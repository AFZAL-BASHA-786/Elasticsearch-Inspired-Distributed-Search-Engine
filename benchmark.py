"""
benchmark.py : EXPERIMENT 2 - "measure latency, tail latency, and throughput as shard
               count changes".

Guide for this file : ../docs/code_guides/11_benchmark_GUIDE.md
My notes behind it  : 03_In_Depth_Implementation/04_Tail_at_Scale_Latency.md
                        (sections 4, 5 and 6: core-to-shard limit, warm-up, measuring rules)
                      doubts 02_Medium.../02_Concurrency_vs_Parallelism.md
                        ("the load generator can be the bottleneck")

For EACH shard count N (1, 2, 4, 8, 16) - a separate run with a freshly built index:

    build index (if missing) -> start shard server processes -> start the coordinator as
    its OWN process -> WARM UP (unmeasured) -> measure -> stop everything

What is measured, and how (the rules from section 6 of my tail-latency notes):

  1. WARM-UP first. The whole query set runs once and the results are thrown away, so page
     faults, channel handshakes and Python start-up effects do not land in the p99.
  2. LATENCY with an OPEN-LOOP load generator: queries are sent on a fixed schedule
     (--rate per second) whether or not earlier ones have finished, and latency is counted
     from the SCHEDULED send time. A closed-loop client would stop sending while the system
     stalls and hide the stall ("coordinated omission").
  3. THROUGHPUT with a CLOSED LOOP: --concurrency queries are kept in flight for a few
     seconds; completed queries / seconds = maximum queries per second.
  4. Percentiles are computed ONCE from the pooled raw latencies (never averaged).
  5. The load generator's own CPU use is reported. If it is near 100% of one core, the
     number measures the test tool, not the search engine.
  6. Everything else stays fixed across N: same corpus, same query file, same k, same
     machine, OMP_NUM_THREADS = 1 in every shard process.

Usage:
    python benchmark.py                          # N = 1 2 4 8 16
    python benchmark.py --shards 1 2 4 8 --rate 50 --stats-mode dfs
"""

import argparse
import asyncio
import json
import multiprocessing
import os
import sys
import time
from typing import List

import grpc

import coordinator as coordinator_module
import search_pb2
import search_pb2_grpc
from build_index import ForwardIndex, ensure_layout
from cluster import ClusterManager
from common import (DEFAULT_COORDINATOR_PORT, DEFAULT_CORPUS, DEFAULT_QUERIES, HERE, HOST,
                    summarize_latencies)

try:
    import psutil                       # optional: only used to read the coordinator's CPU time
except ImportError:                     # the benchmark works without it
    psutil = None


def load_queries(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


async def wait_for_coordinator(stub, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            reply = await stub.ClusterHealth(search_pb2.ClusterHealthRequest(), timeout=1.0)
            if reply.color == "GREEN":
                return
        except grpc.aio.AioRpcError:
            pass
        await asyncio.sleep(0.2)
    raise RuntimeError("coordinator did not become ready (GREEN)")


async def warm_up(stub, queries: List[str], k: int) -> None:
    """One unmeasured pass over the ENTIRE query set (8 in flight at a time)."""
    sem = asyncio.Semaphore(8)

    async def one(q):
        async with sem:
            await stub.Query(search_pb2.QueryRequest(query=q, k=k, timeout_ms=5000))
    await asyncio.gather(*(one(q) for q in queries))


async def open_loop_latency(stub, queries: List[str], k: int, rate: float, total: int) -> dict:
    """Send query i at t0 + i / rate, no matter what. Latency = reply time - SCHEDULED time."""
    latencies, shard_ms, coord_ms = [], [], []
    statuses = {}
    t0 = time.perf_counter() + 0.05

    async def one(i: int, scheduled: float):
        reply = await stub.Query(search_pb2.QueryRequest(query=queries[i % len(queries)], k=k,
                                                         timeout_ms=2000))
        latencies.append((time.perf_counter() - scheduled) * 1000.0)
        shard_ms.append(reply.slowest_shard_ms)
        coord_ms.append(reply.took_ms)
        name = search_pb2.ResultStatus.Name(reply.status)
        statuses[name] = statuses.get(name, 0) + 1

    tasks = []
    for i in range(total):
        scheduled = t0 + i / rate
        # Windows timers tick every ~15 ms, far too coarse to pace queries 10 ms apart.
        # So: sleep coarsely while far away, then yield in a tight loop for the last part.
        while True:
            remaining = scheduled - time.perf_counter()
            if remaining <= 0:
                break
            await asyncio.sleep(remaining - 0.02 if remaining > 0.03 else 0)
        tasks.append(asyncio.ensure_future(one(i, scheduled)))
    await asyncio.gather(*tasks)
    out = summarize_latencies(latencies)
    out["statuses"] = statuses
    out["shard_compute_p50"] = summarize_latencies(shard_ms)["p50"]      # slowest shard's own took_ms
    out["coordinator_p50"] = summarize_latencies(coord_ms)["p50"]        # inside the coordinator
    return out


async def closed_loop_throughput(stub, queries: List[str], k: int, concurrency: int,
                                 seconds: float) -> dict:
    """Keep `concurrency` queries in flight for `seconds`; count how many complete."""
    completed = 0
    latencies = []
    stop_at = time.perf_counter() + seconds

    async def worker(offset: int):
        nonlocal completed
        i = offset
        while time.perf_counter() < stop_at:
            sent = time.perf_counter()
            await stub.Query(search_pb2.QueryRequest(query=queries[i % len(queries)], k=k,
                                                     timeout_ms=5000))
            latencies.append((time.perf_counter() - sent) * 1000.0)
            completed += 1
            i += 7
    cpu0, wall0 = time.process_time(), time.perf_counter()
    await asyncio.gather(*(worker(w * 13) for w in range(concurrency)))
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    out = summarize_latencies(latencies)
    out.update({"qps": completed / wall, "completed": completed,
                # CPU seconds used by THIS process per wall second. Near 100% (one full
                # core) would mean the load generator itself is the bottleneck.
                "load_generator_cpu_pct": 100.0 * cpu / wall})
    return out


async def measure(address: str, queries: List[str], k: int, rate: float, total: int,
                  concurrency: int, seconds: float, coordinator_pid=None) -> dict:
    async with grpc.aio.insecure_channel(address) as channel:
        stub = search_pb2_grpc.CoordinatorServiceStub(channel)
        await wait_for_coordinator(stub)
        await warm_up(stub, queries, k)                  # results discarded
        await warm_up(stub, queries, k)
        latency = await open_loop_latency(stub, queries, k, rate, total)
        proc = psutil.Process(coordinator_pid) if (psutil and coordinator_pid) else None
        cpu0 = sum(proc.cpu_times()[:2]) if proc else 0.0
        wall0 = time.perf_counter()
        throughput = await closed_loop_throughput(stub, queries, k, concurrency, seconds)
        if proc:
            throughput["coordinator_cpu_pct"] = (100.0 * (sum(proc.cpu_times()[:2]) - cpu0)
                                                 / (time.perf_counter() - wall0))
        return {"latency": latency, "throughput": throughput}


def run_one(n: int, args, queries: List[str], fwd: ForwardIndex) -> dict:
    routing = ensure_layout(n, args.copies, args.nodes, args.corpus, tag=args.tag, fwd=fwd)
    manager = ClusterManager(routing, quiet=True)
    manager.start_all()
    # The coordinator runs as its OWN process (its own core, its own GIL), exactly as it
    # would in a deployment; the load generator is this process.
    coordinator = multiprocessing.Process(
        target=coordinator_module.run_server,
        kwargs=dict(routing_path=routing, port=args.port, stats_mode=args.stats_mode, verbose=False),
        daemon=True)
    coordinator.start()
    try:
        result = asyncio.run(measure("%s:%d" % (HOST, args.port), queries, args.k, args.rate, args.total,
                                     args.concurrency, args.seconds, coordinator.pid))
    finally:
        coordinator.kill()
        coordinator.join(timeout=5)
        manager.stop_all()
    result.update({"shards": n, "copies": args.copies, "processes": len(manager.copies)})
    return result


async def _remote_layout(address: str) -> dict:
    """Ask a running coordinator what it is serving (shards, copies, which nodes)."""
    async with grpc.aio.insecure_channel(address) as channel:
        stub = search_pb2_grpc.CoordinatorServiceStub(channel)
        await wait_for_coordinator(stub)
        reply = await stub.ClusterHealth(search_pb2.ClusterHealthRequest(), timeout=2.0)
        return {"shards": len({c.shard_id for c in reply.copies}),
                "processes": len(reply.copies),
                "nodes": sorted({c.address.rsplit(":", 1)[0] for c in reply.copies})}


def run_remote(args, queries: List[str]) -> dict:
    """MULTI-LAPTOP mode: the shard servers (on the node laptops) and the coordinator are
    already running; this process is only the load generator. Nothing is started locally.
    Every row is tagged with the machines it ran on, so results stay honest about the setup.
    """
    layout = asyncio.run(_remote_layout(args.remote))
    result = asyncio.run(measure(args.remote, queries, args.k, args.rate, args.total,
                                 args.concurrency, args.seconds,
                                 args.coordinator_pid or None))
    result.update(layout)
    result["copies"] = layout["processes"] // max(layout["shards"], 1)
    return result


def plot(rows: List[dict], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    xs = [r["shards"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (("p50", "p50"), ("p95", "p95"), ("p99", "p99")):
        ax1.plot(xs, [r["latency"][key] for r in rows], marker="o", label=label)
    ax1.plot(xs, [r["latency"]["shard_compute_p50"] for r in rows], marker="s", linestyle="--",
             label="slowest shard compute (p50)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_xlabel("shards N")
    ax1.set_ylabel("latency (ms)")
    ax1.set_title("Latency vs shard count (open loop)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(xs, [r["throughput"]["qps"] for r in rows], marker="o", color="tab:green")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(x) for x in xs])
    ax2.set_xlabel("shards N")
    ax2.set_ylabel("queries per second")
    ax2.set_title("Max throughput vs shard count (closed loop)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True


def main():
    ap = argparse.ArgumentParser(description="Latency / tail latency / throughput vs shard count.")
    ap.add_argument("--shards", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--rate", type=float, default=100.0, help="open-loop queries per second")
    ap.add_argument("--total", type=int, default=3000, help="measured queries in the latency test")
    ap.add_argument("--concurrency", type=int, default=32, help="in-flight queries in the throughput test")
    ap.add_argument("--seconds", type=float, default=6.0, help="duration of the throughput test")
    ap.add_argument("--stats-mode", choices=["cached", "dfs"], default="cached")
    ap.add_argument("--port", type=int, default=DEFAULT_COORDINATOR_PORT)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--queries-file", default=DEFAULT_QUERIES)
    ap.add_argument("--tag", default="", help="layout folder suffix (use one per corpus)")
    ap.add_argument("--remote", default="",
                    help="multi-laptop: HOST:PORT of an already-running coordinator; measure it "
                         "and start nothing locally. Rows accumulate in results/benchmark_remote.json")
    ap.add_argument("--coordinator-pid", type=int, default=0,
                    help="with --remote, if the coordinator runs on THIS laptop: its pid, to report its CPU")
    args = ap.parse_args()

    # THE LOAD GENERATOR MUST NOT DISTORT WHAT IT MEASURES.
    # The pacing loop of the open-loop test keeps this process's main thread busy, and a
    # busy Python thread holds the GIL. gRPC delivers replies on its own I/O thread, which
    # then has to WAIT for the GIL - by default up to 5 ms (the "switch interval"). The first
    # version of this benchmark reported p99 = 11 ms; 6 ms of that was this effect inside
    # the test tool, not the search engine. Asking for the GIL to be handed over every
    # 0.1 ms removes it. (The "bottleneck illusion" from my notes, met in real life.)
    sys.setswitchinterval(0.0001)

    queries = load_queries(args.queries_file)
    print("physical cores: %s | logical processors: %s | queries: %d | k = %d | stats mode: %s"
          % (psutil.cpu_count(logical=False) if psutil else "?", os.cpu_count(), len(queries),
             args.k, args.stats_mode))
    remote = bool(args.remote)
    fwd = None
    if not remote:
        print("analysing corpus once ...")
        fwd = ForwardIndex(args.corpus)

    rows = []
    remote_path = os.path.join(HERE, "results", "benchmark_remote.json")
    if remote and os.path.exists(remote_path):
        with open(remote_path, "r", encoding="utf-8") as f:
            rows = json.load(f).get("rows", [])          # earlier runs (other shard counts)
    header = ("%-6s %-5s | %-8s %-8s %-8s %-8s | %-13s %-11s | %-9s %-10s %-10s"
              % ("shards", "procs", "p50 ms", "p95 ms", "p99 ms", "mean ms", "shard cmp p50",
                 "coord p50", "max QPS", "loadgen %", "coord %"))
    print()
    print("latency: open loop at %.0f queries/s, %d measured queries after warm-up" % (args.rate, args.total))
    print("throughput: closed loop, %d in flight for %.0f s" % (args.concurrency, args.seconds))
    print(header)
    for n in ([None] if remote else args.shards):
        row = run_remote(args, queries) if remote else run_one(n, args, queries, fwd)
        n = row["shards"]
        if remote:
            print("measured coordinator %s serving %d shards on %s"
                  % (args.remote, n, ", ".join(row["nodes"])))
            rows = [r for r in rows if r["shards"] != n]  # a re-run replaces the old row
        rows.append(row)
        lat, thr = row["latency"], row["throughput"]
        print("%-6d %-5d | %-8.2f %-8.2f %-8.2f %-8.2f | %-13.2f %-11.2f | %-9.0f %-10.0f %-10s"
              % (n, row["processes"], lat["p50"], lat["p95"], lat["p99"], lat["mean"],
                 lat["shard_compute_p50"], lat["coordinator_p50"], thr["qps"],
                 thr["load_generator_cpu_pct"],
                 "%.0f" % thr["coordinator_cpu_pct"] if "coordinator_cpu_pct" in thr else "?"))
        bad = {s: c for s, c in lat["statuses"].items() if s != "COMPLETE"}
        if bad:
            print("       WARNING: non-COMPLETE replies during the latency test: %s" % bad)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows.sort(key=lambda r: r["shards"])
    suffix = ("" if args.stats_mode == "cached" else "_" + args.stats_mode) + \
             ("_" + args.tag if args.tag else "")
    if remote:
        suffix = "_remote"
    json_path = os.path.join(HERE, "results", "benchmark%s.json" % suffix)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "physical_cores": psutil.cpu_count(logical=False) if psutil else None,
                   "logical_processors": os.cpu_count(), "rows": rows}, f, indent=2)
    print()
    print("saved -> %s" % json_path)
    png_path = os.path.join(HERE, "results", "benchmark%s.png" % suffix)
    if plot(rows, png_path):
        print("plot  -> %s" % png_path)
    print("columns: 'shard cmp p50' = the slowest shard's own compute time (took_ms); "
          "'coord p50' = time inside the coordinator;\n'loadgen %' and 'coord %' = CPU of "
          "one core used by the load generator / the coordinator during the throughput test.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
