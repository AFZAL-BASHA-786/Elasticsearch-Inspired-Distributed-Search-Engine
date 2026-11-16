"""
demo_failover.py : EXPERIMENT 3 - the FOUR-ACT failover demonstration, fully automated.

Guide for this file : ../docs/code_guides/12_demo_failover_GUIDE.md
My notes behind it  : 03_In_Depth_Implementation/03_Fault_Tolerance_and_Replication.md
                        (section 5 and section 10.3: the four acts and their pass criteria)
                      02_Medium_Architecture_and_Design/03_gRPC_and_Communication.md
                        (section 7.3: fast failure vs slow failure)

The brief: "Demonstrate one unavailable shard replica and report whether queries complete
or explicitly return a partial-result status."

Layout: 3 shards x 2 copies on 3 nodes (anti-affinity):  S0 -> node A + node C
                                                         S1 -> node B + node A
                                                         S2 -> node C + node B

  Act 1  nothing fails                -> GREEN   every query COMPLETE          (the baseline)
  Act 2  KILL one copy of S0          -> YELLOW  every query still COMPLETE, same results
  Act 3  KILL the other copy of S0    -> RED     every query PARTIAL, failed_shards = [0]
  Act 4  RESTART both                 -> GREEN   COMPLETE again, same results, no data repair
  Act 5  FREEZE one copy of S1 (bonus)-> the SLOW failure: still COMPLETE, but the query that
                                         hits the frozen copy pays one full 200 ms deadline

A steady stream of queries runs THROUGH every act, and each fault is injected in the middle
of the stream - so the log proves that no query failed at the moment of the failure.

Usage:
    python demo_failover.py
"""

import argparse
import asyncio
import json
import os
import time
from typing import List

from build_index import ensure_layout
from cluster import ClusterManager
from common import DEFAULT_CORPUS, HERE, shard_of, summarize_latencies
from coordinator import Coordinator

NUM_SHARDS, COPIES, NODES = 3, 2, 3
CANDIDATE_QUERIES = ["heartbeat failover", "raft consensus", "apple pie recipe", "cheap flights", "quantum energy",
                     "garden soil", "guitar melody"]


async def query_stream(coordinator: Coordinator, query: str, k: int, seconds: float,
                       inject=None, inject_after_s: float = 1.0) -> List[dict]:
    """Send the demo query every ~20 ms for `seconds`; call `inject` once, mid-stream."""
    loop = asyncio.get_event_loop()
    records = []
    started = time.perf_counter()
    injected_at = 0.0 if inject is None else None        # None = the fault has not happened yet
    while time.perf_counter() - started < seconds:
        if injected_at is None and time.perf_counter() - started >= inject_after_s:
            # Fault injection runs in a helper thread so the event loop (heartbeats, other
            # tasks) is never blocked by it.
            await loop.run_in_executor(None, inject)
            injected_at = time.perf_counter() - started
        sent = time.perf_counter()
        reply = await coordinator.search(query, k)
        reply["latency_ms"] = (time.perf_counter() - sent) * 1000.0
        reply["after_fault"] = injected_at is not None
        reply["seconds_after_fault"] = (sent - started - injected_at) if injected_at is not None else -1.0
        records.append(reply)
        await asyncio.sleep(0.02)
    return records


def summarize(act: str, action: str, records: List[dict], baseline_hits, tracked_doc: int,
              color: str) -> dict:
    """Count only the queries sent AFTER the fault was injected (Act 1 has no fault, so
    all of its queries count). `settled_after_s` = how long after the fault the LAST change
    of status happened, i.e. how long the system needed to reach its new steady state."""
    after = [r for r in records if r["after_fault"]]
    lat = summarize_latencies([r["latency_ms"] for r in after])
    last = after[-1]
    settled = 0.0
    for prev, cur in zip(after, after[1:]):
        if prev["status"] != cur["status"]:
            settled = cur["seconds_after_fault"]
    return {
        "act": act, "action": action, "colour_at_end": color,
        "queries_before_fault": len(records) - len(after),
        "queries": len(after),
        "COMPLETE": sum(r["status"] == "COMPLETE" for r in after),
        "PARTIAL": sum(r["status"] == "PARTIAL" for r in after),
        "FAILED": sum(r["status"] == "FAILED" for r in after),
        "status_at_end": last["status"],
        "settled_after_s": settled,
        "rescued_by_failover": sum(1 for r in after if r["retries"] > 0 and r["status"] == "COMPLETE"),
        "p50_ms": lat["p50"], "p99_ms": lat["p99"], "max_ms": lat["max"],
        "complete_results_identical_to_baseline": all(
            r["hits"] == baseline_hits for r in records if r["status"] == "COMPLETE"),
        "tracked_S0_doc_present_at_end": any(d == tracked_doc for d, _ in last["hits"]),
        "failed_shards_at_end": {str(s): e for s, e in last["failed_shards"].items()},
    }


def print_row(row: dict) -> None:
    print("  after the fault: queries=%d  COMPLETE=%d  PARTIAL=%d  FAILED=%d  rescued by failover=%d"
          % (row["queries"], row["COMPLETE"], row["PARTIAL"], row["FAILED"],
             row["rescued_by_failover"]))
    print("  status at end: %s (settled %.1f s after the fault)"
          % (row["status_at_end"], row["settled_after_s"]))
    print("  latency p50=%.1f ms  p99=%.1f ms  max=%.1f ms | colour at end: %s"
          % (row["p50_ms"], row["p99_ms"], row["max_ms"], row["colour_at_end"]))
    print("  COMPLETE results identical to baseline: %s | tracked S0 document present: %s"
          % (row["complete_results_identical_to_baseline"], row["tracked_S0_doc_present_at_end"]))
    if row["failed_shards_at_end"]:
        print("  failed_shards: %s" % row["failed_shards_at_end"])


async def run_demo(routing: str, manager: ClusterManager, k: int, act_seconds: float) -> dict:
    coordinator = Coordinator(routing, stats_mode="cached", verbose=True)
    await coordinator.start()
    try:
        # Pick a demo query whose top-k contains a document that lives in SHARD 0. That
        # document is the visible evidence in Act 3: it must disappear from the results.
        query, baseline_hits, tracked = None, None, None
        for candidate in CANDIDATE_QUERIES:
            hits = (await coordinator.search(candidate, k))["hits"]
            in_s0 = [d for d, _ in hits if shard_of(d, NUM_SHARDS) == 0]
            if in_s0:
                query, baseline_hits, tracked = candidate, hits, in_s0[0]
                break
        if query is None:
            raise RuntimeError("no demo query has a top-k document in shard 0")
        for _ in range(50):                              # warm-up (page cache, channels)
            await coordinator.search(query, k)
        print("\ndemo query: %r, k = %d. Tracked document %d lives in shard 0 "
              "(murmur3(%d) mod 3 = 0).\n" % (query, k, tracked, tracked))

        rows = []

        print("=== ACT 1: all GREEN - the baseline ===")
        recs = await query_stream(coordinator, query, k, act_seconds)
        rows.append(summarize("1", "nothing fails (baseline)", recs, baseline_hits, tracked,
                              coordinator.cluster_color()))
        print_row(rows[-1])

        print("\n=== ACT 2: KILL shard 0 copy 0 (one unavailable replica) ===")
        recs = await query_stream(coordinator, query, k, act_seconds + 3.0,
                                  inject=lambda: manager.kill_copy(0, 0))
        rows.append(summarize("2", "kill S0 copy 0", recs, baseline_hits, tracked,
                              coordinator.cluster_color()))
        print_row(rows[-1])

        print("\n=== ACT 3: KILL shard 0 copy 1 too (no copy of S0 left) ===")
        recs = await query_stream(coordinator, query, k, act_seconds + 3.0,
                                  inject=lambda: manager.kill_copy(0, 1))
        rows.append(summarize("3", "kill S0 copy 1 as well", recs, baseline_hits, tracked,
                              coordinator.cluster_color()))
        print_row(rows[-1])

        print("\n=== ACT 4: RESTART both copies of shard 0 ===")

        def restart_both():
            manager.restart_copy(0, 0)
            manager.restart_copy(0, 1)
        recs = await query_stream(coordinator, query, k, act_seconds + 5.0, inject=restart_both)
        rows.append(summarize("4", "restart both S0 copies", recs, baseline_hits, tracked,
                              coordinator.cluster_color()))
        print_row(rows[-1])

        print("\n=== ACT 5 (bonus): FREEZE shard 1 copy 0 - the slow failure ===")
        recs = await query_stream(coordinator, query, k, act_seconds + 3.0,
                                  inject=lambda: manager.freeze_copy(1, 0))
        rows.append(summarize("5", "freeze S1 copy 0 (alive but silent)", recs, baseline_hits,
                              tracked, coordinator.cluster_color()))
        print_row(rows[-1])
        manager.resume_copy(1, 0)

        return {"query": query, "k": k, "tracked_doc_in_shard_0": tracked, "acts": rows}
    finally:
        await coordinator.stop()


def main():
    ap = argparse.ArgumentParser(description="Four-act failover demonstration.")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--act-seconds", type=float, default=3.0)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    args = ap.parse_args()

    routing = ensure_layout(NUM_SHARDS, COPIES, NODES, args.corpus)
    manager = ClusterManager(routing, quiet=True)
    manager.start_all()
    print("started %d shard server processes:" % len(manager.copies))
    print("\n".join(manager.status_lines()))
    try:
        report = asyncio.run(run_demo(routing, manager, args.k, args.act_seconds))
    finally:
        manager.stop_all()

    print("\n================ SUMMARY ================")
    print("(counts are for queries sent AFTER each fault was injected)")
    print("%-4s %-36s %-7s %-8s %-9s %-8s %-7s %-10s %-8s %-8s %-8s %s"
          % ("act", "action", "colour", "queries", "COMPLETE", "PARTIAL", "FAILED",
             "end status", "p50 ms", "p99 ms", "max ms", "S0 doc?"))
    for r in report["acts"]:
        print("%-4s %-36s %-7s %-8d %-9d %-8d %-7d %-10s %-8.1f %-8.1f %-8.1f %s"
              % (r["act"], r["action"], r["colour_at_end"], r["queries"], r["COMPLETE"],
                 r["PARTIAL"], r["FAILED"], r["status_at_end"], r["p50_ms"], r["p99_ms"],
                 r["max_ms"], "yes" if r["tracked_S0_doc_present_at_end"] else "NO"))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "demo_failover.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
