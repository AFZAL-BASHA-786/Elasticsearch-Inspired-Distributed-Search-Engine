"""
cluster.py : the LAUNCHER. Reads routing_table.json and starts ONE PROCESS PER SHARD COPY.

Guide for this file : ../docs/code_guides/06_cluster_GUIDE.md
My notes behind it  : doubts 01_Basics_and_Intuition/01_What_Is_A_Distributed_Search_Engine.md
                        (Doubt 6: "how does the program become a process, who launches them?")
                      03_In_Depth_Implementation/03_Fault_Tolerance_and_Replication.md
                        (section 10.3: the four-act failover demonstration)

One PROGRAM (shard_server.py on disk), MANY PROCESSES (one per shard copy, each with its
own arguments, its own memory, its own Python interpreter and therefore its own GIL).
Uses Python's `multiprocessing` so every copy is a real, separate OS process.

It is also the "fault injector" for the demonstration:
    kill   S C   -> the process dies       -> FAST failure  (connection refused, UNAVAILABLE)
    freeze S C   -> the process is paused  -> SLOW failure  (silence until DEADLINE_EXCEEDED)
    resume S C   -> un-pause a frozen process
    restart S C  -> start the process again (no data repair needed: the files never changed)

Interactive use (Terminal 1):
    python cluster.py --shards 3
    cluster> status
    cluster> kill 0 0
    cluster> restart 0 0
    cluster> quit
"""

import argparse
import ctypes
import multiprocessing
import os
import signal
import sys
import time
from typing import Dict, Tuple

import grpc

import search_pb2
import search_pb2_grpc
import shard_server
from common import load_routing_table, routing_table_path


def _suspend_process(pid: int, resume: bool = False) -> None:
    """Freeze / un-freeze a whole process WITHOUT killing it.

    A frozen process still exists: the OS keeps its socket open and keeps ACCEPTING bytes
    into the socket buffer on its behalf, but the process never READS them and never
    replies. To the coordinator the connection looks perfectly healthy - only the deadline
    rescues the query. (notes: doubts/02_03_deadlines.md, "accepting vs reading")
    """
    if os.name == "nt":
        PROCESS_SUSPEND_RESUME = 0x0800
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if not handle:
            raise OSError("cannot open process %d" % pid)
        try:
            ntdll = ctypes.windll.ntdll
            (ntdll.NtResumeProcess if resume else ntdll.NtSuspendProcess)(handle)
        finally:
            kernel32.CloseHandle(handle)
    else:
        os.kill(pid, signal.SIGCONT if resume else signal.SIGSTOP)


class ClusterManager:
    """Starts, kills, freezes and restarts shard server processes for one routing table."""

    def __init__(self, routing_path: str, quiet: bool = True, node: str = ""):
        self.routing_path = os.path.abspath(routing_path)
        self.base_dir = os.path.dirname(self.routing_path)
        self.table = load_routing_table(self.routing_path)
        self.quiet = quiet
        # MULTI-LAPTOP: with node="node_A" this launcher starts ONLY the copies the routing
        # table places on node_A - one launcher runs on each laptop. Empty = every copy
        # (single-machine mode).
        self.node = node
        # (shard_id, copy_id) -> copy description from the routing table
        self.copies: Dict[Tuple[int, int], dict] = {}
        for entry in self.table["shards"]:
            for copy in entry["copies"]:
                if not node or copy["node"] == node:
                    self.copies[(entry["shard_id"], copy["copy_id"])] = copy
        self.procs: Dict[Tuple[int, int], multiprocessing.Process] = {}
        self.frozen = set()

    # ------------------------------------------------------------------ start / stop
    def start_copy(self, shard_id: int, copy_id: int) -> None:
        copy = self.copies[(shard_id, copy_id)]
        old = self.procs.get((shard_id, copy_id))
        if old is not None and old.is_alive():
            return
        host, port = copy["address"].rsplit(":", 1)
        # Where to LISTEN: a copy addressed as 127.0.0.x is local-only. A copy addressed by a
        # real LAN IP must accept connections from other laptops, so it listens on 0.0.0.0
        # (every network interface of this laptop).
        bind = host if host.startswith("127.") or host == "localhost" else "0.0.0.0"
        index_dir = os.path.join(self.base_dir, copy["path"])
        if not os.path.isdir(index_dir):
            raise RuntimeError("missing %s - copy this node's folder from the build laptop"
                               % index_dir)
        proc = multiprocessing.Process(
            target=shard_server.serve,
            kwargs=dict(shard_id=shard_id, copy_id=copy_id, port=int(port),
                        index_dir=index_dir, host=bind, quiet=self.quiet),
            name="shard%d-copy%d" % (shard_id, copy_id),
            daemon=True,                 # if the launcher dies, its shard servers die too
        )
        proc.start()
        self.procs[(shard_id, copy_id)] = proc
        self.frozen.discard((shard_id, copy_id))

    def start_all(self, wait: bool = True) -> None:
        for key in sorted(self.copies):
            self.start_copy(*key)
        if wait:
            self.wait_ready()

    def wait_ready(self, timeout_s: float = 30.0) -> None:
        """Block until every running copy answers a Health call ("serving")."""
        deadline = time.time() + timeout_s
        pending = set(self.procs)
        restarts: Dict[Tuple[int, int], int] = {}
        while pending and time.time() < deadline:
            for key in sorted(pending):
                if self._is_serving(self.copies[key]["address"]):
                    pending.discard(key)
                elif not self.procs[key].is_alive():
                    # The process exited during start-up - almost always because its port
                    # was momentarily taken. Try again a few times before giving up.
                    restarts[key] = restarts.get(key, 0) + 1
                    if restarts[key] > 3:
                        raise RuntimeError("shard %d copy %d cannot start on %s (port in use?)"
                                           % (key[0], key[1], self.copies[key]["address"]))
                    time.sleep(0.5)
                    self.start_copy(*key)
            if pending:
                time.sleep(0.1)
        if pending:
            raise RuntimeError("shard copies did not become ready: %s" % sorted(pending))

    @staticmethod
    def _is_serving(address: str, timeout_s: float = 0.5) -> bool:
        try:
            with grpc.insecure_channel(address) as channel:
                stub = search_pb2_grpc.ShardServiceStub(channel)
                return stub.Health(search_pb2.HealthRequest(), timeout=timeout_s).serving
        except grpc.RpcError:
            return False

    def stop_all(self) -> None:
        for key in list(self.procs):
            self.kill_copy(*key)
        self.procs.clear()

    # ------------------------------------------------------------------ fault injection
    def kill_copy(self, shard_id: int, copy_id: int) -> None:
        """FAST failure: the process is gone; the OS refuses connections on its port."""
        proc = self.procs.get((shard_id, copy_id))
        if proc is None or not proc.is_alive():
            return
        if (shard_id, copy_id) in self.frozen:           # a frozen process must be resumed to die cleanly
            _suspend_process(proc.pid, resume=True)
            self.frozen.discard((shard_id, copy_id))
        proc.kill()
        proc.join(timeout=5)

    def freeze_copy(self, shard_id: int, copy_id: int) -> None:
        """SLOW failure: alive but silent. Only the per-attempt deadline ends the wait."""
        proc = self.procs.get((shard_id, copy_id))
        if proc is not None and proc.is_alive():
            _suspend_process(proc.pid)
            self.frozen.add((shard_id, copy_id))

    def resume_copy(self, shard_id: int, copy_id: int) -> None:
        proc = self.procs.get((shard_id, copy_id))
        if proc is not None and proc.is_alive() and (shard_id, copy_id) in self.frozen:
            _suspend_process(proc.pid, resume=True)
            self.frozen.discard((shard_id, copy_id))

    def restart_copy(self, shard_id: int, copy_id: int, wait: bool = True) -> None:
        """Recovery = run the program again. The .npy files were never modified, so the
        restarted copy is still byte-identical to the other copy: nothing to re-sync."""
        self.kill_copy(shard_id, copy_id)
        self.start_copy(shard_id, copy_id)
        if wait:
            deadline = time.time() + 30
            while time.time() < deadline:
                if self._is_serving(self.copies[(shard_id, copy_id)]["address"]):
                    return
                time.sleep(0.1)
            raise RuntimeError("copy did not come back")

    # ------------------------------------------------------------------ reporting
    def status_lines(self):
        lines = []
        for key in sorted(self.copies):
            copy = self.copies[key]
            proc = self.procs.get(key)
            if proc is None or not proc.is_alive():
                state, pid = "KILLED", "-"
            elif key in self.frozen:
                state, pid = "FROZEN", str(proc.pid)
            else:
                state, pid = "RUNNING", str(proc.pid)
            lines.append("  shard %-2d copy %d  %-7s %-16s pid %-7s %s"
                         % (key[0], key[1], copy["node"], copy["address"], pid, state))
        return lines


def interactive(manager: ClusterManager) -> None:
    """A tiny console so failures can be injected by hand during a demonstration."""
    print("commands: status | kill S C | freeze S C | resume S C | restart S C | quit")
    actions = {"kill": manager.kill_copy, "freeze": manager.freeze_copy,
               "resume": manager.resume_copy, "restart": manager.restart_copy}
    while True:
        try:
            line = input("cluster> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd = line[0].lower()
        if cmd in ("quit", "exit", "q"):
            break
        if cmd == "status":
            print("\n".join(manager.status_lines()))
        elif cmd in actions and len(line) == 3:
            try:
                key = (int(line[1]), int(line[2]))
                if key not in manager.copies:
                    print("no such copy: shard %d copy %d" % key)
                    continue
                actions[cmd](*key)
                print("ok: %s shard %d copy %d" % (cmd, key[0], key[1]))
            except Exception as exc:                     # keep the console alive
                print("error:", exc)
        else:
            print("commands: status | kill S C | freeze S C | resume S C | restart S C | quit")


def check_reachable(routing_path: str) -> bool:
    """MULTI-LAPTOP pre-flight: from THIS laptop, send one Health call to every shard copy
    in the routing table and say which ones answer. Run it on the coordinator laptop after
    the node laptops have started, BEFORE starting the coordinator. It tests the whole path
    at once: right IP, firewall open, shard server running, gRPC working."""
    table = load_routing_table(routing_path)
    print("%-6s %-5s %-8s %-22s %s" % ("shard", "copy", "node", "address", "reachable?"))
    all_ok = True
    per_shard_ok = {}
    for entry in table["shards"]:
        for copy in entry["copies"]:
            ok = ClusterManager._is_serving(copy["address"], timeout_s=3.0)
            all_ok = all_ok and ok
            per_shard_ok[entry["shard_id"]] = per_shard_ok.get(entry["shard_id"], False) or ok
            print("%-6d %-5d %-8s %-22s %s"
                  % (entry["shard_id"], copy["copy_id"], copy["node"], copy["address"],
                     "yes" if ok else "NO  <- check IP, firewall, and that the node is started"))
    dead_shards = [s for s, ok in per_shard_ok.items() if not ok]
    if all_ok:
        print("all copies reachable -> the coordinator will start GREEN")
    elif dead_shards:
        print("shards with NO reachable copy: %s -> queries would be PARTIAL (RED)" % dead_shards)
    else:
        print("every shard has at least one reachable copy -> COMPLETE answers, but YELLOW")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="Start one process per shard copy.")
    ap.add_argument("--shards", type=int, default=3, help="which build to launch (shard count)")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--routing", default="", help="explicit path to a routing_table.json")
    ap.add_argument("--node", default="",
                    help="multi-laptop: start only this node's copies, e.g. --node node_A")
    ap.add_argument("--check", action="store_true",
                    help="multi-laptop: start nothing; test that every copy in the routing "
                         "table is reachable from this laptop")
    args = ap.parse_args()

    path = args.routing or routing_table_path(args.shards, args.copies, args.tag)
    if not os.path.exists(path):
        sys.exit("no routing table at %s\nrun:  python build_index.py --shards %d --copies %d"
                 % (path, args.shards, args.copies))
    if args.check:
        sys.exit(0 if check_reachable(path) else 1)
    manager = ClusterManager(path, quiet=False, node=args.node)
    if not manager.copies:
        sys.exit("the routing table places no copy on node %r" % args.node)
    print("launching %d shard copies%s from %s"
          % (len(manager.copies), " of " + args.node if args.node else "", path))
    manager.start_all()
    print("all copies are serving.")
    try:
        interactive(manager)
    finally:
        manager.stop_all()
        print("cluster stopped.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
