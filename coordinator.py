"""
coordinator.py : the HEAD LIBRARIAN. Holds NO documents and NO index - only the routing
                 table, the health "whiteboard" and a small cache of statistics.

Guide for this file : ../docs/code_guides/07_coordinator_GUIDE.md
My notes behind it  : 01_Basics_and_Intuition/02_The_Scatter_Gather_Pattern.md
                      02_Medium_Architecture_and_Design/02_Concurrency_vs_Parallelism.md
                      02_Medium_Architecture_and_Design/03_gRPC_and_Communication.md (sec. 14)
                      03_In_Depth_Implementation/02_k_Way_Heap_Merges.md        (Method A)
                      03_In_Depth_Implementation/03_Fault_Tolerance_and_Replication.md
                      (all under  self learning/learning_guide/)

ONE PROCESS, ONE THREAD, ONE asyncio EVENT LOOP (the single "waiter"):
  The coordinator is I/O-bound: it spends >90% of its time WAITING for shards. One thread
  can overlap hundreds of waits because a parked task costs no CPU. It never scores a
  document - a long computation would block the only thread and freeze every other query.

What happens for one query (the 9 steps of my scatter-gather notes):
  1 receive  2 analyse (once, for all shards)  3 get GLOBAL statistics (exact integer sums)
  4 choose ONE live copy per shard (least outstanding requests)
  5 SCATTER: all shard requests leave at the same time, each with a deadline
  6 GATHER : fold each reply into a bounded MIN-HEAP the moment it arrives (Method A)
  7 failover: UNAVAILABLE / DEADLINE_EXCEEDED -> mark SUSPECT -> try the other copy (budgeted)
  8 the heap now holds the global top-k
  9 respond: hits + status COMPLETE or PARTIAL + which shards failed and why

Run it (Terminal 2, after `python cluster.py --shards 3` in Terminal 1):
    python coordinator.py --shards 3
One-shot test without starting a server:
    python coordinator.py --shards 3 --query "raft consensus"
"""

import argparse
import asyncio
import heapq
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import grpc

import search_pb2
import search_pb2_grpc
from common import (DEFAULT_COORDINATOR_PORT, HOST, hit_key, load_routing_table, query_terms,
                    routing_table_path)

# ======================================================================================
# THE DIALS  (every number my notes argued about, in one place)
# ======================================================================================
CLIENT_TIMEOUT_S = 0.500       # the client's TOTAL time budget: one absolute deadline per query
PER_ATTEMPT_TIMEOUT_S = 0.200  # deadline of ONE call to ONE copy. Short enough to leave room
                               # for a second attempt inside the 500 ms budget (500/200 = 2.5)
MAX_ATTEMPTS = 2               # ATTEMPT BUDGET: 1 try + at most 1 retry per shard per query.
                               # Bounded on purpose: unlimited retries on an overloaded
                               # system multiply the load (a "retry storm").
HEARTBEAT_INTERVAL_S = 1.0     # the intercom roll call, once per second
HEARTBEAT_TIMEOUT_S = 0.200    # a heartbeat reply later than this counts as a miss
DOWN_AFTER_MISSES = 3          # 1 miss is a blip; 3 in a row is a real failure (~3 s to detect)
UP_AFTER_SUCCESSES = 2         # a copy must answer twice in a row before it is trusted again
STATS_REFRESH_S = 30.0         # "refreshed periodically": re-collect the cached statistics

# Status codes that mean "THIS COPY is the problem -> another copy may succeed".
# INVALID_ARGUMENT is NOT here: then THE REQUEST is wrong and every copy would refuse it.
RETRYABLE = {
    grpc.StatusCode.UNAVAILABLE,          # process killed       -> FAST failure
    grpc.StatusCode.DEADLINE_EXCEEDED,    # process frozen/slow  -> SLOW failure (full deadline)
    grpc.StatusCode.RESOURCE_EXHAUSTED,   # copy's queue is full
    grpc.StatusCode.INTERNAL,             # unexpected error inside the shard server
}

UP, SUSPECT, DOWN = "UP", "SUSPECT", "DOWN"

CHANNEL_OPTIONS = [
    # Keep-alive pings check the DOOR (is the connection still intact?), heartbeats check
    # the WORKER (is the shard process responsive?). Both are needed.
    ("grpc.keepalive_time_ms", 10000),
    ("grpc.keepalive_timeout_ms", 5000),
    ("grpc.keepalive_permit_without_calls", 1),
    # After a copy dies, the channel retries the connection with back-off. gRPC's default
    # back-off grows up to 2 minutes; cap it at 1 s so a RESTARTED copy is noticed quickly.
    ("grpc.initial_reconnect_backoff_ms", 200),
    ("grpc.min_reconnect_backoff_ms", 200),
    ("grpc.max_reconnect_backoff_ms", 1000),
]


class ShardCopy:
    """One row of the whiteboard: one copy of one shard, its tube, and its health."""

    def __init__(self, shard_id: int, copy_id: int, node: str, address: str):
        self.shard_id = shard_id
        self.copy_id = copy_id
        self.node = node
        self.address = address
        # THE CHANNEL = the pneumatic tube, installed ONCE at start-up and reused for every
        # query. It is one long-lived TCP connection running HTTP/2, which MULTIPLEXES:
        # hundreds of in-flight calls share it, each as its own stream id. Opening a new
        # connection per call would cost 2 extra round trips and, under load, run the OS
        # out of ports (TIME_WAIT).
        self.channel = grpc.aio.insecure_channel(address, options=CHANNEL_OPTIONS)
        self.stub = search_pb2_grpc.ShardServiceStub(self.channel)
        self.state = UP
        self.misses = 0          # consecutive missed heartbeats
        self.successes = 0       # consecutive answered heartbeats
        self.outstanding = 0     # calls sent to this copy that have not come back yet

    def label(self) -> str:
        return "S%d c%d (%s)" % (self.shard_id, self.copy_id, self.address)


class ShardResult:
    """Outcome of asking ONE SHARD (after trying up to MAX_ATTEMPTS of its copies)."""

    def __init__(self, shard_id: int):
        self.shard_id = shard_id
        self.reply = None                    # StatsReply / SearchReply when some copy answered
        self.errors: List[str] = []          # one line per failed attempt
        self.retries = 0                     # failover attempts that were needed

    @property
    def ok(self) -> bool:
        return self.reply is not None


class TopKMerger:
    """METHOD A: the bounded min-heap, the "bouncer" of my k-way merge notes.

    * the heap never holds more than k hits                      -> memory O(k)
    * the ROOT is the WORST hit currently kept = the admission threshold
    * each shard's reply is folded in the moment it arrives      -> streaming, no waiting
    * replies are sorted best-first, so the first hit that cannot beat the root ends that
      list: every later hit in it is worse too                   -> EARLY TERMINATION

    Why a MIN-heap when we want the HIGHEST scores? Because the only question asked about a
    new hit is "does it beat the weakest member of the current top-k?" - and the weakest
    member is the one to evict. Keeping it at the root makes the check O(1).

    Heap items are (score, -doc_id): Python compares tuples left to right, so the smallest
    item is the lowest score, and among equal scores the LARGEST doc id (the worst by our
    tie-break rule: score descending, then doc id ascending).
    """

    def __init__(self, k: int):
        self.k = k
        self.heap: List[Tuple[float, int]] = []
        self.examined = 0                    # how many hits were actually looked at
        self.received = 0                    # how many hits arrived in total

    def offer(self, hits) -> None:
        """Fold one shard's local top-k (already sorted best-first) into the heap."""
        self.received += len(hits)
        for hit in hits:
            self.examined += 1
            item = hit_key(hit.score, hit.doc_id)
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, item)          # case 1: room left -> push
            elif item > self.heap[0]:
                heapq.heapreplace(self.heap, item)       # case 2: beats the root -> evict the root
            else:
                break                                    # case 3: rejected -> skip the rest of this list

    def result(self) -> List[Tuple[int, float]]:
        """Pop everything: items come out WORST first, so reverse to get best-first."""
        heap = list(self.heap)
        worst_first = [heapq.heappop(heap) for _ in range(len(heap))]
        return [(-neg_doc, score) for score, neg_doc in reversed(worst_first)]


class Coordinator:
    """Scatter-gather manager with a health table, failover and consistent global statistics.

    stats_mode:
      "cached" (default) statistics are collected from the shards ONCE (and re-collected
                         periodically). The corpus is static, so the cache is EXACT, and a
                         query needs only ONE round trip.
      "dfs"              round 1 (GetStats) before EVERY query, then round 2 (Search):
                         always exact, but two round trips and the tail is paid twice.
      "local"            EXPERIMENT ONLY: every shard scores with its OWN statistics, to
                         measure how far rankings drift from the single-node index.
    """

    def __init__(self, routing_path: str, stats_mode: str = "cached", verbose: bool = True,
                 heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S):
        if stats_mode not in ("cached", "dfs", "local"):
            raise ValueError("stats_mode must be cached, dfs or local")
        self.table = load_routing_table(routing_path)
        self.stats_mode = stats_mode
        self.verbose = verbose
        self.heartbeat_interval_s = heartbeat_interval_s
        self.copies_per_shard: int = self.table["copies_per_shard"]
        self.copies: Dict[int, List[ShardCopy]] = {}     # shard id -> its copies (the routing table)
        self.shard_ids: List[int] = [e["shard_id"] for e in self.table["shards"]]
        # ---- statistics kept PER SHARD as exact integers; global values are SUMS of them ----
        self.shard_totals: Dict[int, Tuple[int, int]] = {}      # shard -> (N_s, sum_dl_s)
        self.df_cache: Dict[str, Dict[int, int]] = {}           # term  -> {shard: df_s(t)}
        self._round_robin = 0
        self._tasks: List[asyncio.Task] = []
        self._color = "GREEN"
        self._started = False

    # ==================================================================================
    # START / STOP
    # ==================================================================================
    async def start(self) -> None:
        """Open all N x C channels BEFORE the first query, so no query pays a handshake."""
        # Channels belong to the event loop that creates them, so create them here (inside
        # the running loop), not in __init__.
        for entry in self.table["shards"]:
            self.copies[entry["shard_id"]] = [
                ShardCopy(entry["shard_id"], c["copy_id"], c["node"], c["address"])
                for c in entry["copies"]
            ]
        # The first roll call decides the initial whiteboard: answer -> UP, silence -> DOWN.
        await asyncio.gather(*(self._first_roll_call(c) for c in self._all_copies()))
        self._color = self.cluster_color()
        self._log("cluster is %s  (%d shards x %d copies, stats mode: %s)"
                  % (self._color, len(self.shard_ids), self.copies_per_shard, self.stats_mode))
        if self.stats_mode != "dfs":
            # Collect N_s and sum_dl_s from one copy of every shard, once, at start-up.
            await self._collect_stats([], self.shard_ids, asyncio.get_event_loop().time() + 2.0)
        self._tasks.append(asyncio.ensure_future(self._heartbeat_loop()))
        if self.stats_mode != "dfs":
            self._tasks.append(asyncio.ensure_future(self._stats_refresh_loop()))
        self._started = True

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks = []
        await asyncio.gather(*(c.channel.close() for c in self._all_copies()))

    def _all_copies(self) -> List[ShardCopy]:
        return [c for sid in self.shard_ids for c in self.copies[sid]]

    def _log(self, message: str) -> None:
        if self.verbose:
            print("[coordinator %s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)

    # ==================================================================================
    # THE WHITEBOARD: health states, heartbeats, cluster colour
    # ==================================================================================
    #
    #          missed heartbeat, or a failed query call
    #    UP -------------------------------------------> SUSPECT
    #     ^                                                 |
    #     | 2 answered heartbeats in a row                  | 3 missed heartbeats in a row
    #     |                                                 v
    #     +----------------------------------------------- DOWN
    #
    #   UP      : healthy, preferred for queries
    #   SUSPECT : something went wrong recently; used only if no UP copy is available
    #   DOWN    : confirmed failed; never sent queries (heartbeats still check on it)

    def _set_state(self, copy: ShardCopy, new_state: str, reason: str) -> None:
        if copy.state == new_state:
            return
        self._log("%s : %s -> %s  (%s)" % (copy.label(), copy.state, new_state, reason))
        copy.state = new_state
        color = self.cluster_color()
        if color != self._color:
            self._log("CLUSTER %s -> %s" % (self._color, color))
            self._color = color

    def cluster_color(self) -> str:
        """GREEN : every shard has all its copies working
        YELLOW: every shard has >= 1 working copy, but some copy is missing
                (redundancy is lost, NO data is lost -> queries are still COMPLETE)
        RED   : some shard has NO working copy -> queries are PARTIAL

        Only DOWN counts as "not working". SUSPECT does not change the colour, so a
        half-second network blip does not make the operator's dashboard flash yellow.
        """
        color = "GREEN"
        for sid in self.shard_ids:
            live = sum(1 for c in self.copies[sid] if c.state != DOWN)
            if live == 0:
                return "RED"
            if live < self.copies_per_shard:             # also true if the allocator left a copy unassigned
                color = "YELLOW"
        return color

    async def _probe(self, copy: ShardCopy) -> bool:
        """One heartbeat: "Room A, box S0, are you there?" -> "Here!" """
        try:
            reply = await copy.stub.Health(search_pb2.HealthRequest(), timeout=HEARTBEAT_TIMEOUT_S)
            return reply.serving and reply.shard_id == copy.shard_id
        except grpc.aio.AioRpcError:
            return False

    async def _first_roll_call(self, copy: ShardCopy) -> None:
        try:
            await asyncio.wait_for(copy.channel.channel_ready(), timeout=3.0)
        except asyncio.TimeoutError:
            pass                                         # start anyway: the copy may come up later
        if await self._probe(copy):
            copy.successes = UP_AFTER_SUCCESSES
        else:
            copy.state, copy.misses = DOWN, DOWN_AFTER_MISSES
            self._log("%s did not answer the first roll call -> DOWN" % copy.label())

    async def _heartbeat_one(self, copy: ShardCopy) -> None:
        if await self._probe(copy):
            copy.misses = 0
            copy.successes += 1
            if copy.state != UP and copy.successes >= UP_AFTER_SUCCESSES:
                self._set_state(copy, UP, "%d heartbeats answered in a row" % copy.successes)
        else:
            copy.successes = 0
            copy.misses += 1
            if copy.misses >= DOWN_AFTER_MISSES:
                self._set_state(copy, DOWN, "%d heartbeats missed in a row" % copy.misses)
            elif copy.state == UP:
                self._set_state(copy, SUSPECT, "missed a heartbeat")

    async def _heartbeat_loop(self) -> None:
        """SAFETY NET 2 (slow, a few seconds): keeps FUTURE queries away from dead copies
        and notices when a copy comes back. Heartbeats go to EVERY copy, DOWN ones too -
        that is the only way the coordinator learns that a restarted copy is alive again."""
        while True:
            await asyncio.sleep(self.heartbeat_interval_s)
            await asyncio.gather(*(self._heartbeat_one(c) for c in self._all_copies()))

    def _on_call_failure(self, copy: ShardCopy, method: str, code: grpc.StatusCode) -> None:
        """SAFETY NET 1 (instant): a REAL query is a free health check. If it fails, do not
        wait for the next roll call - mark the copy SUSPECT right now so later queries
        avoid it. The heartbeats then give the official verdict (DOWN or back to UP)."""
        copy.successes = 0
        if copy.state == UP:
            self._set_state(copy, SUSPECT, "%s failed with %s" % (method, code.name))

    # ==================================================================================
    # ROUTING + FAILOVER: one call to one SHARD, trying its copies in turn
    # ==================================================================================
    def _pick_copy(self, shard_id: int, already_tried: set) -> Optional[ShardCopy]:
        """LEAST-OUTSTANDING-REQUESTS routing.

        The coordinator cannot see a shard server's CPU. What it CAN see is its own tally
        of calls it sent to each copy that have not come back yet. Fewest outstanding =
        probably the shortest queue. Order of preference:
            1. UP before SUSPECT            (DOWN copies are never chosen)
            2. channel READY before not     (a broken tube is a bad first choice: the
                                             channel notices a dead connection at once,
                                             seconds before the heartbeats confirm it)
            3. fewer outstanding requests   (the smart traffic cop)
            4. rotate between equals        (spread the load when both are idle)
        """
        candidates = [c for c in self.copies[shard_id]
                      if c.state != DOWN and c.copy_id not in already_tried]
        if not candidates:
            return None
        self._round_robin += 1
        n = len(self.copies[shard_id])
        ready = grpc.ChannelConnectivity.READY
        return min(candidates, key=lambda c: (c.state != UP, c.channel.get_state() != ready,
                                              c.outstanding,
                                              (c.copy_id + self._round_robin) % n))

    async def _call_shard(self, shard_id: int, method: str, request, deadline: float) -> ShardResult:
        """Call `method` ("GetStats" or "Search") on ONE copy of `shard_id`.

        On a retryable failure try ANOTHER copy - but never more than MAX_ATTEMPTS copies
        (attempt budget) and never past the client's absolute deadline (time budget).
        Whichever budget runs out first ends the attempts for this shard.
        Retrying is safe because every call only READS immutable files (idempotent).
        """
        loop = asyncio.get_event_loop()
        result = ShardResult(shard_id)
        tried = set()
        for attempt in range(MAX_ATTEMPTS):
            copy = self._pick_copy(shard_id, tried)
            if copy is None:
                if not tried:
                    result.errors.append("no live copy (every copy is DOWN)")
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                result.errors.append("client time budget used up")
                break
            if attempt > 0:
                result.retries += 1
            tried.add(copy.copy_id)
            timeout = min(PER_ATTEMPT_TIMEOUT_S, remaining)
            copy.outstanding += 1
            try:
                # `await` parks this task; the event loop serves other queries meanwhile.
                # gRPC turns the timeout into a deadline that travels WITH the request.
                result.reply = await getattr(copy.stub, method)(request, timeout=timeout)
                return result                                        # success
            except grpc.aio.AioRpcError as err:
                code = err.code()
                result.errors.append("copy %d (%s): %s" % (copy.copy_id, copy.address, code.name))
                if code not in RETRYABLE:
                    break               # e.g. INVALID_ARGUMENT: a bug, every copy would refuse it
                budget_ended = (code == grpc.StatusCode.DEADLINE_EXCEEDED
                                and timeout < PER_ATTEMPT_TIMEOUT_S)
                if not budget_ended:    # do not blame the copy if the CLIENT's budget ran out
                    self._on_call_failure(copy, method, code)
            finally:
                copy.outstanding -= 1
        return result                                                # every attempt failed

    # ==================================================================================
    # CONSISTENT GLOBAL STATISTICS  (exact integer sums, ONE division)
    # ==================================================================================
    async def _collect_stats(self, terms: List[str], shard_ids: List[int],
                             deadline: float) -> Dict[int, ShardResult]:
        """ROUND 1: ask ONE COPY of each shard for N_s, sum_dl_s and df_s(t).

        One copy per shard, never every copy: replicas hold the SAME documents, so counting
        both copies would double that shard's df and ruin the statistics.
        """
        results = await asyncio.gather(*(
            self._call_shard(sid, "GetStats",
                             search_pb2.StatsRequest(shard_id=sid, terms=terms), deadline)
            for sid in shard_ids))
        for r in results:
            if r.ok:
                self.shard_totals[r.shard_id] = (r.reply.doc_count, r.reply.total_doc_length)
                freqs = dict(r.reply.doc_freqs)
                for t in terms:
                    self.df_cache.setdefault(t, {})[r.shard_id] = freqs.get(t, 0)
        return {r.shard_id: r for r in results}

    def _has_stats(self, shard_id: int, terms: List[str]) -> bool:
        return shard_id in self.shard_totals and all(
            shard_id in self.df_cache.get(t, {}) for t in terms)

    def _global_stats(self, terms: List[str], shard_ids: List[int]) -> Tuple[int, float, Dict[str, int]]:
        """N = SUM N_s        df(t) = SUM df_s(t)        avgdl = (SUM sum_dl_s) / (SUM N_s)

        NEVER "the average of the shards' averages": that gives every shard the same weight
        regardless of its size. We add EXACT INTEGERS and divide ONCE - which reproduces,
        bit for bit, the avgdl a single node would compute over the whole collection.
        """
        n = sum(self.shard_totals[s][0] for s in shard_ids)
        total_length = sum(self.shard_totals[s][1] for s in shard_ids)
        avgdl = total_length / n if n else 0.0                       # the ONE division
        df = {t: sum(self.df_cache[t][s] for s in shard_ids) for t in terms}
        return n, avgdl, df

    async def _stats_refresh_loop(self) -> None:
        """ "...or refreshed periodically": re-collect the cached statistics every 30 s.
        With a static corpus the numbers never change, so this only proves the cache is
        still exact (and would pick up a rebuilt index)."""
        while True:
            await asyncio.sleep(STATS_REFRESH_S)
            terms = sorted(self.df_cache)
            deadline = asyncio.get_event_loop().time() + 2.0
            for i in range(0, max(len(terms), 1), 2000):             # modest message sizes
                await self._collect_stats(terms[i:i + 2000], self.shard_ids, deadline)

    # ==================================================================================
    # THE QUERY: analyse -> statistics -> scatter -> gather + merge -> respond
    # ==================================================================================
    async def search(self, query: str, k: int = 10, timeout_s: float = CLIENT_TIMEOUT_S,
                     cursor: Optional[Tuple[float, int]] = None) -> dict:
        started = time.perf_counter()                    # high-resolution stopwatch for took_ms
        # ONE absolute deadline for the whole query. Both rounds and every retry share it,
        # so failover can never overrun the client's budget.
        deadline = asyncio.get_event_loop().time() + (timeout_s if timeout_s > 0 else CLIENT_TIMEOUT_S)
        k = k if k > 0 else 10

        # STEP 2 - ANALYSE ONCE. The coordinator sends TERMS, not raw text, so no shard can
        # analyse differently. Sorted -> fixed addition order -> bit-identical scores.
        terms = query_terms(query)
        failed: Dict[int, List[str]] = {}
        retries = 0
        if not terms:                                    # nothing searchable (e.g. only stop words)
            return self._reply([], "COMPLETE", failed, started, terms, 0.0, 0, len(self.shard_ids))

        # STEP 3 - GLOBAL STATISTICS
        if self.stats_mode == "dfs":
            # Two round trips per query: stats first, then search.
            round1 = await self._collect_stats(terms, self.shard_ids, deadline)
            for sid, r in round1.items():
                retries += r.retries
                if not r.ok:
                    failed[sid] = r.errors
            # Statistics are summed over the shards that ANSWERED, and only those shards are
            # searched (a shard that cannot count cannot search either).
            search_shards = [sid for sid in self.shard_ids if round1[sid].ok]
        else:
            # Cached: fetch only what is missing (a term never seen before), then ONE round.
            missing = [sid for sid in self.shard_ids if not self._has_stats(sid, terms)]
            if missing:
                round1 = await self._collect_stats(terms, missing, deadline)
                for sid, r in round1.items():
                    retries += r.retries
                    if not r.ok:
                        failed[sid] = r.errors + ["no statistics available for this shard"]
            search_shards = [sid for sid in self.shard_ids if self._has_stats(sid, terms)]

        if not search_shards:
            return self._reply([], "FAILED", failed, started, terms, 0.0, retries, 0)
        n, avgdl, df = self._global_stats(terms, search_shards)

        def build_request(shard_id: int) -> search_pb2.SearchRequest:
            if self.stats_mode == "local":
                # EXPERIMENT: hand each shard ITS OWN numbers -> every shard scores in its
                # own "currency". Shows why global statistics are required.
                n_s, sum_dl_s = self.shard_totals[shard_id]
                stats = (n_s, (sum_dl_s / n_s if n_s else 0.0),
                         {t: self.df_cache[t][shard_id] for t in terms})
            else:
                stats = (n, avgdl, df)
            return search_pb2.SearchRequest(
                shard_id=shard_id, terms=terms, k=k,
                global_doc_count=stats[0], global_avg_doc_length=stats[1],
                global_doc_freqs=stats[2],                           # they ride INSIDE the request
                has_cursor=cursor is not None,
                cursor_score=cursor[0] if cursor else 0.0,
                cursor_doc_id=cursor[1] if cursor else 0,
            )

        # STEP 5 - SCATTER: one task per shard, all in flight at the same time.
        # Latency becomes the MAX of the shard times instead of their SUM.
        tasks = [asyncio.ensure_future(self._call_shard(sid, "Search", build_request(sid), deadline))
                 for sid in search_shards]

        # STEP 6 - GATHER + MERGE (Method A): handle each reply the moment it arrives.
        # By the time the slowest shard answers, the others are already merged.
        merger = TopKMerger(k)
        successful = 0
        slowest_shard_ms = 0.0
        for finished in asyncio.as_completed(tasks):
            result = await finished
            retries += result.retries
            if result.ok:
                successful += 1
                merger.offer(result.reply.hits)          # no re-scoring: the coordinator only COMPARES
                slowest_shard_ms = max(slowest_shard_ms, result.reply.took_ms)
            else:
                failed[result.shard_id] = result.errors  # every copy tried for this shard failed

        # STEP 9 - RESPOND, and be honest about completeness:
        #   answered == all shards -> COMPLETE (even if a replica had to step in)
        #   answered <  all shards -> PARTIAL  + which shards are missing and why
        #   answered == 0          -> FAILED
        if not failed:
            status = "COMPLETE"
        elif successful > 0:
            status = "PARTIAL"
        else:
            status = "FAILED"
        return self._reply(merger.result(), status, failed, started, terms, slowest_shard_ms,
                           retries, successful)

    def _reply(self, hits, status, failed, started, terms, slowest_shard_ms, retries,
               successful) -> dict:
        return {
            "hits": hits,                                # [(global doc id, score), ...] best first
            "status": status,
            "shards_total": len(self.shard_ids),
            "shards_successful": successful,
            "failed_shards": {sid: failed[sid] for sid in sorted(failed)},
            "took_ms": (time.perf_counter() - started) * 1000.0,
            "terms": terms,
            "cluster_color": self._color,
            "slowest_shard_ms": slowest_shard_ms,
            "retries": retries,
        }

    def health_report(self) -> dict:
        return {
            "color": self.cluster_color(),
            "copies": [{"shard_id": c.shard_id, "copy_id": c.copy_id, "node": c.node,
                        "address": c.address, "state": c.state, "misses": c.misses,
                        "outstanding": c.outstanding} for c in self._all_copies()],
        }


# ======================================================================================
# THE FRONT DESK: the client-facing gRPC service (runs on the SAME event loop)
# ======================================================================================
class CoordinatorService(search_pb2_grpc.CoordinatorServiceServicer):
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator

    async def Query(self, request, context):
        cursor = (request.cursor_score, request.cursor_doc_id) if request.has_cursor else None
        reply = await self.coordinator.search(request.query, request.k,
                                              request.timeout_ms / 1000.0, cursor)
        return search_pb2.QueryReply(
            hits=[search_pb2.Hit(doc_id=d, score=s) for d, s in reply["hits"]],
            # PARTIAL is a FIELD of a successful reply, not a gRPC error: a partial answer
            # is still a useful answer, as long as it SAYS it is partial.
            status=search_pb2.ResultStatus.Value(reply["status"]),
            shards_total=reply["shards_total"],
            shards_successful=reply["shards_successful"],
            failed_shards=[search_pb2.FailedShard(shard_id=sid, errors=errs)
                           for sid, errs in reply["failed_shards"].items()],
            took_ms=reply["took_ms"],
            terms=reply["terms"],
            cluster_color=reply["cluster_color"],
            slowest_shard_ms=reply["slowest_shard_ms"],
            retries=reply["retries"],
        )

    async def ClusterHealth(self, request, context):
        report = self.coordinator.health_report()
        return search_pb2.ClusterHealthReply(
            color=report["color"],
            copies=[search_pb2.CopyHealth(**c) for c in report["copies"]])


async def _serve(routing_path: str, port: int, stats_mode: str, verbose: bool,
                 bind: str = HOST) -> None:
    coordinator = Coordinator(routing_path, stats_mode=stats_mode, verbose=verbose)
    await coordinator.start()
    server = grpc.aio.server()
    search_pb2_grpc.add_CoordinatorServiceServicer_to_server(CoordinatorService(coordinator), server)
    # bind = 127.0.0.1: only clients on this laptop. bind = 0.0.0.0: clients (e.g. a load
    # generator) on other laptops can reach the front desk too.
    address = "%s:%d" % (bind, port)
    if server.add_insecure_port(address) == 0:
        raise SystemExit("could not bind %s" % address)
    await server.start()
    coordinator._log("front desk open on %s  (Ctrl+C to stop)" % address)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(0)
        await coordinator.stop()


def run_server(routing_path: str, port: int = DEFAULT_COORDINATOR_PORT,
               stats_mode: str = "cached", verbose: bool = True, bind: str = HOST) -> None:
    """Process entry point (also used by benchmark.py to run the coordinator as its own process)."""
    try:
        asyncio.run(_serve(routing_path, port, stats_mode, verbose, bind))
    except KeyboardInterrupt:
        pass


def print_reply(reply: dict) -> None:
    print("terms %s | status %s | shards %d/%d | cluster %s | %.2f ms (slowest shard %.2f ms, retries %d)"
          % (reply["terms"], reply["status"], reply["shards_successful"], reply["shards_total"],
             reply["cluster_color"], reply["took_ms"], reply["slowest_shard_ms"], reply["retries"]))
    for sid, errors in reply["failed_shards"].items():
        print("  FAILED shard %d: %s" % (sid, "; ".join(errors)))
    for rank, (doc_id, score) in enumerate(reply["hits"], 1):
        print("  %2d. doc %-8d score %.6f" % (rank, doc_id, score))


async def _one_shot(routing_path: str, stats_mode: str, query: str, k: int) -> None:
    coordinator = Coordinator(routing_path, stats_mode=stats_mode)
    await coordinator.start()
    try:
        print_reply(await coordinator.search(query, k))
    finally:
        await coordinator.stop()


def main():
    ap = argparse.ArgumentParser(description="The coordinator (asyncio scatter-gather).")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--routing", default="", help="explicit path to a routing_table.json")
    ap.add_argument("--port", type=int, default=DEFAULT_COORDINATOR_PORT)
    ap.add_argument("--bind", default=HOST,
                    help="front-desk interface; 0.0.0.0 lets clients on other laptops connect")
    ap.add_argument("--stats-mode", choices=["cached", "dfs", "local"], default="cached")
    ap.add_argument("--query", default="", help="run ONE query and exit (no server)")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    path = args.routing or routing_table_path(args.shards, args.copies, args.tag)
    if not os.path.exists(path):
        sys.exit("no routing table at %s - run build_index.py first" % path)
    if args.query:
        asyncio.run(_one_shot(path, args.stats_mode, args.query, args.k))
    else:
        run_server(path, args.port, args.stats_mode, bind=args.bind)


if __name__ == "__main__":
    main()
