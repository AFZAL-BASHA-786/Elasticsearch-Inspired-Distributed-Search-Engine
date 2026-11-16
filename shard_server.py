"""
shard_server.py : ONE shard copy = ONE operating-system process (the Assistant Librarian).

Guide for this file : ../docs/code_guides/05_shard_server_GUIDE.md
My notes behind it  : 02_Medium_Architecture_and_Design/01_Inverted_Indexes_and_Memory_Mapping.md
                      02_Medium_Architecture_and_Design/02_Concurrency_vs_Parallelism.md
                      03_In_Depth_Implementation/01_Consistent_Ranking_with_BM25.md
                      (all under  self learning/learning_guide/)

A plain Python script becomes a "shard server" when it does three things
(notes: doubts 02_Medium.../01_Inverted_Indexes..., "how a program becomes a shard server"):

  1. CLAIMS A SLICE OF DATA : memory-maps (mmap) the .npy files of ITS shard copy.
  2. OPENS A NETWORK PORT   : a gRPC server waits for the coordinator's request forms.
  3. DOES LOCAL MATH        : looks terms up in ITS lexicon, slices ITS arrays, computes BM25
                              with the GLOBAL statistics the coordinator attached, and
                              returns ITS local top-k.

WHY ONE PROCESS PER SHARD COPY (and not threads)?
  Scoring is CPU-bound. Inside one Python process only the thread holding the GIL (the
  "talking stick") can run Python code, so 4 shard threads would take turns on ~1 core.
  Separate processes = separate interpreters = separate GILs = true parallelism.
  Bonus: killing one process IS the brief's "one unavailable shard replica".

Run one copy by hand:
    python shard_server.py --shard-id 0 --copy-id 0 --port 20051 ^
           --index-dir index_data/shards_3_copies_2/node_A/shard_0_copy_0
Normally cluster.py starts one such process per row of routing_table.json.
"""

import os

# MUST be set BEFORE numpy is imported. NumPy's math libraries would otherwise start their
# own hidden threads in EVERY shard process: 8 processes x 8 hidden threads fighting over
# 8 cores ("a CPU riot"). We give up library-level parallelism to get predictable
# process-level parallelism: 1 shard process = 1 thread = 1 core.
# (notes: doubts 02_Medium.../02_Concurrency_vs_Parallelism.md, "OMP_NUM_THREADS=1")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                 # noqa: E402
import json                     # noqa: E402
import math                     # noqa: E402
import time                     # noqa: E402
from concurrent import futures  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import grpc                     # noqa: E402
import numpy as np              # noqa: E402

import search_pb2               # noqa: E402
import search_pb2_grpc          # noqa: E402
from common import B, K1        # noqa: E402


# ======================================================================================
# PART 1: THE STORAGE LAYER  (memory-mapped, read-only, immutable)
# ======================================================================================
def _mmap(path: str) -> np.ndarray:
    """Memory-map a .npy file READ-ONLY.

    Nothing is read from disk here. The OS only reserves virtual addresses and says "these
    addresses ARE the file". The first touch of a 4 KB page raises a PAGE FAULT, the OS
    pulls that page into its PAGE CACHE, and every later read runs at RAM speed.
      * no copy: the NumPy array IS the page-cache memory (it is not a Python heap object,
        so the garbage collector never has to walk millions of postings)
      * shared : every process on the node that maps the same file shares the same pages
    """
    try:
        return np.load(path, mmap_mode="r")
    except ValueError:
        # An EMPTY array cannot be memory-mapped (there are zero bytes to map). This only
        # happens for a shard that received no documents (tiny corpora, many shards).
        return np.load(path)


class ShardIndex:
    """One shard copy, opened for searching.

    ON DISK (memory-mapped, the big ones):  doc_ids, term_freqs, doc_lengths, doc_global_ids
    IN RAM  (small):                         the LEXICON and the two numbers of shard_stats.json

    The same class is used by the single-node baseline (compare_rankings.py loads the
    1-shard build with it), so the comparison is honest: there is NO separate scoring code.
    """

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        with open(os.path.join(index_dir, "shard_stats.json"), "r", encoding="utf-8") as f:
            stats = json.load(f)
        self.shard_id: int = stats["shard_id"]
        self.doc_count: int = stats["doc_count"]                 # N_s      (exact integer)
        self.total_doc_length: int = stats["total_doc_length"]   # sum_dl_s (exact integer)

        # The long "paper tapes": every posting list glued end to end.
        self.doc_ids = _mmap(os.path.join(index_dir, "doc_ids.npy"))            # local doc ids
        self.term_freqs = _mmap(os.path.join(index_dir, "term_freqs.npy"))      # tf per posting
        self.doc_lengths = _mmap(os.path.join(index_dir, "doc_lengths.npy"))    # dl per local doc
        self.doc_global_ids = _mmap(os.path.join(index_dir, "doc_global_ids.npy"))  # local -> global

        # THE LEXICON (in RAM): term -> (offset, length).
        #   offset = where this term's slice starts on the tapes
        #   length = how many postings it has = df_s(t)  ("df for free": never stored separately)
        # On disk it is the sorted vocabulary + the chained bookmark tape offsets.npy:
        #   start = offsets[term_id]      end = offsets[term_id + 1]      length = end - start
        offsets = np.load(os.path.join(index_dir, "offsets.npy"))
        with open(os.path.join(index_dir, "vocabulary.txt"), "r", encoding="utf-8") as f:
            vocabulary = f.read().split("\n") if offsets.size > 1 else []
        self.lexicon: Dict[str, Tuple[int, int]] = {
            term: (int(offsets[i]), int(offsets[i + 1] - offsets[i]))
            for i, term in enumerate(vocabulary)
        }

        # The ACCUMULATOR: one float per local document, reused for every query.
        # For a 17,000-document shard this is 136 KB. Only the touched slots are reset.
        self._scores = np.zeros(self.doc_count, dtype=np.float64)

    # ---------------------------------------------------------------- statistics (round 1)
    def doc_freq(self, term: str) -> int:
        """df_s(t) = length of the term's posting list. 0 if this shard never saw the term."""
        entry = self.lexicon.get(term)
        return entry[1] if entry else 0

    # ---------------------------------------------------------------- search (round 2)
    def search(self, terms: List[str], k: int, global_doc_count: int,
               global_avg_doc_length: float, global_doc_freqs: Dict[str, int],
               cursor: Optional[Tuple[float, int]] = None) -> List[Tuple[int, float]]:
        """Return this shard's local top-k as [(GLOBAL doc id, score), ...], best first.

        BM25 for one document d and query q  (notes: BM25 file, section 2):

            score(d, q) = SUM over t in q of   idf(t) * tf * (k1 + 1) / (tf + k1 * L(d))

            idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))      rarity of the term  (GLOBAL N, df)
            L(d)   = 1 - b + b * dl(d) / avgdl                length multiplier   (GLOBAL avgdl)

        Who knows what:
            tf, dl        -> facts about ONE document: this shard knows them exactly (local)
            N, df, avgdl  -> facts about the WHOLE collection: no shard knows them, so the
                             coordinator sums exact integers from all shards and sends the
                             result inside the request. Every shard then scores in ONE
                             "currency" and a document gets the SAME score no matter which
                             shard holds it or how many shards exist.
        """
        if self.doc_count == 0 or global_avg_doc_length <= 0:
            return []
        scores = self._scores
        touched: List[np.ndarray] = []

        # `terms` arrives SORTED from the coordinator. We add each document's term
        # contributions in exactly this order on every shard (and on the single node), so
        # the floating-point rounding is identical everywhere -> bit-identical scores.
        for term in terms:
            entry = self.lexicon.get(term)               # 1. one hash lookup in RAM
            if entry is None:
                continue                                 # term absent here: contributes nothing
            df = global_doc_freqs.get(term, 0)
            if df <= 0:
                continue
            offset, length = entry
            # 2. SLICE the tapes. The cost depends only on THIS posting list's length, not
            #    on how large the rest of the index is: predictable read times, which is
            #    what tail-latency measurements need. (Page faults happen here if cold.)
            docs = self.doc_ids[offset:offset + length]
            tf = self.term_freqs[offset:offset + length].astype(np.float64)

            # 3. idf: how surprising is a match on this term?  (one number per term)
            idf = math.log(1.0 + (global_doc_count - df + 0.5) / (df + 0.5))

            # 4. L(d): judge the count relative to the document's length. avgdl is the
            #    YARDSTICK - it must be the same global number on every shard.
            dl = self.doc_lengths[docs].astype(np.float64)
            length_multiplier = 1.0 - B + B * (dl / global_avg_doc_length)

            # 5. saturated, length-adjusted count: can never exceed k1 + 1 = 2.2, so
            #    repeating a word 100 times cannot beat matching a NEW query term.
            # Whole-array NumPy operations run in compiled C (no Python loop over postings).
            scores[docs] += idf * (tf * (K1 + 1.0)) / (tf + K1 * length_multiplier)
            touched.append(docs)

        if not touched:
            return []
        # Candidates = documents that matched at least one term. Work is proportional to
        # the posting-list lengths; documents that match nothing are never looked at.
        candidates = touched[0] if len(touched) == 1 else np.unique(np.concatenate(touched))
        candidates = np.asarray(candidates)
        cand_scores = scores[candidates]
        scores[candidates] = 0.0                         # reset the accumulator for the next query

        # search_after cursor ("bookmark"): keep only hits strictly WORSE than the last hit
        # of the previous page, i.e. lower score, or equal score and larger doc id.
        if cursor is not None:
            c_score, c_doc = cursor
            cand_global = self.doc_global_ids[candidates]
            keep = (cand_scores < c_score) | ((cand_scores == c_score) & (cand_global > c_doc))
            candidates, cand_scores = candidates[keep], cand_scores[keep]
            if candidates.size == 0:
                return []

        # LOCAL TOP-K without sorting everything:
        #   find the k-th best score, keep everything at or above it (this keeps ALL ties
        #   at the boundary), then fully order only those few.
        if candidates.size > k:
            kth_score = np.partition(cand_scores, candidates.size - k)[candidates.size - k]
            keep = cand_scores >= kth_score
            candidates, cand_scores = candidates[keep], cand_scores[keep]
        # TOTAL ORDER: score DESCENDING, ties -> smaller doc id first. Local ids were handed
        # out in increasing global-id order, so "smaller local id" == "smaller global id".
        order = np.lexsort((candidates, -cand_scores))[:k]
        top_local = candidates[order]
        top_global = self.doc_global_ids[top_local]      # translate local ids -> GLOBAL ids
        return [(int(g), float(s)) for g, s in zip(top_global, cand_scores[order])]


# ======================================================================================
# PART 2: THE gRPC SERVICE  (the Assistant Librarian's mail slot)
# ======================================================================================
class ShardService(search_pb2_grpc.ShardServiceServicer):
    def __init__(self, index: ShardIndex, copy_id: int):
        self.index = index
        self.copy_id = copy_id

    def _check_shard(self, request_shard_id: int, context) -> None:
        # Protobuf already checked the SHAPE of the form (right fields, right types).
        # The shard checks the LOGIC: "is this form really for me?". A request for shard 2
        # delivered to shard 1 is a routing BUG, and every copy would refuse it too, so the
        # coordinator must NOT retry it (INVALID_ARGUMENT is not retryable).
        if request_shard_id != self.index.shard_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          "this is shard %d, request was for shard %d"
                          % (self.index.shard_id, request_shard_id))

    def GetStats(self, request, context):
        """Round 1: report EXACT INTEGER counts - never an average, never an idf."""
        self._check_shard(request.shard_id, context)
        return search_pb2.StatsReply(
            shard_id=self.index.shard_id,
            doc_count=self.index.doc_count,                      # N_s
            total_doc_length=self.index.total_doc_length,        # sum_dl_s
            doc_freqs={t: self.index.doc_freq(t) for t in request.terms},   # df_s(t)
        )

    def Search(self, request, context):
        """Round 2: my local top-k, scored with the global statistics in the request."""
        self._check_shard(request.shard_id, context)
        if request.k == 0:
            # In proto3 "k = 0" and "k was never set" look identical, so 0 is invalid.
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "k must be > 0")
        if not context.is_active():
            # The request waited in my queue until its deadline passed (or the coordinator
            # cancelled it). Nobody is waiting for the answer: do not waste CPU on it.
            return search_pb2.SearchReply(shard_id=self.index.shard_id)

        started = time.perf_counter()
        cursor = (request.cursor_score, request.cursor_doc_id) if request.has_cursor else None
        hits = self.index.search(list(request.terms), request.k, request.global_doc_count,
                                 request.global_avg_doc_length, dict(request.global_doc_freqs),
                                 cursor)
        took_ms = (time.perf_counter() - started) * 1000.0
        return search_pb2.SearchReply(
            shard_id=self.index.shard_id,
            hits=[search_pb2.Hit(doc_id=d, score=s) for d, s in hits],   # scores travel as raw 8 bytes
            took_ms=took_ms,
        )

    def Health(self, request, context):
        """Heartbeat reply: "Here!" """
        return search_pb2.HealthReply(shard_id=self.index.shard_id, copy_id=self.copy_id,
                                      serving=True)


def _exit_with_launcher() -> None:
    """If this process was started by cluster.py (multiprocessing), exit when cluster.py dies.

    On Windows, child processes outlive a launcher that is force-killed. Without this check
    a crashed launcher would leave "zombie" shard servers that keep answering, so a node
    that should be gone would still look alive. A tiny background thread checks once a
    second whether the launcher is still there.
    """
    import multiprocessing
    import threading
    parent = multiprocessing.parent_process()
    if parent is None:                                   # started by hand: nothing to watch
        return

    def watch():
        while parent.is_alive():
            time.sleep(1.0)
        os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


def serve(shard_id: int, copy_id: int, port: int, index_dir: str, host: str = "127.0.0.1",
          quiet: bool = False) -> None:
    """Entry point of one shard server PROCESS. Blocks until the process is killed."""
    index = ShardIndex(index_dir)
    if index.shard_id != shard_id:
        raise SystemExit("index in %s is shard %d, not %d" % (index_dir, index.shard_id, shard_id))

    server = grpc.server(
        # max_workers = 1: this shard copy computes ONE query at a time; requests that arrive
        # while it is busy WAIT IN ITS QUEUE (notes: Concurrency file, section 6.3). That
        # queueing delay is real and is supposed to show up in the throughput/tail results.
        # More throughput comes from more COPIES, not from more threads (GIL).
        futures.ThreadPoolExecutor(max_workers=1),
        # If more than this many requests are queued here, refuse new ones immediately with
        # RESOURCE_EXHAUSTED so the coordinator can try the other copy instead of waiting.
        maximum_concurrent_rpcs=256,
        options=[
            # Allow the coordinator's keep-alive pings on an idle channel; by default a
            # gRPC server treats frequent pings from an idle client as abuse.
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.min_ping_interval_without_data_ms", 5000),
        ],
    )
    search_pb2_grpc.add_ShardServiceServicer_to_server(ShardService(index, copy_id), server)
    address = "%s:%d" % (host, port)
    if server.add_insecure_port(address) == 0:
        raise SystemExit("could not bind %s (port already in use?)" % address)
    server.start()
    _exit_with_launcher()
    if not quiet:
        print("[shard %d copy %d] pid %d listening on %s | N_s=%d sum_dl_s=%d terms=%d (mmap %s)"
              % (shard_id, copy_id, os.getpid(), address, index.doc_count,
                 index.total_doc_length, len(index.lexicon), index_dir), flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Serve ONE shard copy (one process per copy).")
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--copy-id", type=int, default=0)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--index-dir", required=True, help="folder holding this copy's .npy files")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    serve(args.shard_id, args.copy_id, args.port, args.index_dir, args.host)
