"""
common.py : the few things that EVERY program in this project must agree on.

Guide for this file : ../docs/code_guides/02_common_GUIDE.md

Why a shared file?
    The indexer, the shard servers, the coordinator and the single-node baseline must use
      * the SAME analyser  (how text becomes terms)
      * the SAME hash      (which shard a document belongs to)
      * the SAME constants (k1, b)
      * the SAME ordering  (score descending, then doc_id ascending)
    If any one of them disagreed, the distributed ranking would stop matching the
    single-node ranking for no real reason.
"""

import json
import os
import re
from typing import Dict, List

# --------------------------------------------------------------------------------------
# BM25 constants (notes: 03_In_Depth_Implementation/01_Consistent_Ranking_with_BM25.md)
# --------------------------------------------------------------------------------------
K1 = 1.2    # term saturation dial. Ceiling of the tf part is k1 + 1 = 2.2 (stops keyword stuffing)
B = 0.75    # length normalisation dial. 0 = ignore length, 1 = fully proportional

# --------------------------------------------------------------------------------------
# Where things live on disk
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
INDEX_ROOT = os.path.join(HERE, "index_data")
DEFAULT_CORPUS = os.path.join(DATA_DIR, "corpus.jsonl")
DEFAULT_QUERIES = os.path.join(DATA_DIR, "queries.txt")

# Why not gRPC's customary 50051? On Windows the ports 49152-65535 are the DYNAMIC range the
# OS hands out as source ports for OUTGOING connections. A shard server asking for 50081
# can find it already taken by some unrelated connection and fail to start (this really
# happened during benchmarking). Ports around 20000 are outside the dynamic range on both
# Windows and Linux.
DEFAULT_BASE_PORT = 20051          # first shard server port; copies take 20051, 20052, ...
DEFAULT_COORDINATOR_PORT = 20050   # the Head Librarian's front desk
HOST = "127.0.0.1"


def layout_dir(num_shards: int, copies: int = 2, tag: str = "") -> str:
    """Folder that holds ONE complete build, e.g. index_data/shards_3_copies_2/.

    The shard count is frozen inside one build. To "change the shard count" we build a
    fresh folder for each N (notes: doubts 01_Basics/01_What_Is..., Doubt 3:
    "N is fixed within one run, and changes BETWEEN runs").
    """
    name = "shards_%d_copies_%d" % (num_shards, copies)
    if tag:
        name += "_" + tag
    return os.path.join(INDEX_ROOT, name)


def routing_table_path(num_shards: int, copies: int = 2, tag: str = "") -> str:
    return os.path.join(layout_dir(num_shards, copies, tag), "routing_table.json")


def load_routing_table(path: str) -> dict:
    """The routing table is the ONE source of truth for "which copy lives where".

    The allocator writes it (build_index.py), the launcher reads it to start processes
    (cluster.py) and the coordinator reads it to know whom to call (coordinator.py).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# The analyser: raw text -> list of terms
# (notes: 02_Medium.../01_Inverted_Indexes_and_Memory_Mapping.md, section 2 "Analysis")
#
# RULE: the query MUST go through exactly the same analyser as the documents.
# If documents were lowercased and the query was not, "Raft" would match nothing.
# --------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stop words are bundled here (no download needed). They are dropped at index time AND at
# query time, so they are also NOT counted in document lengths (dl, avgdl).
STOP_WORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is it its "
    "of on or she that the their them they this to was we were will with you your".split()
)


def analyze(text: str) -> List[str]:
    """ "Raft, Consensus & the Protocol!" -> ["raft", "consensus", "protocol"]

    Steps: lowercase -> split on anything that is not a letter/digit -> drop stop words.
    Duplicates are KEPT here (the indexer needs them to count term frequency).
    """
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in STOP_WORDS]


def query_terms(text: str) -> List[str]:
    """Analyse a QUERY: same analyser, then de-duplicate and SORT.

    Why sorted?  Floating-point addition is not associative:
        (0.1 + 0.2) + 0.3 = 0.6000000000000001     but     0.1 + (0.2 + 0.3) = 0.6
    If every shard adds a document's term contributions in the SAME (alphabetical) order,
    every shard and the single-node baseline do the same rounding, so scores come out
    bit-for-bit identical (notes: 01_Basics/03_Why_Scoring_Gets_Messy.md, section 10).

    Query semantics chosen (the brief leaves them open):
      * OR semantics  : a document matches if it contains ANY query term
      * repeated terms: counted once ("raft raft" == "raft")
    """
    return sorted(set(analyze(text)))


# --------------------------------------------------------------------------------------
# The dealing rule:  shard_of(doc) = murmur3(doc_id) mod N
# (notes: 01_Basics/01_What_Is_A_Distributed_Search_Engine.md, section 4.1)
#
# TRAP avoided: Python's built-in hash() on strings is randomised per process
# (PYTHONHASHSEED), so the indexer and another process would disagree about where a
# document lives. murmur3 is a fixed, stable hash - the family Elasticsearch uses.
# It is written out here in pure Python so the project needs no extra package.
# --------------------------------------------------------------------------------------
def murmur3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3, x86 32-bit variant. Returns an unsigned 32-bit integer."""
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h = seed & 0xFFFFFFFF
    n = len(data)
    rounded = n - (n % 4)
    for i in range(0, rounded, 4):                       # body: 4 bytes at a time
        k = int.from_bytes(data[i:i + 4], "little")
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    tail = data[rounded:]                                # tail: the last 1-3 bytes
    if tail:
        k = int.from_bytes(tail, "little")
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
    h ^= n                                               # finalisation ("avalanche")
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def shard_of(doc_id: int, num_shards: int) -> int:
    """Which shard does this WHOLE document go to?  (A document is never cut into pieces.)

    Used ONCE, at index-build time. At query time the hash is NOT used: we search by words,
    and a word tells us nothing about which shard holds it, so every shard is asked.
    """
    return murmur3_32(str(doc_id).encode("utf-8")) % num_shards


# --------------------------------------------------------------------------------------
# The total order of hits:  score DESCENDING, then doc_id ASCENDING
# (notes: 03_In_Depth/02_k_Way_Heap_Merges.md, section 1.1)
#
# As a Python tuple:  key = (score, -doc_id).  A BIGGER key means a BETTER hit:
#     higher score wins; on equal score the smaller doc_id gives the bigger -doc_id.
# Both the shard (sorting its local top-k) and the coordinator (merging) use this rule,
# so the global top-k is UNIQUE - it never depends on which shard replied first.
# --------------------------------------------------------------------------------------
def hit_key(score: float, doc_id: int):
    return (score, -doc_id)


# --------------------------------------------------------------------------------------
# Percentiles for latency reports (notes: 03_In_Depth/04_Tail_at_Scale_Latency.md, sec. 1)
# --------------------------------------------------------------------------------------
def percentile(sorted_values: List[float], p: float) -> float:
    """Nearest-rank percentile on an ALREADY SORTED list.

    p99 of 3,000 values = the 2,970th value in sorted order. No averaging, no max/min:
    we point at the boundary element (notes: doubts/02_03_deadlines.md, "3000 queries").
    Always pool the RAW latencies and call this once - never average two p99 values.
    """
    if not sorted_values:
        return float("nan")
    rank = int(-(-p * len(sorted_values) // 100))        # ceil(p/100 * n)
    rank = min(max(rank, 1), len(sorted_values))
    return sorted_values[rank - 1]


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    s = sorted(latencies_ms)
    if not s:
        return {"count": 0}
    return {
        "count": len(s),
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "mean": sum(s) / len(s),
        "max": s[-1],
    }
