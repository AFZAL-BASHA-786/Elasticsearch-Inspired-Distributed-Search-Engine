"""
compare_rankings.py : EXPERIMENT 1 - "Compare rankings against a single-node index".

Guide for this file : ../docs/code_guides/10_compare_rankings_GUIDE.md
My notes behind it  : 01_Basics_and_Intuition/03_Why_Scoring_Gets_Messy.md
                      03_In_Depth_Implementation/01_Consistent_Ranking_with_BM25.md (sec. 7, 8)
                      03_In_Depth_Implementation/02_k_Way_Heap_Merges.md  (sec. 2, exactness proof)

Three things are compared for every query:

  A. THE ANSWER KEY (single node): the whole corpus in ONE shard, searched in-process with
     the SAME ShardIndex class the shard servers use. No sharding, no network, no merge.
  B. AN INDEPENDENT CHECK: a few queries are also scored by a tiny brute-force BM25 written
     straight from the formula with plain Python dictionaries. If the flat-array code had a
     bug, A and the cluster would agree with each other but not with this.
  C. THE DISTRIBUTED SYSTEM at N = 2, 3, 4, 8 ... shards, through the real coordinator and
     real shard server processes, in each statistics mode:
        cached / dfs -> GLOBAL statistics  -> must match A EXACTLY (same docs, same order,
                                              scores equal bit for bit)
        local        -> each shard's OWN statistics -> expected to DRIFT; we measure how much

Usage:
    python compare_rankings.py                                  # N = 2 3 4 8, all three modes
    python compare_rankings.py --shards 2 16 --modes cached local
    python compare_rankings.py --partition topic --modes cached local   # deliberately skewed shards
"""

import argparse
import asyncio
import json
import math
import os
from typing import Dict, List, Tuple

from build_index import ForwardIndex, ensure_layout
from cluster import ClusterManager
from common import (B, DEFAULT_CORPUS, DEFAULT_QUERIES, HERE, K1, analyze, layout_dir,
                    load_routing_table, query_terms)
from coordinator import Coordinator
from shard_server import ShardIndex

Hits = List[Tuple[int, float]]


def load_queries(path: str, limit: int) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    queries = list(dict.fromkeys(queries))              # drop repeated queries, keep the order
    return queries[:limit] if limit else queries


# --------------------------------------------------------------------------------------
# A. the single-node answer key
# --------------------------------------------------------------------------------------
def single_node_search(index: ShardIndex, query: str, k: int, cursor=None) -> Hits:
    """One shard that holds EVERYTHING: its own statistics ARE the global statistics."""
    terms = query_terms(query)
    n = index.doc_count
    avgdl = index.total_doc_length / n                  # the same integers, the same ONE division
    df = {t: index.doc_freq(t) for t in terms}
    return index.search(terms, k, n, avgdl, df, cursor)


# --------------------------------------------------------------------------------------
# B. the independent brute-force reference (plain dictionaries, straight from the formula)
# --------------------------------------------------------------------------------------
def brute_force_reference(corpus_path: str, queries: List[str], k: int) -> Dict[str, Hits]:
    wanted = {t for q in queries for t in query_terms(q)}
    postings: Dict[str, Dict[int, int]] = {t: {} for t in wanted}    # term -> {doc: tf}
    doc_len: Dict[int, int] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tokens = analyze(rec["text"])
            doc_len[rec["doc_id"]] = len(tokens)
            for tok in tokens:
                if tok in wanted:
                    postings[tok][rec["doc_id"]] = postings[tok].get(rec["doc_id"], 0) + 1
    n = len(doc_len)
    avgdl = sum(doc_len.values()) / n
    out = {}
    for q in queries:
        scores: Dict[int, float] = {}
        for t in query_terms(q):                         # sorted -> same addition order
            df = len(postings[t])
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for doc, tf in postings[t].items():
                length_multiplier = 1.0 - B + B * (doc_len[doc] / avgdl)
                scores[doc] = scores.get(doc, 0.0) + idf * (tf * (K1 + 1.0)) / (tf + K1 * length_multiplier)
        out[q] = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return out


# --------------------------------------------------------------------------------------
# C. the distributed system
# --------------------------------------------------------------------------------------
async def distributed_search_all(routing_path: str, mode: str, queries: List[str], k: int,
                                 paginate: int = 0) -> dict:
    coordinator = Coordinator(routing_path, stats_mode=mode, verbose=False)
    await coordinator.start()
    try:
        results, statuses = {}, {}
        for q in queries:
            reply = await coordinator.search(q, k, timeout_s=5.0)
            results[q] = reply["hits"]
            statuses[reply["status"]] = statuses.get(reply["status"], 0) + 1
        pages = {}
        for q in queries[:paginate]:                     # page 1 + page 2 through the cursor
            first = (await coordinator.search(q, k, timeout_s=5.0))["hits"]
            second = []
            if first:
                cursor = (first[-1][1], first[-1][0])    # (score, doc id) of the last hit
                second = (await coordinator.search(q, k, timeout_s=5.0, cursor=cursor))["hits"]
            pages[q] = first + second
        return {"results": results, "statuses": statuses, "pages": pages}
    finally:
        await coordinator.stop()


def compare(reference: Dict[str, Hits], candidate: Dict[str, Hits], k: int) -> dict:
    """How close is `candidate` to `reference`, query by query?"""
    exact = same_ranking = top1_same = 0
    overlap_sum = 0.0
    max_score_diff = 0.0
    examples = []
    for q, ref in reference.items():
        got = candidate[q]
        ref_ids, got_ids = [d for d, _ in ref], [d for d, _ in got]
        if ref == got:                                   # same docs, same order, scores bit-identical
            exact += 1
        if ref_ids == got_ids:
            same_ranking += 1
            for (_, a), (_, b) in zip(ref, got):
                max_score_diff = max(max_score_diff, abs(a - b))
        elif len(examples) < 3:
            examples.append({"query": q, "single_node": ref_ids[:5], "distributed": got_ids[:5]})
        if ref_ids[:1] == got_ids[:1]:
            top1_same += 1
        overlap_sum += len(set(ref_ids) & set(got_ids)) / max(len(ref_ids), 1)
    total = len(reference)
    return {"queries": total, "bit_identical": exact, "same_ranking": same_ranking,
            "top1_same": top1_same, "mean_overlap_at_k": overlap_sum / total,
            "max_score_diff_when_same_ranking": max_score_diff, "examples": examples}


def main():
    ap = argparse.ArgumentParser(description="Compare distributed rankings with the single-node index.")
    ap.add_argument("--shards", type=int, nargs="+", default=[2, 3, 4, 8])
    ap.add_argument("--modes", nargs="+", default=["cached", "dfs", "local"],
                    choices=["cached", "dfs", "local"])
    ap.add_argument("--queries", type=int, default=500, help="how many queries from the query file")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--partition", choices=["hash", "topic"], default="hash")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--queries-file", default=DEFAULT_QUERIES)
    ap.add_argument("--routing", default="",
                    help="multi-laptop: routing_table.json of a cluster that is ALREADY running "
                         "on other machines; compare it and start nothing locally")
    args = ap.parse_args()

    queries = load_queries(args.queries_file, args.queries)
    tag = "" if args.partition == "hash" else args.partition
    print("analysing corpus once ...")
    fwd = ForwardIndex(args.corpus)

    # ---- A. the answer key ------------------------------------------------------------
    ensure_layout(1, 1, 1, args.corpus, fwd=fwd)
    single = ShardIndex(os.path.join(layout_dir(1, 1), "node_A", "shard_0_copy_0"))
    reference = {q: single_node_search(single, q, args.k) for q in queries}
    reference_2k = {q: single_node_search(single, q, 2 * args.k) for q in queries[:50]}
    print("single-node answer key: %d documents, %d queries, k = %d"
          % (single.doc_count, len(queries), args.k))

    # ---- B. the independent check -----------------------------------------------------
    sample = queries[:25]
    brute = brute_force_reference(args.corpus, sample, args.k)
    check = compare({q: reference[q] for q in sample}, brute, args.k)
    print("independent brute-force BM25 vs single node: %d / %d queries bit-identical"
          % (check["bit_identical"], check["queries"]))

    # ---- C. the distributed system ----------------------------------------------------
    report = {"k": args.k, "queries": len(queries), "partition": args.partition,
              "documents": single.doc_count,
              "brute_force_check": {"queries": check["queries"],
                                    "bit_identical": check["bit_identical"]},
              "runs": []}
    print()
    print("%-7s %-7s %-9s %-14s %-13s %-10s %-12s %s"
          % ("shards", "mode", "queries", "bit-identical", "same ranking", "top-1 same",
             "overlap@%d" % args.k, "statuses"))
    if args.routing:
        # MULTI-LAPTOP: the shard servers already run on the node laptops. The coordinator
        # (in this process) reaches them through the IP addresses in the routing table.
        remote_table = load_routing_table(args.routing)
        runs = [(remote_table["num_shards"], args.routing)]
        report["routing"] = args.routing
        report["nodes"] = sorted({c["address"].rsplit(":", 1)[0]
                                  for e in remote_table["shards"] for c in e["copies"]})
        print("comparing the RUNNING cluster on %s" % ", ".join(report["nodes"]))
    else:
        runs = [(n, None) for n in args.shards]
    for n, remote_routing in runs:
        routing = remote_routing or ensure_layout(n, args.copies, args.nodes, args.corpus,
                                                  partition=args.partition, tag=tag, fwd=fwd)
        manager = ClusterManager(routing, quiet=True)
        if not remote_routing:
            manager.start_all()
        try:
            for mode in args.modes:
                paginate = 50 if mode == "cached" else 0
                out = asyncio.run(distributed_search_all(routing, mode, queries, args.k, paginate))
                c = compare(reference, out["results"], args.k)
                row = dict(c, shards=n, mode=mode, statuses=out["statuses"])
                if paginate:
                    # search_after: page 1 + page 2 (via the cursor) must equal the single
                    # node's top-2k. Deep pagination without asking shards for 2k hits.
                    pg = compare(reference_2k, out["pages"], 2 * args.k)
                    row["cursor_pages_bit_identical"] = "%d/%d" % (pg["bit_identical"], pg["queries"])
                report["runs"].append(row)
                print("%-7d %-7s %-9d %-14s %-13s %-10s %-12s %s"
                      % (n, mode, c["queries"],
                         "%d (%.1f%%)" % (c["bit_identical"], 100.0 * c["bit_identical"] / c["queries"]),
                         "%d" % c["same_ranking"], "%d" % c["top1_same"],
                         "%.3f" % c["mean_overlap_at_k"], out["statuses"]))
                if paginate:
                    print("        cursor pagination (page 1 + page 2 == single-node top-%d): %s"
                          % (2 * args.k, row["cursor_pages_bit_identical"]))
        finally:
            manager.stop_all()

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    name = "compare_rankings%s.json" % ("_remote" if args.routing else ("_" + tag if tag else ""))
    path = os.path.join(HERE, "results", name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print()
    print("saved -> %s" % path)
    print("reading the table: 'cached' and 'dfs' use GLOBAL statistics and must be 100% "
          "bit-identical;\n'local' uses per-shard statistics and shows the drift that global "
          "statistics remove.")


if __name__ == "__main__":
    main()
