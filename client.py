"""
client.py : the USER at the front desk. Sends a query to the coordinator and prints the reply.

Guide for this file : ../docs/code_guides/08_client_GUIDE.md

The client knows NOTHING about shards, replicas or nodes. It talks to one address (the
coordinator) and gets back hits + an explicit status (COMPLETE / PARTIAL).

Usage (Terminal 3):
    python client.py "raft consensus"
    python client.py "apple pie" -k 5 --titles
    python client.py "cheap flights" -k 5 --pages 3      # search_after cursor pagination
    python client.py --health                            # the coordinator's whiteboard
    python client.py "heartbeat failover" -k 5 --repeat 600 --interval-ms 100 --quiet
                                                         # a 1-minute stream, one line per query
"""

import argparse
import json
import sys
import time

import grpc

import search_pb2
import search_pb2_grpc
from common import DEFAULT_COORDINATOR_PORT, DEFAULT_CORPUS, HOST


def load_titles(doc_ids, corpus_path):
    """The optional FETCH phase of "query then fetch": only AFTER the merge do we look up
    the content of the k winners. Shards returned just (doc id, score) pairs."""
    wanted, titles = set(doc_ids), {}
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not wanted:
                    break
                rec = json.loads(line)
                if rec["doc_id"] in wanted:
                    titles[rec["doc_id"]] = rec.get("title", "")
                    wanted.discard(rec["doc_id"])
    except OSError:
        pass
    return titles


def print_health(stub):
    reply = stub.ClusterHealth(search_pb2.ClusterHealthRequest(), timeout=2.0)
    print("cluster colour: %s" % reply.color)
    print("  shard copy node    address           state    misses outstanding")
    for c in reply.copies:
        print("  %-5d %-4d %-7s %-17s %-8s %-6d %d"
              % (c.shard_id, c.copy_id, c.node, c.address, c.state, c.misses, c.outstanding))


def main():
    ap = argparse.ArgumentParser(description="Query the distributed search engine.")
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--coordinator", default="%s:%d" % (HOST, DEFAULT_COORDINATOR_PORT))
    ap.add_argument("--timeout-ms", type=int, default=500, help="the client's total time budget")
    ap.add_argument("--titles", action="store_true", help="look up titles of the winners")
    ap.add_argument("--pages", type=int, default=1, help="fetch N pages with a search_after cursor")
    ap.add_argument("--repeat", type=int, default=1, help="send the same query N times")
    ap.add_argument("--interval-ms", type=int, default=0, help="pause between repeats")
    ap.add_argument("--quiet", action="store_true",
                    help="one line per query (status, shards, time, top doc ids) - for watching "
                         "a stream of queries while copies are killed and restarted")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--health", action="store_true", help="print the health table and exit")
    args = ap.parse_args()

    with grpc.insecure_channel(args.coordinator) as channel:
        stub = search_pb2_grpc.CoordinatorServiceStub(channel)
        try:
            if args.health:
                print_health(stub)
                return
            if not args.query:
                sys.exit("give a query, e.g.  python client.py \"raft consensus\"")
            for iteration in range(args.repeat):
                if iteration and args.interval_ms:
                    time.sleep(args.interval_ms / 1000.0)
                if args.quiet:
                    # STREAM MODE: one compact line per query, so a change of status
                    # (COMPLETE -> PARTIAL -> COMPLETE) is easy to see as it happens.
                    sent = time.perf_counter()
                    reply = stub.Query(search_pb2.QueryRequest(query=args.query, k=args.k,
                                                               timeout_ms=args.timeout_ms),
                                       timeout=args.timeout_ms / 1000.0 + 1.0)
                    wall_ms = (time.perf_counter() - sent) * 1000.0
                    failed = ",".join(str(f.shard_id) for f in reply.failed_shards) or "-"
                    print("%s  %-8s shards %d/%d  cluster %-6s %7.1f ms  retries %d  failed shards: %-3s top: %s"
                          % (time.strftime("%H:%M:%S"), search_pb2.ResultStatus.Name(reply.status),
                             reply.shards_successful, reply.shards_total, reply.cluster_color,
                             wall_ms, reply.retries, failed,
                             [h.doc_id for h in reply.hits[:5]]), flush=True)
                    continue
                cursor = None
                rank = 0
                for page in range(args.pages):
                    request = search_pb2.QueryRequest(query=args.query, k=args.k,
                                                      timeout_ms=args.timeout_ms)
                    if cursor is not None:
                        # The bookmark: "skip everything at or above the LAST hit I saw".
                        # Each shard then returns only k hits again, instead of page x k.
                        request.has_cursor = True
                        request.cursor_score, request.cursor_doc_id = cursor
                    sent = time.perf_counter()
                    reply = stub.Query(request, timeout=args.timeout_ms / 1000.0 + 1.0)
                    wall_ms = (time.perf_counter() - sent) * 1000.0
                    status = search_pb2.ResultStatus.Name(reply.status)
                    print("page %d | terms %s | status %s | shards %d/%d | cluster %s | "
                          "%.2f ms end-to-end (coordinator %.2f ms, slowest shard %.2f ms, retries %d)"
                          % (page + 1, list(reply.terms), status, reply.shards_successful,
                             reply.shards_total, reply.cluster_color, wall_ms, reply.took_ms,
                             reply.slowest_shard_ms, reply.retries))
                    for failed in reply.failed_shards:
                        print("  !! shard %d UNAVAILABLE: %s"
                              % (failed.shard_id, "; ".join(failed.errors)))
                    titles = load_titles([h.doc_id for h in reply.hits], args.corpus) if args.titles else {}
                    for hit in reply.hits:
                        rank += 1
                        print("  %3d. doc %-8d score %.6f  %s"
                              % (rank, hit.doc_id, hit.score, titles.get(hit.doc_id, "")))
                    if not reply.hits:
                        print("  (no more hits)")
                        break
                    cursor = (reply.hits[-1].score, reply.hits[-1].doc_id)
        except grpc.RpcError as err:
            sys.exit("cannot reach the coordinator at %s: %s\n"
                     "start it with:  python coordinator.py --shards 3"
                     % (args.coordinator, err.code().name))


if __name__ == "__main__":
    main()
