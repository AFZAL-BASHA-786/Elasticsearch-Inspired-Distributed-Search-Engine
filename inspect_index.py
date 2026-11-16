"""
inspect_index.py : open ONE shard copy and SHOW the raw arrays - the "open the box" tool.

Guide for this file : ../docs/code_guides/09_inspect_index_GUIDE.md
My notes behind it  : 02_Medium_Architecture_and_Design/01_Inverted_Indexes_and_Memory_Mapping.md
                      (sections 5.1 and 5.2: the arrays and the lookup, step by step)

Use it to check by hand that the files on disk are exactly what I drew in my notes:

    python generate_corpus.py --tiny
    python build_index.py --corpus data/tiny_corpus.jsonl --shards 1 --copies 1 --nodes 1 --tag tiny
    python inspect_index.py                      # prints all arrays of the tiny index
    python inspect_index.py --term raft          # the 5-step lookup for one term

On a real shard:
    python inspect_index.py --dir index_data/shards_3_copies_2/node_A/shard_0_copy_0 --term apple
"""

import argparse
import os

from common import layout_dir
from shard_server import ShardIndex


def main():
    ap = argparse.ArgumentParser(description="Print the arrays of one shard copy.")
    ap.add_argument("--dir", default=os.path.join(layout_dir(1, 1, "tiny"), "node_A", "shard_0_copy_0"))
    ap.add_argument("--term", default="", help="trace the lookup of this term")
    ap.add_argument("--limit", type=int, default=20, help="max entries to print per array")
    args = ap.parse_args()

    index = ShardIndex(args.dir)
    vocabulary = sorted(index.lexicon, key=lambda t: index.lexicon[t][0])
    print("shard copy     : %s" % args.dir)
    print("shard_stats    : N_s = %d documents, sum_dl_s = %d tokens  (exact integers)"
          % (index.doc_count, index.total_doc_length))
    print("vocabulary     : %d terms, %d postings" % (len(vocabulary), index.doc_ids.size))
    print("doc_ids is a memory map : %s" % type(index.doc_ids).__name__)

    def show(name, values):
        values = list(values[:args.limit])
        more = " ..." if len(values) == args.limit else ""
        print("%-15s= %s%s" % (name, values, more))

    print()
    show("doc_ids.npy", index.doc_ids)
    show("term_freqs.npy", index.term_freqs)
    show("doc_lengths.npy", index.doc_lengths)
    show("doc_global_ids", index.doc_global_ids)
    print()
    print("LEXICON (kept in RAM):  term -> (offset, length)      length = df_s(term)")
    for term in vocabulary[:args.limit]:
        offset, length = index.lexicon[term]
        print("  %-14s -> offset %-6d length %-6d  slice [%d : %d]"
              % (term, offset, length, offset, offset + length))
    if len(vocabulary) > args.limit:
        print("  ... %d more terms" % (len(vocabulary) - args.limit))

    if args.term:
        print()
        print("LOOKUP of the term %r, step by step:" % args.term)
        entry = index.lexicon.get(args.term)
        if entry is None:
            print("  1. lexicon lookup: NOT in this shard -> df_s = 0, contributes nothing")
            return
        offset, length = entry
        print("  1. lexicon lookup (one hash lookup in RAM) : offset = %d, length = %d" % (offset, length))
        print("  2. slice range                             : [%d : %d]" % (offset, offset + length))
        print("  3. doc_ids[%d:%d]    = %s" % (offset, offset + length,
                                               list(index.doc_ids[offset:offset + length][:args.limit])))
        print("     term_freqs[%d:%d] = %s" % (offset, offset + length,
                                               list(index.term_freqs[offset:offset + length][:args.limit])))
        print("  4. df for free                             : df_s = end - start = %d" % length)
        docs = index.doc_ids[offset:offset + length][:args.limit]
        print("  5. doc_lengths[docs]                       : %s" % list(index.doc_lengths[docs]))
        print("     global ids of those docs                : %s" % list(index.doc_global_ids[docs]))


if __name__ == "__main__":
    main()
