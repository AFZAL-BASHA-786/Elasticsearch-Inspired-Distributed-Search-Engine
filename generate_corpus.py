"""
generate_corpus.py : creates the "supplied document collection" and a query set.

Guide for this file : ../docs/code_guides/03_generate_corpus_GUIDE.md

The brief says "over a supplied document collection". No collection was supplied with the
project folder, so this script creates one that is
  * STATIC        - generated once with a fixed seed, never modified afterwards
  * REPRODUCIBLE  - the same seed gives byte-identical files on any machine
  * REALISTIC enough to show the effects my notes talk about:
      - document lengths vary a lot (short notes vs long articles)  -> the b / avgdl yardstick
      - some words are common, some are rare (Zipf-like)            -> idf matters
      - documents have topics                                        -> queries are meaningful

You can replace data/corpus.jsonl with any real collection: one JSON object per line with
the fields  {"doc_id": <int>, "title": <str>, "text": <str>} .

Usage:
    python generate_corpus.py                 # 50,000 documents + 500 queries
    python generate_corpus.py --docs 100000
    python generate_corpus.py --tiny          # the 4-document example from my notes
"""

import argparse
import json
import os

import numpy as np

from common import DATA_DIR, DEFAULT_CORPUS, DEFAULT_QUERIES

# Topic vocabularies. The FIRST words of each list are used most often (Zipf-like).
TOPICS = {
    "distributed": "distributed consensus raft paxos replica shard node cluster leader quorum "
                   "partition failover heartbeat latency throughput coordinator replication "
                   "gossip election log commit snapshot rpc deadline timeout idempotent "
                   "linearizable membership rebalance".split(),
    "search": "search index query ranking relevance document term posting inverted score "
              "tokenizer analyzer retrieval recall precision snippet crawler bm25 idf "
              "frequency lexicon stemming pagination cursor facet highlight synonym boost".split(),
    "cooking": "recipe apple pie butter flour sugar oven bake salt pepper garlic onion tomato "
               "pasta sauce cheese simmer roast grill spice cinnamon vanilla dough whisk "
               "knead marinate saute broth".split(),
    "gardening": "garden soil seed plant water compost flower rose tomato prune harvest "
                 "mulch fertilizer greenhouse weed shrub bulb seedling pollinate trellis "
                 "perennial orchard cactus photosynthesis irrigation nursery".split(),
    "sports": "match team goal player coach league score season tournament stadium referee "
              "penalty striker defender championship training cricket wicket innings football "
              "tennis serve marathon sprint medal transfer".split(),
    "physics": "energy particle quantum wave field force mass electron photon gravity "
               "relativity entropy momentum spin neutron proton laser superconductor boson "
               "spectrum orbit plasma collider entanglement".split(),
    "travel": "flight hotel ticket cheap flights airport luggage passport visa itinerary "
              "beach mountain tour guide booking destination cruise hostel backpack museum "
              "island resort layover".split(),
    "finance": "market stock bank loan interest inflation bond investor portfolio dividend "
               "equity budget tax credit currency exchange profit revenue audit hedge "
               "mortgage insurance startup valuation".split(),
    "health": "health doctor patient vaccine diet exercise sleep protein vitamin heart "
              "blood pressure therapy diagnosis symptom clinic surgery immune infection "
              "nutrition hydration allergy dosage".split(),
    "music": "music song guitar piano melody rhythm concert album band singer chord "
             "orchestra violin drum lyric tempo harmony studio jazz symphony tuning "
             "playlist acoustic".split(),
}

# General filler words: appear in documents of EVERY topic (they are not stop words, so
# they are indexed and act as the "common, low-idf" terms).
GENERAL = ("system time people work new way year day world life use first good data part "
           "place case week point group problem fact number long small large high different "
           "important early public able information level change result example study "
           "report process often still while also many most other some more very just "
           "only over after before under between through during about against").split()

# Rare words: each lands in only a handful of documents -> very high idf.
RARE = ("zebra quasar xylophone obsidian labyrinth nebula saffron glacier falcon mosaic "
        "pendulum origami tundra citadel lantern velvet ember cobalt harbor meadow "
        "kestrel willow topaz ivory juniper marble thistle comet lagoon prism").split()


def zipf_probs(n: int, s: float = 1.0) -> np.ndarray:
    """Zipf-like weights: the i-th word is used in proportion to 1 / i^s."""
    w = 1.0 / np.arange(1, n + 1) ** s
    return w / w.sum()


def generate(num_docs: int, seed: int, corpus_path: str, queries_path: str, num_queries: int):
    rng = np.random.default_rng(seed)
    topic_names = list(TOPICS)
    # Topics are not equally popular: a few topics have many documents, some have few.
    topic_weights = zipf_probs(len(topic_names), 0.6)
    general = np.array(GENERAL)
    general_p = zipf_probs(len(general), 0.9)
    topic_arr = {t: np.array(w) for t, w in TOPICS.items()}
    topic_p = {t: zipf_probs(len(w), 1.0) for t, w in TOPICS.items()}
    rare = np.array(RARE)
    rare_p = zipf_probs(len(rare), 1.2)

    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8", newline="\n") as f:
        for doc_id in range(num_docs):
            topic = topic_names[rng.choice(len(topic_names), p=topic_weights)]
            # Log-normal lengths: most documents are short, a few are very long.
            length = int(np.clip(rng.lognormal(mean=4.2, sigma=0.7), 6, 900))
            n_topic = int(length * 0.40)
            n_other = int(length * 0.08)
            n_general = length - n_topic - n_other
            words = list(rng.choice(topic_arr[topic], size=n_topic, p=topic_p[topic]))
            words += list(rng.choice(general, size=n_general, p=general_p))
            other = topic_names[rng.integers(len(topic_names))]
            words += list(rng.choice(topic_arr[other], size=n_other, p=topic_p[other]))
            if rng.random() < 0.04:                      # 4% of documents get one rare word
                words.append(str(rng.choice(rare, p=rare_p)))
            rng.shuffle(words)
            rec = {"doc_id": doc_id, "title": "%s note #%d" % (topic, doc_id),
                   "text": " ".join(words)}
            f.write(json.dumps(rec) + "\n")

    # ---- the query set: the SAME file is used for every shard count (fair comparison) ----
    queries = []
    for _ in range(num_queries):
        topic = topic_names[rng.choice(len(topic_names), p=topic_weights)]
        n_terms = int(rng.choice([1, 2, 3], p=[0.30, 0.45, 0.25]))
        terms = list(rng.choice(topic_arr[topic], size=n_terms, replace=False))
        roll = rng.random()
        if roll < 0.15:                                  # sometimes a rare term (high idf)
            terms[-1] = str(rng.choice(rare))
        elif roll < 0.30:                                # sometimes a very common term (low idf)
            terms[-1] = str(rng.choice(general[:20]))
        queries.append(" ".join(terms))
    with open(queries_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(queries) + "\n")
    print("wrote %d documents  -> %s" % (num_docs, corpus_path))
    print("wrote %d queries    -> %s" % (num_queries, queries_path))


def generate_tiny(path: str):
    """The 4-document running example from my notes
    (02_Medium.../01_Inverted_Indexes_and_Memory_Mapping.md, section 2 and 5.1).

    Build it with ONE shard and inspect_index.py prints exactly the arrays drawn there:
        doc_ids     = [0, 1, 3, 3, 2, 1, 0, 0, 3, 2]
        term_freqs  = [1, 2, 1, 1, 1, 1, 1, 1, 2, 1]
        offsets     = [0, 3, 4, 5, 6, 7, 9, 10]
        doc_lengths = [3, 3, 2, 4]
    """
    docs = ["raft consensus protocol", "paxos consensus consensus",
            "gardening tips", "distributed raft consensus raft"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, text in enumerate(docs):
            f.write(json.dumps({"doc_id": i, "title": "d%d" % i, "text": text}) + "\n")
    print("wrote 4 documents -> %s" % path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the static document collection.")
    ap.add_argument("--docs", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--queries-file", default=DEFAULT_QUERIES)
    ap.add_argument("--tiny", action="store_true",
                    help="write data/tiny_corpus.jsonl (the 4-document example from the notes)")
    args = ap.parse_args()
    if args.tiny:
        generate_tiny(os.path.join(DATA_DIR, "tiny_corpus.jsonl"))
    else:
        generate(args.docs, args.seed, args.corpus, args.queries_file, args.queries)
