"""
build_index.py : the OFFLINE phase. Runs ONCE per shard count, before any server starts.

Guide for this file : ../docs/code_guides/04_build_index_GUIDE.md
My notes behind it  : self learning/learning_guide/02_Medium_Architecture_and_Design/
                        01_Inverted_Indexes_and_Memory_Mapping.md      (sections 5, 6, 7)
                      self learning/learning_guide/03_In_Depth_Implementation/
                        03_Fault_Tolerance_and_Replication.md          (section 10.1, allocator)

It plays TWO roles from my component table (01_What_Is_A_Distributed_Search_Engine.md, sec. 7):

  1. INDEXER   : read documents -> break into terms -> deal each WHOLE document to a shard
                 with murmur3(doc_id) mod N -> build each shard's inverted index as FLAT,
                 CONTIGUOUS 1-D NumPy arrays.
  2. ALLOCATOR : decide which node holds which copy (anti-affinity: two copies of a shard
                 never share a node) and write routing_table.json.

STRICTLY STATIC (the critical constraint):
  * no dynamic ingestion, no in-memory write buffer, no B-tree, no LSM-tree, no compaction
  * after this script finishes, NOTHING ever writes to these files again
  * a replica is a byte-for-byte FILE COPY: copying files is the whole replication protocol

Files written for ONE shard copy  (e.g. index_data/shards_3_copies_2/node_A/shard_0_copy_0/):

    doc_ids.npy         int32  one entry per posting   <- the long "paper tape" of doc ids
    term_freqs.npy      int32  one entry per posting   <- the parallel tape of counts (tf)
    offsets.npy         int64  V+1 entries             <- the "bookmark tape" (chained boundaries)
    vocabulary.txt      V sorted terms, one per line   <- line number = term id
    doc_lengths.npy     int32  one entry per document  <- dl(d), indexed by LOCAL doc id
    doc_global_ids.npy  int64  one entry per document  <- LOCAL id -> corpus-wide (GLOBAL) id
    shard_stats.json    N_s and sum_dl_s               <- exact integers, for global statistics

Usage:
    python build_index.py --shards 3                      # 3 shards x 2 copies on 3 nodes
    python build_index.py --shards 1 2 4 8                # one fresh build per shard count
    python build_index.py --shards 1 --copies 1 --nodes 1 # the single-node baseline
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from array import array
from collections import Counter
from typing import Dict, List

import numpy as np

from common import (DEFAULT_BASE_PORT, DEFAULT_CORPUS, HOST, analyze, layout_dir, murmur3_32,
                    shard_of)


# ======================================================================================
# STEP 1: ANALYSE THE CORPUS ONCE  ->  a compact FORWARD index (document -> terms)
# ======================================================================================
class ForwardIndex:
    """What we know after reading every document once.

    The documents themselves are a FORWARD mapping (document -> words). Search needs the
    opposite (word -> documents), hence "INVERTED" index. We first collect the forward
    form compactly, then invert it once per shard.

    All postings of all documents are stored end to end, exactly like the final index:
        fwd_term[i], fwd_tf[i], fwd_doc[i]  = "document fwd_doc[i] contains term fwd_term[i]
                                               fwd_tf[i] times"
    """

    def __init__(self, corpus_path: str):
        docs = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        # Documents are processed in INCREASING global doc_id order. Later this makes
        # "smaller local id" also mean "smaller global id" inside every shard, so the
        # tie-break rule (equal score -> smaller doc id first) gives the same order inside
        # a shard and in the coordinator's merge (notes: file 01, section 5.4).
        docs.sort(key=lambda d: d["doc_id"])
        ids = [d["doc_id"] for d in docs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate doc_id in corpus")

        term_to_id: Dict[str, int] = {}          # term -> temporary id (first-seen order)
        fwd_term, fwd_tf, fwd_doc = array("i"), array("i"), array("i")
        doc_len = array("i")
        for pos, d in enumerate(docs):
            # Only the TEXT is indexed. The title is display metadata (client.py shows it),
            # so the index stores ids and counts, never the documents' text itself.
            tokens = analyze(d["text"])
            # dl(d) = number of index tokens AFTER analysis. Stop words were dropped by the
            # analyser, so they are not counted - dl and avgdl must be counted with the
            # same tokeniser that builds the postings (notes: BM25 file, section 1.2).
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():     # tf(t, d): count of t inside d
                tid = term_to_id.get(term)
                if tid is None:
                    tid = term_to_id[term] = len(term_to_id)
                fwd_term.append(tid)
                fwd_tf.append(tf)
                fwd_doc.append(pos)

        self.num_docs = len(docs)
        self.global_ids = np.array(ids, dtype=np.int64)            # position -> global doc id
        self.topics = [d.get("title", "").split(" ")[0] for d in docs]   # only for --partition topic
        self.doc_len = np.frombuffer(doc_len, dtype=np.int32).copy()
        self.fwd_term = np.frombuffer(fwd_term, dtype=np.int32).copy()
        self.fwd_tf = np.frombuffer(fwd_tf, dtype=np.int32).copy()
        self.fwd_doc = np.frombuffer(fwd_doc, dtype=np.int32).copy()

        # Give every term its FINAL rank in the alphabetically sorted global vocabulary.
        terms = sorted(term_to_id)                                 # sorted vocabulary
        rank_of_tmp = np.empty(len(terms), dtype=np.int32)
        for rank, term in enumerate(terms):
            rank_of_tmp[term_to_id[term]] = rank
        self.terms = terms
        self.fwd_rank = rank_of_tmp[self.fwd_term]                 # alphabetical rank per posting


# ======================================================================================
# STEP 2: DEAL DOCUMENTS TO SHARDS  (the "dealing rule")
# ======================================================================================
def assign_shards(fwd: ForwardIndex, num_shards: int, partition: str) -> np.ndarray:
    """Return shard number for every document. Each WHOLE document goes to exactly ONE shard.

    hash  (default): murmur3(doc_id) mod N -> deterministic, balanced, no lookup table.
    topic (experiment only): all documents of one topic land on the same shard. This is the
          deliberately SKEWED split (the cookbooks-vs-sports example in
          01_Basics/03_Why_Scoring_Gets_Messy.md, section 3) used to show how badly LOCAL
          statistics drift. Never use it for a real deployment.
    """
    if partition == "hash":
        return np.array([shard_of(int(g), num_shards) for g in fwd.global_ids], dtype=np.int32)
    if partition == "topic":
        return np.array([murmur3_32(t.encode("utf-8")) % num_shards for t in fwd.topics],
                        dtype=np.int32)
    raise ValueError("unknown partition: " + partition)


# ======================================================================================
# STEP 3: INVERT ONE SHARD  ->  flat arrays + offsets (the CSR layout)
# ======================================================================================
def build_shard_arrays(fwd: ForwardIndex, doc_shard: np.ndarray, shard_id: int) -> dict:
    """Build the inverted index of ONE shard as flat, contiguous 1-D arrays.

    The recipe is exactly section 6 of my notes:
        1. emit (term, local doc id, tf) triples for this shard's documents
        2. SORT all triples by (term, doc id)          <- this sort IS the "inversion"
        3. walk the sorted list: doc ids -> doc_ids, tfs -> term_freqs,
           count postings per term -> PREFIX SUM -> offsets
    """
    in_shard = doc_shard == shard_id                     # which documents belong here
    positions = np.flatnonzero(in_shard)                 # their positions, in global-id order
    n_docs = int(positions.size)

    # LOCAL ids 0, 1, 2, ... are handed out in increasing GLOBAL id order.
    # Why local ids at all? If this shard owned global documents 900000, 900001, 900002,
    # a doc_lengths array indexed by global id would need 900003 slots, almost all empty.
    # With local ids it needs 3: a DENSE array with no gaps (notes: doubts 02/01, "What is this").
    local_of = np.cumsum(in_shard) - 1                   # position -> local id (valid where in_shard)

    keep = in_shard[fwd.fwd_doc]                         # postings that belong to this shard
    rank = fwd.fwd_rank[keep]                            # term (as alphabetical rank)
    local_doc = local_of[fwd.fwd_doc[keep]].astype(np.int32)
    tf = fwd.fwd_tf[keep]

    # THE INVERSION: sort the triples by (term, local doc id).
    # np.lexsort sorts by the LAST key first, so (local_doc, rank) means "by rank, then doc".
    order = np.lexsort((local_doc, rank))
    rank, local_doc, tf = rank[order], local_doc[order], tf[order]

    # This shard's own vocabulary = the terms that actually occur in it, still sorted.
    present_ranks, counts = np.unique(rank, return_counts=True)
    vocabulary = [fwd.terms[r] for r in present_ranks]

    # PREFIX SUM turns "postings per term" into chained bookmarks:
    #   counts  [3, 1, 1, 1, 1, 2, 1]  ->  offsets [0, 3, 4, 5, 6, 7, 9, 10]   (V + 1 numbers)
    # offsets[t] is the END of term t-1 AND the START of term t, so one number does double
    # duty and we never store separate (start, end) pairs.
    offsets = np.zeros(len(vocabulary) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    doc_lengths = fwd.doc_len[positions]
    return {
        "doc_ids": np.ascontiguousarray(local_doc, dtype=np.int32),
        "term_freqs": np.ascontiguousarray(tf, dtype=np.int32),
        "offsets": offsets,
        "vocabulary": vocabulary,
        "doc_lengths": np.ascontiguousarray(doc_lengths, dtype=np.int32),
        "doc_global_ids": np.ascontiguousarray(fwd.global_ids[positions], dtype=np.int64),
        "stats": {
            "shard_id": shard_id,
            # The two EXACT INTEGERS the coordinator sums. The shard never reports its own
            # average: "ship counts, not ratios" (notes: BM25 file, section 7.4).
            "doc_count": n_docs,                                   # N_s
            "total_doc_length": int(doc_lengths.sum(dtype=np.int64)),   # sum_dl_s
            "num_terms": len(vocabulary),
            "num_postings": int(local_doc.size),
        },
    }


def write_shard_copy(directory: str, arrays: dict, num_shards: int) -> None:
    """Write one shard copy to disk. After this, the files are never touched again."""
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, "doc_ids.npy"), arrays["doc_ids"])
    np.save(os.path.join(directory, "term_freqs.npy"), arrays["term_freqs"])
    np.save(os.path.join(directory, "offsets.npy"), arrays["offsets"])
    np.save(os.path.join(directory, "doc_lengths.npy"), arrays["doc_lengths"])
    np.save(os.path.join(directory, "doc_global_ids.npy"), arrays["doc_global_ids"])
    with open(os.path.join(directory, "vocabulary.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(arrays["vocabulary"]))
    stats = dict(arrays["stats"], num_shards=num_shards)
    with open(os.path.join(directory, "shard_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def directory_digest(directory: str) -> str:
    """One sha256 over every file of a shard copy - used to PROVE replicas are identical."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(directory)):
        with open(os.path.join(directory, name), "rb") as f:
            h.update(name.encode("utf-8"))
            h.update(f.read())
    return h.hexdigest()


# ======================================================================================
# STEP 4: THE ALLOCATOR  ->  which node stores which copy
# ======================================================================================
def node_name(index: int) -> str:
    return "node_" + (chr(ord("A") + index) if index < 26 else str(index))


def parse_hosts(spec: str) -> Dict[str, str]:
    """ "node_A=192.168.43.21,node_B=192.168.43.22"  ->  {"node_A": "192.168.43.21", ...}

    MULTI-MACHINE DEPLOYMENT: the routing table is the only place that says where a copy
    lives, so putting real IP addresses here is all the coordinator needs to work across
    laptops. A node missing from the map stays on 127.0.0.1 (this machine).
    """
    hosts = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "=" not in part:
            raise ValueError("--hosts entries look like node_A=192.168.1.11, got %r" % part)
        node, ip = (x.strip() for x in part.split("=", 1))
        hosts[node] = ip
    return hosts


def allocate(num_shards: int, copies: int, num_nodes: int, base_port: int,
             hosts: Dict[str, str] = None) -> dict:
    """Place copy r of shard s on node (s - r) mod M.   (notes: file 03, section 10.1)

    ANTI-AFFINITY: the C copies of a shard must sit on C DIFFERENT nodes. A node is one
    "failure domain" (same power, same disk, same OS): copies on the same node die together,
    so a replica next to its original protects nothing.

        N = 3 shards, C = 2 copies, M = 3 nodes:
            S0 -> node A, node C      S1 -> node B, node A      S2 -> node C, node B
        Kill ANY one node and every shard still has a live copy.

    The one hard rule is M >= C. If C > M the allocator does NOT break the rule to squeeze
    the extra copy in: it leaves that copy UNASSIGNED and the cluster starts YELLOW.
    """
    shards, unassigned = [], []
    port = base_port
    for s in range(num_shards):
        copy_list = []
        for r in range(copies):
            if r >= num_nodes:                           # no legal node left for this copy
                unassigned.append({"shard_id": s, "copy_id": r,
                                   "reason": "anti-affinity: only %d nodes for %d copies"
                                             % (num_nodes, copies)})
                continue
            node = node_name((s - r) % num_nodes)
            copy_list.append({
                "copy_id": r,
                "node": node,
                # One shard server PROCESS per shard copy; its address is "node + port".
                # Single machine: every node is 127.0.0.1. Several laptops (--hosts):
                # each node's real IP address.
                "address": "%s:%d" % ((hosts or {}).get(node, HOST), port),
                "path": "%s/shard_%d_copy_%d" % (node, s, r),      # relative to the layout folder
            })
            port += 1
        shards.append({"shard_id": s, "copies": copy_list})

    # Refuse to continue if any shard has two copies on the same node.
    for entry in shards:
        nodes = [c["node"] for c in entry["copies"]]
        if len(nodes) != len(set(nodes)):
            raise RuntimeError("anti-affinity violated for shard %d" % entry["shard_id"])
    return {"shards": shards, "unassigned": unassigned}


# ======================================================================================
# Putting it together
# ======================================================================================
def build(fwd: ForwardIndex, num_shards: int, copies: int, num_nodes: int, base_port: int,
          partition: str = "hash", tag: str = "", quiet: bool = False,
          hosts: Dict[str, str] = None) -> str:
    """Build one complete layout (all shards, all copies, routing table). Returns its folder."""
    out_dir = layout_dir(num_shards, copies, tag)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)                           # a rebuild is always from scratch
    os.makedirs(out_dir)

    placement = allocate(num_shards, copies, num_nodes, base_port, hosts)
    unknown = set(hosts or {}) - {c["node"] for e in placement["shards"] for c in e["copies"]}
    if unknown:
        raise ValueError("--hosts names nodes that hold no copy: %s" % sorted(unknown))
    doc_shard = assign_shards(fwd, num_shards, partition)

    total_docs, total_len = 0, 0
    for entry in placement["shards"]:
        s = entry["shard_id"]
        arrays = build_shard_arrays(fwd, doc_shard, s)
        total_docs += arrays["stats"]["doc_count"]
        total_len += arrays["stats"]["total_doc_length"]
        digests = []
        first_dir = None
        for copy in entry["copies"]:
            directory = os.path.join(out_dir, copy["path"])
            if first_dir is None:
                write_shard_copy(directory, arrays, num_shards)    # the "primary" (just a label)
                first_dir = directory
            else:
                shutil.copytree(first_dir, directory)              # REPLICATION = file copy
            digests.append(directory_digest(directory))
        if len(set(digests)) > 1:
            raise RuntimeError("replicas of shard %d are not byte-identical" % s)
        entry["sha256"] = digests[0] if digests else ""
        entry["doc_count"] = arrays["stats"]["doc_count"]
        if not quiet:
            st = arrays["stats"]
            print("  shard %-2d  docs=%-7d tokens=%-9d terms=%-6d postings=%-9d copies on %s"
                  % (s, st["doc_count"], st["total_doc_length"], st["num_terms"],
                     st["num_postings"], ", ".join(c["node"] for c in entry["copies"])))

    table = {
        "num_shards": num_shards,
        "copies_per_shard": copies,
        "num_nodes": num_nodes,
        "partitioner": "murmur3_32(str(doc_id)) mod num_shards" if partition == "hash" else partition,
        # Only for humans / the comparison report. The coordinator NEVER reads these two
        # numbers: it always collects N_s and sum_dl_s from the shards themselves.
        "corpus": {"doc_count": total_docs, "total_doc_length": total_len},
        "shards": placement["shards"],
        "unassigned": placement["unassigned"],
    }
    with open(os.path.join(out_dir, "routing_table.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    if not quiet:
        if placement["unassigned"]:
            print("  WARNING: %d copies UNASSIGNED (cluster will start YELLOW)"
                  % len(placement["unassigned"]))
        print("  routing table -> %s" % os.path.join(out_dir, "routing_table.json"))
    return out_dir


def write_node_bundles(out_dir: str) -> List[str]:
    """MULTI-LAPTOP: write ONE ZIP PER NODE with everything that laptop needs.

        <layout>/deploy/node_A.zip
            search_engine/*.py, search.proto, requirements.txt     the program
            search_engine/index_data/<layout>/routing_table.json    who lives where
            search_engine/index_data/<layout>/node_A/...            ONLY this node's shard copies
            search_engine/start_node_A.bat                          double-click to start the node
            search_engine/README_NODE.txt                           the 4 steps for that laptop

    "Deploying a node" is therefore: copy one zip, unzip, install, start. A node never gets
    the corpus or another node's files - replication is nothing more than copying files.
    """
    import zipfile
    with open(os.path.join(out_dir, "routing_table.json"), "r", encoding="utf-8") as f:
        table = json.load(f)
    layout_name = os.path.basename(out_dir)
    code_dir = os.path.dirname(os.path.abspath(__file__))
    code_files = sorted(n for n in os.listdir(code_dir)
                        if n.endswith((".py", ".proto")) or n == "requirements.txt")
    routing_rel = "index_data\\%s\\routing_table.json" % layout_name
    deploy_dir = os.path.join(out_dir, "deploy")
    os.makedirs(deploy_dir, exist_ok=True)
    nodes = {}                                           # node -> [(shard, copy, address)]
    for entry in table["shards"]:
        for copy in entry["copies"]:
            nodes.setdefault(copy["node"], []).append((entry["shard_id"], copy["copy_id"],
                                                       copy["address"]))
    written = []
    for node, copies in sorted(nodes.items()):
        ports = ",".join(a.rsplit(":", 1)[1] for _, _, a in copies)
        start_cmd = "python cluster.py --routing %s --node %s" % (routing_rel, node)
        readme = "\r\n".join([
            "THIS LAPTOP IS %s  (address in the routing table: %s)" % (node, copies[0][2].rsplit(":", 1)[0]),
            "It serves: " + ", ".join("shard %d copy %d on port %s" % (s, c, a.rsplit(":", 1)[1])
                                      for s, c, a in copies),
            "",
            "1. Unzip anywhere. Open PowerShell in the search_engine folder.",
            "2. Once:  python -m pip install -r requirements.txt",
            "3. Once, in PowerShell run AS ADMINISTRATOR (opens the ports in the firewall):",
            "     New-NetFirewallRule -DisplayName \"DSE shards\" -Direction Inbound -Protocol TCP "
            "-LocalPort %s -Action Allow -Profile Any" % ports,
            "4. Start the node (or double-click start_%s.bat):" % node,
            "     " + start_cmd,
            "",
            "Console commands: status | kill S C | freeze S C | resume S C | restart S C | quit",
            "This laptop's IP must still be the one above (check with: ipconfig).",
            "",
        ])
        bat = "@echo off\r\ncd /d \"%~dp0\"\r\n" + start_cmd + "\r\npause\r\n"
        zip_path = os.path.join(deploy_dir, node + ".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for name in code_files:
                z.write(os.path.join(code_dir, name), "search_engine/" + name)
            z.write(os.path.join(out_dir, "routing_table.json"),
                    "search_engine/index_data/%s/routing_table.json" % layout_name)
            node_dir = os.path.join(out_dir, node)
            for base, _, names in os.walk(node_dir):
                for name in names:
                    full = os.path.join(base, name)
                    rel = os.path.relpath(full, out_dir).replace("\\", "/")
                    z.write(full, "search_engine/index_data/%s/%s" % (layout_name, rel))
            z.writestr("search_engine/README_NODE.txt", readme)
            z.writestr("search_engine/start_%s.bat" % node, bat)
        written.append(zip_path)
    return written


def ensure_layout(num_shards: int, copies: int, num_nodes: int, corpus: str = DEFAULT_CORPUS,
                  base_port: int = DEFAULT_BASE_PORT, partition: str = "hash", tag: str = "",
                  fwd: ForwardIndex = None) -> str:
    """Build the layout only if it does not exist yet (used by the experiment scripts)."""
    path = os.path.join(layout_dir(num_shards, copies, tag), "routing_table.json")
    if os.path.exists(path):
        return path
    if fwd is None:
        fwd = ForwardIndex(corpus)
    build(fwd, num_shards, copies, num_nodes, base_port, partition, tag, quiet=True)
    return path


def main():
    ap = argparse.ArgumentParser(description="Offline indexer + allocator (static, immutable index).")
    ap.add_argument("--shards", type=int, nargs="+", default=[3],
                    help="one or more shard counts; each gets its own fresh build")
    ap.add_argument("--copies", type=int, default=2, help="copies per shard (1 = no replica)")
    ap.add_argument("--nodes", type=int, default=3, help="number of (simulated) nodes")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    ap.add_argument("--partition", choices=["hash", "topic"], default="hash")
    ap.add_argument("--tag", default="", help="suffix for the layout folder name")
    ap.add_argument("--hosts", default="",
                    help="multi-laptop deployment: node_A=IP,node_B=IP,... "
                         "(see docs/05_MULTI_LAPTOP_SETUP.md)")
    ap.add_argument("--bundle", action="store_true",
                    help="multi-laptop: also write one ready-to-copy zip per node into <layout>/deploy/")
    args = ap.parse_args()
    hosts = parse_hosts(args.hosts)

    t0 = time.time()
    print("analysing corpus %s ..." % args.corpus)
    fwd = ForwardIndex(args.corpus)
    print("  %d documents, %d distinct terms, %d postings, %d tokens  (%.1f s)"
          % (fwd.num_docs, len(fwd.terms), fwd.fwd_term.size, int(fwd.doc_len.sum()),
             time.time() - t0))
    for n in args.shards:
        t1 = time.time()
        tag = args.tag or ("" if args.partition == "hash" else args.partition)
        print("building N = %d shards x %d copies on %d nodes (partition: %s)"
              % (n, args.copies, args.nodes, args.partition))
        out_dir = build(fwd, n, args.copies, args.nodes, args.base_port, args.partition, tag,
                        hosts=hosts)
        routing_rel = os.path.relpath(os.path.join(out_dir, "routing_table.json"))
        if hosts:
            print("  node -> laptop: %s" % ", ".join("%s=%s" % kv for kv in sorted(hosts.items())))
        if args.bundle:
            for zip_path in write_node_bundles(out_dir):
                print("  bundle -> %s  (%.1f MB)"
                      % (os.path.relpath(zip_path), os.path.getsize(zip_path) / 1e6))
            print("  copy each zip to its laptop, unzip, and follow README_NODE.txt inside")
        elif hosts:
            print("  copy each node_X folder to its laptop, then on that laptop run:")
            for node in sorted(hosts):
                print("    python cluster.py --routing %s --node %s" % (routing_rel, node))
        if hosts:
            print("  when all nodes are started, check from this laptop:")
            print("    python cluster.py --routing %s --check" % routing_rel)
        print("  done in %.1f s" % (time.time() - t1))


if __name__ == "__main__":
    main()
