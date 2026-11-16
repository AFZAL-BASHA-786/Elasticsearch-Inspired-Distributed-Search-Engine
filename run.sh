#!/usr/bin/env bash
# run.sh : everything in ONE terminal, start to finish (Git Bash / Linux / macOS).
#
#   bash run.sh              # steps 1-6 (about 2 minutes)
#   bash run.sh --benchmark  # also step 7 (about 3 more minutes)
#
# The step-by-step version with explanations is in ../docs/02_HOW_TO_RUN.md
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

echo; echo "=== 1. Compile the contract (search.proto -> Python stubs) ==="
$PY -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. search.proto

echo; echo "=== 2. Create the static document collection (only if missing) ==="
[ -f data/corpus.jsonl ] || $PY generate_corpus.py
$PY generate_corpus.py --tiny

echo; echo "=== 3. OFFLINE: build the indexes (immutable .npy files + routing table) ==="
$PY build_index.py --corpus data/tiny_corpus.jsonl --shards 1 --copies 1 --nodes 1 --tag tiny
$PY build_index.py --shards 3

echo; echo "=== 4. Open the box: the arrays from my notes, on disk ==="
$PY inspect_index.py --term raft

echo; echo "=== 5. EXPERIMENT 1: rankings vs the single-node index ==="
$PY compare_rankings.py --shards 2 3 4 8

echo; echo "=== 6. EXPERIMENT 3: the four-act failover demo (COMPLETE -> PARTIAL -> COMPLETE) ==="
$PY demo_failover.py

if [ "${1:-}" = "--benchmark" ]; then
  echo; echo "=== 7. EXPERIMENT 2: latency / tail latency / throughput vs shard count ==="
  $PY benchmark.py --shards 1 2 4 8 16
fi

echo; echo "All done. Results are in search_engine/results/"
