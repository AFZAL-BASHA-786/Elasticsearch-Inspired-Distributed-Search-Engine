# run_all.ps1 : everything in ONE terminal, start to finish (Windows PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File run_all.ps1            # steps 1-6 (about 2 minutes)
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -Benchmark # also step 7 (about 3 more minutes)
#
# The step-by-step version with explanations is in ../docs/02_HOW_TO_RUN.md
param([switch]$Benchmark)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "`n=== 1. Compile the contract (search.proto -> Python stubs) ===" -ForegroundColor Cyan
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. search.proto

Write-Host "`n=== 2. Create the static document collection (only if missing) ===" -ForegroundColor Cyan
if (-not (Test-Path data\corpus.jsonl)) { python generate_corpus.py }
python generate_corpus.py --tiny

Write-Host "`n=== 3. OFFLINE: build the indexes (immutable .npy files + routing table) ===" -ForegroundColor Cyan
python build_index.py --corpus data\tiny_corpus.jsonl --shards 1 --copies 1 --nodes 1 --tag tiny
python build_index.py --shards 3

Write-Host "`n=== 4. Open the box: the arrays from my notes, on disk ===" -ForegroundColor Cyan
python inspect_index.py --term raft

Write-Host "`n=== 5. EXPERIMENT 1: rankings vs the single-node index ===" -ForegroundColor Cyan
python compare_rankings.py --shards 2 3 4 8

Write-Host "`n=== 6. EXPERIMENT 3: the four-act failover demo (COMPLETE -> PARTIAL -> COMPLETE) ===" -ForegroundColor Cyan
python demo_failover.py

if ($Benchmark) {
    Write-Host "`n=== 7. EXPERIMENT 2: latency / tail latency / throughput vs shard count ===" -ForegroundColor Cyan
    python benchmark.py --shards 1 2 4 8 16
}

Write-Host "`nAll done. Results are in search_engine\results\" -ForegroundColor Green
