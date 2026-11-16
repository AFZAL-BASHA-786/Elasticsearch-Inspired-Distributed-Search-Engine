# Running the search engine on several laptops

**Goal:** one laptop is the **coordinator**, the other laptops are **nodes** that hold shard copies — a real distributed deployment instead of processes on one machine.

## What the code provides for this

| Option | File | What it does |
|---|---|---|
| `--hosts node_A=IP,node_B=IP,…` | `build_index.py` | writes each node's **real IP address** into `routing_table.json` |
| `--bundle` | `build_index.py` | writes **one zip per node** (`deploy/node_A.zip`, …) containing the code, the routing table, that node's shard files, a `README_NODE.txt` and a `start_node_A.bat` |
| `--node node_A` | `cluster.py` | on a node laptop, starts **only that node's** shard copies; they listen on `0.0.0.0` so other laptops can reach them |
| `--check` | `cluster.py` | from the coordinator laptop, tests that **every shard copy is reachable** (IP, firewall, server, gRPC in one go) |
| `--bind 0.0.0.0` | `coordinator.py` | (optional) lets a client or load generator on **another** laptop reach the coordinator |
| `--routing <file>` | `compare_rankings.py` | compares the rankings of the **already-running** multi-laptop cluster with the single-node index |
| `--remote HOST:PORT` | `benchmark.py` | measures an **already-running** coordinator; starts nothing locally; results accumulate in `results/benchmark_remote.json` |

Also: a shard server started by `cluster.py` now **exits when its launcher exits**, so closing a node's launcher window really takes that node's copies down (no leftover processes).

**Tested on one machine** with three "nodes" on three different network addresses (this laptop's Wi-Fi IP, `127.0.0.2`, `127.0.0.3`), each started from its **own unzipped bundle folder** with its own launcher:
- `--check` said NO for all copies before the nodes started and yes for all after;
- all 6 copies UP;
- rankings bit-identical to the single node (200 / 200 queries, cached and dfs);
- remote benchmark worked;
- killing node_B's launcher → both its copies DOWN, cluster YELLOW, **80 / 80 queries still COMPLETE**.

It has **not yet been run on separate laptops** — that is what this guide is for.

---

## 0. The plan

Layout: **3 shards × 2 copies on 3 node laptops**, plus **1 coordinator laptop**:

| Laptop | Role | Shard copies (chosen by the allocator) | Ports |
|---|---|---|---|
| **Laptop 0** | coordinator, client, load generator | none | 20050 |
| **Laptop A** | node_A | S0 copy 0, S1 copy 1 | 20051, 20054 |
| **Laptop B** | node_B | S1 copy 0, S2 copy 1 | 20053, 20056 |
| **Laptop C** | node_C | S0 copy 1, S2 copy 0 | 20052, 20055 |

Any one node laptop can die and every shard still has a live copy.

Only 2 or 3 laptops? See §9.

---

## 1. Network

1. Put **all laptops on the same network**. Best: a **phone hotspot** or a home router. College/hostel Wi-Fi usually blocks laptop-to-laptop traffic.
2. On every laptop run `ipconfig` and note the **IPv4 Address** of the Wi-Fi adapter. Example used below:

   | Laptop | IP |
   |---|---|
   | Laptop 0 | 192.168.43.20 |
   | Laptop A | 192.168.43.21 |
   | Laptop B | 192.168.43.22 |
   | Laptop C | 192.168.43.23 |

3. Keep every laptop connected for the whole session — an IP can change after reconnecting (then redo §3).
4. On every laptop: Settings → Network → Wi-Fi → (your network) → set it to **Private**.

---

## 2. Python on every laptop

Install the **same Python version** on all laptops (tested with 3.9), with "Add Python to PATH" ticked. Nothing else is needed on the node laptops yet — the code arrives in the bundle in §4.

Laptop 0 already has the full `search_engine` folder (this project).

---

## 3. Build the index and the bundles (on Laptop 0)

```powershell
cd "<project>\search_engine"
python generate_corpus.py                          # skip if data\corpus.jsonl exists
python build_index.py --shards 3 --copies 2 --nodes 3 --tag lan --bundle --hosts "node_A=192.168.43.21,node_B=192.168.43.22,node_C=192.168.43.23"
```

Use **your** three IPs. Actual output:

```
  shard 0   docs=16548 ... copies on node_A, node_C
  shard 1   docs=17009 ... copies on node_B, node_A
  shard 2   docs=16443 ... copies on node_C, node_B
  node -> laptop: node_A=192.168.43.21, node_B=192.168.43.22, node_C=192.168.43.23
  bundle -> index_data\shards_3_copies_2_lan\deploy\node_A.zip  (3.0 MB)
  bundle -> index_data\shards_3_copies_2_lan\deploy\node_B.zip  (3.0 MB)
  bundle -> index_data\shards_3_copies_2_lan\deploy\node_C.zip  (3.0 MB)
  copy each zip to its laptop, unzip, and follow README_NODE.txt inside
  when all nodes are started, check from this laptop:
    python cluster.py --routing index_data\shards_3_copies_2_lan\routing_table.json --check
```

---

## 4. Deploy: one zip per laptop (this *is* replication)

Send `node_A.zip` to Laptop A, `node_B.zip` to Laptop B, `node_C.zip` to Laptop C (USB stick, WhatsApp/Telegram to yourself, Google Drive — each is only ~3 MB).

Each zip contains the program, the routing table, **only that node's** shard files, and a `README_NODE.txt` with that laptop's exact commands. On each node laptop:

1. **Unzip** anywhere and open PowerShell in the `search_engine` folder inside it.
2. **Install** (once):
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. **Open the firewall** (once). PowerShell **as Administrator** — the exact line with that laptop's ports is in its `README_NODE.txt`, e.g. for node_A:
   ```powershell
   New-NetFirewallRule -DisplayName "DSE shards" -Direction Inbound -Protocol TCP -LocalPort 20051,20054 -Action Allow -Profile Any
   ```
   If Windows asks whether Python may use the network, allow **Private and Public**. (Remove later: `Remove-NetFirewallRule -DisplayName "DSE shards"`.)

---

## 5. Start the nodes, then check

On each node laptop, double-click `start_node_A.bat` (or `_B`, `_C`), or type the command from its README:

```powershell
python cluster.py --routing index_data\shards_3_copies_2_lan\routing_table.json --node node_A
```

It prints `launching 2 shard copies of node_A …`, one `listening on 0.0.0.0:<port>` line per copy, then `all copies are serving.` and the `cluster>` console (kill / freeze / restart work for that laptop's copies).

**Then, on Laptop 0, one command checks everything** (right IPs, firewalls, servers, gRPC):

```powershell
python cluster.py --routing index_data\shards_3_copies_2_lan\routing_table.json --check
```

```
shard  copy  node     address                reachable?
0      0     node_A   192.168.43.21:20051    yes
0      1     node_C   192.168.43.23:20052    yes
1      0     node_B   192.168.43.22:20053    yes
1      1     node_A   192.168.43.21:20054    yes
2      0     node_C   192.168.43.23:20055    yes
2      1     node_B   192.168.43.22:20056    yes
all copies reachable -> the coordinator will start GREEN
```

A `NO` line names the copy to fix (see §10). Do not go on until every line says `yes`.

---

## 6. Start the coordinator (Laptop 0)

```powershell
python coordinator.py --routing index_data\shards_3_copies_2_lan\routing_table.json
```

Expect `cluster is GREEN (3 shards x 2 copies, stats mode: cached)`. A copy reported `DOWN` at start = that address is unreachable → run the `--check` from §5 again.

(Add `--bind 0.0.0.0` only if a client will run on a *different* laptop; then also open port 20050 in Laptop 0's firewall.)

---

## 7. Use it (Laptop 0, second window)

```powershell
python client.py --health                                  # all 6 copies UP, with the node IPs
python client.py "heartbeat failover" -k 5 --titles
```

### 7.1 Rankings across machines = single node?

```powershell
python compare_rankings.py --routing index_data\shards_3_copies_2_lan\routing_table.json
```

Expected: `cached` and `dfs` 100% bit-identical, `local` drifting. Saved to `results\compare_rankings_remote.json`.

### 7.2 Latency and throughput across machines

Get the coordinator's process id (so its CPU is reported), then measure:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like "*coordinator.py*" | Select-Object ProcessId
python benchmark.py --remote 127.0.0.1:20050 --coordinator-pid <that id>
```

Each run adds one row (for the shard count the coordinator is serving) to `results\benchmark_remote.json` and redraws `results\benchmark_remote.png`. **To compare shard counts**, repeat §3–§7.2 with `--shards 2`, `--shards 6` etc. (rebuild with `--bundle`, send the new zips, restart nodes and coordinator); the rows accumulate. Note in your report how many laptops, their CPUs, and the network used.

---

## 8. The failure demonstration on real machines

On Laptop 0, start a stream and watch it plus the coordinator window:

```powershell
python client.py "heartbeat failover" -k 5 --repeat 1200 --interval-ms 100 --quiet
```

| Act | What you do | Expected |
|---|---|---|
| 1 | nothing | `COMPLETE shards 3/3 cluster GREEN` |
| 2 | on Laptop A: `kill 0 0` | still `COMPLETE`, same hits; after ~3 s `YELLOW` |
| 3 | on Laptop C: `kill 0 1` | `PARTIAL shards 2/3 … failed shards: 0`, different hits; `RED` |
| 4 | `restart 0 0` on A, `restart 0 1` on C | back to `COMPLETE`, `GREEN`, original hits |
| 5 | **turn off Wi-Fi on Laptop B** (or close its lid) | still `COMPLETE` — S1 and S2 have copies on A and C. Some queries take ~200 ms: a vanished machine sends no "connection refused", so only the **deadline** ends the wait (the true slow failure). Both B copies DOWN, `YELLOW` |
| 6 | Wi-Fi on Laptop B back on | after 2 heartbeats both copies UP, `GREEN` |
| 7 | close the launcher window on Laptop B | same as act 5, but a fast failure (connection refused) |
| 8 (optional) | take Laptops A **and** B away | S1 has no copy → `PARTIAL`, failed shard 1 |

Act 5 is the one a single laptop cannot show: a whole machine and both copies on it fail together, and replicas on other machines save every query.

---

## 9. Fewer laptops

| You have | Do this |
|---|---|
| **3 laptops** | Laptop 0 is the coordinator **and** node_C: use Laptop 0's IP for `node_C` in §3 and also run `cluster.py … --node node_C` on it |
| **2 laptops** | `--nodes 2` and `--hosts "node_A=<laptop 1 IP>,node_B=<laptop 0 IP>"`; run `--node node_A` on laptop 1 and `--node node_B` plus the coordinator on laptop 0. Every shard then has one copy on each laptop, so either laptop can disappear without losing a shard |

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `--check` says `NO` for a copy | in this order: is that node's launcher running? does `ipconfig` on that laptop still show the IP in the table? was the firewall rule added (§4 step 3)? is the Wi-Fi network set to Private? If all yes: the Wi-Fi isolates clients → use a phone hotspot |
| `--check` says `NO` for **every** copy on all laptops | the network blocks laptop-to-laptop traffic (typical for college Wi-Fi) → phone hotspot |
| `missing …\node_B\shard_1_copy_0 - copy this node's folder` | you started the wrong node name on this laptop, or unzipped another laptop's bundle |
| `the routing table places no copy on node 'node_X'` | wrong `--node` name for this layout |
| `cannot start … (port in use?)` | an old launcher is still running on that laptop; `quit` it |
| Coordinator shows a copy DOWN at start | that IP/port unreachable; fix, then restart the coordinator after all nodes are up |
| Queries slow (20–100 ms) | Wi-Fi power saving or weak signal: laptops on power, close to the hotspot, power plan "Best performance" |
| An IP changed | rebuild with the new `--hosts … --bundle`, send the new zips, restart |
| `python` is not recognised on a node laptop | reinstall Python with "Add Python to PATH", or use `py` instead of `python` |

---

## 11. Checklist for the day (about 30 minutes with 4 laptops)

- [ ] All laptops on one hotspot, network set to Private, on charger
- [ ] `ipconfig` on each → IPs written down
- [ ] Laptop 0: `build_index.py … --tag lan --bundle --hosts …`
- [ ] Each node laptop: zip received → unzip → `pip install -r requirements.txt` → firewall rule → `start_node_X.bat`
- [ ] Laptop 0: `cluster.py --routing … --check` → six times `yes`
- [ ] Laptop 0: `coordinator.py --routing …` → GREEN
- [ ] `client.py --health`, one query, then the stream
- [ ] `compare_rankings.py --routing …` → 100% bit-identical for cached and dfs
- [ ] `benchmark.py --remote 127.0.0.1:20050 --coordinator-pid …`
- [ ] Failure acts from §8; **take screenshots or a screen recording** of the stream and the coordinator log
- [ ] Note the date, number of laptops, CPU models / core counts, and the network — you will be asked
- [ ] Save `results\compare_rankings_remote.json`, `results\benchmark_remote.json` and `.png`

---

## 12. What you can claim afterwards

Only after you actually run it:

- "Deployed across **N laptops** (one coordinator, N−1 storage nodes) over Wi-Fi with gRPC."
- "Verified failover with a **real machine failure**: with one laptop off the network, all queries stayed COMPLETE via replicas on other machines; losing every copy of a shard returned PARTIAL."
- "Rankings across machines bit-identical to a single-node index."
- Latency / throughput numbers **with the setup stated** (e.g. "p50 X ms over Wi-Fi, 4 laptops").

The résumé line to switch to after the run is in `Mathworks Interview/RESUME_LINE_BY_LINE_ANSWERS.md`.
