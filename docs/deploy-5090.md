# 5090 Deployment Guide — NInfer + Qwen3.8-27B NVFP4 with /metrics

Everything needed to replicate the 6000 Pro setup on a 5090 box. The 5090 is
compute 12.0 (sm_120) — same family as the 6000 Pro — so the **same branch
builds unmodified**: `CMAKE_CUDA_ARCHITECTURES=120a` covers both.

## Prerequisites on the 5090 box

| Component | Version | Notes |
|---|---|---|
| CUDA toolkit | 13.0+ (13.1 tested) | `nvcc --version` must show 13.x |
| CMake | ≥ 3.28 | hard requirement |
| Ninja | any recent | build generator |
| GCC | 13+ | C++20 |
| Python | 3.10+ | conversion tooling only |
| GPU | RTX 5090 (sm_120) | driver 570+ for CUDA 13 |
| Disk | ~60 GB free | 18 GB artifact + 15 GB build + source |

## 1. Clone the fork

```bash
git clone -b nvfp4-upstream-master git@github.com:alantheprice/ninfer-4090.git ninfer
cd ninfer
```

(HTTPS alternative: `https://github.com/alantheprice/ninfer-4090.git`)

This branch = upstream master `98dada0e` + 17 newer upstream perf commits +
our deltas: `/metrics` (Prometheus), `/slots`, `/usage` (incl. energy
accounting), `--auto-long-anchors`, `--electricity-rate`, `--metrics-state`
persistence, NVFP4 v2→v3 whitelist fix.

## 2. Build (sm_120a — same target as the 6000 Pro)

```bash
PATH=/usr/local/cuda-13.1/bin:$PATH \
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=120a
PATH=/usr/local/cuda-13.1/bin:$PATH ninja -C build -j$(nproc)
```

Binary: `build/apps/ninfer-serve`

## 3. The model artifact

Two options:

**A. Copy from the workstation** (fastest, same artifact):

```bash
# on the workstation:
scp /home/aprice/models/qwen3_8_27b_nvfp4_v3.ninfer <5090-box>:/home/<user>/models/
```

**B. Regenerate from the source NVFP4 artifact** (if transferring 18 GB is
inconvenient): download `Ostfralla/Qwen3.8-27B-NVFP4-NInfer` from HuggingFace
(v2 container), then upgrade in place:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
    qwen3_8_27b_nvfp4.ninfer qwen3_8_27b_nvfp4_v3.ninfer
```

(The whitelist already accepts this artifact's 1307-object layout.)

**Important**: the artifact is tied to the engine lineage. Use the
`nvfp4-upstream-master` branch's converter tooling — do NOT mix artifacts
from the `qwen3_5_9b` branch lineage.

## 4. Serve command (validated config)

```bash
sudo ./build/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4_v3.ninfer \
  --host 0.0.0.0 --port 8006 \
  --model-id qwen3.8-27b \
  --max-context 262144 \
  --kv-capacity 1700000 \
  --kv-dtype fp8 \
  --max-concurrency 8 \
  --max-pending-requests 64 \
  --pending-timeout-ms 900000 \
  --default-max-tokens 32768 \
  --spec mtp --draft-tokens 4 \
  --lm-head-draft \
  --preserve-thinking --vision \
  --prefill-chunk 2688 \
  --electricity-rate 0.125 \
  --metrics-state /var/lib/ninfer/metrics-state.json \
  --request-log-jsonl /var/log/ninfer/requests.jsonl
```

Measured on the 6000 Pro (should be similar on a 5090 — same arch, slightly
higher clocks, 32 GB vs 96 GB VRAM):

- prefill: 7.6–7.8K tok/s
- decode: 245 tok/s solo, 2,247–2,752 tok/s @ C=8
- MTP-4 acceptance ~77%
- energy: ~0.9 kWh / 1M output tokens solo (562 W)

**5090 VRAM note (32 GB vs 96 GB)**: `--kv-capacity 1700000` will NOT fit.
Budget: weights 16.8 GB + runtime ~10 GB + KV per-token fp8 ≈ 0.5 MB/K token…
scale `--kv-capacity` down to ~500000–700000 tokens (fits in ~10–14 GB KV) or
drop `--vision` (saves ~2 GB). Also consider `--max-concurrency 4` to halve
state-slot overhead. Concretely, start with:

```
--kv-capacity 600000 --max-concurrency 4
```

and raise until `capacity | ... | free N GiB` in the startup log shows 2–4 GiB
headroom.

## 5. systemd unit

```ini
[Unit]
Description=NInfer serve - Qwen3.8-27B NVFP4 (v3, fp8 KV, MTP-4 dt4)
After=network-online.target

[Service]
Type=simple
User=<youruser>
ExecStartPre=/bin/sh -c 'for i in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && exit 0; sleep 2; done; exit 1'
ExecStart=<path>/ninfer-serve <flags from step 4>
Restart=always
RestartSec=5
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp ninfer-5090.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ninfer-5090.service
```

`Restart=always` + the `ExecStartPre` GPU wait loop handle crash-recovery and
boot ordering (the GPU must be up before NVML init).

## 6. Endpoints

| Path | What |
|---|---|
| `/usage` | human JSON: uptime, tokens (in/cached/out), throughput, lanes, energy (daily + 30-day, cost at `--electricity-rate`) |
| `/metrics` | Prometheus: llama.cpp-compatible counters + ninfer reuse-path/lanes/KV/pressure |
| `/slots` | per-lane state + queue depth |
| `/health` | availability |

## Gotchas (learned the hard way on the 6000 Pro)

1. **systemctl restart can hang** during engine shutdown — use
   `systemctl stop --no-block`, wait for the port to free, kill -9 leftovers,
   then `start --no-block`.
2. **NVML arg order**: `nvmlDeviceGetHandleByIndex_v2(index, &device)` — if
   power sampling reads 0, check `[nvml]` stderr lines.
3. **Client prefix stability dominates cache-hit rate**: keep system prompts
   byte-identical across turns; volatile content at the END of the prompt.
   A mutating client can halve your hit rate (see results.md).
4. **Power cap matters for prefill only** — decode throughput was flat
   300–600 W on the 6000 Pro; don't expect cap changes to move decode.
5. **fp8 KV + chunk 2688** were worth +32% prefill vs defaults; revisit if
   the 5090 firmware differs.
