# ALFa homelab: architecture and 27B benchmarks

Local LLM inference over llama.cpp RPC, plus QLoRA training on a separate lock. No secrets, hostnames, or LAN addresses.

Operator runbooks stay in private `alfa-ai`. This repository is the public architecture and benchmark note.

Trainer configs: [alfa-train](https://github.com/Curt-Alfrey-s-Org/alfa-train) (private; disk path often `alfa-ai-train`).

References: [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md), [tensor-split](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md), [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [LiteLLM](https://docs.litellm.ai/docs/proxy/configs), [Axolotl](https://docs.axolotl.ai/), [PEFT](https://huggingface.co/docs/peft/index).

---

## Summary

Multiple NVIDIA GPUs on the LAN are one model via llama.cpp RPC. A CPU coordinator on the NAS holds the GGUF and shards layers to `rpc-server` processes. Clients use an OpenAI-compatible proxy (LiteLLM) on the NAS.

Only one of three GGUFs is loaded per cluster start. Inference and QLoRA cannot share the 12 GB gaming GPUs; a lock file on the Windows orchestrator blocks cluster start while training is active.

| Load | File class | Role |
|------|------------|------|
| Qwen3.8-27B Q8_0 | ~29 GB dense | Default inference |
| Qwen2.5-72B-Instruct Q5_K_M | ~52 GB dense | Exclusive inference; needs opt-in GPUs |
| Qwen3-Coder-Next Q5_K_M | ~57 GB MoE (~3B active) | Exclusive inference; needs opt-in GPUs |

Switching models is a full cluster stop/start. 27B and 72B/80B are not concurrent.

Slow/older GPUs remain in the inference mesh for VRAM (72B/80B fit), not 27B tok/s.

---

## Architecture

```
Product apps (own API + widget)
        │  POST /v1/chat/completions
        ▼
LiteLLM (NAS)     aliases: frontier-code, frontier
        ▼
llama-server (NAS CPU)     GGUF owner, --tensor-split
        │  RPC
        ▼
rpc-server × N (one process per GPU)
```

| Node | Role |
|------|------|
| NAS (TrueNAS) | GGUF hub, LiteLLM, ALFa API, llama-server, RTX 3050 tail shard |
| Dual-GPU Linux | 3060 12 GB + 5060 8 GB; largest 27B slices |
| Windows PC | Cluster start/stop UI; 3060 12 GB shard (WSL Docker) |
| Gaming PCs | Opt-in at start (required for 72B/80B). Two 5070 12 GB are default QLoRA ranks when inference is stopped |
| Optional | GTX 1070 8 GB, GTX 1660 Super 6 GB — extra VRAM, not throughput |
| Other VMs | App hosts, CPU RAG (embed/rerank), hypervisor, IoT Pi — no frontier GPU |

`rpc-server` is single-client until disconnect. After start, do not TCP-probe shard RPC ports or restart one shard. Health: coordinator `/health` plus a completion that returns text.

---

## Hardware

Inference without gaming opt-in (27B Q8):

| Node | CPU / RAM | GPU | Inference job |
|------|-----------|-----|----------------|
| NAS | Ryzen 9 7900X, ~46 GB ECC | RTX 3050 6 GB | Coordinator, proxy, API, tail shard |
| Dual-GPU Linux | i7-9700K, 48 GB | 3060 12 GB + 5060 8 GB | Two shards |
| Windows | Ryzen 9 7900X, 32 GB | 3060 12 GB | Orchestrator + one shard |

Opt-in at start:

| Node | GPU | Inference | QLoRA |
|------|-----|-----------|--------|
| Gaming A | 5070 12 GB | Opt-in | Rank 0 (default 14B) |
| Gaming B | 5070 12 GB | Opt-in | Rank 1 (default 14B) |
| Gaming C | 5060 8 GB | Opt-in | Not on default 14B job |
| Spare | GTX 1070 8 GB | Opt-in | No |
| GPU VM | GTX 1660 Super 6 GB | Opt-in | No |

VRAM (approx.):

| Layout | Physical | Assigned (`-ts`) | Use |
|--------|----------|------------------|-----|
| NAS + Linux dual-GPU + Windows 3060 | ~35–45 GB usable | — | 27B Q8 (~29 GB weights + KV for 4 slots) |
| All opt-in GPUs | ~84 GB | ~62–68 GB | 72B / 80B exclusive |
| Default train | 24 GB (2×12) | — | 14B QLoRA |

A 4-GPU ~40 GB train layout is documented; mixed 8 GB ranks failed before checkpoint, so the GUI does not launch it.

---

## Software

- One git clone per host; GitHub is the sync bus (no SCP of source).
- Artifacts ≥100 MB seed on the NAS; workers read NFS/SMB. No worker HF/Docker Hub pulls for those blobs.
- Docker: rpc-server, llama-server, LiteLLM, ALFa API, RAG, Axolotl.
- Client model ids: `frontier-code` (default), `frontier`. Both map to the GGUF loaded this boot.
- Windows GUI: separate Run vs Train screens.

---

## Models

### Qwen3.8-27B Q8_0 (default)

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), ~29 GB. Q8_0 vs FP16: fit and quality on this mesh; FP16 not used.

Coordinator: `--parallel 4`, `--ctx-size 163840` (40960/slot), KV `q4_0`.

Keep-warm: NAS cron every 15 min, four concurrent streaming completions to LiteLLM on localhost (all parallel slots). Cron user must read the env file that contains the proxy key.

### Qwen2.5-72B-Instruct Q5_K_M (exclusive)

[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct), ~52 GB dense. Requires all opt-in GPUs. `PARALLEL=1`, context 86016 (YaRN); 65536 if 6–8 GB cards are near OOM. Stop 27B first.

### Qwen3-Coder-Next Q5_K_M (exclusive)

[Qwen3-Coder-Next GGUF](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF), ~57 GB MoE. Same membership as 72B. `frontier-code` and `frontier` both alias this GGUF while loaded.

---

## Client repos

ALFa owns inference, LiteLLM, admin API, and chat-widget design. It does not store product databases.

Each product repo has its own API and DB. The widget calls that API; the API calls LiteLLM (`frontier-code` / `frontier`). Wiring is env only (`LITELLM_BASE_URL`, model alias). Product repos do not modify ALFa.

| Client | Function | Path |
|--------|----------|------|
| Contracts bot | Listing search / Q&A | Widget → contracts API → LiteLLM |
| Rankings | League rankings Q&A | Same; rankings DB only |
| Market pie | Market UI narratives | Same; market DB only |
| Credit-check | Credit-check workflows | Same; its DB only |
| ATEMS | App operator chat | Same |
| RV/auto finder | Listing assistant | Same |
| Auth service | Login | Not a GPU client |
| WordPress | Optional dock | Widget; no other product DB |

A contracts chat must not see rankings or market data. ALFa guest chat has no product DBs.

Model switch (27B → 80B) is cluster restart only. Client aliases stay the same.

---

## Benchmarks (Qwen3.8-27B Q8_0, full opt-in mesh)

Method: OpenAI streaming + `usage`.

- **TTFT** = time to first token.
- **Decode tok/s** = `completion_tokens / (total_s − TTFT)`.
- Wall tok/s (includes prefill) is not used as the health metric.

2026-08-17, Instruct, short completion:

| Metric | Value |
|--------|--------|
| UI TTFT | ~192 s |
| Decode after first token | ~4.4 tok/s |
| Coordinator, 15 prompt tokens | ~101 s prefill, then ~3.4 tok/s |

Cause: hybrid Gated DeltaNet + RPC over 1 Gbit Ethernet, not a failed pool.

Long UI threads resend full history each turn; ~600 s HTTP timeout is common. Use a new chat for smoke tests.

| Condition | Typical |
|-----------|---------|
| GGUF still loading | ~2 min coordinator not ready |
| First completion after idle | ~3–4 min wall for a short reply |
| Keep-warm (4 streams / 15 min) | Reduces cold unused slots; does not guarantee 1–4 s TTFT |
| Occasional warm short prompt | ~1–4 s TTFT (not reliable) |
| Typical short prompt, this 27B mesh | ~90–200 s TTFT, then ~3.5–4.4 tok/s |

72B / 80B: no decode log yet. Expect lower tok/s than 27B Q8 (`PARALLEL=1`, larger weights, same 1 Gbit RPC).

LAN (measured):

| Path | Result |
|------|--------|
| Dual-GPU Linux NIC | 1 Gbit negotiated |
| iperf3 Linux → Windows orchestrator | ~930–940 Mbit/s |
| ICMP to NAS | ~0.25 ms |
| 10G NICs on Windows | Present; RPC was not on that NIC |

Decode is network-bound until RPC is on >1 Gbit.

---

## Training (alfa-train)

[alfa-train](https://github.com/Curt-Alfrey-s-Org/alfa-train): Axolotl QLoRA. ALFa API does not train. GUI Start 14B stops inference, then launches trainers.

| Item | Value |
|------|--------|
| Base | [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) |
| Method | QLoRA SFT, JSONL `content`, seq 2048, micro-batch 1, gradient checkpointing |
| Parallel | torchrun / DeepSpeed ZeRO-2 |
| GPUs | 2× 5070 12 GB |
| Image | Official Axolotl (NAS tarball, `docker load`) |
| Metrics | TensorBoard; WinForms is start/stop/lock only |

Axolotl 13–14B QLoRA VRAM table: ~16–20 GB → two 12 GB cards.

| Dataset | Source | Use |
|---------|--------|-----|
| Repo JSONL | Multiple local git trees as completion rows | Default 14B job |
| Sleep pack | Feedback, error logs, curator lessons | Separate, operator-gated job |

Default 14B corpus is multi-repo, not ALFa-only.

Pipeline: app feedback → nightly curator (no weight update) → export + secret scan on NAS → QLoRA → held-out eval ≥ ~70% → optional LoRA GGUF and coordinator `--lora` after full inference restart. Base 27B/72B/80B GGUFs stay unchanged until that gate.

---

## Constraints

1. Full cluster start/stop only. No single-shard restart while the coordinator is connected.
2. GPU membership only at start.
3. Hub-first artifacts.
4. Same git commit on every clone before start.
5. Only the coordinator may connect to shard RPC ports.
6. Shard `restart: no`.
7. Decode gate before LiteLLM/API.
8. Train XOR infer on the 5070s (lock file).
