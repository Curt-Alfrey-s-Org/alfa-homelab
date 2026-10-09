# ALFa homelab: architecture and 27B benchmarks

LAN llama.cpp RPC inference. No secrets, hostnames, or LAN addresses.

Operator runbooks stay in private `alfa-ai`. This repository is the public architecture and benchmark note.

References: [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md), [tensor-split](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md), [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [LiteLLM](https://docs.litellm.ai/docs/proxy/configs).

---

## Summary

The cluster turns several NVIDIA GPUs on the LAN into one loaded GGUF. A CPU coordinator on the NAS owns the weights and splits layers across `rpc-server` processes (one process per GPU). Application UIs do not talk to those GPUs. Each product backend sends OpenAI-compatible `POST /v1/chat/completions` to LiteLLM on the NAS, which forwards to the coordinator.

Only one GGUF is in VRAM at a time. The default is Qwen3.8-27B Q8_0. Two larger loads (72B dense and 80B-class MoE) replace that file for the whole boot; they are not extra models sitting beside 27B.

Start and stop are a single orchestrated cycle from a Windows desktop GUI. Extra GPUs (gaming desktops, spare cards) are optional at that start. If a box is off or in use for games, it is left out. Membership cannot change until the next full stop/start.

| Load | File class | When |
|------|------------|------|
| Qwen3.8-27B Q8_0 | ~29 GB dense | Default. Runs on the always-on GPUs |
| Qwen2.5-72B-Instruct Q5_K_M | ~52 GB dense | Exclusive. Needs every extra GPU opted in |
| Qwen3-Coder-Next Q5_K_M | ~57 GB MoE (~3B active) | Exclusive. Same membership as 72B |

Older or slower cards stay in the mesh for VRAM so the large files fit. They are not there to raise 27B decode speed.

---

## Architecture

llama.cpp RPC is a coordinator plus remote backends. The coordinator (`llama-server` on NAS CPU) maps the GGUF with `--tensor-split` and opens an RPC session to each `rpc-server`. Each backend holds a slice of layers in that GPU's VRAM. Official RPC docs: backends are the set passed at coordinator start; there is no hot-add or hot-remove.

```
Product backends (own API, own database)
        │  POST /v1/chat/completions
        ▼
LiteLLM (NAS)     aliases: frontier-code, frontier
        ▼
llama-server (NAS CPU)     GGUF owner, --tensor-split
        │  RPC
        ▼
rpc-server × N (one process per GPU)
```

| Node | Role in inference |
|------|-------------------|
| NAS (TrueNAS) | GGUF hub, LiteLLM, ALFa HTTP API, llama-server, RTX 3050 as the small tail shard |
| Dual-GPU Linux | 3060 12 GB + 5060 8 GB. Largest 27B slices |
| Windows PC | Start/stop GUI. 3060 12 GB shard in WSL Docker |
| Extra desktops | Opt-in at start only. Required for 72B/80B |
| Spare / older GPUs | GTX 1070 8 GB, GTX 1660 Super 6 GB. Extra VRAM, not throughput |
| Other VMs | App hosts and CPU RAG (embed/rerank). No frontier GGUF on those GPUs |

`rpc-server` accepts one client until that client disconnects. After the coordinator is up, do not TCP-probe shard RPC ports and do not restart a single shard. Health is coordinator `/health` plus a completion that returns text.

---

## Starting inference and GPU opt-in

Inference is started from a Windows WinForms app on the orchestrator PC (desktop shortcut). Home has two workspaces: **Run models** and **Train**. Run models is stop, start, pool radio (27B / 72B / 80B), and which extra GPUs to include. It is not a per-host compose click.

**Always-on GPUs** (every inference start):

- NAS 3050
- Linux 3060 + 5060
- Windows orchestrator 3060

Those three boxes are the 27B Q8 mesh. They are enough for the default load. They are not enough for 72B or 80B.

**27B opt-in GPUs** (chosen at this start only):

- One 5060 8 GB gaming PC
- Two 5070 12 GB gaming PCs (blocked while the training lock is set; see [Training](#training))
- Spare GTX 1070 8 GB
- VM GTX 1660 Super 6 GB

**72B / 80B mesh** (exclusive load; same shard ports as the 27B opt-in, so 27B and 72B/80B never share a GPU at the same time):

- All five opt-in GPUs above. The two 5070 cards are required for the full 72B/80B mesh

Each extra host has a checkbox (or an equivalent Y/N prompt if start is run from the CLI). Check it only when that PC is on, reachable over SSH, and you are willing to give the GPU to inference for this session. Leave it unchecked if the machine is off, someone is gaming, or you do not need the VRAM.

| Goal | Action |
|------|--------|
| 27B on the core boxes | Start with no extra GPUs checked |
| 27B plus extra VRAM | Check whichever extra boxes are actually available (leave the two 5070 PCs out while training) |
| 72B or 80B | Check every extra GPU, including both 5070 PCs. Start fails or OOMs if the mesh is short |
| Take a GPU back for games | Full **Stop**, then **Start** with that box unchecked. Do not kill one shard while the coordinator is connected |
| Change model (27B to 80B, or the reverse) | Full stop, pick the other radio, start again |

Non-interactive CLI flags exist for the same choices (core only, last selection, or an explicit include list). The GUI is the normal path.

Start order, in short: write the active shard list from the checkboxes, fail-fast if a required host is down, bring shards up, then start the coordinator with that list, then LiteLLM/API after a decode that returns text. Stop tears down every host that was in the last active list.

---

## Hardware (inference)

Core mesh (27B Q8 without extra GPUs):

| Node | CPU / RAM | GPU | Job |
|------|-----------|-----|-----|
| NAS | Ryzen 9 7900X, ~46 GB ECC | RTX 3050 6 GB | Coordinator, LiteLLM, API, tail shard |
| Dual-GPU Linux | i7-9700K, 48 GB | 3060 12 GB + 5060 8 GB | Two shards |
| Windows | Ryzen 9 7900X, 32 GB | 3060 12 GB | Orchestrator + one shard |

Opt-in mesh (27B; this start):

| GPU | Typical use |
|-----|-------------|
| 5060 8 GB (desktop C) | Extra 27B slice |
| 5070 12 GB (desktops A and B) | Extra 27B slice; required for the full 72B/80B mesh. Not available to 27B while the training lock is set |
| GTX 1070 8 GB | VRAM only |
| GTX 1660 Super 6 GB | VRAM only |

VRAM (approx.):

| Layout | Physical | Assigned (`-ts`) | Load |
|--------|----------|------------------|------|
| Core only | ~38 GB physical | 35.0 GB (`12,8,10,5`) | 27B Q8 (~29 GB weights + q4_0 KV) |
| Core + 5060, 1070, 1660 Super | ~60 GB physical | 52.0 GB (`12,8,5,10,7,5,5`) | 27B without the two 5070 PCs (for example while training) |
| Core + all five opt-in | ~84 GB physical | 70.0 GB (`12,8,5,9,9,10,7,5,5`) | 27B with every opt-in GPU |
| 72B / 80B full mesh | ~84 GB physical | ~62 GB | Exclusive 72B/80B (both 5070 PCs required; not concurrent with 27B) |

---

## Software (inference)

- One git clone per host. GitHub is the sync bus. Source is not copied with SCP.
- Files ≥100 MB seed on the NAS. Workers read NFS/SMB. Workers do not pull those blobs from Hugging Face or Docker Hub.
- Docker runs `rpc-server`, `llama-server`, LiteLLM, the ALFa API, and CPU RAG sidecars.
- Cluster aliases are `frontier-code` (production default) and `frontier`. Both names are remapped to whichever GGUF this boot loaded. Product apps do not select GGUF files.

---

## Models

### Qwen3.8-27B Q8_0 (default)

Architecture: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Production GGUF: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-Q8_0.gguf` (~29 GB). Q8_0 is the production quant on this mesh: it fits the mesh VRAM with KV for five parallel slots. FP16 is not used.

Coordinator flags: `--parallel 5`, `--ctx-size 262144` (52428 per slot; Qwen3.8 native max), `--batch-size 8192`, `--ubatch-size 1024`, KV `q4_0`. Five slots are two interactive and three automated (keep-warm and similar). This sizing assumes the 5060 opt-in PC joins at cluster start.

Keep-warm: NAS cron every 15 minutes sends five concurrent streaming completions to LiteLLM on localhost so unused slots do not go fully cold. The cron user must be able to read the env file that holds the proxy key. Keep-warm does not turn 90–200 s TTFT into 1–4 s on this hardware; it only trims idle-slot cold start.

### Qwen2.5-72B-Instruct Q5_K_M (exclusive)

[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) GGUF [Q5_K_M](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF), ~52 GB dense. The core boxes cannot hold it. Start with every opt-in GPU checked, after 27B is stopped. `PARALLEL=1`, context 86016 (YaRN); drop to 65536 if the 6–8 GB cards are near OOM.

### Qwen3-Coder-Next Q5_K_M (exclusive)

[Qwen3-Coder-Next GGUF](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF), ~57 GB MoE with ~3B active. Same GPU membership as 72B. While this file is loaded, `frontier-code` and `frontier` both alias it. Apps keep the same env; only the cluster start radio changes.

---

## How applications use inference

ALFa is the inference service: GGUF load, LiteLLM, admin HTTP, and the shared chat-widget design. It does not store the other apps' business data.

Each other repo is a consumer. It keeps its own API, auth, and database. A chat widget on that site posts to **that** API (typically an app-local AI route), not to ALFa's guest chat. The product API then calls LiteLLM with OpenAI-compatible chat completions. Production default alias is `frontier-code`. Some apps use other LiteLLM alias names; those names are configured on the proxy to the same loaded GGUF. Coupling is environment only (`LITELLM_BASE_URL`, model alias). Consumer repos do not vendor or patch the inference stack.

What they use it for:

- Chat and Q&A in the product UI (completions against that product's own records).
- Short narratives or explanations the product API asks the model to write from data it already loaded.
- Operator chat inside an app, still scoped to that app's API.

What they do not use it for:

- Login and session services (no GPU).
- Cross-app retrieval. A chat opened on one product must not read another product's database. ALFa guest chat has no product DBs either.

Changing 27B to 80B is a cluster restart. Client aliases stay valid, so app env does not need a GGUF or weight-name edit for that switch.

---

## Benchmarks (Qwen3.8-27B Q8_0, core + opt-in GPUs)

Method: OpenAI streaming plus `usage`.

- **TTFT** = time to first token.
- **Decode tok/s** = `completion_tokens / (total_s − TTFT)`.
- Wall tok/s (prefill included) is not the health metric. Prefill on this mesh is tens to hundreds of seconds; mixing it into tok/s hides decode.

2026-08-17, Instruct, short completion, all extra GPUs in:

| Metric | Value |
|--------|--------|
| UI TTFT | ~192 s |
| Decode after first token | ~4.4 tok/s |
| Coordinator, 15 prompt tokens | ~101 s prefill, then ~3.4 tok/s |

This is hybrid Gated DeltaNet plus RPC over 1 Gbit Ethernet. It is not a sign that shards failed to attach.

Long UI threads resend the full history every turn. A ~600 s HTTP timeout is common. Use a new chat for smoke tests.

| Condition | Typical |
|-----------|---------|
| GGUF still loading | ~2 min coordinator not ready |
| First completion after idle | ~3–4 min wall for a short reply |
| Keep-warm (4 streams / 15 min) | Fewer fully cold unused slots; not 1–4 s TTFT |
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

Decode stays network-bound until RPC is on a faster path than 1 Gbit.

---

## Training

Training is a separate workspace on the same Windows GUI, not a second mode of the inference start screen. Configs live in the private `alfa-train` repo (disk path often `alfa-ai-train`). The ALFa API does not train.

The two 5070 12 GB cards are used for inference **or** for QLoRA, not both. Starting a 7B train job sets a lock so those two GPUs cannot join the 27B opt-in mesh or a 72B/80B start. The core 27B mesh stays up while training.

| Item | Value |
|------|--------|
| Base | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| Method | QLoRA SFT, JSONL `content`, seq 512, micro-batch 1, gradient checkpointing |
| Parallel | torchrun DDP (Axolotl's default when DeepSpeed/FSDP are unset: [multi-GPU](https://docs.axolotl.ai/docs/multi-gpu.html)). Each rank holds a full replica; DDP does not pool VRAM |
| GPUs | 2× 5070 12 GB |
| Image | Official Axolotl (NAS tarball, `docker load`) |
| Metrics | TensorBoard. The GUI is start/stop/lock only |

Axolotl 7–8B QLoRA VRAM table: ~10–14 GB per replica ([choosing method](https://docs.axolotl.ai/docs/choosing_method.html)), so one 12 GB card holds a replica at seq 512. Longer sequences hit CUDA errors on these 12 GB WSL cards. 14B QLoRA (~16–20 GB) is parked and the GUI does not start it. FSDP+QLoRA is not used for Qwen ([fsdp_qlora](https://docs.axolotl.ai/docs/fsdp_qlora.html)).

| Dataset | Source | Use |
|---------|--------|-----|
| Repo JSONL | Multiple local git trees as completion rows | Default 7B job |
| Cluster FAQ chat | Short chat-format Q&A about the cluster | Mixed into the default 7B job |
| Sleep pack | Feedback, error logs, curator lessons | Separate, operator-gated job |

The default 7B corpus is multi-repo, not ALFa-only. Each new run continues the existing LoRA adapter on the full rebuilt corpus (not only the new diffs) unless a fresh adapter is chosen.

Pipeline: app feedback → nightly curator (no weight update) → export + secret scan on NAS → QLoRA → held-out smoke eval vs base (~70% keyword threshold, informational) and a live chat gate → optional LoRA GGUF and coordinator `--lora` after a full inference restart. Base 27B/72B/80B GGUFs stay unchanged until that gate.

References: [Axolotl](https://docs.axolotl.ai/), [PEFT](https://huggingface.co/docs/peft/index).

---

## Constraints

1. Full cluster start/stop only. No single-shard restart while the coordinator is connected.
2. GPU membership is chosen at start and then frozen.
3. Hub-first artifacts (≥100 MB on the NAS before workers read them).
4. Same git commit on every clone before start.
5. Only the coordinator may connect to shard RPC ports.
6. Shard `restart: no`.
7. Decode gate before LiteLLM/API.
