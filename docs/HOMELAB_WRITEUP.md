# Homelab write-up: ALFa AI chat cluster + alfa-train

How this house runs local LLMs for websites and bots, and how it fine-tunes on its own code. **No secrets, accounts, hostnames, or LAN addresses.**

This file is the **public** copy. Day-to-day operator docs live in a private repo.

Trainer project (private): [alfa-train](https://github.com/Curt-Alfrey-s-Org/alfa-train) (folder on disk is often `alfa-ai-train`).

Official references: [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md), [multi-GPU / `--tensor-split`](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md), [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [LiteLLM proxy](https://docs.litellm.ai/docs/proxy/configs), [Axolotl](https://docs.axolotl.ai/), [PEFT / LoRA](https://huggingface.co/docs/peft/index).

---

## What this is, in plain language

I did not buy one giant GPU. I glued together the GPUs I already have (gaming PCs, a Linux box with two cards, a NAS with a small card) so they act like **one model**.

**When I want chat:** a program on the NAS loads a model file (GGUF) into CPU RAM, then sends layers over the home network to each GPU. Anything that speaks OpenAI’s API (websites, bots, scripts) talks to a proxy on the NAS. That proxy talks to the NAS program. The NAS program talks to the GPUs.

**When I want to fine-tune:** I stop chat completely, then two gaming GPUs run a trainer (`alfa-train`). Chat and training cannot share those GPUs. A lock file on the Windows PC blocks “start chat” until training is stopped.

**I only load one of three models at a time:**

| I pick this at start | What it is | When I use it |
|----------------------|------------|----------------|
| **Qwen3.8-27B Q8** | 27 billion parameters, high-quality 8-bit weights | Everyday chat and coding for all the apps |
| **Qwen2.5-72B Instruct Q5** | Dense 72B (every layer is used) | Longer / harder questions when I can spare the extra GPUs |
| **Qwen3-Coder-Next 80B Q5** | Mixture-of-experts: ~80B on disk, ~3B “on” per token | Coding-heavy sessions |

Switching 27B → 72B or 80B means **stop the cluster, start it again** with that choice. The GPUs cannot hold 27B and 72B together.

Accuracy matters more than speed. Old/slow GPUs stay in the chat pool on purpose (extra memory for 72B/80B). They do not make 27B faster.

---

## How the pieces fit (architecture)

Think of four layers. Users only ever see the top one.

```
Websites and bots  (contracts, rankings, market, and so on)
        │  each app has its own API + chat widget
        │  they send OpenAI-style /v1/chat/completions
        ▼
LiteLLM on the NAS   (names the model “frontier-code” / “frontier”)
        │
        ▼
llama.cpp server on the NAS CPU   (holds the GGUF, splits work)
        │  RPC over the LAN
        ▼
One process per GPU   (rpc-server: 3060s, 5060s, 5070s, 3050, optional 1070 / 1660 Super)
```

**NAS (TrueNAS):** download store for big files, OpenAI proxy, ALFa website/API (“the brain”), and the CPU process that owns the GGUF. Small GPU here only holds a **tail** of the model.

**Linux PC with two GPUs (3060 12 GB + 5060 8 GB):** holds the **biggest slices** of the model whenever chat is running.

**Windows desktop (3060 12 GB):** the **start/stop button** for the whole house, plus one more GPU slice (Docker in WSL).

**Gaming PCs (two 5070 12 GB, one 5060 8 GB):** I am asked yes/no **each time I start chat**. If I am playing or those PCs are off, chat still works on the NAS + Linux dual-GPU + Windows 3060. If I say yes, their VRAM joins the same model (needed for 72B and 80B). Those same 5070s are the **default trainers** when chat is stopped.

**Optional extras:** an old 1070 and a 1660 Super can join chat if they are on. They are extra memory, not extra speed.

**Separate VMs (no frontier GPU):** app servers, a CPU box for search/embed (RAG), a hypervisor, a Pi for home automation.

llama.cpp `rpc-server` talks to **one** client until that client disconnects. After chat is up, do not poke GPU RPC ports or restart one GPU container. Health check = the NAS server `/health` plus a **real short answer**, not a port ping.

---

## Hardware (what each machine is for)

Machines that **run chat whenever I start the cluster** (enough for 27B Q8):

| Machine | CPU / RAM | GPU | Job while chat is up |
|---------|-----------|-----|----------------------|
| NAS | Ryzen 9 7900X, ~46 GB ECC | RTX 3050 6 GB | Model file, proxy, website/API, tiny GPU slice |
| Dual-GPU Linux | i7-9700K, 48 GB | 3060 12 GB + 5060 8 GB | Two GPU slices (largest) |
| Windows desktop | Ryzen 9 7900X, 32 GB | 3060 12 GB | Cluster start/stop UI + one GPU slice |

Machines I **invite at start** (needed for 72B/80B; optional for 27B):

| Machine | GPU | Chat | Train |
|---------|-----|------|--------|
| Gaming PC A | 5070 12 GB | Opt-in slice | Default trainer GPU 0 |
| Gaming PC B | 5070 12 GB | Opt-in slice | Default trainer GPU 1 |
| Gaming PC C | 5060 8 GB | Opt-in slice | Not used on the default 14B train job |
| Spare PC | GTX 1070 8 GB | Opt-in only | No |
| Small GPU VM | GTX 1660 Super 6 GB | Opt-in tail | No |

Memory ballpark:

- Chat on NAS + dual-GPU Linux + Windows 3060: about **35–45 GB** of GPU memory the model can use (27B Q8 is ~29 GB of weights, plus KV cache for four simultaneous chats).
- Chat with every optional GPU: about **84 GB** on the cards / **62–68 GB** we actually assign (that is what 72B and 80B need).
- Default training: **two 12 GB cards** (24 GB) for 14B QLoRA. A four-GPU ~40 GB layout exists on paper; mixed 8 GB cards died overnight before a checkpoint, so the GUI does not start that.

---

## Software (no secrets)

- Each PC has its **own git clone**. GitHub is how they stay in sync. I do not copy source trees with SCP.
- **Big files live on the NAS first** (model weights, Docker images). Other PCs read them from the LAN. Trainers do not download 20 GB from the public internet.
- **Docker** runs the GPU workers, the NAS server, the proxy, the website, search, and the trainer.
- Apps ask for models named **`frontier-code`** (default) and **`frontier`**. Those names always point at **whichever of the three GGUFs I loaded this boot**.
- Desktop GUI on Windows: Home is two doors — **Run chat** vs **Train** — not one mixed screen.

---

## The three models (the only ones I load)

### Everyday — Qwen3.8-27B Q8

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) **Q8_0** (~29 GB). I stay on 8-bit rather than 16-bit: quality is already the point, and 16-bit would not fit this mesh as comfortably.

NAS server settings: four chats at once (`--parallel 4`), 163840 tokens total context (40960 per chat), KV cache `q4_0`.

A cron job on the NAS every 15 minutes sends **four tiny streaming requests** to the proxy on localhost so all four chat slots have been used once. That cron must be able to **read** the env file that holds the proxy key.

This is what every website/bot uses unless I have switched the whole house to 72B or 80B.

### Harder questions — dense 72B

[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) **Q5_K_M** (~52 GB). Dense = every parameter can fire. Needs **every optional GPU** as well as the NAS/Linux/Windows GPUs. One chat at a time, context 86016 (YaRN). Stop 27B first. If a 6–8 GB card is almost full, I drop context to 65536.

### Coding sessions — 80B MoE coder

[Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) **Q5_K_M** (~57 GB). Mixture-of-experts: large on disk, small “active” per token (~3B). Same “everyone joins, 27B is off” rule as 72B. Both `frontier-code` and `frontier` point at this coder so the apps do not 404.

---

## How other git repos use this (the actual point of the cluster)

**alfa-ai is only the brain.** It owns the GPU pool, the OpenAI proxy, the admin site, and the shared chat-widget design. It does **not** store another app’s business data.

Each product is its **own repo**, its **own database**, its **own API**. The chat bubble on that site talks to **that** API. That API then calls the NAS proxy with `frontier-code` (or `frontier`). Environment variables only: proxy URL + model name. Those repos are not allowed to edit alfa-ai.

| Product repo (examples) | What the human uses it for | How it uses ALFa |
|-------------------------|----------------------------|------------------|
| Government **contracts** bot | Search / explain listings | Chat widget → contracts API → LiteLLM |
| **Rankings** / pool-league site | Player rankings, site Q&A | Same pattern, rankings DB only |
| **Market** pie bot | Market UI + narratives | Same pattern, market DB only |
| **Credit-check** bot | Credit-check workflows | Same pattern, its DB only |
| **ATEMS** | That app’s operators | Same pattern |
| **RV/auto finder** | Listings assistant | Same pattern |
| **Auth** service | Login for the apps | Not a GPU client |
| Public **WordPress** sites | Optional ALFa dock | Theme/widget; still no other bot’s data |

Isolation is the rule: a chat opened on the contracts site must not see rankings or market data. The brain’s own guest chat has **no** bot databases at all.

So day to day: I start the cluster on 27B Q8, keep-warm runs, and every one of those apps is just another OpenAI client on the LAN. If 27B is not enough for a coding binge, I stop, start 80B, and the **same** `frontier-code` name now hits the coder. Apps do not change code for that switch.

The shared look of the chat bubble (avatar, dock size, colors) is defined in alfa-ai and copied into each frontend on purpose. Behavior and data stay in the product repo.

---

## Measured chat speed (Qwen3.8-27B Q8, every GPU that joined)

How I count it (OpenAI streaming + `usage`):

- **Time to first token (TTFT)** = how long you wait before the first word.
- **Decode tok/s** = tokens after that / time after that — how fast it types.
- Ignore “tokens / total time” on long prompts; that mixes waiting and typing.

Logged **2026-08-17**, Instruct, short reply:

| Metric | Measured |
|--------|----------|
| TTFT in the web UI | **~192 s** |
| Typing after the first token | **~4.4 tok/s** |
| NAS probe, 15-token prompt | **~101 s** of prefill, then **~3.4 tok/s** |

That is the live 27B setup, not a crashed pool. Hybrid architecture + splitting layers over a **1 Gbit** network is slow to start and then types at a few tokens per second.

If you keep one huge thread, the website sends the **entire history** every turn and often hits a **~10 minute** timeout. New chat for a smoke test.

| Situation | Typical |
|-----------|---------|
| NAS still loading the GGUF | **~2 min** not ready |
| First answer after a long idle | **~3–4 min** for a short completion |
| Keep-warm every 15 min (4 streams) | Stops unused slots from being ice-cold; does not promise a 2 s first token |
| Lucky warm short question | **~1–4 s** to first token, not reliable |
| Normal short question on this 27B pool | **~90–200 s** to first token, then **~3.5–4.4 tok/s** |

72B and 80B: **no typing-speed log yet**. Expect slower than 27B (bigger file, one chat at a time, same 1 Gbit RPC).

Network (measured): the dual-GPU Linux box is **1 Gbit**; iperf3 from there to the Windows desktop was **~930–940 Mbit/s**. Ping to the NAS is **~0.25 ms**. 10G cards exist in a PC but were **not** the NIC actually carrying RPC. Until RPC rides a faster link, typing speed stays network-limited.

---

## alfa-train: fine-tune on this house’s own code

[alfa-train](https://github.com/Curt-Alfrey-s-Org/alfa-train) is configs + scripts for **QLoRA**. The brain never trains. Starting 14B QLoRA from the Windows GUI **stops chat first**, then trains.

Goals:

1. Learn from **this house’s git repos and operator notes**, not only a public dump.
2. Produce a **LoRA**, convert to GGUF, optionally attach on the NAS server (`--lora`) after a test score.
3. Leave the 27B / 72B / 80B **base files frozen** until that test passes.

Method: 4-bit base + LoRA ([bitsandbytes](https://huggingface.co/docs/bitsandbytes) + PEFT) via Axolotl. Their VRAM table: **13–14B QLoRA ~16–20 GB** — so **two 12 GB cards**, not one 8 GB card.

### What the GUI actually starts

| Knob | Value |
|------|--------|
| Base | [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) |
| Job | QLoRA SFT, JSONL of raw file text (`content`), sequence 2048, batch 1 |
| Parallel | torchrun / DeepSpeed ZeRO-2 |
| GPUs | the two 12 GB gaming cards |
| Image | Official Axolotl (loaded from the NAS tarball) |
| Graphs | **TensorBoard**. The Windows window is only start/stop/lock. |

### Two datasets people mix up

| Dataset | Built from | Used for |
|---------|------------|----------|
| **Repo JSONL** | Other git trees (bots, this brain, trainer docs, …) as plain text | Default 14B QLoRA |
| **Sleep pack** | Thumbs-up/down, error logs, nightly “lessons” | A second, operator-gated train path |

A 14B run that only ate the brain repo was the wrong diet. The code corpus is **multi-repo**. Sleep packs do not replace it.

### Loop: apps → notes → train → maybe LoRA on chat

1. People use the bots; thumbs and errors get logged in the brain.
2. A nightly job writes lesson drafts (still no training).
3. An export + secret scan lands a pack on the NAS.
4. I start alfa-train. If held-out eval is **~70%+**, I may convert the LoRA to GGUF and restart chat with `--lora` on the NAS only.

Production 27B/72B/80B files are not overwritten by a half-finished adapter.

---

## Ops rules that kept this from melting

1. One button starts chat, one button stops it. Never restart a single GPU worker while chat is up.
2. Gaming PCs join only at start, never mid-session.
3. Download once, on the NAS.
4. Same git commit on every PC before start.
5. Only the NAS server may speak to GPU RPC ports.
6. GPU workers do not auto-restart; a crash must look like a crash.
7. A real short completion must work before the proxy/website come up.
8. **Train or chat on the 5070s, never both.** The lock file enforces that.

---

## Good at / bad at

**Good:** every local app is a normal OpenAI client; 27B Q8 with ~40k tokens per chat slot; I can spend an evening on real 72B or an 80B-class coder without an 80 GB card; I can QLoRA on the same PCs that otherwise serve those apps; LoRA is gated, not silently swapped into production.

**Bad:** first-token wait; tokens/sec vs one 4090; two big chat models at once; training while the websites still need a GPU; keep-warm does not make Qwen3.8 cache like a plain transformer; mixing 8 GB cards into a 14B overnight job.

---

## If you copy the idea

1. Prove one dual-GPU PC + NAS CPU server, 27B Q8, one chat at a time, one real answer.
2. Add more GPUs only by stopping and starting the whole pool.
3. Keep a download hub. Workers never pull 20 GB from the internet.
4. Point each product repo at the proxy with env vars; keep their data in their repo.
5. Put training in a second repo with a hard lock against chat.
6. Train 14B on two matching 12 GB cards before you add 8 GB cards.
7. 72B and 80B are a **full stop/start to a different file**, not a second model sitting next to 27B.

Happy to answer architecture questions. This page will not grow env files, keys, or a LAN map.
