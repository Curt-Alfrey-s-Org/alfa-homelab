# alfa-homelab

Public write-up of a home **local-LLM cluster**: several NVIDIA GPUs acting as one model for websites and bots, plus a sibling trainer for QLoRA on that house’s own git repos.

**Read this:** [docs/HOMELAB_WRITEUP.md](docs/HOMELAB_WRITEUP.md)

That article is the copy meant for forums. It has no secrets, LAN addresses, or account names.

Operator runbooks and deploy scripts stay in a **private** repo. This repository is only the public description.

## What you will find

- How chat is split across a NAS CPU coordinator and GPU workers
- The three models that actually get loaded (27B Q8 daily, dense 72B, 80B MoE coder)
- How other product repos call the cluster (OpenAI-style API, one app’s data per chat)
- Measured 27B speed on this hardware
- How training is locked so it cannot run at the same time as chat
