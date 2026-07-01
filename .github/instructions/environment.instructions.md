---
description: "Use when editing environment or reward code. Covers reward_transform() coupling with YAML config and CLI overrides."
applyTo: "PowerGridworld/**"
---

# Environment & Reward Dependencies

- `reward_transform()` in `multiagent_env.py` computes VPP, voltage, and EV rewards using penalty params from `evovernight13node_vpp.yaml`.
- VPP reward uses `vpp_reward_penalty` (quadratic) and `vpp_reward_linear_penalty` (linear), divided by `n_active_agents` (D2 per-capita normalization).
- `voltage_penalty` is applied locally when `cooperative_voltage=False` (current setting).
- EV component rewards (unserved_penalty, urgency) are divided by `reward_scale`, which is overridden by CLI `--reward-scale 100000`.
- YAML defaults can be overridden by CLI args in `run_hgteam_mlp.py`. Always check both before assuming a value.
