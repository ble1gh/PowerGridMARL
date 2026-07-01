---
description: "Use when editing YAML config files. Covers how config values interact with CLI overrides and environment code."
applyTo: "BenchMARL/benchmarl/conf/**"
---

# Config File Dependencies

- Algorithm YAML defaults are loaded by `get_from_yaml()` in `run_hgteam_mlp.py`, then overridden by CLI arguments.
- `evovernight13node_vpp.yaml` reward params (`vpp_reward_penalty`, `voltage_penalty`, etc.) are used directly in `multiagent_env.py:reward_transform()`.
- `reward_scale` in the YAML (default=1) is overridden by CLI `--reward-scale` (currently 100000) — this divides EV component rewards only.
- `lmbda` and `encoder_update_mode` in algorithm YAMLs can be overridden by CLI `--lmbda` and `--encoder-update-mode`.
- HAPPO yaml uses `critic_coef: 2.0` + `smooth_l1`; PPO yaml uses `critic_coef: 1.0` + `l2`. These are intentionally different.
