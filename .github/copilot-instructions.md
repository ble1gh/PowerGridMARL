# PowerGridMARL — Copilot Project Context

## Architecture

HGTeam (Heterogeneous Graph Team): MARL for power grid VPP control using GNNs.

- **HGTeamBase** (shared): GNN/embedding init, graph construction, edge filtering
- **HGTeam** (PPO variant): ClipPPOLoss, single optimizer
- **HGTeamHAPPO** (HAPPO variant): Sequential 3-group updates with importance-weighted factor propagation
- **HGTeamSAC** (SAC variant): Off-policy with Q-networks

All share `HGTeamBase` in `BenchMARL/benchmarl/algorithms/HGTeam.py`.

### Agent Groups (HAPPO)
| Group | Count | Max Padded | Notes |
|-------|-------|-----------|-------|
| EV | 10–20 | 20 | Random arrival/departure; demand response |
| PV | 25 | 25 | Fixed count; generation curtailment |
| Storage | 5–8 | 8 | Flexible charge/discharge |
| **Total** | 40–53 | **53** | Variable per episode; `active_mask` tracks live agents |

### Episode Structure
- 97 steps (48h with 30-min `control_timedelta=1800`)
- All agent slots max-padded; inactive agents masked in loss, advantages, and GNN edges

## Key Design Constraints (D1–D12)
Full details in `docs/variable_agent_design_choices.md`. Critical invariants:
- **D6**: Do NOT normalize advantages across the agent dimension — this erases inter-agent reward structure. Use per-slot normalization only (S8a).
- **D7**: Zero advantages for inactive agents (not just masked loss)
- **D8**: Geometric-mean HAPPO factor normalization
- **D2**: VPP reward divided by `n_active_agents` (per-capita)
- **D11**: `separate_forward` mode — GNN frozen during HAPPO head updates; fresh forward+backward from all groups at 1/n_groups scaling, single step
- **D12**: `coop_encoder` mode — Phase 0 pre-head shared-GNN update with per-agent, per-group advantages (S8a-normalized, D7-zeroed), followed by Phase 1 HAPPO head updates with GNN frozen. Cooperation is enforced by the shared encoder objective/optimizer step; no HAPPO factor in Phase 0.

## Reward Hierarchy (Current — April 2026)
Per-step per-agent at typical error magnitudes:
1. **VPP** (err_norm=1): ≈ −0.19 per agent ← dominant signal
2. **PV utilization** (100%): +0.10
3. **Voltage** (0.01pu violation): −0.10
4. **EV urgency**: ≈ −0.00025

Config in `evovernight13node_vpp.yaml`: `vpp_reward_penalty=1.0`, `vpp_reward_linear_penalty=5.0`, `voltage_penalty=10`.
CLI `--reward-scale 100000` divides EV component rewards.

## File Dependency Map
Changes in one file often require checking others:
- `HGTeam.py` ↔ `HGTeamHA.py`: HAPPO overrides `process_batch`, `_get_loss`, `_get_parameters`, `train_groups`. Both use `HGTeamBase`. Also aliases `get_critic`, `_get_shared_critic`, `_compute_other_actions_dim`, `_split_shared_gnn_param_groups`.
- `HGTeamHA.py` ↔ `HGTeam.py` `HGTeamLoss`: `HGTeamHAPPOLoss` extends `HGTeamLoss` (which extends `ClipPPOLoss`).
- `hgteam_modules.py`: Shared `EmbeddingProcessor`, `HyperNetworkJoiner`, `reparameterize()`, `merge_embedding_losses()`.
- `heterognn.py` ↔ `HGTeam.py`: `_tensordict_to_hetero_data()` builds graphs; `_use_action_edge_features` flag on critic GNN.
- `multiagent_env.py` ↔ `evovernight13node_vpp.yaml`: `reward_transform()` uses yaml penalty params. `reward_scale` is overridden by CLI.
- `run_hgteam_mlp.py`: CLI args override yaml defaults. Always check both before assuming a value.

## Training Config (Current HAPPO Run)
- `--lr 1e-4`, `--frames-per-batch 12288`, `--n-envs-per-worker 32`
- `--minibatch-size 1024`, `--n-minibatch-iters 4` → 48× sample reuse per group
- `--lmbda 0.95` (effective GAE horizon ≈19 steps)
- `--encoder-update-mode coop_encoder`, `--gnn-mode concat`
- `--critic-use-other-actions true`
- `--evaluation-episodes 20`, GPU: `h200_2g.35gb` (35 GB MIG)

## Build & Run
```bash
module load benchmarl/nightly
export PYTHONPATH=$PWD/PowerGridworld:$PWD/BenchMARL:$PYTHONPATH
sbatch EV_Charging_HGTeamHA_1node.sbatch
```
