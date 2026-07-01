# Agent quickstart — HGTeam codebase

Concise facts for implementers. **Publication framing (contributions, domains, design ids)** → [`paper/outline.md`](../paper/outline.md).
Encoder math note → [`docs/hgt_vs_heteroconv_attention.md`](hgt_vs_heteroconv_attention.md).

## 1. What the code implements

**HGTeam**: heterogeneous MARL with variable team size, shared GNN encoder, and BenchMARL training. Variants: **`HGTeam`** (policy-gradient / PPO-style loss), **`HGTeamHA`** (HAPPO), **`HGTeamSAC`** (SAC). Shared base: **`HGTeam.py`** (`HGTeamBase`).

**Domains in repo**: **VPP / feeder** (`PowerGridworld` + `PowerGridworldVariable` / `PowerGridworldGraph`); **SMACv2** (`benchmarl/environments/smacv2_variable/`, `smacv2/`).

## 2. File map (edit here first)

| Concern | Path |
|--------|------|
| Shared HGTeam logic, loss, batching | `BenchMARL/benchmarl/algorithms/HGTeam.py` |
| HAPPO order, factors, encoder phases | `BenchMARL/benchmarl/algorithms/HGTeamHA.py` |
| SAC variant | `BenchMARL/benchmarl/algorithms/HGTeamSAC.py` |
| Embeddings, hypernet, SNI hooks | `BenchMARL/benchmarl/algorithms/hgteam_modules.py` |
| Stable hetero GNN, `active_mask` on edges | `BenchMARL/benchmarl/models/heterognn.py` |
| Opt-in low-rank edge-conditioned HGT | `BenchMARL/benchmarl/models/edgeweightedHGT.py` |
| Variable grid wrapper + specs | `BenchMARL/benchmarl/environments/PowerGridworldVariable/common.py` |
| Graph + padding wrapper | `BenchMARL/benchmarl/environments/PowerGridworldGraph/common.py` |
| Rewards, `reward_transform`, penalties | `PowerGridworld/gridworld/multiagent_env.py` |
| Scenario instance | `BenchMARL/.../PowerGridworldVariable/evovernight13node_*.py` |
| HAPPO YAML defaults | `BenchMARL/benchmarl/conf/algorithm/hgteamhappo.yaml` |
| CLI overrides | `run_hgteam_mlp.py`, `run_hgteam_experiment.py` |
| SLURM template (often overrides YAML) | `EV_Charging_HGTeamHA_1node.sbatch` |

## 3. Shapes (VPP scenario)

Typical padded roster (see outline §III.C): **EV 10–20** (pad 20), **PV 25**, **storage 5–8** (pad 8) → **53** slots. Batches are often `(n_envs, T, n_agents, …)` but **agent dim is not guaranteed at a fixed index** — write dim‑aware code.

## 4. Invariants (do not break)

Aligns with **`paper/outline.md`** (size-invariant credit, masking, HAPPO factor).

- **D2**: VPP / shared cooperative reward terms **per active agent** (per‑capita), not scaled by padded roster alone.
- **D6–D8, S8a**: no global advantage norm across all agents; **inactive slots zeroed in advantages** (not loss-only mask); HAPPO factor uses **geometric‑mean style** normalization; **per‑slot active‑only** norm where documented in code.
- **`active_mask`**: required for masks / GNN **edges**; **not** a concatenated model input feature; exclude from feature pipelines.

## 5. `encoder_update_mode` (short)

YAML default: **`separate_forward`** (`hgteamhappo.yaml`). Modes:

- **`accumulated`**: GNN grads accumulate across group head updates; one encoder step after heads.
- **`separate_forward`**: freeze GNN → HAPPO heads → unfreeze → encoder PPO‑style step on factored advantages.
- **`coop_encoder`**: Phase 0 `_coop_encoder_update()` (per‑agent, per‑group adv, **no** HAPPO factor); Phase 1 sequential HAPPO with GNN frozen; **re‑eval old logprobs** after Phase 0 GNN change.

## 6. SNI (ratio stability)

**`EmbeddingProcessor.deterministic_mode()`**: eval / loss uses **μ not sampled z** to avoid ratio noise. **Do not reuse exhausted context managers** — construct fresh per use.

## 7. Config vs jobs

**YAML ≠ what SLURM runs.** Inspect **both** `hgteamhappo.yaml` and the **`sbatch` / CLI** before assuming defaults (`encoder_update_mode`, `use_beta`, batch sizes, etc.).

## 8. Commands (typical CHPC)

```bash
module load benchmarl/nightly   # or your module
export PYTHONPATH=$PWD/PowerGridworld:$PWD/BenchMARL:$PYTHONPATH
sbatch EV_Charging_HGTeamHA_1node.sbatch
```

## 9. After edits

Run narrow **`pytest`** / **`smoke_test_*.py`**. Re‑check: device‑safe masking, inactive agents (loss, adv, edges, logging), ratio metrics if touching PPO path. For `EdgeWeightedHGT` changes, run `python smoke_test_edgeweightedhgt.py` plus the relevant opt-in pipeline smoke such as `python smoke_test_vpp_pipeline.py --critic-model edgeweightedhgt --tier synthetic`.

If behavior surprises: first suspect **YAML vs CLI override**, then **encoder phase ordering** under `coop_encoder`.
