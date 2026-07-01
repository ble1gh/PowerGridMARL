# AGENTS — PowerGridMARL

Orientation for humans and agents. **HGTeam / HAPPO / GNN invariants** → [`docs/agent_quickstart.md`](docs/agent_quickstart.md). Variable-agent design notes → [`docs/variable_agent_design_choices.md`](docs/variable_agent_design_choices.md). **Paper framing (contributions, domains, notation)** → [`paper/outline.md`](paper/outline.md).

## What this is

Heterogeneous MARL (**HGTeam** family) on **variable-composition** teams: primary domain is **distribution-grid VPP** (`PowerGridworld` + BenchMARL `PowerGridworldVariable` / `PowerGridworldGraph`); **SMACv2** variants live under `BenchMARL/benchmarl/environments/smacv2*`. Code is an **in-tree BenchMARL fork** (`BenchMARL/`) plus the **`gridworld`** package (`PowerGridworld/`). Stack: TorchRL / TensorDict, PyTorch, optional PyG.

**OpenDSS**: [`OpenDSSSolver`](PowerGridworld/gridworld/distribution_system/opendss.py) calls **`import opendssdirect as dss`**. [`PowerGridworld/pyproject.toml`](PowerGridworld/pyproject.toml) declares **`opendssdirect-py`** for a minimal grid-only install. Full MARL stacks use root **`requirements-benchmarl-*.txt`**, which pins **`dss-python`** (+ **`dss-python-backend`**) (DSS-Extensions engine). Those stacks may or may not install **`opendssdirect`** as a separate import name—**verify** `python -c "import opendssdirect"` in your env; if it fails, install **`opendssdirect-py`** alongside the locked requirements so solver code matches deps.

## Where things live

| Area | Path | Notes |
|------|------|--------|
| Algorithms | `BenchMARL/benchmarl/algorithms/` | `HGTeam*.py`, `hgteam_modules.py` |
| Models | `BenchMARL/benchmarl/models/` | `heterognn.py`, `transformer.py`, … |
| Grid ↔ TorchRL | `BenchMARL/.../PowerGridworldVariable/`, `.../PowerGridworldGraph/` | `VariableAgentMultiAgentEnv`, `PaddedMultiAgentEnv`, `active_mask` |
| Physics / reward | `PowerGridworld/gridworld/` | `MultiAgentEnv`, DER agents, OpenDSS |
| SMACv2 | `BenchMARL/benchmarl/environments/smacv2_variable/` (and `smacv2/`) | Second benchmark domain |
| Runners / smoke | repo root | `run_hgteam_*.py`, `run_mat_experiment.py`, `smoke_test_*.py` |
| HPC | repo root | `*.sbatch`, `launch_*.sh` |
| Lint | root `pyproject.toml` | Ruff `target-version = py310`; excludes some BenchMARL/PowerGridworld paths |

## Runbook

1. **Python**: **3.10** for `requirements-benchmarl-*` / Ruff; `PowerGridworld` alone allows `3.8–3.10`.
2. **Install**: matching `requirements-benchmarl-*.txt` (includes editable `BenchMARL`).
3. **Cwd / path**: run from **repo root**; scripts usually `sys.path`‑prepend `BenchMARL` and often `PowerGridworld` before `benchmarl` / `gridworld` imports (see nearest `run_*.py` or `smoke_test_*.py`).
4. **Checks**: `pytest test_bus_connectivity.py -q` (OpenDSS wiring); targeted `smoke_test_*.py` or short `run_hgteam_*` with low `--max-n-frames` for integration; `ruff check .` / `ruff format` from root.

## Reading order

- **Grid env / reward** → `PowerGridworld/gridworld/multiagent_env.py` → `PowerGridworldVariable` or `PowerGridworldGraph` `common.py` → scenario `evovernight13node_*.py`.
- **HGTeam / HAPPO / GNN** → `docs/agent_quickstart.md` → `HGTeam.py` → `HGTeamHA.py` or `HGTeamSAC.py` → `models/heterognn.py` + `algorithms/hgteam_modules.py`.
- **Defaults vs jobs** → `BenchMARL/benchmarl/conf/algorithm/hgteamhappo.yaml` **and** CLI / `EV_Charging_HGTeamHA_1node.sbatch` (they diverge).

## Cursor usage

First message: goal, acceptance criteria, 3–8 `@` files or one folder; `@AGENTS.md` once per thread. Prefer `.cursor/rules/` over pasting long theory. New thread when the task topic changes.
