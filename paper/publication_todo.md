# Publication Readiness Checklist — JMLR Submission

## 1. Experiments & Results (blocks everything in Sections VI–VIII)

### 1.1 PowerGridworld Domain
- [ ] Full HGTeam (HAPPO + coop_encoder) training run — convergence, final metrics
- [ ] Baseline runs: MAPPO, stock HAPPO, MAT (same environment interface)
- [ ] Ablation runs (Table 2 in outline):
  - [ ] −D2 (raw team reward, not per-capita)
  - [ ] −D3 (no edge filtering for inactive agents)
  - [ ] −D8 (product-mean or arithmetic-mean HAPPO factor)
  - [ ] −D10 (global advantage normalization)
  - [ ] −D12 (no coop_encoder — last-group-only encoder update)
  - [ ] −GNN (gnn_mode=none, MLP only)
- [ ] Variable-composition generalization (Figure 2 left): train on [10–20] EVs, eval on fixed compositions (10, 12, 15, 18, 20)
- [ ] Multiple seeds (≥3, ideally 5) for all runs

### 1.2 SMACv2 Domain
- [ ] Full HGTeam on Protoss 10v10 (primary)
- [ ] Full HGTeam on Protoss 20v20 (scaling test)
- [ ] Baseline runs: MAPPO, stock HAPPO, MAT, QMIX
- [ ] Ablation runs (same set as PowerGrid, both domains in every row of Table 2)
- [ ] Graph mode ablation (Table 3): ego_entity vs shared-graph vs flat-MLP actor
- [ ] Star vs full ego topology comparison
- [ ] Multiple seeds for all runs

### 1.3 Analysis Figures
- [ ] Figure 2: Variable-composition generalization (two panels, both domains)
- [ ] Figure 3: HAPPO factor magnitude distribution — geometric-mean vs product (two panels)
- [ ] Figure 4: Learning curves (mean return vs frames, both domains)
- [ ] Figure 5: Example rollouts (VPP 48h decomposition; entity attention weights)

---

## 2. Paper Writing (by section, keyed to outline.md)

### 2.1 Can Write Now (✅ in outline)
- [ ] Section I: Introduction (4 paragraphs framed in outline)
- [ ] Section II: Related Work (MARL, GNNs in MARL, Power Systems, Combat)
- [ ] Section III: Problem Formulation — Variable-Composition Dec-POMDP, Information Structure
- [ ] Section IV: Method — Architecture, Graph Construction, Credit Assignment, Coop Encoder
- [ ] Appendix A: Full D1–D12 + D-ego-1 through D-ego-5 catalog
- [ ] Appendix B: Environment details (OpenDSS feeder, reward equations, obs/action specs)

### 2.2 Draft Structure, Fill After Results (⚠️)
- [ ] Section V: Experimental Setup (tables with config values, baselines, evaluation protocol)
- [ ] Section VII: Discussion (structure written, arguments filled from results)

### 2.3 Wait for Results (❌)
- [ ] Section VI: Results — Tables 1–3, Figures 2–5, all narrative
- [ ] Section VIII: Conclusion
- [ ] Appendix C: Hyperparameter tables (final values from successful runs)
- [ ] Appendix D: Extended results (per-type breakdown, voltage profiles, win rate by composition)

### 2.4 Figures & Diagrams
- [ ] Figure 1: Architecture diagram (shared-graph vs ego-entity, GNN → policy heads)
- [ ] Table formatting for all results tables
- [ ] LaTeX template (JMLR style file: `jmlr.cls`)

---

## 3. Repository & Reproducibility

### 3.1 Critical: PowerGridworld Not In Git
- [ ] Add PowerGridworld/ to git tracking (currently 0 tracked files — it exists on disk but is ignored/untracked)
- [ ] Or: set up as a git submodule if you want to keep it as a separate repo
- [ ] Verify the PowerGridworld version on disk matches what was used for all experiments

### 3.2 Repository Hygiene
- [ ] Commit `.gitignore` (currently untracked at repo root)
- [ ] Add `.gitignore` rules for: `*slurmjob*`, `hgteamhappo_*/`, `*.bak`, `__pycache__/`, `debug_*.py`, `check_*.py`, `smoke_test_*.py`
- [ ] Remove `HGTeam.py.bak` and `debug_hypernetwork_gradients.py.bak` from working directory
- [ ] Decide what to do with the 7 untracked `hgteamhappo_*` experiment output directories (delete or move to scratch)
- [ ] Commit `paper/` directory (outline, second_domain.md, this checklist)
- [ ] Commit `docs/` if it contains the design decisions doc (`variable_agent_design_choices.md`)

### 3.3 Documentation
- [ ] Rewrite root `README.md` (currently 1 paragraph) to include:
  - Project description and contribution summary
  - Repository structure diagram (BenchMARL fork + PowerGridworld + paper)
  - Installation instructions (module load or pip install, PYTHONPATH setup)
  - How to reproduce main results (point to sbatch scripts or a reproduce script)
  - How to run a single training job (example command)
  - Citation (BibTeX)
- [ ] Add `CHANGES.md` documenting what was modified from upstream BenchMARL (fork diff summary)
- [ ] Add root `LICENSE` file (or clarify licensing: BenchMARL is MIT, PowerGridworld has its own license)

### 3.4 Reproducibility Artifacts
- [ ] `requirements.txt` or `environment.yml` at repo root with pinned versions (pytorch, torchrl, torch_geometric, etc.)
- [ ] Verify `requirements-benchmarl-3.10.3.txt` and `requirements-benchmarl-nightly.txt` are complete and correct
- [ ] Reproduction script: `scripts/reproduce_powergrid.sh` and `scripts/reproduce_smacv2.sh` (or a single script with arguments)
- [ ] Document any CHPC-specific setup (module loads, MIG GPU slicing) and provide generic alternatives
- [ ] Pin random seeds in reproduction scripts

---

## 4. Code Quality

- [ ] Remove dead code / unused imports in HGTeam.py, HGTeamHA.py, HGTeamSAC.py
- [ ] Verify all three encoder_update_modes still pass smoke tests after any further changes
- [ ] Add docstrings to key public methods (HGTeamBase, HGTeamHAPPO, loss classes) — JMLR reviewers may inspect code
- [ ] Ensure no hardcoded paths or credentials in tracked files
- [ ] Review `run_hgteam_mlp.py` CLI argument documentation (--help should be self-explanatory)

---

## 5. JMLR-Specific Requirements

- [ ] Paper formatted with `jmlr.cls` LaTeX style
- [ ] Abstract ≤ 200 words
- [ ] Paper length: JMLR has no hard limit but 25–40 pages (including appendix) is typical
- [ ] Code submission: JMLR requires code availability — link to GitHub repo or include as supplementary
- [ ] Reproducibility checklist (JMLR has an optional ML Reproducibility Checklist — consider filling it)
- [ ] Ensure all referenced results are reproducible from the provided code + instructions

---

## Priority Order

**Phase 1 — Unblock results** (do first):
1. Finalize SMACv2 integration (variable environments, ego-entity GNN)
2. Launch full training runs (PowerGrid + SMACv2, HGTeam + baselines)
3. Launch ablation runs

**Phase 2 — Write while runs are in progress**:
4. Write Sections I–IV (can write now)
5. Draft Section V experimental setup
6. Create Figure 1 architecture diagram
7. Repository cleanup (3.1–3.2)

**Phase 3 — After results land**:
8. Write Sections VI–VIII
9. Create all results figures and tables
10. Write Appendices C–D

**Phase 4 — Pre-submission polish**:
11. Rewrite README and add reproduction docs (3.3–3.4)
12. Code quality pass (4)
13. JMLR formatting and checklist (5)
14. Internal review / advisor feedback loop
