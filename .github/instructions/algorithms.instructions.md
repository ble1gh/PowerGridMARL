---
description: "Use when editing HGTeam algorithm files. Covers cross-file dependencies between HGTeam.py, HGTeamHA.py, HGTeamSAC.py, hgteam_modules.py, and heterognn.py."
applyTo: "BenchMARL/benchmarl/algorithms/**"
---

# Algorithm File Dependencies

- **HGTeamBase** (in HGTeam.py) is the shared base for all variants. Changes here affect PPO, HAPPO, and SAC.
- **HGTeamHAPPO** (HGTeamHA.py) overrides `process_batch`, `_get_loss`, `_get_parameters`, and `train_groups`. It uses `HGTeamHAPPOLoss(HGTeamLoss)` — changes to `HGTeamLoss.forward()` propagate to HAPPO.
- **hgteam_modules.py** has shared components: `EmbeddingProcessor`, `HyperNetworkJoiner`, `reparameterize()`, `merge_embedding_losses()`. Used by all variants.
- **heterognn.py**: `HeteroGNN._forward()` is called during both collection and training. `_tensordict_to_hetero_data()` in HGTeam.py builds the graph. `_use_action_edge_features` flag gates critic action collection.
- The HAPPO variant supports three encoder update modes: `accumulated` (GNN in all 3 group optimizers at lr/3), `separate_forward` (GNN frozen during head updates, fresh forward+backward+step afterwards — D11), and `coop_encoder` (cooperative objective trains GNN — D12). Current default: `coop_encoder`.
- Design constraints D1–D12 in `docs/variable_agent_design_choices.md` must be preserved. Key: D6 (no cross-agent advantage normalization), D7 (zero inactive advantages), D8 (geometric-mean HAPPO factor), D12 (cooperative encoder objective).
