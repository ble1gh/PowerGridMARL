# SMACv2 as Second Domain: Detailed Plan

## Purpose

This document describes how SMACv2 serves as a second experimental domain for the JMLR paper, what it demonstrates for each of the three contributions, and how to implement it.

The critical question for JMLR reviewers is: *"Is HGTeam a general variable-composition MARL framework, or a power-grid-specific system?"* SMACv2 answers this by demonstrating all three contributions in a completely different domain — tactical combat with heterogeneous unit types and agent death.

---

## Why SMACv2 Specifically

### Natural Alignment with the Three Contributions

| Property | PowerGridworld | SMACv2 | Why Both Matter |
|----------|---------------|--------|-----------------|
| **Variable composition** | EV arrival/departure (10→20 per episode) | Unit death in combat (20→0 over episode) | Two distinct mechanisms: stochastic arrival vs. attrition |
| **Heterogeneous types** | EV/PV/Storage (3 types, different obs/action spaces) | Stalker/Zealot/Colossus (3 types, different combat roles) | Both have asymmetric capabilities per type |
| **Reward structure** | Mixed-motive: cooperative VPP tracking + local objectives | Common team reward; role-specific survival/positioning pressures emerge through dynamics | PowerGridworld tests reward decomposition; SMACv2 cleanly isolates variable-$N$ common-reward attrition |
| **Topology** | Physical grid bus connections | Spatial proximity on the battlefield | Both have structure the GNN can exploit |

### Contrasts That Strengthen the Paper

| Dimension | PowerGridworld | SMACv2 |
|-----------|---------------|--------|
| Composition change | Gradual arrival/departure | Permanent death (irreversible) |
| Agent count trajectory | Increases (EVs arriving) | Monotonically decreases (units dying) |
| Action space | Continuous [0,1] | Discrete (move/attack/stop) |
| Episode length | 97 steps (48h) | 50–200 steps (variable, battle-dependent) |
| Reward density | Dense per-step (VPP error) | Sparse-ish (damage + win bonus) |
| Cooperation structure | Explicit setpoint tracking | Implicit tactical coordination |

These contrasts preempt the reviewer objection *"your design decisions might only work for dense-reward continuous-action settings."*

---

## How Each Contribution Is Demonstrated

### Contribution 1: Variable-Composition Framework (D1–D7, D10)

**PowerGridworld demonstration**: Agents arrive and depart stochastically. Between episodes, team composition varies (10–20 EVs, 5–8 Storage). Within episodes, EVs connect and disconnect randomly.

**SMACv2 demonstration**: Team starts at full strength (e.g., 9 Stalkers + 9 Zealots + 2 Colossi = 20). Units die during combat, reducing the active count. Different battles have different casualty patterns — sometimes Zealots (melee) die early while Stalkers (ranged) survive; other times the reverse.

**What SMACv2 adds that PowerGridworld alone cannot show**:
- **Within-episode monotonic decrease**: The framework must handle a team that shrinks to near-zero, not just fluctuates around a mean. This stress-tests D3 (edge filtering) and D6 (masked loss reduction) at extreme sparsity — when only 2–3 of 20 agents remain alive.
- **Irreversible composition change**: Dead units don't return. The GNN must learn representations that degrade gracefully as the graph loses nodes permanently.
- **Different masking dynamics**: In PowerGridworld, an inactive EV slot might become active later (arrival). In SMACv2, once an agent's mask flips to inactive, it stays inactive. This tests whether the framework's assumptions hold for both reversible and irreversible composition changes.

**Key ablation for C1 on SMACv2**: Remove D6+D7+D10 (use standard global advantage normalization, don't zero inactive advantages). Expected failure mode: as units die mid-episode, surviving agents receive inflated gradient signal from the growing fraction of zero-padded slots. This effect should be more dramatic in SMACv2 than PowerGridworld because the active fraction drops much lower (e.g., 3/20 = 15% alive vs. PowerGridworld's typical 40/53 = 75%).

### Contribution 2: Geometric-Mean HAPPO Factor (D8)

**PowerGridworld demonstration**: Group sizes vary between episodes (10–20 EVs vs. 5–8 Storage). The geometric mean normalizes factor variance across episodes with different team compositions.

**SMACv2 demonstration**: Group sizes vary within a single episode as units die. A Zealot group that starts at 9 agents might be at 3 by mid-episode. The factor F_g = ∏ r_i would have variance proportional to 9 early in training but proportional to 3 late in an episode. With geometric mean, F_g = (∏ r_i)^{1/n}, the variance is O(1/n) regardless.

**What SMACv2 adds**:
- **Intra-episode variance change**: The same group has different sizes at different timesteps within a single trajectory. This is a harder test than PowerGridworld, where group sizes are fixed within an episode (only varying across episodes).
- **Correlated composition and advantage magnitude**: Units die when losing a fight, which correlates with negative advantage values. Without D8, the factor for a dying group shrinks in dimensionality at exactly the moments when advantages are most negative — creating a systematic bias. The geometric mean breaks this correlation.

**Key ablation for C2 on SMACv2**: Compare geometric mean vs. product (original HAPPO) vs. arithmetic mean. Plot the factor magnitude distribution as a function of group size (number of alive units). Expected result: product factor has exponentially wider spread for larger groups; geometric mean is constant.

### Contribution 3: Two-Phase Cooperative Encoder Training (D11/D12)

**PowerGridworld demonstration**: Three agent types with mixed cooperative (VPP) + individual (voltage, EV urgency) objectives. The coop_encoder mode trains the shared GNN with a per-agent cooperative PPO surrogate (Formulation B: per-agent advantages and ratios aggregated over active agents) while heads retain per-group HAPPO updates.

**SMACv2 demonstration**: Three unit types (Stalker/Zealot/Colossus) with a shared team reward (battle outcome + damage shaping). The environment still creates different tactical pressures by role: melee Zealots must advance, ranged Stalkers often kite, and Colossi need protection. These are dynamics-induced specialisation pressures, not separate individual reward terms. The encoder must learn a representation that supports all three roles under a shrinking active roster.

**What SMACv2 adds**:
- **Role pressure is implicit, not reward-engineered**: In PowerGridworld, cooperative and individual signals are explicit reward components. In SMACv2, all active agents receive the same per-capita team reward in the wrapper; survival matters because alive agents continue acting and accumulating future team reward, not because there is a private survival reward. This tests whether coop_encoder remains useful in the common-reward limit where the mixed-motive axis is removed.
- **Asymmetric role importance**: Colossi are rare (10% weight) but disproportionately impactful (area damage, high HP). The balanced gradient accumulation in D11 must prevent the numerically dominant Stalker/Zealot groups from overwhelming the Colossus group's encoder gradient. SMACv2 makes this asymmetry more extreme than PowerGridworld (where PV is 25 agents vs. Storage's 5–8, but PV's individual contribution is smaller).
- **Tighter coupling between types**: In combat, a Zealot's optimal action depends on where Stalkers are providing covering fire. In PowerGridworld, agent coupling is mediated by the grid (voltage constraints). SMACv2's direct spatial coupling tests whether the GNN encoder learns to condition each type's representation on the other types' states.

**Key ablation for C3 on SMACv2**: Compare coop_encoder vs. separate_forward (D11 only) vs. last-group-only (vanilla HAPPO). Expected result: coop_encoder learns coordinated tactics (Zealots engage while Stalkers kite) faster than separate_forward, which in turn outperforms last-group-only.

---

## Experimental Design for SMACv2

### Configuration

Use **Protoss 10v10** as the primary battle:
- 3 unit types: Stalker (45%), Zealot (45%), Colossus (10%)
- Expected composition: ~4–5 Stalkers, 4–5 Zealots, 0–1 Colossus per episode
- Max pad: 5 Stalkers + 5 Zealots + 1 Colossus = 11 max agent slots (much smaller than PowerGridworld's 53, which also demonstrates scale-independence)

Use **Protoss 20v20** as the scaling test:
- ~9 Stalkers, 9 Zealots, 2 Colossi = 20 max agent slots
- Tests the same framework at double the agent count

### Metrics

| Metric | What It Shows |
|--------|---------------|
| **Win rate** | Primary performance metric (analogous to VPP tracking error) |
| **Mean episode return** | Learning efficiency |
| **Return vs. alive count** | Does policy quality degrade gracefully as units die? C1 contribution. |
| **Factor magnitude histogram** | D8 contribution — factor variance vs. group size |
| **Win rate by composition** | D12 contribution — performance when Colossus-heavy vs. Zealot-heavy draws |

### Baselines (SMACv2 only)

| Baseline | What It Tests |
|----------|---------------|
| **MAPPO (single group)** | Standard homogeneous MARL — no per-type policy, no GNN, no D8 |
| **HAPPO (per-type groups, no D1–D8)** | Stock HAPPO with dead agents handled naively (zero rewards, global advantage norm) |
| **HGTeam without GNN** | gnn_mode=none — tests whether the GNN (C1/C3) actually helps vs. MLP-only |
| **QMIX** | Value decomposition baseline (standard SMAC benchmark algorithm) |

MAPPO and QMIX have published SMACv2 results in the literature, so you can reference those for validation without necessarily re-running them.

---

## Implementation Strategy

### Foundational Design Principle: Common Knowledge vs. Private Knowledge

The two domains have fundamentally different partial-observability structures, and the GNN construction adapts accordingly:

| | PowerGrid | SMACv2 |
|---|---|---|
| **Common knowledge** | Grid topology, team composition (who's active, types, participation scores) — slow-changing, broadcast-able | None — no shared battlefield state |
| **Private knowledge** | Local bus voltage, power injection, SOC — fast-changing, only locally relevant | Each agent's sight-range-limited observation of nearby allies and enemies |
| **GNN encodes** | Common knowledge (shared graph, one forward pass for all agents) | Private knowledge (per-agent ego-centric entity graph) |
| **Policy head receives** | GNN embedding + private observation (concatenated) | GNN self-node embedding + move features (concatenated) |
| **`exclude_observations_from_node_features`** | `True` — observations bypass GNN | `False` — decomposed observations ARE the GNN input |
| **Dec-exec story** | Broadcast GNN inputs (topology + composition change slowly) or broadcast embeddings | Each agent runs GNN locally on its own private observation |
| **Critic** | Centralized (full information access) | Centralized (full information access) |

This adapts the GNN to each domain's information structure while keeping the variable-composition framework (D1–D7, D10), credit assignment (D8), and encoder training (D11/D12) completely identical.

### Architecture: What to Build

```
BenchMARL/benchmarl/environments/smacv2_variable/
├── __init__.py          # (already exists)
├── common.py            # VariableSmacv2Env + observation decomposition

BenchMARL/benchmarl/conf/task/smacv2_variable/
├── protoss_10_vs_10.yaml   # (already exists)
├── protoss_10_vs_11.yaml   # (already exists)
├── protoss_20_vs_20.yaml   # (already exists)
└── protoss_20_vs_23.yaml   # (already exists)
```

Changes span: `common.py` (entity decomposition), `HGTeam.py` (ego-entity GNN construction, actor pipeline), `heterognn.py` (ego-entity readout), algorithm YAMLs (`actor_graph_mode` flag), task YAMLs (`entity_obs` flag).

### What common.py Must Do

The wrapper sits **between** TorchRL's SMACv2Env and BenchMARL's experiment loop. It transforms the standard SMACv2 interface (single "agents" group) into the HGTeam interface (per-type groups, active_mask, entity-decomposed observations).

#### 1. Override group_map: Split by Unit Type (already implemented)

Agents are assigned to per-type groups (stalker, zealot, colossus) at episode reset, with max-padded slots and active_mask tracking live agents. This is complete.

#### 2. Derive active_mask from Action Mask (already implemented)

A unit is alive if it has at least one non-noop valid action. Exposed as per-type and flat active_masks. This is complete.

#### 3. Observation Entity Decomposition (NEW)

Each agent's flat 182-dim observation (Protoss 10v10 config) is decomposed into structured entity features:

| Section | Per-entity dim | Count | Total | Slice |
|---------|-------|-------|-------|-------|
| Move feats | — | — | 4 | `[0:4]` |
| Enemy feats | 9 | 10 | 90 | `[4:94]` reshaped to `(10, 9)` |
| Ally feats | 9 | 9 | 81 | `[94:175]` reshaped to `(9, 9)` |
| Own feats | — | — | 7 | `[175:182]` |

Enemy features per entity (9-dim): `[shootable, distance, rel_x, rel_y, health, shield, type_bit_0, type_bit_1, type_bit_2]`
Ally features per entity (9-dim): `[visible, distance, rel_x, rel_y, health, shield, type_bit_0, type_bit_1, type_bit_2]`
Own features (7-dim): `[health, shield, pos_x, pos_y, type_bit_0, type_bit_1, type_bit_2]`
Move features (4-dim): `[can_north, can_south, can_east, can_west]`

Out-of-sight entities have all-zero features — the sight-range constraint is respected by construction.

Output keys per type group (e.g., stalker):
- `(stalker, "entity_enemy")`: shape `(max_stalkers, n_enemies, 9)`
- `(stalker, "entity_ally")`: shape `(max_stalkers, n_allies, 9)`
- `(stalker, "entity_self")`: shape `(max_stalkers, 7)`
- `(stalker, "move_feats")`: shape `(max_stalkers, 4)`

Gated by `entity_obs: True` in the YAML config.

#### 4. No Proximity Graph — Ego-Entity Graph Instead (REVISED)

~~The earlier plan used a shared proximity graph with `proximity_threshold=6.0` for agent-agent edges.~~

**New approach**: The GNN processes per-agent ego-centric entity graphs, not a shared agent-agent graph. Each agent's graph has nodes for every entity in its observation (self, allies, enemies). Connectivity within the ego graph is determined by the chosen topology (star, full, or bipartite — see Design Decisions section).

No explicit position-based proximity graph is needed because the entity features already encode relative distance (normalized by sight range). Entities beyond sight range are all-zeros and are filtered from the ego graph via entity-level active masking.

#### 5. Participation Scores (REVISED)

In the shared-graph mode, participation scores served as the GNN's node features (encoding "who is active and how much"). In ego-entity mode, the entity features replace this role — each entity node's features directly encode its state (health, shield, type, relative position).

The `participation_score` key is retained for compatibility with the framework (D3, D5, D7 use it for masking), but set to binary alive/dead:
```python
participation = active_mask.float().unsqueeze(-1)  # (n_agents, 1)
```

#### 6. Reward Handling (unchanged)

Per-capita broadcast of team reward to alive agents (D2). Already implemented.

### What Changes in the Algorithm Layer

#### `actor_graph_mode` Flag

Add `actor_graph_mode: str = "shared"` to `HGTeamBase.__init__()`. Values:
- `"shared"`: Current behavior (PowerGrid). Shared graph, participation_score node features, observations concatenated after GNN.
- `"ego_entity"`: Per-agent ego-centric entity graph from decomposed observations. Observations ARE the GNN input.

Set via CLI `--actor-graph-mode ego_entity`.

#### Ego-Entity GNN Construction

When `actor_graph_mode == "ego_entity"`, `_get_or_build_shared_actor_gnn()` builds:

**Node types**: `{self_entity, enemy, stalker_ally, zealot_ally, colossus_ally}`
- `self_entity`: 1 node per agent, 7 features
- `enemy`: n_enemies nodes per agent, 9 features
- `stalker_ally`, `zealot_ally`, `colossus_ally`: per-type ally nodes, 9 features each

Per-type ally node types give TransformerConv separate learned projection matrices per ally type, enabling type-specific attention patterns (e.g., attend differently to melee Zealots vs. ranged Stalkers). The env wrapper parses the 3 type bits in ally features to route each ally to its corresponding node type. Out-of-range allies (all-zero features) are filtered via entity active masking (D-ego-4).

**Edge types** (default: star topology):
- `(enemy, "observed_by", self_entity)` — enemies report to self
- `(self_entity, "observes", enemy)` — self attends to enemies
- `(stalker_ally, "observed_by", self_entity)` — stalker allies report to self
- `(self_entity, "observes", stalker_ally)` — self attends to stalker allies
- (analogous edges for zealot_ally, colossus_ally)

Star topology makes the self-node a hub that aggregates all entity information. This is the closest analogy to the power grid's bus node aggregating connected agent information.

**Batching**: Fold B_env × N_agents ego graphs into PyG's batch dimension. One forward pass processes all per-agent graphs. Extract self_entity embeddings → reshape to (B, N_total, embed_dim) → split into per-type groups.

**GNN weights are shared** across all agents and all per-agent graphs (same TransformerConv parameters). This is identical to how the shared GNN has shared weights — only the input data structure differs.

#### Actor Pipeline

`gnn_mode="concat"` flow with ego_entity:
```
EgoEntityGNN(entity features) → self-node embedding (embed_dim)
EmbeddingProcessor → embedding_z
Concat [embedding_z ‖ move_feats] → input (embed_dim + 4)
MLP → logits
```

Raw observation is NOT concatenated (it's already consumed by the entity GNN). Only the 4-dim move features (which don't fit the entity graph structure) are appended.

#### HAPPO / coop_encoder Compatibility

No structural changes. `self._shared_actor_gnn` points to the ego-entity GNN. All references in `HGTeamHA.train_groups()` work unchanged:
- Phase 1 (cooperative encoder): Unfreeze ego GNN, cooperative PPO objective using per-agent advantages and per-agent ratios aggregated over active agents, step GNN optimizer
- Phase 2 (HAPPO heads): Freeze ego GNN, update per-group MLP heads with HAPPO factor. Heads train against freshly updated embeddings from the GNN step.
- `_split_shared_gnn_param_groups`: GNN params get scaled LR — still applies

### What Does NOT Need to Change

- **HGTeamHA.py**: HAPPO factor propagation operates on per-group log-probs. Agnostic to embedding source.
- **hgteam_modules.py**: EmbeddingProcessor, HyperNetworkJoiner unchanged.
- **Critic architecture**: Centralized critic retains full information access (shared graph or MLP).
- **D1–D12**: All variable-composition design decisions apply identically.

### Configuration

```yaml
# smacv2_variable/protoss_10_vs_10.yaml additions:
entity_obs: True                    # Enable observation decomposition
# proximity_threshold: 6.0  (deprecated — ego-entity replaces proximity graph)
```

```yaml
# conf/algorithm/hgteamha.yaml additions:
actor_graph_mode: "shared"          # default for PowerGrid
# CLI: --actor-graph-mode ego_entity  (for SMACv2)
```

---

## What Reviewers Will Ask (and Answers)

### "You have two different graph architectures. Is this really one method?"

The contributions (C1: variable-composition with D1–D12, C2: geometric-mean HAPPO factor, C3: cooperative encoder training) are completely unchanged between shared-graph and ego-entity modes. The same HeteroGNN module, same TransformerConv layers, same training algorithm. What adapts is the **input representation** — the graph construction layer maps each domain's natural information structure into the GNN. This is analogous to how a CNN can process natural images or spectrograms: the architecture and training are the same, the input modality differs. The `actor_graph_mode` flag is a single configuration choice, not a separate system.

The ablation running shared-graph mode on SMACv2 (see below) demonstrates that both modes share the same code path and produce comparable results.

### "The common-knowledge / private-knowledge distinction is ad-hoc."

The distinction is grounded in Dec-POMDP information structure. In the grid domain, topology and composition are part of the *common prior* — all agents can access them (they're properties of the infrastructure, not individual percepts). In SMACv2, each agent's observation is *private* by definition (sight-range-limited). The GNN construction follows directly: common knowledge → shared graph; private knowledge → per-agent ego-graph. This is not a post-hoc rationalization; it's the formal definition of partial observability in each domain.

### "SMACv2 is discrete-action. Doesn't HGTeam assume continuous actions?"

HGTeam's contributions (D1–D12) are action-space-agnostic. The variable-composition framework, geometric-mean factor, and cooperative encoder training apply identically to discrete and continuous policies. The PPO/HAPPO loss function handles both. PowerGridworld demonstrates continuous; SMACv2 demonstrates discrete. This strengthens the generality claim.

### "For SMACv2, what does the entity GNN add over a flat MLP?"

Each agent's flat observation already encodes what it can see. The entity decomposition + GNN adds **relational inductive bias**: it learns entity-level attention (e.g., prioritize low-health enemies, attend to nearby allies) rather than memorizing position-dependent feature slots in a flat vector. This is the same advantage entity transformers (UPDeT, REFIL) have demonstrated. The key ablation is ego_entity GNN vs. flat MLP (no GNN): both respect sight range, but the GNN imposes entity structure.

### "Your ego-entity GNN with star topology is just an entity transformer."

Yes — with star topology, the ego-entity GNN with TransformerConv reduces to entity-level cross-attention. This is by design. The advantage of the GNN formulation: (a) heterogeneous edge types provide separate learned projections for ally vs. enemy relationships, (b) non-star topologies (ablation) enable entity-entity reasoning, and (c) the same module handles both domain modes. We intentionally use this rather than a bespoke entity transformer to maintain framework unity.

### "How does D4 (actions as edge features) work with discrete actions?"

D4 can be omitted for SMACv2 experiments. In the ego-entity GNN, there are no inter-agent edges on the actor side — only entity edges within each agent's ego-graph. D4 applies only to the centralized critic (if the critic uses a shared graph). The paper can state: *"D4 is demonstrated on PowerGridworld (continuous actions, critic graph); SMACv2 experiments use an observation-based critic."*

### "QMIX already handles SMACv2 with dead agents. What's new?"

QMIX handles death implicitly — dead agents' Q-values are set to 0 in the mixer. But QMIX (a) assumes homogeneous agents (single network for all types), (b) uses value decomposition rather than policy gradient (no factor normalization needed), and (c) has no mechanism for balancing encoder gradients across types. HGTeam's C1–C3 address problems that don't exist in QMIX's framework but do exist in policy gradient methods (HAPPO/PPO) applied to heterogeneous teams.

### "You never demonstrate decentralized execution."

Fair critique, standard for MARL methods papers. However, the ego-entity mode is immediately deployable for decentralized execution — each agent only needs its own observation and the shared GNN weights (no inter-agent communication required). For the grid domain, agents would need broadcast of composition/topology information (which changes infrequently) to compute embeddings locally. We acknowledge this as a limitation and note that the ego-entity architecture is deployment-ready by construction.

---

## Paper Integration

### Where SMACv2 Results Appear

**Section IV (Method)**: New subsection A′ describes how the GNN adapts to each domain's information structure (common knowledge vs. private knowledge). Two-panel architecture figure shows both modes.

**Section V (Experimental Setup)**: One subsection describes PowerGridworld VPP control; one subsection describes SMACv2 combat. Both share the same method (Section IV).

**Section VI (Results)**:
- **Table 1**: Main comparison — two panels side by side. Left: PowerGridworld (VPP error, voltage violations). Right: SMACv2 (win rate, episode return).
- **Table 2**: Ablation — both domains in every row. Makes clear that each design decision matters in both settings.
- **Table 3** (new): Graph mode ablation — shared vs. ego_entity on SMACv2, ego_entity vs. shared on PowerGrid (if feasible). Shows that the mode choice matters and that the principled choice (matching the domain's information structure) wins.
- **Figure (composition generalization)**: PowerGridworld shows performance vs. EV count. SMACv2 shows performance vs. percentage of units surviving at episode midpoint.
- **Figure (D8 factor analysis)**: Factor magnitude distribution at different group sizes. PowerGridworld provides across-episode variation; SMACv2 provides within-episode variation. Both on the same figure, different panels.

### What SMACv2 Adds to Each Section

| Section | Without SMACv2 | With SMACv2 |
|---------|---------------|-------------|
| **Abstract** | "...demonstrated on power grid VPP control" | "...demonstrated on VPP control and tactical combat" |
| **Contribution bullets** | "We present a framework..." | "We present a *general* framework validated across *two distinct domains* with contrasting partial-observability structures..." |
| **Method** | Method described, seems domain-specific | Method described with principled domain adaptation: common-knowledge graph (grid) vs. private-knowledge ego-graph (combat) |
| **Experiments** | Thorough but single-domain | Two-domain validation with contrasting properties |
| **Discussion** | "Limitations: single domain" | "Our framework adapts to both shared and private information structures, continuous and discrete action spaces, dense and sparse rewards, and both arrival and attrition composition dynamics" |

### Narrative Flow

The paper intro motivates variable-composition MARL with two examples:
1. *"In power grids, distributed energy resources connect and disconnect dynamically..."*
2. *"In tactical combat, team members are lost to attrition, requiring the surviving team to adapt..."*

Section IV introduces the common-knowledge / private-knowledge distinction as the principled basis for the GNN's domain adaptation. Both examples reappear in experiments, closing the narrative loop.

---

## Design Decisions for SMACv2 Ego-Entity GNN

### D-ego-1: Ego-Entity Graph Topology

**Default: Star (self as hub)**. Self-entity node connects to all enemy and ally entity nodes. Entities don't connect to each other.
- ~38 edges per agent (self→19 + 19→self for 10v10)
- Self-node aggregates all entity information via attention
- Direct analogy: self-node is like the grid bus node, entities are like connected DERs
- **Ablation**: Full connectivity (~380 edges) as a more expensive but more expressive option

### D-ego-2: Per-Type Ally Node Types

Each ally unit type gets its own node type: `stalker_ally`, `zealot_ally`, `colossus_ally`. This gives TransformerConv separate learned projection matrices per ally type, enabling type-specific attention patterns.
- 5 node types total: `{self_entity, enemy, stalker_ally, zealot_ally, colossus_ally}`
- Env wrapper parses the 3 type bits in ally features to route each ally to the correct node type
- Out-of-range allies (all-zero features, zero type bits) are filtered via entity active masking (D-ego-4) rather than assigned to a default type
- Aligns with HGTeam's core heterogeneous design philosophy — explicit type structure in the graph, not implicit in features
- **Ablation**: Single "ally" node type (3 node types total) — simpler, tests whether explicit per-type structure helps

### D-ego-3: Move Features Concatenated After GNN

Move features (4-dim: can_north/south/east/west) are non-relational — they describe terrain around the agent, not entity relationships. Concatenated with the self-node embedding before the MLP, not included in the entity graph.

### D-ego-4: Entity Active Masking

Entity nodes with all-zero features (out of sight range, or padded slots beyond actual unit count) are treated as inactive. Edges to/from inactive entities are filtered, consistent with D3. Implementation: `entity_active = (entity_feats.abs().sum(-1) > 0)`.

### D-ego-5: Agent Folding Into Batch Dimension

B_env × N_agents per-agent ego graphs are folded into PyG's batch dimension. One HeteroData batch, one TransformerConv forward pass. Self-entity embeddings extracted and reshaped to (B, N, embed_dim), then split into per-type groups.

Computational estimate (10v10, batch=32): 320 ego graphs × ~20 nodes = 6,400 nodes. Manageable. For 20v20: 640 × ~40 = 25,600 nodes. May require reduced hidden dim (32 instead of 64).

---

## Key Ablations for SMACv2

| Ablation | What It Tests |
|----------|---------------|
| **ego_entity vs. shared graph** (actor) | Does respecting sight-range observability matter? Quantifies the value of local-only information vs. implicit all-to-all communication |
| **ego_entity vs. flat MLP** (no GNN) | Does the entity-relational structure help? Tests whether decomposing obs into entities + GNN beats processing the flat observation |
| **Star vs. full ego topology** | Does entity-entity reasoning help beyond self-aggregation? |
| **−D8 (geometric mean)** | Factor normalization matters more in SMACv2 (within-episode group size change) than PowerGrid (across-episode only) |
| **−D6/D7 (masked loss)** | Catastrophic at low alive fractions (3/20 = 15%) — SMACv2 is the harder test |
| **−D12 (coop_encoder)** | Does cooperative encoder help when type coupling is implicit (tactical coordination) vs. explicit (VPP tracking)? |

---

## Implementation Timeline Estimate

| Step | Description | Dependency |
|------|-------------|------------|
| 1 | Write `smacv2_variable/common.py` skeleton + yaml configs | None |
| 2 | Validate: single episode runs, group_map correct, active_mask derived, shapes match | Step 1 |
| 3 | Validate: HGTeam forward pass works with SMACv2 tensordicts | Step 2 |
| 4 | Test: 1-seed short training run (1M frames), verify learning signal | Step 3 |
| 5 | Full runs: HGTeam + baselines (MAPPO, stock HAPPO) × 3 seeds | Step 4 |
| 6 | Ablation runs: −D2, −D3, −D8, −D10, −D11/D12 × 3 seeds | Step 4 |

Steps 1–4 are engineering work (~1 week). Steps 5–6 are compute-only (runs in parallel with PowerGridworld experiments, limited only by queue depth).

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| **SMACv2 requires StarCraft II binary** on CHPC nodes | Check if sc2 is available or installable on compute nodes. If not, use PySC2's headless mode. Alternatively, use TorchRL's `fake_tensordict` for initial debugging. |
| **Protoss 10v10 compositions too homogeneous** (always ~5/5/0–1) | Use `dist_type: "fixed_teams"` with explicit variable counts, e.g., Stalkers ∈ [3,7], Zealots ∈ [3,7], Colossi ∈ [0,2]. This gives more composition diversity per episode. |
| **Discrete actions break something in HGTeam** | HGTeam already supports discrete via BenchMARL's categorical policy. The GNN output feeds into a categorical head instead of a continuous one. Test early (Step 3). |
| **Win rate is too noisy for clean ablation tables** | Use 20+ evaluation episodes per checkpoint (same as PowerGridworld). SMACv2 variance is well-studied; 3 seeds × 20 eval episodes should suffice. |
| **The proximity graph adds no value over fully-connected** | This is fine — report both. If fully-connected works equally well, it means the GNN's value comes from type-aware message passing (C3), not topology. That's still a contribution. |
