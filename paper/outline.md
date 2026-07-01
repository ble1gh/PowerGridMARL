# HGTeam: Heterogeneous Graph Teams for Variable-Composition Multi-Agent Reinforcement Learning

**Target venue**: Journal of Machine Learning Research (JMLR)

---

## Status Key
- ✅ **WRITE NOW** — Content is stable regardless of final results
- ⚠️ **DRAFT STRUCTURE** — Write the framing/prose shell, leave placeholders for numbers
- ❌ **WAIT** — Depends on results that don't exist yet

## Headline Contribution (framing discipline)

The paper's **primary** contribution is the **cooperative encoder objective (D12)**:
a principled mechanism that trains a shared heterogeneous-graph encoder with a
cooperative PPO surrogate, while per-group HAPPO-style heads update sequentially
on mixed-motive per-agent rewards. Everything else — the variable-composition
graph mechanics, the geometric-mean HAPPO factor, the edge-aware heterogeneous
graph architecture —
is in service of this headline, not a co-equal contribution. Keep this
discipline throughout the writing.

---

## I. Introduction ✅

### Para 1: The Variable-Team, Mixed-Motive Problem
- Real-world multi-agent systems rarely have fixed team composition OR a single shared objective.
- *Distribution-grid VPP control (PowerGridworld)*: EVs, PV, and storage connect and disconnect dynamically. Each agent has **self-interested** objectives (an EV prioritises its own charging deadline; PV prioritises its own utilisation) that can directly contradict the **team** VPP-tracking objective. Composition varies between and within episodes.
- *Tactical combat (SMACv2)*: Team members die during a mission, permanently changing the surviving team's capability mix. A fully cooperative reward, but with within-episode team-size changes.
- Standard MARL methods (MAPPO, HAPPO, QMIX, MAT) assume fixed agent count and typically common reward. HGTeam's two domains stress these assumptions differently: fixed composition fails in both; the common-reward assumption fails in PowerGridworld.

### Para 2: Why This Breaks Things (technical gap)
- Reward magnitude becomes a function of team size, confounding value estimation.
- HAPPO's sequential importance-weight factor has variance that grows with group size; SMACv2 changes group size *within* a trajectory.
- GNN-based communication corrupts active representations when inactive nodes participate.
- Parameter-sharing scalability methods (SHPPO, EPC) handle variable team size via architectural genericity but assume common reward and do not offer per-type credit assignment.
- HAPPO-family methods give per-type credit assignment and monotonic improvement but assume fixed $N$ and common reward.
- **Nobody currently bridges these requirements** — variable composition, per-type credit assignment, and mixed-motive per-agent rewards in the same scalable framework.

### Para 3: Our Contribution (the pitch)
- We propose **HGTeam**, a framework for variable-composition, heterogeneous, mixed-motive cooperative MARL.
- **Primary contribution**: a *cooperative encoder objective* (D12) that trains a shared heterogeneous-graph encoder via a cooperative per-agent PPO surrogate while per-group HAPPO-style heads update sequentially on mixed-motive per-agent advantages. This is the paper's theoretical and algorithmic centre.
- **Secondary contribution**: a *variable-composition credit-assignment layer* — per-capita reward normalisation (D2), masked loss (D6–D7), and a geometric-mean HAPPO factor (D8) — that makes HAPPO's sequential trust-region machinery numerically stable under variable $N$.
- **Supporting contribution**: a *domain-adaptive edge-aware heterogeneous graph encoder* inspired by the **Heterogeneous Graph Transformer (HGT)** of Hu et al. (2020), operating in a *shared-graph* mode when common-knowledge topology exists (power grid) and an *ego-entity* mode when only private-knowledge observations are available (combat). The current implementation uses `HeteroConv` + `TransformerConv`, not vanilla `HGTConv`, so grid and critic edge attributes can condition attention and messages.
- Validated on VPP control (IEEE 13-node feeder, 40–53 agents, mixed-motive rewards) and SMACv2 tactical combat (Protoss 10v10 / 20v20, up to 36 agents, common reward).
- Scalability head-to-head vs. **SHPPO**, and a **HGTeam+Hyper** variant (HGTeam with SHPPO-style hypernetwork heads) demonstrates the orthogonality of our contributions to SHPPO's.

### Para 4: Paper organization
- Standard paragraph.

---

## II. Related Work ✅

### A. Cooperative and Heterogeneous Multi-Agent RL
- CTDE paradigm: MAPPO (Yu et al., NeurIPS 2022), MADDPG (Lowe et al., 2017), QMIX (Rashid et al., 2018).
- **Heterogeneous-agent sequential updates**: HATRPO/HAPPO (Kuba et al., ICLR 2022) via multi-agent advantage decomposition; HAML (Kuba et al., JMLR 2024) as the operator-theoretic generalisation. These provide the only clean monotonic-improvement guarantees for heterogeneous policies, but require **fixed $N$ and common reward**.
- HARL toolkit (Zhong et al., 2023): benchmarked family of HAPPO descendants.

### B. Scalable and Variable-Team MARL
- **SHPPO** (Zhang et al., ICML 2024) — parameter-shared backbone with a latent-network-generated heterogeneous layer. Zero-shot generalisation to unseen team sizes. Assumes common reward.
- **EPC — Evolutionary Population Curriculum** (Long et al., ICLR 2020) — scaling MARL via curricula over population sizes.
- **Entity-based MARL**: REFIL (Iqbal et al., ICML 2021), UPDeT (Hu et al., ICLR 2021) — handle variable entity observations via entity-level attention, flat agent rosters, fixed policy count.
- **MAT — Multi-Agent Transformer** (Wen et al., NeurIPS 2022) — sequence-model view of cooperative MARL; handles variable $N$ via attention but fixed-$N$ in training loop.
- Gap: none of the above simultaneously handles variable $N$ **and** mixed-motive rewards **and** per-type credit assignment.

### C. Graph Neural Networks in MARL
- **Heterogeneous Graph Transformer (HGT)** (Hu et al., WWW 2020) — meta-relation-parameterised attention GNN for type-diverse graphs. HGTeam adopts the meta-relation framing but implements an edge-aware relation-typed transformer encoder with PyG `HeteroConv` + `TransformerConv`, because PowerGridworld and action-conditioned critics require continuous edge attributes.
- DGN (Jiang et al., ICLR 2020), MAGNet, attention-based communication (CommNet, MAAC, TarMAC).
- Action-conditioned critics: COMA (Foerster et al., AAAI 2018); our per-agent advantages play a similar counterfactual-baseline role without requiring a centralised counterfactual critic.
- **Common-knowledge reasoning** (Nayyar et al., 2013): prescriptor decomposition of Dec-POMDPs separating common and private information; we operationalise this decomposition at the architecture level.

### D. Mixed-Motive and General-Sum MARL
- Sequential social dilemmas (Leibo et al., AAMAS 2017; Jaques et al., ICML 2019; Köster et al., NeurIPS 2022).
- PettingZoo MPE (Terry et al., 2021) as the canonical general-sum benchmark.
- Most existing mixed-motive work studies 2–8 fixed agents; the variable-$N$ axis is largely unexplored.

### E. MARL for Power Systems and Combat
- MARL for grid control (voltage regulation, demand response, EV charging): mostly single-agent RL or centralised optimisation.
- SMACv2 (Ellis et al., 2023): randomised unit types and start positions; prior methods assume fixed rosters.
- Gap: no MARL method simultaneously handles variable agent availability, topology-aware heterogeneous GNNs, and mixed-motive rewards across contrasting domains.

---

## III. Problem Formulation ✅

### A. Variable-Composition Set Markov Game
- Formal definition: tuple $(\mathcal{N}_\text{max}, \mathcal{G}, \mathcal{S}, \{\mathcal{O}_i\}, \{\mathcal{A}_i\}, T, \{R_i\}, \gamma, \mu)$, where $\mathcal{G}$ is the set of agent **types** (groups) and at each episode reset an instance $n_g$ of each group is sampled.
- **Active mask** $m_i^t \in \{0,1\}$ per slot per timestep — may change within episode (EV arrival/departure; unit death).
- **Per-agent rewards** $R_i$ are in general **distinct across agents** and decompose as
  $$R_i = w_c \cdot R_\text{team} + w_i \cdot R^\text{self}_i,$$
  where $R_\text{team}$ is a shared team objective and $R^\text{self}_i$ is an agent-type-specific individual objective. When $w_i = 0$ for all $i$, the game reduces to the common-reward Dec-POMDP of HAPPO; when $w_c = 0$ it reduces to a set of independent MDPs. The interesting regime — and the regime of PowerGridworld — is in between.
- Joint observation, action, reward only over active agents; inactive slots are zero-padded (D1).
- Key distinction from standard Dec-POMDP: agent set is time-varying, stochastic, and rewards are **general-sum**.

### B. Information Structure: Common Knowledge vs. Private Knowledge
- Decompose available information into *common knowledge* $I_C$ (accessible to all agents, or cheaply broadcast) and *private knowledge* $I_i$ (accessible only to agent $i$).
- **Power grid**: $I_C$ = feeder topology, active team composition, types. $I_i$ = local bus voltage, power injection, SOC.
- **Combat**: $I_C = \emptyset$. $I_i$ = sight-range-limited entity observation.
- This decomposition determines the GNN encoder's construction (Section IV.B): shared graph over $I_C$ vs. per-agent ego-graph over $I_i$.

### C. Domain 1: VPP Control (PowerGridworld) — **mixed-motive**
- IEEE 13-node distribution feeder with OpenDSS AC power flow.
- Agent types:
  - EV (10–20): demand response, random arrival/departure, charge scheduling.
  - PV (25): generation curtailment, stochastic irradiance.
  - Storage (5–8): bidirectional charge/discharge, SOC tracking.
- VPP setpoint tracking: collective production target of 25 kW.
- **Mixed-motive reward decomposition** (table):
  - Shared / cooperative: VPP production tracking (per-capita, D2).
  - Local / individual: voltage violation penalty at violating bus; EV charging-urgency penalty; PV self-utilisation reward.
- Episode: 97 steps (48 h, 30-min intervals).
- Composition dynamics: between-episode + within-episode (EV arrival/departure).

### D. Domain 2: Tactical Combat (SMACv2) — **common-reward variable-$N$**
- StarCraft Multi-Agent Challenge v2 with randomised Protoss unit types.
- Agent types: Stalker (~45%), Zealot (~45%), Colossus (~10%).
- Team reward: battle outcome + incremental damage.
- Discrete action space.
- Episode: 50–200 steps.
- Composition dynamics: monotonically decreasing (permanent death).
- Partial observability: sight-range-limited; out-of-range entities zero-filled.
- **Note**: SMACv2 is the *common-reward* half of the story. It tests variable-$N$ HAPPO without the mixed-motive axis, which is the clean ablation against PowerGridworld.

### E. Why Standard Methods Fail (across the two domains)
- Value-target scale $\propto n_\text{active}$ without D2.
- HAPPO factor variance $\propto n_\text{active}$ without D8.
- GNN representations corrupted by inactive nodes without D3.
- Common-reward methods cannot represent the mixed-motive decomposition in PowerGridworld.
- HAPPO's theorem requires fixed $N$ and common reward. Fixed $N$ fails in both domains; common reward fails in PowerGridworld but holds in SMACv2, making SMACv2 the clean common-reward variable-$N$ ablation.
- Motivates the three design layers in Section IV.

---

## IV. Method ✅ (leave hyperparameters as symbols)

### A. Architecture Overview
- **Figure 1**: Full architecture diagram (two panels — PowerGrid + SMACv2).
  - **PowerGrid panel**: IEEE feeder topology → shared heterogeneous graph (agent nodes + grid nodes) → edge-aware relation-typed transformer encoder → per-group MLP policy heads → HAPPO sequential update.
  - **SMACv2 panel**: Per-agent observation → entity decomposition → ego-centric relation-typed transformer encoder → per-agent embedding → per-group MLP policy heads → HAPPO sequential update.
  - Visual emphasis: the D1–D12 machinery is identical in both panels; only the graph construction layer differs.
- One paragraph walking through a single forward pass for each domain.
- Explicitly state: same encoder family, same training algorithm, same design decisions. The `actor_graph_mode` flag is a single configuration choice.

### B. Edge-Aware Heterogeneous Graph Encoder

HGTeam's encoder is HGT-inspired but not vanilla `HGTConv`. We use PyG
`HeteroConv` with a relation-specific `TransformerConv` for each
source-relation-destination tuple. This preserves relation-typed transformer
message passing while supporting continuous edge attributes through
`TransformerConv(edge_dim=...)`. That edge-feature path is load-bearing in
PowerGridworld, where line, transformer, switch, mapping, and action-conditioned
critic edges carry physical or counterfactual information. See
`docs/hgt_vs_heteroconv_attention.md` for the mathematical comparison with
vanilla HGT.

### C. Domain-Adaptive Graph Construction ✅

*Central methodological insight: the GNN encodes the relational component of
available information. What counts as relational depends on the domain's
information structure.*

> In the power grid, the coupling structure between agents — physical topology
> and team composition — constitutes *common knowledge*: information shared
> among all agents that changes on a slow timescale relative to control. Each
> agent's local observations (bus voltage, power injection, SOC) constitute
> *private knowledge*. The relation-typed encoder encodes the common knowledge into
> per-agent embeddings via a single shared graph; private observations are
> processed directly by each agent's policy head.
>
> In tactical combat, there is no common knowledge about the battlefield. Each
> agent observes a private, sight-range-limited view of nearby entities. The
> natural relational structure is therefore *ego-centric*: each agent's flat
> observation is decomposed into entity-level features (self, per-ally,
> per-enemy), and a per-agent relation-typed encoder processes this ego-graph. Out-of-range
> entities have all-zero features, so the GNN's input respects the sight-range
> constraint by construction.
>
> Both modes encode the structured, relational component of available
> information into the same embedding space. The variable-composition
> mechanics (D1–D7, D10), geometric-mean HAPPO factor (D8), and cooperative
> encoder objective (D12) operate identically in both modes. Only the graph
> construction adapts.

### D. Variable-Composition Graph Mechanics

*Problem*: How to build a valid heterogeneous graph when agents appear and
disappear.

1. **Variable topology with active masking** (D1, D3).
   - Max-padded slots; at reset, sample agent counts, set active_mask.
   - Edge filtering: remove all edges where src or dst is inactive.
   - Transformer attention computed only over active neighbours.
   - Math: relation-typed attention equation with filtered edge set $E_\text{active} \subset E_\text{max}$.
2. **Ego-entity graph construction** (SMACv2).
   - Per-agent observation decomposed into entity nodes: self_entity (7d), enemy (9d), per-type allies (9d) for stalker/zealot/colossus.
   - 5 node types total; per-type ally nodes give the relation-typed encoder separate projections per ally type.
   - Star topology; $B_\text{env} \times N_\text{agents}$ ego graphs folded into PyG batch dimension.
   - Entity active masking: zero entities filtered.
   - Move features (4d) concatenated after GNN — non-relational.
3. **Actions as edge features for scalable critics** (D4).
   - Edge $(j \to i)$ carries $a_j$ as edge_attr; self-loops carry 0-vector.
   - Counterfactual $V(s, a_{-i})$ semantics in one forward pass.
4. **Embedding safety net** (D5). Post-GNN multiply by active_mask.

### E. Credit Assignment under Variable $N$ and Mixed-Motive Rewards

*Problem*: How to compute well-behaved policy gradients when team size varies
and agents have different reward functions.

1. **Per-capita reward normalisation** (D2). Team-level reward divided by $n_\text{active}$ before broadcasting. Value targets become $n$-independent.
2. **Masked loss reduction** (D6, D7). Per-element loss with explicit active-mask mean; inactive advantages and value targets zeroed.
3. **Geometric-mean HAPPO factor** (D8).
   - Standard HAPPO factor: $\log F_g = \sum_i \log r_i$, variance $\propto n$.
   - Ours: $\log \tilde F_g = (1/n) \sum_i \log r_i$, variance $\propto 1/n$.
   - **Conjecture** (stated explicitly as such; see Section IV.G): in the small-clip limit, the geometric-mean factor preserves HAPPO's monotonic-improvement property up to $O(\epsilon^2)$.
   - Especially critical for SMACv2: group size changes within a single trajectory.
4. **Per-slot advantage normalisation** (D10). Normalise $\mu, \sigma$ per slot over batch, active entries only; preserve the inter-agent reward-scale hierarchy (critical for mixed-motive).

### F. Cooperative Encoder Objective (D12) — **primary contribution**

*Problem*: HAPPO updates groups sequentially. A naive shared encoder receives
unbalanced gradients and may be biased toward the last-updated group.

**Algorithm (two-phase update, one PPO iteration)**:

- **Phase 1 — Cooperative encoder update.** Unfreeze the shared heterogeneous encoder $\theta^E$. Fresh forward pass for all groups. Compute a clipped PPO surrogate using **per-agent advantages from per-group critics** $\hat A_i$ (not a homogenised team advantage). Single backward + optimiser step on $\theta^E$ only.
- **Phase 2 — HAPPO head updates.** Freeze $\theta^E$. For each group $g$ in a random order, compute the HAPPO sequential surrogate with the geometric-mean factor (D8) propagated across groups. Update only the per-group MLP policy head $\theta^H_g$.

Phase 1 trains the encoder to maximise the **sum** of cooperative and
self-interested objectives (since per-agent advantages carry both). Phase 2
lets each group specialise its head against the updated encoder.

Alternatives considered (and reported as ablations):
- **Joint encoder + heads update** (standard MAPPO-style): reproduces the last-group-bias failure.
- **Balanced gradient accumulation** (D11 / `separate_forward` mode): fixes the gradient imbalance but does not give the clean theoretical structure below.
- **Frozen encoder** (pretrained and held fixed): loses adaptivity.
- **Per-group encoders**: quadratic parameter growth with types.

### F.1 Theoretical Justification

*Key claim.* Using per-agent advantages $\hat A_i$ in the Phase 1 encoder surrogate
optimises the same mixed-motive per-capita objective as the full game, and the
per-agent formulation has lower variance than a shared cooperative advantage.

**Setup.** The mixed-motive per-capita objective is
$$J(\theta) = \frac{1}{N_\text{active}} \sum_i \mathbb{E}\left[\sum_t \gamma^t R_i(s_t, a_t)\right].$$
By the policy-gradient theorem and linearity of the gradient,
$$\nabla_\theta J = \frac{1}{N_\text{active}} \sum_i \mathbb{E}\left[\nabla_\theta \log \pi_i(a_i \mid o_i; \theta) \cdot A_i\right].$$
The Phase 1 clipped PPO surrogate
$$L(\theta) = \frac{1}{N_\text{active}} \sum_i \min\bigl(\rho_i \hat A_i,\; \mathrm{clip}(\rho_i) \hat A_i\bigr)$$
is a first-order approximation of $J$ around $\theta_\text{old}$, and $\nabla_\theta L$
is an unbiased estimator of $\nabla_\theta J$.

**Proposition 1 (variance reduction).** Both the shared cooperative advantage
$\hat A_\text{coop} = (1/N)\sum_i \hat A_i$ and the per-agent advantage
decomposition yield unbiased estimators of $\nabla_\theta J$. The per-agent
formulation has lower gradient variance whenever the per-group critics $V_i$
satisfy the standard counterfactual-baseline conditions (Foerster et al.,
2018). [Formal statement and proof in Appendix.]

**Why this matters for representation learning.** A shared cooperative advantage
produces gradients where every agent type's embedding is pushed in the same
direction, leading to a lazy equilibrium of undifferentiated embeddings. Per-
agent advantages provide **type-specific gradient pressure**: on a given
timestep, EV embeddings may be pushed to change (negative advantage) while PV
embeddings are reinforced (positive advantage). This sign difference is the
gradient-level mechanism that drives embedding separation.

**Connection to mixed-motive rewards.** In PowerGridworld, shared reward
components create correlated advantages (cooperative pressure); individual
components create divergent advantages (specialisation pressure). Averaging
into a cooperative advantage erases the individual components. The per-agent
formulation lets the encoder learn representations that balance both — it
optimises the sum of cooperative and individual objectives, not just the
cooperative component.

**Remark.** Phase 1 does not include a HAPPO importance-weighting factor
because the Phase 1 update applies to all agents simultaneously, not
sequentially. The sequential trust-region factorisation applies only to Phase
2's per-group head updates, where HAPPO's machinery remains valid *conditional
on the encoder at the start of Phase 2*.

### G. Theoretical Results (at-a-glance)

- **Theorem 1 (Cooperative Encoder Update is Sound).** [Precise statement in Appendix.] Under a standard PPO-clip assumption and conditional on the encoder parameters at iteration $k$, the two-phase update satisfies: (i) Phase 1's encoder step is a stochastic-gradient ascent step on the mixed-motive per-capita objective $J$, with per-agent advantages serving as valid counterfactual baselines; (ii) Phase 2's sequential head updates preserve HAPPO's monotonic improvement *with respect to the heads and per-agent conditional advantages* at the updated encoder.
- **Conjecture 1 (Variable-$N$ HAPPO with Geometric Mean).** In the small-clip limit $\epsilon \to 0$, HAPPO with the geometric-mean factor (D8) recovers HAPPO's monotonic-improvement property up to $O(\epsilon^2)$, with an $n$-dependent effective clip range. We discuss the technical obstacle (HAPPO's exact factorisation requires a *product* factor) and present an approximation argument; empirical validation is in Section VI.E.

---

## V. Experimental Setup ⚠️ (write prose, leave number placeholders)

### A. Domain 1: PowerGridworld VPP Control (Mixed-Motive)
- IEEE 13-node distribution feeder (OpenDSS steady-state AC power flow).
- Agent counts, buses, timing — table.
- **Reward decomposition table** (cooperative vs. individual, per type).

### B. Domain 2: SMACv2 Tactical Combat (Common-Reward Variable-$N$)
- Protoss 10v10 (primary) and 20v20 (scaling test).
- Observation entity decomposition (4d move, 9d enemy, 9d ally, 7d self).
- Ego-entity relation-typed transformer encoder with star topology.

### C. Baselines
- **MAPPO** (parameter sharing within groups).
- **HAPPO** (per-type groups; naive variable-$N$ handling: zero rewards, global advantage norm, no D1–D8).
- **SHPPO** (parameter-shared backbone + latent hypernetwork heads). Direct scalability competitor.
- **EPC — Evolutionary Population Curriculum** (Long et al., ICLR 2020). Direct variable-$N$ curriculum competitor.
- **MAT — Multi-Agent Transformer**. Attention-based cooperative MARL baseline.
- **QMIX** (SMACv2 only; standard value-decomposition baseline).
- All baselines adapted to variable agents via D1 (max-padding) for fairness.

### D. Ablation Variants

| Variant | What Changes | Strongest Test |
| --- | --- | --- |
| −D2 | Raw team reward (not per-capita) | PowerGrid (reward confound) |
| −D3 | No edge filtering | Both (GNN corruption) |
| −D8 | Product / arithmetic-mean HAPPO factor | SMACv2 (intra-episode size change) |
| −D10 | Global advantage normalisation | Both (reward hierarchy) |
| −D12 | Last-group-only encoder (no coop_encoder) | Both (primary ablation) |
| D11 only | Balanced accumulation instead of coop_encoder | Both (isolates coop objective contribution) |
| −edge-aware encoder | remove relation-typed transformer encoder or replace edge-aware relations with a simpler non-edge-aware variant | Both (encoder ablation) |
| −GNN | `gnn_mode=none` (MLP only) | Both (relational structure) |
| HGTeam+Hyper | HGTeam with SHPPO-style hypernetwork heads | Scalability (composes with SHPPO) |
| ego→shared | Shared-graph actor on SMACv2 | SMACv2 (info-structure test) |
| ego→MLP | Flat MLP actor on SMACv2 | SMACv2 (entity structure) |
| Star→full | Full ego-entity connectivity | SMACv2 |

### E. Scalability Protocol — train-at-$N$, test-at-$M$

- **PowerGridworld**: train on EV counts sampled from $[10, 20]$; **test** on fixed $\{8, 10, 12, 15, 18, 20, 24\}$. The $\{8, 24\}$ test points are **out of distribution**.
- **SMACv2**: train on Protoss 10v10; test on **10v10 (seen)** and **20v20 (OOD)** with identical hyperparameters.
- Same protocol applied to all baselines (SHPPO, EPC, MAPPO, HAPPO, MAT).
- Report: seen-size performance, interpolation, extrapolation; mean ± seed-std.

### F. Evaluation Protocol
- Metrics:
  - PowerGrid: mean per-capita return; per-component return breakdown (VPP tracking, voltage violation, individual); VPP tracking error (kW); voltage violation rate.
  - SMACv2: win rate; mean episode return; return vs. alive count.
- Seeds: [TBD]; evaluation episodes per checkpoint: [TBD].
- Training: frames, batch size, minibatch iterations (table).

---

## VI. Results ❌ (wait for runs)

### A. Main Comparison
- **Table 1**: HGTeam vs. baselines — two panels (PowerGrid / SMACv2).
- Demonstrates generality: same method, two contrasting domains, both SOTA or competitive.

### B. Coop Encoder Ablation (primary)
- **Table 2**: `accumulated` vs. `separate_forward` (D11) vs. `coop_encoder` (D12).
  - Both domains.
  - Headline ablation — isolates the paper's primary contribution.

### C. Full Design-Decision Ablation
- **Table 3**: Remove each of D2, D3, D8, D10, edge-aware relation typing, GNN in turn.
- Report degradation relative to full HGTeam.
- Highlight which decisions cause catastrophic failure vs. graceful degradation.

### D. Scalability and Zero-Shot Generalisation
- **Table 4 / Figure 2**: train-at-$N$, test-at-$M$ table for HGTeam, SHPPO, EPC, MAPPO, HAPPO.
- **HGTeam+Hyper** variant reported as a compositional result: does combining HGTeam with SHPPO-style heads yield additive gains?
- Key claim: HGTeam matches or beats SHPPO on scalability **and** handles mixed-motive rewards, which SHPPO cannot.

### E. HAPPO Factor Analysis (D8)
- **Figure 3**: factor magnitude distribution at different group sizes.
  - PowerGrid: across-episode variation.
  - SMACv2: within-episode variation.
  - Compare geometric-mean vs. product factor.
- Empirical validation of Conjecture 1.

### F. Information Structure and Graph Mode
- **Table 5**: shared vs. ego-entity actor on each domain; edge-aware relation-typed encoder vs. simpler GNN/MLP variants.
- Synthesises the common/private knowledge story.

### G. Learning Dynamics
- **Figure 4**: learning curves (mean return vs. frames), HGTeam vs. baselines, both domains.

### H. Learned Behaviour Analysis
- **Figure 5**: example rollouts.
  - PowerGrid: VPP production vs. setpoint over 48 h, decomposed by agent type.
  - SMACv2: relation-typed attention on entity nodes at different battle stages.

---

## VII. Discussion ⚠️ (draft structure, fill after results)

### A. When Does the Cooperative Encoder Objective Matter?
- D12 vs. D11 vs. naive: when is the gap largest?
- Hypothesis: the gap is largest when (i) group sizes are imbalanced and (ii) per-type rewards diverge from the team reward.

### B. Variable-$N$ HAPPO in the Wild
- D8 empirical validation; limits of the geometric-mean factor; when the factor variance is the actual bottleneck.

### C. Information Structure and Architecture Choice
- When shared-graph vs. ego-entity is the principled choice.
- Hybrid cases with partial common knowledge.
- Connection to common-knowledge Dec-POMDP (Nayyar et al., 2013).

### D. HGTeam and SHPPO: Orthogonal or Substitutable?
- HGTeam+Hyper variant: do the two heterogeneity mechanisms compose?
- If yes: SHPPO-style hypernetworks should be adopted. If no: the relation-typed encoder carries the load.

### E. Mixed-Motive MARL at Scale
- Revisit: HAPPO + mixed-motive at variable $N$ was previously an open problem. What does HGTeam's experience suggest for general mixed-motive MARL?

### F. Scalability
- Computational cost: wall-clock on PowerGrid (53 agents) vs. SMACv2 (20 agents, denser ego graphs).
- Memory: max-padding cost; ego-entity batching cost.
- Edge-aware relation-typed encoder vs. simpler GNN/MLP compute profile.

### G. Limitations
- Two domains; other information structures untested.
- **Conjecture 1 is not proved**. The geometric-mean factor's monotonic-improvement property is established only empirically and in the small-clip limit.
- The mixed-motive axis is only exercised in PowerGridworld; SMACv2 is fully cooperative.
- No decentralised-execution demonstration (though ego-entity mode is deployment-ready by construction).
- OpenDSS is steady-state, not transient.
- No self-play / team-vs-team competitive results; "mixed-motive" here means per-agent general-sum rewards, not team-vs-team.

---

## VIII. Conclusion ❌ (wait for results)

- HGTeam is a framework for variable-composition, heterogeneous, **mixed-motive** cooperative MARL.
- Primary contribution: the cooperative encoder objective (D12) reconciles a shared heterogeneous graph encoder with per-group HAPPO-style heads.
- Secondary contribution: variable-$N$ credit assignment (D2, D6–D8, D10) keeps HAPPO numerically stable under variable team size.
- Supporting: edge-aware heterogeneous graph encoder with a domain-adaptive graph-construction layer (shared vs. ego-entity).
- Validated on two contrasting domains: continuous mixed-motive infrastructure control (PowerGridworld) and discrete common-reward tactical combat (SMACv2). Direct head-to-head against the closest scalability competitor (SHPPO).
- Future work: (a) formal proof of Conjecture 1; (b) extension to team-vs-team self-play; (c) partial-common-knowledge domains; (d) decentralised-execution deployment.

---

## Appendix (Supplementary Material) ✅

### A. Full Design Decision Catalog
- D1–D12 + D-ego-1..5 from `variable_agent_design_choices.md`, reformatted.
- Grouped into: (A) Environment handling, (B) Graph construction, (C) Loss & advantage mechanics, (D) Encoder training.

### B. Environment Details
- PowerGrid: OpenDSS feeder topology, full reward function equations, observation/action spaces.
- SMACv2: unit-type statistics, observation entity decomposition, sight-range parameters, Protoss config.

### C. Hyperparameter Tables
- Algorithm (per domain), HGT architecture, environment.

### D. Theorem 1 Proof and Conjecture 1 Discussion
- Formal statement of Theorem 1 with full proof.
- Technical discussion of Conjecture 1: why the geometric-mean factor breaks exact HAPPO factorisation; first-order argument; empirical support.

### E. Extended Results
- Per-agent-type return breakdown (both domains).
- Voltage profile analysis per bus (PowerGrid).
- Win rate by unit-composition draw (SMACv2).
- Factor magnitude histograms.
- Full scalability table (all methods × all test sizes).

### F. Common-Knowledge Formalism
- Formal Dec-POMDP information-structure decomposition.
- Definition of $I_C$ and $I_i$ for each domain.
- Connection to prescriptor approach (Nayyar et al., 2013).

---

## Figures To Create

| Figure | Status | Description |
|--------|--------|-------------|
| Fig 1: Architecture diagram | ✅ Draw now | Two-panel: PowerGrid shared-graph encoder + SMACv2 ego-entity encoder |
| Fig 2: Scalability (train-at-$N$, test-at-$M$) | ❌ Need runs | HGTeam, SHPPO, EPC, MAPPO, HAPPO on both domains |
| Fig 3: D8 factor analysis | ❌ Need runs | Geometric vs. product factor, across- and within-episode |
| Fig 4: Learning curves | ❌ Need runs | Both domains |
| Fig 5: Rollout analysis | ❌ Need runs | PowerGrid 48 h tracking + SMACv2 entity attention |
| Fig 6: IEEE feeder topology | ✅ Draw now | Standard diagram with agent placement |

---

## Tables To Create

| Table | Status | Description |
|-------|--------|-------------|
| Table 1: Main comparison | ❌ Need runs | Two-panel, HGTeam vs. baselines |
| Table 2: Coop encoder ablation | ❌ Need runs | D12 vs. D11 vs. naive — **primary ablation** |
| Table 3: Full ablation | ❌ Need runs | Remove D2/D3/D8/D10/edge-aware encoder/GNN individually |
| Table 4: Scalability | ❌ Need runs | Train-at-$N$, test-at-$M$; all methods |
| Table 5: Graph mode | ❌ Need runs | Shared vs. ego-entity; edge-aware relation-typed encoder vs. simpler variants |
| Table A1: Hyperparameters | ⚠️ Draft now | Algorithm, graph encoder, environment |
| Table A2: Design decisions | ✅ Write now | D1–D12 + D-ego summary |

---

## Writing Priority Order

1. **Section III** (Problem Formulation) — write to completion; the mixed-motive formulation is the foundation.
2. **Section II** (Related Work) — write to completion.
3. **Section IV.F and IV.F.1** (Cooperative encoder + theoretical justification) — this is the headline; write it well.
4. **Section IV.B–E** (edge-aware heterogeneous graph encoder, graph construction, variable-composition mechanics, credit assignment) — prose; symbols for hyperparameters.
5. **Section IV.G** (Theoretical results at-a-glance) — state Theorem 1 and Conjecture 1 precisely.
6. **Section I** (Introduction) — write after III and IV are solid.
7. **Section V** (Experimental Setup) — prose shell, number placeholders.
8. **Appendix A–D** (Design catalog, environment, hyperparameters, Theorem 1 proof) — draft now.
9. **Figures 1 & 6** — draw architecture (two-panel) and feeder topology.
10. **Sections VI, VII, VIII** — wait for results.
