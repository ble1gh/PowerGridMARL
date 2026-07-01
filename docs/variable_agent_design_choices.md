# Design Choices for Variable Agent Counts and Mixed-Motive Rewards

Quick agent onboarding: see `docs/agent_quickstart.md` for architecture, file map, and current training semantics.
For the theoretical foundations see `paper/appendix_theorem_proofs.md`.
For the research background see `docs/hgteam_research_background.md`.

**Terminology note.** "Mixed-motive" (or "general-sum with a cooperative team
reward") is used throughout instead of the older phrase "mixed cooperative-
competitive". PowerGridworld's reward decomposes as $R_i = w_c R_\text{team} + w_i R^\text{self}_i$; SMACv2 uses a common reward. This terminology change
is consistent with Leibo et al. (AAMAS 2017) and the sequential-social-dilemma
literature. Team-vs-team competition is out of scope for the first paper.

This document catalogues every design decision made to handle (i) variable
numbers of agents across episodes and (ii) mixed-motive per-agent rewards.
Each section states the **problem**, the **decision** taken, the
**alternatives** considered, and the **justification**.

**Organisational layers** (restructured):

- **A. Environment handling** — D1 (padding + active mask), D2 (per-capita
  reward normalisation).
- **B. Graph construction & message passing** — D3 (edge filtering), D4
  (actions as edge features), D5 (embedding safety net).
- **C. Loss & advantage mechanics** — D6 (masked loss reduction), D7 (zero
  advantages for inactive), D8 (geometric-mean HAPPO factor; **Conjecture 1**),
  D10 (per-slot masked advantage normalisation).
- **D. Encoder training & objectives** — D11 (balanced gradient accumulation),
  D12 (cooperative encoder objective; **Theorem 1** — headline).
- **E. Logging & evaluation** — D9 (masked return logging).
- **F. Cross-cutting invariants** — D13 (per-$N$ scaling across reward, loss,
  entropy, KL).

---

## A. Environment handling

### D1. Max-Padded Agent Slots with Binary Active Mask

| | |
|---|---|
| **Problem** | TorchRL and BenchMARL expect a fixed tensor layout across episodes, but agent counts vary: EVs ∈ [10, 20], PVs = 25, Storage ∈ [5, 8]. |
| **Decision** | Allocate the maximum number of slots per type (20 EV, 25 PV, 8 Storage = 53 total). At each episode reset, sample a random count per type and set a per-slot boolean `active_mask`. Inactive slots receive zero observations, zero actions, and zero rewards. |
| **Alternatives** | (a) Dynamic tensor shapes per episode — incompatible with vectorised batching in TorchRL's replay buffers and collectors. (b) Separate policies per agent count — combinatorial explosion. |
| **Justification** | Padding is the standard approach for variable-length sequences (cf. NLP). The active mask lets every downstream component selectively ignore inactive slots without architectural changes to TorchRL. |
| **Files** | `BenchMARL/benchmarl/environments/PowerGridworldVariable/common.py` |

### D2. Per-Agent VPP Reward Normalisation (per-capita team reward)

| | |
|---|---|
| **Problem** | The VPP (Virtual Power Plant) production-tracking reward is a single team-level scalar broadcast to every active agent. When that scalar is the same regardless of team size, the total reward summed across agents scales linearly with $n_\text{active}$, creating a confound: the critic and advantage estimator observe systematically different reward magnitudes for different episode compositions. |
| **Decision** | Divide the VPP reward by $n_\text{active}$ before broadcasting, so each agent receives `reward / n`. The value function targets per-agent expected return rather than a share of a size-dependent total. |
| **Alternatives** | (a) Leave reward un-normalised — critic must implicitly learn the $n$-dependence. (b) Normalise advantages instead of rewards — doesn't fix the value target scale. (c) Normalise at the critic output — more complex, entangles architecture with reward structure. |
| **Justification** | Normalising at the source is simplest and keeps the reward semantics clean: each agent's reward reflects its *per-capita* contribution to the team objective, independent of how many agents happen to be present. This is analogous to how cooperative game theory divides coalition value by coalition size (Shapley, 1953). Combined with D13, this is one of several places where per-$N$ scaling is applied consistently. |
| **Files** | `PowerGridworld/gridworld/multiagent_env.py` (`reward_transform`) |

---

## B. Graph construction & message passing

### D3. Edge Filtering — Remove Edges to/from Inactive Nodes

| | |
|---|---|
| **Problem** | The heterogeneous GNN sends messages along all edges, including those involving inactive (zero-padded) agent nodes. Inactive nodes emit non-trivial messages because attention GNNs have learned bias terms ($\alpha_{i,j} \propto \text{softmax}(\cdot + \mathbf{b})$), so even zero-featured nodes produce non-zero attention scores and pollute active nodes' representations. |
| **Decision** | In `_tensordict_to_hetero_data()`, remove all agent–agent edges where either endpoint is inactive. This is done per edge type by indexing into the flattened `active_mask` and keeping only edges where `src_active AND dst_active`. |
| **Alternatives** | (a) Attention masking inside the GNN layer (set attention weights to $-\infty$) — PyG's `TransformerConv` does not support a native attention mask parameter. (b) Zero the node features (already done) — insufficient because bias terms in $Q/K$ projections still produce non-zero attention contributions. (c) Post-hoc zeroing of inactive embeddings — doesn't prevent corruption of *active* node embeddings during aggregation. |
| **Justification** | Edge filtering is the only complete solution: it prevents inactive nodes from participating in message passing at all. With edges removed, attention softmax is computed only over active neighbours, and active nodes' representations are uncontaminated. This applies to the current edge-aware `HeteroConv` + `TransformerConv` encoder and would also apply to any future edge-feature-compatible HGT-style variant. |
| **Files** | `BenchMARL/benchmarl/models/heterognn.py` (`_tensordict_to_hetero_data`, active mask edge filtering) |

### D4. Actions as GNN Edge Features (Option B) — Replacing Flat Action Concatenation

| | |
|---|---|
| **Problem** | The original centralised critic used `_compute_augmented_obs()` to flat-concatenate all 53 agents' actions into each agent's observation. This creates 53 zero-padded action slots, most of which are noise from inactive agents, and scales the critic input dimension linearly with max agent count. |
| **Decision** | Instead of concatenation, pass other agents' actions as **edge attributes** on the GNN's interaction edges. For edge $(j \to i)$, `edge_attr = a_j`. Self-loop edges carry `edge_attr = 0` so agent $i$ never sees its own action, yielding $V(s, a_{-i})$ counterfactual semantics. Combined with D3 (edge filtering), inactive agents' actions never enter the computation. |
| **Alternatives** | (a) Per-agent counterfactual forward passes — one GNN pass per agent to compute $V(s, a_{-i})$, requiring 53× compute. (b) Keep flat concatenation but mask inactive slots — still $O(\text{max\_agents})$ input dim, wastes capacity on padding. |
| **Justification** | Option B achieves the same counterfactual semantics as Option A in a single forward pass by leveraging `TransformerConv`'s native `edge_dim` support: $\alpha_{i,j} = \text{softmax}\left(\frac{(W_3 x_i)^T (W_4 x_j + W_6 e_{ij})}{\sqrt{d}}\right)$. The edge attribute $e_{ij} = a_j$ naturally conditions attention on the sender's action. This reduces compute from $O(n)$ forward passes to $O(1)$, and the critic's input spec no longer depends on `max_agents`. |
| **Files** | `BenchMARL/benchmarl/models/heterognn.py` (`_collect_action_edge_features`, `_tensordict_to_hetero_data`); `BenchMARL/benchmarl/algorithms/HGTeam.py` (`_compute_other_actions_dim`) |

### D5. Zero-Embedding Safety Net for Inactive Agents

| | |
|---|---|
| **Problem** | Even with edge filtering (D3), numerical residuals from layer norms, bias terms in output projections, or floating-point accumulation could leave inactive agent embedding slots with small non-zero values. |
| **Decision** | After the GNN forward pass, multiply each agent's output embedding by its `active_mask` value (0 or 1). Belt-and-suspenders safeguard applied in `_forward()`. |
| **Alternatives** | Rely solely on edge filtering — likely sufficient in practice, but hard to guarantee across all layer types and future architecture changes. |
| **Justification** | Cheap (one element-wise multiply) and eliminates any residual leakage. Prevents inactive embeddings from affecting downstream MLP/Transformer heads. |
| **Files** | `BenchMARL/benchmarl/models/heterognn.py` (`_forward`, post-GNN active_mask multiply) |

---

## C. Loss & advantage mechanics

### D6. Loss Reduction = "none" with Masked Mean

| | |
|---|---|
| **Problem** | TorchRL's `ClipPPOLoss` aggregates (mean or sum) over all elements including inactive agent slots. Inactive slots have zero advantages (see D7) but potentially non-zero log-probs and entropy, which would corrupt gradient estimates. A standard mean over all 53 slots dilutes the gradient by including ~50% inactive agents. |
| **Decision** | Initialise `ClipPPOLoss` with `reduction="none"` to obtain per-element losses. In `HGTeamLoss.forward()`, apply masked reduction: `(loss * active_mask).sum() / active_mask.sum()`. |
| **Alternatives** | (a) Standard mean reduction — dilutes gradients by $53/n_\text{active}$, biased toward smaller groups. (b) Sum reduction — gradient magnitude scales with $n_\text{active}$. (c) Filter tensordict to only active agents before loss — complex reshape, breaks TorchRL's module assumptions. |
| **Justification** | Masked mean is gradient-unbiased: each active agent contributes equally regardless of padding. Combined with D7 (zero advantages), the policy gradient $\nabla \log \pi(a \mid s) \cdot A$ is exactly zero for inactive slots even before masking, but masking also handles the entropy and critic loss terms. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeam.py` (`HGTeamLoss.forward`, `_get_loss`, `reduction="none"`) |

### D7. Zero Advantages and Value Targets for Inactive Agents

| | |
|---|---|
| **Problem** | GAE computes advantages from TD errors $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. Even with zero rewards, the critic may predict non-zero values for inactive slots (e.g., due to shared parameters processing the zero observation into a non-zero value), producing non-zero advantages that inject noise into the policy gradient. |
| **Decision** | After GAE computation, explicitly zero out `advantage[inactive] = 0` and set `value_target[inactive] = state_value[inactive].detach()`. The second assignment ensures the critic loss for inactive slots is zero (target equals prediction). |
| **Alternatives** | (a) Train the critic to predict zero for inactive slots — would require an auxiliary loss or architectural constraint. (b) Mask only in the loss (D6) — works for loss, but non-zero advantages would still affect advantage normalisation statistics (see D10). |
| **Justification** | Direct zeroing is simple and correct. It prevents inactive agents from (i) contributing policy gradient signal and (ii) skewing the advantage normalisation, which divides by the standard deviation of all advantages including inactive slots. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeam.py` (`process_batch`), `BenchMARL/benchmarl/algorithms/HGTeamHA.py` (`process_batch`) |

### D8. HAPPO Factor: Geometric-Mean Normalisation

| | |
|---|---|
| **Problem** | HAPPO computes a per-group importance weight factor $F_g = \prod_{i \in g} \frac{\pi_\text{new}(a_i \mid s)}{\pi_\text{old}(a_i \mid s)}$ and passes it to subsequent groups as an advantage multiplier. In log-space, $\log F_g = \sum_i \log r_i$. With variable agent counts, the sum has a different number of terms per sample (10–20 EVs, 5–8 Storage). The variance of $\log F_g$ grows linearly with $n_\text{active}$ (and the variance of $F_g$ grows exponentially), causing: (i) systematic clipping bias for larger groups, and (ii) inconsistent factor magnitude passed to downstream groups across samples. |
| **Decision** | Divide `log_group_ratio` by $n_\text{active}$, making the factor the **geometric mean** of per-agent ratios: $F_g = \left(\prod_i r_i\right)^{1/n}$. |
| **Alternatives** | (a) No normalisation (original HAPPO) — factor variance is $n$-dependent. (b) Per-agent clipping before summing — more conservative, but variance of the sum still grows with $n$. (c) Hybrid: per-agent clip + normalise — most controlled but furthest from the original algorithm. |
| **Justification (engineering)** | The geometric mean preserves the monotone-improvement guarantee *locally* (if all per-agent ratios are near 1, the mean is near 1) while ensuring the factor has consistent scale regardless of team size. This is mathematically equivalent to raising the standard HAPPO factor to the power $1/n$, which compresses but does not lose information about policy-change magnitude. |
| **Theoretical status** | **Conjecture 1** in `paper/appendix_theorem_proofs.md`. The geometric-mean substitution breaks the exact equivalence with HAPPO's multi-agent advantage decomposition (which requires a *product* factor). Monotonic improvement is recovered only in the small-clip limit up to $O(\epsilon^2)$, with an effective clip range that becomes $n$-dependent. Empirical validation is required. Do **not** cite D8 as theoretically equivalent to standard HAPPO in publications. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeamHA.py` (`train_groups`, geometric-mean normalisation of `log_group_ratio`) |

### D10. Per-Slot Masked Advantage Normalisation

| | |
|---|---|
| **Problem** | TorchRL's built-in advantage normalisation (`_standardize`) includes inactive (zero-padded) advantages in its statistics. After D7 zeros inactive advantages, a partially-active slot (e.g., EV slot 15, active ~45% of the time) has its per-slot mean pulled toward zero and its std compressed. When the normalisation divides by this compressed std, the few real advantage values are **inflated**, giving rarely-active slots disproportionately large gradient signals. Meanwhile, always-active slots (PV agents) get relatively dampened gradients. |
| **Decision** | Disable TorchRL's `normalize_advantage` and implement per-agent-slot masked normalisation in `process_batch()`, after D7 zeroing. For each slot, compute mean and std over only the batch entries where that slot is active. Re-zero inactive entries after normalisation since $(0 - \mu)/\sigma \neq 0$. |
| **Alternatives** | (a) Global masked normalisation (single mean/std across all active entries) — simpler, but erases inter-agent advantage scale differences that reflect the heterogeneous reward structure. In a mixed-motive environment, agent types with higher marginal impact should retain larger advantage magnitudes. (b) Leave TorchRL's normalisation enabled and accept the zero-contamination — biases policy gradient toward rarely-active slots. (c) No advantage normalisation at all — possible but can lead to training instability. |
| **Justification** | Per-slot normalisation preserves the natural advantage scale hierarchy across agent types while ensuring each slot has zero-mean unit-variance advantages within its own active-entry distribution. This is critical in mixed-motive settings: the self-reward dimension (EV urgency, PV utilisation, voltage penalty) requires that gradient magnitudes reflect actual reward-signal differences. Cross-agent normalisation would artificially equalise these signals. |
| **Related work** | This is essentially masked advantage normalisation; it is not gradient-reweighting in the PCGrad / CAGrad sense. We intentionally do **not** do inter-task gradient reconciliation because agents are not solving distinct tasks in the multi-task sense — they share environment dynamics and differ only in reward weighting. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeam.py` (`process_batch`, `_get_loss`), `BenchMARL/benchmarl/algorithms/HGTeamHA.py` (`process_batch`, `_get_loss`) |

---

## D. Encoder training & objectives

### D11. Balanced GNN Gradient Accumulation (`separate_forward` Mode)

| | |
|---|---|
| **Problem** | In HAPPO's `separate_forward` encoder-update mode, the shared actor GNN is frozen during per-group PPO loops, then unfrozen and updated separately afterwards. The original implementation used only the last group's data for this update, meaning the GNN gradient was 100% biased toward whichever group happened to be last in the random HAPPO ordering. Over many iterations the bias averages out, but within any single iteration one agent type dominates the encoder update — problematic because GNN parameters serving different edge types (e.g., EV↔Storage interaction) only receive gradient when the relevant group is selected. |
| **Decision** | Accumulate gradients from all groups before stepping: for each group, sample a minibatch, forward through that group's loss, and call `(loss / n_groups).backward()`. The `1/n_groups` scaling ensures the accumulated `.grad` tensors equal the mean gradient across groups. A single `optimizer.step()` then applies one coherent Adam update. |
| **Alternatives** | (a) Last-group only (original) — simplest, zero extra compute, but systematically biased. (b) Round-robin cycling — zero extra compute, but at any single iteration the gradient is still 100% from one group. (c) Loss averaging (sum losses, one backward) — mathematically identical to gradient accumulation but ~3× peak memory (all 3 computation graphs alive). (d) Weighted gradient accumulation (weight by group size) — adds a hyperparameter; unclear whether larger groups deserve more gradient signal. |
| **Justification** | Gradient accumulation achieves identical results to loss averaging with 1/3 the peak memory. Each group contributes equally regardless of agent count. The uniform $1/n_\text{groups}$ weighting treats each group's learning signal as equally important, avoiding implicit prioritisation of larger groups. |
| **Relation to D12** | D11 is the *non-cooperative* encoder update (each group contributes its own selfish PPO gradient, averaged). D12 replaces this with an explicit cooperative PPO surrogate. D11 remains available as an ablation baseline; D12 is the headline. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeamHA.py` (`train_groups`, `separate_forward` branch) |

### D12. Cooperative Encoder Objective (`coop_encoder` mode) — **headline**

**Implementation status.** The current code in `HGTeamHA.py` already
implements the formulation described below (encoder-first Phase 0, per-agent
advantages, per-agent ratios, per-capita aggregation). This corresponds to
**Formulation B** in `paper/appendix_theorem_proofs.md`; Theorem 1 applies
directly. An earlier version of this document described an out-of-date
Formulation A + heads-first scheme; that description was never accurate to
the current code and has been removed.

| | |
|---|---|
| **Problem** | The environment has mixed-motive structure: all agents share a team-level VPP-tracking reward (cooperative), but also receive local rewards — voltage penalties, PV utilisation bonuses, EV urgency costs — that can contradict the team term. Both `accumulated` and `separate_forward` modes train the shared actor GNN using per-group HAPPO losses, meaning the encoder gradient is a mixture of each group's *selfish* advantage. There is no explicit cooperative signal: the encoder happens to learn cooperation through the shared VPP component of per-group advantages, but this signal is diluted by self components and entangled with HAPPO's sequential update dynamics. |
| **Decision** | Two-phase training with **encoder update first** (Phase 0 in the code; conceptually Phase 1 in the paper): |
|    | **Phase 0 — cooperative encoder update.** Unfreeze the GNN. For each minibatch and each group, compute per-agent log-probs under the current policy and the behaviour-policy snapshot, yielding per-agent ratios $\rho_i$ and per-agent advantages $\hat A_i$ from that group's critic (already S8a-normalised and D7-zeroed). Accumulate a **sum of per-agent clipped PPO terms** across groups, mask by `active_mask`, divide by total active count, and step the GNN optimiser. No HAPPO factor is applied — Phase 0 runs before any sequential head update. |
|    | **Phase 1 — HAPPO head updates.** Freeze the updated encoder. For each group $g$ in random order, re-evaluate old log-probs under the new encoder (so the HAPPO factor reflects head-only drift), then run standard PPO on the per-group head with the cumulative geometric-mean HAPPO factor (D8) propagated across groups. |
| **Surrogate (Formulation B)** | $L^{\text{coop}}(\theta^E) = \dfrac{1}{N_\text{total\_active}} \sum_{g \in \mathcal{G}} \sum_{i \in g} m_i \cdot \min\bigl(\rho_i \hat A_i,\; \text{clip}(\rho_i, 1 \pm \epsilon) \hat A_i\bigr)$ where $m_i$ is agent $i$'s active mask. Per-agent ratios (not geometric-mean aggregated). Per-agent advantages (not mean aggregated). |
| **Alternatives** | (a) `accumulated` mode — GNN participates in all groups' backward passes at `lr/n_groups`. Simple, but GNN gradient is the mean of selfish gradients, not a cooperative gradient. (b) `separate_forward` / D11 — balanced gradient accumulation after HAPPO. Not cooperative. (c) Separate cooperative critic network (mean-pooling MLP, virtual node, attention pooling) — adds parameters and a new optimiser; higher complexity. (d) **Formulation A** — single scalar per sample: mean advantage × geometric-mean ratio, one clipped PPO term. Both A and B are unbiased estimators of the same per-capita cooperative gradient (Proposition 2 in `paper/appendix_theorem_proofs.md` §3.4); they differ in variance (B uses per-agent baselines) and in clipping behaviour (B clips each agent, A clips only the aggregate ratio). The code uses Formulation B because per-agent clipping and per-agent baselines are both beneficial under heterogeneity. |
| **Justification** | The cooperative encoder update is a standard (non-sequential) PPO step whose gradient is an **unbiased** estimator of the mixed-motive per-capita objective gradient (Theorem 1 in `paper/appendix_theorem_proofs.md` §3.7). Per-agent advantages act as per-agent counterfactual baselines, yielding lower gradient variance than a shared cooperative advantage and providing type-specific gradient pressure that drives embedding differentiation across agent types. Phase-1 HAPPO head updates at the updated encoder improve $\bar J$ to first order regardless of the HAPPO factor (Theorem 2 in appendix §4). The HAPPO factor's higher-order effect in the mixed-motive variable-$N$ setting is a conservative trust-region coupling rather than a theorem-level guarantee; whether it helps convergence is an empirical question (see appendix §4.5 for proposed ablations). |
| **Connection to mixed-motive** | Individual actor heads decide how much each agent cooperates vs. self-optimises. The shared GNN encoder represents the cooperative structure of the environment. Per-agent advantages naturally preserve the multi-objective structure: shared reward components create correlated advantages across types (cooperative pressure), self components create divergent advantages (specialisation pressure). Averaging into a single cooperative advantage (Formulation A) would erase the self components, losing specialisation signal. |
| **Zero cooperative critic parameters** | By linearity of expectation, $V_\text{coop}(s) = \frac{1}{N_\text{active}} \sum_i V^i(s)$ is exactly the mean of per-agent value functions, which the existing per-group critics already provide. No new critic parameters are needed. Cooperative-critic loss is logged as a tracking metric only. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeamHA.py`: `_coop_encoder_update` (the Phase 0 implementation, lines ~384–683); `train_groups` (dispatch at lines ~734–736; Phase-1 head loop with old-log-prob re-evaluation at lines ~756+); `_get_parameters` (strips GNN from head optimisers in `coop_encoder`); `__init__` validation. |

**Re-evaluation of old log-probs in Phase 1.** Because Phase 0 has already
stepped the GNN before any head update, the collection-time old log-probs no
longer reflect the current policy. The code re-evaluates old log-probs under
the new encoder for every group (including the first) in `coop_encoder` mode
(see `_need_reeval` at `HGTeamHA.py:773`), so the HAPPO factor captures only
head-level drift within Phase 1 and not the encoder change from Phase 0. This
is the correct behaviour for the two-phase theorem (appendix §4).

#### Considered cooperative-critic architectures (legacy)

| Option | Description | New Params | Pros | Cons |
|--------|-------------|-----------|------|------|
| 1. Mean Pooling + MLP | Pool GNN agent embeddings → MLP → scalar $V_\text{coop}$ | ~4K | Independent cooperative value | Training disconnection; pooling lossy |
| 2. Virtual Critic Node | Add a virtual node in the GNN that aggregates all agents' messages | ~2K | Topology-aware pooling within GNN | Invasive GNN change; highest risk |
| 3. Attention Pooling (PMA) | Perceiver-style cross-attention over agent embeddings | ~8K | Learned adaptive weighting | Most params; overkill if linearity holds |
| **4. Value averaging** | $V_\text{coop} = \frac{1}{N_\text{active}} \sum_i V^i$ | **0** | Mathematically exact by linearity; zero risk | Assumes linearity of value under mean reward |

Option 4 is used because it is simplest, adds no hyperparameters, and is
mathematically correct under D2's per-capita reward convention. If cooperative
advantage variance proves too high, Option 3 (attention pooling) is the
identified fallback.

---

## E. Logging & evaluation

### D9. Masked Return Logging

| | |
|---|---|
| **Problem** | Episode return is typically computed as the mean over all agent slots. With inactive slots contributing zero reward, the reported return is diluted: a 20-agent EV episode has the same total reward spread across 20 slots averaged with 0 inactive slots, but a 10-agent episode averages across 10 active + 10 zero slots, reporting half the per-capita return. This makes learning curves misleading and incomparable across episodes with different compositions. |
| **Decision** | Compute `masked_mean_return = sum(return_i * ever_active_i) / n_ever_active` where `ever_active_i` is True if agent $i$ was active at any timestep during the episode. Log this alongside `n_active_agents` as an auxiliary metric. |
| **Alternatives** | (a) Report raw mean over all slots — misleading as discussed. (b) Report total return (sum) — scales with $n$, not comparable. (c) Report per-type returns — useful but doesn't give a single summary metric. |
| **Justification** | The masked mean reflects the average per-active-agent return, which is the quantity the per-agent policy is optimising. Logging `n_active_agents` alongside allows post-hoc analysis of performance as a function of team size. |
| **Files** | `BenchMARL/benchmarl/experiment/logger.py` (`log_evaluation`, `_log_individual_and_group_rewards`) |

---

## F. Cross-cutting invariants

### D13. Per-$N$ Scaling Semantics (reward, loss, entropy, KL)

| | |
|---|---|
| **Problem** | Variable $N$ shows up in at least four places with different defaults: (i) **team reward** aggregation across agents, (ii) **PPO loss** reduction, (iii) **entropy regularisation**, (iv) **KL divergence** reported between old and new policies. Each of these must be made $N$-invariant consistently, or subtle composition bugs appear (e.g., entropy coefficient effectively scales with $N$, causing over-regularisation for large teams). |
| **Decision** | Unify on **per-active-agent** semantics across all four: |
|    | 1. **Reward**: $R_i \leftarrow (1/n_\text{active}) R_\text{team} + R^\text{self}_i$ (D2). |
|    | 2. **Loss**: masked mean over active agents (D6). |
|    | 3. **Entropy**: $H(\pi) = (1/n_\text{active}) \sum_{i \text{ active}} H(\pi_i)$; entropy coefficient is then a per-capita quantity. |
|    | 4. **KL**: $\text{KL} = (1/n_\text{active}) \sum_{i \text{ active}} \text{KL}(\pi^i_\text{old}, \pi^i_\text{new})$; reported as a per-capita metric. |
|    | Coefficients (entropy, value-loss, KL) in config files are **per-capita** coefficients by convention. |
| **Alternatives** | (a) Use sum semantics everywhere — numerically bad for large $N$, and coefficients must be retuned across team sizes. (b) Mix semantics (sum for reward, mean for loss) — creates composition bugs. (c) Leave it to individual components to decide — the status quo before this decision; causes silent drift. |
| **Justification** | Consistent per-active-agent semantics is the minimum discipline required for HGTeam to be hyperparameter-invariant across team sizes. It is also the convention used in the theoretical analysis (Theorem 1, Conjecture 1) and in the per-capita reward normalisation (D2). Every config file should be audited for consistency: if any coefficient drifts from per-capita semantics, transfer across sizes is unreliable. |
| **Files** | All loss-construction sites: `BenchMARL/benchmarl/algorithms/HGTeam.py`, `HGTeamHA.py`, `HGTeamIA.py`, `HGTeamMA.py`; the TorchRL `ClipPPOLoss` subclass in use. Every `yaml` in `BenchMARL/benchmarl/conf/` that defines `entropy_coef`, `kl_coef`, `value_coef` should have a one-line comment stating "per-capita". |
| **Status** | **To audit**. This cross-cutting decision retrofits an invariant onto the codebase; the entropy and KL per-capita semantics are the places most likely to have drifted. A focused audit PR is recommended before training the next set of scalability experiments. |

---

## Summary Table

| ID | Group | Decision | Key Insight |
|----|-------|----------|-------------|
| D1 | A. Env | Max-padded slots + active_mask | Standard variable-length padding pattern |
| D2 | A. Env | VPP reward / $n_\text{active}$ | Per-capita reward is $n$-independent |
| D3 | B. Graph | Edge filtering for inactive nodes | Only complete solution for attention-based GNNs |
| D4 | B. Graph | Actions as edge_attr (Option B) | Single forward pass + natural counterfactual |
| D5 | B. Graph | Zero-embedding safety net | Belt-and-suspenders against numerical residuals |
| D6 | C. Loss | reduction="none" + masked mean | Unbiased gradient over active agents only |
| D7 | C. Loss | Zero advantage + value_target for inactive | Prevents advantage normalisation skew |
| D8 | C. Loss | HAPPO geometric-mean factor | Factor variance independent of $n_\text{active}$ (**Conjecture 1**) |
| D9 | E. Logging | Masked return / $n_\text{ever\_active}$ | Correct per-capita evaluation metric |
| D10 | C. Loss | Per-slot masked advantage normalisation | Preserves inter-agent scale in mixed-motive |
| D11 | D. Encoder | Balanced gradient accumulation (`separate_forward`) | Equal group contribution to shared GNN update |
| D12 | D. Encoder | Cooperative encoder objective (`coop_encoder`) | **Headline**: encoder learns cooperation; heads decide strategy (**Theorem 1**) |
| D13 | F. Cross-cut | Per-$N$ scaling semantics | Per-capita discipline for reward, loss, entropy, KL |
