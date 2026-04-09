# Design Choices for Variable Agent Counts

This document catalogues every design decision made to handle variable numbers
of agents across episodes.  Each section states the **problem**, the
**decision** taken, the **alternatives** considered, and the **justification**.

---

## Environment Layer

### D1. Max-Padded Agent Slots with Binary Active Mask

| | |
|---|---|
| **Problem** | TorchRL and BenchMARL expect a fixed tensor layout across episodes, but agent counts vary: EVs ∈ [10, 20], PVs = 25, Storage ∈ [5, 8]. |
| **Decision** | Allocate the maximum number of slots per type (20 EV, 25 PV, 8 Storage = 53 total). At each episode reset, sample a random count per type and set a per-slot boolean `active_mask`. Inactive slots receive zero observations, zero actions, and zero rewards. |
| **Alternatives** | (a) Dynamic tensor shapes per episode — incompatible with vectorized batching in TorchRL's replay buffers and collectors. (b) Separate policies per agent count — combinatorial explosion. |
| **Justification** | Padding is the standard approach for variable-length sequences (cf. NLP). The active mask lets every downstream component selectively ignore inactive slots without architectural changes to TorchRL. |
| **Files** | `BenchMARL/benchmarl/environments/PowerGridworldVariable/common.py` |

### D2. Per-Agent VPP Reward Normalization (S2a)

| | |
|---|---|
| **Problem** | The VPP (Virtual Power Plant) production-tracking reward is a single team-level scalar broadcast to every active agent. When that scalar is the same regardless of team size, the total reward summed across agents scales linearly with *n*_active, creating a confound: the critic and advantage estimator observe systematically different reward magnitudes for different episode compositions. |
| **Decision** | Divide the VPP reward by *n*_active before broadcasting, so each agent receives reward / *n*. The value function targets per-agent expected return rather than a share of a size-dependent total. |
| **Alternatives** | (a) Leave reward un-normalized — critic must implicitly learn the *n*-dependence. (b) Normalize advantages instead of rewards — doesn't fix the value target scale. (c) Normalize at the critic output — more complex, entangles architecture with reward structure. |
| **Justification** | Normalizing at the source is simplest and keeps the reward semantics clean: each agent's reward reflects its *per-capita* contribution to the team objective, independent of how many agents happen to be present. This is analogous to how cooperative game theory divides coalition value by coalition size (Shapley, 1953). |
| **Files** | `PowerGridworld/gridworld/multiagent_env.py` (`reward_transform`) |

---

## GNN / Message Passing Layer

### D3. Edge Filtering — Remove Edges to/from Inactive Nodes

| | |
|---|---|
| **Problem** | The heterogeneous GNN (TransformerConv) sends messages along all edges, including those involving inactive (zero-padded) agent nodes. Inactive nodes emit non-trivial messages because TransformerConv has learned bias terms ($\alpha_{i,j} \propto \text{softmax}(... + \mathbf{b})$), so even zero-featured nodes produce non-zero attention scores and pollute active nodes' representations. |
| **Decision** | In `_tensordict_to_hetero_data()`, remove all agent–agent edges where either endpoint is inactive. This is done per edge type by indexing into the flattened `active_mask` and keeping only edges where `src_active AND dst_active`. |
| **Alternatives** | (a) Attention masking inside TransformerConv (set attention weights to −∞) — PyG's TransformerConv doesn't support a native attention mask parameter. (b) Zero the node features (already done) — insufficient because bias terms in Q/K projections still produce non-zero attention contributions. (c) Post-hoc zeroing of inactive embeddings — doesn't prevent corruption of *active* node embeddings during aggregation. |
| **Justification** | Edge filtering is the only complete solution: it prevents inactive nodes from participating in message passing at all. With edges removed, TransformerConv's attention softmax is computed only over active neighbours, and active nodes' representations are uncontaminated. |
| **Files** | `BenchMARL/benchmarl/models/heterognn.py` (`_tensordict_to_hetero_data`, active mask edge filtering) |

### D4. Actions as GNN Edge Features (Option B) — Replacing Flat Action Concatenation

| | |
|---|---|
| **Problem** | The original centralised critic used `_compute_augmented_obs()` to flat-concatenate all 53 agents' actions into each agent's observation. This creates 53 zero-padded action slots, most of which are noise from inactive agents, and scales the critic input dimension linearly with max agent count. |
| **Decision** | Instead of concatenation, pass other agents' actions as **edge attributes** on the GNN's TransformerConv interaction edges. For edge (j → i), `edge_attr = a_j`. Self-loop edges carry `edge_attr = 0` so agent *i* never sees its own action, yielding V(s, a_{−i}) counterfactual semantics. Combined with D3 (edge filtering), inactive agents' actions never enter the computation. |
| **Alternatives** | (a) Per-agent counterfactual forward passes — one GNN pass per agent to compute V(s, a_{−i}), requiring 53× compute. (b) Keep flat concatenation but mask inactive slots — still O(max_agents) input dim, wastes capacity on padding. |
| **Justification** | Option B achieves the same counterfactual semantics as Option A in a single forward pass by leveraging TransformerConv's native `edge_dim` support: $\alpha_{i,j} = \text{softmax}\left(\frac{(W_3 x_i)^T (W_4 x_j + W_6 e_{ij})}{\sqrt{d}}\right)$. The edge attribute $e_{ij} = a_j$ naturally conditions attention on the sender's action. This reduces compute from O(*n*) forward passes to O(1), and the critic's input spec no longer depends on `max_agents`. |
| **Files** | `BenchMARL/benchmarl/models/heterognn.py` (action collection in `_collect_action_edge_features`, edge attr construction in `_tensordict_to_hetero_data`); `BenchMARL/benchmarl/algorithms/HGTeam.py` (`_compute_other_actions_dim`) |

### D5. Zero-Embedding Safety Net for Inactive Agents

| | |
|---|---|
| **Problem** | Even with edge filtering (D3), numerical residuals from layer norms, bias terms in output projections, or floating-point accumulation could leave inactive agent embedding slots with small non-zero values. |
| **Decision** | After the GNN forward pass, multiply each agent's output embedding by its `active_mask` value (0 or 1). This is a belt-and-suspenders safeguard applied in `_forward()`. |
| **Alternatives** | Rely solely on edge filtering — likely sufficient in practice, but hard to guarantee across all layer types and future architecture changes. |
| **Justification** | Cheap (one element-wise multiply) and eliminates any residual leakage. Prevents inactive embeddings from affecting downstream MLP/Transformer heads. |
| **Files** | `BenchMARL/benchmarl/models/heterognn.py` (`_forward`, post-GNN active_mask multiply) |

---

## Policy Gradient / Loss Layer

### D6. Loss Reduction = "none" with Masked Mean

| | |
|---|---|
| **Problem** | TorchRL's `ClipPPOLoss` aggregates (mean or sum) over all elements including inactive agent slots. Inactive slots have zero advantages (see D7) but potentially non-zero log-probs and entropy, which would corrupt gradient estimates. A standard mean over all 53 slots dilutes the gradient by including ~50% inactive agents. |
| **Decision** | Initialize `ClipPPOLoss` with `reduction="none"` to obtain per-element losses. In `HGTeamLoss.forward()`, apply masked reduction: `(loss * active_mask).sum() / active_mask.sum()`. This computes the mean loss over active agents only. |
| **Alternatives** | (a) Standard mean reduction — dilutes gradients by 53/n_active factor, biased toward smaller groups. (b) Sum reduction — gradient magnitude scales with n_active. (c) Filter tensordict to only active agents before loss — complex reshape, breaks TorchRL's module assumptions. |
| **Justification** | Masked mean is gradient-unbiased: each active agent contributes equally regardless of how many inactive padding slots exist. Combined with D7 (zero advantages), the policy gradient $\nabla \log \pi(a|s) \cdot A$ is exactly zero for inactive slots even before masking, but masking also handles the entropy and critic loss terms. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeam.py` (`HGTeamLoss.forward`, masked reduction), (`_get_loss`, `reduction="none"`) |

### D7. Zero Advantages and Value Targets for Inactive Agents

| | |
|---|---|
| **Problem** | GAE computes advantages from TD errors $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. Even with zero rewards, the critic may predict non-zero values for inactive slots (e.g., due to shared parameters processing the zero observation into a non-zero value), producing non-zero advantages that inject noise into the policy gradient. |
| **Decision** | After GAE computation, explicitly zero out `advantage[inactive] = 0` and set `value_target[inactive] = state_value[inactive].detach()`. The second assignment ensures the critic loss for inactive slots is zero (target equals prediction). |
| **Alternatives** | (a) Train the critic to predict zero for inactive slots — would require an auxiliary loss or architectural constraint. (b) Mask only in the loss (D6) — works for loss, but non-zero advantages would still affect advantage normalization statistics. |
| **Justification** | Direct zeroing is simple and correct. It prevents inactive agents from (i) contributing policy gradient signal and (ii) skewing the advantage normalization, which divides by the standard deviation of all advantages including inactive slots. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeam.py` (`process_batch`, D7 zeroing), `BenchMARL/benchmarl/algorithms/HGTeamHA.py` (`process_batch`, D7 zeroing) |

### D8. HAPPO Factor: Geometric-Mean Normalization (Option A)

| | |
|---|---|
| **Problem** | HAPPO computes a per-group importance weight factor $F_g = \prod_{i \in g} \frac{\pi_\text{new}(a_i \mid s)}{\pi_\text{old}(a_i \mid s)}$ and passes it to subsequent groups as an advantage multiplier. In log-space, this is $\log F_g = \sum_i \log r_i$. With variable agent counts, the sum has a different number of terms per sample (10–20 for EVs, 5–8 for Storage). The variance of $\log F_g$ grows linearly with *n*_active (and the variance of $F_g$ grows exponentially), causing: (i) systematic clipping bias for larger groups (the ratio always hits the clip boundary), and (ii) inconsistent factor magnitude passed to downstream groups across samples with different team compositions. |
| **Decision** | Divide `log_group_ratio` by *n*_active, making the factor the **geometric mean** of per-agent ratios: $F_g = \left(\prod_i r_i\right)^{1/n}$. This normalizes the factor's variance to be independent of group size. |
| **Alternatives** | (a) No normalization (original HAPPO) — factor variance is *n*-dependent, breaks the assumption that all samples in a batch have comparable factor magnitudes. (b) Per-agent clipping before summing — more conservative, limits each agent's contribution to $[\log(1-\epsilon), \log(1+\epsilon)]$, but variance of the sum still grows with *n*. (c) Hybrid: per-agent clip + normalize — most controlled but furthest from the original algorithm. |
| **Justification** | Standard HAPPO (Zhong et al., 2024) assumes a fixed agent count. With variable counts, the joint ratio's magnitude becomes a function of *n* rather than policy divergence. The geometric mean preserves the monotone improvement guarantee locally (if all per-agent ratios are near 1, the mean is near 1) while ensuring the factor has consistent scale regardless of how many agents are active. This is mathematically equivalent to raising the standard HAPPO factor to the power $1/n$, which compresses but does not lose information about policy change magnitude. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeamHA.py` (`train_groups`, geometric-mean normalization of `log_group_ratio`) |

---

## Logging & Evaluation

### D9. Masked Return Logging (S1a / S1b)

| | |
|---|---|
| **Problem** | Episode return is typically computed as the mean over all agent slots. With inactive slots contributing zero reward, the reported return is diluted: a 20-agent EV episode has the same total reward spread across 20 slots averaged with 0 inactive slots, but a 10-agent episode averages across 10 active + 10 zero slots, reporting half the per-capita return. This makes learning curves misleading and incomparable across episodes with different compositions. |
| **Decision** | Compute `masked_mean_return = sum(return_i * ever_active_i) / n_ever_active` where `ever_active_i` is True if agent *i* was active at any timestep during the episode. Log this alongside `n_active_agents` as an auxiliary metric. |
| **Alternatives** | (a) Report raw mean over all slots — misleading as discussed. (b) Report total return (sum) — scales with *n*, not comparable. (c) Report per-type returns — useful but doesn't give a single summary metric. |
| **Justification** | The masked mean reflects the average per-active-agent return, which is the quantity the per-agent policy is optimizing. Logging `n_active_agents` alongside allows post-hoc analysis of performance as a function of team size. |
| **Files** | `BenchMARL/benchmarl/experiment/logger.py` (`log_evaluation`, `_log_individual_and_group_rewards`) |

### D10. Per-Slot Masked Advantage Normalization (S8a)

| | |
|---|---|
| **Problem** | TorchRL's built-in advantage normalization (`_standardize`) includes inactive (zero-padded) advantages in its statistics. After D7 zeros inactive advantages, a partially-active slot (e.g., EV slot 15, active ~45% of the time) has its per-slot mean pulled toward zero and its std compressed. When the normalization divides by this compressed std, the few real advantage values are **inflated**, giving rarely-active slots disproportionately large gradient signals. Meanwhile, always-active slots (PV agents) get relatively dampened gradients. |
| **Decision** | Disable TorchRL's `normalize_advantage` and implement per-agent-slot masked normalization in `process_batch()`, after D7 zeroing. For each slot, compute mean and std over only the batch entries where that slot is active. Re-zero inactive entries after normalization since $(0 - \mu)/\sigma \neq 0$. |
| **Alternatives** | (a) Global masked normalization (single mean/std across all active entries) — simpler, but erases inter-agent advantage scale differences that reflect the heterogeneous reward structure. In a mixed cooperative-competitive environment, agent types with higher marginal impact should retain larger advantage magnitudes. (b) Leave TorchRL's normalization enabled and accept the zero-contamination — biases policy gradient toward rarely-active slots. (c) No advantage normalization at all — possible but can lead to training instability from raw advantage scale. |
| **Justification** | Per-slot normalization preserves the natural advantage scale hierarchy across agent types (EVs, PVs, Storage) while ensuring each slot has zero-mean unit-variance advantages within its own active-entry distribution. This is critical in mixed cooperative-competitive settings: the competitive dimension (agents competing for grid capacity) requires that gradient magnitudes reflect actual reward-signal differences. Cross-agent normalization would artificially equalize these signals, handicapping the agent type that should learn faster. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeam.py` (`process_batch`, `_get_loss`), `BenchMARL/benchmarl/algorithms/HGTeamHA.py` (`process_batch`, `_get_loss`) |

---

## GNN Encoder Update Layer

### D11. Balanced GNN Gradient Accumulation (`separate_forward` Mode)

| | |
|---|---|
| **Problem** | In HAPPO's `separate_forward` encoder update mode, the shared actor GNN is frozen during per-group PPO loops, then unfrozen and updated separately afterwards. The original implementation used only the last group's data for this update, meaning the GNN gradient was 100% biased toward whichever group happened to be last in the random HAPPO ordering. Over many iterations the bias averages out, but within any single iteration one agent type dominates the encoder update — problematic because GNN parameters serving different edge types (e.g., EV↔Storage interaction convolutions) only receive gradient when the relevant group is selected. |
| **Decision** | Accumulate gradients from all groups before stepping: for each group, sample a minibatch, forward through that group's loss, and call `(loss / n_groups).backward()`. The `1/n_groups` scaling ensures the accumulated `.grad` tensors equal the mean gradient across groups. A single `optimizer.step()` then applies one coherent Adam update. |
| **Alternatives** | (a) Last-group only (original) — simplest, zero extra compute, but systematically biased. (b) Round-robin cycling — zero extra compute, but at any single iteration the gradient is still 100% from one group; Adam's momentum smooths this over time but edge-type-specific parameters suffer stale gradients for 2/3 of iterations. (c) Loss averaging (sum losses, one backward) — mathematically identical to gradient accumulation, but requires holding all 3 computation graphs simultaneously in GPU memory (~3× peak memory). (d) Weighted gradient accumulation (weight by group size) — adds a hyperparameter; unclear whether larger groups deserve more gradient signal in a mixed cooperative-competitive setting. |
| **Justification** | Gradient accumulation (Option B) achieves identical results to loss averaging with 1/3 the peak memory (only one group's graph alive at a time). Each group contributes equally regardless of agent count, consistent with D10's principle of preserving inter-agent scale differences. The uniform `1/n_groups` weighting treats each group's learning signal as equally important, avoiding implicit prioritization of larger groups. The cost — 3 forward+backward passes instead of 1 — is negligible since the GNN update runs once per training iteration while per-group PPO loops run `n_optimizer_steps × n_minibatches` passes each. |
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeamHA.py` (`train_groups`, section 5: `separate_forward` branch) |

---

## Cooperative-Competitive Encoder Layer

### D12. Cooperative Encoder Objective (`coop_encoder` Mode)

| | |
|---|---|
| **Problem** | The environment has mixed cooperative-competitive structure: all agents share a team-level VPP-tracking reward (cooperative), but also receive local rewards — voltage penalties, PV utilization bonuses, EV urgency costs (competitive in the sense that agents compete for limited grid capacity). Both `accumulated` and `separate_forward` modes train the shared actor GNN using per-group HAPPO losses, meaning the encoder gradient is a mixture of each group's selfish advantage. There is no explicit cooperative signal — the encoder happens to learn cooperation through the shared VPP component of per-group advantages, but this signal is diluted by local components and entangled with HAPPO's sequential update dynamics. |
| **Decision** | Add `coop_encoder` mode with two-phase training: **Phase 1** (identical to `separate_forward`) freezes the GNN and updates per-group heads with HAPPO factor propagation. **Phase 2** unfreezes the GNN and trains it with a cooperative PPO surrogate that explicitly optimises the mean return across all active agents. |
| **Alternatives** | (a) `accumulated` mode (D11-variant) — GNN participates in all 3 group backward passes at lr/3 with gradient accumulation. Simple, but GNN gradient = (1/3)Σ∇L_g, which is a Pareto signal (average of selfish gradients), not a cooperative gradient. Mathematically: $(1/G)\sum_g \nabla_\phi L_g \neq \nabla_\phi L(\text{mean}_i(r^i))$ in general. (b) `separate_forward` (D11) — balanced gradient accumulation across groups, same Pareto-not-cooperative distinction. (c) Separate cooperative critic network — adds new parameters (pooling head + MLP), requires its own GAE and optimiser. More flexible but higher complexity and risk of critic/encoder disconnection. (d) Value averaging with cooperative PPO surrogate (**chosen, Option 4**) — zero new parameters, mathematically exact cooperative value function by linearity. |
| **Justification** | The core insight is the cooperative-competitive decomposition: individual actor heads decide how much each agent cooperates or competes, while the shared GNN encoder should represent the cooperative structure of the environment. By training the encoder with mean return, it learns representations that benefit all agents on average. Heads, updated separately with HAPPO, then decide per-agent strategy using these cooperative embeddings. The mathematical foundation is linearity of expectation and GAE: $V_{\text{coop}}(s) = \frac{1}{N_{\text{active}}} \sum_i V^i(s)$ requires **zero new parameters** — the cooperative value is exactly the mean of per-agent values already computed by the existing shared critic. Similarly, cooperative advantages = mean of per-agent raw advantages $(V^i_{\text{target}} - V^i)$. The cooperative PPO surrogate uses a global importance ratio $r = \exp\left(\frac{1}{N_{\text{active}}} \sum_i \log \pi_{\text{new}}(a_i \mid o_i) - \frac{1}{N_{\text{active}}} \sum_i \log \pi_{\text{old}}(a_i \mid o_i)\right)$, i.e., the geometric mean of per-agent ratios. The encoder loss is $-L_{\text{coop}}^{\text{clip}} + \beta_{\text{VIB}} \cdot D_{\text{KL}}$. Cooperative critic loss ($\|V_{\text{coop}} - V_{\text{coop}}^{\text{target}}\|^2$) is logged as a tracking metric only — it measures the existing per-agent critics' accuracy on the cooperative value, not a training signal. |

#### Considered cooperative critic architectures

| Option | Description | New Params | Pros | Cons |
|--------|-------------|-----------|------|------|
| 1. Mean Pooling + MLP | Pool GNN agent embeddings → MLP → scalar $V_{\text{coop}}$ | ~4K | Independent cooperative value | Training disconnection; pooling lossy |
| 2. Virtual Critic Node | Add a virtual node in the GNN that aggregates all agents' messages | ~2K | Learns topology-aware pooling within GNN | Invasive GNN change; highest risk |
| 3. Attention Pooling (PMA) | Perceiver-style cross-attention over agent embeddings | ~8K | Learned adaptive weighting | Most params; overkill if linearity holds |
| **4. Value Averaging** | $V_{\text{coop}} = \frac{1}{N_{\text{active}}} \sum_i V^i$ | **0** | **Mathematically exact by linearity; zero risk** | Assumes linearity of value under mean reward |

Option 4 was chosen because it is the simplest, adds no hyperparameters, and is mathematically correct: by linearity of expectation, the mean of per-agent values (each trained on per-agent rewards that include the per-capita VPP component) exactly equals the value of the mean return. If cooperative advantage variance proves too high in practice, Option 3 (Attention Pooling) is the identified fallback.

| | |
|---|---|
| **Files** | `BenchMARL/benchmarl/algorithms/HGTeamHA.py` (`train_groups`, section 5: `coop_encoder` branch; `_get_parameters` strips GNN for coop_encoder; validation in `__init__`) |

---

## Summary Table

| ID | Layer | Decision | Key Insight |
|----|-------|----------|-------------|
| D1 | Env | Max-padded slots + active_mask | Standard variable-length padding pattern |
| D2 | Env | VPP reward / *n*_active | Per-capita reward is *n*-independent |
| D3 | GNN | Edge filtering for inactive nodes | Only complete solution for TransformerConv |
| D4 | GNN | Actions as edge_attr (Option B) | Single forward pass + natural counterfactual |
| D5 | GNN | Zero-embedding safety net | Belt-and-suspenders against numerical residuals |
| D6 | Loss | reduction="none" + masked mean | Unbiased gradient over active agents only |
| D7 | Loss | Zero advantage + value_target for inactive | Prevents advantage normalization skew |
| D8 | Loss | HAPPO geometric-mean factor | Factor variance independent of *n*_active |
| D9 | Logging | Masked return / *n*_ever_active | Correct per-capita evaluation metric |
| D10 | Loss | Per-slot masked advantage normalization | Preserves inter-agent scale in mixed coop-competitive |
| D11 | GNN | Balanced gradient accumulation (`separate_forward`) | Equal group contribution to shared GNN update |
| D12 | GNN | Cooperative encoder objective (`coop_encoder`) | Encoder learns cooperation; heads decide strategy |
