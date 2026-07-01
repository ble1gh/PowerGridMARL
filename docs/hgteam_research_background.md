# HGTeam Research Background

**Audience.** A future AI agent (or collaborator) joining this project mid-stream
who must advise on research, experiments, and writing toward a JMLR submission on
the HGTeam family of algorithms.

**Goal statement (user's words).** *"Truly scalable variable (and heterogeneous)
team, mixed cooperative-competitive MARL."* In the game-theoretic literature the
precise term for what this project actually targets is **mixed-motive
(general-sum) MARL**: each agent $i$ has its own reward
$R_i = w_c R_\text{team} + w_i R^\text{self}_i$, so the team has a shared
objective **and** per-agent self-interested objectives that can directly
contradict it. This is distinct from "cooperative-competitive" in the
team-vs-team sense (e.g., soccer, competitive SMAC). Throughout this document
the project's setting is called **mixed-motive** unless team-vs-team competition
is specifically meant.

This document assumes you have already skimmed `AGENTS.md`,
`paper/outline.md`, `paper/second_domain.md`, and
`docs/variable_agent_design_choices.md`. Its job is to give you the *theoretical*
and *bibliographic* scaffolding those documents don't, so you can judge design
choices against the literature rather than on their own terms.

---

## 1. Executive summary

HGTeam is an umbrella name for a small family of PPO-style MARL algorithms
(HGTeamIA, HGTeamHA, HGTeamMA — independent / heterogeneous / multi-agent
flavours) built on top of a **heterogeneous graph neural network (GNN) encoder**
that is shared across agents and trained with **CTDE**. Three papers anchor its
intellectual lineage:

1. **HAPPO / HATRPO** (Kuba et al., ICLR 2022, and the HAML paper, JMLR 2023) —
   sequential, monotonic policy updates for heterogeneous agents, derived from a
   *multi-agent advantage decomposition lemma*. Provides the only currently known
   clean theoretical guarantees for cooperative MARL with non-shared policies.
2. **HGT — Heterogeneous Graph Transformer** (Hu et al., WWW 2020) — an attention
   GNN that parameterises messages by **meta-relation triplets** (source type,
   edge type, destination type) and handles node/edge type heterogeneity in a
   principled, scalable way.
3. **SHPPO** (Zhang et al., ICML 2024) — a scalability recipe: keep a
   parameter-shared PPO backbone, but insert a *latent strategy network* that
   generates the weights of a small **heterogeneous layer** per agent, giving
   zero-shot generalisation to unseen team sizes while preserving learning
   efficiency.

HGTeam, as currently implemented, can be read as an attempt to fuse these three
ideas: **HAPPO's credit assignment** (with adaptations for variable $N$), a
**heterogeneous GNN encoder** (the stable default is HGT-shaped but implemented
with PyG's `HeteroConv` + `TransformerConv`), and **parameter sharing within
type** (SHPPO-like in spirit but currently without an explicit latent
heterogeneous-layer mechanism). The opt-in `EdgeWeightedHGT` backend in
`BenchMARL/benchmarl/models/edgeweightedHGT.py` is the publishable edge-aware HGT
path: it uses HGT-style relation transforms with low-rank edge-conditioned
modulation while preserving the existing `HeteroGNN` defaults for current jobs.
The 12
design decisions (D1–D12) plus the ego-entity decisions (D-ego-1 to D-ego-5)
encode the engineering compromises that make this work over a **variable** agent
population on PowerGridworld and SMACv2.

The project's three headline contributions, as currently framed in
`paper/outline.md`, are:

1. A GNN-based encoder that scales to variable and heterogeneous team sizes.
2. A HAPPO-style credit-assignment mechanism adapted for variable $N$ via the
   **geometric-mean HAPPO factor** (D8).
3. A **cooperative encoder objective** (D12 / `encoder_update_mode="coop_encoder"`)
   that reconciles centralised encoder training with decentralised
   heterogeneous-agent losses.

The rest of this document unpacks each of these claims against the literature,
flags what is solid, what is load-bearing but under-theorised, and what is still
missing.

---

## 2. The three foundational papers, in depth

### 2.1 HAPPO and HAML

**Setting.** Fully cooperative Markov game (aka Dec-MDP / common-reward Dec-POMDP)
with a **fixed**, finite set of heterogeneous agents $\mathcal{N}=\{1,\dots,n\}$.
Each agent has its own policy $\pi^i$, and the team optimises the *shared*
expected return $J(\boldsymbol{\pi})$.

**Key lemma (multi-agent advantage decomposition).** For any permutation
$i_{1:n}$ of the agents and any joint policy $\boldsymbol{\pi}$,

$$A^{\boldsymbol{\pi}}_{i_{1:n}}(s, a^{i_{1:n}}) = \sum_{m=1}^{n} A^{\boldsymbol{\pi}}_{i_m}(s, a^{i_{1:m-1}}, a^{i_m}),$$

where $A^{\boldsymbol{\pi}}_{i_m}$ is the **conditional** advantage of agent
$i_m$ given the actions already chosen by agents $i_{1:m-1}$. This is *exact* (not
a bound) and holds for any base policy.

**Consequence.** If each agent in turn improves its own conditional advantage,
**joint** return is guaranteed not to decrease. HATRPO/HAPPO implement this by:

- Sampling a **random permutation** of agents each iteration.
- Updating agents sequentially, each with a PPO/TRPO surrogate multiplied by a
  cumulative importance-sampling **factor** $M^{i_{1:m}} = \prod_{k<m}
  (\pi^{i_k}_{\text{new}} / \pi^{i_k}_{\text{old}})$ that absorbs the effect of
  earlier updates.
- In HAPPO, clipping both the per-agent ratio and the product with the factor.

**HAML (JMLR 2023).** Generalises HAPPO to a broader class of *heterogeneous
mirror-descent* updates (including KL, Bregman divergences, and RL objectives
other than policy gradients), with **monotonic improvement** and **convergence to
a Nash equilibrium** of the fully cooperative game under mild assumptions.

**Why this matters for HGTeam.**

- HAPPO is the *only* currently-accepted framework that gives monotonic
  guarantees for **non-shared** (heterogeneous) policies in cooperative MARL.
- Every theoretical claim HGTeam can realistically make about its
  heterogeneous-agent objective flows through the advantage decomposition lemma
  and HAML's operator view.
- The **load-bearing assumption** is a **fixed** $n$, **fixed agent identities**,
  and **full participation** in each episode. None of these hold in HGTeam.

**Limitations relative to HGTeam's goal.**

| HAPPO assumes | HGTeam needs |
| --- | --- |
| Fixed $n$ across episodes | $n$ varies per episode and within episode (SMACv2 deaths) |
| Each agent samples one action per step | Some agents are dead / masked / absent |
| Common reward for all agents | Mixed-motive: per-agent rewards $R_i = w_c R_\text{team} + w_i R^\text{self}_i$ (PowerGridworld) |
| No parameter sharing required, but identity matters | Type-shared parameters + identity broken by entry/exit |
| IS factor $M^{i_{1:m}}$ is a **product** over fixed $n$ agents | Product over variable $n$ blows up / vanishes → motivates D8 |

Any theoretical result you claim for HGTeam **must be derived explicitly**; you
cannot inherit HAPPO's guarantees by invocation.

### 2.2 HGT — Heterogeneous Graph Transformer

**Setting.** A graph $G = (V, E, \tau_V, \tau_E)$ where each node has a **type**
$\tau(v) \in \mathcal{A}$ and each edge has a **type**
$\phi(e) \in \mathcal{R}$. A *meta-relation* is the triplet
$\langle \tau(s), \phi(e), \tau(t) \rangle$.

**Core mechanism.** At layer $l$, for an edge $e = (s, t)$ of meta-relation
$\langle \tau(s), \phi(e), \tau(t)\rangle$:

- **Typed projections.** $K^l(s) = K\text{-Linear}_{\tau(s)}(h^{l-1}_s)$,
  $Q^l(t) = Q\text{-Linear}_{\tau(t)}(h^{l-1}_t)$,
  $V^l(s) = V\text{-Linear}_{\tau(s)}(h^{l-1}_s)$.
- **Edge-typed attention.** Attention score uses a relation-specific weight
  matrix $W^{ATT}_{\phi(e)}$ and a prior $\mu_{\langle\tau(s),\phi(e),\tau(t)\rangle}$:

  $$\text{ATT-head}(s, e, t) = (K^l(s) W^{ATT}_{\phi(e)}) Q^l(t)^\top \cdot \frac{\mu_{\langle\tau(s),\phi(e),\tau(t)\rangle}}{\sqrt{d}}.$$

- **Edge-typed messages.** $\text{MSG}(s,e,t) = V^l(s) W^{MSG}_{\phi(e)}$.
- **Target-typed aggregation and update.** Multi-head concatenation, softmax over
  all source–edge pairs into $t$, residual + gated update (in HGT the gate is
  typed too).

**Why this matters for HGTeam.**

- HGT is the principled way to handle **multiple edge types** (e.g. electrical
  vs. observation edges in PowerGridworld, or vision-vs-communication in SMACv2
  ego graphs) without collapsing them.
- The $\mu$-prior gives a cheap, learnable modulation of meta-relation importance
  — exactly the kind of inductive bias one wants when the **topology** varies.

**What is *not* HGT in the current code.** `heterognn.py` uses `HeteroConv` +
`TransformerConv`. This:

- Has **per-edge-type convolutions** (good, heterogeneous at the relation level).
- Uses `TransformerConv`'s standard multi-head attention *within* each relation,
  which is **node-type agnostic** (the same $K/Q/V$ projections apply regardless
  of the source type within that relation — inside a `HeteroConv` each relation
  is already type-specific, so this is partially OK).
- Has **no learned meta-relation prior $\mu$** and **no explicit typed
  message/attention weights $W^{ATT}_\phi, W^{MSG}_\phi$ factorised as HGT
  prescribes**.

So the current model is best described as a **relation-typed TransformerConv
stack**, not HGT. This is important for honest framing: either call it
"HGT-inspired" / "relation-typed GAT", or upgrade to true HGT (PyG has
`HGTConv`). The naming "HGTeam" implicitly promises the latter.

### 2.3 SHPPO — Scalable Heterogeneous PPO

**Setting.** Parameter-shared PPO (MAPPO-style) over a team where **zero-shot
generalisation** to unseen team sizes is the target.

**Core mechanism.**

- A **shared backbone** processes each agent's observation.
- A **latent network** produces a per-agent latent $z_i$ from its private
  observation (and, crucially, not from the team size — so the mechanism is size
  agnostic).
- A **heterogeneous layer** is a small MLP whose **weights are generated by
  $z_i$** (a hypernetwork). This gives each agent an effectively unique policy
  head while keeping the number of learnable parameters $O(1)$ in $N$.
- Training is vanilla PPO on the combined module.

**What SHPPO buys you.**

- **Zero-shot scalability**: trained at $n=5$, deployable at $n=20$ without
  retraining, because the shared backbone and latent-to-weights map generalise.
- **Within-team heterogeneity** without an explosion of parameters.
- Compatibility with CTDE (centralised critic sees all latents).

**Limitations.**

- No theoretical improvement guarantees; it rides on MAPPO's (which themselves
  hold only in the shared-policy, fixed-$N$ limit).
- Heterogeneity is *emergent* from the latent network; it cannot be steered by
  known agent types (you get differentiation only where the data forces it).
- Zero-shot transfer quality depends strongly on how well the latent captures
  the right invariances.

**Where HGTeam currently stands vs. SHPPO.**

- HGTeam has **type-level** heterogeneity (different node types in the graph,
  different actor heads per group) but **not instance-level** heterogeneity of
  the SHPPO kind.
- HGTeam's GNN encoder *could* be viewed as an SHPPO-style mechanism: the
  per-agent embedding $h_i^L$ plays the role of SHPPO's latent $z_i$. But the
  *policy head* downstream is **not** a hypernetwork — it is a plain MLP shared
  within a group. So the HGTeam actor is less expressive per agent than SHPPO's.
- If zero-shot transfer to unseen team sizes is a headline claim, SHPPO is the
  most direct baseline and the most honest target to either match or beat.

---

## 3. HGTeam in context

### 3.1 What HGTeam is, in one sentence per layer

- **Encoder.** A shared heterogeneous GNN (`HeteroConv` + `TransformerConv` in
  the current code) operating on a per-step graph whose nodes are agents and
  relevant environment entities and whose edges encode physical, topological, or
  observational relations. Outputs a per-agent embedding $h_i$.
- **Actor.** Per-*group* MLP head that maps $h_i$ (plus optional raw observation
  passthrough) to action distribution parameters. Parameters are **shared within
  group** and **separate across groups** (this is the "heterogeneous" in HGTeam).
- **Critic.** Centralised; consumes per-agent embeddings / global features;
  returns a per-agent value (HAPPO uses per-agent advantages).
- **Objective.** HAPPO-style sequential PPO update per group, with the
  geometric-mean HAPPO factor (D8) replacing the product factor so the surrogate
  neither explodes nor vanishes when $N$ varies.
- **Encoder-update modes.** `accumulated` (default), `separate_forward`, and
  `coop_encoder` — the last treats the encoder as a centralised shared module
  with its own cooperative objective, which is the proposed contribution.

### 3.2 Variable $N$: the actual research problem

The user's phrase "truly scalable variable team" packs several distinct problems
that are worth separating — agents often conflate them and reviewers will want
them disentangled.

| Axis | What it means | HGTeam's answer |
| --- | --- | --- |
| **Train-time variable $N$** | Different episodes have different $N$ | Padding + `active_mask` (D1–D3); graph construction per-step from active set |
| **Within-episode $N$ changes** | Agents enter/die mid-episode | Active-mask propagation; GAE/returns on active agents; D10 balanced accumulation |
| **Test-time unseen $N$** | Deploy on $N$ not seen in training | Parameter sharing within type; GNN size-invariance; encoder handles arbitrary node counts |
| **Heterogeneous types at test time** | New type combinations unseen in training | Currently **not** directly supported without retraining heads; this is a gap |
| **Compositional generalisation** | Combine known types in new counts/ratios | Partially via GNN; depends on encoder architecture |

**Important distinction.** SHPPO addresses axis 3 (unseen $N$) elegantly.
HAPPO addresses neither scaling axis. HGT addresses the *representation* side of
axes 4–5. HGTeam is the first system, to the author's knowledge, that
**simultaneously** tries to attack axes 1–3 with HAPPO-style guarantees. That is
the most defensible novelty claim, but it requires a rigorous theoretical
treatment to land.

### 3.3 "Mixed cooperative-competitive" — precise terminology

The term "mixed cooperative-competitive" in the MARL literature is ambiguous and
typically evokes **team-vs-team** settings (MADDPG on PettingZoo's
`simple_tag`, competitive SMAC against learned opponents). HGTeam's actual
setting is different and more precise: a **mixed-motive / general-sum Markov
game**.

- **PowerGridworld is mixed-motive.** Each agent $i$ has a reward
  $R_i = w_c R_\text{team} + w_i R^\text{self}_i$. The team term $R_\text{team}$
  (VPP production tracking) is shared and cooperative. The self-term
  $R^\text{self}_i$ (voltage violation penalty at the agent's bus, EV charging
  urgency, PV self-utilisation) is agent-specific and can directly **contradict**
  the team objective (an EV may want to charge when the VPP wants it to
  discharge). This is the canonical structure of a sequential social dilemma
  (Leibo et al., AAMAS 2017; Jaques et al., ICML 2019).
- **SMACv2 is common-reward.** All agents share the team win/loss reward; it is
  fully cooperative. The "competitive" element is a scripted game AI, not a
  learned opponent, so it does not constitute team-vs-team competition in the
  game-theoretic sense. SMACv2 is therefore the **common-reward variable-$N$**
  half of the experimental story.

**Recommendation for the paper.** Replace "mixed cooperative-competitive" with
**"mixed-motive"** (or "general-sum with cooperative team reward") throughout.
State the reward decomposition $R_i = w_c R_\text{team} + w_i R^\text{self}_i$
in the problem formulation. This is both more accurate and sharper positioning,
and it converts what I previously flagged as a "biggest mismatch" into a
defensible, under-explored research regime: **mixed-motive MARL at variable
$N$ with per-type credit assignment** is genuinely open.

**Important theoretical consequence.** HAPPO's monotonic-improvement theorem
requires a **common reward**. In a mixed-motive / general-sum setting the
theorem does not directly apply, so any HAPPO-style guarantee must be restated
*per agent's own reward* (Nash-style equilibrium language) rather than *joint
team return* (monotonic-improvement language). This is a harder theoretical
regime, not an easier one, and it is one of the load-bearing technical
challenges the paper must acknowledge. See §4 for the landscape and
`paper/appendix_theorem_proofs.md` for the formal statements.

The team-vs-team competitive extension (self-play on SMACv2 or an adversarial
PowerGridworld scenario) remains valid future work, but it is **not** what the
current artefact implements and should be explicitly scoped out of the first
paper.

---

## 4. Theoretical landscape for variable-$N$ monotonic improvement

This section exists because, if HGTeam wants to claim any HAPPO-style
theoretical guarantee, it needs to navigate the following gaps explicitly rather
than implicitly.

### 4.1 The fixed-$N$ and common-reward dependencies of HAPPO's proofs

HAPPO's monotonic-improvement theorem requires **two** assumptions that HGTeam
violates:

1. **Fixed $N$**: the advantage decomposition lemma is stated over a *fixed*
   ordered set $i_{1:n}$. Variable $N$ turns the state space into a union over
   $n$ of the $n$-agent state spaces, and the policy becomes a distribution
   over policies indexed by $n$.
2. **Common reward**: the lemma decomposes the *joint* advantage
   $A^{\boldsymbol{\pi}}_{i_{1:n}}$ under a single scalar return. In a
   mixed-motive / general-sum game each agent has its own return $J_i$, so the
   joint-advantage object does not exist as a single scalar; the best you can
   state is a per-agent Nash-style equilibrium condition, not a monotonic-
   improvement condition on team return.

The variable-$N$ gap can be partially closed by a reformulation; the mixed-
motive gap is a more fundamental reframing (equilibrium-style guarantees rather
than monotonic-improvement).

Two clean ways to handle this formally:

1. **Type-conditioned policy** $\pi(a \mid o, \text{type})$ shared across
   instances. Then HAPPO updates happen at the *type* level with expected
   advantage averaged over realised instances. This is close to what SHPPO
   does. Monotonic improvement holds *in expectation* under mild technical
   conditions (a projection argument), but with looser constants.
2. **Set-structured MDPs / Entity MDPs** (Jiang et al., 2020; Iqbal et al.,
   2021). The environment is a **set Markov game**. Policies are permutation-
   invariant and size-invariant via a GNN / set transformer. Monotonic
   improvement is recoverable if the GNN satisfies certain Lipschitz conditions
   on the joint entity set.

HGTeam's current framing is closer to (1); the GNN encoder gives it mechanical
access to (2). You should pick a framing and **state the theorem you would
need** before writing the proof.

### 4.2 The geometric-mean HAPPO factor (D8)

D8 replaces the HAPPO factor $M = \prod_{k} r_k$ (where $r_k =
\pi^{i_k}_{\text{new}}/\pi^{i_k}_{\text{old}}$) with
$\tilde M = \left(\prod_k r_k\right)^{1/n}$.

**What this gains.**
- Numerical stability for variable $n$: a product of 16 agent ratios vs. 2 does
  not differ by $\sim 10^6$ in magnitude.
- A single hyperparameter (clip range) that works across team sizes.

**What this loses.**
- The advantage decomposition lemma *exactly* equates the joint surrogate to a
  sum of conditional surrogates when the factor is the **product** of
  importance weights; with a geometric mean this equality breaks. You recover it
  only in the *limit of small updates*, and with an effective clip range that is
  $n$-dependent.
- Equivalently, the geometric mean changes the semantic meaning of the policy
  ratio from "how likely is this joint action under new vs old policies" to "the
  per-agent average likelihood ratio." Monotonic improvement is therefore a
  **conjecture** unless shown.

**What you can honestly claim.**
- An **approximation** argument: for small clip ranges $\epsilon$, the geometric
  mean and the product agree to first order, so HAPPO's guarantees hold up to
  $O(\epsilon^2)$.
- A **trust-region** argument (HAML-style): define the update operator using the
  geometric mean; prove it remains a valid mirror-descent operator. This is
  non-trivial and likely requires an additional weighting.
- **Empirical evidence** that the modified factor behaves no worse than the
  product factor on fixed-$N$ and better on variable-$N$.

A JMLR paper will require at least one of the two theoretical arguments, plus
the empirical evidence.

### 4.3 Cooperative encoder objective (D12 / `coop_encoder`)

This is HGTeam's most original algorithmic contribution after D8. The encoder is
shared across agents (good for data efficiency), but HAPPO requires per-agent
policy parameters (good for heterogeneity). The cooperative encoder objective
reconciles these by:

- Treating the encoder as a **separate module** with its own objective.
- Updating it with a **sum/mean of per-agent PPO surrogates** (a cooperative
  signal) while the per-agent heads are updated sequentially à la HAPPO.

Literature-wise, this is close to:

- **Shared critic, separate actors** MAPPO-style designs (Yu et al., 2022).
- **CTDE encoders with distinct execution heads** as in QMIX / VDN / MA-Trans.
- **Multi-task learning** gradient aggregation (PCGrad, CAGrad) — the coop
  encoder essentially averages / accumulates competing agent gradients.

Novelty claim is strongest if framed as: **"a principled way to combine a shared
CTDE encoder with per-agent HAPPO policy updates, recovering HAPPO's monotonic
improvement for the heads while gaining the sample efficiency of a shared
encoder."** That framing also suggests the natural theorem: conditional on the
encoder at iteration $k$, HAPPO's guarantees for the heads hold step-by-step;
the encoder update is a cooperative PPO step that does not degrade the joint
return in expectation.

---

## 5. Experimental landscape

### 5.1 PowerGridworld VPP control

- **Continuous actions, dense rewards, no agent death, fixed topology per
  scenario, variable $N$ across scenarios.**
- This is a **clean test** of axes 1 (train-time variable $N$) and 3 (test-time
  unseen $N$). It is a **poor test** of axis 2 (within-episode changes) and of
  heterogeneous-type combinations at test time.
- The right ablations here are: per-type counts seen in train vs. test; transfer
  to new grid topologies; performance vs. MAPPO, IPPO, HAPPO (fixed-$N$
  baseline).

### 5.2 SMACv2

- **Discrete actions, sparse rewards, agent death (axis 2), heterogeneous
  units.**
- Tests axes 2 and 4. Tests axis 3 via the 10v10 vs. 20v20 scenarios.
- The **ego-entity GNN** variant (D-ego-1 to D-ego-5) is specifically for the
  partially observable, per-agent view SMACv2 exposes. It is a different
  architectural mode than PowerGridworld's shared-graph; that duality is worth
  discussing explicitly, since it is currently justified by "each domain needs a
  different GNN" without a principled criterion.

### 5.3 Missing domains (optional but useful)

- **PettingZoo simple_spread / simple_tag**: classic heterogeneous MARL
  benchmarks, fixed $N$, but cheap and useful as a sanity check.
- **MPE with heterogeneous agents**: good for ablations.
- **MAgent** or **Neural MMO**: truly variable $N$ with death and birth; tests
  axis 2 under more extreme conditions.
- **A team-vs-team competitive environment** (e.g. SMACv2 self-play) — only
  needed if a future paper wants to extend the mixed-motive framing to
  team-vs-team competition.

---

## 6. Research opportunities and risks

### 6.1 Opportunities

1. **Theorem: variable-$N$ monotonic improvement.** State and prove the
   type-conditioned analogue of HAPPO's monotonic improvement. Even a restricted
   version (e.g. "if agent entry/exit is independent of actions") would be a
   contribution.
2. **Edge-feature-compatible HGT-style encoder.** The current implementation
   intentionally uses `HeteroConv` + `TransformerConv` because grid and critic
   edges need continuous edge attributes. A future opportunity is to recover more
   of vanilla HGT's meta-relation factorization and learned prior $\mu$ without
   losing the edge-feature path. See `docs/hgt_vs_heteroconv_attention.md`.
3. **Hypernetwork policy heads (SHPPO-style).** Add a SHPPO-like hypernetwork
   policy head driven by $h_i$. This converts HGTeam from "type-level
   heterogeneous" to "instance-level heterogeneous" and gives a direct zero-shot
   scalability knob.
4. **Coop encoder ablation.** Ablate `accumulated` vs. `separate_forward`
   vs. `coop_encoder` systematically; this is the cleanest way to isolate D12's
   contribution.
5. **Geometric-mean factor analysis.** Compare product, geometric-mean, and
   arithmetic-mean factors on fixed-$N$ HAPPO benchmarks; see if the geometric
   mean is Pareto-better or merely equivalent in the fixed-$N$ regime. If
   equivalent, the argument reduces to "geometric mean handles variable $N$ for
   free."
6. **Set-MDP framing.** Restate the problem as a set Markov game and derive
   the size-invariance properties of the GNN encoder. This is a clean section
   for a JMLR paper.

### 6.2 Risks / things reviewers will push on

1. **Terminology precision** (see §3.3). Use "mixed-motive" not "mixed
   cooperative-competitive"; define the reward decomposition early. Scope out
   team-vs-team competition for this paper.
2. **"HGT" without vanilla HGTConv** (see §2.2). Be precise: the current encoder
   is HGT-inspired and relation typed, but not PyG `HGTConv`, because edge
   features are load-bearing for grid and critic graphs.
3. **D8 without theory.** Will be flagged by any theoretically-minded reviewer.
   Stated as Conjecture 1 in `paper/appendix_theorem_proofs.md`.
4. **Fixed-$N$ HAPPO inheritance.** Do not quietly reuse HAPPO's guarantees;
   HAPPO requires common reward **and** fixed $N$, neither of which holds in
   PowerGridworld.
5. **Baselines.** SHPPO is the closest scalable baseline; excluding it would be
   a red flag. Likewise, EPC (Evolutionary Population Curriculum, Long et al.,
   2020), MAPPO, IPPO, and at least one attention-based baseline (e.g. CommNet
   or MAAC) are expected.
6. **Train/test protocol for scalability.** A train-at-$N$-test-at-$M$ table is
   expected; make sure the protocol is consistent across methods and that seeds
   are fair.
7. **Compute budget / reproducibility.** JMLR reviewers weight this. Seeds,
   wall-clock, configs, and public-code readiness all matter.

---

## 7. One-page cheat sheet for future agents

Read this first if you only have 90 seconds.

- **Project goal.** Scalable, variable, heterogeneous-team, **mixed-motive**
  cooperative MARL. Team-vs-team competition is out of scope for the first
  paper.
- **Reward structure.** PowerGridworld: $R_i = w_c R_\text{team} + w_i
  R^\text{self}_i$ (mixed-motive / general-sum). SMACv2: common reward
  (variable-$N$ cooperative).
- **Algorithm.** HGTeam = HAPPO heads + shared heterogeneous graph encoder
  (current code: edge-aware `HeteroConv` + `TransformerConv`, HGT-inspired but
  not vanilla `HGTConv`) + cooperative encoder objective.
- **Key novelties (ordered by defensibility).**
  1. Cooperative encoder objective (D12) — **headline**; Theorem 1 in appendix.
  2. Geometric-mean HAPPO factor for variable $N$ (D8) — Conjecture 1.
  3. Heterogeneous graph encoder adapted to variable and type-diverse teams.
- **Core environments.** PowerGridworld (continuous, dense, mixed-motive);
  SMACv2 (discrete, sparse, common-reward with death).
- **Core invariants.** `active_mask` is the single source of truth for "agent
  exists this step"; every surrogate/objective must respect it.
- **Biggest literature debts.** HAPPO/HAML (theory), HGT (encoder design),
  SHPPO (scalability baseline), MAPPO/IPPO (baselines), sequential social
  dilemma literature (mixed-motive framing).
- **Biggest open theoretical question.** Does HAPPO's per-agent credit
  assignment carry to variable $N$ with the geometric-mean factor? See
  Conjecture 1.
- **Biggest framing risk.** HAPPO's theorems require common reward; don't
  inherit them silently when running PowerGridworld. State equilibrium-style
  claims for mixed-motive, monotonic-improvement claims for SMACv2.
- **Before submitting anything.** Decide: (i) final paper terminology for the
  edge-aware relation-typed encoder (do not call it vanilla `HGTConv`); (ii)
  finalise the mixed-motive problem-formulation language in the paper; (iii)
  complete Theorem 1 and Conjecture 1 writeups in
  `paper/appendix_theorem_proofs.md`.

---

## 8. References (for the future agent)

- Kuba, J. G. et al., *Trust Region Policy Optimisation in Multi-Agent
  Reinforcement Learning*, ICLR 2022. (HATRPO/HAPPO.)
- Kuba, J. G. et al., *Heterogeneous-Agent Mirror Learning*, JMLR 2024. (HAML.)
- Hu, Z. et al., *Heterogeneous Graph Transformer*, WWW 2020. (HGT.)
- Zhang, Z. et al., *Scalable and Heterogeneous PPO for Cooperative MARL*, 2024.
  (SHPPO.)
- Yu, C. et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games*, NeurIPS 2022. (MAPPO.)
- Ellis, B. et al., *SMACv2: An Improved Benchmark for Cooperative MARL*, 2023.
- Iqbal, S. et al., *Randomized Entity-wise Factorization for Multi-Agent RL*,
  ICML 2021. (Entity MDPs.)
- Long, Q. et al., *Evolutionary Population Curriculum for Scaling MARL*, ICLR
  2020. (EPC; scalability baseline.)

---

*Author's note for future agents.* Keep the scope of the first JMLR paper
narrow: **cooperative, variable and heterogeneous $N$, with a principled
encoder-plus-HAPPO contribution**, and a clear experimental story across
PowerGridworld and SMACv2. Every claim this document flags as "needs theory" or
"needs rescoping" should be resolved before writing the final draft.
