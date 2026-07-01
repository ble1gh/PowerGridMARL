# Appendix: Theoretical Results for HGTeam

**Status.** Working draft to support `paper/outline.md`. Theorem 1
(cooperative encoder soundness) and Theorem 2 (Phase-1-after-Phase-0
first-order improvement) are stated and proved. Conjecture 1 (geometric-mean
HAPPO factor) is stated with a first-order argument and a clear description
of what remains open, but its empirical necessity is now itself in question
(see §4.5 ablations). References to the main paper use the outline section
numbers (`paper/outline.md`).

**Correction.** An earlier revision of this appendix claimed Formulation A
was biased. That claim was wrong — both A and B are unbiased estimators of
the per-capita cooperative gradient; they differ in variance and in
clipping behaviour (§3.4, §3.5). The "bias vs. unbiased" framing has been
replaced everywhere with "variance" and "trust-region tightness."

**Contents.**
1. Notation and problem setup
2. Background: PPO surrogate, HAPPO sequential decomposition
3. Theorem 1 (Cooperative Encoder Update is Sound)
   - Formulation A: single-scalar (mean advantage + geometric-mean ratio)
   - Formulation B: per-agent (per-agent advantages + per-agent ratios) — **preferred**
   - Equivalence Lemma: when A and B coincide
   - Proposition 2 / 2': both A and B are unbiased; B has lower variance and tighter per-agent clipping
   - Formulation A vs. B under the clip
   - Formulation C: single-agent joint-policy view — how it relates to A and B
   - Statement of Theorem 1
4. Phase-1-after-Phase-0 soundness: HAPPO heads at the updated encoder
   - 4.1 Setting and assumptions
   - 4.2 First-order improvement (Theorem 2)
   - 4.3 What the HAPPO factor does and does not do
   - 4.4 Stale advantage correction (a subtle mismatch)
   - 4.5 Empirical diagnostics to test whether HAPPO helps
5. Conjecture 1 (Variable-$N$ HAPPO with Geometric-Mean Factor)
6. Open questions

---

## 1. Notation and problem setup

We model the environment as a **variable-composition set Markov game** following
`paper/outline.md` §III.A.

**State, observation, action.** Let $\mathcal{S}$ be the global state space.
There is a finite set of agent **types** $\mathcal{G}$ (e.g.
$\{\text{EV}, \text{PV}, \text{Storage}\}$). At the start of each episode an
active set $\mathcal{N}_t \subseteq \{1, \dots, N_\text{max}\}$ is sampled and
may change over time. Each agent $i \in \mathcal{N}_t$ has observation
$o_i \in \mathcal{O}$ and action $a_i \in \mathcal{A}$. Let
$\boldsymbol{a} = (a_i)_{i \in \mathcal{N}_t}$ be the joint action.

**Rewards.** Each agent has a per-agent reward
$$R_i(s, \boldsymbol{a}) = w_c R_\text{team}(s, \boldsymbol{a}) + w_i R^\text{self}_i(s, a_i),$$
with $w_c \ge 0, w_i \ge 0$. When $w_i = 0 \,\forall i$ the game reduces to the
common-reward Dec-POMDP of HAPPO. When $w_c = 0$ it reduces to independent
MDPs. PowerGridworld lives in between; SMACv2 has $w_i = 0$.

**Policies and parameters.** Each agent $i$ of type $g = g(i)$ has a policy
$\pi_i(a_i \mid o_i; \theta^E, \theta^H_g)$ with **shared encoder parameters**
$\theta^E$ (the heterogeneous graph transformer) and **group-shared head
parameters** $\theta^H_g$. Write $\theta = (\theta^E, \{\theta^H_g\}_{g \in \mathcal{G}})$.

We train under CTDE: per-group critics $V^i(s; \phi_g)$ with $g = g(i)$; access
to the global state $s$ at training time.

**Per-capita objective.** Throughout this appendix and consistent with D2, D13,
we work with the **per-capita return**
$$J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \gamma^t \bar R(s_t, \boldsymbol{a}_t)\right], \qquad \bar R(s, \boldsymbol{a}) \;=\; \frac{1}{N_\text{active}(s)} \sum_{i \in \mathcal{N}(s)} R_i(s, \boldsymbol{a}).$$
Under the per-capita convention, by linearity of expectation and of GAE,
$$V_\text{coop}(s) \;=\; \frac{1}{N_\text{active}(s)} \sum_i V^i(s), \qquad \hat A_\text{coop}(s, \boldsymbol{a}) \;=\; \frac{1}{N_\text{active}(s)} \sum_i \hat A_i(s, \boldsymbol{a}),$$
which is the justification for the zero-new-parameters cooperative critic in
D12.

**Policy ratios.** For per-agent ratios we write $\rho_i(\theta) = \pi_i(a_i \mid o_i; \theta) / \pi_i(a_i \mid o_i; \theta_\text{old})$.
The clip operator is $\text{clip}(x, 1 \pm \epsilon) = \min(\max(x, 1-\epsilon), 1+\epsilon)$.

**Active-count convention.** Throughout, sums are over active agents only;
$N_\text{active}$ varies per state. We suppress the dependence on $s$ for
readability when context is clear.

---

## 2. Background: PPO surrogate and HAPPO sequential decomposition

### 2.1 Single-agent PPO

For a single-agent policy, the clipped PPO surrogate of Schulman et al. (2017)
is
$$L^{\text{PPO}}(\theta) = \mathbb{E}\left[\min\bigl(\rho(\theta) \hat A,\; \text{clip}(\rho, 1 \pm \epsilon) \hat A\bigr)\right].$$
Under mild assumptions $\nabla_\theta L^{\text{PPO}}|_{\theta = \theta_\text{old}} = \mathbb{E}[\nabla_\theta \log \pi(a \mid s; \theta) \hat A]$, so at initialisation $L^{\text{PPO}}$ is a first-order surrogate for the policy-gradient objective.

### 2.2 HAPPO sequential decomposition (Kuba et al. 2022)

In the **common-reward fixed-$N$** setting, the multi-agent advantage
decomposition lemma says: for any permutation $i_{1:n}$ of the agents and any
joint policy $\boldsymbol{\pi}$,
$$A^{\boldsymbol{\pi}}_{i_{1:n}}(s, \boldsymbol{a}) \;=\; \sum_{m=1}^n A^{\boldsymbol{\pi}}_{i_m}\!\bigl(s, a^{i_{1:m-1}}, a^{i_m}\bigr).$$
HAPPO exploits this by updating agents sequentially with a per-agent PPO
surrogate multiplied by a cumulative importance factor
$M^{i_{1:m}}(\theta) = \prod_{k<m} \rho_{i_k}(\theta)$.

This appendix does not inherit HAPPO's theorem; we instead build analogous
results for the mixed-motive variable-$N$ setting from scratch.

---

## 3. Theorem 1 — Cooperative Encoder Update is Sound

**Setting.** Fix any stage of training with current parameters
$\theta_\text{old} = (\theta^E_\text{old}, \{\theta^H_{g,\text{old}}\})$. We
consider one update to the encoder $\theta^E$ while holding all heads frozen
(Phase 1 of D12, encoder-first ordering).

**Assumptions (standard PPO).**

- **A1** (smoothness). $J(\theta)$ is differentiable in $\theta^E$ with
  Lipschitz gradient; per-agent log-policies are differentiable in $\theta^E$
  almost everywhere.
- **A2** (small-clip regime). The clip range $\epsilon$ is small enough that
  all per-agent ratios $\rho_i$ at the updated parameters lie within
  $[1-\epsilon, 1+\epsilon]$ for the sampled trajectories. Equivalently, the
  update is within the PPO trust region.
- **A3** (bounded advantages). Per-agent advantages $\hat A_i$ are bounded
  with finite variance.
- **A4** (consistent baselines). Per-group critics $V^i$ are unbiased
  estimators of $\mathbb{E}\bigl[\sum_t \gamma^t R_i \mid s\bigr]$, or are
  empirically consistent under GAE.

Under A2 the PPO clip is not active, so $\min(\rho_i \hat A_i, \text{clip}(\rho_i) \hat A_i) = \rho_i \hat A_i$; this lets us analyse the surrogate as a plain importance-weighted objective in a neighbourhood of $\theta_\text{old}$.

We analyse **two** cooperative-encoder surrogates and their relationship.

### 3.1 Formulation A (single-scalar: mean advantage + geometric-mean ratio)

The single-scalar formulation — not the one implemented in `HGTeamHA.py`,
which uses Formulation B below, but important as a theoretical contrast — is,
per sample,
$$L^{\text{coop}}_A(\theta^E) \;=\; \mathbb{E}\left[\min\bigl(\bar\rho(\theta^E)\, \bar{\hat A},\; \text{clip}(\bar\rho, 1 \pm \epsilon)\, \bar{\hat A}\bigr)\right],$$
where
$$\bar{\hat A} \;=\; \frac{1}{N_\text{active}} \sum_i \hat A_i, \qquad \bar\rho(\theta^E) \;=\; \Bigl(\prod_i \rho_i(\theta^E)\Bigr)^{1/N_\text{active}} \;=\; \exp\!\left(\frac{1}{N_\text{active}} \sum_i \log \rho_i(\theta^E)\right).$$
The advantage is the arithmetic mean, the ratio is the geometric mean
(equivalently: the per-capita log-ratio).

**Lemma A.1 (first-order expansion of $L^{\text{coop}}_A$).** At
$\theta^E = \theta^E_\text{old}$, $\rho_i = 1$ and $\bar\rho = 1$. For small
$\Delta\theta^E = \theta^E - \theta^E_\text{old}$, writing
$\ell_i = \log \pi_i(a_i \mid o_i; \theta^E)$, a Taylor expansion gives
$$\log \bar\rho \;=\; \frac{1}{N_\text{active}} \sum_i (\ell_i - \ell_{i,\text{old}}) \;=\; \frac{1}{N_\text{active}} \sum_i \nabla_{\theta^E} \ell_i \cdot \Delta\theta^E + O(\|\Delta\theta^E\|^2).$$
Hence
$$\bar\rho - 1 \;=\; \frac{1}{N_\text{active}} \sum_i \nabla_{\theta^E} \ell_i \cdot \Delta\theta^E + O(\|\Delta\theta^E\|^2).$$

**Proposition A (gradient of $L^{\text{coop}}_A$).** Under A1–A4 and A2,
$$\nabla_{\theta^E} L^{\text{coop}}_A \big|_{\theta_\text{old}} \;=\; \mathbb{E}\left[\,\bar{\hat A} \cdot \frac{1}{N_\text{active}} \sum_i \nabla_{\theta^E} \log \pi_i\,\right].$$

*Proof.* Differentiate $\bar\rho \cdot \bar{\hat A}$ in $\theta^E$:
$\nabla_{\theta^E} \bar\rho = \bar\rho \cdot \nabla_{\theta^E} \log \bar\rho = \bar\rho \cdot (1/N_\text{active}) \sum_i \nabla_{\theta^E} \log \pi_i$. Evaluating at $\theta^E_\text{old}$ where $\bar\rho = 1$, and noting A2 makes the clip inactive,
$$\nabla_{\theta^E} L^{\text{coop}}_A \big|_{\theta_\text{old}} = \mathbb{E}\left[\bar{\hat A} \cdot \frac{1}{N_\text{active}} \sum_i \nabla_{\theta^E} \log \pi_i\right].\qquad\square$$

**Interpretation of A.** The encoder receives a single scalar $\bar{\hat A}$
multiplied by the *mean* per-agent score function. Every agent's score
function is weighted by the **same** advantage.

### 3.2 Formulation B (per-agent: per-agent advantages + per-agent ratios) — **implemented, preferred**

The formulation implemented in `HGTeamHA.py::_coop_encoder_update` and
reflected in `paper/outline.md` §IV.F.1 is
$$L^{\text{coop}}_B(\theta^E) \;=\; \mathbb{E}\left[\frac{1}{N_\text{active}} \sum_i \min\bigl(\rho_i(\theta^E) \hat A_i,\; \text{clip}(\rho_i, 1 \pm \epsilon) \hat A_i\bigr)\right].$$

**Proposition B (gradient of $L^{\text{coop}}_B$).** Under A1–A4 and A2,
$$\nabla_{\theta^E} L^{\text{coop}}_B \big|_{\theta_\text{old}} \;=\; \mathbb{E}\left[\frac{1}{N_\text{active}} \sum_i \hat A_i \cdot \nabla_{\theta^E} \log \pi_i\right].$$

*Proof.* Under A2 the clip is inactive, so each per-agent term reduces to
$\rho_i \hat A_i$. Differentiating and evaluating at $\theta^E_\text{old}$
where $\rho_i = 1$:
$$\nabla_{\theta^E} L^{\text{coop}}_B \big|_{\theta_\text{old}} = \mathbb{E}\left[\frac{1}{N_\text{active}} \sum_i \hat A_i \cdot \nabla_{\theta^E} \log \pi_i\right].\qquad\square$$

**Interpretation of B.** Each agent's score function is weighted by **its own**
advantage $\hat A_i$. This is the standard multi-agent policy-gradient
estimator.

**Proposition C (B is an unbiased estimator of the cooperative gradient).** The
gradient $\nabla_{\theta^E} L^{\text{coop}}_B |_{\theta_\text{old}}$ is an
unbiased estimator of $\nabla_{\theta^E} J(\theta)|_{\theta_\text{old}}$, the
per-capita policy-gradient objective.

*Proof.* By the per-capita multi-agent policy-gradient theorem (linearity of
the gradient through $\bar R$ and the policy-gradient theorem per agent),
$$\nabla_\theta J \;=\; \mathbb{E}\left[\frac{1}{N_\text{active}} \sum_i \nabla_\theta \log \pi_i \cdot A_i\right].$$
Per-group critics $V^i$ from A4 are valid baselines for their respective
agents; subtracting $V^{g(i)}$ from each agent's return does not change the
expected gradient but reduces variance. Hence
$\mathbb{E}[\sum_i \hat A_i \nabla_\theta \log \pi_i / N_\text{active}] = \nabla_\theta J$, proving Proposition C. $\square$

**Observation.** Proposition A shows Formulation A produces a gradient of the
form $\mathbb{E}[\bar{\hat A} \cdot \overline{\nabla \log \pi}]$, with the
mean score function scaled by the mean advantage. Proposition B produces
$\mathbb{E}[\overline{\hat A \cdot \nabla \log \pi}]$. These differ whenever
advantages and score functions are not perfectly correlated across agents —
which is precisely the regime where heterogeneity carries information.

### 3.3 Equivalence Lemma: when A and B coincide

The conditions under which Formulations A and B give the same expected
gradient are important both for theoretical clarity and for understanding why
B is preferred.

**Lemma 1 (Formulation A = Formulation B under symmetry).** If for each state
$s$ and joint action $\boldsymbol{a}$, either (i) all per-agent advantages are
equal, $\hat A_i(s, \boldsymbol{a}) = \hat A(s, \boldsymbol{a})$ for all $i$,
or (ii) all per-agent score functions $\nabla_{\theta^E} \log \pi_i(a_i \mid o_i)$ are equal across agents, then
$$\nabla_{\theta^E} L^{\text{coop}}_A \big|_{\theta_\text{old}} = \nabla_{\theta^E} L^{\text{coop}}_B \big|_{\theta_\text{old}}.$$

*Proof.* Condition (i): $\bar{\hat A} = \hat A$, and $\sum_i \hat A_i \nabla_{\theta^E} \log \pi_i = \hat A \sum_i \nabla_{\theta^E} \log \pi_i$. Both formulations yield $\mathbb{E}[\hat A \cdot (1/N) \sum_i \nabla_{\theta^E} \log \pi_i]$. Condition (ii): let $g_\star = \nabla_{\theta^E} \log \pi_i$ for all $i$. Then
$(1/N) \sum_i \hat A_i g_\star = \bar{\hat A} g_\star$ and
$\bar{\hat A} \cdot (1/N) \sum_i g_\star = \bar{\hat A} g_\star$. $\square$

**When does condition (i) hold?** Only under common reward **and** symmetric
value functions **and** symmetric trajectories — a strong coincidence. In
PowerGridworld's mixed-motive setting $R_i \ne R_j$ in general, so
$\hat A_i \ne \hat A_j$, and condition (i) fails.

**When does condition (ii) hold?** Only under full parameter sharing at the
head **and** identical observations across agents. With per-group heads and
heterogeneous observations, condition (ii) fails.

**Consequence.** In HGTeam's actual operating regime, **Formulations A and B
are not equivalent**. They differ by the cross-term between advantage
heterogeneity and score-function heterogeneity across agents.

### 3.4 Proposition (Variance structure): B uses per-agent baselines, A uses an averaged baseline

**Both A and B are unbiased.** An earlier draft of this appendix incorrectly
claimed that Formulation A is biased under heterogeneity. That claim was wrong;
this section replaces it with the correct analysis.

Let $\hat g_A$ and $\hat g_B$ denote the single-sample gradient estimators of
$L^{\text{coop}}_A$ and $L^{\text{coop}}_B$ at $\theta^E_\text{old}$, writing
$g_i = \nabla_{\theta^E} \log \pi_i(a_i \mid o_i)$:
$$\hat g_A \;=\; \bar{\hat A} \cdot \frac{1}{N} \sum_i g_i, \qquad \hat g_B \;=\; \frac{1}{N} \sum_i \hat A_i\, g_i.$$

**Proposition 2 (both estimators unbiased).** *Under A1–A4 and A2, both
$\hat g_A$ and $\hat g_B$ are unbiased estimators of
$\nabla_{\theta^E} J(\theta)|_{\theta_\text{old}}$, the per-capita
policy-gradient objective.*

*Proof.* Unbiasedness of B is Proposition C. For A, use linearity and the
multi-agent policy-gradient theorem. Writing
$\nabla J_j = \nabla_{\theta^E} \mathbb{E}[\sum_t \gamma^t R_j]$:
$$\mathbb{E}[\hat g_A] \;=\; \mathbb{E}\Bigl[(1/N^2)\sum_{i,j} \hat A_j\, g_i\Bigr] \;=\; \frac{1}{N^2}\sum_{i,j} \mathbb{E}[\hat A_j\, g_i].$$
Because per-agent baselines $V^{g(j)}$ do not depend on any agent's action,
$\hat A_j$ satisfies $\mathbb{E}_{\boldsymbol{a}\sim\boldsymbol\pi}[\hat A_j \mid s] = 0$,
so by the policy-gradient theorem $\mathbb{E}[\hat A_j g_i] = \nabla J_j$
for *any* $i$. Substituting:
$\mathbb{E}[\hat g_A] = (1/N^2) \cdot N \cdot \sum_j \nabla J_j = (1/N)\sum_j \nabla J_j = \nabla J$. $\square$

**What actually differs between A and B.** The difference is in the
per-sample variance and the way baselines are assigned, not in the expected
gradient. Write
$$\hat g_B - \hat g_A \;=\; \frac{1}{N} \sum_i (\hat A_i - \bar{\hat A})\,(g_i - \bar g),$$
the within-sample cross-agent covariance of advantage heterogeneity and
score-function heterogeneity. This quantity has **mean zero** (the population
covariance across agents collapses to $\nabla J - \nabla J = 0$) but nonzero
per-sample magnitude, so $\hat g_A$ and $\hat g_B$ disagree sample-by-sample
and have different variances.

**Baseline interpretation.** Formulation A is equivalent to running PPO on
the cooperative reward $\bar R = (1/N)\sum_i R_i$ with a **shared** baseline
$V_\text{coop} = (1/N)\sum_i V^i$. Formulation B is equivalent to running PPO
on $\bar R$ with **per-agent** baselines $V^{g(i)}$, i.e. each agent's score
function is paired with the baseline of its own group. By the standard
control-variate theory of COMA (Foerster et al. 2018), the per-agent
baseline scheme reduces gradient variance when per-group critics carry
information not captured by the pooled critic. In homogeneous limits (common
reward, symmetric policies), the two schemes coincide.

**Proposition 2' (variance comparison, heuristic).** *Under A1–A4 and the
assumption that per-group critics are more informative baselines for their
own agents than the pooled average is (formally: $\mathrm{Var}[\hat A_i \mid s, g(i)] < \mathrm{Var}[\bar{\hat A} \mid s]$
for agent $i$'s gradient term), Formulation B has lower gradient variance
than Formulation A.*

We state this as a heuristic proposition rather than a theorem because the
variance relation depends on the trained-critic quality; in pathological
regimes where per-group critics are severely under-trained, A can have lower
variance than B by washing out noisy per-group residuals.

**Takeaway.** Formulation B is preferred **on variance grounds**, not on bias
grounds. Both formulations are unbiased estimators of the per-capita
cooperative gradient. B is implemented in the code because it provides
per-agent baselines (lower variance in typical regimes) and because its
per-agent clipping is more forgiving when per-agent ratios diverge (under
heterogeneous policies). A is an honest baseline that would also converge to
the same objective, just with higher gradient variance.

### 3.5 Formulation A vs. B under the clip

The gradient analyses above are at the point $\theta^E = \theta^E_\text{old}$,
where $\rho_i = 1$ for all $i$ and the clip is inactive. Off-initialisation,
the two formulations differ more substantively:

- A clips the **single** aggregated ratio $\bar\rho$. If even one $\rho_i$ is
  far from 1, $\bar\rho$ can still be close to 1 (the log-ratios average),
  so A under-clips.
- B clips each $\rho_i$ independently. Extreme per-agent ratios are clipped
  regardless of the other agents' ratios, giving B a tighter effective trust
  region in the heterogeneous regime.

For HGTeam this favours B: under heterogeneous policies (different groups,
different observations), per-agent ratios can diverge while the aggregated
ratio $\bar\rho$ stays near 1. A would therefore allow larger per-agent
moves than the PPO trust region is designed to tolerate. This is an
additional reason to prefer B in practice, independent of variance.

### 3.6 Formulation C (single-agent joint-policy view)

A third framing, raised in discussion, treats the full team as a single
"super-agent" whose policy is the joint policy
$\pi_\text{joint}(\boldsymbol{a} \mid s; \theta) = \prod_i \pi_i(a_i \mid o_i; \theta)$,
and runs single-agent PPO on this super-agent.

Define
$$\rho_\text{joint}(\theta^E) \;=\; \prod_i \rho_i(\theta^E) \;=\; \exp\!\left(\sum_i \log \rho_i\right),$$
and the "team advantage" $A_\text{team} = \sum_i A_i$ (via linearity of GAE
under the team-summed reward). The single-agent PPO surrogate on the super-
agent is
$$L^{\text{coop}}_C(\theta^E) \;=\; \mathbb{E}\left[\min\bigl(\rho_\text{joint} A_\text{team},\; \text{clip}(\rho_\text{joint}) A_\text{team}\bigr)\right].$$

**Lemma 2 (relationship between A, B, C).** Under A2,
$\nabla_{\theta^E} L^{\text{coop}}_C |_{\theta_\text{old}} = \mathbb{E}[A_\text{team} \cdot \sum_i \nabla_{\theta^E} \log \pi_i]$. Since $A_\text{team} = N_\text{active} \bar{\hat A}$, this equals $N_\text{active}$ times the gradient of Formulation A. Thus **C is a rescaled A** (gradient is $N_\text{active}^2$ times the per-capita A gradient), differing only by an $N_\text{active}$-dependent scale.

**Consequence.** C is **not** equivalent to B. C inherits all the same biases
as A under heterogeneity and additionally has $O(N)$ log-ratio variance
(because $\log \rho_\text{joint} = \sum_i \log \rho_i$ has variance scaling
with $N$). C is what naive "single-agent joint-policy PPO" would do and is
*exactly* the regime D8 was introduced to avoid.

**Takeaway.** The three formulations lie on a spectrum:

| | Ratio | Advantage | Expected gradient | Log-ratio variance | Clip behaviour |
|---|---|---|---|---|---|
| **A** | geometric mean $\bar\rho$ | mean $\bar{\hat A}$ | $\nabla J$ (unbiased) | $O(1/N)$ | clips aggregate only |
| **B** | per-agent $\rho_i$ | per-agent $\hat A_i$ | $\nabla J$ (unbiased) | $O(1/N)$ (no joint log-ratio) | clips each agent |
| **C** | product $\rho_\text{joint}$ | sum $A_\text{team}$ | $N \cdot \nabla J$ (rescaled) | $O(N)$ | clips aggregate only |

All three are gradient-consistent estimators of the same cooperative objective
(up to the $N$-dependent scale of C). B is preferred on two grounds:
(1) per-agent baselines give lower gradient variance when per-group critics
are informative; (2) per-agent clipping enforces the PPO trust region
agent-by-agent, preventing any one agent's ratio from drifting far outside
$[1-\epsilon, 1+\epsilon]$ while the aggregate stays near 1.

### 3.7 Statement of Theorem 1

**Theorem 1 (Soundness of the Cooperative Encoder Update).** *Under assumptions
A1–A4, the Formulation-B cooperative PPO surrogate $L^{\text{coop}}_B$ has the
following properties at $\theta = \theta_\text{old}$:*

*(i) Gradient consistency.* $\nabla_{\theta^E} L^{\text{coop}}_B |_{\theta_\text{old}}$ *is an unbiased estimator of the per-capita policy gradient* $\nabla_{\theta^E} J(\theta)|_{\theta_\text{old}}$ *(Proposition C).*

*(ii) First-order improvement.* *For a sufficiently small gradient-ascent step
$\Delta \theta^E = \eta \nabla_{\theta^E} L^{\text{coop}}_B$ with $\eta > 0$,*
$$J(\theta_\text{new}) \;\ge\; J(\theta_\text{old}) + \eta \,\|\nabla_{\theta^E} J\|^2 - O(\eta^2),$$
*so the encoder update increases the per-capita objective in expectation.*

*(iii) Variance properties.* *Formulation A is also gradient-consistent (an
unbiased estimator of* $\nabla_{\theta^E} J$*), but B has lower gradient
variance than A in any regime where per-group critics are more informative
baselines than the averaged critic (Proposition 2'). B additionally enforces
per-agent clipping, which gives a tighter effective trust region under
heterogeneous policies.*

*Proof.* (i) is Proposition C. (ii) follows from the standard gradient-ascent
step bound under smoothness A1: $J(\theta_\text{new}) \ge J(\theta_\text{old}) + \langle \nabla J, \Delta\theta^E \rangle - \tfrac{L}{2} \|\Delta\theta^E\|^2 = J(\theta_\text{old}) + \eta \|\nabla J\|^2 - O(\eta^2)$. (iii) is Proposition 2' and §3.5. $\square$

**Remark.** Theorem 1 does *not* require fixed $N$ or common reward. The
per-capita framing absorbs variable $N$; the mixed-motive reward enters only
via the structure of $\hat A_i$, which remains a valid policy-gradient
estimator through the per-group critics. This is why Theorem 1 is the
paper's cleanest theoretical result.

---

## 4. Phase-1-after-Phase-0 soundness: HAPPO heads at the updated encoder

Phase 0 performs the Formulation-B cooperative encoder update (§3; Theorem 1).
Phase 1 freezes the updated encoder $\theta^E_1$ and runs sequential HAPPO on
the per-group heads in a random order $g_1, \dots, g_G$.

This section answers the question: *given that Phase 0 has already moved the
encoder, does Phase 1 monotonically improve the per-capita cooperative
objective $\bar J$?* The main result is that Phase 1 improves $\bar J$ to
**first order** regardless of the HAPPO factor, because the factor only
rescales per-group step sizes. The HAPPO factor's theorem-level guarantee
(which couples per-group updates into a joint monotonic-improvement statement)
**does not transfer** to the mixed-motive variable-$N$ setting for reasons
made precise below.

### 4.1 Setting and assumptions

**State after Phase 0.**
- Encoder: $\theta^E_1 = \theta^E_0 + \Delta^E$, where $\Delta^E$ is the
  Phase-0 step.
- Heads: $\theta^H_{g,0}$ for every group $g$ (unchanged).
- Old log-probs for Phase 1 are **re-evaluated under $(\theta^E_1, \theta^H_0)$**
  (`HGTeamHA.py::_need_reeval`). Per-agent ratios in Phase 1 therefore measure
  head drift from $\theta^H_0$ only, not Phase-0 encoder drift.
- Advantages used in Phase 1 are those computed at the start of `train_groups`,
  i.e. under the pre-Phase-0 parameters $\theta_0 = (\theta^E_0, \theta^H_0)$.
  Critics use a separate encoder (`_shared_gnn_critic`), so Phase 0 does not
  move the critic, and the advantages are stable w.r.t. critic evaluation
  between Phase 0 and Phase 1. They do, however, reflect the collection-time
  policy, not $\theta^E_1$. We return to this mismatch in §4.4.

**Per-group surrogate.** For group $g$ at position $m$ in the random order,
with cumulative factor $\tilde F_g$ (D8; geometric mean),
$$L^{\text{HAPPO}}_g(\theta^H_g) \;=\; \tilde F_g \cdot \mathbb{E}\!\left[\frac{1}{n_g}\sum_{i \in g} \min\bigl(\rho_i(\theta^H_g)\,\hat A_i,\; \text{clip}(\rho_i, 1\pm\epsilon)\,\hat A_i\bigr)\right],$$
where $\rho_i(\theta^H_g) = \pi_i(a_i \mid o_i; \theta^E_1, \theta^H_g) / \pi_i(a_i \mid o_i; \theta^E_1, \theta^H_{g,0})$.

**Assumption A5 (positive factor).** $\tilde F_g > 0$ almost surely. This
holds whenever all preceding-group ratios are positive, which they are since
they are ratios of probability densities.

### 4.2 First-order improvement (Theorem 2)

**Key observation.** The HAPPO factor $\tilde F_g$ is a **scalar that does not
depend on $\theta^H_g$** — it is a function of *other* groups' ratios, which
have already been fixed by the time we update group $g$. So with respect to
the parameters being updated, $\tilde F_g$ is a constant.

**Proposition 3 (per-group gradient direction).** *Under A1–A5 and A2, at
$\theta^H_g = \theta^H_{g,0}$,*
$$\nabla_{\theta^H_g} L^{\text{HAPPO}}_g \;=\; \tilde F_g \cdot \frac{1}{n_g} \sum_{i \in g} \hat A_i \nabla_{\theta^H_g} \log \pi_i.$$
*In expectation this equals* $(\tilde F_g / n_g) \cdot \sum_{i \in g} \nabla_{\theta^H_g} J_i$.

*Proof.* Under A2 the clip is inactive; $\rho_i = 1$ at initialisation. As in
Proposition B, differentiating gives the per-agent sum. Since $\theta^H_g$
parameterises only agents in group $g$, $\nabla_{\theta^H_g} \log \pi_i = 0$
for $i \notin g$. The factor $\tilde F_g$ passes through as a scalar.
Applying the policy-gradient theorem (A4 ensures valid baselines) gives the
expected form. $\square$

**Relation to per-capita gradient.** Because
$\bar J = (1/N_\text{total}) \sum_i J_i$ and $\theta^H_g$ only affects agents
$i \in g$, we have $\nabla_{\theta^H_g} \bar J = (1/N_\text{total}) \sum_{i \in g} \nabla_{\theta^H_g} J_i$. Substituting,
$$\mathbb{E}[\nabla_{\theta^H_g} L^{\text{HAPPO}}_g] \;=\; \tilde F_g \cdot \frac{N_\text{total}}{n_g}\,\nabla_{\theta^H_g} \bar J.$$

**This is the crux of the analysis.** The expected per-group gradient is
**parallel** to the per-group projection of $\nabla \bar J$, with a positive
scalar $\tilde F_g \cdot N_\text{total}/n_g$. Positive alignment is sufficient
for first-order improvement.

**Theorem 2 (Phase-1 first-order improvement).** *Under A1–A5 and A2, a
single sweep of Phase 1 with per-group step sizes $\eta_g > 0$ satisfies*
$$\mathbb{E}[\bar J(\theta_\text{new})] \;\ge\; \bar J(\theta_\text{old}; \theta^E_1) + \sum_g \eta_g \tilde F_g \frac{N_\text{total}}{n_g} \|\nabla_{\theta^H_g} \bar J\|^2 - O(\max_g \eta_g^2),$$
*where the expectation is over trajectory sampling and A1 smoothness bounds
the second-order term.*

*Proof.* Write the first-order expansion of $\bar J$ around $\theta_\text{old}$
(heads only, encoder frozen at $\theta^E_1$):
$$\bar J(\theta_\text{new}) \;=\; \bar J(\theta_\text{old}) + \sum_g \langle \nabla_{\theta^H_g} \bar J, \Delta\theta^H_g \rangle + O\bigl(\max_g \|\Delta\theta^H_g\|^2\bigr).$$
Since heads live in **disjoint parameter spaces**, the cross terms between
different groups vanish — i.e., the Hessian of $\bar J$ w.r.t.
$(\theta^H_g, \theta^H_{g'})$ for $g \ne g'$ contributes only to the
second-order term. Taking expectation and using Proposition 3 with
$\Delta\theta^H_g = \eta_g \nabla_{\theta^H_g} L^{\text{HAPPO}}_g$:
$$\mathbb{E}[\bar J(\theta_\text{new})] \;\ge\; \bar J(\theta_\text{old}) + \sum_g \eta_g \langle \nabla_{\theta^H_g} \bar J, \tilde F_g (N_\text{total}/n_g) \nabla_{\theta^H_g} \bar J \rangle - O(\max_g \eta_g^2).$$
Each inner product reduces to $\|\nabla_{\theta^H_g} \bar J\|^2$, giving the
stated bound. $\square$

**Corollary 1 (HAPPO factor does not affect first-order correctness).** *The
geometric-mean factor $\tilde F_g$ (D8) enters only as a **positive scalar
rescaling** of each group's step size. Setting $\tilde F_g = 1$ (i.e.,
independent per-group PPO) yields the same first-order improvement bound,
just without the rescaling.*

**Corollary 2 (per-group-order invariance to first order).** *Because
$\tilde F_g$ is positive and the per-group gradients are disjoint in
parameter space, any random ordering of groups gives the same first-order
bound.*

These corollaries are the soundness result. At first order, the HAPPO
machinery is cosmetic in the mixed-motive variable-$N$ setting: it rescales
step sizes but does not create or destroy gradient signal.

### 4.3 What the HAPPO factor does and does not do

To answer the user's concern "is the HAPPO-based update helping overall
convergence?" we need to go beyond first order and ask what the factor
contributes.

**In the original HAPPO setting (common reward, fixed $N$).** The factor
$F_g = \prod_{k<m} \rho_{i_k}$ makes the *sum* of sequential per-agent
surrogates equal to the joint advantage via the multi-agent advantage
decomposition lemma. This is the mechanism by which HAPPO's Theorem 1
upgrades per-agent PPO improvements to a **joint** monotonic improvement. It
is a higher-order effect: without the factor, per-agent PPO improvements need
not aggregate into joint improvement because earlier agents' changes shift
the distribution from which later agents' advantages were computed.

**In HGTeam's setting (mixed motive, variable $N$, geometric-mean factor).**
Three things break the HAPPO mechanism:

1. **Mixed-motive reward.** The advantage decomposition lemma assumes a
   common scalar return. Under $R_i = w_c R_\text{team} + w_i R^\text{self}_i$,
   no single scalar advantage exists to decompose.
2. **Geometric-mean factor.** $\tilde F_g = F_g^{1/n_{<g}}$ is not the
   product $F_g$ needed by the decomposition (Conjecture 1 attempts to
   recover a first-order approximation).
3. **Variable $N$.** The decomposition lemma is stated per-permutation on a
   fixed set. Under variable $N$ it must be stated set-by-set, which works
   but invalidates any cross-episode joint-return statement.

**What $\tilde F_g$ provides instead.** A **coupled trust region**: if
earlier groups moved far (high $|\log \rho_{i_k}|$), $\tilde F_g$ is far from
1, so later groups' effective step size is reduced. This is a conservative
heuristic — it shrinks later groups' updates in response to earlier
changes — but it does not recover HAPPO's monotonic-improvement theorem.

**Consequence for the user's concern.** In the small-clip regime (A2), all
$\log \rho_{i_k}$ are $O(\epsilon)$, so $\log \tilde F_g$ is $O(\epsilon)$
and $\tilde F_g \approx 1 + O(\epsilon)$. The factor's contribution is
second-order; with geometric-mean normalisation it is *even smaller* for
larger $n_{<g}$. So we expect the factor to have **small but nonzero impact
on convergence**, whose sign depends on whether per-group gradients reinforce
or conflict:

- **Reinforcing regime** (per-group gradients aligned with $\nabla \bar J$):
  factor is mildly conservative, slight slowdown but no divergence.
- **Conflicting regime** (per-group gradients partially anti-aligned):
  factor is mildly helpful — it dampens later-group updates after a
  large-magnitude earlier-group step, preventing overshoot.

**Honest statement.** Without a theorem-level guarantee, the HAPPO factor is
a **heuristic trust-region coupling** in the mixed-motive variable-$N$
setting. Whether it provides net benefit over independent per-group PPO is
an **empirical question** (see §4.5).

### 4.4 Stale advantage correction (a subtle mismatch)

Phase 1 uses advantages $\hat A_i$ computed at $\theta_0 = (\theta^E_0, \theta^H_0)$,
but ratios $\rho_i$ are measured relative to the re-evaluated base
$(\theta^E_1, \theta^H_0)$. Under standard PPO, advantages and ratios should
be evaluated at the same base. The mismatch introduces a correction term
bounded by how much Phase 0 moved the encoder.

**Proposition 4 (advantage staleness is second-order).** *Under A1–A4 and
assuming the Phase-0 encoder step satisfies $\|\Delta^E\| \le \delta$, the
difference between the stale-advantage surrogate gradient and the fresh-
advantage surrogate gradient is $O(\delta)$. Consequently, the Phase-1
first-order improvement bound in Theorem 2 incurs an additive correction of
order* $\sum_g \eta_g \tilde F_g \cdot O(\delta) \cdot \|\nabla_{\theta^H_g} \bar J\|$.

*Proof sketch.* Define $\hat A_i(\theta)$ as the on-policy advantage at
parameters $\theta$. By continuity of the value function (A1) and smoothness
of the encoder in $\theta^E$,
$\hat A_i(\theta^E_1) - \hat A_i(\theta^E_0) = \langle \nabla_{\theta^E} \hat A_i, \Delta^E \rangle + O(\|\Delta^E\|^2) = O(\delta)$.
Substituting into Proposition 3 gives the first-order staleness correction;
the effect on $\bar J$-improvement is the stated $O(\delta)$ term times the
gradient norm. $\square$

**Consequence.** If Phase 0 takes small steps, Phase 1 advantages are
approximately fresh and the Theorem 2 bound is essentially tight. If Phase 0
takes aggressive steps, Phase 1's per-group improvement direction deviates
from $\nabla \bar J$ by $O(\delta / \|\nabla \bar J\|)$. When the encoder
moves **far** in Phase 0, Phase 1 can in principle step *against* $\bar J$
if the stale advantages misdirect the gradient.

**Two honest mitigations.**

1. **Keep Phase 0 conservative.** Since Phase 0 uses the same clip $\epsilon$
   and Phase 0 is a standard PPO step, its $\|\Delta^E\|$ is already
   bounded. As long as Phase 0 does not hit the clip on many samples, $\delta$
   is small.
2. **Recompute advantages between Phase 0 and Phase 1.** This costs one
   additional critic forward pass per minibatch but eliminates the staleness
   correction. It is the theoretically clean choice; the cost is modest
   because the critic is small relative to the actor encoder.

**Recommendation.** For the JMLR paper, either (a) report Phase-0 clip-
fraction and $\|\Delta^E\|$ diagnostics to show staleness is small in
practice, or (b) implement the advantage recomputation and remove the
caveat. (b) is cleaner.

### 4.5 Empirical diagnostics to test whether HAPPO helps

Theorem 2 and §4.3 give the honest picture: the HAPPO factor provides no
first-order gradient signal and its higher-order effect is a conservative
coupled trust region. The user's empirical concern ("I'm not 100% confident
the HAPPO-based update is helping overall convergence") is therefore a
legitimate open question. Here are concrete ablations that would resolve it.

**Ablation 1 (HAPPO-factor ablation).** Run three configurations on
PowerGridworld and SMACv2, matched otherwise:

| Config | Phase 0 | Phase 1 factor | Phase 1 advantages | What it tests |
|---|---|---|---|---|
| **Full** | Formulation B | $\tilde F_g$ (D8) | stale (current) | Current HGTeam |
| **NoFactor** | Formulation B | $\tilde F_g = 1$ (independent PPO) | stale | Does factor help? |
| **NoFactorFresh** | Formulation B | $\tilde F_g = 1$ | recomputed | Does factor help once staleness is fixed? |

If **NoFactor** matches or beats **Full**, the factor is not contributing
and should be dropped. If **Full** beats **NoFactor**, the factor's
conservative trust-region coupling is providing real benefit (worth keeping,
worth further theoretical investigation). **NoFactorFresh** controls for
staleness.

**Ablation 2 (advantage staleness).** Run **Full** vs.
**FullFresh** (Phase 0 + Phase-1 advantage recomputation + D8 factor). If
**FullFresh** uniformly matches **Full** (within noise), the staleness term
is negligible and §4.4's caveat can be dropped empirically. If **FullFresh**
beats **Full** by a non-trivial margin, advantage recomputation is worth the
cost.

**Ablation 3 (Phase-1 ordering).** Run **Full** with (a) random group order
(current), (b) fixed order (always EV, PV, Storage), (c) reverse fixed
order. Theorem 2 Corollary 2 says all should match to first order; if they
don't, the HAPPO factor is doing something non-trivial at higher order.

**Ablation 4 (phase order).** Swap Phase 0 and Phase 1: run **HeadsFirst**
(HAPPO head loop first, then coop encoder update). Compare with **Full**
(encoder first). The paper's theoretical argument rests on encoder-first
ordering; this ablation tests whether encoder-first is empirically as well
as theoretically preferred.

**Diagnostics to log alongside any ablation.**

1. **Factor histogram.** Distribution of $\tilde F_g$ per group across
   training. If concentrated near 1, the factor is doing very little.
2. **Per-group gradient cosine.** Cosine similarity between
   $\nabla_{\theta^H_g} L^{\text{HAPPO}}_g$ and an estimate of
   $\nabla_{\theta^H_g} \bar J$ (from held-out trajectories). Directly
   measures whether per-group updates align with the cooperative gradient.
3. **Phase-0 step norm** $\|\Delta^E\|$ and clip fraction. Directly measures
   §4.4's $\delta$.
4. **Per-group return improvement.** $\Delta J_g$ per iteration, split into
   $\Delta J^\text{team}_g$ and $\Delta J^\text{self}_g$. Reveals whether
   the mixed-motive reward components conflict in practice.

**Recommended first run.** Ablation 1 on PowerGridworld for 5 seeds, 1M
steps. If NoFactor is within 1σ of Full, the JMLR paper can honestly drop
HAPPO from the contribution list and rest on the cooperative encoder
update alone — arguably a **cleaner story**, not a weaker one. If the
factor helps, its empirical contribution becomes a headline result
alongside Theorem 1.

---

## 5. Conjecture 1 — Variable-$N$ HAPPO with Geometric-Mean Factor

**Setting.** Consider HAPPO-style sequential per-group updates under variable
$N$. The standard HAPPO factor is
$$F_g(\theta) \;=\; \prod_{k < g} \prod_{i \in g_k} \rho_i(\theta).$$
D8 replaces this with
$$\tilde F_g(\theta) \;=\; \Bigl(\prod_{k < g} \prod_{i \in g_k} \rho_i(\theta)\Bigr)^{1/n_{<g}}, \qquad n_{<g} = \sum_{k<g} n_{g_k}.$$

**Why the substitution is not innocent.** HAPPO's monotonic-improvement
theorem relies on the identity
$$L^{\text{HAPPO}}_{g}(\theta) \;=\; F_g(\theta) \cdot \mathbb{E}[\rho_g \hat A_g],$$
together with the multi-agent advantage-decomposition lemma, to equate the
sequential surrogate to a sum of conditional advantages. Replacing $F_g$ by
$\tilde F_g$ breaks this equality exactly: $\tilde F_g = F_g^{1/n_{<g}}$ is a
non-linear function of $F_g$, so
$\mathbb{E}[\tilde F_g \rho_g \hat A_g] \ne \mathbb{E}[F_g \rho_g \hat A_g]$
except at $F_g = 1$.

**First-order argument (what we can currently claim).** In the small-clip
regime A2, $\log F_g = \sum_{i < g} \log \rho_i$ is $O(\epsilon)$. A Taylor
expansion of $F_g = 1 + \sum_i \log \rho_i + O(\epsilon^2)$ and
$\tilde F_g = 1 + (1/n_{<g}) \sum_i \log \rho_i + O(\epsilon^2)$ gives
$$F_g - \tilde F_g \;=\; \bigl(1 - 1/n_{<g}\bigr) \sum_i \log \rho_i + O(\epsilon^2) \;=\; O(\epsilon).$$
To first order in $\epsilon$, the two factors differ by a constant multiplier
$(1 - 1/n_{<g})$, which can be absorbed into an effective clip range
$\tilde\epsilon = \epsilon \cdot n_{<g}/(n_{<g} + 1)$ or similar. Under this
reparameterisation HAPPO's monotonic-improvement bound is recovered up to
$O(\epsilon^2)$.

**What remains open.**

1. The rescaling is **state-dependent** (via $n_{<g}$, which varies with the
   active set at each state), so the "effective clip range" is not a single
   scalar across a minibatch. A uniform-clip implementation therefore
   corresponds to HAPPO with a clip range that silently varies by $n_{<g}$;
   the question is whether any such variation introduces a second-order bias
   that accumulates across iterations.
2. The second-order residual is not clearly bounded without additional
   assumptions on the curvature of $J$ in $\theta$.
3. A HAML-style operator view (treat the update as a heterogeneous mirror-
   descent step with a modified Bregman divergence) may give a cleaner route;
   this has not yet been attempted.

**Conjecture 1.** *Under A1–A4 and A2, HAPPO with the geometric-mean factor
$\tilde F_g$ is monotonically improving in the common-reward component of $J$
up to $O(\epsilon^2)$. Equivalently, for sufficiently small $\epsilon$, there
exists a choice of clip schedule $\tilde\epsilon = c(n_{<g}) \cdot \epsilon$
under which standard HAPPO's monotonic-improvement theorem applies verbatim to
the geometric-mean factor.*

The conjecture is to be supported empirically (Section VI.E of the main
paper: factor-magnitude histograms, product vs. geometric-mean comparison on
fixed-$N$ HAPPO benchmarks) and, if a proof emerges, promoted to a theorem.

---

## 6. Open questions

These are the theoretical loose ends that a future agent (or the user)
should work on, in rough order of importance:

1. **Empirical resolution of §4.5 ablations.** Before chasing more theory,
   run Ablation 1: if the HAPPO factor is not empirically helpful in the
   mixed-motive setting, the paper should drop Conjecture 1 entirely and
   rest on Theorem 1 + Theorem 2. This is the single most important
   remaining question, and it is empirical rather than theoretical.
2. **Proof of Conjecture 1** via a HAML-style operator argument — only
   worth doing if Ablation 1 shows the factor is empirically helpful.
   Otherwise this is chasing a theorem for a mechanism that isn't paying
   its way.
3. **Variance bound for Formulation A vs. B.** Proposition 2' is stated
   heuristically; a clean theorem bounding the variance gap in terms of
   per-group critic quality would tighten the argument for B.
4. **Single-agent GNN framing formalised.** Is there a natural interpretation
   of the encoder update as a single-agent RL problem (treating the encoder's
   "action" as the embedding it produces, and the heads + environment as a
   deterministic downstream transition)? If so, this would give a cleaner
   theoretical story for Phase 0.
5. **Set-Markov-game convergence.** If the paper reframes to a set Markov
   game (per `paper/outline.md` §III.A), is there a size-invariance property
   of the encoder under which variable-$N$ convergence can be stated for the
   full two-phase update?
6. **Mixed-motive equilibrium characterisation.** In what sense does HGTeam
   converge in the mixed-motive PowerGridworld setting? A correlated-
   equilibrium or Nash-equilibrium statement would be the natural target but
   requires more structure on the game (e.g., concavity of per-agent
   returns).

---

## References used in this appendix

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Kuba et al., *Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning*, ICLR 2022. (HAPPO.)
- Kuba et al., *Heterogeneous-Agent Mirror Learning*, JMLR 2024. (HAML.)
- Foerster et al., *Counterfactual Multi-Agent Policy Gradients*, AAAI 2018. (COMA; counterfactual baselines.)
- Leibo et al., *Multi-agent Reinforcement Learning in Sequential Social Dilemmas*, AAMAS 2017.
- Shapley, *A Value for n-Person Games*, 1953.
