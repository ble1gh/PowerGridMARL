# Low-Rank vs Basis Modulation for Edge-Conditioned HGT

## 1. The Starting Point

Vanilla HGT has a relation-specific attention/message transform:

$$
W_{\mathrm{ATT}}^r,\quad W_{\mathrm{MSG}}^r
$$

for meta-relation

$$
r = \langle \tau(j), \phi(j,i), \tau(i) \rangle .
$$

To make HGT edge-aware, we want:

$$
W_{\mathrm{ATT}}^r(e_{ji}),\quad W_{\mathrm{MSG}}^r(e_{ji})
$$

where $e_{ji}$ is a fixed-dimensional continuous edge feature vector.

The unconstrained version is:

$$
W^r(e) = W_0^r + \Delta W^r(e),
$$

where an MLP maps $e$ to a full $d \times d$ matrix. This is expressive but expensive and reviewer-risky. Low-rank and basis modulation are structured ways to define $\Delta W^r(e)$.

### Implementation in this repo

`BenchMARL/benchmarl/models/edgeweightedHGT.py` implements the low-rank version as `EdgeWeightedHGT`. It is an opt-in BenchMARL model, separate from the current stable `HeteroGNN` defaults. For each relation and head it applies additive low-rank corrections to both the HGT attention transform and the HGT message transform:

$$
A^r_h(e) = A^r_h + U^r_{\mathrm{att},h}\operatorname{diag}(g^r_{\mathrm{att},h}(e))V^r_{\mathrm{att},h},
\qquad
M^r_h(e) = M^r_h + U^r_{\mathrm{msg},h}\operatorname{diag}(g^r_{\mathrm{msg},h}(e))V^r_{\mathrm{msg},h}.
$$

Relations with no edge features do not build edge gates. Setting `low_rank=0` disables modulation. With the default zero-initialized edge-gate output layer, edge features are initially a no-op, so the model starts at the vanilla-HGT relation-transform limit and learns edge-conditioned corrections only through training.

---

## 2. Low-Rank Modulation

Low-rank modulation says the edge feature can only change the relation matrix along a small number of learned directions:

$$
\Delta W^r(e) = U^r \operatorname{diag}(g^r(e)) V^r
$$

where:

$$
U^r \in \mathbb{R}^{d \times k},\quad
V^r \in \mathbb{R}^{k \times d},\quad
g^r(e) \in \mathbb{R}^k,\quad
k \ll d.
$$

The edge feature network outputs only $k$ scalars:

$$
g^r(e) = \mathrm{MLP}_r(e).
$$

So the edge-conditioned matrix is:

$$
W^r(e) = W_0^r + U^r \operatorname{diag}(g^r(e)) V^r.
$$

### Meaning

This says:

1. $V^r$ projects the node representation into a small latent interaction space.
2. $g^r(e)$ gates those latent directions according to the edge feature.
3. $U^r$ maps the modulated signal back to the model dimension.

So the edge feature does **not** invent an arbitrary relation matrix. It chooses how strongly to activate a small set of learned relation-adjustment directions.

In attention:

$$
s_{ji}^r =
\frac{
q_i^\top
\left[
W_{\mathrm{ATT},0}^r
+
U_{\mathrm{ATT}}^r
\operatorname{diag}(g_{\mathrm{ATT}}^r(e_{ji}))
V_{\mathrm{ATT}}^r
\right]
k_j
}{\sqrt d}
+
\mu_r .
$$

In messages:

$$
m_{ji}^r =
\left[
W_{\mathrm{MSG},0}^r
+
U_{\mathrm{MSG}}^r
\operatorname{diag}(g_{\mathrm{MSG}}^r(e_{ji}))
V_{\mathrm{MSG}}^r
\right]
v_j .
$$

If relation $r$ has no edge features, set $g^r(e)=0$, giving:

$$
W^r(e)=W_0^r.
$$

That recovers vanilla HGT.

### Literature Connections

- **Low-rank matrix factorization**: classical numerical linear algebra and statistical learning. The core idea is that many useful matrix variations live in a low-dimensional subspace.
- **LoRA**: Low-Rank Adaptation of Large Language Models, Hu et al. (2021/2022). LoRA writes parameter changes as low-rank updates:

$$
W' = W + BA
$$

with rank much smaller than $d$. Your version is edge-conditioned LoRA for HGT relation matrices.
- **Adapter methods / parameter-efficient fine-tuning**: also use small bottleneck updates to alter large models without replacing the whole parameter matrix.

### Reviewer View

A reviewer would likely see low-rank modulation as conservative and mathematically clean.

Strengths:
- Preserves vanilla HGT as the base model.
- Adds edge features as controlled corrections.
- Efficient parameter sharing remains central.
- Easy ablation: rank $k=0$ is vanilla HGT.
- Good for stability.

Weaknesses:
- Rank $k$ is a new hyperparameter.
- If $k$ is too small, edge features may be under-expressive.
- Interpretability is moderate, not perfect: the $k$ latent directions are learned, not necessarily human-labeled.

---

## 3. Basis Modulation

Basis modulation says the edge feature chooses a mixture of learned relation-update matrices:

$$
\Delta W^r(e) =
\sum_{b=1}^{B}
\beta_b^r(e) B_b^r
$$

where:

$$
B_b^r \in \mathbb{R}^{d \times d}
$$

are learned basis matrices, and

$$
\beta^r(e) = \mathrm{MLP}_r(e) \in \mathbb{R}^B
$$

are edge-feature-dependent coefficients.

So:

$$
W^r(e) =
W_0^r
+
\sum_{b=1}^{B}
\beta_b^r(e) B_b^r.
$$

### Meaning

This says the model has a small dictionary of relation-transform templates. The edge feature chooses a weighted mixture.

For example, for grid line edges, the bases might specialize into patterns like:

- high-impedance edge behavior,
- closed-switch behavior,
- transformer-like coupling,
- weak electrical coupling,
- strong electrical coupling.

The model is not forced to make those bases human-interpretable, but the structure makes that kind of specialization possible.

In attention:

$$
s_{ji}^r =
\frac{
q_i^\top
\left[
W_{\mathrm{ATT},0}^r
+
\sum_{b=1}^{B}
\beta_{\mathrm{ATT},b}^r(e_{ji})
B_{\mathrm{ATT},b}^r
\right]
k_j
}{\sqrt d}
+
\mu_r .
$$

In messages:

$$
m_{ji}^r =
\left[
W_{\mathrm{MSG},0}^r
+
\sum_{b=1}^{B}
\beta_{\mathrm{MSG},b}^r(e_{ji})
B_{\mathrm{MSG},b}^r
\right]
v_j .
$$

Again, if there are no edge features:

$$
\beta^r(e)=0
$$

and:

$$
W^r(e)=W_0^r.
$$

### Literature Connections

- **R-GCN basis decomposition**: Schlichtkrull et al. (2018), Relational Graph Convolutional Networks. R-GCN reduces relation-specific parameters using:

$$
W_r = \sum_{b=1}^{B} a_{rb} V_b.
$$

Your version makes the coefficients depend on edge features:

$$
W_r(e) = W_{r,0} + \sum_b \beta_b(e) V_b.
$$

- **Mixture-of-experts**: a gating network chooses a weighted mixture of expert transformations.
- **Dynamic Filter Networks**: Jia et al. (2016), where filters are generated dynamically from input.
- **CondConv**: Yang et al. (2019), where convolution kernels are input-conditioned mixtures of experts.
- **Edge-conditioned convolution**: Simonovsky and Komodakis (2017), where edge labels/features generate filters for graph convolution.

### Reviewer View

A reviewer may find basis modulation more interpretable than low-rank modulation.

Strengths:
- Very natural for heterogeneous relations.
- Direct connection to R-GCN basis decomposition.
- Edge features select among relation-transform templates.
- Easier to inspect coefficients $\beta_b(e)$.
- Can be shared across relations if desired.

Weaknesses:
- Basis matrices can be parameter-heavy.
- Interpretability is suggestive, not guaranteed.
- If each relation has its own bases, parameter count may grow quickly.
- If bases are shared globally, implementation and explanation become more complex.

---

## 4. Low-Rank vs Basis: Core Difference

Low-rank modulation says:

$$
\Delta W(e)
\text{ lives in a rank-}k\text{ matrix family.}
$$

Basis modulation says:

$$
\Delta W(e)
\text{ lives in the span of }B\text{ learned matrices.}
$$

Low-rank constrains the **rank** of the update.

Basis modulation constrains the **dictionary** of possible updates.

They can overlap. If every basis matrix $B_b$ is low-rank, then basis modulation is also low-rank-ish. But conceptually they emphasize different things:

- Low-rank: efficient adaptation directions.
- Basis: mixture of learned relation mechanisms.

## 5. My Recommendation for HGTeam

For a first edge-aware HGT implementation, I would choose:

$$
W^r(e) =
W_0^r
+
U^r \operatorname{diag}(g^r(e)) V^r.
$$

That is low-rank modulation.

Why:
- It is easier to implement.
- It is cheaper.
- It is less likely to destabilize training.
- It has a clean vanilla-HGT special case.
- It is easy to explain as “LoRA-style edge-conditioned relation adaptation.”

Then, if interpretability becomes central, try basis modulation as a second model:

$$
W^r(e) =
W_0^r
+
\sum_b \beta_b^r(e) B_b^r.
$$

That gives better “which learned relational template is active?” diagnostics.

## 6. Best Reviewer-Safe Framing

I would write:

> We extend HGT to edge-attributed heterogeneous graphs by retaining the vanilla meta-relation matrix $W_0^r$ and adding a structured edge-conditioned correction $\Delta W^r(e)$. In the absence of edge features, $\Delta W^r(e)=0$, so the layer exactly reduces to vanilla HGT. We consider low-rank modulation as the default because it preserves HGT’s parameter-sharing structure while allowing continuous physical edge attributes to modulate attention and message passing.

That is clean, defensible, and directly tied to the grid motivation.