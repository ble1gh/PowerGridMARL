# HGT vs. HGTeam's HeteroConv Transformer Attention

This note compares the graph attention math in three related but different architectures:

1. **Vanilla HGT**: the Heterogeneous Graph Transformer from Hu et al. (2020), often implemented as PyG `HGTConv`.
2. **Current HGTeam encoder**: `HeteroConv` over relation-specific `TransformerConv` layers in `BenchMARL/benchmarl/models/heterognn.py`.
3. **Opt-in EdgeWeightedHGT**: `BenchMARL/benchmarl/models/edgeweightedHGT.py`, a custom HGT-style layer with low-rank edge-conditioned relation transforms.

The short version:

- Vanilla HGT is explicitly designed around **meta-relations** and learns type/relation-specific attention, message, and prior terms.
- HGTeam's current encoder is **HGT-inspired**, but intentionally uses `TransformerConv` inside `HeteroConv` because grid and critic edges need edge features.
- `EdgeWeightedHGT` restores the HGT-style relation-transform structure while adding continuous edge features through low-rank modulation.
- With no edge features, HGTeam reduces to a relation-typed TransformerConv stack, not vanilla HGT.
- With edge features, HGTeam becomes an edge-aware relation-typed TransformerConv stack; this is the important practical difference for PowerGridworld.
- `EdgeWeightedHGT` is not a thin wrapper around PyG `HGTConv`: PyG `HGTConv.forward()` does not accept external `edge_attr_dict`, so the repo implements the edge-conditioned HGT equations directly.

---

## 1. Basic Message Passing

A graph neural network updates a node representation by collecting messages from its neighbors.

Let:

- $i$ be the destination node.
- $j$ be a source neighbor of $i$.
- $h_i^{(\ell)}$ be node $i$'s embedding at layer $\ell$.
- $\mathcal{N}(i)$ be the source neighbors that send messages to $i$.

A generic message-passing layer looks like:

$$
h_i^{(\ell+1)}
= \operatorname{Update}\left(
h_i^{(\ell)},
\operatorname{Aggregate}_{j \in \mathcal{N}(i)}
\operatorname{Message}(h_j^{(\ell)}, h_i^{(\ell)}, e_{ji})
\right).
$$

The edge feature $e_{ji}$ may or may not exist.

Attention-based GNNs choose how much to listen to each neighbor:

$$
h_i^{(\ell+1)}
= \sum_{j \in \mathcal{N}(i)}
\alpha_{ji} \cdot m_{ji},
$$

where:

- $\alpha_{ji}$ is the attention weight from source $j$ to destination $i$.
- $m_{ji}$ is the message sent from $j$ to $i$.

The architectures below differ in how they compute $\alpha_{ji}$ and $m_{ji}$.

---

## 2. Homogeneous Transformer-Style Graph Attention

Before heterogeneity, consider a normal transformer-style graph attention layer.

Define:

$$
q_i = W_Q h_i,
\qquad
k_j = W_K h_j,
\qquad
v_j = W_V h_j.
$$

Then:

$$
\alpha_{ji}
= \operatorname{softmax}_{j \in \mathcal{N}(i)}
\left(
\frac{q_i^\top k_j}{\sqrt{d}}
\right),
$$

and:

$$
h_i'
= \sum_{j \in \mathcal{N}(i)} \alpha_{ji} v_j.
$$

This is homogeneous because the same $W_Q$, $W_K$, and $W_V$ apply to all nodes and all edges.

---

## 3. Vanilla HGT

HGT is designed for graphs with multiple node and edge types.

Each node has a type:

$$
\tau(i) \in \mathcal{A}
$$

and each edge has a relation type:

$$
\phi(j,i) \in \mathcal{R}.
$$

The key object in HGT is the **meta-relation**:

$$
r = \langle \tau(j), \phi(j,i), \tau(i) \rangle.
$$

For example, in a grid graph:

$$
\langle \text{EV}, \text{connected_to}, \text{grid_node} \rangle
$$

and

$$
\langle \text{PV}, \text{connected_to}, \text{grid_node} \rangle
$$

are different meta-relations because the source type differs.

## 3.1 HGT Typed Projections

Vanilla HGT uses node-type-specific projections:

$$
q_i = W_Q^{\tau(i)} h_i,
$$

$$
k_j = W_K^{\tau(j)} h_j,
$$

$$
v_j = W_V^{\tau(j)} h_j.
$$

This means a PV node and an EV node are projected through different matrices even if they have the same feature dimension.

## 3.2 HGT Relation-Specific Attention

HGT then modifies the key-query interaction using a relation-specific attention matrix:

$$
s_{ji}
=
\frac{
q_i^\top W_{\phi(j,i)}^{ATT} k_j
}{\sqrt{d}}.
$$

HGT also includes a learned prior for the full meta-relation:

$$
\mu_{\langle \tau(j), \phi(j,i), \tau(i) \rangle}.
$$

So the attention score is often written as:

$$
s_{ji}
=
\frac{
q_i^\top W_{\phi(j,i)}^{ATT} k_j
}{\sqrt{d}}
\mu_{\langle \tau(j), \phi(j,i), \tau(i) \rangle},
$$

or equivalently with the prior as a multiplicative/logit scaling term, depending on notation.

The attention weight is:

$$
\alpha_{ji}
=
\operatorname{softmax}_{j \in \mathcal{N}(i)}
\left(s_{ji}\right).
$$

## 3.3 HGT Relation-Specific Messages

The message also receives a relation-specific transform:

$$
m_{ji}
=
W_{\phi(j,i)}^{MSG} v_j.
$$

The updated node representation is:

$$
\tilde h_i
=
\sum_{j \in \mathcal{N}(i)}
\alpha_{ji} m_{ji}.
$$

HGT then applies a destination-type-specific output/update transform:

$$
h_i'
=
\operatorname{Update}_{\tau(i)}(h_i, \tilde h_i).
$$

The exact update includes residuals, normalization, activation, and sometimes a type-specific skip/gate.

## 3.4 What Vanilla HGT Learns Separately

Vanilla HGT separates:

- **source node type** through $W_K^{\tau(j)}$ and $W_V^{\tau(j)}$,
- **destination node type** through $W_Q^{\tau(i)}$ and output/update transforms,
- **edge relation type** through $W_{\phi}^{ATT}$ and $W_{\phi}^{MSG}$,
- **full meta-relation importance** through $\mu_{\langle \tau(j), \phi, \tau(i) \rangle}$.

This is why HGT is mathematically clean for heterogeneous graphs.

## 3.5 What Vanilla HGT Does Not Give Us Here

PyG's `HGTConv` does not expose the edge-feature path HGTeam needs.

In PowerGridworld, edges are not merely labels like "line" or "transformer". They can carry physical attributes:

- line features,
- transformer features,
- switch features,
- action-edge features for critics,
- mapping edges between agents and grid nodes.

Those attributes are not optional implementation details. They encode physical coupling and counterfactual action conditioning.

That is the reason HGTeam moved away from direct `HGTConv`.

---

## 4. HGTeam's Current Encoder

HGTeam currently builds:

$$
\operatorname{HeteroConv}
\left(
\left\{
(\text{src}, \text{rel}, \text{dst})
\mapsto
\operatorname{TransformerConv}_{(\text{src}, \text{rel}, \text{dst})}
\right\}
\right).
$$

In code terms:

```python
conv_dict[(src, rel, dst)] = TransformerConv(...)
self.convs.append(HeteroConv(conv_dict, aggr="sum"))
```

Each full edge type tuple gets its own `TransformerConv` instance. Therefore the relation type still matters strongly: an EV-to-grid edge and a Storage-to-grid edge do not share the same convolution parameters unless the model is explicitly built that way.

## 4.1 Current Encoder Without Edge Features

For one relation $r = (\text{src}, \text{rel}, \text{dst})$, `TransformerConv` without edge features computes something close to:

$$
q_i^r = W_Q^r h_i,
$$

$$
k_j^r = W_K^r h_j,
$$

$$
v_j^r = W_V^r h_j.
$$

The attention score is:

$$
s_{ji}^r
=
\frac{(q_i^r)^\top k_j^r}{\sqrt{d}},
$$

and:

$$
\alpha_{ji}^r
=
\operatorname{softmax}_{j \in \mathcal{N}_r(i)}
\left(s_{ji}^r\right).
$$

The relation-specific output is:

$$
\tilde h_{i,r}
=
\sum_{j \in \mathcal{N}_r(i)}
\alpha_{ji}^r v_j^r.
$$

Then `HeteroConv(..., aggr="sum")` aggregates all incoming relation outputs for node type $\tau(i)$:

$$
h_i'
=
\sum_{r \in \mathcal{R}_{\rightarrow \tau(i)}}
\tilde h_{i,r}.
$$

This is why the current model is still heterogeneous: the parameters are separated by full edge tuple.

But it is not vanilla HGT because there is no explicit HGT-style factorization into:

- node-type projection matrices $W_Q^{\tau(i)}, W_K^{\tau(j)}, W_V^{\tau(j)}$ shared across relations,
- relation matrices $W_{\phi}^{ATT}, W_{\phi}^{MSG}$,
- meta-relation priors $\mu_{\langle \tau(j), \phi, \tau(i) \rangle}$.

Instead, the full relation tuple gets its own complete TransformerConv parameter set.

## 4.2 Current Encoder With Edge Features

When a relation has edge features, HGTeam passes `edge_dim` to `TransformerConv`.

Let:

$$
e_{ji} \in \mathbb{R}^{d_e}
$$

be the edge feature.

For one relation $r$, PyG `TransformerConv` uses an edge projection. Conceptually:

$$
q_i^r = W_Q^r h_i,
$$

$$
k_j^r = W_K^r h_j + W_E^r e_{ji},
$$

$$
v_j^r = W_V^r h_j + W_E^r e_{ji}.
$$

Then:

$$
s_{ji}^r
=
\frac{(q_i^r)^\top k_j^r}{\sqrt{d}},
$$

$$
\alpha_{ji}^r
=
\operatorname{softmax}_{j \in \mathcal{N}_r(i)}
\left(s_{ji}^r\right),
$$

and:

$$
\tilde h_{i,r}
=
\sum_{j \in \mathcal{N}_r(i)}
\alpha_{ji}^r v_j^r.
$$

The important part is that the edge feature enters twice:

1. It changes the key, so it changes **attention weight**.
2. It changes the value/message, so it changes **message content**.

This is the mathematical behavior HGTeam needs for grid edges and action-conditioned critic edges.

---

## 5. Difference 1: Parameter Factorization

Vanilla HGT factorizes heterogeneity:

$$
\text{node type} + \text{edge type} + \text{destination type}
$$

into separate pieces.

For attention:

$$
q_i = W_Q^{\tau(i)} h_i,
\qquad
k_j = W_K^{\tau(j)} h_j,
$$

then:

$$
q_i^\top W_{\phi}^{ATT} k_j.
$$

HGTeam's current relation-typed TransformerConv does not factorize this way. For each full tuple $r$:

$$
q_i^r = W_Q^r h_i,
\qquad
k_j^r = W_K^r h_j.
$$

So the full edge type owns its own query/key/value projections.

## Consequence

Vanilla HGT shares information across meta-relations more structurally.

For example:

$$
\langle \text{EV}, \text{connects}, \text{grid} \rangle
$$

and

$$
\langle \text{PV}, \text{connects}, \text{grid} \rangle
$$

share the same destination-type projection for `grid` in vanilla HGT, and may share relation components for `connects`.

In the current HGTeam encoder, those two edge tuples can have entirely separate `TransformerConv` modules. This is more flexible but less parameter-efficient and less explicitly tied to HGT's meta-relation decomposition.

---

## 6. Difference 2: Relation Priors

Vanilla HGT has a learned meta-relation prior:

$$
\mu_{\langle \tau(j), \phi, \tau(i) \rangle}.
$$

This prior can make one relation globally more or less important before looking at node features.

Current HGTeam does not have an explicit $\mu$ term.

Instead, relation importance is learned implicitly through:

- separate `TransformerConv` parameters per relation tuple,
- attention logits from node and edge features,
- the `HeteroConv(..., aggr="sum")` aggregation,
- downstream normalization and policy/critic losses.

## Consequence

Vanilla HGT can learn something like:

> "messages from PV nodes through electrical edges into grid nodes tend to matter more than messages from another relation."

as an explicit prior.

Current HGTeam can still learn that behavior, but it is encoded implicitly in the relation-specific attention and value projections rather than a named prior parameter.

---

## 7. Difference 3: Edge Features

This is the decisive difference for HGTeam.

Vanilla HGT, as exposed by PyG `HGTConv`, does not support arbitrary edge feature vectors in the same way `TransformerConv(edge_dim=...)` does.

Current HGTeam supports:

$$
e_{ji}
\rightarrow
W_E^r e_{ji}
$$

inside both attention and message computation.

## Grid Example

Suppose a grid edge has features:

$$
e_{ji}
=
[\text{status}, \text{resistance}, \text{reactance}].
$$

Then HGTeam can learn:

$$
k_j^r = W_K^r h_j + W_E^r e_{ji}.
$$

So two otherwise similar neighboring grid nodes can receive different attention because their electrical connection differs.

## Critic Action-Edge Example

For the critic, an agent-agent edge can carry another agent's action:

$$
e_{ji} = a_j.
$$

Then destination agent $i$ can compute a counterfactual value conditioned on other agents' actions:

$$
V_i(s, a_{-i}).
$$

Self-loop action edges are zeroed, so agent $i$ does not receive its own action as an edge feature:

$$
e_{ii} = 0.
$$

This gives the intended counterfactual semantics in one GNN forward pass.

---

## 8. Difference 4: Aggregation Across Relations

Vanilla HGT computes messages over typed relations and combines them through the HGT layer's update structure, including typed output transforms and skip behavior.

Current HGTeam uses PyG `HeteroConv` with:

$$
\operatorname{aggr} = \text{sum}.
$$

So if node $i$ receives relation-specific outputs:

$$
\tilde h_{i,r_1}, \tilde h_{i,r_2}, \ldots, \tilde h_{i,r_K},
$$

the combined output is:

$$
h_i' = \sum_{k=1}^K \tilde h_{i,r_k}.
$$

Then HGTeam may apply normalization and activation between layers:

$$
h_i' \leftarrow \operatorname{Act}(\operatorname{Norm}(h_i')).
$$

## Consequence

Vanilla HGT has a more specialized typed update rule.

HGTeam's current stack is simpler: each relation computes its own transformed attention output, and `HeteroConv` sums them. This is easier to combine with arbitrary edge features.

---

## 9. Difference 5: What "Vanilla" Means When Edge Features Are Missing

If edge features are absent, current HGTeam does **not** become vanilla HGT.

It becomes:

$$
\text{HeteroConv}(\text{TransformerConv per full edge tuple}).
$$

That is closer to:

> a relation-typed graph attention network using transformer attention

than to:

> HGTConv with meta-relation priors and factorized type/relation parameters.

## What Is Preserved

- Separate node types.
- Separate edge types.
- Separate relation-specific message passing.
- Transformer-style attention.
- Multi-head attention if configured.
- Heterogeneous input/output dictionaries.

## What Is Not Preserved

- HGT's explicit meta-relation prior $\mu$.
- HGT's exact type-specific projection factorization.
- HGT's relation-specific attention/message matrices in the canonical form.
- HGT's destination-type update/gating structure.

---

## 10. Difference 6: Expressivity vs. Inductive Bias

Current HGTeam is not simply weaker or stronger than vanilla HGT. The tradeoff is more specific.

## Vanilla HGT

Strengths:

- Strong heterogeneous inductive bias.
- Efficient parameter sharing across node and relation types.
- Clear meta-relation interpretation.
- Good paper-level mathematical cleanliness.

Weaknesses for this project:

- No convenient arbitrary edge-feature path in PyG `HGTConv`.
- Harder to encode physical line/switch/transformer attributes directly.
- Harder to encode action-conditioned critic edges directly.

## HGTeam's Current Encoder

Strengths:

- Supports edge features directly.
- Treats physical grid attributes as first-class inputs.
- Supports action-as-edge-feature critic semantics.
- Still relation typed through `HeteroConv`.
- Simple to instantiate across shared-grid and ego-entity graph modes.

Weaknesses:

- Less faithful to vanilla HGT.
- Fewer explicit meta-relation parameters.
- Potentially more parameters because each full edge tuple can own a whole `TransformerConv`.
- The paper must avoid saying "we use HGTConv directly."

---

## 11. What the Paper Should Say

A precise description would be:

> HGTeam uses an edge-aware, relation-typed graph transformer encoder. The design is inspired by HGT's meta-relation view of heterogeneous graphs, but uses PyG `HeteroConv` with relation-specific `TransformerConv` layers rather than vanilla `HGTConv`, because the power-grid and critic graphs require continuous edge attributes. This preserves relation-specific transformer message passing while allowing line, transformer, switch, mapping, and action-edge features to condition attention and messages.

Avoid:

> We use HGT directly through PyG `HGTConv`.

Also avoid:

> HGTConv is an ablation baseline.

That reverses the current implementation.

---

## 12. Cheat Sheet

| Feature | Vanilla HGT / `HGTConv` | Current HGTeam encoder |
|---|---|---|
| Node types | Yes | Yes |
| Edge/relation types | Yes | Yes |
| Full meta-relation prior $\mu$ | Yes | No explicit prior |
| Type-specific Q/K/V projections | Yes, factorized by node type | Relation-specific modules own projections |
| Relation-specific attention/message transforms | Yes, canonical HGT form | Yes, but as full `TransformerConv` modules per edge tuple |
| Arbitrary edge features | Not in the needed PyG `HGTConv` path | Yes via `TransformerConv(edge_dim=...)` |
| Grid physical edge attributes | Awkward / unsupported directly | Supported |
| Action-as-edge-feature critic | Awkward / unsupported directly | Supported |
| Best paper phrase | "Vanilla HGT" | "HGT-inspired edge-aware relation-typed transformer encoder" |

---

## 13. Bottom Line

HGTeam's current encoder is mathematically closest to:

$$
\boxed{
\text{edge-aware relation-typed TransformerConv inside HeteroConv}
}
$$

not:

$$
\boxed{
\text{vanilla HGTConv}
}
$$

When edge features are absent, it is still relation typed and transformer-based, but it does not recover the exact HGT equations. When edge features are present, it departs further from vanilla HGT in a deliberate way: edge attributes become part of attention and message content, which is essential for grid modeling and scalable action-conditioned critics.
