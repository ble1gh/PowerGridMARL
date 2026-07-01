"""Ego-entity GNN for SMACv2 environments.

Each agent constructs a small ego-centric heterogeneous graph with node types:
- ``self_entity``: the agent's own features (own_dim)
- ``enemy``: all enemy entity features (entity_dim each)
- ``{type}_ally``: per-type ally features (entity_dim each)

Message passing via TransformerConv computes an embedding for the
``self_entity`` node, which is used downstream as the GNN output.

The forward pass folds (B, N_group) into a flat batch of small ego graphs
via PyG batching, runs message passing, then unfolds back to (B, N_group, output_dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as tgnn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torch_geometric.data import HeteroData, Batch


class EgoEntityGNNWrapper(TensorDictModuleBase):
    """TensorDictModule wrapper for EgoEntityGNN.

    Reads per-group entity observations from the tensordict, runs the GNN,
    and writes per-group ``gnn_embedding`` keys back.
    """

    def __init__(self, ego_gnn: "EgoEntityGNN", all_groups: list[str]):
        # Build in/out key lists
        in_keys = []
        out_keys = []
        for g in all_groups:
            in_keys.extend([
                (g, "entity_self"),
                (g, "entity_enemy"),
                (g, "active_mask"),
            ])
            for at in all_groups:
                in_keys.append((g, f"entity_{at}_ally"))
            out_keys.append((g, "gnn_embedding"))

        super().__init__()
        self.ego_gnn = ego_gnn
        self.all_groups = all_groups
        self.in_keys = in_keys
        self.out_keys = out_keys

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        results = self.ego_gnn(tensordict)
        for group, embedding in results.items():
            tensordict.set((group, "gnn_embedding"), embedding)
        return tensordict


class EgoEntityGNN(nn.Module):
    """Ego-centric entity GNN for per-agent entity observations.

    Args:
        group_map: Agent group name → list of agent indices.
        all_groups: Ordered list of agent group names.
        n_enemies: Number of enemy entities.
        entity_dim: Feature dimension for enemy/ally entities.
        own_dim: Feature dimension for the agent's own features.
        move_feats_dim: Dimension of move features (not used in GNN, passed through).
        ally_type_max: Dict[group_name, max_allies_of_that_type].
        output_dim: GNN output embedding dimension.
        num_layers: Number of TransformerConv layers.
        heads: Number of attention heads.
        concat_heads: Whether to concatenate (True) or average (False) heads.
        use_beta: Whether to use beta (learnable skip) in TransformerConv.
        self_loops: Whether to add self-loops.
        topology: ``"star"`` (self-node is hub) or ``"full"`` (all-to-all).
        norm_class: Optional normalization class name (e.g., ``"LayerNorm"``).
        device: Torch device.
    """

    def __init__(
        self,
        group_map: dict[str, list],
        all_groups: list[str],
        n_enemies: int,
        entity_dim: int,
        own_dim: int,
        move_feats_dim: int,
        ally_type_max: dict[str, int],
        output_dim: int,
        num_layers: int = 2,
        heads: int = 2,
        concat_heads: bool = False,
        use_beta: bool = True,
        self_loops: bool = True,
        topology: str = "star",
        norm_class: str | None = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.group_map = group_map
        self.all_groups = all_groups
        self.n_enemies = n_enemies
        self.entity_dim = entity_dim
        self.own_dim = own_dim
        self.move_feats_dim = move_feats_dim
        self.ally_type_max = ally_type_max
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.topology = topology
        self.self_loops = self_loops

        # Node types: self_entity, enemy, {type}_ally for each type
        self.node_types = ["self_entity", "enemy"] + [f"{g}_ally" for g in all_groups]

        # Input projection to common hidden dim
        hidden_dim = output_dim
        self.input_projs = nn.ModuleDict()
        self.input_projs["self_entity"] = nn.Linear(own_dim, hidden_dim)
        self.input_projs["enemy"] = nn.Linear(entity_dim, hidden_dim)
        for g in all_groups:
            self.input_projs[f"{g}_ally"] = nn.Linear(entity_dim, hidden_dim)

        # Build edge types based on topology
        self.edge_types = self._build_edge_types()

        # TransformerConv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if norm_class else None

        for layer_i in range(num_layers):
            conv_dict = {}
            for src, rel, dst in self.edge_types:
                in_channels = hidden_dim
                out_channels = hidden_dim
                conv_dict[(src, rel, dst)] = tgnn.TransformerConv(
                    in_channels=in_channels,
                    out_channels=out_channels // heads if concat_heads else out_channels,
                    heads=heads,
                    concat=concat_heads,
                    beta=use_beta,
                    edge_dim=None,
                )
            self.convs.append(tgnn.HeteroConv(conv_dict, aggr="sum"))

            if norm_class:
                norm_dict = {}
                for nt in self.node_types:
                    norm_dict[nt] = self._make_norm(norm_class, hidden_dim)
                self.norms.append(nn.ModuleDict(norm_dict))

        # Output projection for self_entity node
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.to(device)

    def _build_edge_types(self) -> list[tuple[str, str, str]]:
        """Build edge type list based on topology."""
        entity_node_types = ["enemy"] + [f"{g}_ally" for g in self.all_groups]

        if self.topology == "star":
            # Star: self_entity ↔ each entity type
            edges = []
            for nt in entity_node_types:
                edges.append((nt, f"{nt}_to_self", "self_entity"))
                edges.append(("self_entity", f"self_to_{nt}", nt))
            if self.self_loops:
                edges.append(("self_entity", "self_loop", "self_entity"))
            return edges

        elif self.topology == "full":
            # Full: all-to-all between all node types
            all_nt = ["self_entity"] + entity_node_types
            edges = []
            for src in all_nt:
                for dst in all_nt:
                    if src == dst:
                        if self.self_loops:
                            edges.append((src, f"{src}_self_loop", dst))
                    else:
                        edges.append((src, f"{src}_to_{dst}", dst))
            return edges
        else:
            raise ValueError(f"Unknown topology: {self.topology}")

    @staticmethod
    def _make_norm(norm_class_name: str, dim: int) -> nn.Module:
        if norm_class_name == "LayerNorm":
            return nn.LayerNorm(dim)
        elif norm_class_name == "BatchNorm1d":
            return nn.BatchNorm1d(dim)
        elif norm_class_name == "InstanceNorm1d":
            return nn.InstanceNorm1d(dim)
        else:
            return nn.LayerNorm(dim)

    def forward(self, tensordict) -> dict[str, torch.Tensor]:
        """Run ego-entity GNN on all agent groups.

        Args:
            tensordict: TensorDict with per-group entity observations.
                Expected keys per group ``g``:
                - ``(g, "entity_self")``: ``(B, N_g, own_dim)``
                - ``(g, "entity_enemy")``: ``(B, N_g, n_enemies, entity_dim)``
                - ``(g, "entity_{t}_ally")``: ``(B, N_g, max_allies_t, entity_dim)``
                - ``(g, "active_mask")``: ``(B, N_g)``

        Returns:
            Dict mapping group name → embedding tensor ``(B, N_g, output_dim)``.
        """
        results = {}

        for group in self.all_groups:
            # Get entity observations
            entity_self = tensordict.get((group, "entity_self"))  # (B, N_g, own_dim) or (N_g, own_dim)
            entity_enemy = tensordict.get((group, "entity_enemy"))  # (B, N_g, n_enemies, ed) or (N_g, n_enemies, ed)
            active_mask = tensordict.get((group, "active_mask"))  # (B, N_g) or (N_g,)

            # Handle unbatched case
            if entity_self.dim() == 2:
                entity_self = entity_self.unsqueeze(0)
                entity_enemy = entity_enemy.unsqueeze(0)
                active_mask = active_mask.unsqueeze(0)

            B, N_g = entity_self.shape[:2]

            # Collect ally features per type
            ally_feats = {}
            for at in self.all_groups:
                key = f"entity_{at}_ally"
                af = tensordict.get((group, key))  # (B, N_g, max_at, ed)
                if af.dim() == 3:
                    af = af.unsqueeze(0)
                ally_feats[at] = af

            # Fold B*N_g into flat batch of ego graphs
            embeddings = self._forward_ego_graphs(
                entity_self, entity_enemy, ally_feats, active_mask, B, N_g
            )
            results[group] = embeddings  # (B, N_g, output_dim)

        return results

    def _forward_ego_graphs(
        self,
        entity_self: torch.Tensor,  # (B, N_g, own_dim)
        entity_enemy: torch.Tensor,  # (B, N_g, n_enemies, entity_dim)
        ally_feats: dict[str, torch.Tensor],  # {type: (B, N_g, max_at, entity_dim)}
        active_mask: torch.Tensor,  # (B, N_g)
        B: int,
        N_g: int,
    ) -> torch.Tensor:
        """Build and process ego graphs for all agents in a group."""
        device = entity_self.device
        BN = B * N_g

        # --- Build node features (fold B*N_g into batch dim) ---
        # self_entity: one node per ego graph → (BN, own_dim)
        self_feats = entity_self.reshape(BN, self.own_dim)

        # enemy: n_enemies nodes per ego graph → (BN * n_enemies, entity_dim)
        enemy_feats = entity_enemy.reshape(BN, self.n_enemies, self.entity_dim)

        # ally: per type
        ally_feats_flat = {}
        for at in self.all_groups:
            max_at = self.ally_type_max[at]
            ally_feats_flat[at] = ally_feats[at].reshape(BN, max_at, self.entity_dim)

        # --- Project to hidden dim ---
        x_dict = {}
        x_dict["self_entity"] = self.input_projs["self_entity"](self_feats)  # (BN, hidden)

        enemy_flat = enemy_feats.reshape(BN * self.n_enemies, self.entity_dim)
        x_dict["enemy"] = self.input_projs["enemy"](enemy_flat)  # (BN*n_enemies, hidden)

        for at in self.all_groups:
            max_at = self.ally_type_max[at]
            af = ally_feats_flat[at].reshape(BN * max_at, self.entity_dim)
            x_dict[f"{at}_ally"] = self.input_projs[f"{at}_ally"](af)  # (BN*max_at, hidden)

        # --- Build edge indices ---
        edge_index_dict = self._build_edge_indices(BN, device)

        # --- Build batch vectors ---
        batch_dict = {}
        b_vec = torch.arange(BN, device=device)
        batch_dict["self_entity"] = b_vec  # (BN,)
        batch_dict["enemy"] = b_vec.repeat_interleave(self.n_enemies)
        for at in self.all_groups:
            max_at = self.ally_type_max[at]
            batch_dict[f"{at}_ally"] = b_vec.repeat_interleave(max_at)

        # --- Build HeteroData ---
        data = HeteroData()
        for nt in self.node_types:
            data[nt].x = x_dict[nt]
            data[nt].batch = batch_dict[nt]

        for (src, rel, dst), ei in edge_index_dict.items():
            data[src, rel, dst].edge_index = ei

        # --- Message passing ---
        x_dict_cur = {nt: data[nt].x for nt in self.node_types}

        for layer_i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict_cur, edge_index_dict)
            # Apply normalization + residual + ReLU
            for nt in self.node_types:
                if nt in x_dict_new:
                    if self.norms is not None:
                        x_dict_new[nt] = self.norms[layer_i][nt](x_dict_new[nt])
                    x_dict_new[nt] = x_dict_new[nt] + x_dict_cur[nt]  # residual
                    x_dict_new[nt] = torch.relu(x_dict_new[nt])
                else:
                    x_dict_new[nt] = x_dict_cur[nt]
            x_dict_cur = x_dict_new

        # --- Extract self_entity embedding ---
        self_embed = x_dict_cur["self_entity"]  # (BN, hidden)
        output = self.output_proj(self_embed)  # (BN, output_dim)

        # Zero inactive agents
        active_flat = active_mask.reshape(BN)
        output = output * active_flat.unsqueeze(-1).float()

        # Unfold back to (B, N_g, output_dim)
        return output.reshape(B, N_g, self.output_dim)

    def _build_edge_indices(
        self, BN: int, device: torch.device
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """Build batched edge indices for ego graphs.

        In each ego graph:
        - self_entity has 1 node (index 0)
        - enemy has n_enemies nodes (indices 0..n_enemies-1)
        - {type}_ally has max_allies nodes

        For batched graphs, we offset by graph_id * n_nodes_of_type.
        """
        edge_index_dict = {}
        b = torch.arange(BN, device=device)

        for src_type, rel, dst_type in self.edge_types:
            n_src = self._n_nodes_of_type(src_type)
            n_dst = self._n_nodes_of_type(dst_type)

            if src_type == dst_type:
                # Self-loops: connect each node to itself
                n = n_src
                # For each graph, self-loop on each node
                node_ids = torch.arange(n, device=device)
                # Batch: offset by graph_id * n
                src_ids = (b.unsqueeze(1) * n + node_ids.unsqueeze(0)).reshape(-1)
                dst_ids = src_ids.clone()
                edge_index_dict[(src_type, rel, dst_type)] = torch.stack([src_ids, dst_ids])
            else:
                # Cross-type edges: bipartite
                src_nodes = torch.arange(n_src, device=device)
                dst_nodes = torch.arange(n_dst, device=device)

                # All src → all dst (or limited for star)
                if self.topology == "star":
                    # Star: edges only between entity types and self_entity
                    if n_src == 1:
                        # self_entity → entity_type: 1 src to each dst
                        src_local = torch.zeros(n_dst, dtype=torch.long, device=device)
                        dst_local = dst_nodes
                    elif n_dst == 1:
                        # entity_type → self_entity: each src to 1 dst
                        src_local = src_nodes
                        dst_local = torch.zeros(n_src, dtype=torch.long, device=device)
                    else:
                        # Should not happen in star topology
                        continue
                else:
                    # Full: all-to-all bipartite
                    src_local = src_nodes.repeat_interleave(n_dst)
                    dst_local = dst_nodes.repeat(n_src)

                n_edges = src_local.shape[0]
                # Batch offset
                src_offset = b.unsqueeze(1) * n_src  # (BN, 1)
                dst_offset = b.unsqueeze(1) * n_dst

                src_ids = (src_offset + src_local.unsqueeze(0)).reshape(-1)
                dst_ids = (dst_offset + dst_local.unsqueeze(0)).reshape(-1)
                edge_index_dict[(src_type, rel, dst_type)] = torch.stack([src_ids, dst_ids])

        return edge_index_dict

    def _n_nodes_of_type(self, node_type: str) -> int:
        if node_type == "self_entity":
            return 1
        elif node_type == "enemy":
            return self.n_enemies
        else:
            # {type}_ally
            group_name = node_type.replace("_ally", "")
            return self.ally_type_max[group_name]
