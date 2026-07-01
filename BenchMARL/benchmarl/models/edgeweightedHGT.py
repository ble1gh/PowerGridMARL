from __future__ import annotations

import math
import warnings
from dataclasses import MISSING, dataclass

import torch
import torch_geometric
from tensordict.utils import _unravel_key_to_tuple
from torch import Tensor, nn
from torch_geometric.utils import softmax

from benchmarl.models.common import ModelConfig
from benchmarl.models.heterognn import HeteroGNN


def _edge_type_name(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


class EdgeGateNetwork(nn.Module):
    """Map continuous edge features to low-rank attention/message gates."""

    def __init__(
        self,
        edge_dim: int,
        heads: int,
        low_rank: int,
        hidden_dim: int,
        num_layers: int,
        activation_class: type[nn.Module],
        activation_kwargs: dict | None,
        zero_init: bool,
    ) -> None:
        super().__init__()
        if edge_dim <= 0:
            raise ValueError(f"edge_dim must be positive, got {edge_dim}")
        if low_rank <= 0:
            raise ValueError(f"low_rank must be positive, got {low_rank}")

        activation_kwargs = activation_kwargs or {}
        output_dim = 2 * heads * low_rank
        layers: list[nn.Module] = []
        in_dim = edge_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_class(**activation_kwargs))
            in_dim = hidden_dim
        final = nn.Linear(in_dim, output_dim)
        if zero_init:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.heads = heads
        self.low_rank = low_rank

    def forward(self, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        gates = self.net(edge_attr)
        gates = gates.view(edge_attr.shape[0], self.heads, 2, self.low_rank)
        return gates[:, :, 0, :], gates[:, :, 1, :]


class EdgeWeightedHGTLayer(nn.Module):
    """A vanilla-HGT layer with optional additive low-rank edge modulation.

    The layer deliberately keeps the HGT softmax domain: attention logits from
    all relations arriving at the same destination node type are normalized
    together by destination node index.
    """

    def __init__(
        self,
        in_channels: dict[str, int],
        out_channels: int,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        edge_features_dims: dict | None,
        heads: int,
        low_rank: int,
        edge_gate_hidden_dim: int,
        edge_gate_num_layers: int,
        edge_gate_activation_class: type[nn.Module],
        edge_gate_activation_kwargs: dict | None,
        edge_gate_scale: float,
        zero_init_edge_gates: bool,
        modulate_attention: bool,
        modulate_message: bool,
        allow_missing_edge_features: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if out_channels % heads != 0:
            raise ValueError(
                f"EdgeWeightedHGT requires out_channels divisible by heads, "
                f"got out_channels={out_channels}, heads={heads}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_types = node_types
        self.edge_types = edge_types
        self.edge_features_dims = edge_features_dims or {}
        self.heads = heads
        self.low_rank = low_rank
        self.head_dim = out_channels // heads
        self.edge_gate_scale = edge_gate_scale
        self.modulate_attention = modulate_attention
        self.modulate_message = modulate_message
        self.allow_missing_edge_features = allow_missing_edge_features
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.kqv_lin = nn.ModuleDict(
            {
                node_type: nn.Linear(in_channels[node_type], 3 * out_channels)
                for node_type in node_types
            }
        )
        self.out_lin = nn.ModuleDict(
            {node_type: nn.Linear(out_channels, out_channels) for node_type in node_types}
        )
        self.skip = nn.ParameterDict(
            {node_type: nn.Parameter(torch.ones(1)) for node_type in node_types}
        )

        self.att_rel = nn.ParameterDict()
        self.msg_rel = nn.ParameterDict()
        self.prior_rel = nn.ParameterDict()
        self.att_u = nn.ParameterDict()
        self.att_v = nn.ParameterDict()
        self.msg_u = nn.ParameterDict()
        self.msg_v = nn.ParameterDict()
        self.edge_gates = nn.ModuleDict()

        for edge_type in edge_types:
            name = _edge_type_name(edge_type)
            self.att_rel[name] = nn.Parameter(
                torch.empty(heads, self.head_dim, self.head_dim)
            )
            self.msg_rel[name] = nn.Parameter(
                torch.empty(heads, self.head_dim, self.head_dim)
            )
            self.prior_rel[name] = nn.Parameter(torch.ones(heads))

            edge_dim = self._edge_dim_for_relation(edge_type)
            if low_rank > 0 and edge_dim > 0 and (modulate_attention or modulate_message):
                self.edge_gates[name] = EdgeGateNetwork(
                    edge_dim=edge_dim,
                    heads=heads,
                    low_rank=low_rank,
                    hidden_dim=edge_gate_hidden_dim,
                    num_layers=edge_gate_num_layers,
                    activation_class=edge_gate_activation_class,
                    activation_kwargs=edge_gate_activation_kwargs,
                    zero_init=zero_init_edge_gates,
                )
                if modulate_attention:
                    self.att_u[name] = nn.Parameter(
                        torch.empty(heads, self.head_dim, low_rank)
                    )
                    self.att_v[name] = nn.Parameter(
                        torch.empty(heads, low_rank, self.head_dim)
                    )
                if modulate_message:
                    self.msg_u[name] = nn.Parameter(
                        torch.empty(heads, self.head_dim, low_rank)
                    )
                    self.msg_v[name] = nn.Parameter(
                        torch.empty(heads, low_rank, self.head_dim)
                    )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.kqv_lin.values():
            module.reset_parameters()
        for module in self.out_lin.values():
            module.reset_parameters()
        for param in self.att_rel.values():
            nn.init.xavier_uniform_(param)
        for param in self.msg_rel.values():
            nn.init.xavier_uniform_(param)
        for param in self.att_u.values():
            nn.init.xavier_uniform_(param)
        for param in self.att_v.values():
            nn.init.xavier_uniform_(param)
        for param in self.msg_u.values():
            nn.init.xavier_uniform_(param)
        for param in self.msg_v.values():
            nn.init.xavier_uniform_(param)

    def _edge_dim_for_relation(self, edge_type: tuple[str, str, str]) -> int:
        _src, rel, _dst = edge_type
        edge_dim = self.edge_features_dims.get(rel)
        if edge_dim is None and "interact" in rel:
            edge_dim = self.edge_features_dims.get("interaction")
        return int(edge_dim or 0)

    @staticmethod
    def _low_rank_update(x: Tensor, gate: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # Computes U diag(g(e)) V x without materializing an edge-wise matrix.
        projected = torch.einsum("ehd,hrd->ehr", x, v)
        return torch.einsum("ehr,hdr->ehd", projected * gate, u)

    def _relation_edge_attr(
        self,
        edge_type: tuple[str, str, str],
        edge_attr_dict: dict[tuple[str, str, str], Tensor] | None,
        n_edges: int,
    ) -> Tensor | None:
        edge_dim = self._edge_dim_for_relation(edge_type)
        if edge_dim <= 0:
            return None

        edge_attr = None if edge_attr_dict is None else edge_attr_dict.get(edge_type)
        if edge_attr is None:
            if self.allow_missing_edge_features or n_edges == 0:
                return None
            raise ValueError(
                f"Relation {edge_type} is configured with edge_dim={edge_dim}, "
                "but no edge_attr tensor was provided."
            )
        if edge_attr.shape[-1] != edge_dim:
            raise ValueError(
                f"Relation {edge_type} expected edge_attr dim {edge_dim}, "
                f"got {edge_attr.shape[-1]}."
            )
        return edge_attr

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        edge_attr_dict: dict[tuple[str, str, str], Tensor] | None = None,
    ) -> dict[str, Tensor]:
        q_dict: dict[str, Tensor] = {}
        k_dict: dict[str, Tensor] = {}
        v_dict: dict[str, Tensor] = {}
        for node_type, x in x_dict.items():
            if node_type not in self.kqv_lin:
                continue
            k, q, v = self.kqv_lin[node_type](x).chunk(3, dim=-1)
            q_dict[node_type] = q.view(-1, self.heads, self.head_dim)
            k_dict[node_type] = k.view(-1, self.heads, self.head_dim)
            v_dict[node_type] = v.view(-1, self.heads, self.head_dim)

        incoming: dict[str, list[tuple[Tensor, Tensor, Tensor]]] = {
            node_type: [] for node_type in self.node_types if node_type in x_dict
        }

        for edge_type, edge_index in edge_index_dict.items():
            src, _rel, dst = edge_type
            if src not in k_dict or dst not in q_dict:
                continue
            if edge_type not in self.edge_types:
                continue

            name = _edge_type_name(edge_type)
            src_index, dst_index = edge_index
            n_edges = int(edge_index.shape[1])
            if n_edges == 0:
                continue

            q_i = q_dict[dst][dst_index]
            k_j = k_dict[src][src_index]
            v_j = v_dict[src][src_index]

            rel_key = torch.einsum("ehd,hdf->ehf", k_j, self.att_rel[name])
            rel_msg = torch.einsum("ehd,hdf->ehf", v_j, self.msg_rel[name])

            edge_attr = self._relation_edge_attr(edge_type, edge_attr_dict, n_edges)
            if edge_attr is not None and name in self.edge_gates:
                gate_att, gate_msg = self.edge_gates[name](edge_attr)
                gate_att = gate_att * self.edge_gate_scale
                gate_msg = gate_msg * self.edge_gate_scale
                if self.modulate_attention and name in self.att_u:
                    rel_key = rel_key + self._low_rank_update(
                        k_j, gate_att, self.att_u[name], self.att_v[name]
                    )
                if self.modulate_message and name in self.msg_u:
                    rel_msg = rel_msg + self._low_rank_update(
                        v_j, gate_msg, self.msg_u[name], self.msg_v[name]
                    )

            scores = (q_i * rel_key).sum(dim=-1)
            scores = scores * self.prior_rel[name].view(1, self.heads)
            scores = scores / math.sqrt(self.head_dim)
            incoming[dst].append((scores, dst_index, rel_msg))

        out_dict: dict[str, Tensor] = {}
        for node_type, x in x_dict.items():
            if node_type not in self.out_lin:
                out_dict[node_type] = x
                continue

            n_nodes = x.shape[0]
            out = x.new_zeros(n_nodes, self.heads, self.head_dim)
            if incoming.get(node_type):
                scores = torch.cat([item[0] for item in incoming[node_type]], dim=0)
                dst_index = torch.cat([item[1] for item in incoming[node_type]], dim=0)
                messages = torch.cat([item[2] for item in incoming[node_type]], dim=0)
                alpha = softmax(scores, dst_index, num_nodes=n_nodes)
                weighted = self.dropout(messages * alpha.unsqueeze(-1))
                out_flat = x.new_zeros(n_nodes, self.out_channels)
                out_flat = out_flat.index_add(
                    0, dst_index, weighted.reshape(-1, self.out_channels)
                )
            else:
                out_flat = out.reshape(n_nodes, self.out_channels)

            transformed = self.out_lin[node_type](torch.nn.functional.gelu(out_flat))
            if transformed.shape[-1] == x.shape[-1]:
                alpha_skip = self.skip[node_type].sigmoid()
                transformed = alpha_skip * transformed + (1 - alpha_skip) * x
            out_dict[node_type] = transformed

        return out_dict


class EdgeWeightedHGT(HeteroGNN):
    """BenchMARL heterogeneous graph model with low-rank edge-conditioned HGT layers."""

    def __init__(
        self,
        topology: str,
        self_loops: bool,
        position_key: str | None,
        exclude_pos_from_node_features: bool | None,
        velocity_key: str | None,
        edge_radius: float | None,
        pos_features: int | None,
        vel_features: int | None,
        edge_features_key: str | None = None,
        agent_node_index_key: str | None = None,
        node_types: list[str] | None = None,
        agent_groups: list[str] | None = None,
        node_features_keys: dict[str, str] | None = None,
        edge_types: list[tuple] | None = None,
        node_features_dims: dict[str, int] | None = None,
        edge_features_dims: dict | None = None,
        grid_edge_keys: dict[str, str] | None = None,
        exclude_observations_from_node_features: bool = False,
        cat_observations_to_output: bool = False,
        num_layers: int = 1,
        activation_class: type[nn.Module] = nn.ReLU,
        activation_kwargs: dict | None = None,
        norm_class: type[nn.Module] | None = None,
        norm_kwargs: dict | None = None,
        prune_non_agent_final_layer: bool = False,
        gnn_hidden_dim: int | None = None,
        heads: int = 1,
        low_rank: int = 4,
        edge_gate_hidden_dim: int = 32,
        edge_gate_num_layers: int = 2,
        edge_gate_activation_class: type[nn.Module] = nn.ReLU,
        edge_gate_activation_kwargs: dict | None = None,
        edge_gate_scale: float = 1.0,
        zero_init_edge_gates: bool = True,
        modulate_attention: bool = True,
        modulate_message: bool = True,
        allow_missing_edge_features: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        self.heads = heads
        self.low_rank = low_rank
        self.edge_gate_hidden_dim = edge_gate_hidden_dim
        self.edge_gate_num_layers = edge_gate_num_layers
        self.edge_gate_activation_class = edge_gate_activation_class
        self.edge_gate_activation_kwargs = edge_gate_activation_kwargs
        self.edge_gate_scale = edge_gate_scale
        self.zero_init_edge_gates = zero_init_edge_gates
        self.modulate_attention = modulate_attention
        self.modulate_message = modulate_message
        self.allow_missing_edge_features = allow_missing_edge_features
        self.dropout = dropout

        # Reuse the mature HeteroGNN TensorDict/spec frontend, then replace
        # only the message-passing stack with edge-weighted HGT layers.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            super().__init__(
                topology=topology,
                self_loops=self_loops,
                gnn_class=torch_geometric.nn.TransformerConv,
                gnn_kwargs={"heads": heads, "concat": False, "beta": True},
                position_key=position_key,
                exclude_pos_from_node_features=exclude_pos_from_node_features,
                velocity_key=velocity_key,
                edge_radius=edge_radius,
                pos_features=pos_features,
                vel_features=vel_features,
                edge_features_key=edge_features_key,
                agent_node_index_key=agent_node_index_key,
                node_types=node_types,
                agent_groups=agent_groups,
                node_features_keys=node_features_keys,
                edge_types=edge_types,
                node_features_dims=node_features_dims,
                edge_features_dims=edge_features_dims,
                grid_edge_keys=grid_edge_keys,
                exclude_observations_from_node_features=exclude_observations_from_node_features,
                cat_observations_to_output=cat_observations_to_output,
                num_layers=num_layers,
                activation_class=activation_class,
                activation_kwargs=activation_kwargs,
                norm_class=norm_class,
                norm_kwargs=norm_kwargs,
                prune_non_agent_final_layer=prune_non_agent_final_layer,
                gnn_hidden_dim=gnn_hidden_dim,
                **kwargs,
            )

        if self.gnn_out_channels % self.heads != 0:
            raise ValueError(
                f"EdgeWeightedHGT requires gnn_hidden_dim/output_features divisible by heads, "
                f"got hidden_dim={self.gnn_out_channels}, heads={self.heads}."
            )
        if not self.edge_features_dims:
            # HeteroGNN._forward uses this dictionary as the signal to pass
            # HeteroData.edge_attr_dict into conv layers. Keep that path active
            # for EdgeWeightedHGT without assigning features to any real relation.
            self.edge_features_dims = {"__edgeweightedhgt_edge_attr_forward__": 0}
        self.convs = self._build_edgeweighted_layers().to(self.device)

    def _node_input_dim(self, node_type: str) -> int:
        if node_type in self.input_features_per_group:
            return self.input_features_per_group[node_type]
        if self.node_features_dims and node_type in self.node_features_dims:
            return self.node_features_dims[node_type]
        if self.node_features_keys and node_type in self.node_features_keys:
            feature_key = self.node_features_keys[node_type]
            for key, spec in self.input_spec.items(True, True):
                key_tuple = _unravel_key_to_tuple(key)
                if key_tuple[-1] == feature_key:
                    return spec.shape[-1] if len(spec.shape) > 0 else 1
        return 1

    def _build_edgeweighted_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        hidden_channels = self.gnn_out_channels
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                in_channels = {
                    node_type: self._node_input_dim(node_type) for node_type in self.node_types
                }
            else:
                in_channels = {node_type: hidden_channels for node_type in self.node_types}

            edge_types = []
            for src, rel, dst in self.edge_types:
                if (
                    self.prune_non_agent_final_layer
                    and layer_index == self.num_layers - 1
                    and dst not in self.agent_groups
                ):
                    continue
                edge_types.append((src, rel, dst))

            layers.append(
                EdgeWeightedHGTLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    node_types=self.node_types,
                    edge_types=edge_types,
                    edge_features_dims=self.edge_features_dims,
                    heads=self.heads,
                    low_rank=self.low_rank,
                    edge_gate_hidden_dim=self.edge_gate_hidden_dim,
                    edge_gate_num_layers=self.edge_gate_num_layers,
                    edge_gate_activation_class=self.edge_gate_activation_class,
                    edge_gate_activation_kwargs=self.edge_gate_activation_kwargs,
                    edge_gate_scale=self.edge_gate_scale,
                    zero_init_edge_gates=self.zero_init_edge_gates,
                    modulate_attention=self.modulate_attention,
                    modulate_message=self.modulate_message,
                    allow_missing_edge_features=self.allow_missing_edge_features,
                    dropout=self.dropout,
                )
            )
        return layers


@dataclass
class EdgeWeightedHGTConfig(ModelConfig):
    """Dataclass config for :class:`~benchmarl.models.EdgeWeightedHGT`."""

    topology: str = MISSING
    self_loops: bool = MISSING

    cat_observations_to_output: bool = False

    position_key: str | None = None
    pos_features: int | None = 0
    velocity_key: str | None = None
    vel_features: int | None = 0
    exclude_pos_from_node_features: bool | None = None
    edge_radius: float | None = None

    edge_features_key: str | None = None
    agent_node_index_key: str | None = None

    agent_groups: list[str] | None = None
    node_types: list[str] | None = None
    node_features_keys: dict[str, str] | None = None
    edge_types: list[tuple] | None = None
    node_features_dims: dict[str, int] | None = None
    edge_features_dims: dict | None = None
    grid_edge_keys: dict[str, str] | None = None
    exclude_observations_from_node_features: bool = False
    prune_non_agent_final_layer: bool = False
    gnn_hidden_dim: int | None = None
    num_layers: int = 1

    activation_class: type[nn.Module] = nn.ReLU
    activation_kwargs: dict | None = None
    norm_class: type[nn.Module] | None = None
    norm_kwargs: dict | None = None

    heads: int = 1
    low_rank: int = 4
    edge_gate_hidden_dim: int = 32
    edge_gate_num_layers: int = 2
    edge_gate_activation_class: type[nn.Module] = nn.ReLU
    edge_gate_activation_kwargs: dict | None = None
    edge_gate_scale: float = 1.0
    zero_init_edge_gates: bool = True
    modulate_attention: bool = True
    modulate_message: bool = True
    allow_missing_edge_features: bool = False
    dropout: float = 0.0

    @staticmethod
    def associated_class():
        return EdgeWeightedHGT
