from __future__ import annotations

import importlib
import inspect
from dataclasses import MISSING, dataclass
from math import prod

import torch
import torch.nn.functional as F
import torch_geometric
from tensordict import TensorDictBase
from tensordict.utils import NestedKey, _unravel_key_to_tuple
from torch import Tensor, nn

from benchmarl.models.common import Model, ModelConfig

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    import torch_geometric
    from torch_geometric.transforms import BaseTransform

    class _RelVel(BaseTransform):
        """Transform that reads graph.vel and writes node1.vel - node2.vel in the edge attributes"""

        def __init__(self):
            pass

        def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
            (row, col), vel, pseudo = data.edge_index, data.vel, data.edge_attr

            cart = vel[row] - vel[col]
            cart = cart.view(-1, 1) if cart.dim() == 1 else cart

            if pseudo is not None:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = cart
            return data


TOPOLOGY_TYPES = {"full", "empty", "from_pos", "adjacency"}


class HeteroGNN(Model):
    """A heterogeneous GNN model using HeteroConv and TransformerConv.

    This model supports heterogeneous graphs with multiple edge types and optional edge features.
    It can be used as a decentralized actor or critic (value function).

    Args:
        topology (str): Topology of the graph adoption. Options: "full", "empty", "from_pos", "adjacency".
            This mainly controls the construction of the "default" interaction edges between agents.
        self_loops (bool): Whether to add self-loops to the graph.
        gnn_class (Type[torch_geometric.nn.MessagePassing]): The GNN layer class (recommended: TransformerConv).
        gnn_kwargs (dict, optional): Arguments for the GNN class.
        position_key (str, optional): Key for agent positions (used if topology="from_pos").
        exclude_pos_from_node_features (bool, optional): Whether to exclude position from node features.
        velocity_key (str, optional): Key for agent velocities.
        edge_radius (float, optional): Radius for "from_pos" topology.
        edge_features_key (str, optional): Key for explicit edge features (multi-dimensional adjacency).
            Used if topology="adjacency". Expected shape (..., n_agents, n_agents, n_edge_features).
        pos_features (int, optional): Number of position features.
        vel_features (int, optional): Number of velocity features.
        node_types (List[str], optional): List of node types. Defaults to ["agent"].
        edge_types (List[Tuple[str, str, str]], optional): List of edge types as (src, rel, dst).
            Defaults to [("agent", "interaction", "agent")].
        edge_features_dims (Dict[str, int], optional): Dictionary mapping edge type relation names to feature dimensions.
            e.g., {"interaction": 10}. Used to initialize TransformerConv with edge_dim.

    """

    def __init__(
        self,
        topology: str,
        self_loops: bool,
        gnn_class: type[torch_geometric.nn.MessagePassing],
        gnn_kwargs: dict | None,
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
        **kwargs,
    ):
        self.topology = topology
        self.self_loops = self_loops
        self.position_key = position_key
        self.velocity_key = velocity_key
        self.exclude_pos_from_node_features = exclude_pos_from_node_features
        self.edge_radius = edge_radius
        self.pos_features = pos_features
        self.edge_features_key = edge_features_key
        self.vel_features = vel_features
        self.grid_edge_keys = grid_edge_keys
        self.agent_node_index_key = agent_node_index_key
        self.edge_features_dims = edge_features_dims

        self.agent_groups = (
            agent_groups if agent_groups is not None else [kwargs.get("agent_group")]
        )
        # Ensure agent_groups are in node_types
        default_node_types = (
            self.agent_groups + ["grid_node"]
            if (edge_features_key or grid_edge_keys)
            else self.agent_groups
        )
        self.node_types = node_types if node_types else default_node_types

        self.node_features_keys = node_features_keys if node_features_keys else {}
        self.node_features_dims = node_features_dims if node_features_dims else {}

        # Default edge types: agent <-> agent (for all groups)
        if edge_types is None:
            self.edge_types = []
            for g1 in self.agent_groups:
                for g2 in self.agent_groups:
                    if g1 == g2:
                        # Intra-type: e.g. (EV, EV_self_interact, EV)
                        rel = f"{g1}_self_interact"
                    else:
                        # Cross-type: e.g. (EV, EV_interact_PV, PV)
                        rel = f"{g1}_interact_{g2}"
                    self.edge_types.append((g1, rel, g2))

            if self.grid_edge_keys:
                for rel in self.grid_edge_keys:
                    self.edge_types.append(("grid_node", rel, "grid_node"))

            # Add mapping edges
            if self.agent_node_index_key or (self.grid_edge_keys is not None):
                for g in self.agent_groups:
                    self.edge_types.append((g, "mapping", "grid_node"))
                    self.edge_types.append(("grid_node", "mapping_rev", g))

        else:
            self.edge_types = edge_types

        self.edge_features_dims = edge_features_dims if edge_features_dims else {}
        self.grid_edge_keys = grid_edge_keys
        self.exclude_observations_from_node_features = exclude_observations_from_node_features
        self.cat_observations_to_output = cat_observations_to_output
        self.num_layers = num_layers
        self.prune_non_agent_final_layer = prune_non_agent_final_layer

        self.activation_class = activation_class
        self.activation_kwargs = activation_kwargs if activation_kwargs is not None else {}
        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}

        super().__init__(**kwargs)

        if self.pos_features > 0:
            self.pos_features += 1  # We will add also 1-dimensional distance
        self.edge_features = self.pos_features + self.vel_features

        # Calculate input features per agent group (supports heterogeneous dims)
        self.input_features_per_group = {}
        for group in self.agent_groups:
            dim = 0
            # Add explicit node feature dimensions for this group
            if self.node_features_dims and group in self.node_features_dims:
                dim += self.node_features_dims[group]
            # Add observation dimensions if not excluded
            if not self.exclude_observations_from_node_features:
                dim += sum(
                    spec.shape[-1]
                    for key, spec in self.input_spec.items(True, True)
                    if _unravel_key_to_tuple(key)[-1]
                    not in (position_key, velocity_key, "active_mask")
                    and (
                        self.agent_node_index_key is None
                        or _unravel_key_to_tuple(key)[-1] != self.agent_node_index_key
                    )
                    and group in _unravel_key_to_tuple(key)
                )
            if self.position_key is not None and not self.exclude_pos_from_node_features:
                dim += self.pos_features - 1
            if self.velocity_key is not None:
                dim += self.vel_features
            self.input_features_per_group[group] = dim

        # Backward-compatible scalar: use max across groups (for single-group this is exact)
        self.input_features = (
            max(self.input_features_per_group.values()) if self.input_features_per_group else 0
        )

        # Use dummy features if input is empty
        self.use_dummy_node_features = self.input_features == 0
        if self.use_dummy_node_features:
            self.input_features = 1
            for g in self.input_features_per_group:
                self.input_features_per_group[g] = 1

        self.output_features = self.output_leaf_spec.shape[-1]

        # When the final output is very low-dimensional (e.g. critic value = 1),
        # using output_features as the hidden dim for *all* GNN layers creates a
        # severe bottleneck (e.g. 1 channel × 3 heads = 3-dim hidden space).
        # gnn_hidden_dim lets the user specify a wider intermediate dimension;
        # the existing output_proj layer handles the final mapping.
        self.gnn_out_channels = (
            gnn_hidden_dim if gnn_hidden_dim is not None else self.output_features
        )

        # If concatenating observations to output, we must reserve space in the output vector
        # so that (GNN_Output + Observations) matches the expected output spec.
        if self.cat_observations_to_output:
            obs_dim = 0
            for key, spec in self.input_spec.items(True, True):
                # We only concatenate features belonging to the agent group
                # This logic mirrors the gathering in _forward
                key_tuple = _unravel_key_to_tuple(key)
                is_agent_feature = False
                for g in self.agent_groups:
                    if g in key_tuple:
                        is_agent_feature = True
                        break

                if is_agent_feature:
                    # Exclude pos/vel as they are handled separately (and usually not cat'ed if excluded)
                    # In _forward: ... if key not in (pos, vel)
                    if key_tuple[-1] in (self.position_key, self.velocity_key):
                        continue
                    obs_dim += spec.shape[-1]

            if obs_dim >= self.output_features:
                raise ValueError(
                    f"Output spec size ({self.output_features}) is smaller than observation size ({obs_dim}) "
                    f"but cat_observations_to_output=True. Increase output size (e.g. intermediate_sizes)."
                )

            self.output_features -= obs_dim

        if gnn_kwargs is None:
            gnn_kwargs = {}

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = self.activation_class(**self.activation_kwargs)

        # When concat=True, the actual output dim of each conv layer is
        # heads * out_channels, not just out_channels. We must track
        # this so that subsequent layers use the correct in_channels.
        concat_heads = gnn_kwargs.get("concat", False)
        n_heads = gnn_kwargs.get("heads", 1)
        effective_hidden_dim = (
            self.gnn_out_channels * n_heads if concat_heads else self.gnn_out_channels
        )

        for i in range(self.num_layers):
            conv_dict = {}
            for src, rel, dst in self.edge_types:
                # On the final layer, skip edge types whose destination is not
                # an agent group — their output is discarded by _forward(),
                # leaving those parameters with permanently zero gradients.
                if (
                    self.prune_non_agent_final_layer
                    and i == self.num_layers - 1
                    and dst not in self.agent_groups
                ):
                    continue

                # Determine edge_dim
                edge_dim = self.edge_features_dims.get(rel, None)
                # Fall back to base relation name (e.g. "interaction" for
                # per-pair names like "EV_interact_PV")
                if edge_dim is None and "interact" in rel:
                    edge_dim = self.edge_features_dims.get("interaction", None)
                if edge_dim == 0:
                    edge_dim = None

                if (
                    edge_dim is None
                    and rel == "interaction"
                    and (self.position_key or self.velocity_key)
                ):
                    edge_dim = self.edge_features

                # Layer-specific kwargs
                current_kwargs = gnn_kwargs.copy()
                current_kwargs.update({"out_channels": self.gnn_out_channels})

                # Handle in_channels logic
                if i == 0:
                    # Helper to resolve dims to avoid lazy Init failure
                    def _resolve_dim(nt):
                        # Per-group dimension (supports heterogeneous agent dims)
                        if nt in self.input_features_per_group:
                            return self.input_features_per_group[nt]

                        # Explicit dimension override
                        if self.node_features_dims and nt in self.node_features_dims:
                            return self.node_features_dims[nt]

                        if self.node_features_keys and nt in self.node_features_keys:
                            f_key = self.node_features_keys[nt]
                            for k, v in self.input_spec.items(True, True):
                                kt = _unravel_key_to_tuple(k)
                                if kt[-1] == f_key:
                                    return v.shape[-1]
                        return -1

                    in_channels_src = _resolve_dim(src)
                    in_channels_dst = _resolve_dim(dst)
                else:
                    # Hidden layers: Input is previous layer's actual output dim.
                    # With concat=True the output is heads * out_channels, not
                    # just out_channels.
                    in_channels_src = effective_hidden_dim
                    in_channels_dst = effective_hidden_dim

                # TransformerConv support for edge_dim and lazy init
                if "edge_dim" in inspect.getfullargspec(gnn_class).args and edge_dim is not None:
                    current_kwargs["edge_dim"] = edge_dim

                current_kwargs["in_channels"] = (in_channels_src, in_channels_dst)

                # Use the full edge tuple as key to ensure each edge type gets its own parameters
                # even if they share the same relation name.
                conv_dict[(src, rel, dst)] = gnn_class(**current_kwargs)

            self.convs.append(torch_geometric.nn.HeteroConv(conv_dict, aggr="sum"))

            # Create norms for this layer (one per node type, stored in ModuleDict)
            # We apply norm/act only for intermediate layers, keeping the last layer linear
            # (standard for composable blocks or output heads)
            if self.norm_class and i < self.num_layers - 1:
                self.norms.append(
                    nn.ModuleDict(
                        {
                            node_type: self.norm_class(effective_hidden_dim, **self.norm_kwargs)
                            for node_type in self.node_types
                        }
                    )
                )
            else:
                self.norms.append(None)

        # If concat heads are used, the final GNN layer outputs
        # effective_hidden_dim (= heads * out_channels) which differs from
        # self.output_features. Add a per-node-type projection layer to map
        # back to the expected output_features.
        if effective_hidden_dim != self.output_features:
            # When pruning, only agent groups produce output from the final
            # layer so we only need projection layers for those node types.
            proj_node_types = (
                [nt for nt in self.node_types if nt in self.agent_groups]
                if self.prune_non_agent_final_layer
                else self.node_types
            )
            self.output_proj = nn.ModuleDict(
                {
                    node_type: nn.Linear(effective_hidden_dim, self.output_features)
                    for node_type in proj_node_types
                }
            )
        else:
            self.output_proj = None

        self.convs = self.convs.to(self.device)
        self.norms = self.norms.to(self.device)
        self.act = self.act.to(self.device)
        if self.output_proj is not None:
            self.output_proj = self.output_proj.to(self.device)

        self.edge_index = _get_edge_index(
            topology=self.topology,
            self_loops=self.self_loops,
            device=self.device,
            n_agents=self.n_agents,
        )
        # Unified key-resolution cache: {(suffix, group): resolved_full_key}
        # Populated on first forward pass, eliminates repeated key-tree scans.
        self._resolved_keys = {}
        # Pre-computed eye matrices for self-loop removal: {(n, device): Tensor}
        self._eye_cache = {}

    def _perform_checks(self) -> None:
        # The base Model._perform_checks raises if len(out_keys) > 1.
        # In multi-group mode we intentionally have one out_key per group,
        # so we skip the base check and replicate the rest here.
        if len(self.agent_groups) > 1:
            # Replicate base checks except the out_keys length restriction
            if not self.input_has_agent_dim and not self.centralised:
                raise ValueError(
                    "If input does not have an agent dimension the model should be marked as centralised"
                )
            if self.agent_group in self.input_spec and self.input_spec[
                self.agent_group
            ].shape != (self.n_agents,):
                raise ValueError(
                    "If the agent group is in the input specs, its shape should be the number of agents"
                )
            if self.agent_group in self.output_spec and self.output_spec[
                self.agent_group
            ].shape != (self.n_agents,):
                raise ValueError(
                    "If the agent group is in the output specs, its shape should be the number of agents"
                )
        else:
            super()._perform_checks()

        if self.topology not in TOPOLOGY_TYPES:
            raise ValueError(
                f"Got topology: {self.topology} but only available options are {TOPOLOGY_TYPES}"
            )
        if self.topology == "from_pos" and self.position_key is None:
            raise ValueError("If topology is from_pos, position_key must be provided")
        if (
            self.topology == "adjacency"
            and self.edge_features_key is None
            and self.grid_edge_keys is None
        ):
            raise ValueError(
                "If topology is adjacency, edge_features_key or grid_edge_keys must be provided"
            )
        if self.position_key is not None and self.exclude_pos_from_node_features is None:
            raise ValueError(
                "exclude_pos_from_node_features needs to be specified when position_key is provided"
            )
        if self.position_key is not None and self.pos_features <= 0:
            raise ValueError(f"Position key specified but pos_features is {self.pos_features}")
        elif self.position_key is None and self.pos_features > 0:
            raise ValueError(
                f"If no position_key is given, pos_features needs to be 0, got: {self.pos_features}"
            )
        if self.velocity_key is not None and self.vel_features <= 0:
            raise ValueError(f"Velocity key specified but vel_features is {self.vel_features}")
        elif self.velocity_key is None and self.vel_features > 0:
            raise ValueError(
                f"If no velocity_key is given, vel_features needs to be 0, got: {self.vel_features}"
            )

        if not self.input_has_agent_dim:
            pass  # Multi-group mode: agent_dim absence is valid for shared critic GNN

        for edge in self.edge_types:
            if len(edge) != 3:
                raise ValueError(
                    f"Each element in edge_types must be a tuple of length 3 (src, rel, dst), got {edge}"
                )
            src, rel, dst = edge
            if src not in self.node_types or dst not in self.node_types:
                raise ValueError(
                    f"Edge type {edge} refers to undefined node types. Available node_types: {self.node_types}"
                )

        if not self.exclude_observations_from_node_features:
            # In multi-group mode, different groups may have different n_agents,
            # so we skip the shape validation that assumes a single n_agents.
            if len(self.agent_groups) <= 1:
                input_shape = None
                for input_key, input_spec in self.input_spec.items(True, True):
                    # Skip validation for special keys
                    key_leaf = _unravel_key_to_tuple(input_key)[-1]
                    if key_leaf in (self.position_key, self.velocity_key):
                        continue
                    if (
                        self.agent_node_index_key is not None
                        and key_leaf == self.agent_node_index_key
                    ):
                        continue
                    # Always skip known edge index keys injected by environment
                    if key_leaf == "agent_grid_edge_index":
                        continue
                    if key_leaf == "active_mask":
                        continue
                    if self.grid_edge_keys is not None and key_leaf in self.grid_edge_keys.values():
                        continue
                    if (
                        self.node_features_keys is not None
                        and key_leaf in self.node_features_keys.values()
                    ):
                        continue

                    if len(input_spec.shape) == 2:
                        if input_shape is None:
                            input_shape = input_spec.shape[:-1]
                        else:
                            if input_spec.shape[:-1] != input_shape:
                                raise ValueError(
                                    f"GNN inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                                )
                    else:
                        raise ValueError(
                            f"GNN input value {input_key} from {self.input_spec} has an invalid shape"
                        )

                if input_shape is not None and input_shape[-1] != self.n_agents:
                    raise ValueError(
                        f"The second to last input spec dimension should be the number of agents, got {self.input_spec}"
                    )
        if (
            self.output_has_agent_dim
            and len(self.agent_groups) <= 1
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the GNN output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Run heterogeneous message-passing and return updated embeddings.

        Builds per-group node features from observations, constructs inter-
        and intra-group edges (filtered by ``active_mask``), runs
        TransformerConv layers, and writes group embeddings back into
        *tensordict* under each group's ``out_key``.
        """
        # Determine device from model parameters
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # Fallback if no parameters
            device = self.device

        # Build keys list once for all lookups in this forward pass
        all_keys = list(tensordict.keys(True, True))

        x_dict = {}
        pos_dict = {}
        vel_dict = {}
        obs_dict = {}  # Store observations for concatenation if needed
        active_masks = {}  # Per-group boolean active masks for edge filtering

        # Gather inputs for each agent group (node type)
        for group in self.agent_groups:
            # Collect active_mask for edge filtering (inactive agents excluded from message passing)
            am_key = self._resolve_key(all_keys, "active_mask", group)
            if am_key is not None:
                am_val = tensordict.get(am_key)
                if am_val is not None:
                    active_masks[group] = am_val.bool()

            # Check if this agent group has explicit node features defined
            # If so, we'll use those instead of observations
            has_explicit_features = (
                self.node_features_keys is not None and group in self.node_features_keys
            )

            # Reusable logic to fetch observations for this group
            # We fetch them if we need them for node features OR for output concatenation
            # Note: observations are fetched even when has_explicit_features=True,
            # because __init__ computes input_features as explicit + obs dims.
            observations = []
            if (
                not self.exclude_observations_from_node_features
            ) or self.cat_observations_to_output:
                observations = [
                    tensordict.get(in_key)
                    for in_key in self.in_keys
                    if group in _unravel_key_to_tuple(in_key)
                    and _unravel_key_to_tuple(in_key)[-1]
                    not in (self.position_key, self.velocity_key, "active_mask")
                ]

            if self.cat_observations_to_output and observations:
                obs_dict[group] = torch.cat(observations, dim=-1)

            input_list = []

            # Add explicit node features if provided
            if has_explicit_features:
                # Load explicit agent node features (e.g., participation_score)
                key = self.node_features_keys[group]
                full_key = self._resolve_key(all_keys, key, None)
                if full_key:
                    explicit_feats = tensordict.get(full_key)
                    input_list.append(explicit_feats)

            # Add observations if not excluded (independent of explicit features)
            if not self.exclude_observations_from_node_features:
                input_list.extend(observations)

            # Retrieve position and velocity for this group
            pos = None
            vel = None
            if self.position_key is not None:
                pos_key = self._resolve_key(all_keys, self.position_key, group)
                if pos_key:
                    pos = tensordict.get(pos_key)
                    if not self.exclude_pos_from_node_features:
                        input_list.append(pos)
                    pos_dict[group] = pos

            if self.velocity_key is not None:
                vel_key = self._resolve_key(all_keys, self.velocity_key, group)
                if vel_key:
                    vel = tensordict.get(vel_key)
                    input_list.append(vel)
                    vel_dict[group] = vel

            if not input_list and self.use_dummy_node_features:
                # Use tensordict.batch_size for env batch dims and self.n_agents
                # for the agent count.  Avoids picking up LSTM hidden states or
                # other multi-dimensional tensors as the shape reference.
                batch_dims = tensordict.batch_size  # e.g. (256,) or ()
                n_agents_in_group = self.n_agents
                dummy = torch.zeros(
                    *batch_dims, n_agents_in_group, 1, device=device, dtype=torch.float
                )
                input_list.append(dummy)

            if input_list:
                x_group = torch.cat(input_list, dim=-1)
                x_dict[group] = x_group

        # Load explicit node features for NON-AGENT types (e.g., grid_node)
        # Agent groups are handled above
        if self.node_features_keys:
            for node_type, key in self.node_features_keys.items():
                if node_type not in x_dict and node_type not in self.agent_groups:
                    full_key = self._resolve_key(all_keys, key, None)
                    if full_key:
                        x_dict[node_type] = tensordict.get(full_key)

        edge_attr_dense_dict = {}
        if self.grid_edge_keys:
            for rel, key in self.grid_edge_keys.items():
                full_key = self._resolve_key(all_keys, key, None)
                if full_key:
                    edge_attr_dense_dict[rel] = tensordict.get(full_key)

        # Legacy support
        if self.edge_features_key is not None and not edge_attr_dense_dict:
            full_edge_key = self._resolve_key(all_keys, self.edge_features_key, None)
            val = tensordict.get(full_edge_key)
            # Default legacy name 'edge'
            if val is not None:
                edge_attr_dense_dict["edge"] = val

        agent_grid_edge_indices = {}
        if self.agent_node_index_key is not None:
            # Support per-type mapping edges: try {group}_agent_grid_edge_index for each group
            for group in self.agent_groups:
                per_type_key = f"{group}_agent_grid_edge_index"
                full_key = self._resolve_key(all_keys, per_type_key, None)
                if full_key is not None:
                    agent_grid_edge_indices[group] = tensordict.get(full_key)
            # Fallback: single shared agent_grid_edge_index (backward compat)
            if not agent_grid_edge_indices:
                full_agent_idx_key = self._resolve_key(all_keys, self.agent_node_index_key, None)
                val = tensordict.get(full_agent_idx_key)
                if val is not None:
                    # Assign to first (or only) agent group for backward compat
                    agent_grid_edge_indices[self.agent_groups[0]] = val

        # Ensure all inputs are on the correct device
        x_dict = {k: v.to(device) for k, v in x_dict.items()}

        # Enforce explicit dimensions (padding if needed)
        if self.node_features_dims:
            for node_type, expected_dim in self.node_features_dims.items():
                if node_type in x_dict and expected_dim > 0:
                    current = x_dict[node_type]
                    if current.shape[-1] != expected_dim:
                        diff = expected_dim - current.shape[-1]
                        if diff > 0:
                            x_dict[node_type] = F.pad(current, (0, diff))
                elif node_type not in x_dict and expected_dim > 0:
                    # If the node type is missing from inputs but we expect it,
                    # we should try to create it with correct dimensions (zeros)
                    # This is critical for grid_node which might be missing but needs Dim=2
                    ref = None
                    n_cnt = 0

                    # Try to get N from edge_attr (specific for grid_node)
                    if node_type == "grid_node" and edge_attr_dense_dict:
                        ref_k = next(iter(edge_attr_dense_dict.keys()))
                        ref = edge_attr_dense_dict[ref_k]
                        n_cnt = ref.shape[-2]

                    if ref is not None and n_cnt > 0:
                        batch_shape = ref.shape[:-2]
                        x_dict[node_type] = torch.zeros(
                            *batch_shape, n_cnt, expected_dim, device=ref.device, dtype=torch.float
                        )

        pos_dict = {k: v.to(device) for k, v in pos_dict.items()}
        vel_dict = {k: v.to(device) for k, v in vel_dict.items()}
        edge_attr_dense_dict = {k: v.to(device) for k, v in edge_attr_dense_dict.items()}

        # Build HeteroData Batch
        edge_index = self.edge_index
        if edge_index is not None:
            edge_index = edge_index.to(device)

        # Collect per-group action tensors for action-edge-features
        # (critic_use_other_actions via edge attributes).
        # Only enabled for critic GNNs with action_edge_features flag.
        action_dict = None
        if getattr(self, "_use_action_edge_features", False):
            action_dict = {}
            act_dim = self.edge_features_dims.get("interaction", 1)
            for group in self.agent_groups:
                act_key = self._resolve_key(all_keys, "action", group)
                act_val = tensordict.get(act_key) if act_key is not None else None
                if act_val is not None:
                    action_dict[group] = act_val.to(device)
                else:
                    # Fallback: zero actions (setup/warmup calls or vmap passes
                    # where action values are not populated).
                    action_dict[group] = torch.zeros(
                        *x_dict[group].shape[:-1], act_dim, device=device
                    )

        data_batch = _tensordict_to_hetero_data(
            x_dict=x_dict,
            pos_dict=pos_dict,
            vel_dict=vel_dict,
            edge_index=edge_index,
            edge_attr_dense_dict=edge_attr_dense_dict,
            agent_grid_edge_indices=agent_grid_edge_indices,
            self_loops=self.self_loops,
            edge_radius=self.edge_radius,
            device=device,
            edge_types=self.edge_types,
            node_types=self.node_types,
            agent_groups=self.agent_groups,
            eye_cache=self._eye_cache,
            active_masks=active_masks if active_masks else None,
            action_dict=action_dict,
        )

        # Forward pass
        x_dict_out = data_batch.x_dict

        for i, conv in enumerate(self.convs):
            edge_attr = None
            if self.edge_features > 0 or self.edge_features_dims:
                try:
                    edge_attr = data_batch.edge_attr_dict
                except KeyError:
                    pass

            if edge_attr is not None:
                x_dict_out = conv(x_dict_out, data_batch.edge_index_dict, edge_attr)
            else:
                x_dict_out = conv(x_dict_out, data_batch.edge_index_dict)

            if i < self.num_layers - 1:
                # Apply Norm
                if self.norms[i]:
                    temp_dict = {}
                    for node_key, x in x_dict_out.items():
                        if node_key in self.norms[i]:
                            temp_dict[node_key] = self.norms[i][node_key](x)
                        else:
                            temp_dict[node_key] = x
                    x_dict_out = temp_dict

                # Apply Activation
                x_dict_out = {k: self.act(v) for k, v in x_dict_out.items()}

        # Project from effective_hidden_dim back to output_features if needed
        if self.output_proj is not None:
            x_dict_out = {
                k: self.output_proj[k](v) if k in self.output_proj else v
                for k, v in x_dict_out.items()
            }

        out_dict = x_dict_out

        # Map outputs back to TensorDict
        for group in self.agent_groups:
            if group in out_dict:
                res_agent = out_dict[group]
                if group in x_dict:
                    batch_size = x_dict[group].shape[:-2]
                    n_agents = x_dict[group].shape[-2]
                    res = res_agent.view(*batch_size, n_agents, self.output_features)

                    # Zero out embeddings for inactive agents (safety net)
                    if group in active_masks:
                        am = active_masks[group].to(res.device)
                        while am.dim() < res.dim():
                            am = am.unsqueeze(-1)
                        res = res * am.float()

                    # Concatenate original observations if requested
                    if self.cat_observations_to_output and not self.centralised:
                        if group in obs_dict:
                            res = torch.cat([res, obs_dict[group]], dim=-1)

                    if self.centralised:
                        res = res.mean(dim=-2)

                    if group == self.agent_group:
                        tensordict.set(self.out_key, res)
                    else:
                        leaf_key = _unravel_key_to_tuple(self.out_key)[-1]
                        tensordict.set((group, leaf_key), res)

        return tensordict

    def _resolve_key(
        self, all_keys: list[NestedKey], key: str, group: str | None = None
    ) -> NestedKey | None:
        """Resolve a TensorDict key suffix, caching the result across forward passes."""
        cache_key = (key, group)
        if cache_key not in self._resolved_keys:
            result = self._get_key_terminating_with(all_keys, key, group)
            if result is not None:
                self._resolved_keys[cache_key] = result
            return result
        return self._resolved_keys[cache_key]

    def _get_key_terminating_with(
        self, keys: list[NestedKey], key: str, group: str | None = None
    ) -> NestedKey | None:
        """Find the first nested key in *keys* ending with *key*, optionally scoped to *group*."""
        for k in keys:
            k_tuple = _unravel_key_to_tuple(k)
            if k_tuple[-1] == key:
                if group is None or group in k_tuple:
                    if k_tuple[0] != "next":
                        return k
        # If group is None, try looser match
        if group is None:
            for k in keys:
                k_tuple = _unravel_key_to_tuple(k)
                if k_tuple[-1] == key and k_tuple[0] != "next":
                    return k
        return None


def _get_edge_index(topology: str, self_loops: bool, n_agents: int, device: str):
    if topology == "full":
        adjacency = torch.ones(n_agents, n_agents, device=device, dtype=torch.long)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)
        if not self_loops:
            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    elif topology == "empty":
        if self_loops:
            edge_index = (
                torch.arange(n_agents, device=device, dtype=torch.long).unsqueeze(0).repeat(2, 1)
            )
        else:
            edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
    elif topology == "from_pos" or topology == "adjacency":
        edge_index = None
    else:
        raise ValueError(f"Topology {topology} not supported")

    return edge_index


def _tensordict_to_hetero_data(
    x_dict: dict[str, Tensor],
    edge_index: Tensor | None,
    self_loops: bool,
    device: str,
    edge_types: list[tuple],
    node_types: list[str],
    agent_groups: list[str] = None,
    pos_dict: dict[str, Tensor] = None,
    vel_dict: dict[str, Tensor] = None,
    edge_attr_dense_dict: dict[str, Tensor] = None,
    agent_grid_edge_indices: dict[str, Tensor] = None,
    edge_radius: float | None = None,
    eye_cache: dict = None,
    active_masks: dict[str, Tensor] | None = None,
    action_dict: dict[str, Tensor] | None = None,
) -> torch_geometric.data.Batch:
    # Infer batch size — prefer grid adjacencies (shape ...batch, N, N, F)
    # over x_dict, because agent tensors might have extra dims (e.g. LSTM states)
    if edge_attr_dense_dict:
        _adj = next(iter(edge_attr_dense_dict.values()))
        batch_size = prod(_adj.shape[:-3]) if _adj.ndim > 3 else 1
    elif x_dict:
        first_group = next(iter(x_dict.values()))
        batch_size = prod(first_group.shape[:-2])
    else:
        batch_size = 1

    b = torch.arange(batch_size, device=device)

    # Create the HeteroData batch directly
    batch_data = torch_geometric.data.HeteroData()

    # Helper to get grid node count
    n_grid_nodes = 0
    if edge_attr_dense_dict:
        count_tensor = next(iter(edge_attr_dense_dict.values()))
        n_grid_nodes = count_tensor.shape[-2]

    # Initialize Agent node types
    for node_type in node_types:
        if node_type in x_dict:
            x_agent = x_dict[node_type]
            n_agents = x_agent.shape[-2]

            x_agent_flat = x_agent.view(-1, x_agent.shape[-1])
            batch_vector = torch.repeat_interleave(b, n_agents)

            batch_data[node_type].x = x_agent_flat
            batch_data[node_type].batch = batch_vector
            batch_data[node_type].ptr = torch.arange(
                0, (batch_size + 1) * n_agents, n_agents, device=device
            )

        elif node_type == "grid_node" or node_type not in x_dict:
            if n_grid_nodes > 0:
                count = n_grid_nodes
                # Initialize dummy features for grid nodes
                # (Batch*N, 1)
                batch_data[node_type].x = torch.zeros(batch_size * count, 1, device=device)
                batch_vector_node = torch.repeat_interleave(b, count)
                batch_data[node_type].batch = batch_vector_node
                batch_data[node_type].ptr = torch.arange(
                    0, (batch_size + 1) * count, count, device=device
                )

    # Dictionary to store pre-calculated dense edge indices to avoid recalculation
    dense_edge_indices = {}

    # Process dense edge features if present (GRID ADJACENCY DICT)
    if edge_attr_dense_dict:
        for rel_name, edge_attr_dense in edge_attr_dense_dict.items():
            n_grid_nodes = edge_attr_dense.shape[-2]  # (Batch, N, N, F)

            edge_attr_dense_flat = edge_attr_dense.view(batch_size, n_grid_nodes, n_grid_nodes, -1)

            mask = edge_attr_dense_flat.abs().sum(dim=-1) > 0  # (Batch, N, N)
            if not self_loops:
                # Remove diagonal (eye matrix is cached to avoid re-allocation)
                eye_key = (n_grid_nodes, device)
                if eye_cache is not None and eye_key in eye_cache:
                    diag_mask = eye_cache[eye_key]
                else:
                    diag_mask = torch.eye(n_grid_nodes, device=device, dtype=torch.bool).unsqueeze(
                        0
                    )
                    if eye_cache is not None:
                        eye_cache[eye_key] = diag_mask
                mask &= ~diag_mask

            nonzero_indices = mask.nonzero()
            b_idx = nonzero_indices[:, 0]
            row = nonzero_indices[:, 1]
            col = nonzero_indices[:, 2]

            # Create global graph indices
            src = row + b_idx * n_grid_nodes
            dst = col + b_idx * n_grid_nodes
            batched_idx = torch.stack([src, dst], dim=0)
            batched_attr = edge_attr_dense_flat[b_idx, row, col]

            dense_edge_indices[rel_name] = (batched_idx, batched_attr)

    # Default agent_groups from x_dict keys if not provided (backward compat)
    if agent_groups is None:
        agent_groups = [k for k in x_dict if k != "grid_node"]
    if agent_grid_edge_indices is None:
        agent_grid_edge_indices = {}

    # Prepare mapping (agent <-> grid_node) per agent type
    batched_mapping_edge_indices = {}
    for group, agent_grid_edge_index in agent_grid_edge_indices.items():
        if agent_grid_edge_index is None or group not in x_dict:
            continue
        agent_grid_edge_index = agent_grid_edge_index.to(device)
        if agent_grid_edge_index.dim() == 2:
            agent_grid_edge_index = agent_grid_edge_index.unsqueeze(0).repeat(batch_size, 1, 1)

        n_mapped_agents = x_dict[group].shape[-2]
        M = agent_grid_edge_index.shape[-1]

        b_idx = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, M).view(-1)
        agent_grid_edge_index_flat = agent_grid_edge_index.view(batch_size, 2, M)

        local_src = agent_grid_edge_index_flat[:, 0, :].reshape(-1)
        local_dst = agent_grid_edge_index_flat[:, 1, :].reshape(-1)

        global_src = local_src + b_idx * n_mapped_agents
        global_dst = local_dst + b_idx * n_grid_nodes

        batched_mapping_edge_indices[group] = torch.stack([global_src, global_dst], dim=0)

    # Handle Edge Indices for all types
    for src, rel, dst in edge_types:
        can_build_edge = False
        current_edge_index = None
        current_edge_attr = None

        # 1. Predefined simple topology (for agent-agent interaction edges)
        is_src_agent = src in agent_groups
        is_dst_agent = dst in agent_groups
        if (
            edge_index is not None
            and src in x_dict
            and dst in x_dict
            and src == dst
            and is_src_agent
        ):
            x_src = x_dict[src]
            n_agents = x_src.shape[-2]
            n_edges = edge_index.shape[1]
            batch_repeat = torch.repeat_interleave(b, n_edges)
            batch_edge_index = edge_index.repeat(1, batch_size) + batch_repeat * n_agents
            current_edge_index = batch_edge_index
            can_build_edge = True

        # 1b. Build full agent graph when topology="adjacency" but no explicit edge data
        elif (
            edge_index is None
            and src in x_dict
            and dst in x_dict
            and is_src_agent
            and is_dst_agent
            and rel not in dense_edge_indices
        ):
            x_src = x_dict[src]
            x_dst = x_dict[dst]
            n_src = x_src.shape[-2]
            n_dst = x_dst.shape[-2]
            if src == dst:
                # Intra-type full graph
                full_adj = torch.ones(n_src, n_src, device=device, dtype=torch.long)
                agent_edge_index, _ = torch_geometric.utils.dense_to_sparse(full_adj)
                if not self_loops:
                    agent_edge_index, _ = torch_geometric.utils.remove_self_loops(agent_edge_index)
            else:
                # Cross-type full bipartite graph
                rows = torch.arange(n_src, device=device).repeat_interleave(n_dst)
                cols = torch.arange(n_dst, device=device).repeat(n_src)
                agent_edge_index = torch.stack([rows, cols], dim=0)
            n_edges = agent_edge_index.shape[1]
            batch_repeat = torch.repeat_interleave(b, n_edges)
            # Offset by per-type node counts
            src_offset = batch_repeat * n_src
            dst_offset = batch_repeat * n_dst
            current_edge_index = torch.stack(
                [
                    agent_edge_index[0].repeat(batch_size) + src_offset,
                    agent_edge_index[1].repeat(batch_size) + dst_offset,
                ],
                dim=0,
            )
            can_build_edge = True

        # 2. Grid Adjacency (Matches explicit relation name)
        elif rel in dense_edge_indices and "grid" in src and src == dst:
            current_edge_index, current_edge_attr = dense_edge_indices[rel]
            can_build_edge = True

        # 3. Radius Graph
        elif pos_dict is not None and src in pos_dict and dst in pos_dict:
            pos_src = pos_dict[src].view(-1, pos_dict[src].shape[-1])
            pos_dst = pos_dict[dst].view(-1, pos_dict[dst].shape[-1])
            batch_src = batch_data[src].batch
            batch_dst = batch_data[dst].batch

            if src == dst:
                current_edge_index = torch_geometric.nn.pool.radius_graph(
                    pos_src, batch=batch_src, r=edge_radius, loop=self_loops
                )
            else:
                row, col = torch_geometric.nn.pool.radius(
                    x=pos_src, y=pos_dst, r=edge_radius, batch_x=batch_src, batch_y=batch_dst
                )
                current_edge_index = torch.stack([col, row], dim=0)

            can_build_edge = True

        # 4. Mapping Edges (Per-type agent <-> grid_node)
        elif "mapping" in rel and batched_mapping_edge_indices:
            # Determine which agent group this edge type refers to
            if is_src_agent and "grid" in dst and src in batched_mapping_edge_indices:
                current_edge_index = batched_mapping_edge_indices[src]
                can_build_edge = True
            elif "grid" in src and is_dst_agent and dst in batched_mapping_edge_indices:
                current_edge_index = torch.flip(batched_mapping_edge_indices[dst], [0])
                can_build_edge = True

        if can_build_edge and current_edge_index is not None:
            # --- Active mask edge filtering for agent-agent edges ---
            # Remove edges where source or destination is an inactive agent.
            if active_masks is not None and is_src_agent and is_dst_agent:
                n_src = x_dict[src].shape[-2]
                n_dst = x_dict[dst].shape[-2]
                # Build flat boolean masks: (batch_size * n_agents,)
                src_am = active_masks.get(src)
                dst_am = active_masks.get(dst)
                if src_am is not None or dst_am is not None:
                    if src_am is not None:
                        src_flat = src_am.reshape(-1)  # (batch_size * n_src,)
                    else:
                        src_flat = torch.ones(batch_size * n_src, dtype=torch.bool, device=device)
                    if dst_am is not None:
                        dst_flat = dst_am.reshape(-1)
                    else:
                        dst_flat = torch.ones(batch_size * n_dst, dtype=torch.bool, device=device)
                    # Keep only edges where both endpoints are active
                    keep = src_flat[current_edge_index[0]] & dst_flat[current_edge_index[1]]
                    current_edge_index = current_edge_index[:, keep]
                    if current_edge_attr is not None:
                        current_edge_attr = current_edge_attr[keep]

            # --- Action edge features for agent-agent edges (Option B) ---
            # For edge (j→i), edge_attr = a_j.  Self-loops get edge_attr = 0
            # so agent i never sees its own action → V(s, a_{-i}) semantics.
            if action_dict is not None and is_src_agent and is_dst_agent:
                src_actions = action_dict.get(src)
                if src_actions is not None:
                    n_src = x_dict[src].shape[-2]
                    # Flatten actions to (batch_size * n_src, action_dim)
                    act_dim = src_actions.shape[-1]
                    act_flat = src_actions.reshape(-1, act_dim)  # (B*n_src, d)
                    # Look up sender action for each edge
                    edge_act = act_flat[current_edge_index[0]]  # (n_edges, d)
                    # Zero self-loop actions (src == dst node index means self-loop)
                    if src == dst:
                        is_self_loop = current_edge_index[0] == current_edge_index[1]
                        edge_act = edge_act.clone()
                        edge_act[is_self_loop] = 0.0
                    # Attach as edge_attr (add to existing if present)
                    if current_edge_attr is not None:
                        current_edge_attr = torch.cat([current_edge_attr, edge_act], dim=-1)
                    else:
                        current_edge_attr = edge_act

            batch_data[src, rel, dst].edge_index = current_edge_index
            if current_edge_attr is not None:
                batch_data[src, rel, dst].edge_attr = current_edge_attr

    return batch_data.to(device)


@dataclass
class HeteroGnnConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.HeteroGNN`."""

    topology: str = MISSING
    self_loops: bool = MISSING
    gnn_class: type[torch_geometric.nn.MessagePassing] = MISSING

    cat_observations_to_output: bool = False
    gnn_kwargs: dict | None = None

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

    @staticmethod
    def associated_class():
        return HeteroGNN
