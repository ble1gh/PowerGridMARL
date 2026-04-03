#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import warnings
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type, List

import torch
from torch import nn
import torch_geometric.nn as tgnn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.hgteam_modules import (
    EmbeddingProcessor,
    HyperNetworkJoiner,
    merge_embedding_losses,
    reparameterize,
)
from benchmarl.models.common import ModelConfig
from benchmarl.beta_param_extractor import BetaParamExtractor
from benchmarl.independent_beta import IndependentBeta
from benchmarl.tanh_normal_entropy import TanhNormalWithEntropy
from benchmarl.models import MlpConfig, LstmConfig, HeteroGnnConfig
from benchmarl.models.heterognn import HeteroGNN

class HGTeamLoss(ClipPPOLoss):
    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictBase
    critic_network_params: TensorDictBase
    target_actor_network_params: TensorDictBase
    target_critic_network_params: TensorDictBase

    def __init__(self, algorithm, group, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm
        self.group = group

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # --- PRE-FORWARD: make participation scores grad-enabled ----
        # Participation scores are environment observations (no grad by default).
        # We clone+detach+requires_grad BEFORE super().forward() so the actor
        # consumes the grad-enabled version and .grad is populated after backward.
        self._grad_participation_ref = None
        self._grad_embedding_z_ref = None

        if self.algorithm.gnn_agent_node_feature_key is not None:
            part_key = f"{self.group}_{self.algorithm.gnn_agent_node_feature_key}"
            part = tensordict.get(part_key, None)
            if part is None:
                part = tensordict.get((self.group, part_key), None)
                part_key = (self.group, part_key)
            if part is not None:
                part_clone = part.clone().detach().requires_grad_(True)
                tensordict.set(part_key, part_clone)
                part_clone.retain_grad()
                self._grad_participation_ref = part_clone

        # --- Snapshot z tensors BEFORE any mutations -----------------------
        # Capture originals for ALL groups so that VIB SNI (and future
        # transforms like detach_z) can mutate freely; a single restore pass
        # at the end is order-independent.
        _z_stash: dict[tuple, torch.Tensor] = {}
        _z_keys = ("embedding_z", "embedding_z_token")
        for g in self.algorithm.group_map.keys():
            for zk in _z_keys:
                val = tensordict.get((g, zk), None)
                if val is not None:
                    _z_stash[(g, zk)] = val

        # --- VIB SNI: use deterministic mu for PPO loss computation --
        # Selective Noise Injection (Igl et al., NeurIPS 2019): during the
        # PPO update, replace stochastic z_query with its deterministic mean
        # so the importance ratio is not corrupted by re-sampled noise.
        # The stochastic z is restored afterwards for the VIB KL term.
        if self.algorithm.use_vib:
            embedding_mu = tensordict.get((self.group, "embedding_mu"), None)
            embedding_z_pre = tensordict.get((self.group, "embedding_z"), None)
            if embedding_mu is not None and embedding_z_pre is not None:
                tensordict.set((self.group, "embedding_z"), embedding_mu)

        # --- FORWARD: standard PPO losses --------------------------
        out = super().forward(tensordict)

        # --- Restore z tensors from snapshot ------------------------
        for key, val in _z_stash.items():
            tensordict.set(key, val)

        # --- POST-FORWARD: retain_grad on embedding_z --------------
        embedding_z = tensordict.get((self.group, "embedding_z"), None)
        if embedding_z is not None and embedding_z.requires_grad:
            embedding_z.retain_grad()
            self._grad_embedding_z_ref = embedding_z

        # Also retain_grad for z_token in split-z mode
        z_token = tensordict.get((self.group, "embedding_z_token"), None)
        if z_token is not None and z_token.requires_grad:
            z_token.retain_grad()
            self._grad_embedding_z_token_ref = z_token

        # If embeddings are missing (because PPO used saved params from rollout),
        # run the actor module to generate them and attach gradients.
        if embedding_z is None and self.algorithm.gnn_mode != "none":
            self.actor_network.get_dist_params(tensordict)
            embedding_z = tensordict.get((self.group, "embedding_z"), None)
            if embedding_z is not None and embedding_z.requires_grad:
                embedding_z.retain_grad()
                self._grad_embedding_z_ref = embedding_z

        # Compute and merge embedding auxiliary losses into loss_objective
        merge_embedding_losses(
            self.algorithm, self.group, tensordict, out, "loss_objective"
        )

        return out

# EmbeddingProcessor, HyperNetworkJoiner, reparameterize, and
# merge_embedding_losses are imported from hgteam_modules.py.


class HGTeamBase(Algorithm):
    """Shared architecture base for HGTeam variants (PPO, SAC, HAPPO).

    Encapsulates the GNN encoder, embedding processor, actor pipeline builder,
    and auxiliary embedding losses.  Subclasses add loss-specific params
    (e.g. clip_epsilon for PPO, alpha_init for SAC).
    """

    def __init__(
        self,
        share_param_critic: bool,
        scale_mapping: str,
        scale_lb: float,
        use_tanh_normal: bool,
        use_beta: bool,
        beta_min_param: float,
        share_critic_across_groups: bool = False,
        centralised_value_per_agent: bool = False,
        gnn_mode: str = "none",  # "none", "concat", "hypernetwork", or "learned_query"
        z_dim: int = None,
        hypernet_actor_feature_dim: int = None,
        stochastic_z: bool = False,
        embedding_entropy_coef: float = 0.0,
        embedding_diversity_coef: float = 0.0,
        # GNN configuration parameters
        gnn_num_layers: int = 2,
        gnn_heads: int = 4,
        gnn_concat_heads: bool = False,
        gnn_use_beta: bool = True,
        gnn_self_loops: bool = True,
        gnn_agent_node_feature_key: Optional[str] = "participation_score",
        gnn_agent_node_feature_dim: int = 1,
        gnn_norm_class: Optional[str] = None,
        critic_embed_dim: int = 32,
        # Split-z parameters (learned_query mode)
        split_z: bool = False,
        z_token_dim: int = 32,
        z_query_dim: int = 32,
        stochastic_z_query: bool = True,
        # VIB (Variational Information Bottleneck) parameters
        use_vib: bool = False,
        vib_beta: float = 0.01,
        vib_warmup_frames: int = 500_000,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_critic = share_param_critic
        self.scale_mapping = scale_mapping
        self.scale_lb = scale_lb
        self.use_tanh_normal = use_tanh_normal
        self.use_beta = use_beta
        self.beta_min_param = beta_min_param
        self.share_critic_across_groups = share_critic_across_groups
        self.centralised_value_per_agent = centralised_value_per_agent
        self.gnn_mode = gnn_mode
        if gnn_mode not in ("none", "concat", "hypernetwork", "learned_query"):
            raise ValueError(f"gnn_mode must be 'none', 'concat', 'hypernetwork', or 'learned_query', got '{gnn_mode}'")
        self.z_dim = z_dim
        self.hypernet_actor_feature_dim = hypernet_actor_feature_dim
        self.stochastic_z = stochastic_z

        # Mode-specific validation (hard errors)
        if gnn_mode != "none" and z_dim is None:
            raise ValueError(f"z_dim is required when gnn_mode='{gnn_mode}'")
        if gnn_mode == "hypernetwork" and hypernet_actor_feature_dim is None:
            raise ValueError("hypernet_actor_feature_dim is required when gnn_mode='hypernetwork'")
        if gnn_mode != "none" and split_z and gnn_mode != "learned_query":
            raise ValueError(f"split_z is only supported with gnn_mode='learned_query', got '{gnn_mode}'")

        # Mode-specific validation (warnings for ignored flags)
        if gnn_mode == "none":
            _ignored = []
            if stochastic_z:
                _ignored.append("stochastic_z")
            if z_dim is not None:
                _ignored.append("z_dim")
            if hypernet_actor_feature_dim is not None:
                _ignored.append("hypernet_actor_feature_dim")
            if split_z:
                _ignored.append("split_z")
            if embedding_entropy_coef > 0:
                _ignored.append("embedding_entropy_coef")
            if embedding_diversity_coef > 0:
                _ignored.append("embedding_diversity_coef")
            if _ignored:
                warnings.warn(
                    f"gnn_mode='none' but the following GNN-related flags are set "
                    f"and will be ignored: {', '.join(_ignored)}",
                    stacklevel=2,
                )
        else:
            if gnn_mode != "hypernetwork" and hypernet_actor_feature_dim is not None:
                warnings.warn(
                    f"hypernet_actor_feature_dim={hypernet_actor_feature_dim} is set "
                    f"but only used when gnn_mode='hypernetwork' (current: '{gnn_mode}')",
                    stacklevel=2,
                )
            if split_z and stochastic_z:
                warnings.warn(
                    "stochastic_z=True is ignored when split_z=True; "
                    "use stochastic_z_query to control stochasticity in split-z mode",
                    stacklevel=2,
                )
            if not split_z:
                _split_ignored = []
                if z_token_dim != 32:
                    _split_ignored.append(f"z_token_dim={z_token_dim}")
                if z_query_dim != 32:
                    _split_ignored.append(f"z_query_dim={z_query_dim}")
                if not stochastic_z_query:
                    _split_ignored.append("stochastic_z_query=False")
                if _split_ignored:
                    warnings.warn(
                        f"split_z=False but split-z params are set to non-defaults "
                        f"and will be ignored: {', '.join(_split_ignored)}",
                        stacklevel=2,
                    )

        self.embedding_entropy_coef = embedding_entropy_coef
        self.embedding_diversity_coef = embedding_diversity_coef

        # GNN configuration
        self.gnn_num_layers = gnn_num_layers
        self.gnn_heads = gnn_heads
        self.gnn_concat_heads = gnn_concat_heads
        self.gnn_use_beta = gnn_use_beta
        self.gnn_self_loops = gnn_self_loops
        self.gnn_agent_node_feature_key = gnn_agent_node_feature_key
        self.gnn_agent_node_feature_dim = gnn_agent_node_feature_dim

        # Resolve norm class string → nn.Module class (or None)
        _norm_map = {
            "layernorm": nn.LayerNorm,
            "batchnorm1d": nn.BatchNorm1d,
            "instancenorm1d": nn.InstanceNorm1d,
            "groupnorm": nn.GroupNorm,
        }
        if gnn_norm_class is not None and gnn_norm_class.lower() != "none":
            key = gnn_norm_class.lower().replace("_", "")
            if key not in _norm_map:
                raise ValueError(
                    f"Unknown gnn_norm_class '{gnn_norm_class}'. "
                    f"Supported: {list(_norm_map.keys())}"
                )
            self.gnn_norm_class = _norm_map[key]
        else:
            self.gnn_norm_class = None

        self.critic_embed_dim = critic_embed_dim

        # Split-z: separate GNN output into z_token (KV) and z_query (cross-attn)
        self.split_z = split_z
        self.z_token_dim = z_token_dim
        self.z_query_dim = z_query_dim
        self.stochastic_z_query = stochastic_z_query

        # VIB: KL(q(z|x) || N(0,I)) with beta warm-up and selective noise injection
        self.use_vib = use_vib
        self.vib_beta = vib_beta
        self.vib_warmup_frames = vib_warmup_frames

        self.shared_critic_module = None
        self._shared_projection = None
        self._shared_gnn_critic = None
        self._shared_actor_gnn = None


    def _filter_shared_gnn_params(
        self,
        all_params: list,
        shared_gnn,
        group: str,
        role: str = "actor",
    ) -> list:
        """Remove shared-GNN parameters whose gradients are structurally zero
        for *group*'s loss.

        Specifically, on the **final** GNN layer, conv sub-modules whose
        destination node type is NOT *group* produce hidden states that are
        never consumed by this group's output path.  Their parameters receive
        zero gradients from this group's backward, polluting Adam's momentum
        and variance estimates with zeros.  Similarly, ``output_proj`` entries
        for other groups are unused.

        Layer-0 (and intermediate) conv params, norms, and the group's own
        final-layer convs/proj all DO receive gradients and are kept.
        """
        from benchmarl.models.heterognn import HeteroGNN

        # Unwrap TensorDictModule / TensorDictSequential wrappers to find
        # the underlying HeteroGNN nn.Module.
        gnn_module = shared_gnn
        for _ in range(10):  # bounded unwrap
            if isinstance(gnn_module, HeteroGNN):
                break
            if hasattr(gnn_module, 'module'):
                gnn_module = gnn_module.module
            else:
                break

        if not isinstance(gnn_module, HeteroGNN):
            return all_params  # Can't identify GNN; return as-is

        # Build set of data_ptr()s to exclude.  We use data_ptr (the raw
        # storage address) because TorchRL's actor_network_params values may
        # be views/references rather than the exact same Python objects as the
        # nn.Parameter instances on the module.
        exclude_ptrs = set()

        final_layer_idx = gnn_module.num_layers - 1
        if 0 <= final_layer_idx < len(gnn_module.convs):
            final_conv = gnn_module.convs[final_layer_idx]
            for edge_key, conv_module in final_conv.convs.items():
                # edge_key is (src, rel, dst)
                dst = edge_key[2] if isinstance(edge_key, tuple) else None
                if dst is not None and dst != group and dst in gnn_module.agent_groups:
                    for p in conv_module.parameters():
                        exclude_ptrs.add(p.data_ptr())

        if gnn_module.output_proj is not None:
            for node_type, proj in gnn_module.output_proj.items():
                if node_type != group and node_type in gnn_module.agent_groups:
                    for p in proj.parameters():
                        exclude_ptrs.add(p.data_ptr())

        if not exclude_ptrs:
            return all_params

        filtered = [p for p in all_params if p.data_ptr() not in exclude_ptrs]
        return filtered


    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        # Determine actor output based on gnn_mode
        if self.gnn_mode == "hypernetwork":
            # Hypernetwork mode: MLP outputs features, GNN embedding generates weights
            feature_dim = self.hypernet_actor_feature_dim
            actor_output_spec = Composite(
                {
                    group: Composite(
                        {"actor_features": Unbounded(shape=(n_agents, feature_dim))},
                        shape=(n_agents,),
                    )
                }
            )
        else:
            # None, Concat, or Learned-query mode: actor outputs logits directly
            actor_output_spec = Composite(
                {
                    group: Composite(
                        {"logits": Unbounded(shape=logits_shape)},
                        shape=(n_agents,),
                    )
                }
            )

        # Instantiate Base/Actor Model (first stream)
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )

        # Build GNN module if gnn_mode is not "none"
        if self.gnn_mode != "none":
            # Use the shared actor GNN that sees ALL agent groups as separate
            # node types with cross-group interaction edges. This enables
            # the encoder to learn role differentiation between agent types.

            # --- Compute GNN output dimensionality ----------------------
            if self.split_z and self.gnn_mode == "learned_query":
                # Split-z mode: GNN produces [z_token | z_query_raw]
                z_query_raw_dim = (
                    self.z_query_dim * 2 if self.stochastic_z_query
                    else self.z_query_dim
                )
                gnn_output_dim = self.z_token_dim + z_query_raw_dim
            else:
                gnn_output_dim = (
                    self.z_dim * 2
                    if self.stochastic_z
                    else self.z_dim
                )

            gnn_stream_module = self._get_or_build_shared_actor_gnn(gnn_output_dim)

            # --- Create per-group embedding processor -------------------
            if self.split_z and self.gnn_mode == "learned_query":
                embedding_processor = EmbeddingProcessor(
                    embedding_dim=gnn_output_dim,
                    stochastic=False,  # legacy flag unused in split path
                    split_z=True,
                    z_token_dim=self.z_token_dim,
                    z_query_dim=self.z_query_dim,
                    stochastic_query=self.stochastic_z_query,
                )
                # Outputs: z_token, z_query, query_mean, query_logvar
                # We also write z_query as "embedding_z" for backward compat
                # with diversity/entropy loss computation.
                embedding_processor_module = TensorDictModule(
                    embedding_processor,
                    in_keys=[(group, "gnn_embedding")],
                    out_keys=[
                        (group, "embedding_z_token"),
                        (group, "embedding_z"),       # z_query (alias)
                        (group, "embedding_mu"),       # pre-noise mean (for VIB KL)
                        (group, "embedding_logvar"),
                    ],
                )
            else:
                embedding_processor = EmbeddingProcessor(
                    embedding_dim=gnn_output_dim,
                    stochastic=self.stochastic_z,
                )
                embedding_processor_module = TensorDictModule(
                    embedding_processor,
                    in_keys=[(group, "gnn_embedding")],
                    out_keys=[(group, "embedding_z"), (group, "embedding_mu"), (group, "embedding_logvar")],
                )

        if self.gnn_mode == "hypernetwork":
            # Hypernetwork mode: GNN embeddings generate weights for actor
            feature_dim = self.hypernet_actor_feature_dim
            
            joiner = HyperNetworkJoiner(
                embedding_dim=gnn_output_dim,           # From GNN (2x if stochastic)
                feature_dim=feature_dim,                # From Actor
                output_dim=logits_shape[-1],
                device=self.device,
                stochastic_embedding=self.stochastic_z,
            )
            
            joiner_module = TensorDictModule(
                joiner,
                in_keys=[(group, "actor_features"), (group, "gnn_embedding")],
                out_keys=[(group, "logits"), (group, "embedding_z"), (group, "embedding_logvar")]
            )

            # Sequence: Actor -> GNN -> Joiner
            actor_module = TensorDictSequential(
                actor_module, 
                gnn_stream_module, 
                joiner_module
            )
            
        elif self.gnn_mode == "concat":
            # Concat mode: GNN embeddings concatenated with observations as input to MLP
            # Flow: GNN -> EmbeddingProcessor -> Concat with obs -> MLP -> logits
            
            # Get the latent dim (after stochastic processing)
            latent_dim = self.z_dim
            
            # Get observation shape for this group
            obs_shape = self.observation_spec[group, "observation"].shape
            obs_dim = obs_shape[-1]  # Last dimension is feature dim
            
            # Create concatenation module
            def concat_obs_embedding(obs, embedding_z):
                # obs: (..., n_agents, obs_dim)
                # embedding_z: (..., n_agents, latent_dim)
                return torch.cat([obs, embedding_z], dim=-1)
            
            concat_module = TensorDictModule(
                concat_obs_embedding,
                in_keys=[(group, "observation"), (group, "embedding_z")],
                out_keys=[(group, "concat_input")]
            )
            
            # Create new actor input spec with concatenated features
            concat_input_spec = Composite(
                {
                    group: Composite(
                        {"concat_input": Unbounded(shape=(n_agents, obs_dim + latent_dim))},
                        shape=(n_agents,),
                    )
                }
            )
            
            # Create MLP that takes concatenated input
            mlp_actor = model_config.get_model(
                input_spec=concat_input_spec,
                output_spec=actor_output_spec,
                agent_group=group,
                input_has_agent_dim=True,
                n_agents=n_agents,
                centralised=False,
                share_params=self.experiment_config.share_policy_params,
                device=self.device,
                action_spec=self.action_spec,
            )
            
            # Sequence: GNN -> EmbeddingProcessor -> Concat -> MLP
            actor_module = TensorDictSequential(
                gnn_stream_module,
                embedding_processor_module,
                concat_module,
                mlp_actor
            )

        elif self.gnn_mode == "learned_query":
            # Learned-query mode: GNN produces embedding(s) which the actor
            # (e.g. a Transformer with use_z_as_query=True) reads from the
            # tensordict during its forward pass.
            # When split_z=True the tensordict will contain both
            # "embedding_z_token" (for KV prepend) and "embedding_z" (query).
            # When split_z=False the single "embedding_z" serves both roles.
            # Flow: GNN -> EmbeddingProcessor -> Actor (reads embedding keys)
            actor_module = TensorDictSequential(
                gnn_stream_module,
                embedding_processor_module,
                actor_module,
            )

        if continuous:
            if self.use_beta:
                # Use Beta distribution for bounded [0,1] actions
                extractor_module = TensorDictModule(
                    BetaParamExtractor(min_param=self.beta_min_param),
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "alpha"), (group, "beta")],
                )
                policy = ProbabilisticActor(
                    module=TensorDictSequential(actor_module, extractor_module),
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "alpha"), (group, "beta")],
                    out_keys=[(group, "action")],
                    distribution_class=IndependentBeta,
                    distribution_kwargs={
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    },
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                extractor_module = TensorDictModule(
                    NormalParamExtractor(scale_mapping=self.scale_mapping, scale_lb=self.scale_lb),
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "loc"), (group, "scale")],
                )
                policy = ProbabilisticActor(
                    module=TensorDictSequential(actor_module, extractor_module),
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "loc"), (group, "scale")],
                    out_keys=[(group, "action")],
                    distribution_class=(
                        IndependentNormal if not self.use_tanh_normal else TanhNormalWithEntropy
                    ),
                    distribution_kwargs=(
                        {
                            "low": self.action_spec[(group, "action")].space.low,
                            "high": self.action_spec[(group, "action")].space.high,
                        }
                        if self.use_tanh_normal
                        else {}
                    ),
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )

        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )

        return policy


    def _compute_embedding_losses(
        self, group: str, batch: TensorDictBase
    ) -> dict:
        """Compute embedding-related losses from the batch after forward pass.
        
        Returns dict with:
        - embedding_entropy: Raw entropy value (for logging)
        - embedding_diversity: Raw diversity value (for logging)
        - loss_embedding_entropy: Penalizes high variance (encourages certainty)
        - loss_embedding_diversity: Rewards L2 distance between agent embeddings
        - embedding_z_token_norm: Norm of z_token (split-z mode only)
        - diag_actor_features_mean/std: Actor feature statistics (hypernetwork mode)
        - diag_logits_mean/std/min/max: Logits statistics before Beta extraction
        - diag_alpha_beta_mean_diff: Mean difference between alpha-half and beta-half logits
        """
        losses = {}
        
        # Get embedding stats from batch
        # In split-z mode, "embedding_z" is an alias for z_query
        embedding_z = batch.get((group, "embedding_z"), None)
        if not torch.is_tensor(embedding_z):
            # Embeddings not found - this means hypernetwork is not enabled or forward failed
            return losses
            
        embedding_logvar = batch.get((group, "embedding_logvar"), None)
        if not torch.is_tensor(embedding_logvar):
            embedding_logvar = None

        # Always log embedding (z_query) norm for debugging
        losses["embedding_z_norm"] = embedding_z.norm(dim=-1).mean().detach()

        # Log z_token norm when in split-z mode
        z_token = batch.get((group, "embedding_z_token"), None)
        if z_token is not None:
            losses["embedding_z_token_norm"] = z_token.norm(dim=-1).mean().detach()

        if embedding_logvar is not None:
            # Mean entropy across all dimensions and agents
            entropy = 0.5 * (1 + embedding_logvar)  # Simplified, ignoring constants
            entropy_mean = entropy.mean()
            losses["embedding_entropy"] = entropy_mean.detach()
            
            if self.embedding_entropy_coef > 0:
                # We want to MINIMIZE entropy, so we add positive entropy as loss
                losses["loss_embedding_entropy"] = entropy_mean * self.embedding_entropy_coef

        # --- VIB: KL(q(z|x) || N(0,I)) with beta warm-up ---
        # Alemi et al. 2017 "Deep Variational Information Bottleneck"
        # Beta warm-up: Bowman et al. 2016 / Fu et al. 2019
        if self.use_vib and embedding_logvar is not None:
            embedding_mu = batch.get((group, "embedding_mu"), None)
            if torch.is_tensor(embedding_mu):
                # KL per agent: sum over latent dims, mean over agents & batch
                # KL(N(mu,sigma^2) || N(0,I)) = -0.5 * sum_j(1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
                kl_per_element = -0.5 * (
                    1 + embedding_logvar - embedding_mu.pow(2) - embedding_logvar.exp()
                )
                kl_per_agent = kl_per_element.sum(dim=-1)  # sum over z_query_dim
                kl_mean = kl_per_agent.mean()  # mean over agents & batch

                # Linear beta warm-up over collected frames.
                # total_frames is maintained by Experiment and survives
                # checkpoint resume (restored from state_dict).
                total_frames = self.experiment.total_frames
                if self.vib_warmup_frames > 0:
                    beta_effective = min(
                        self.vib_beta,
                        self.vib_beta * total_frames / self.vib_warmup_frames,
                    )
                else:
                    beta_effective = self.vib_beta

                losses["vib_kl"] = kl_mean.detach()
                losses["vib_beta_effective"] = torch.tensor(beta_effective, device=embedding_z.device).detach()
                losses["vib_mu_norm"] = embedding_mu.norm(dim=-1).mean().detach()
                losses["vib_std_mean"] = torch.exp(0.5 * embedding_logvar).mean().detach()
                losses["loss_vib_kl"] = kl_mean * beta_effective

        # Diversity: pairwise cosine similarity distance loss
        # Computes L_d = 1/(n(n-1)) * sum_{i} sum_{j!=i} Norm(1 - cos_sim(z_i, z_j))
        # where Norm is min-max normalization over all pairwise distances
        # embedding_z shape: (..., n_agents, embedding_dim)
        if embedding_z.dim() >= 3:
            n = embedding_z.shape[-2]
            if n >= 2:
                # L2-normalize embeddings for cosine similarity
                z_norm = torch.nn.functional.normalize(embedding_z, dim=-1)
                # Pairwise cosine similarity: (..., n_agents, n_agents)
                cos_sim = torch.matmul(z_norm, z_norm.transpose(-2, -1))
                # Pairwise distance: 1 - cos_sim  (range [0, 2])
                pairwise_dist = 1.0 - cos_sim

                # Extract off-diagonal elements (i != j)
                mask = ~torch.eye(n, dtype=torch.bool, device=embedding_z.device)
                off_diag = pairwise_dist[..., mask]  # (..., n*(n-1))

                # Min-max normalization per batch element (Eq. 9)
                d_min = off_diag.min(dim=-1, keepdim=True).values
                d_max = off_diag.max(dim=-1, keepdim=True).values
                normalized = (off_diag - d_min) / (d_max - d_min + 1e-12)

                # Mean over all pairs, then over batch
                diversity = normalized.mean(dim=-1).mean()
            else:
                diversity = torch.tensor(0.0, device=embedding_z.device)

            losses["embedding_diversity"] = diversity.detach()

            if self.embedding_diversity_coef > 0:
                 # We want to MAXIMIZE diversity, so we MINIMIZE negative diversity
                 losses["loss_embedding_diversity"] = -diversity * self.embedding_diversity_coef
        
        # Additional diagnostics specific to hypernetwork mode
        if self.gnn_mode == "hypernetwork":
            # Actor features produced by the base actor MLP
            actor_features = batch.get((group, "actor_features"), None)
            if actor_features is not None:
                # Compute scalar statistics averaged over all dims (batch, agents, features)
                losses["diag_actor_features_mean"] = actor_features.mean().detach()
                losses["diag_actor_features_std"] = actor_features.std().detach()
                # Norm gives overall scale
                losses["diag_actor_features_norm"] = actor_features.norm(dim=-1).mean().detach()
            
            # Logits before BetaParamExtractor
            logits = batch.get((group, "logits"), None)
            if logits is not None and logits.numel() > 0:
                # Aggregate statistics
                losses["diag_logits_mean"] = logits.mean().detach()
                losses["diag_logits_std"] = logits.std().detach()
                losses["diag_logits_min"] = logits.min().detach()
                losses["diag_logits_max"] = logits.max().detach()
                
                # Split into alpha/beta halves to detect symmetry
                action_dim = logits.shape[-1] // 2
                alpha_half = logits[..., :action_dim]
                beta_half = logits[..., action_dim:]
                alpha_mean = alpha_half.mean().detach()
                beta_mean = beta_half.mean().detach()
                losses["diag_alpha_beta_mean_diff"] = (alpha_mean - beta_mean)

            # Compute lin_beta gradient norms per GNN layer if available
            try:
                policy_for_loss = self.get_policy_for_loss(group)
                # ProbabilisticActor may wrap a TensorDictSequential in `module`
                root_module = getattr(policy_for_loss, "module", policy_for_loss)
                # Find the HeteroGNN module in the policy
                hetero_gnns = [m for m in root_module.modules() if isinstance(m, HeteroGNN)]
                if hetero_gnns:
                    gnn = hetero_gnns[0]
                    # For each layer, log per-edge-type and average lin_beta gradient norms
                    for i, hetero_conv in enumerate(gnn.convs):
                        beta_grad_norms = []
                        # hetero_conv.convs is a ModuleDict mapping (src, rel, dst) -> TransformerConv
                        for edge_key, conv in hetero_conv.convs.items():
                            if hasattr(conv, "lin_beta") and conv.lin_beta is not None and hasattr(conv.lin_beta, "weight"):
                                if conv.lin_beta.weight.grad is not None:
                                    grad_norm = conv.lin_beta.weight.grad.norm().detach()
                                    beta_grad_norms.append(grad_norm)
                                    # edge_key is a tuple (src, rel, dst); use relation name
                                    try:
                                        rel = edge_key[1]
                                    except Exception:
                                        rel = str(edge_key)
                                    # Sanitize rel for logging
                                    rel = str(rel).replace(" ", "_")
                                    losses[f"diag_gnn_lin_beta_grad_norm_layer_{i}_{rel}"] = grad_norm
                        if beta_grad_norms:
                            # Log mean gradient norm per layer
                            mean_grad_norm = torch.stack(beta_grad_norms).mean()
                            losses[f"diag_gnn_lin_beta_grad_norm_layer_{i}"] = mean_grad_norm
            except Exception:
                # Avoid crashing training due to diagnostics
                pass
        
        return losses

    #####################
    # Custom new methods
    #####################


    def _get_or_build_shared_actor_gnn(self, gnn_output_dim: int):
        """Lazily build and return a shared actor GNN that sees ALL agent groups.

        The GNN has one node type per agent group plus ``grid_node``, with
        cross-group ``interaction`` edges between every pair of agent types,
        ``mapping`` / ``mapping_rev`` edges between each agent type and the
        grid, and grid-internal adjacency edges.  This allows each agent's
        embedding to be informed by the participation scores and graph
        positions of agents in *every* group, enabling cross-group role
        differentiation.

        The same ``nn.Module`` instance is reused across all groups' actor
        pipelines, so gradients from every group's loss flow into it.

        Note: because the shared parameters appear in every group's
        optimizer, _get_parameters() returns them in a separate param group
        with lr scaled by 1/num_groups to compensate for the multiple steps.
        """
        if self._shared_actor_gnn is not None:
            return self._shared_actor_gnn

        all_groups = list(self.group_map.keys())

        # --- Node features for every agent type + grid ---------
        node_features_keys = {"grid_node": "grid_node_features"}
        node_features_dims = {"grid_node": 2}
        if self.gnn_agent_node_feature_key is not None:
            for g in all_groups:
                per_type_key = f"{g}_{self.gnn_agent_node_feature_key}"
                node_features_keys[g] = per_type_key
                node_features_dims[g] = self.gnn_agent_node_feature_dim

        # --- GNN config ----------------------------------------
        # --- Edge feature dims: per-pair interaction names ----
        # The GNN auto-generates per-pair relation names like
        # "EV_self_interact", "EV_interact_PV", etc.  Register
        # them all with dim 0 (no edge features).  The HeteroGNN
        # also falls back to "interaction" for any *_interact_*
        # relation not found in this dict.
        edge_features_dims = {
            "line_adjacency": 3,
            "transformer_adjacency": 3,
            "switch_adjacency": 1,
            "interaction": 0,   # fallback for any interact edge
            "mapping": 0,
            "mapping_rev": 0,
        }
        for g1 in all_groups:
            edge_features_dims[f"{g1}_self_interact"] = 0
            for g2 in all_groups:
                if g1 != g2:
                    edge_features_dims[f"{g1}_interact_{g2}"] = 0

        # --- Validate key contract --------------------------------
        # The GNN resolves keys at runtime via leaf-name matching
        # (_get_key_terminating_with).  Collect all leaf names visible
        # in observation_spec so we can warn about anything missing.
        _obs_leaf_names = set()
        for k in self.observation_spec.keys(True, True):
            _obs_leaf_names.add(k[-1] if isinstance(k, tuple) else k)

        _missing = []
        # Grid node features (hard-coded as "grid_node_features")
        if "grid_node_features" not in _obs_leaf_names:
            _missing.append("grid_node_features")
        # Grid adjacency matrices
        for adj in ("line_adjacency", "transformer_adjacency", "switch_adjacency"):
            if adj not in _obs_leaf_names:
                _missing.append(adj)
        # Per-group agent↔grid edge indices
        for g in all_groups:
            eidx = f"{g}_agent_grid_edge_index"
            if eidx not in _obs_leaf_names:
                _missing.append(eidx)
        # Per-group agent node features (config-dependent)
        if self.gnn_agent_node_feature_key is not None:
            for g in all_groups:
                feat = f"{g}_{self.gnn_agent_node_feature_key}"
                if feat not in _obs_leaf_names:
                    _missing.append(feat)

        if _missing:
            warnings.warn(
                f"Actor GNN key contract: the following keys are expected "
                f"by the GNN but not found in observation_spec: {_missing}. "
                f"The GNN may silently produce degraded results. Verify "
                f"that the environment provides all required graph keys.",
                stacklevel=2,
            )

        gnn_conf = HeteroGnnConfig(
            topology="adjacency",
            self_loops=self.gnn_self_loops,
            gnn_class=tgnn.TransformerConv,
            gnn_kwargs={
                "heads": self.gnn_heads,
                "concat": self.gnn_concat_heads,
                "beta": self.gnn_use_beta,
            },
            grid_edge_keys={
                "line_adjacency": "line_adjacency",
                "transformer_adjacency": "transformer_adjacency",
                "switch_adjacency": "switch_adjacency",
            },
            edge_features_dims=edge_features_dims,
            node_features_keys=node_features_keys,
            node_features_dims=node_features_dims,
            # Per-type edge indices are auto-discovered by HeteroGNN
            # as {group}_agent_grid_edge_index for each group.
            agent_node_index_key="agent_grid_edge_index",
            agent_groups=all_groups,  # ← all groups as node types
            exclude_observations_from_node_features=True,
            cat_observations_to_output=False,
            num_layers=self.gnn_num_layers,
            norm_class=self.gnn_norm_class,
            prune_non_agent_final_layer=True,
            pos_features=0,
            vel_features=0,
            edge_radius=0,
        )

        # --- Input spec: all groups' observations --------------
        shared_input_spec = Composite()
        for g in all_groups:
            shared_input_spec.set(
                g, self.observation_spec[g].clone().to(self.device)
            )

        # --- Output spec: each group gets its own embedding ----
        shared_output_spec = Composite()
        for g in all_groups:
            n_g = len(self.group_map[g])
            shared_output_spec.set(
                g,
                Composite(
                    {"gnn_embedding": Unbounded(shape=(n_g, gnn_output_dim))},
                    shape=(n_g,),
                ),
            )

        primary_group = all_groups[0]
        n_primary = len(self.group_map[primary_group])

        self._shared_actor_gnn = gnn_conf.get_model(
            input_spec=shared_input_spec,
            output_spec=shared_output_spec,
            agent_group=primary_group,
            input_has_agent_dim=True,
            n_agents=n_primary,
            centralised=False,
            share_params=True,
            device=self.device,
            action_spec=self.action_spec,
        )
        return self._shared_actor_gnn


    def _add_graph_keys_to_spec(self, spec):
        """Add GNN-required graph keys from observation_spec into *spec*.

        Always adds keys for every agent group in ``self.group_map``.
        """
        cfg = self.critic_model_config
        groups_to_add = list(self.group_map.keys())

        # Grid adjacency matrices (shared across types)
        if cfg.grid_edge_keys:
            for spec_key in cfg.grid_edge_keys.values():
                if spec_key in self.observation_spec.keys():
                    spec.set(
                        spec_key,
                        self.observation_spec[spec_key].clone().to(self.device),
                    )

        # Node features
        if cfg.node_features_keys:
            for node_type, spec_key in cfg.node_features_keys.items():
                if node_type == "grid_node":
                    if spec_key in self.observation_spec.keys():
                        spec.set(
                            spec_key,
                            self.observation_spec[spec_key].clone().to(self.device),
                        )
                else:
                    for g in groups_to_add:
                        per_type_key = f"{g}_{spec_key}"
                        if per_type_key in self.observation_spec.keys():
                            spec.set(
                                per_type_key,
                                self.observation_spec[per_type_key].clone().to(self.device),
                            )

        # Agent-grid edge index
        if cfg.agent_node_index_key:
            for g in groups_to_add:
                key = f"{g}_agent_grid_edge_index"
                if key in self.observation_spec.keys():
                    spec.set(
                        key,
                        self.observation_spec[key].clone().to(self.device),
                    )


class HGTeam(HGTeamBase):
    """Heterogeneous Graph Team PPO - Multi Agent PPO with GNN-based agent embeddings.

    Args:
        share_param_critic (bool): Whether to share the parameters of the critics within agent groups
        gnn_mode (str): How to use GNN embeddings. Options:
            - "none": No GNN, standard MLP actor
            - "concat": GNN embeddings concatenated with observations as MLP input
            - "hypernetwork": GNN embeddings generate weights for actor network
            - "learned_query": GNN produces embedding_z which the actor reads
              from the tensordict (e.g. Transformer cross-attention via use_z_as_query)
        clip_epsilon (scalar): weight clipping threshold in the clipped PPO loss equation.
        entropy_coef (scalar): entropy multiplier when computing the total loss.
        critic_coef (scalar): critic loss multiplier when computing the total
        loss_critic_type (str): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1".
        lmbda (float): The GAE lambda
        scale_mapping (str): positive mapping function to be used with the std.
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        use_tanh_normal (bool): if ``True``, use TanhNormal as the continuyous action distribution with support bound
            to the action domain. Otherwise, an IndependentNormal is used.
        use_beta (bool): if ``True``, use Beta distribution for bounded [0,1] actions. Takes precedence over use_tanh_normal.
        beta_min_param (float): minimum parameter value for Beta distribution to ensure numerical stability.
        minibatch_advantage (bool): if ``True``, advantage computation is perfomend on minibatches of size
            ``experiment.config.on_policy_minibatch_size`` instead of the full
            ``experiment.config.on_policy_collected_frames_per_batch``, this helps not exploding memory usage
        share_critic_across_groups (bool): if ``True``, the critic module will be shared across all agent groups.
            This is only possible if a global state is present.
        centralised_value_per_agent (bool): if ``True``, the centralised critic will output a value for each agent
            instead of a single value for the group. This is useful when using models like HeteroGNN that can
            produce individual values for each agent.

    """

    def __init__(
        self,
        clip_epsilon: float,
        entropy_coef: float,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        minibatch_advantage: bool,
        critic_use_other_actions: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda
        self.minibatch_advantage = minibatch_advantage
        self.critic_use_other_actions = critic_use_other_actions


    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        # Loss
        loss_module = HGTeamLoss(
            algorithm=self,
            group=group,
            actor_network=policy_for_loss,
            critic_network=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coef,
            critic_coeff=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=True,
            normalize_advantage_exclude_dims=(1,),
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, 
            gamma=self.experiment_config.gamma, 
            lmbda=self.lmbda,
            vectorized=False,
            deactivate_vmap=True,  # Clearly disable vmap for sparse graphs (dynamic shapes)
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        # --- Actor parameters ---
        # The actor_network_params TensorDict contains all params from the
        # actor pipeline, including the shared GNN.  In a multi-group setup
        # the shared GNN appears in every group's optimizer, so it would
        # receive one Adam step per group (effectively N× learning rate).
        #
        # Fix: split params into (a) per-group MLP/head params at full LR
        # and (b) shared GNN params at LR/num_groups.  Returning param groups
        # (list of dicts) makes Adam use the per-group LR override.
        #
        # We also filter out shared-GNN final-layer params whose gradients
        # are structurally zero for this group (other-group output convs).
        actor_params = list(loss.actor_network_params.flatten_keys().values())

        if self.gnn_mode != "none" and self._shared_actor_gnn is not None:
            n_before = len(actor_params)
            actor_params = self._filter_shared_gnn_params(
                actor_params, self._shared_actor_gnn, group, role="actor"
            )
            n_after = len(actor_params)
            if n_before != n_after:
                print(f"[HGTeam] {group}/actor: filtered {n_before - n_after} "
                      f"other-group GNN params ({n_before} → {n_after})")

            # Separate shared GNN params from per-group params for LR scaling
            actor_params = self._split_shared_gnn_param_groups(
                actor_params, self._shared_actor_gnn
            )

        # --- Critic parameters ---
        critic_params = list(loss.critic_network_params.flatten_keys().values())

        if self._shared_gnn_critic is not None:
            n_before = len(critic_params)
            critic_params = self._filter_shared_gnn_params(
                critic_params, self._shared_gnn_critic, group, role="critic"
            )
            n_after = len(critic_params)
            if n_before != n_after:
                print(f"[HGTeam] {group}/critic: filtered {n_before - n_after} "
                      f"other-group GNN params ({n_before} → {n_after})")

            # Separate shared GNN params from per-group params for LR scaling
            critic_params = self._split_shared_gnn_param_groups(
                critic_params, self._shared_gnn_critic
            )

        return {
            "loss_objective": actor_params,
            "loss_critic": critic_params,
        }

    def _split_shared_gnn_param_groups(self, all_params, shared_gnn):
        """Split params into per-group (full LR) and shared GNN (scaled LR).

        Because the shared GNN appears in every group's optimizer, it receives
        one Adam step per group per training iteration — effectively N× the
        intended learning rate.  We return param groups so that Adam applies
        ``lr / num_groups`` to the shared GNN params.

        Note: The 1/N LR scaling corrects the effective learning rate, but
        Adam's second-moment estimate (v) still sees N independent steps of
        |g/N|² each, yielding v ≈ g²/N instead of the single-step v = g².
        The smaller sqrt(v) makes effective step sizes ~√N larger than a
        single aggregated update.  With N=3 groups, this is a ~√3 ≈ 1.73×
        over-correction — acceptable in practice but not mathematically exact.

        TODO: For exact equivalence, replace this with a gradient-accumulating
        Adam wrapper that buffers N calls to step(), sums their gradients,
        and performs a single real Adam update on the Nth call.  This would
        be transparent to BenchMARL's per-group optimizer loop and eliminate
        the √N bias in Adam's variance estimate.

        Returns a list of param-group dicts (accepted by torch.optim.Adam).
        """
        num_groups = len(self.group_map)
        if num_groups <= 1:
            return all_params  # No scaling needed with a single group

        shared_ptrs = {p.data_ptr() for p in shared_gnn.parameters()}
        gnn_params = [p for p in all_params if p.data_ptr() in shared_ptrs]
        other_params = [p for p in all_params if p.data_ptr() not in shared_ptrs]

        if not gnn_params:
            return all_params  # No shared GNN params found

        scaled_lr = self.experiment_config.lr / num_groups
        return [
            {"params": other_params},
            {"params": gnn_params, "lr": scaled_lr},
        ]

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        # HGTeam uses the same stochastic actor for collection
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        batch = batch.to(self.device)
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        # Pre-compute augmented critic observations with other agents' actions
        if self.critic_use_other_actions:
            self._augment_critic_observations(batch)

        loss = self.get_loss_and_updater(group)[0]
        if self.minibatch_advantage:
            increment = -(
                -self.experiment.config.train_minibatch_size(self.on_policy)
                // batch.shape[1]
            )
        else:
            increment = batch.batch_size[0] + 1
        last_start_index = 0
        start_index = increment
        minibatches = []
        while last_start_index < batch.shape[0]:
            minibatch = batch[last_start_index:start_index]
            minibatches.append(minibatch)
            with torch.no_grad():
                loss.value_estimator(
                    minibatch,
                    params=loss.critic_network_params,
                    target_params=loss.target_critic_network_params,
                )
            last_start_index = start_index
            start_index += increment

        batch = torch.cat(minibatches, dim=0)
        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase, batch: TensorDictBase = None
    ) -> TensorDictBase:
        loss_vals.set(
            "loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"]
        )
        del loss_vals["loss_entropy"]
        
        # Note: embedding losses are already added to loss_objective in HGTeamLoss.forward()
        # We don't add them again here to avoid double-counting
        
        return loss_vals

    def _compute_other_actions_dim(self) -> int:
        """Total dimension of all agents' actions across all groups."""
        return sum(
            len(self.group_map[g]) * self.action_spec[g, "action"].shape[-1]
            for g in self.group_map
        )

    def _augment_critic_observations(self, batch: TensorDictBase):
        """Pre-compute augmented critic observations for all groups.

        For each agent i in each group, creates:
            critic_obs_i = [obs_i, a_{-i}]
        where a_{-i} contains every other agent's action (across all groups)
        with agent i's own action zeroed out (counterfactual baseline).

        Also computes augmented observations for the ``next`` sub-tensordict
        (needed for GAE bootstrapping), using actions shifted forward by one
        timestep.  The last timestep gets zero-padded actions (masked by the
        done flag in GAE).
        """
        type_order = list(self.group_map.keys())

        # Idempotency guard: skip if already augmented (process_batch is
        # called once per group, but we augment all groups at once).
        if batch.get((type_order[0], "critic_obs"), None) is not None:
            return

        # Gather all current actions
        all_actions = {g: batch.get((g, "action")) for g in type_order}

        # Augment current-state observations for every group
        for group in type_order:
            obs = batch.get((group, "observation"))
            critic_obs = self._compute_augmented_obs(
                obs, group, all_actions, type_order
            )
            batch.set((group, "critic_obs"), critic_obs)

        # Handle "next" state for GAE bootstrapping
        next_td = batch.get("next", None)
        if next_td is not None:
            # Approximate next-state other-actions by shifting current
            # actions forward by one timestep along dim 0 (time).
            next_actions = {}
            for g in type_order:
                act = all_actions[g]
                shifted = torch.zeros_like(act)
                if act.shape[0] > 1:
                    shifted[:-1] = act[1:]
                next_actions[g] = shifted

            for group in type_order:
                next_obs = next_td.get((group, "observation"), None)
                if next_obs is not None:
                    next_critic_obs = self._compute_augmented_obs(
                        next_obs, group, next_actions, type_order
                    )
                    next_td.set((group, "critic_obs"), next_critic_obs)

    def _compute_augmented_obs(
        self,
        obs: torch.Tensor,
        target_group: str,
        all_actions: dict,
        type_order: list,
    ) -> torch.Tensor:
        """Return ``[obs_i, a_{-i}]`` for every agent *i* in *target_group*.

        * Same-group actions are included with agent *i*'s own action zeroed.
        * Other-group actions are fully visible to every agent.

        Returns a tensor of shape ``(..., n_agents, obs_dim + total_actions_dim)``.
        """
        n_self = len(self.group_map[target_group])
        parts = [obs]

        for g in type_order:
            acts = all_actions[g]
            n_g = len(self.group_map[g])
            d_g = acts.shape[-1]

            if g == target_group:
                # Same group: for each viewer agent i, zero out its own action
                # expanded: (..., n_self, n_g, d_g)
                expanded = acts.unsqueeze(-3).expand(
                    *acts.shape[:-2], n_self, n_g, d_g
                )
                mask = (1.0 - torch.eye(n_g, device=acts.device)).unsqueeze(-1)
                masked = expanded * mask
                parts.append(masked.flatten(-2, -1))
            else:
                # Other group: all actions visible to every agent
                flat = acts.flatten(-2, -1)  # (..., n_g * d_g)
                flat = flat.unsqueeze(-2).expand(
                    *obs.shape[:-1], flat.shape[-1]
                )
                parts.append(flat)

        return torch.cat(parts, dim=-1)

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])

        # ================================================================
        # Shared critic path — always used for HeteroGNN critics so that
        # the critic graph contains cross-group interaction edges.
        # ================================================================
        if self.share_critic_across_groups or isinstance(self.critic_model_config, HeteroGnnConfig):
            if not self.share_param_critic:
                raise ValueError(
                    "Using a HeteroGNN critic or sharing critic across groups "
                    "requires share_param_critic=True"
                )
            return self._get_shared_critic(group, n_agents)

        # ================================================================
        # Per-type independent critic path (MLP critics only)
        # ================================================================
        if self.share_param_critic and not self.centralised_value_per_agent:
            critic_output_spec = Composite({"state_value": Unbounded(shape=(1,))})
        else:
            critic_output_spec = Composite(
                {
                    group: Composite(
                        {"state_value": Unbounded(shape=(n_agents, 1))},
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:
            input_has_agent_dim = False
            critic_input_spec = self.state_spec
        else:
            input_has_agent_dim = True
            if self.critic_use_other_actions:
                obs_dim = self.observation_spec[group, "observation"].shape[-1]
                other_actions_dim = self._compute_other_actions_dim()
                augmented_dim = obs_dim + other_actions_dim
                critic_input_spec = Composite(
                    {group: Composite(
                        {"critic_obs": Unbounded(shape=(n_agents, augmented_dim))},
                        shape=(n_agents,),
                    )}
                )
            else:
                critic_input_spec = Composite(
                    {group: self.observation_spec[group].clone().to(self.device)}
                )
            # Include root-level graph keys the GNN needs
            if isinstance(self.critic_model_config, HeteroGnnConfig):
                self._add_graph_keys_to_spec(critic_input_spec)

        # Build model
        critic_config = self.critic_model_config
        if (self.centralised_value_per_agent
                and isinstance(critic_config, HeteroGnnConfig)):
            critic_config.prune_non_agent_final_layer = True

        # Per-type key overrides
        if isinstance(critic_config, HeteroGnnConfig):
            import copy
            critic_config = copy.deepcopy(critic_config)
            if critic_config.agent_node_index_key is not None:
                critic_config.agent_node_index_key = f"{group}_agent_grid_edge_index"
            if critic_config.node_features_keys and "agents" in critic_config.node_features_keys:
                orig_key = critic_config.node_features_keys["agents"]
                critic_config.node_features_keys["agents"] = f"{group}_{orig_key}"

        value_module = critic_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=not self.centralised_value_per_agent,
            input_has_agent_dim=input_has_agent_dim,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )

        if self.share_param_critic and not self.centralised_value_per_agent:
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(
                    *value.shape[:-1], n_agents, 1
                ),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(value_module, expand_module)

        return value_module

    # ------------------------------------------------------------------
    # Shared-critic helpers
    # ------------------------------------------------------------------

    def _get_shared_critic(self, group: str, n_agents: int) -> TensorDictModule:
        """Build (or reuse) a shared critic where each agent type is a native
        heterogeneous node type in the GNN.

        Architecture::

            Per-type obs ──► Native HeteroGNN (each type = own node type)
            Graph keys   ──►   with per-type input projections
                                          │
                                  Per-type state_values
                                          │
                               Direct output: (group, state_value)
        """
        import copy
        type_order = list(self.group_map.keys())
        total_agents = sum(len(a) for a in self.group_map.values())

        # --- Build shared GNN critic on first call -----------------------
        if self._shared_gnn_critic is None:
            # Build input spec with each agent type as its own group
            critic_input_spec = Composite()
            for t in type_order:
                n_t = len(self.group_map[t])
                if self.critic_use_other_actions:
                    obs_dim = self.observation_spec[t, "observation"].shape[-1]
                    other_actions_dim = self._compute_other_actions_dim()
                    augmented_dim = obs_dim + other_actions_dim
                    critic_input_spec.set(
                        t,
                        Composite(
                            {"critic_obs": Unbounded(shape=(n_t, augmented_dim))},
                            shape=(n_t,),
                        ),
                    )
                else:
                    critic_input_spec.set(
                        t,
                        self.observation_spec[t].clone().to(self.device),
                    )

            # Add graph keys for ALL types
            if isinstance(self.critic_model_config, HeteroGnnConfig):
                self._add_graph_keys_to_spec(critic_input_spec)

            # Build per-type node_features_keys and dims from the base config
            critic_config = copy.deepcopy(self.critic_model_config)
            
            if isinstance(critic_config, HeteroGnnConfig):
                # Set agent_groups so the GNN knows all types
                critic_config.agent_groups = list(type_order)
                
                # Build per-type node feature keys (e.g., EV -> EV_participation_score)
                base_node_features_keys = dict(critic_config.node_features_keys or {})
                base_node_features_dims = dict(critic_config.node_features_dims or {})
                
                # Remove the generic "agents" placeholder if present
                agents_feature_key = base_node_features_keys.pop("agents", None)
                agents_feature_dim = base_node_features_dims.pop("agents", None)
                
                # Add per-type entries
                for t in type_order:
                    if agents_feature_key is not None:
                        base_node_features_keys[t] = f"{t}_{agents_feature_key}"
                    if agents_feature_dim is not None:
                        base_node_features_dims[t] = agents_feature_dim
                
                critic_config.node_features_keys = base_node_features_keys
                critic_config.node_features_dims = base_node_features_dims
                
                # Use a per-type edge key convention; the GNN's _forward will
                # auto-discover {group}_agent_grid_edge_index for each group
                critic_config.agent_node_index_key = "agent_grid_edge_index"
                
                if self.centralised_value_per_agent:
                    critic_config.prune_non_agent_final_layer = True

            # Output spec: per-type state_value
            if self.centralised_value_per_agent:
                # Every group gets its own (group, state_value) output
                critic_output_spec = Composite()
                for t in type_order:
                    n_t = len(self.group_map[t])
                    critic_output_spec.set(
                        t,
                        Composite(
                            {"state_value": Unbounded(shape=(n_t, 1))},
                            shape=(n_t,),
                        ),
                    )
            else:
                critic_output_spec = Composite(
                    {"state_value": Unbounded(shape=(1,))}
                )

            # The primary agent_group for the model (used for self.out_key, etc.)
            primary_group = type_order[0]
            n_primary = len(self.group_map[primary_group])

            self._shared_gnn_critic = critic_config.get_model(
                input_spec=critic_input_spec,
                output_spec=critic_output_spec,
                n_agents=n_primary,  # Primary group count; multi-group uses per-group counts internally
                centralised=not self.centralised_value_per_agent,
                input_has_agent_dim=True,
                agent_group=primary_group,
                share_params=True,
                device=self.device,
                action_spec=self.action_spec,
            )

        # --- Per-group wrapper -------------------------------------------
        if self.centralised_value_per_agent:
            # The GNN writes (group, "state_value") for each group in agent_groups.
            # For non-primary groups, the output is at (group, "state_value") already.
            # We just need a pass-through that reads the correct key.
            if group == type_order[0]:
                # Primary group: GNN output is already at the correct key
                return self._shared_gnn_critic
            else:
                # Non-primary: GNN writes output at (group, "state_value") via
                # the multi-group output mapping in _forward. Just return the GNN.
                return self._shared_gnn_critic
        else:
            # Fully centralised: single scalar value, expand per group
            expand_fn = lambda value: value.unsqueeze(-2).expand(
                *value.shape[:-1], n_agents, 1
            )
            expand_module = TensorDictModule(
                expand_fn,
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            return TensorDictSequential(
                self._shared_gnn_critic,
                expand_module,
            )


@dataclass
class HGTeamConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.HGTeam`."""

    # PPO base parameters
    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    scale_lb: float = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING
    use_beta: bool = MISSING
    beta_min_param: float = MISSING

    # HGTeam-specific parameters
    share_critic_across_groups: bool = MISSING
    centralised_value_per_agent: bool = MISSING
    gnn_mode: str = MISSING  # "none", "concat", "hypernetwork", or "learned_query"
    z_dim: int = MISSING
    hypernet_actor_feature_dim: int = MISSING
    stochastic_z: bool = MISSING
    embedding_entropy_coef: float = MISSING
    embedding_diversity_coef: float = MISSING
    
    # GNN configuration parameters
    gnn_num_layers: int = MISSING
    gnn_heads: int = MISSING
    gnn_concat_heads: bool = MISSING
    gnn_use_beta: bool = MISSING
    gnn_self_loops: bool = MISSING
    gnn_agent_node_feature_key: Optional[str] = MISSING
    gnn_agent_node_feature_dim: int = MISSING
    critic_use_other_actions: bool = MISSING  # Condition value fn on other agents' actions (counterfactual)
    gnn_norm_class: Optional[str] = MISSING  # "LayerNorm", "BatchNorm1d", etc. or null/None
    critic_embed_dim: int = MISSING  # Common dimension for per-type observation projections in shared critic

    # Split-z parameters (learned_query mode): separate GNN output into
    # z_token (deterministic, KV prepend) and z_query (cross-attention query).
    split_z: bool = MISSING
    z_token_dim: int = MISSING
    z_query_dim: int = MISSING
    stochastic_z_query: bool = MISSING

    # VIB (Variational Information Bottleneck) parameters
    use_vib: bool = MISSING
    vib_beta: float = MISSING
    vib_warmup_frames: int = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return HGTeam

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
