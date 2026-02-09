#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

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
from benchmarl.models.common import ModelConfig
from benchmarl.beta_param_extractor import BetaParamExtractor
from benchmarl.independent_beta import IndependentBeta
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
        # Get standard PPO losses
        out = super().forward(tensordict)
        
        # If embeddings are missing (because PPO used saved params from rollout),
        # we must run the actor module to generate them and attach gradients.
        
        # Check if embedding is present in the input batch
        has_embedding = tensordict.get((self.group, "embedding_z"), None) is not None
        
        if not has_embedding:
            # We need to run the underlying policy module to generate embeddings
            # We use get_dist_params which runs the underlying module chain
            self.actor_network.get_dist_params(tensordict)
        
        # Compute embedding losses using the input batch (tensordict)
        embedding_losses = self.algorithm._compute_embedding_losses(self.group, tensordict)
        
        # Log if we are missing losses unexpectedly
        if not embedding_losses:
             # Check if we should have them
             if self.algorithm.gnn_mode != "none": 
                 pass # Could print debug here if needed

        for k, v in embedding_losses.items():
            out.set(k, v)
            # Add auxiliary losses to the main objective so they are optimized
            if k.startswith("loss_") and "loss_objective" in out.keys():
                out.set("loss_objective", out.get("loss_objective") + v)
                
        return out

class HyperNetworkJoiner(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feature_dim: int,
        output_dim: int,
        device: torch.device,
        stochastic_embedding: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.stochastic_embedding = stochastic_embedding

        # If stochastic, embedding_dim is actually 2x (mean + logvar)
        # so the actual latent dim is embedding_dim // 2
        if stochastic_embedding:
            self.latent_dim = embedding_dim // 2
        else:
            self.latent_dim = embedding_dim

        # Generators (operate on latent_dim, not raw embedding_dim)
        self.weight_generator = nn.Linear(
            self.latent_dim, feature_dim * output_dim, device=device
        )
        self.bias_generator = nn.Linear(self.latent_dim, output_dim, device=device)

    def forward(self, features: torch.Tensor, embedding: torch.Tensor):
        # features: (..., n_agents, feature_dim)
        # embedding: (..., n_agents, embedding_dim)
        # Returns: (logits, embedding_mean, embedding_logvar) if stochastic, else (logits, embedding, None)

        if self.stochastic_embedding:
            # Split embedding into mean and log_var
            mean, log_var = embedding.chunk(2, dim=-1)
            # Clamp log_var to prevent numerical instability
            log_var = torch.clamp(log_var, min=-10, max=2)
            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + eps * std  # Sampled latent
        else:
            mean = embedding
            log_var = None
            z = embedding

        weights = self.weight_generator(z)
        weights = weights.view(*weights.shape[:-1], self.feature_dim, self.output_dim)
        
        bias = self.bias_generator(z)

        # logits = features * weights + bias
        logits = torch.einsum("...f,...fo->...o", features, weights)
        return logits + bias, z, log_var


class EmbeddingProcessor(nn.Module):
    """Process GNN embeddings for concat mode - handles stochastic sampling and outputs embedding stats."""
    def __init__(self, embedding_dim: int, stochastic: bool = False):
        super().__init__()
        self.stochastic = stochastic
        if stochastic:
            self.latent_dim = embedding_dim // 2
        else:
            self.latent_dim = embedding_dim
    
    def forward(self, embedding: torch.Tensor):
        # embedding: (..., n_agents, embedding_dim)
        # Returns: (z, log_var) where z is the processed embedding
        if self.stochastic:
            mean, log_var = embedding.chunk(2, dim=-1)
            log_var = torch.clamp(log_var, min=-10, max=2)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = embedding
            log_var = None
        return z, log_var


class HGTeam(Algorithm):
    """Heterogeneous Graph Team PPO - Multi Agent PPO with GNN-based agent embeddings.

    Args:
        share_param_critic (bool): Whether to share the parameters of the critics within agent groups
        gnn_mode (str): How to use GNN embeddings. Options:
            - "none": No GNN, standard MLP actor
            - "concat": GNN embeddings concatenated with observations as MLP input
            - "hypernetwork": GNN embeddings generate weights for actor network
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
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: bool,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        use_beta: bool,
        beta_min_param: float,
        minibatch_advantage: bool,
        share_critic_across_groups: bool = False,
        centralised_value_per_agent: bool = False,
        gnn_mode: str = "none",  # "none", "concat", or "hypernetwork"
        hypernet_hidden_dim: int = None,
        hypernet_feature_dim: int = None,
        stochastic_hypernet: bool = False,
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
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_critic = share_param_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal
        self.use_beta = use_beta
        self.beta_min_param = beta_min_param
        self.minibatch_advantage = minibatch_advantage
        self.share_critic_across_groups = share_critic_across_groups
        self.centralised_value_per_agent = centralised_value_per_agent
        self.gnn_mode = gnn_mode
        if gnn_mode not in ("none", "concat", "hypernetwork"):
            raise ValueError(f"gnn_mode must be 'none', 'concat', or 'hypernetwork', got '{gnn_mode}'")
        self.hypernet_hidden_dim = hypernet_hidden_dim
        self.hypernet_feature_dim = hypernet_feature_dim
        self.stochastic_hypernet = stochastic_hypernet
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

        self.shared_critic_module = None


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
            normalize_advantage=False,
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
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

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
            feature_dim = self.hypernet_feature_dim
            actor_output_spec = Composite(
                {
                    group: Composite(
                        {"actor_features": Unbounded(shape=(n_agents, feature_dim))},
                        shape=(n_agents,),
                    )
                }
            )
        else:
            # None or Concat mode: MLP outputs logits directly
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
            # GNN Stream: generates embeddings from graph structure
            # 
            # The GNN uses:
            # - grid_node features: current load forecast (2D)
            # - agent node features: configurable via gnn_agent_node_feature_key (e.g., participation_score)
            # 
            # Agent node features can be set to static features like participation_score
            # that remain constant from connection until disconnection, allowing the GNN 
            # to learn embeddings based on "task difficulty" rather than dynamic state.
            
            # Build node_features_keys based on config
            node_features_keys = {"grid_node": "grid_node_features"}
            node_features_dims = {"grid_node": 2}
            if self.gnn_agent_node_feature_key is not None:
                node_features_keys["agents"] = self.gnn_agent_node_feature_key
                node_features_dims["agents"] = self.gnn_agent_node_feature_dim
            
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
                    "switch_adjacency": "switch_adjacency"
                },
                edge_features_dims={
                    "line_adjacency": 3,
                    "transformer_adjacency": 3,
                    "switch_adjacency": 1,
                    "interaction": 0,
                    "mapping": 0,
                    "mapping_rev": 0
                },
                node_features_keys=node_features_keys,
                node_features_dims=node_features_dims,
                agent_node_index_key="agent_grid_edge_index",
                exclude_observations_from_node_features=True,  # GNN uses graph features only
                cat_observations_to_output=False,  # Observations handled separately in HGTeam
                num_layers=self.gnn_num_layers,
                norm_class=self.gnn_norm_class,
                prune_non_agent_final_layer=True,
                pos_features=0, vel_features=0, edge_radius=0
            )
            
            # If stochastic, GNN outputs mean + logvar (2x embedding dim)
            gnn_output_dim = self.hypernet_hidden_dim * 2 if self.stochastic_hypernet else self.hypernet_hidden_dim
            
            gnn_stream_output_spec = Composite(
                {
                    group: Composite(
                        {"gnn_embedding": Unbounded(shape=(n_agents, gnn_output_dim))},
                        shape=(n_agents,),
                    )
                }
            )
            
            gnn_stream_module = gnn_conf.get_model(
                input_spec=actor_input_spec,
                output_spec=gnn_stream_output_spec,
                agent_group=group,
                input_has_agent_dim=True,
                n_agents=n_agents,
                centralised=False, # GNN runs per-agent (but uses graph)
                share_params=True, # Typically GNNs share params
                device=self.device,
                action_spec=self.action_spec
            )
            
            # Create embedding processor to handle stochastic sampling and output embedding stats
            embedding_processor = EmbeddingProcessor(
                embedding_dim=gnn_output_dim,
                stochastic=self.stochastic_hypernet,
            )
            embedding_processor_module = TensorDictModule(
                embedding_processor,
                in_keys=[(group, "gnn_embedding")],
                out_keys=[(group, "embedding_z"), (group, "embedding_logvar")]
            )

        if self.gnn_mode == "hypernetwork":
            # Hypernetwork mode: GNN embeddings generate weights for actor
            feature_dim = self.hypernet_feature_dim
            
            joiner = HyperNetworkJoiner(
                embedding_dim=gnn_output_dim,           # From GNN (2x if stochastic)
                feature_dim=feature_dim,                # From Actor
                output_dim=logits_shape[-1],
                device=self.device,
                stochastic_embedding=self.stochastic_hypernet,
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
            latent_dim = self.hypernet_hidden_dim
            
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
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                extractor_module = TensorDictModule(
                    NormalParamExtractor(scale_mapping=self.scale_mapping),
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "loc"), (group, "scale")],
                )
                policy = ProbabilisticActor(
                    module=TensorDictSequential(actor_module, extractor_module),
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "loc"), (group, "scale")],
                    out_keys=[(group, "action")],
                    distribution_class=(
                        IndependentNormal if not self.use_tanh_normal else TanhNormal
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
            minimbatch = batch[last_start_index:start_index]
            minibatches.append(minimbatch)
            with torch.no_grad():
                loss.value_estimator(
                    minimbatch,
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

    def _compute_embedding_losses(
        self, group: str, batch: TensorDictBase
    ) -> dict:
        """Compute embedding-related losses from the batch after forward pass.
        
        Returns dict with:
        - embedding_entropy: Raw entropy value (for logging)
        - embedding_diversity: Raw diversity value (for logging)
        - loss_embedding_entropy: Penalizes high variance (encourages certainty)
        - loss_embedding_diversity: Rewards L2 distance between agent embeddings
        - diag_actor_features_mean/std: Actor feature statistics (hypernetwork mode)
        - diag_logits_mean/std/min/max: Logits statistics before Beta extraction
        - diag_alpha_beta_mean_diff: Mean difference between alpha-half and beta-half logits
        """
        losses = {}
        
        # Get embedding stats from batch
        embedding_z = batch.get((group, "embedding_z"), None)
        
        if embedding_z is None:
            # Embeddings not found - this means hypernetwork is not enabled or forward failed
            return losses
            
        embedding_logvar = batch.get((group, "embedding_logvar"), None)

        # Always log embedding norm for debugging
        losses["embedding_z_norm"] = embedding_z.norm(dim=-1).mean().detach()

        if embedding_logvar is not None:
            # Mean entropy across all dimensions and agents
            entropy = 0.5 * (1 + embedding_logvar)  # Simplified, ignoring constants
            entropy_mean = entropy.mean()
            losses["embedding_entropy"] = entropy_mean.detach()
            
            if self.embedding_entropy_coef > 0:
                # We want to MINIMIZE entropy, so we add positive entropy as loss
                losses["loss_embedding_entropy"] = entropy_mean * self.embedding_entropy_coef
        
        # Diversity: compute pairwise distances between agent embeddings
        # embedding_z shape: (batch_size, n_agents, embedding_dim)
        if embedding_z.dim() >= 3:
            # simple mean pairwise distance
            # For simplicity let's use variance from mean if n_agents is large, 
            # or exact pairwise.
            # Let's use variance of the mean embedding across agents as a proxy for diversity
            # Higher variance across agents = more diversity
            
            # center = embedding_z.mean(dim=-2, keepdim=True)
            # dist = ((embedding_z - center) ** 2).sum(dim=-1).mean()
            
            # Or pairwise:
            # This can be heavy for many agents.
            # Let's stick to variance from centroid for now.
            diversity = torch.var(embedding_z, dim=-2).sum(dim=-1).mean() # Sum var over dims, mean over batch
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

            # Compute lin_beta norms per GNN layer if available
            try:
                policy_for_loss = self.get_policy_for_loss(group)
                # ProbabilisticActor may wrap a TensorDictSequential in `module`
                root_module = getattr(policy_for_loss, "module", policy_for_loss)
                # Find the HeteroGNN module in the policy
                hetero_gnns = [m for m in root_module.modules() if isinstance(m, HeteroGNN)]
                if hetero_gnns:
                    gnn = hetero_gnns[0]
                    # For each layer, log per-edge-type and average lin_beta weight norms
                    for i, hetero_conv in enumerate(gnn.convs):
                        beta_norms = []
                        # hetero_conv.convs is a ModuleDict mapping (src, rel, dst) -> TransformerConv
                        for edge_key, conv in hetero_conv.convs.items():
                            if hasattr(conv, "lin_beta") and conv.lin_beta is not None and hasattr(conv.lin_beta, "weight"):
                                wnorm = conv.lin_beta.weight.norm().detach()
                                beta_norms.append(wnorm)
                                # edge_key is a tuple (src, rel, dst); use relation name
                                try:
                                    rel = edge_key[1]
                                except Exception:
                                    rel = str(edge_key)
                                # Sanitize rel for logging
                                rel = str(rel).replace(" ", "_")
                                losses[f"diag_gnn_lin_beta_norm_layer_{i}_{rel}"] = wnorm
                        if beta_norms:
                            # Log mean norm per layer
                            mean_norm = torch.stack(beta_norms).mean()
                            losses[f"diag_gnn_lin_beta_norm_layer_{i}"] = mean_norm
            except Exception:
                # Avoid crashing training due to diagnostics
                pass
        
        return losses

    #####################
    # Custom new methods
    #####################

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
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
            critic_input_spec = Composite(
                {group: self.observation_spec[group].clone().to(self.device)}
            )

        if self.share_critic_across_groups:
            if self.state_spec is None:
                # Functionality extension: Allow sharing critic without explicit global state
                # This is useful for GNN critics that act on the full graph (all observations)
                input_has_agent_dim = True
                critic_input_spec = self.observation_spec.clone().to(self.device)
                
            if not self.share_param_critic:
                raise ValueError(
                    "Sharing critic across groups requires share_param_critic=True"
                )

        if self.share_critic_across_groups and self.shared_critic_module is not None:
            value_module = self.shared_critic_module
        else:
            # When using per-agent critic values, the GNN only extracts agent
            # node outputs (grid_node outputs are discarded).  Prune the final
            # GNN layer to avoid dead parameters with zero gradients.
            if (self.centralised_value_per_agent
                    and isinstance(self.critic_model_config, HeteroGnnConfig)):
                self.critic_model_config.prune_non_agent_final_layer = True

            value_module = self.critic_model_config.get_model(
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
            if self.share_critic_across_groups:
                self.shared_critic_module = value_module

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
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING
    use_beta: bool = MISSING
    beta_min_param: float = MISSING

    # HGTeam-specific parameters
    share_critic_across_groups: bool = MISSING
    centralised_value_per_agent: bool = MISSING
    gnn_mode: str = MISSING  # "none", "concat", or "hypernetwork"
    hypernet_hidden_dim: int = MISSING
    hypernet_feature_dim: int = MISSING
    stochastic_hypernet: bool = MISSING
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
    gnn_norm_class: Optional[str] = None  # "LayerNorm", "BatchNorm1d", etc. or null/None

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
