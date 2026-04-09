#  HGTeamSAC — SAC-based variant of HGTeam
#
#  Replaces PPO with SAC to provide differentiable gradient paths from Q-value
#  to the actor GNN via mu (direct) and optionally through action (indirect).

import copy
from collections.abc import Iterable
from dataclasses import MISSING, dataclass

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Composite, Unbounded
from torchrl.objectives import LossModule, SACLoss, ValueEstimators

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.algorithms.HGTeam import HGTeamBase
from benchmarl.algorithms.hgteam_modules import (
    EmbeddingProcessor,
    merge_embedding_losses,
)
from benchmarl.models import HeteroGnnConfig

# ======================================================================
# Loss wrapper
# ======================================================================


class HGTeamSACLoss(SACLoss):
    """SACLoss wrapper that adds HGTeam embedding auxiliary losses.

    Follows the same wrapping pattern as HGTeamLoss(ClipPPOLoss):
    pre-forward processing (VIB SNI, z detach, grad tracking), call
    super().forward(), post-forward (restore z, compute embedding losses,
    merge into loss_actor).
    """

    actor_network: TensorDictModule
    qvalue_network: TensorDictModule
    actor_network_params: TensorDictBase
    qvalue_network_params: TensorDictBase
    target_qvalue_network_params: TensorDictBase
    target_actor_network_params: TensorDictBase

    def __init__(self, algorithm: "HGTeamSAC", group: str, *args, **kwargs) -> None:
        """Wrap SACLoss with HGTeam embedding and graph-key propagation.

        Args:
            algorithm: Parent HGTeamSAC instance.
            group: Agent group name this loss operates on.
            *args: Forwarded to ``SACLoss.__init__``.
            **kwargs: Forwarded to ``SACLoss.__init__``.
        """
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm
        self.group = group

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute SAC loss with graph-key propagation and embedding aux losses.

        Pre-forward: propagates static graph keys to ``"next"`` tensordict,
        computes next-actions for peer groups, applies VIB SNI and z-detach.

        Post-forward: restores z, retains grad on embeddings, computes and
        merges embedding auxiliary losses into ``loss_actor``.

        Args:
            tensordict: Batch tensordict with observations, actions, rewards.

        Returns:
            TensorDict with ``loss_actor``, ``loss_qvalue``, ``loss_alpha``,
            and any ``loss_embedding_*`` keys.
        """
        # The actor GNN needs graph structure (adjacency, node features) to run.
        # These are usually static and present in the root tensordict, but not
        # automatically in "next". We must propagate them so the target actor
        # (running on "next") has valid inputs.
        next_td = tensordict.get("next", None)
        if next_td is not None:
            # 1. Global graph keys
            global_keys = [
                "grid_node_features",
                "line_adjacency",
                "transformer_adjacency",
                "switch_adjacency",
            ]
            for k in global_keys:
                if k in tensordict and k not in next_td:
                    next_td.set(k, tensordict.get(k))

            # 2. Per-group keys (features & indices)
            # The GNN expects these at (group, key)
            part_key_suffix = self.algorithm.gnn_agent_node_feature_key
            idx_key_suffix = "agent_grid_edge_index"  # Hardcoded in HGTeam

            for g in self.algorithm.group_map:
                # Participation score (or configured feature)
                if part_key_suffix:
                    # Try (group, key)
                    if (g, part_key_suffix) in tensordict.keys(True, True):
                        if (g, part_key_suffix) not in next_td.keys(True, True):
                            next_td.set((g, part_key_suffix), tensordict.get((g, part_key_suffix)))
                    # Try key at root (fallback, though unlikely for multi-agent)
                    elif part_key_suffix in tensordict:
                        if part_key_suffix not in next_td:
                            next_td.set(part_key_suffix, tensordict.get(part_key_suffix))

                # Agent index
                if (g, idx_key_suffix) in tensordict.keys(True, True):
                    if (g, idx_key_suffix) not in next_td.keys(True, True):
                        next_td.set((g, idx_key_suffix), tensordict.get((g, idx_key_suffix)))

        # --- PRE-FORWARD: Fill missing next-actions for OTHER groups ---
        # The shared Q-network (HeteroGNN) requires inputs from ALL groups.
        # SACLoss only computes next-action for the current group.
        # We must manually compute next-actions for other groups using their
        # *current* policies on a CLONE of next_td to avoid mutating the
        # original batch.
        #
        # Stale-params approximation: groups are trained sequentially within
        # each iteration (Group A → step → Group B → step → …).  When Group
        # B evaluates Group A's policy here, the shared GNN has already been
        # updated by Group A's optimizer step, so the next-actions come from
        # a slightly different policy than the one that collected the data.
        # This is a standard MARL approximation (Citation) and is minor for off-policy
        # SAC (N=3, one gradient step of drift).  For HAPPO, the sequential-
        # update design will handle this differently (frozen encoder).
        if self.algorithm.has_centralized_critic:
            next_td_peer = next_td.clone()
            with torch.no_grad():
                for g in self.algorithm.group_map:
                    if g == self.group:
                        continue

                    has_action = (g, "action") in next_td.keys(True, True)
                    has_mu = (g, "embedding_mu") in next_td.keys(True, True)

                    if not has_action or (self.algorithm.critic_use_mu and not has_mu):
                        other_policy = self.algorithm.get_policy_for_loss(group=g)
                        # Run on clone to avoid polluting the original batch
                        other_policy(next_td_peer)

            # Copy only the keys we need back to next_td (detached)
            for g in self.algorithm.group_map:
                if g == self.group:
                    continue
                if (g, "action") in next_td_peer.keys(True, True):
                    next_td.set(
                        (g, "action"),
                        next_td_peer.get((g, "action")).detach(),
                    )
                if (g, "embedding_mu") in next_td_peer.keys(True, True):
                    next_td.set(
                        (g, "embedding_mu"),
                        next_td_peer.get((g, "embedding_mu")).detach(),
                    )

        # --- PRE-FORWARD: make participation scores grad-enabled ----
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
        # Capture originals for ALL groups so that detach_z and VIB SNI can
        # mutate freely; a single restore pass at the end is order-independent.
        _z_stash: dict[tuple, torch.Tensor] = {}
        _z_keys = ("embedding_z", "embedding_z_token")
        for g in self.algorithm.group_map:
            for zk in _z_keys:
                val = tensordict.get((g, zk), None)
                if val is not None:
                    _z_stash[(g, zk)] = val

        # --- Detach z from Transformer (block indirect GNN gradient) ----
        if self.algorithm.detach_z_from_transformer:
            z_token = tensordict.get((self.group, "embedding_z_token"), None)
            z_query = tensordict.get((self.group, "embedding_z"), None)
            if z_token is not None:
                tensordict.set((self.group, "embedding_z_token"), z_token.detach())
            if z_query is not None:
                tensordict.set((self.group, "embedding_z"), z_query.detach())

        # --- VIB SNI: use deterministic mu for SAC loss computation -----
        if self.algorithm.use_vib:
            embedding_mu = tensordict.get((self.group, "embedding_mu"), None)
            embedding_z_pre = tensordict.get((self.group, "embedding_z"), None)
            if embedding_mu is not None and embedding_z_pre is not None:
                tensordict.set((self.group, "embedding_z"), embedding_mu)

        # --- FORWARD: standard SAC losses ------------------------------
        out = super().forward(tensordict)

        # --- Restore z tensors from snapshot ----------------------------
        for key, val in _z_stash.items():
            tensordict.set(key, val)

        # --- POST-FORWARD: retain_grad on embedding_z -------------------
        embedding_z = tensordict.get((self.group, "embedding_z"), None)
        if embedding_z is not None and embedding_z.requires_grad:
            embedding_z.retain_grad()
            self._grad_embedding_z_ref = embedding_z

        z_token = tensordict.get((self.group, "embedding_z_token"), None)
        if z_token is not None and z_token.requires_grad:
            z_token.retain_grad()
            self._grad_embedding_z_token_ref = z_token

        # --- Compute and merge embedding auxiliary losses into loss_actor ---
        merge_embedding_losses(self.algorithm, self.group, tensordict, out, "loss_actor")

        return out


# ======================================================================
# Algorithm
# ======================================================================


class HGTeamSAC(HGTeamBase):
    """HGTeam with SAC instead of PPO.

    Inherits the full actor pipeline (GNN → EmbeddingProcessor → Transformer
    → Beta distribution) from HGTeamBase.  Replaces the critic with a Q(s,a,μ)
    network and uses SACLoss for differentiable gradient flow from Q to actor.
    """

    def __init__(
        self,
        # SAC-specific parameters
        alpha_init: float = 1.0,
        target_entropy: float | str = "auto",
        num_qvalue_nets: int = 2,
        fixed_alpha: bool = False,
        delay_qvalue: bool = True,
        min_alpha: float | None = None,
        max_alpha: float | None = None,
        loss_function: str = "l2",
        # Gradient control
        detach_action_from_q: bool = False,
        detach_z_from_transformer: bool = True,
        critic_use_mu: bool = True,
        # Separate learning rates
        lr_actor: float = 3e-4,
        lr_encoder: float = 1e-4,
        lr_critic: float = 3e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # SAC-specific
        self.alpha_init = alpha_init
        self.target_entropy = target_entropy
        self.num_qvalue_nets = num_qvalue_nets
        self.fixed_alpha = fixed_alpha
        self.delay_qvalue = delay_qvalue
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.loss_function = loss_function

        # Gradient control
        self.detach_action_from_q = detach_action_from_q
        self.detach_z_from_transformer = detach_z_from_transformer
        self.critic_use_mu = critic_use_mu

        # Separate LRs
        self.lr_actor = lr_actor
        self.lr_encoder = lr_encoder
        self.lr_critic = lr_critic

        # Q-network cache (shared across groups, like _shared_gnn_critic)
        self._shared_qvalue_gnn = None

    # ------------------------------------------------------------------
    # Override: _get_loss
    # ------------------------------------------------------------------
    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> tuple[LossModule, bool]:
        """Create the HGTeamSACLoss module for *group*.

        Configures TD(0) value estimator and sets TensorDict keys.

        Returns:
            ``(loss_module, True)`` — True enables target-network updater.
        """
        if not continuous:
            raise NotImplementedError("HGTeamSAC only supports continuous actions.")

        loss_module = HGTeamSACLoss(
            algorithm=self,
            group=group,
            actor_network=policy_for_loss,
            qvalue_network=self.get_continuous_value_module(group),
            num_qvalue_nets=self.num_qvalue_nets,
            loss_function=self.loss_function,
            alpha_init=self.alpha_init,
            min_alpha=self.min_alpha,
            max_alpha=self.max_alpha,
            action_spec=self.action_spec,
            fixed_alpha=self.fixed_alpha,
            target_entropy=self.target_entropy,
            delay_qvalue=self.delay_qvalue,
        )
        loss_module.set_keys(
            state_action_value=(group, "state_action_value"),
            action=(group, "action"),
            reward=(group, "reward"),
            priority=(group, "td_error"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.experiment_config.gamma)
        return loss_module, True  # True = use target network updater

    # ------------------------------------------------------------------
    # Override: _get_parameters (4 param groups, 3 Adam instances)
    # ------------------------------------------------------------------
    def _get_parameters(self, group: str, loss: LossModule) -> dict[str, Iterable]:
        """Return optimizer param groups with separate LRs for actor/encoder/critic.

        Partitions actor params into GNN encoder (at lr_encoder/N) and
        Transformer head (at lr_actor).  Critic GNN params are similarly
        scaled by 1/N.

        Returns:
            Dict mapping loss keys to parameter lists or param-group dicts.
            Includes ``"loss_alpha"`` if alpha is learnable.
        """
        all_actor_params = list(loss.actor_network_params.flatten_keys().values())

        # Partition actor params into GNN (encoder) vs Transformer (actor)
        gnn_param_ids = set()
        if self._shared_actor_gnn is not None:
            for p in self._shared_actor_gnn.parameters():
                gnn_param_ids.add(id(p))

        # Also include EmbeddingProcessor params in the encoder group
        # (they sit between GNN output and Transformer input)
        embedding_processor_param_ids = set()
        # Walk the actor pipeline to find EmbeddingProcessor modules
        actor_net = loss.actor_network
        root = getattr(actor_net, "module", actor_net)
        for m in root.modules():
            if isinstance(m, EmbeddingProcessor):
                for p in m.parameters():
                    embedding_processor_param_ids.add(id(p))

        encoder_params = []
        transformer_params = []
        for p in all_actor_params:
            if id(p) in gnn_param_ids or id(p) in embedding_processor_param_ids:
                encoder_params.append(p)
            else:
                transformer_params.append(p)

        # Filter shared GNN params for this group (remove other-group params)
        if self.gnn_mode != "none" and self._shared_actor_gnn is not None:
            encoder_params = self._filter_shared_gnn_params(
                encoder_params, self._shared_actor_gnn, group, role="actor"
            )

        # Build param groups with separate LRs
        # Scale shared GNN encoder LR by 1/num_groups to compensate for
        # the shared params receiving one Adam step per group per iteration.
        # Note: same √N bias caveat as HGTeam._split_shared_gnn_param_groups.
        num_groups = len(self.group_map)
        encoder_lr = self.lr_encoder / max(num_groups, 1)
        actor_param_groups = []
        if transformer_params:
            actor_param_groups.append({"params": transformer_params, "lr": self.lr_actor})
        if encoder_params:
            actor_param_groups.append({"params": encoder_params, "lr": encoder_lr})

        # Q-network params
        qvalue_params = list(loss.qvalue_network_params.flatten_keys().values())
        if self._shared_qvalue_gnn is not None:
            qvalue_params = self._filter_shared_gnn_params(
                qvalue_params, self._shared_qvalue_gnn, group, role="critic"
            )
        # Scale shared critic GNN LR
        if self._shared_qvalue_gnn is not None and num_groups > 1:
            shared_critic_ptrs = {p.data_ptr() for p in self._shared_qvalue_gnn.parameters()}
            critic_gnn = [p for p in qvalue_params if p.data_ptr() in shared_critic_ptrs]
            critic_other = [p for p in qvalue_params if p.data_ptr() not in shared_critic_ptrs]
            qvalue_param_groups = []
            if critic_other:
                qvalue_param_groups.append({"params": critic_other, "lr": self.lr_critic})
            if critic_gnn:
                qvalue_param_groups.append(
                    {"params": critic_gnn, "lr": self.lr_critic / num_groups}
                )
        else:
            qvalue_param_groups = [{"params": qvalue_params, "lr": self.lr_critic}]

        items = {
            "loss_actor": actor_param_groups,
            "loss_qvalue": qvalue_param_groups,
        }
        if not self.fixed_alpha:
            items["loss_alpha"] = [loss.log_alpha]

        return items

    # ------------------------------------------------------------------
    # Override: process_batch (no GAE, no minibatch splitting)
    # ------------------------------------------------------------------
    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        """Prepare a collected batch for SAC training.

        Broadcasts shared done/terminated/reward into per-group keys.
        No advantage computation (SAC is off-policy, Q-based).
        """
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
                batch.get(("next", "terminated")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        return batch

    # ------------------------------------------------------------------
    # Override: process_loss_vals (SAC has no entropy to merge)
    # ------------------------------------------------------------------
    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase, batch: TensorDictBase = None
    ) -> TensorDictBase:
        """Post-process SAC loss values (no-op — embedding losses already merged)."""
        # SACLoss returns loss_actor, loss_qvalue, loss_alpha — no merging needed.
        # Embedding aux losses are already added to loss_actor in HGTeamSACLoss.forward().
        return loss_vals

    # ------------------------------------------------------------------
    # Q-network: shared HeteroGNN Q(obs, a, μ) across all agent types
    # ------------------------------------------------------------------
    def get_continuous_value_module(self, group: str) -> TensorDictModule:
        """Build a Q-network that takes per-agent [obs_i, a_i, mu_i] features.

        The Q-network is a single shared HeteroGNN across all agent types.
        Each agent node receives its own concatenated features; cross-agent
        information flows through GNN message passing over edges.
        """
        type_order = list(self.group_map.keys())

        # Determine mu_dim based on the embedding mode
        if self.split_z and self.gnn_mode == "learned_query":
            mu_dim = self.z_query_dim
        elif self.gnn_mode != "none":
            mu_dim = self.z_dim
        else:
            mu_dim = 0

        # --- Build QValueInputPrep modules per group --------------------
        prep_modules = []
        for t in type_order:
            obs_dim = self.observation_spec[t, "observation"].shape[-1]
            action_dim = self.action_spec[t, "action"].shape[-1]
            q_input_key = f"{t}_qvalue_input"

            def _make_prep(grp, o_dim, a_dim, m_dim):
                """Create a closure to avoid late-binding issues."""

                def prep_fn(obs, action, mu):
                    if self.detach_action_from_q:
                        action = action.detach()
                    if m_dim > 0:
                        return torch.cat([obs, action, mu], dim=-1)
                    else:
                        return torch.cat([obs, action], dim=-1)

                return prep_fn

            if self.critic_use_mu and mu_dim > 0:
                prep_modules.append(
                    TensorDictModule(
                        _make_prep(t, obs_dim, action_dim, mu_dim),
                        in_keys=[
                            (t, "observation"),
                            (t, "action"),
                            (t, "embedding_mu"),
                        ],
                        out_keys=[(t, q_input_key)],
                    )
                )
            else:

                def _make_prep_no_mu(grp):
                    def prep_fn(obs, action):
                        if self.detach_action_from_q:
                            action = action.detach()
                        return torch.cat([obs, action], dim=-1)

                    return prep_fn

                prep_modules.append(
                    TensorDictModule(
                        _make_prep_no_mu(t),
                        in_keys=[(t, "observation"), (t, "action")],
                        out_keys=[(t, q_input_key)],
                    )
                )

        # --- Build shared HeteroGNN Q-network ---------------------------
        if self._shared_qvalue_gnn is None:
            critic_input_spec = Composite()
            for t in type_order:
                n_t = len(self.group_map[t])
                obs_dim = self.observation_spec[t, "observation"].shape[-1]
                action_dim = self.action_spec[t, "action"].shape[-1]
                q_input_dim = (
                    obs_dim + action_dim + (mu_dim if self.critic_use_mu and mu_dim > 0 else 0)
                )
                critic_input_spec.set(
                    t,
                    Composite(
                        {f"{t}_qvalue_input": Unbounded(shape=(n_t, q_input_dim))},
                        shape=(n_t,),
                    ),
                )

            # Add graph keys
            if isinstance(self.critic_model_config, HeteroGnnConfig):
                self._add_graph_keys_to_spec(critic_input_spec)

            critic_config = copy.deepcopy(self.critic_model_config)
            if isinstance(critic_config, HeteroGnnConfig):
                critic_config.agent_groups = list(type_order)

                # Build per-type node feature keys (qvalue_input replaces obs)
                critic_config.node_features_keys = {}
                critic_config.node_features_dims = {}

                # Keep grid_node features if present in original config
                base_nfk = dict(self.critic_model_config.node_features_keys or {})
                base_nfd = dict(self.critic_model_config.node_features_dims or {})
                if "grid_node" in base_nfk:
                    critic_config.node_features_keys["grid_node"] = base_nfk["grid_node"]
                if "grid_node" in base_nfd:
                    critic_config.node_features_dims["grid_node"] = base_nfd["grid_node"]

                # Agent types: use qvalue_input as explicit per-node features
                # for each group so the critic receives the concatenated
                # [observation, action, (optional) mu] tensor with the correct
                # per-group agent count.
                for t in type_order:
                    obs_dim = self.observation_spec[t, "observation"].shape[-1]
                    action_dim = self.action_spec[t, "action"].shape[-1]
                    q_input_dim = (
                        obs_dim + action_dim + (mu_dim if self.critic_use_mu and mu_dim > 0 else 0)
                    )
                    critic_config.node_features_keys[t] = f"{t}_qvalue_input"
                    critic_config.node_features_dims[t] = q_input_dim

                # Agent types: use qvalue_input as the sole input feature
                # The original "agents"/"participation_score" features are
                # already part of the observation concatenated into qvalue_input.
                # We exclude observations from node features since qvalue_input
                # already contains them.
                critic_config.exclude_observations_from_node_features = True
                critic_config.cat_observations_to_output = False

                critic_config.agent_node_index_key = "agent_grid_edge_index"

                if self.centralised_value_per_agent:
                    critic_config.prune_non_agent_final_layer = True

            # Output spec: per-agent state_action_value
            critic_output_spec = Composite()
            for t in type_order:
                n_t = len(self.group_map[t])
                critic_output_spec.set(
                    t,
                    Composite(
                        {"state_action_value": Unbounded(shape=(n_t, 1))},
                        shape=(n_t,),
                    ),
                )

            primary_group = type_order[0]
            n_primary = len(self.group_map[primary_group])

            self._shared_qvalue_gnn = critic_config.get_model(
                input_spec=critic_input_spec,
                output_spec=critic_output_spec,
                n_agents=n_primary,
                centralised=not self.centralised_value_per_agent,
                input_has_agent_dim=True,
                agent_group=primary_group,
                share_params=True,
                device=self.device,
                action_spec=self.action_spec,
            )

        # --- Assemble pipeline: prep → GNN → output --------------------
        return TensorDictSequential(*prep_modules, self._shared_qvalue_gnn)

    # ------------------------------------------------------------------
    # Override: _get_policy_for_collection (same as parent)
    # ------------------------------------------------------------------
    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        """Return the collection policy (same as loss policy for SAC)."""
        return policy_for_loss


# ======================================================================
# Config
# ======================================================================


@dataclass
class HGTeamSACConfig(AlgorithmConfig):
    """Configuration for HGTeamSAC — SAC-based variant of HGTeam."""

    # --- HGTeam architecture parameters (same as HGTeamConfig) ----------
    share_param_critic: bool = MISSING
    scale_mapping: str = MISSING
    scale_lb: float = MISSING
    use_tanh_normal: bool = MISSING
    use_beta: bool = MISSING
    beta_min_param: float = MISSING

    share_critic_across_groups: bool = MISSING
    centralised_value_per_agent: bool = MISSING
    gnn_mode: str = MISSING
    z_dim: int = MISSING
    hypernet_actor_feature_dim: int = MISSING
    stochastic_z: bool = MISSING
    embedding_entropy_coef: float = MISSING
    embedding_diversity_coef: float = MISSING

    gnn_num_layers: int = MISSING
    gnn_heads: int = MISSING
    gnn_concat_heads: bool = MISSING
    gnn_use_beta: bool = MISSING
    gnn_self_loops: bool = MISSING
    gnn_agent_node_feature_key: str | None = MISSING
    gnn_agent_node_feature_dim: int = MISSING
    gnn_norm_class: str | None = MISSING
    critic_embed_dim: int = MISSING

    split_z: bool = MISSING
    z_token_dim: int = MISSING
    z_query_dim: int = MISSING
    stochastic_z_query: bool = MISSING

    use_vib: bool = MISSING
    vib_beta: float = MISSING
    vib_warmup_frames: int = MISSING

    # --- SAC-specific parameters ----------------------------------------
    alpha_init: float = MISSING
    target_entropy: float | str = MISSING
    num_qvalue_nets: int = MISSING
    fixed_alpha: bool = MISSING
    delay_qvalue: bool = MISSING
    min_alpha: float | None = MISSING  # null in YAML
    max_alpha: float | None = MISSING  # null in YAML
    loss_function: str = MISSING

    # --- Gradient control -----------------------------------------------
    detach_action_from_q: bool = MISSING
    detach_z_from_transformer: bool = MISSING
    critic_use_mu: bool = MISSING

    # --- Learning rates -------------------------------------------------
    lr_actor: float = MISSING
    lr_encoder: float = MISSING
    lr_critic: float = MISSING

    @staticmethod
    def associated_class() -> type["HGTeamSAC"]:
        return HGTeamSAC

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False

    @staticmethod
    def on_policy() -> bool:
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
