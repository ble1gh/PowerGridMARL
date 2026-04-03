#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""HGTeamHAPPO — Heterogeneous-Agent Proximal Policy Optimisation (HAPPO)
variant of HGTeam.

HAPPO (Kuba et al., 2022) performs sequential policy updates across agent
groups, weighting each group's advantage by a cumulative *factor* — the
product of clipped importance ratios from previously-updated groups.  This
guarantees monotonic improvement under the multi-agent trust-region
constraint.

This file implements the group-level HAPPO variant where each agent group
(e.g. EV, PV, Storage) is a single sequential "slot".  Within a group,
standard PPO with per-agent ratios is used.  The factor propagates
*between* groups only.

Design decisions (see /memories/session/happo-design-decisions.md):
  - Sequencing: group-level (not agent-level)
  - Group ordering: random permutation each epoch (configurable fixed)
  - Factor: cumulative product of clipped group ratios
  - Group ratio: exp(sum_agents(log_pi_new - log_pi_old)) in log-space
  - Encoder update: configurable "accumulated" (default) or "separate_forward"
  - Critic: shared HeteroGNN, reused from HGTeamBase
"""

import random
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.HGTeam import HGTeam, HGTeamBase, HGTeamLoss
from benchmarl.models.common import ModelConfig


class HGTeamHAPPOLoss(HGTeamLoss):
    """ClipPPO loss extended with a HAPPO factor that multiplies advantages.

    The factor is read from the input tensordict at key ``("happo_factor",)``.
    If this key is absent, the loss reduces to standard ClipPPO (factor = 1).
    """

    FACTOR_KEY = "happo_factor"

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # If a factor is stored in the tensordict, inject it into the
        # advantage so that ClipPPOLoss.forward sees ``factor * advantage``.
        # We restore the original advantage after the call.
        adv_key = (self.group, "advantage")
        factor = tensordict.get(self.FACTOR_KEY, None)

        if factor is not None:
            original_adv = tensordict.get(adv_key).clone()
            # Ensure factor broadcasts: (batch,) -> (batch, 1) or (batch, n_agents, 1)
            f = factor
            while f.dim() < original_adv.dim():
                f = f.unsqueeze(-1)
            tensordict.set(adv_key, original_adv * f)
        else:
            original_adv = None

        out = super().forward(tensordict)

        # Restore original advantage
        if original_adv is not None:
            tensordict.set(adv_key, original_adv)

        return out


class HGTeamHAPPO(HGTeamBase):
    """Heterogeneous-Agent Proximal Policy Optimisation with GNN encoders.

    Extends HGTeamBase with HAPPO's sequential group update and factor
    propagation.

    Additional args over HGTeamBase:
        encoder_update_mode: ``"accumulated"`` (default) or ``"separate_forward"``.
            - accumulated: GNN grads accumulate during per-group head backward
              passes; a single ``gnn_optimizer.step()`` is performed after all
              groups.
            - separate_forward: GNN is frozen during head updates; after all
              groups, the GNN is unfrozen and a fresh forward+backward+step
              is performed.
        fixed_order: If True, groups are always updated in the order they
            appear in ``group_map``.  If False (default), a random permutation
            is used each epoch.
    """

    def __init__(
        self,
        clip_epsilon: float,
        entropy_coef: float,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        minibatch_advantage: bool,
        encoder_update_mode: str = "accumulated",
        fixed_order: bool = False,
        critic_use_other_actions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda
        self.minibatch_advantage = minibatch_advantage
        self.encoder_update_mode = encoder_update_mode
        self.fixed_order = fixed_order
        self.critic_use_other_actions = critic_use_other_actions

        if encoder_update_mode not in ("accumulated", "separate_forward"):
            raise ValueError(
                f"encoder_update_mode must be 'accumulated' or "
                f"'separate_forward', got '{encoder_update_mode}'"
            )

    # ------------------------------------------------------------------
    # Loss / parameters / processing — PPO-based, same as HGTeam
    # ------------------------------------------------------------------

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        loss_module = HGTeamHAPPOLoss(
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
            deactivate_vmap=True,
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        actor_params = list(loss.actor_network_params.flatten_keys().values())

        if self.gnn_mode != "none" and self._shared_actor_gnn is not None:
            actor_params = self._filter_shared_gnn_params(
                actor_params, self._shared_actor_gnn, group, role="actor"
            )

            if self.encoder_update_mode == "accumulated":
                # Mode C: all params in one optimizer — GNN accumulates grads
                # across groups, step once at end in train_groups.
                # LR scaling: shared GNN params at lr/n_groups.
                actor_params = self._split_shared_gnn_param_groups(
                    actor_params, self._shared_actor_gnn
                )
            else:
                # Mode B (separate_forward): strip GNN params from per-group
                # optimizer entirely — they'll be handled by gnn_optimizer.
                shared_ptrs = {
                    p.data_ptr() for p in self._shared_actor_gnn.parameters()
                }
                actor_params = [
                    p for p in actor_params
                    if p.data_ptr() not in shared_ptrs
                ]

        critic_params = list(loss.critic_network_params.flatten_keys().values())
        if self._shared_gnn_critic is not None:
            critic_params = self._filter_shared_gnn_params(
                critic_params, self._shared_gnn_critic, group, role="critic"
            )
            critic_params = self._split_shared_gnn_param_groups(
                critic_params, self._shared_gnn_critic
            )

        return {
            "loss_objective": actor_params,
            "loss_critic": critic_params,
        }

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
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
        return loss_vals

    # Reuse HGTeam's critic and shared-critic helpers (inherited from
    # HGTeamBase, but get_critic is defined on HGTeam).
    get_critic = HGTeam.get_critic
    _get_shared_critic = HGTeam._get_shared_critic
    _augment_critic_observations = HGTeam._augment_critic_observations
    _compute_augmented_obs = HGTeam._compute_augmented_obs
    _compute_other_actions_dim = HGTeam._compute_other_actions_dim
    _split_shared_gnn_param_groups = HGTeam._split_shared_gnn_param_groups

    # ------------------------------------------------------------------
    # HAPPO sequential training loop
    # ------------------------------------------------------------------

    def train_groups(self, experiment, batch, current_frames: int):
        """Sequential HAPPO training across groups.

        For each collection iteration:
        1. Process batch for all groups (compute advantages).
        2. Determine group ordering (random or fixed).
        3. For each group in order:
           a. Set HAPPO factor on the loss module.
           b. Run standard PPO optimizer loop (n_optimizer_steps × n_minibatches).
           c. Compute new log-probs, update cumulative factor.
        4. If encoder_update_mode == "accumulated": step GNN optimizer once.
           If encoder_update_mode == "separate_forward": unfreeze GNN, fresh
           forward+backward, step GNN optimizer.
        5. Run callbacks and exploration schedule.
        """
        groups = list(experiment.train_group_map.keys())

        # --- 1. Process batch and fill replay buffers for all groups ---
        for group in groups:
            group_batch = batch.exclude(*experiment._get_excluded_keys(group))
            group_batch = self.process_batch(group, group_batch)
            if not self.has_rnn:
                group_batch = group_batch.reshape(-1)

            group_buffer = experiment.replay_buffers[group]
            group_buffer.extend(group_batch.to(group_buffer.storage.device))

        # --- 2. Determine group ordering ---
        if self.fixed_order:
            group_order = groups
        else:
            group_order = groups.copy()
            random.shuffle(group_order)

        # --- 3. Snapshot old log-probs for all groups ---
        # We need old log-probs to compute the importance ratio after updates.
        # These are already stored in the replay buffer from collection.

        # --- 4. Sequential group updates with factor propagation ---
        # Factor starts as 1 (no weighting for the first group).
        # After each group's PPO update completes, we compute the clipped
        # group importance ratio and accumulate it into the factor for the
        # next group.
        #
        # The factor is per-sample and stored directly into each group's
        # replay buffer under ``HGTeamHAPPOLoss.FACTOR_KEY`` so that
        # sampled minibatches carry the correct factor values.

        cumulative_log_factor = None  # shape: (buffer_size,) in log-space
        factor_key = HGTeamHAPPOLoss.FACTOR_KEY

        for g_idx, group in enumerate(group_order):
            loss_module = experiment.losses[group]
            buffer = experiment.replay_buffers[group]
            buf_len = len(buffer)
            storage_device = buffer.storage.device

            # --- Snapshot old log-probs before update ---
            with torch.no_grad():
                full_data = buffer[:buf_len]
                old_log_probs = full_data.get(
                    (group, "log_prob")
                ).to(experiment.config.train_device)

            # --- Write factor into buffer storage ---
            storage_td = buffer.storage._storage
            if g_idx == 0 or cumulative_log_factor is None:
                # First group: factor = 1 everywhere → no weighting.
                # Ensure the key exists with ones so the loss can read it.
                if factor_key not in storage_td.keys():
                    storage_td.set(
                        factor_key,
                        torch.ones(*storage_td.batch_size, device=storage_device),
                    )
                else:
                    storage_td[factor_key][:buf_len] = 1.0
            else:
                factor_vals = torch.exp(
                    cumulative_log_factor[:buf_len].to(storage_device)
                ).detach()
                if factor_key not in storage_td.keys():
                    storage_td.set(
                        factor_key,
                        torch.ones(*storage_td.batch_size, device=storage_device),
                    )
                storage_td[factor_key][:buf_len] = factor_vals

            # --- Freeze GNN if separate_forward mode ---
            if self.encoder_update_mode == "separate_forward":
                if self._shared_actor_gnn is not None:
                    for p in self._shared_actor_gnn.parameters():
                        p.requires_grad_(False)

            # --- Standard PPO optimizer loop for this group ---
            training_tds = []
            for _ in range(experiment.config.n_optimizer_steps(self.on_policy)):
                for _ in range(
                    -(
                        -experiment.config.train_batch_size(self.on_policy)
                        // experiment.config.train_minibatch_size(self.on_policy)
                    )
                ):
                    training_tds.append(experiment._optimizer_loop(group))
            training_td = torch.stack(training_tds)

            # --- Compute new log-probs and update factor ---
            # Chunked forward pass: process the buffer in minibatches to
            # avoid loading the entire buffer onto GPU at once.  The chunk
            # size matches train_minibatch_size, which is already known to
            # fit in GPU memory from the PPO optimizer loop.
            import contextlib
            chunk_size = experiment.config.train_minibatch_size(self.on_policy)
            with torch.no_grad():
                ctx = (
                    loss_module.actor_network_params.to_module(loss_module.actor_network)
                    if loss_module.functional
                    else contextlib.nullcontext()
                )
                new_lp_chunks = []
                for chunk in full_data.chunk(max(1, -(-full_data.shape[0] // chunk_size)), dim=0):
                    chunk_dev = chunk.to(experiment.config.train_device)
                    with ctx:
                        dist = loss_module.actor_network.get_dist(chunk_dev)
                    action = chunk_dev.get((group, "action"))
                    new_lp_chunks.append(dist.log_prob(action).cpu())
                new_log_probs = torch.cat(new_lp_chunks, dim=0).to(
                    experiment.config.train_device
                )
                # Ensure shape matches old_log_probs
                if new_log_probs.dim() < old_log_probs.dim():
                    new_log_probs = new_log_probs.unsqueeze(-1)
                elif new_log_probs.dim() > old_log_probs.dim():
                    old_log_probs = old_log_probs.squeeze(-1)

            # Group ratio in log-space: sum over agents
            log_ratio_per_agent = new_log_probs - old_log_probs
            # Sum over non-batch dims (agent / action dims) to get per-sample
            # group-level log-ratio.  Batch dims come from the buffer storage.
            n_batch_dims = len(full_data.batch_size)
            extra_dims = log_ratio_per_agent.dim() - n_batch_dims
            if extra_dims > 0:
                sum_dims = tuple(range(n_batch_dims, log_ratio_per_agent.dim()))
                log_group_ratio = log_ratio_per_agent.sum(dim=sum_dims)
            else:
                log_group_ratio = log_ratio_per_agent

            # Clip the ratio: min(ratio, clip(ratio, 1-eps, 1+eps))
            group_ratio = torch.exp(log_group_ratio)
            clipped_ratio = torch.clamp(
                group_ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            )
            factor_update = torch.min(group_ratio, clipped_ratio)
            log_factor_update = torch.log(factor_update.clamp(min=1e-8))

            # Accumulate into cumulative factor (log-space)
            if cumulative_log_factor is None:
                cumulative_log_factor = log_factor_update.cpu()
            else:
                cumulative_log_factor = (
                    cumulative_log_factor + log_factor_update.cpu()
                )

            # --- Log training and callbacks ---
            experiment.logger.log_training(
                group, training_td, step=experiment.n_iters_performed
            )
            experiment._on_train_end(training_td, group)

            # --- Exploration schedule ---
            if isinstance(experiment.group_policies[group], TensorDictSequential):
                explore_layer = experiment.group_policies[group][-1]
            else:
                explore_layer = experiment.group_policies[group]
            if hasattr(explore_layer, "step"):
                explore_layer.step(current_frames)

        # --- 5. GNN encoder update ---
        if self._shared_actor_gnn is not None:
            if self.encoder_update_mode == "accumulated":
                # Mode C: GNN grads have been accumulated during per-group
                # backward passes (since GNN params were in the optimizer
                # with scaled LR).  The per-group optimizer.step() calls
                # already stepped the GNN params with the scaled LR.
                # No additional action needed — this is handled by the
                # _split_shared_gnn_param_groups LR scaling.
                pass

            elif self.encoder_update_mode == "separate_forward":
                # Mode B: Unfreeze GNN, run a fresh forward+backward on
                # the last group's buffer data, then step.
                for p in self._shared_actor_gnn.parameters():
                    p.requires_grad_(True)

                # Build a GNN-only optimizer if not already cached
                if not hasattr(self, "_gnn_optimizer"):
                    gnn_params = list(self._shared_actor_gnn.parameters())
                    self._gnn_optimizer = torch.optim.Adam(
                        gnn_params,
                        lr=self.experiment_config.lr,
                        eps=self.experiment_config.adam_eps,
                    )

                # Use the last group's buffer for the encoder update
                last_group = group_order[-1]
                buffer = experiment.replay_buffers[last_group]
                subdata = buffer.sample().to(experiment.config.train_device)

                # Forward through loss to get gradients on GNN
                loss_vals = experiment.losses[last_group](subdata)
                loss_vals = self.process_loss_vals(last_group, loss_vals, batch=subdata)
                obj_loss = loss_vals.get("loss_objective", None)
                if obj_loss is not None:
                    self._gnn_optimizer.zero_grad()
                    obj_loss.backward()
                    if experiment.config.clip_grad_norm and experiment.config.clip_grad_val is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self._shared_actor_gnn.parameters(),
                            experiment.config.clip_grad_val,
                        )
                    self._gnn_optimizer.step()
                    self._gnn_optimizer.zero_grad()


@dataclass
class HGTeamHAPPOConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`HGTeamHAPPO`."""

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

    # HGTeam-specific parameters (shared encoder)
    share_critic_across_groups: bool = MISSING
    centralised_value_per_agent: bool = MISSING
    gnn_mode: str = MISSING
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
    critic_use_other_actions: bool = MISSING
    gnn_norm_class: Optional[str] = MISSING
    critic_embed_dim: int = MISSING

    # Split-z parameters
    split_z: bool = MISSING
    z_token_dim: int = MISSING
    z_query_dim: int = MISSING
    stochastic_z_query: bool = MISSING

    # VIB parameters
    use_vib: bool = MISSING
    vib_beta: float = MISSING
    vib_warmup_frames: int = MISSING

    # HAPPO-specific parameters
    encoder_update_mode: str = MISSING
    fixed_order: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return HGTeamHAPPO

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
