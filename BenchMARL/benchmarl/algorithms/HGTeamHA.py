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

import contextlib
import random
import warnings
from collections.abc import Iterable
from dataclasses import MISSING, dataclass

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.HGTeam import HGTeam, HGTeamBase, HGTeamLoss
from benchmarl.algorithms.hgteam_modules import EmbeddingProcessor


class HGTeamHAPPOLoss(HGTeamLoss):
    """ClipPPO loss extended with a HAPPO factor that multiplies advantages.

    The factor is read from the input tensordict at key ``("happo_factor",)``.
    If this key is absent, the loss reduces to standard ClipPPO (factor = 1).
    """

    # Re-declare so TorchRL's LossModule sees them on this class directly
    # (it checks __annotations__ on the immediate class, not parents).
    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictBase
    critic_network_params: TensorDictBase
    target_actor_network_params: TensorDictBase
    target_critic_network_params: TensorDictBase

    FACTOR_KEY = "happo_factor"

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute PPO loss with HAPPO factor-weighted advantages.

        Reads ``happo_factor`` from *tensordict* and multiplies it into
        the advantage before delegating to ``HGTeamLoss.forward()``.  The
        original advantage is restored afterwards.

        Args:
            tensordict: Batch tensordict with advantages and optional
                ``happo_factor`` key.

        Returns:
            TensorDict with loss keys (same as ``HGTeamLoss.forward``).
        """
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
        encoder_n_optimizer_steps: int | None = None,
        encoder_lr: float | None = None,
        **kwargs,
    ) -> None:
        """Initialise HGTeamHAPPO with PPO + HAPPO-specific parameters.

        Args:
            clip_epsilon: PPO clipping parameter ε.
            entropy_coef: Entropy bonus coefficient.
            critic_coef: Critic loss weight.
            loss_critic_type: Critic loss type (``"smooth_l1"`` or ``"l2"``).
            lmbda: GAE λ for advantage estimation.
            minibatch_advantage: If True, compute GAE per minibatch.
            encoder_update_mode: ``"accumulated"`` (GNN grads accumulated
                across groups, stepped at lr/3) or ``"separate_forward"``
                (GNN frozen during head updates, separate forward pass).
            fixed_order: If True, use deterministic group ordering.
            critic_use_other_actions: If True, pass actions as GNN edge
                features on the critic.
            **kwargs: Forwarded to ``HGTeamBase.__init__``.
        """
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
        self.encoder_n_optimizer_steps = encoder_n_optimizer_steps
        self.encoder_lr = encoder_lr

        if encoder_update_mode not in ("accumulated", "separate_forward", "coop_encoder"):
            raise ValueError(
                f"encoder_update_mode must be 'accumulated', "
                f"'separate_forward', or 'coop_encoder', got '{encoder_update_mode}'"
            )

    # ------------------------------------------------------------------
    # Loss / parameters / processing — PPO-based, same as HGTeam
    # ------------------------------------------------------------------

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> tuple[LossModule, bool]:
        """Create HGTeamHAPPOLoss for *group* with GAE value estimator.

        Identical to ``HGTeam._get_loss`` except uses ``HGTeamHAPPOLoss``
        which multiplies advantages by the HAPPO factor.

        Returns:
            ``(loss_module, False)`` — False means no target-network updater.
        """
        loss_module = HGTeamHAPPOLoss(
            algorithm=self,
            group=group,
            actor_network=policy_for_loss,
            critic_network=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coef,
            critic_coeff=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            reduction="none",
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

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> dict[str, Iterable]:
        """Return optimizer param groups with encoder-update-mode handling.

        In ``accumulated`` mode, shared GNN params are included at lr/N.
        In ``separate_forward`` mode, GNN params are stripped (handled by
        a dedicated GNN optimizer in ``train_groups``).
        """
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
                # separate_forward / coop_encoder: strip GNN params from
                # per-group optimizer — they're handled by gnn_optimizer.
                shared_ptrs = {p.data_ptr() for p in self._shared_actor_gnn.parameters()}
                actor_params = [p for p in actor_params if p.data_ptr() not in shared_ptrs]

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
        """Return the collection policy (same as loss policy for on-policy)."""
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        """Prepare a collected batch for HAPPO training.

        Same as ``HGTeam.process_batch``: broadcasts shared keys, computes
        GAE advantages, zeros inactive agents (D7), and applies S8a per-slot
        masked advantage normalisation.
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

        loss = self.get_loss_and_updater(group)[0]
        if self.minibatch_advantage:
            increment = -(
                -self.experiment.config.train_minibatch_size(self.on_policy) // batch.shape[1]
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

        # --- Zero out advantages for inactive (padded) agent slots ---
        active_mask = batch.get((group, "active_mask"), None)
        if active_mask is not None:
            adv = batch.get((group, "advantage"))
            vtarg = batch.get((group, "value_target"))
            vval = batch.get((group, "state_value"))
            m = active_mask
            while m.dim() < adv.dim():
                m = m.unsqueeze(-1)
            m = m.expand_as(adv)
            adv[~m] = 0.0
            vtarg[~m] = vval[~m].detach()

            # --- S8a: Per-slot masked advantage normalization ---
            # See HGTeam.process_batch for full rationale.
            agent_dim = active_mask.dim() - 1  # last dim of active_mask
            n_agents = active_mask.shape[agent_dim]
            for slot in range(n_agents):
                slot_active = active_mask.select(agent_dim, slot)  # (*batch_dims) bool
                if slot_active.sum() > 1:
                    slot_adv_view = adv.select(agent_dim, slot)  # (*batch_dims, 1)
                    slot_vals = slot_adv_view[slot_active]  # (n_active, 1)
                    slot_adv_view[slot_active] = (
                        slot_vals - slot_vals.mean()
                    ) / slot_vals.std(correction=0).clamp(min=1e-7)
        else:
            # No active_mask — fall back to standard per-slot normalization.
            warnings.warn(
                f"HGTeamHAPPO.process_batch({group}): 'active_mask' not found. "
                "Falling back to standard per-slot advantage normalization "
                "(no inactive-agent masking).",
                stacklevel=2,
            )
            adv = batch.get((group, "advantage"))
            agent_dim = adv.dim() - 2  # second-to-last: (*batch_dims, n_agents, 1)
            n_agents = adv.shape[agent_dim]
            for slot in range(n_agents):
                slot_adv_view = adv.select(agent_dim, slot)  # (*batch_dims, 1)
                if slot_adv_view.numel() > 1:
                    slot_adv_view.copy_(
                        (slot_adv_view - slot_adv_view.mean())
                        / slot_adv_view.std(correction=0).clamp(min=1e-7)
                    )

        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase, batch: TensorDictBase = None
    ) -> TensorDictBase:
        """Post-process loss values: merge entropy loss into objective."""
        loss_vals.set("loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"])
        del loss_vals["loss_entropy"]
        return loss_vals

    # Reuse HGTeam's critic and shared-critic helpers (inherited from
    # HGTeamBase, but get_critic is defined on HGTeam).
    get_critic = HGTeam.get_critic
    _get_shared_critic = HGTeam._get_shared_critic
    _compute_other_actions_dim = HGTeam._compute_other_actions_dim
    _split_shared_gnn_param_groups = HGTeam._split_shared_gnn_param_groups

    # ------------------------------------------------------------------
    # Cooperative encoder update (Phase 0 — runs before HAPPO head updates)
    # ------------------------------------------------------------------

    def _coop_encoder_update(
        self,
        experiment: "Experiment",  # noqa: F821
        groups: list[str],
    ) -> None:
        """Cooperative GNN update with mean per-agent advantage (D12).

        Runs BEFORE the sequential HAPPO head updates so that the GNN
        receives gradients through clean (pre-HAPPO) head Jacobians.
        The cooperative advantage is the z-scored mean of raw per-agent
        advantages across all active agents.

        After this method returns, the GNN is re-frozen (requires_grad=False)
        so that Phase 1 head updates don't accumulate GNN gradients.
        """
        if self._shared_actor_gnn is None:
            return

        for p in self._shared_actor_gnn.parameters():
            p.requires_grad_(True)

        gnn_lr = self.encoder_lr if self.encoder_lr is not None else self.experiment_config.lr
        if not hasattr(self, "_gnn_optimizer") or self._gnn_lr != gnn_lr:
            gnn_params = list(self._shared_actor_gnn.parameters())
            self._gnn_optimizer = torch.optim.Adam(
                gnn_params,
                lr=gnn_lr,
                eps=self.experiment_config.adam_eps,
            )
            self._gnn_lr = gnn_lr

        train_device = experiment.config.train_device
        chunk_size = experiment.config.train_minibatch_size(self.on_policy)

        # --- Compute cooperative advantage from all group buffers ---
        sum_raw_adv = None
        sum_coop_sval = None
        sum_coop_vtarg = None
        total_active = None
        buf_len = None

        for g in groups:
            gbuf = experiment.replay_buffers[g]
            glen = len(gbuf)
            if buf_len is None:
                buf_len = glen
            gdata = gbuf[:glen]
            vtarg = gdata.get((g, "value_target"))
            sval = gdata.get((g, "state_value"))
            amask = gdata.get((g, "active_mask"), None)

            raw_adv = vtarg - sval
            if amask is not None:
                m = amask.float().unsqueeze(-1)
            else:
                m = torch.ones_like(raw_adv)

            agent_dim = -2
            g_sum = (raw_adv * m).sum(dim=agent_dim)
            g_sval = (sval * m).sum(dim=agent_dim)
            g_vtarg = (vtarg * m).sum(dim=agent_dim)
            g_nact = m.sum(dim=agent_dim)

            if sum_raw_adv is None:
                sum_raw_adv = g_sum
                sum_coop_sval = g_sval
                sum_coop_vtarg = g_vtarg
                total_active = g_nact
            else:
                sum_raw_adv = sum_raw_adv + g_sum
                sum_coop_sval = sum_coop_sval + g_sval
                sum_coop_vtarg = sum_coop_vtarg + g_vtarg
                total_active = total_active + g_nact

        coop_adv = sum_raw_adv / total_active.clamp(min=1)
        coop_sval = sum_coop_sval / total_active.clamp(min=1)
        coop_vtarg = sum_coop_vtarg / total_active.clamp(min=1)

        # Z-score normalize cooperative advantage
        coop_adv = (coop_adv - coop_adv.mean()) / coop_adv.std().clamp(min=1e-7)

        # --- Log cooperative critic loss (tracking only) ---
        coop_critic_loss = ((coop_vtarg - coop_sval) ** 2).mean().item()
        experiment.logger.log(
            {
                "train/coop_encoder/coop_critic_loss": coop_critic_loss,
                "train/coop_encoder/coop_adv_std_raw": (
                    sum_raw_adv / total_active.clamp(min=1)
                ).std().item(),
            },
            step=experiment.n_iters_performed,
        )

        # --- Cooperative clipped PPO update for encoder (with SNI) ---
        n_optimizer_steps = (
            self.encoder_n_optimizer_steps
            if self.encoder_n_optimizer_steps is not None
            else experiment.config.n_optimizer_steps(self.on_policy)
        )
        n_minibatches = max(1, -(-buf_len // chunk_size))

        # VIB beta
        beta_eff = 0.0
        if self.use_vib:
            total_frames = experiment.total_frames
            if self.vib_warmup_frames > 0:
                beta_eff = min(
                    self.vib_beta,
                    self.vib_beta * total_frames / self.vib_warmup_frames,
                )
            else:
                beta_eff = self.vib_beta

        # --- Find EmbeddingProcessors for SNI (one per group) ---
        _ep_map: dict[str, EmbeddingProcessor | None] = {}
        for g in groups:
            _ep_map[g] = None
            lm = experiment.losses[g]
            for _m in lm.actor_network.modules():
                if isinstance(_m, EmbeddingProcessor):
                    _ep_map[g] = _m
                    break

        # --- Pre-load full buffer data per group (on CPU) ---
        gbuf_full: dict[str, TensorDictBase] = {}
        for g in groups:
            gbuf_full[g] = experiment.replay_buffers[g][:buf_len]

        # --- Compute old_log_prob with SNI (collection-time params) ---
        # Nothing has been updated yet (no Phase 1), so these log-probs
        # are from the actual behavior policy — the cleanest baseline.
        old_lp_sni: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for g in groups:
                lm = experiment.losses[g]
                ctx_params = (
                    lm.actor_network_params.to_module(lm.actor_network)
                    if lm.functional
                    else contextlib.nullcontext()
                )
                ep = _ep_map[g]
                lp_chunks = []
                for chunk in gbuf_full[g].chunk(
                    max(1, -(-gbuf_full[g].shape[0] // chunk_size)), dim=0
                ):
                    chunk_dev = chunk.to(train_device)
                    sni_ctx = ep.deterministic_mode() if ep is not None else contextlib.nullcontext()
                    with sni_ctx, ctx_params:
                        dist_old = lm.actor_network.get_dist(chunk_dev)
                    action = chunk_dev.get((g, "action"))
                    lp_chunks.append(dist_old.log_prob(action).cpu())
                old_lp_sni[g] = torch.cat(lp_chunks, dim=0)

        # Logging accumulators
        _log_coop_obj = 0.0
        _log_encoder_loss = 0.0
        _log_vib_kl = 0.0
        _log_ratio_mean = 0.0
        _log_ratio_max = 0.0
        _log_clip_frac = 0.0
        _n_iters = 0

        clip_eps = self.clip_epsilon
        ca = coop_adv

        for _epoch in range(n_optimizer_steps):
            for _mb in range(n_minibatches):
                mb_start = _mb * chunk_size
                mb_end = min(mb_start + chunk_size, buf_len)

                self._gnn_optimizer.zero_grad()

                coop_obj_num = torch.tensor(0.0, device=train_device)
                total_active_count = torch.tensor(0.0, device=train_device)
                vib_kl_total = torch.tensor(0.0, device=train_device)
                n_vib_groups = 0
                mb_ratio_sum = 0.0
                mb_ratio_max = 0.0
                mb_clip_count = 0
                mb_total_count = 0

                ca_mb = ca[mb_start:mb_end].to(train_device)

                for g in groups:
                    gdata_mb = gbuf_full[g][mb_start:mb_end].to(train_device)
                    old_lp_mb = old_lp_sni[g][mb_start:mb_end].to(train_device)

                    lm = experiment.losses[g]
                    ctx_params = (
                        lm.actor_network_params.to_module(lm.actor_network)
                        if lm.functional
                        else contextlib.nullcontext()
                    )
                    ep = _ep_map[g]
                    sni_ctx = ep.deterministic_mode() if ep is not None else contextlib.nullcontext()
                    with sni_ctx, ctx_params:
                        dist_new = lm.actor_network.get_dist(gdata_mb)
                    action = gdata_mb.get((g, "action"))
                    new_lp = dist_new.log_prob(action)

                    log_ratio = new_lp - old_lp_mb
                    ratio = torch.exp(log_ratio)
                    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

                    ca_g = ca_mb
                    while ca_g.dim() < ratio.dim():
                        ca_g = ca_g.unsqueeze(-1)

                    gain1 = ratio * ca_g.detach()
                    gain2 = clipped_ratio * ca_g.detach()
                    gain = torch.min(gain1, gain2)

                    amask = gdata_mb.get((g, "active_mask"), None)
                    if amask is not None:
                        am = amask.float()
                        while am.dim() < gain.dim():
                            am = am.unsqueeze(-1)
                        coop_obj_num = coop_obj_num + (gain * am).sum()
                        total_active_count = total_active_count + am.sum()
                        with torch.no_grad():
                            mb_ratio_sum += (ratio * am).sum().item()
                            active_max = ratio.where(am.bool(), torch.zeros_like(ratio)).max().item()
                            mb_ratio_max = max(mb_ratio_max, active_max)
                            mb_clip_count += ((ratio != clipped_ratio) * am).sum().item()
                            mb_total_count += am.sum().item()
                    else:
                        coop_obj_num = coop_obj_num + gain.sum()
                        total_active_count = total_active_count + torch.tensor(
                            float(gain.numel()), device=train_device,
                        )
                        with torch.no_grad():
                            mb_ratio_sum += ratio.sum().item()
                            mb_ratio_max = max(mb_ratio_max, ratio.max().item())
                            mb_clip_count += (ratio != clipped_ratio).sum().item()
                            mb_total_count += ratio.numel()

                    # VIB KL with active_mask
                    if self.use_vib:
                        mu = gdata_mb.get((g, "embedding_mu"), None)
                        logvar = gdata_mb.get((g, "embedding_logvar"), None)
                        if mu is not None and logvar is not None:
                            kl_per_agent = -0.5 * (
                                1 + logvar - mu.pow(2) - logvar.exp()
                            ).sum(dim=-1)
                            if amask is not None:
                                am_f = amask.float()
                                kl = (kl_per_agent * am_f).sum() / am_f.sum().clamp(min=1)
                            else:
                                kl = kl_per_agent.mean()
                            vib_kl_total = vib_kl_total + kl
                            n_vib_groups += 1

                coop_obj = coop_obj_num / total_active_count.clamp(min=1)

                encoder_loss = -coop_obj
                if n_vib_groups > 0 and beta_eff > 0:
                    encoder_loss = encoder_loss + beta_eff * (vib_kl_total / n_vib_groups)

                encoder_loss.backward()

                if experiment.config.clip_grad_norm and experiment.config.clip_grad_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._shared_actor_gnn.parameters(),
                        experiment.config.clip_grad_val,
                    )
                self._gnn_optimizer.step()
                self._gnn_optimizer.zero_grad()

                _log_coop_obj += coop_obj.item()
                _log_encoder_loss += encoder_loss.item()
                if n_vib_groups > 0:
                    _log_vib_kl += (vib_kl_total / n_vib_groups).item()
                if mb_total_count > 0:
                    _log_ratio_mean += mb_ratio_sum / mb_total_count
                    _log_ratio_max = max(_log_ratio_max, mb_ratio_max)
                    _log_clip_frac += mb_clip_count / mb_total_count
                _n_iters += 1

        # --- Log coop_encoder metrics ---
        if _n_iters > 0:
            log_dict = {
                "train/coop_encoder/coop_obj": _log_coop_obj / _n_iters,
                "train/coop_encoder/encoder_loss": _log_encoder_loss / _n_iters,
                "train/coop_encoder/n_encoder_iters": _n_iters,
                "train/coop_encoder/ratio_mean": _log_ratio_mean / _n_iters,
                "train/coop_encoder/ratio_max": _log_ratio_max,
                "train/coop_encoder/clip_fraction": _log_clip_frac / _n_iters,
            }
            if self.use_vib:
                log_dict["train/coop_encoder/vib_kl_masked"] = _log_vib_kl / _n_iters
                log_dict["train/coop_encoder/vib_beta_eff"] = beta_eff
            experiment.logger.log(log_dict, step=experiment.n_iters_performed)

        # Re-freeze GNN for Phase 1 head updates
        for p in self._shared_actor_gnn.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # HAPPO sequential training loop
    # ------------------------------------------------------------------

    def train_groups(
        self,
        experiment: "Experiment",  # noqa: F821
        batch: TensorDictBase,
        current_frames: int,
    ) -> None:
        """Sequential HAPPO training across groups.

        For each collection iteration:
        1. Process batch for all groups (compute advantages).
        2. Determine group ordering (random or fixed).
        2b. If coop_encoder: run Phase 0 (cooperative GNN update) before
            head updates — gives the GNN clean gradients through unmodified
            head Jacobians.
        3. For each group in order:
           a. Set HAPPO factor on the loss module.
           b. Run standard PPO optimizer loop (n_optimizer_steps × n_minibatches).
           c. Compute new log-probs, update cumulative factor.
        4. If encoder_update_mode == "accumulated": GNN already stepped.
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

        # --- 2b. Phase 0: cooperative GNN update (before head updates) ---
        if self.encoder_update_mode == "coop_encoder" and self._shared_actor_gnn is not None:
            self._coop_encoder_update(experiment, groups)

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
        _factor_diagnostics = {}  # group -> (n_active_per_sample, factor_update)

        for g_idx, group in enumerate(group_order):
            loss_module = experiment.losses[group]
            buffer = experiment.replay_buffers[group]
            buf_len = len(buffer)
            storage_device = buffer.storage.device

            # --- Snapshot old log-probs under the *current* policy ---
            # In accumulated mode, previous groups' PPO updates have already
            # stepped the shared GNN, so the stored collection-time log-probs
            # no longer reflect the current encoder.  We re-evaluate log-probs
            # under the current policy so that the HAPPO factor captures only
            # this group's policy change, not shared-encoder drift from
            # earlier groups.
            # In coop_encoder mode, Phase 0 has updated the GNN, so ALL
            # groups (including g_idx=0) need re-evaluated old_log_probs.

            chunk_size_snap = experiment.config.train_minibatch_size(self.on_policy)
            _need_reeval = (
                (g_idx > 0 and self.encoder_update_mode == "accumulated")
                or self.encoder_update_mode == "coop_encoder"
            )
            with torch.no_grad():
                full_data = buffer[:buf_len]
                if _need_reeval:
                    ctx_snap = (
                        loss_module.actor_network_params.to_module(loss_module.actor_network)
                        if loss_module.functional
                        else contextlib.nullcontext()
                    )
                    lp_chunks = []
                    for chunk in full_data.chunk(
                        max(1, -(-full_data.shape[0] // chunk_size_snap)), dim=0
                    ):
                        chunk_dev = chunk.to(experiment.config.train_device)
                        with ctx_snap:
                            dist_snap = loss_module.actor_network.get_dist(chunk_dev)
                        action_snap = chunk_dev.get((group, "action"))
                        lp_chunks.append(dist_snap.log_prob(action_snap).cpu())
                    old_log_probs = torch.cat(lp_chunks, dim=0).to(experiment.config.train_device)
                else:
                    old_log_probs = full_data.get((group, "log_prob")).to(
                        experiment.config.train_device
                    )

            # --- SNI: recompute sample_log_prob with deterministic z=mu ---
            # Collection stored log_prob with sampled z.  HGTeamLoss now uses
            # deterministic_mode (z=mu) for new_log_prob in PPO, so the buffer's
            # old_log_prob must also use z=mu for consistent importance ratios.
            if self.use_vib:
                _ep = None
                for _m in loss_module.actor_network.modules():
                    if isinstance(_m, EmbeddingProcessor):
                        _ep = _m
                        break
                if _ep is not None:
                    with torch.no_grad():
                        ctx_sni = (
                            loss_module.actor_network_params.to_module(
                                loss_module.actor_network
                            )
                            if loss_module.functional
                            else contextlib.nullcontext()
                        )
                        sni_lp_chunks = []
                        for chunk in full_data.chunk(
                            max(1, -(-full_data.shape[0] // chunk_size_snap)), dim=0
                        ):
                            chunk_dev = chunk.to(experiment.config.train_device)
                            with _ep.deterministic_mode(), ctx_sni:
                                dist_sni = loss_module.actor_network.get_dist(chunk_dev)
                            action_sni = chunk_dev.get((group, "action"))
                            sni_lp_chunks.append(dist_sni.log_prob(action_sni).cpu())
                        sni_log_probs = torch.cat(sni_lp_chunks, dim=0)
                        # Write SNI log-probs back to buffer storage
                        storage_td = buffer.storage._storage
                        storage_td[(group, "log_prob")][:buf_len] = sni_log_probs.to(
                            storage_device
                        )

            # --- Write factor into buffer storage ---
            storage_td = buffer.storage._storage
            if g_idx == 0 or cumulative_log_factor is None:
                # First group: factor = 1 everywhere → no weighting.
                # Ensure the key exists with ones so the loss can read it.
                if factor_key not in storage_td:
                    storage_td.set(
                        factor_key,
                        torch.ones(*storage_td.batch_size, device=storage_device),
                    )
                else:
                    storage_td[factor_key][:buf_len] = 1.0
            else:
                factor_vals = torch.exp(cumulative_log_factor[:buf_len].to(storage_device)).detach()
                if factor_key not in storage_td:
                    storage_td.set(
                        factor_key,
                        torch.ones(*storage_td.batch_size, device=storage_device),
                    )
                storage_td[factor_key][:buf_len] = factor_vals

            # --- Freeze GNN if separate_forward or coop_encoder mode ---
            if self.encoder_update_mode in ("separate_forward", "coop_encoder"):
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

            chunk_size = experiment.config.train_minibatch_size(self.on_policy)
            with torch.no_grad():
                ctx = (
                    loss_module.actor_network_params.to_module(loss_module.actor_network)
                    if loss_module.functional
                    else contextlib.nullcontext()
                )
                # SNI: use deterministic z=mu so new_log_probs are consistent
                # with the SNI-overwritten old_log_probs in buffer (Step 7).
                _ep_factor = None
                if self.use_vib:
                    for _m in loss_module.actor_network.modules():
                        if isinstance(_m, EmbeddingProcessor):
                            _ep_factor = _m
                            break
                new_lp_chunks = []
                for chunk in full_data.chunk(max(1, -(-full_data.shape[0] // chunk_size)), dim=0):
                    chunk_dev = chunk.to(experiment.config.train_device)
                    sni = _ep_factor.deterministic_mode() if _ep_factor is not None else contextlib.nullcontext()
                    with sni, ctx:
                        dist = loss_module.actor_network.get_dist(chunk_dev)
                    action = chunk_dev.get((group, "action"))
                    new_lp_chunks.append(dist.log_prob(action).cpu())
                new_log_probs = torch.cat(new_lp_chunks, dim=0).to(experiment.config.train_device)
                # Ensure shape matches old_log_probs
                if new_log_probs.dim() < old_log_probs.dim():
                    new_log_probs = new_log_probs.unsqueeze(-1)
                elif new_log_probs.dim() > old_log_probs.dim():
                    old_log_probs = old_log_probs.squeeze(-1)

            # Group ratio in log-space: sum over agents
            log_ratio_per_agent = new_log_probs - old_log_probs

            # Mask inactive agent log-ratios so padded slots don't
            # corrupt the HAPPO factor passed to subsequent groups.
            agent_mask = full_data.get((group, "active_mask"), None)
            if agent_mask is not None:
                am = agent_mask.to(log_ratio_per_agent.device).float()
                while am.dim() < log_ratio_per_agent.dim():
                    am = am.unsqueeze(-1)
                log_ratio_per_agent = log_ratio_per_agent * am

            # Sum over non-batch dims (agent / action dims) to get per-sample
            # group-level log-ratio.  Batch dims come from the buffer storage.
            n_batch_dims = len(full_data.batch_size)
            extra_dims = log_ratio_per_agent.dim() - n_batch_dims
            if extra_dims > 0:
                sum_dims = tuple(range(n_batch_dims, log_ratio_per_agent.dim()))
                log_group_ratio = log_ratio_per_agent.sum(dim=sum_dims)
            else:
                log_group_ratio = log_ratio_per_agent

            # --- Mean log-ratio normalization (Option A) ---
            # Divide the summed log-ratio by the number of active agents so
            # the group factor becomes the *geometric mean* of per-agent
            # importance ratios:  (∏ rᵢ)^{1/n} instead of ∏ rᵢ.
            # Without this, the product's variance grows exponentially with
            # agent count, causing systematic clipping bias for larger groups
            # and inconsistent factor magnitude across samples with different
            # numbers of active agents.
            if agent_mask is not None:
                n_active = agent_mask.float().sum(dim=-1).clamp(min=1)
                # n_active shape: (batch,) or (batch, ...) — reduce extra dims
                while n_active.dim() > n_batch_dims:
                    n_active = n_active.sum(dim=-1)
                log_group_ratio = log_group_ratio / n_active.to(log_group_ratio.device)

            # Clip the ratio with HAPPO's *pessimistic* (asymmetric) clamp.
            # min(ratio, clip(ratio, 1-ε, 1+ε)) caps the upside at 1+ε
            # (preventing an over-optimistic factor from inflating downstream
            # advantages) but allows unrestricted downside: if a previous
            # group's policy worsened significantly (ratio ≪ 1-ε), the factor
            # shrinks freely, dampening subsequent groups' updates.
            # This asymmetry is required for the monotonic-improvement
            # guarantee in HAPPO (Kuba et al., 2022, Theorem 1).
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
                cumulative_log_factor = cumulative_log_factor + log_factor_update.cpu()

            # Store per-group factor diagnostics for logging
            _factor_diagnostics[group] = (
                n_active.detach().cpu() if agent_mask is not None else torch.ones(buf_len),
                factor_update.detach().cpu(),
            )

            # --- Log training and callbacks ---
            experiment.logger.log_training(group, training_td, step=experiment.n_iters_performed)
            experiment._on_train_end(training_td, group)

            # --- Exploration schedule ---
            if isinstance(experiment.group_policies[group], TensorDictSequential):
                explore_layer = experiment.group_policies[group][-1]
            else:
                explore_layer = experiment.group_policies[group]
            if hasattr(explore_layer, "step"):
                explore_layer.step(current_frames)

        # --- 4b. HAPPO factor diagnostics ---
        if _factor_diagnostics and cumulative_log_factor is not None:
            cum_factor = torch.exp(cumulative_log_factor)  # (buffer_size,)
            step = experiment.n_iters_performed
            # Scalar summaries (every iteration)
            factor_log = {
                "train/happo_factor/cum_factor_mean": cum_factor.mean().item(),
                "train/happo_factor/cum_factor_std": cum_factor.std().item(),
                "train/happo_factor/cum_factor_max": cum_factor.max().item(),
                "train/happo_factor/cum_factor_min": cum_factor.min().item(),
            }
            for grp, (n_act, f_upd) in _factor_diagnostics.items():
                factor_log[f"train/happo_factor/{grp}_factor_mean"] = f_upd.mean().item()
                factor_log[f"train/happo_factor/{grp}_n_active_mean"] = n_act.float().mean().item()
            experiment.logger.log(factor_log, step=step)

            # Detailed wandb.Table (every 10 iterations) for scatter plots
            if step % 10 == 0:
                try:
                    import wandb
                    if wandb.run is not None:
                        # Total active agents across all groups per sample
                        n_active_total = sum(
                            diag[0].float() for diag in _factor_diagnostics.values()
                        )  # (buffer_size,)
                        # Subsample to cap table size
                        n_samples = min(512, cum_factor.shape[0])
                        idx = torch.randperm(cum_factor.shape[0])[:n_samples]
                        columns = ["n_active_total", "cum_factor"]
                        for grp in _factor_diagnostics:
                            columns += [f"{grp}_n_active", f"{grp}_factor"]
                        data = []
                        for i in idx.tolist():
                            row = [n_active_total[i].item(), cum_factor[i].item()]
                            for grp, (n_act, f_upd) in _factor_diagnostics.items():
                                row += [n_act[i].item(), f_upd[i].item()]
                            data.append(row)
                        table = wandb.Table(columns=columns, data=data)
                        wandb.log({"train/happo_factor_table": table}, step=step)
                except ImportError:
                    pass

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
                # Minibatched clipped PPO update for the shared GNN.
                #
                # After Phase 1 head updates (GNN frozen), we unfreeze the
                # GNN and run n_optimizer_steps × n_minibatches gradient
                # steps using per-group HAPPO factor-weighted advantages.
                #
                # Old log-probs are re-evaluated with post-Phase-1 params
                # + SNI so the importance ratio measures GNN drift only,
                # not head drift from Phase 1.
                for p in self._shared_actor_gnn.parameters():
                    p.requires_grad_(True)

                # Build a GNN-only optimizer if not already cached
                gnn_lr = self.encoder_lr if self.encoder_lr is not None else self.experiment_config.lr
                if not hasattr(self, "_gnn_optimizer") or self._gnn_lr != gnn_lr:
                    gnn_params = list(self._shared_actor_gnn.parameters())
                    self._gnn_optimizer = torch.optim.Adam(
                        gnn_params,
                        lr=gnn_lr,
                        eps=self.experiment_config.adam_eps,
                    )
                    self._gnn_lr = gnn_lr

                train_device = experiment.config.train_device
                chunk_size = experiment.config.train_minibatch_size(self.on_policy)
                n_optimizer_steps = (
                    self.encoder_n_optimizer_steps
                    if self.encoder_n_optimizer_steps is not None
                    else experiment.config.n_optimizer_steps(self.on_policy)
                )
                n_groups = len(group_order)

                # Buffer length (same across groups)
                buf_len = None
                for g in groups:
                    glen = len(experiment.replay_buffers[g])
                    if buf_len is None:
                        buf_len = glen
                n_minibatches = max(1, -(-buf_len // chunk_size))

                # VIB beta
                beta_eff = 0.0
                if self.use_vib:
                    total_frames = experiment.total_frames
                    if self.vib_warmup_frames > 0:
                        beta_eff = min(
                            self.vib_beta,
                            self.vib_beta * total_frames / self.vib_warmup_frames,
                        )
                    else:
                        beta_eff = self.vib_beta

                # Find EmbeddingProcessors for SNI (one per group)
                _ep_map: dict[str, "EmbeddingProcessor | None"] = {}
                for g in groups:
                    _ep_map[g] = None
                    lm = experiment.losses[g]
                    for _m in lm.actor_network.modules():
                        if isinstance(_m, EmbeddingProcessor):
                            _ep_map[g] = _m
                            break

                # Pre-load full buffer data per group (on CPU)
                gbuf_full: dict[str, TensorDictBase] = {}
                for g in groups:
                    gbuf_full[g] = experiment.replay_buffers[g][:buf_len]

                # --- Compute old_log_prob with SNI (post-Phase-1 params) ---
                old_lp_sni: dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    for g in groups:
                        lm = experiment.losses[g]
                        ctx_params = (
                            lm.actor_network_params.to_module(lm.actor_network)
                            if lm.functional
                            else contextlib.nullcontext()
                        )
                        ep = _ep_map[g]
                        lp_chunks = []
                        for chunk in gbuf_full[g].chunk(
                            max(1, -(-gbuf_full[g].shape[0] // chunk_size)), dim=0
                        ):
                            chunk_dev = chunk.to(train_device)
                            sni_ctx = ep.deterministic_mode() if ep is not None else contextlib.nullcontext()
                            with sni_ctx, ctx_params:
                                dist_old = lm.actor_network.get_dist(chunk_dev)
                            action = chunk_dev.get((g, "action"))
                            lp_chunks.append(dist_old.log_prob(action).cpu())
                        old_lp_sni[g] = torch.cat(lp_chunks, dim=0)

                # Logging accumulators
                _log_obj = 0.0
                _log_encoder_loss = 0.0
                _log_vib_kl = 0.0
                _log_ratio_mean = 0.0
                _log_ratio_max = 0.0
                _log_clip_frac = 0.0
                _n_iters = 0

                clip_eps = self.clip_epsilon

                for _epoch in range(n_optimizer_steps):
                    for _mb in range(n_minibatches):
                        mb_start = _mb * chunk_size
                        mb_end = min(mb_start + chunk_size, buf_len)

                        self._gnn_optimizer.zero_grad()

                        obj_num = torch.tensor(0.0, device=train_device)
                        total_active_count = torch.tensor(0.0, device=train_device)
                        vib_kl_total = torch.tensor(0.0, device=train_device)
                        n_vib_groups = 0
                        mb_ratio_sum = 0.0
                        mb_ratio_max = 0.0
                        mb_clip_count = 0
                        mb_total_count = 0

                        for g in groups:
                            gdata_mb = gbuf_full[g][mb_start:mb_end].to(train_device)
                            old_lp_mb = old_lp_sni[g][mb_start:mb_end].to(train_device)

                            lm = experiment.losses[g]
                            ctx_params = (
                                lm.actor_network_params.to_module(lm.actor_network)
                                if lm.functional
                                else contextlib.nullcontext()
                            )
                            ep = _ep_map[g]
                            sni_ctx = ep.deterministic_mode() if ep is not None else contextlib.nullcontext()
                            with sni_ctx, ctx_params:
                                dist_new = lm.actor_network.get_dist(gdata_mb)
                            action = gdata_mb.get((g, "action"))
                            new_lp = dist_new.log_prob(action)

                            # Per-agent importance ratio
                            log_ratio = new_lp - old_lp_mb
                            ratio = torch.exp(log_ratio)
                            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

                            # Per-group advantage (S8a-normalized, D7-zeroed)
                            # with HAPPO factor already in the buffer
                            adv_mb = gdata_mb.get((g, "advantage")).detach()
                            factor_mb = gdata_mb.get(HGTeamHAPPOLoss.FACTOR_KEY, None)
                            if factor_mb is not None:
                                f = factor_mb.detach()
                                while f.dim() < adv_mb.dim():
                                    f = f.unsqueeze(-1)
                                adv_mb = adv_mb * f

                            # Align advantage dims with ratio (ratio may lack
                            # the trailing action dim that advantage has)
                            while adv_mb.dim() > ratio.dim():
                                adv_mb = adv_mb.squeeze(-1)
                            while adv_mb.dim() < ratio.dim():
                                adv_mb = adv_mb.unsqueeze(-1)

                            # Clipped PPO surrogate (pessimistic)
                            gain1 = ratio * adv_mb
                            gain2 = clipped_ratio * adv_mb
                            gain = torch.min(gain1, gain2)

                            amask = gdata_mb.get((g, "active_mask"), None)
                            if amask is not None:
                                am = amask.float()
                                while am.dim() < gain.dim():
                                    am = am.unsqueeze(-1)
                                obj_num = obj_num + (gain * am).sum()
                                total_active_count = total_active_count + am.sum()
                                with torch.no_grad():
                                    mb_ratio_sum += (ratio * am).sum().item()
                                    active_max = ratio.where(am.bool(), torch.zeros_like(ratio)).max().item()
                                    mb_ratio_max = max(mb_ratio_max, active_max)
                                    mb_clip_count += ((ratio != clipped_ratio) * am).sum().item()
                                    mb_total_count += am.sum().item()
                            else:
                                obj_num = obj_num + gain.sum()
                                total_active_count = total_active_count + torch.tensor(
                                    float(gain.numel()), device=train_device,
                                )
                                with torch.no_grad():
                                    mb_ratio_sum += ratio.sum().item()
                                    mb_ratio_max = max(mb_ratio_max, ratio.max().item())
                                    mb_clip_count += (ratio != clipped_ratio).sum().item()
                                    mb_total_count += ratio.numel()

                            # VIB KL with active_mask
                            if self.use_vib:
                                mu = gdata_mb.get((g, "embedding_mu"), None)
                                logvar = gdata_mb.get((g, "embedding_logvar"), None)
                                if mu is not None and logvar is not None:
                                    kl_per_agent = -0.5 * (
                                        1 + logvar - mu.pow(2) - logvar.exp()
                                    ).sum(dim=-1)
                                    if amask is not None:
                                        am_f = amask.float()
                                        kl = (kl_per_agent * am_f).sum() / am_f.sum().clamp(min=1)
                                    else:
                                        kl = kl_per_agent.mean()
                                    vib_kl_total = vib_kl_total + kl
                                    n_vib_groups += 1

                        obj = obj_num / total_active_count.clamp(min=1)

                        # Encoder loss = -clipped_ppo_objective + VIB
                        encoder_loss = -obj
                        if n_vib_groups > 0 and beta_eff > 0:
                            encoder_loss = encoder_loss + beta_eff * (vib_kl_total / n_vib_groups)

                        encoder_loss.backward()

                        if experiment.config.clip_grad_norm and experiment.config.clip_grad_val is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self._shared_actor_gnn.parameters(),
                                experiment.config.clip_grad_val,
                            )
                        self._gnn_optimizer.step()
                        self._gnn_optimizer.zero_grad()

                        _log_obj += obj.item()
                        _log_encoder_loss += encoder_loss.item()
                        if n_vib_groups > 0:
                            _log_vib_kl += (vib_kl_total / n_vib_groups).item()
                        if mb_total_count > 0:
                            _log_ratio_mean += mb_ratio_sum / mb_total_count
                            _log_ratio_max = max(_log_ratio_max, mb_ratio_max)
                            _log_clip_frac += mb_clip_count / mb_total_count
                        _n_iters += 1

                # --- Log separate_forward metrics ---
                if _n_iters > 0:
                    log_dict = {
                        "train/separate_forward/obj": _log_obj / _n_iters,
                        "train/separate_forward/encoder_loss": _log_encoder_loss / _n_iters,
                        "train/separate_forward/n_encoder_iters": _n_iters,
                        "train/separate_forward/ratio_mean": _log_ratio_mean / _n_iters,
                        "train/separate_forward/ratio_max": _log_ratio_max,
                        "train/separate_forward/clip_fraction": _log_clip_frac / _n_iters,
                    }
                    if self.use_vib:
                        log_dict["train/separate_forward/vib_kl_masked"] = _log_vib_kl / _n_iters
                        log_dict["train/separate_forward/vib_beta_eff"] = beta_eff
                    experiment.logger.log(log_dict, step=experiment.n_iters_performed)

            elif self.encoder_update_mode == "coop_encoder":
                # Phase 0 already ran before the head updates — nothing to do.
                pass


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
    gnn_agent_node_feature_key: str | None = MISSING
    gnn_agent_node_feature_dim: int = MISSING
    critic_use_other_actions: bool = MISSING
    gnn_norm_class: str | None = MISSING
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
    encoder_n_optimizer_steps: int | None = MISSING
    encoder_lr: float | None = MISSING

    @staticmethod
    def associated_class() -> type[Algorithm]:
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
