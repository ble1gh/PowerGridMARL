#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import math
from collections.abc import Iterable
from dataclasses import MISSING, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch.distributions import Normal
from torchrl.modules import (
    IndependentNormal,
    ProbabilisticActor,
)
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig

################################################################################
# MAT Model Implementation
# Adapted from https://github.com/PKU-MARL/Multi-Agent-Transformer
################################################################################


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))

    def forward(self, key, value, query, mask=None):
        # Handle arbitrary leading batch dimensions: (..., T, C)
        leading_shape = query.shape[:-2]
        T = query.shape[-2]
        C = query.shape[-1]
        B = 1
        for s in leading_shape:
            B *= s

        # Flatten leading dims
        query = query.reshape(B, T, C)
        key = key.reshape(B, key.shape[-2], C)
        value = value.reshape(B, value.shape[-2], C)

        k = self.key(key).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal attention mask for decoder
        if self.masked:
            mask_shape = (T, T)
            causal_mask = torch.tril(torch.ones(mask_shape, device=query.device)).view(1, 1, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(causal_mask == 0, float("-inf"))
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Mask padding (inactive agents)
        if mask is not None:
            mask = mask.reshape(B, 1, 1, -1)
            att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        # Unflatten leading dims
        return y.reshape(*leading_shape, T, C)


class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 4 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(4 * n_embd, n_embd)),
        )
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, x, x, mask))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 4 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(4 * n_embd, n_embd)),
        )
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)  # Causal self-attn
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=False)  # Cross-attn

    def forward(self, x, rep, mask=None):
        # Causal Self Attention
        x = self.ln1(x + self.attn1(x, x, x))
        # Cross Attention (query=x, key=rep, value=rep)
        x = self.ln2(x + self.attn2(key=rep, value=rep, query=x, mask=mask))
        # MLP
        x = self.ln3(x + self.mlp(x))
        return x


class MultiAgentTransformer(nn.Module):
    def __init__(
        self,
        n_agent,
        obs_dim,
        action_dim,
        n_block,
        n_embd,
        n_head,
        action_type="continuous",
        device="cpu",
    ):
        super().__init__()
        self.n_agent = n_agent
        self.action_type = action_type
        self.device = device

        # Encoder
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim), init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU()
        )
        self.encoder_blocks = nn.ModuleList(
            [EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)]
        )

        # Decoder
        self.action_encoder = nn.Sequential(
            init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU()
        )
        self.decoder_blocks = nn.ModuleList(
            [DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)]
        )

        # Heads
        self.action_head = init_(nn.Linear(n_embd, action_dim))
        self.value_head = init_(nn.Linear(n_embd, 1))

        if action_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_rep(self, obs, mask=None):
        # obs: (..., N, F) - handle arbitrary leading batch dims
        leading_shape = obs.shape[:-2]
        N, F = obs.shape[-2], obs.shape[-1]
        obs_flat = obs.reshape(-1, N, F)
        mask_flat = mask.reshape(-1, mask.shape[-1]) if mask is not None else None

        rep = self.obs_encoder(obs_flat)
        for block in self.encoder_blocks:
            rep = block(rep, mask_flat)
        # Unflatten: (..., N, D)
        return rep.reshape(*leading_shape, N, rep.shape[-1])

    def get_value(self, obs, mask=None):
        rep = self.get_rep(obs, mask)
        val = self.value_head(rep)
        return val

    def get_logits(self, rep, action, mask=None):
        # rep: (..., N, D), action: (..., N, A)
        leading_shape = action.shape[:-2]
        N, A = action.shape[-2], action.shape[-1]
        D = rep.shape[-1]

        rep_flat = rep.reshape(-1, N, D)
        action_flat = action.reshape(-1, N, A)
        mask_flat = mask.reshape(-1, mask.shape[-1]) if mask is not None else None

        action_emb = self.action_encoder(action_flat)  # (-1, N, D)

        # Shift logic: Input to decoder at step t is action at t-1.
        # Step 0 gets zeros.
        shifted_action_emb = torch.zeros_like(action_emb)
        shifted_action_emb[:, 1:, :] = action_emb[:, :-1, :]

        x = shifted_action_emb + rep_flat

        for block in self.decoder_blocks:
            x = block(x, rep_flat, mask_flat)

        logits = self.action_head(x)
        # Unflatten: (..., N, A)
        return logits.reshape(*leading_shape, N, logits.shape[-1])


################################################################################
# BenchMARL Integration
################################################################################


class MATLoss(ClipPPOLoss):
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
        # We need to ensure the shared MAT model has parameters available for optimization
        # BenchMARL/TorchRL might use functional calls, but MAT model is stateful nn.Module in our wrapper
        # The params are in algorithm.mat_model
        return super().forward(tensordict)


class MAT(Algorithm):
    """Multi-Agent Transformer (MAT) Algorithm.
    Reference: https://arxiv.org/abs/2205.14953
    """

    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: float,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        n_block: int,
        n_embd: int,
        n_head: int,
        minibatch_advantage: bool = False,
        **kwargs,
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
        self.n_block = n_block
        self.n_embd = n_embd
        self.n_head = n_head
        self.minibatch_advantage = minibatch_advantage

        self.mat_models = {}  # Per-group MAT models

    def _get_mat_model(self, group: str):
        if group in self.mat_models:
            return self.mat_models[group]

        n_agent = len(self.group_map[group])
        obs_dim = self.observation_spec[group, "observation"].shape[-1]
        action_dim = self.action_spec[group, "action"].shape[-1]

        model = MultiAgentTransformer(
            n_agent=n_agent,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_block=self.n_block,
            n_embd=self.n_embd,
            n_head=self.n_head,
            action_type="continuous",  # BenchMARL mostly continuous for now
            device=self.device,
        ).to(self.device)

        self.mat_models[group] = model
        return model

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> tuple[LossModule, bool]:
        loss_module = MATLoss(
            algorithm=self,
            group=group,
            actor=policy_for_loss,
            critic=self.get_critic(group),  # This shares weights with actor in our impl
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
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> dict[str, Iterable]:
        return {
            "loss_objective": list(self._get_mat_model(group).parameters()),
            "loss_critic": list(self._get_mat_model(group).parameters()),  # Shared parameters
        }

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        # Training Policy (Parallel Teacher Forcing)
        mat_model = self._get_mat_model(group)

        # Wrapper to handle input keys and expected output for ProbabilisticActor
        class ParallelActorWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model  # Register as submodule to handle .to(device)

            def forward(self, obs, action, mask=None):
                # obs: (B, N, F), action: (B, N, A)
                rep = self.model.get_rep(obs, mask)
                mean = self.model.get_logits(rep, action, mask)

                # Return loc and scale
                # std = sigmoid(log_std) * 0.5
                std = torch.sigmoid(self.model.log_std) * 0.5
                std = std.expand_as(mean)
                return mean, std

        actor_module = TensorDictModule(
            ParallelActorWrapper(mat_model),
            in_keys=[
                (group, "observation"),
                (group, "action"),
            ],  # Takes ACTION as input (Teacher Forcing)
            out_keys=[(group, "loc"), (group, "scale")],
        )

        policy = ProbabilisticActor(
            module=actor_module,
            spec=self.action_spec[group, "action"],
            in_keys=[(group, "loc"), (group, "scale")],
            out_keys=[(group, "action")],
            distribution_class=IndependentNormal,  # TanhNormal support could be added
            return_log_prob=True,
            log_prob_key=(group, "log_prob"),
        )
        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        # Collection Policy (Autoregressive Generation)
        mat_model = self._get_mat_model(group)
        n_agents = len(self.group_map[group])
        action_dim = self.action_spec[group, "action"].shape[-1]

        class AutoRegressiveWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model  # Register as submodule to handle .to(device)

            def forward(self, obs, mask=None):
                # obs: (B, N, F)
                B = obs.shape[0]
                rep = self.model.get_rep(obs, mask)
                actions = torch.zeros(B, n_agents, action_dim, device=obs.device)
                log_probs = torch.zeros(B, n_agents, device=obs.device)

                for i in range(n_agents):
                    mean_full = self.model.get_logits(rep, actions, mask)
                    mean_i = mean_full[:, i, :]
                    std = torch.sigmoid(self.model.log_std) * 0.5
                    dist = Normal(mean_i, std)
                    action_i = dist.sample()

                    # Store
                    actions[:, i, :] = action_i

                    # Sum log probs over action dim (shape: (B,) per agent)
                    log_probs[:, i] = dist.log_prob(action_i).sum(dim=-1)

                return actions, log_probs

        collection_policy = TensorDictModule(
            AutoRegressiveWrapper(mat_model),
            in_keys=[(group, "observation")],
            out_keys=[(group, "action"), (group, "log_prob")],
        )

        return collection_policy

    def get_critic(self, group: str) -> TensorDictModule:
        mat_model = self._get_mat_model(group)

        class CriticWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, obs, mask=None):
                return self.model.get_value(obs, mask)

        return TensorDictModule(
            CriticWrapper(mat_model),
            in_keys=[(group, "observation")],  # Critic only needs obs
            out_keys=[(group, "state_value")],
        )

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        # Ensure batch is on the correct device for GAE computation
        batch = batch.to(self.device)

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

    def process_loss_vals(self, group: str, loss_vals: TensorDictBase) -> TensorDictBase:
        loss_vals.set("loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"])
        del loss_vals["loss_entropy"]
        return loss_vals


@dataclass
class MATConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.MAT`."""

    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING

    n_block: int = MISSING
    n_embd: int = MISSING
    n_head: int = MISSING
    minibatch_advantage: bool = MISSING

    @staticmethod
    def associated_class() -> type[Algorithm]:
        return MAT

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False  # Implemented continuous only for this task

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True  # MAT has joint encoder, effectively centralized
