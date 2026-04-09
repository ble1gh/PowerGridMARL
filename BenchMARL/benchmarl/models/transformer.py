#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import MISSING, dataclass

import torch
from tensordict import TensorDictBase
from tensordict.utils import unravel_key_list
from torch import nn
from torchrl.data import Composite, Unbounded
from torchrl.modules import MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self._build_pe(max_len)

    def _build_pe(self, max_len: int):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * (-torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        pe = torch.zeros(max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        # Dynamically extend PE buffer if the sequence is longer than expected
        if seq_len > self.pe.size(0):
            self._build_pe(seq_len * 2)
            self.pe = self.pe.to(x.device)
        pe = self.pe[:seq_len].transpose(0, 1).to(x.device)
        x = x + pe
        return self.dropout(x)


class Transformer(Model):
    """Multi-head transformer that consumes growing observation/action tokens.

    The model keeps a rolling token history so inference can be recurrent. During
    training (no history provided) it processes full sequences. Optionally, the
    query vectors for attention pooling can be taken from a GNN embedding ``z``.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        use_z_as_query: bool,
        append_actions: bool,
        norm_first: bool,
        prepend_z_token: bool = True,
        **kwargs,
    ):
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.history_name = (
            self.agent_group,
            f"_hidden_transformer_history_{self.model_index}",
        )
        self.history_len_name = (
            self.agent_group,
            f"_hidden_transformer_len_{self.model_index}",
        )
        self.rnn_keys = unravel_key_list(["is_init", self.history_name, self.history_len_name])
        self.in_keys += self.rnn_keys

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_z_as_query = use_z_as_query
        self.append_actions = append_actions
        self.norm_first = norm_first
        self.prepend_z_token = prepend_z_token
        self.z_token_proj = None  # lazily initialized on first use once z dim is known

        self.input_features = sum(
            spec.shape[-1]
            for key, spec in self.input_spec.items(True, True)
            if (key if isinstance(key, str) else key[-1]) != "active_mask"
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        self.obs_proj = nn.Linear(self.input_features, self.d_model, device=self.device)

        self.action_dim = self._infer_action_dim()
        if self.append_actions and self.action_dim > 0:
            self.action_proj = nn.Linear(self.action_dim, self.d_model, device=self.device)
        else:
            self.action_proj = None
            self.append_actions = False

        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
            max_len=self.max_seq_len * 2,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            norm_first=self.norm_first,
            device=self.device,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        if self.use_z_as_query:
            self.query_proj = None  # will be instantiated on first use once z dim is known
            self.query_attn = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.nhead,
                dropout=self.dropout,
                batch_first=True,
                device=self.device,
            )
        else:
            self.query_proj = None
            self.query_attn = None

        if self.input_has_agent_dim:
            self.head = MultiAgentMLP(
                n_agent_inputs=self.d_model,
                n_agent_outputs=self.output_features,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                num_cells=[],
                activation_class=nn.Identity,
                layer_class=nn.Linear,
            )
        else:
            self.head = nn.ModuleList(
                [
                    MLP(
                        in_features=self.d_model,
                        out_features=self.output_features,
                        device=self.device,
                        num_cells=[],
                        activation_class=nn.Identity,
                        layer_class=nn.Linear,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _infer_action_dim(self) -> int:
        if self.action_spec is None:
            return 0
        spec = self.action_spec.get((self.agent_group, "action"), None)
        if spec is None:
            return 0
        if hasattr(spec, "space") and hasattr(spec.space, "n"):
            return int(spec.space.n)
        if len(spec.shape) == 0:
            return 1
        return int(spec.shape[-1])

    def _gather_inputs(self, tensordict: TensorDictBase) -> torch.Tensor:
        return torch.cat(
            [
                tensordict.get(in_key)
                for in_key in self.in_keys
                if in_key not in self.rnn_keys
                and (in_key if isinstance(in_key, str) else in_key[-1])
                != "active_mask"
            ],
            dim=-1,
        )

    def _build_token_sequence(
        self,
        obs_emb: torch.Tensor,
        action: torch.Tensor | None,
        z_emb: torch.Tensor | None = None,
    ):
        """Build interleaved token sequence from obs, action, and optional z embeddings.

        When *z_emb* is provided (and ``self.prepend_z_token`` is True), the
        per-timestep token order is ``[z, obs, act]`` (3 tokens/step) or
        ``[z, obs]`` (2 tokens/step if no actions).  This places the GNN
        embedding directly into the KV sequence so self-attention and
        cross-attention can attend to it, fixing the near-vanishing gradient
        problem for agents whose observations are low-variance (e.g. Storage
        SoC).

        Returns:
            tokens: (b, total_tokens, a, d_model)
            obs_positions: 1-D LongTensor with indices of obs tokens in the
                sequence (used by cross-attention to identify which encoded
                vectors correspond to observations).
        """
        has_z = z_emb is not None
        has_act = action is not None and self.action_proj is not None

        if not has_act and not has_z:
            # Original: obs-only tokens
            tokens = obs_emb
            obs_positions = torch.arange(
                tokens.shape[1], device=tokens.device if tokens.dim() > 1 else None
            )
            return tokens, obs_positions

        # --- Promote action_emb to 4-D (b, t, a, d) if present -------------
        action_emb = None
        if has_act:
            action_emb = self.action_proj(action.to(obs_emb.dtype))
            if action_emb.dim() == 2 and obs_emb.dim() == 4:
                action_emb = action_emb.unsqueeze(0).unsqueeze(0)
            elif action_emb.dim() == 3 and obs_emb.dim() == 4:
                action_emb = action_emb.unsqueeze(1)
            if obs_emb.dim() == 3:
                obs_emb = obs_emb.unsqueeze(1)
                action_emb = action_emb.unsqueeze(1)

        # --- Promote z_emb to 4-D (b, t, a, d) if present ------------------
        if has_z:
            if obs_emb.dim() == 3:
                obs_emb = obs_emb.unsqueeze(1)
            if z_emb.dim() == 3:
                z_emb = z_emb.unsqueeze(1)  # (b, a, d) -> (b, 1, a, d)
            # Broadcast z along time dim to match obs_emb if needed
            if z_emb.shape[1] == 1 and obs_emb.shape[1] > 1:
                z_emb = z_emb.expand(-1, obs_emb.shape[1], -1, -1)

        b, t, a, _ = obs_emb.shape

        # --- Stack tokens per timestep ------------------------------------
        if has_z and has_act:
            # 3 tokens/step: [z, obs, act]
            stacked = torch.stack([z_emb, obs_emb, action_emb], dim=-2)  # (b, t, a, 3, d)
            tokens_per_step = 3
            obs_offset = 1  # obs is at position 1 within each triplet
        elif has_z and not has_act:
            # 2 tokens/step: [z, obs]
            stacked = torch.stack([z_emb, obs_emb], dim=-2)  # (b, t, a, 2, d)
            tokens_per_step = 2
            obs_offset = 1
        else:
            # 2 tokens/step: [obs, act]  (original behavior, no z)
            stacked = torch.stack([obs_emb, action_emb], dim=-2)  # (b, t, a, 2, d)
            tokens_per_step = 2
            obs_offset = 0

        tokens = stacked.reshape(b, t * tokens_per_step, a, self.d_model)
        obs_positions = torch.arange(
            obs_offset, t * tokens_per_step, tokens_per_step, device=obs_emb.device
        )
        return tokens, obs_positions

    def _reshape_for_encoder(self, tokens: torch.Tensor):
        if tokens.dim() == 3:
            tokens = tokens.unsqueeze(1)
        tokens = tokens.transpose(1, 2)  # (b, a, t, d)
        b, a, t, d = tokens.shape
        tokens = tokens.reshape(b * a, t, d)
        return tokens, b, a, t

    def _apply_encoder(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        causal_mask: torch.Tensor | None,
    ):
        tokens = self.pos_encoder(tokens)
        encoded = self.encoder(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return encoded

    def _pool_with_query(
        self,
        encoded: torch.Tensor,
        query: torch.Tensor,
        b: int,
        a: int,
        t: int,
        obs_positions: torch.Tensor,
    ):
        encoded = encoded.reshape(b, a, t, self.d_model)
        encoded_obs = encoded[:, :, obs_positions, :]  # (b, a, seq, d)
        if self.use_z_as_query and query is not None:
            query_proj = self.query_proj(query)
            if query_proj.dim() == 3:
                query_proj = query_proj.unsqueeze(2)
            query_proj = query_proj.transpose(1, 2)  # (b, a, seq?, d)
            q = query_proj.reshape(b * a, -1, self.d_model)
            kv = encoded.reshape(b * a, t, self.d_model)
            attn_out, _ = self.query_attn(query=q, key=kv, value=kv)
            pooled = attn_out.reshape(b, a, -1, self.d_model)
            return pooled
        return encoded_obs

    def _project_output(self, features: torch.Tensor, training: bool):
        if self.input_has_agent_dim:
            if training:
                output = self.head.forward(features)
            else:
                output = self.head.forward(features)
                if not self.output_has_agent_dim:
                    output = output[..., 0, :]
        else:
            if training:
                if not self.share_params:
                    output = torch.stack(
                        [net(features) for net in self.head],
                        dim=-2,
                    )
                else:
                    output = self.head[0](features)
            else:
                if not self.share_params:
                    output = torch.stack(
                        [net(features) for net in self.head],
                        dim=-2,
                    )
                else:
                    output = self.head[0](features)
        return output

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = self._gather_inputs(tensordict)
        action_td = tensordict.get((self.agent_group, "action"), None)

        # --- Resolve z vectors for token prepend & query ----------------
        # Split-z mode: two separate vectors from the EmbeddingProcessor
        z_token_vec = tensordict.get((self.agent_group, "embedding_z_token"), None)
        z_query_vec = tensordict.get((self.agent_group, "embedding_z"), None)
        # Legacy mode: single z serves both roles
        if z_token_vec is None:
            # No split — fall back to single embedding_z for both
            z_for_token = z_query_vec  # may be None (no GNN)
            z_for_query = z_query_vec
        else:
            z_for_token = z_token_vec
            z_for_query = z_query_vec  # z_query (alias of embedding_z)

        is_init = tensordict.get("is_init", None)
        history = tensordict.get(self.history_name, None)
        history_len = tensordict.get(self.history_len_name, None)

        training = history is None or history_len is None

        obs_emb = self.obs_proj(obs)
        if self.input_has_agent_dim and obs_emb.dim() == 3:
            obs_emb = obs_emb.unsqueeze(1)

        # Lazily initialize z_token_proj on first use (shared by train & eval)
        if self.prepend_z_token and z_for_token is not None and self.z_token_proj is None:
            self.z_token_proj = nn.Linear(
                z_for_token.shape[-1], self.d_model, device=z_for_token.device
            )

        if training:
            added_batch_dim = False
            # --- Project z_for_token to a d_model token -----------------
            z_emb = None
            if self.prepend_z_token and z_for_token is not None:
                z_emb = self.z_token_proj(z_for_token)
            tokens, obs_positions = self._build_token_sequence(obs_emb, action_td, z_emb=z_emb)
            tokens_flat, b, a, t = self._reshape_for_encoder(tokens)
            token_len = tokens_flat.shape[1]
            causal_mask = torch.triu(
                torch.ones(token_len, token_len, device=tokens.device, dtype=torch.bool),
                1,
            )
            key_padding_mask = None
            encoded = self._apply_encoder(tokens_flat, key_padding_mask, causal_mask)
            encoded = encoded.reshape(b, a, token_len, self.d_model)

            if self.use_z_as_query and z_for_query is not None:
                if self.query_proj is None:
                    self.query_proj = nn.Linear(
                        z_for_query.shape[-1], self.d_model, device=z_for_query.device
                    )
                # If z_for_query has a time dimension (b, t, a, z_dim), transpose
                z_q = z_for_query
                if z_q.dim() == 4:
                    z_q = z_q.transpose(1, 2)

                query_proj = self.query_proj(z_q)  # (b, a?, [t_obs,] d)
                if query_proj.dim() == 2:
                    query_proj = query_proj.unsqueeze(1).unsqueeze(2).expand(-1, a, 1, -1)
                elif query_proj.dim() == 3:
                    query_proj = query_proj.unsqueeze(2)
                # query_proj is now (b, a, t_q, d)
                t_q = query_proj.shape[2]
                query_flat = query_proj.reshape(b * a, t_q, self.d_model)
                kv_flat = encoded.reshape(b * a, token_len, self.d_model)

                # Causal mask for cross-attention when multiple query timesteps
                cross_mask = None
                if t_q > 1:
                    cross_mask = torch.triu(
                        torch.ones(t_q, token_len, device=tokens.device, dtype=torch.bool), 1
                    )

                attn_out, _ = self.query_attn(
                    query=query_flat, key=kv_flat, value=kv_flat, attn_mask=cross_mask
                )
                pooled = attn_out.reshape(b, a, t_q, self.d_model)
                if t_q == 1:
                    pooled = pooled.squeeze(2)  # (b, a, d)
                else:
                    pooled = pooled.transpose(1, 2)  # (b, t_obs, a, d)
            else:
                t_obs = len(obs_positions)
                encoded_obs = encoded[:, :, obs_positions, :]  # (b, a, t_obs, d)
                if t_obs == 1:
                    pooled = encoded_obs.squeeze(2)  # (b, a, d)
                else:
                    pooled = encoded_obs.transpose(1, 2)  # (b, t_obs, a, d)

            output = self._project_output(pooled, training=True)
            tensordict.set(("next", *self.history_name), None)
            tensordict.set(("next", *self.history_len_name), None)
        else:
            added_batch_dim = False
            if obs_emb.dim() == 2:
                obs_emb = obs_emb.unsqueeze(0)
                added_batch_dim = True
            # Keep z vectors in sync: if we added a batch dim to obs_emb,
            # also add one to z so the agent dim is not mistaken for batch.
            if added_batch_dim:
                if z_for_token is not None and z_for_token.dim() == 2:
                    z_for_token = z_for_token.unsqueeze(0)
                if z_for_query is not None and z_for_query.dim() == 2:
                    z_for_query = z_for_query.unsqueeze(0)
            # Project z_for_token for eval path (after batch-dim adjustment)
            z_emb_eval = None
            if self.prepend_z_token and z_for_token is not None:
                z_emb_eval = self.z_token_proj(z_for_token)
            # Ensure obs_emb is 4D (b, t, a, d) before building token sequence
            if self.input_has_agent_dim and obs_emb.dim() == 3:
                obs_emb = obs_emb.unsqueeze(1)  # (b, 1, a, d)
            tokens, obs_positions = self._build_token_sequence(obs_emb, action_td, z_emb=z_emb_eval)
            tokens = tokens.transpose(1, 2)  # (b, a, t, d)
            if is_init is not None:
                reset_mask = is_init.to(torch.bool).view(tokens.shape[0], 1, 1, 1)
                if history is not None:
                    history = torch.where(reset_mask, 0, history)
                if history_len is not None:
                    history_len = torch.where(reset_mask.view(tokens.shape[0], 1), 0, history_len)
            new_tokens = tokens
            if new_tokens.dim() == 3:
                new_tokens = new_tokens.unsqueeze(2)
            b, a, t_new, _ = new_tokens.shape
            if history is None:
                history = torch.zeros(
                    (b, self.n_agents, self.max_seq_len, self.d_model),
                    device=self.device,
                )
                history_len = torch.zeros((b, self.n_agents), device=self.device, dtype=torch.long)
            history = history.to(new_tokens.device)
            history_len = history_len.to(new_tokens.device).long()

            combined = torch.cat([history, new_tokens], dim=2)
            token_count = combined.shape[2]
            total_len = torch.clamp(history_len + t_new, max=self.max_seq_len)
            if token_count > self.max_seq_len:
                combined = combined[:, :, -self.max_seq_len :, :]
                token_count = combined.shape[2]

            mask_positions = torch.arange(token_count, device=new_tokens.device)
            key_padding_mask = mask_positions.unsqueeze(0).unsqueeze(0) >= total_len.unsqueeze(-1)
            tokens_flat = combined.transpose(1, 2).reshape(b * a, token_count, self.d_model)
            causal_mask = torch.triu(
                torch.ones(token_count, token_count, device=new_tokens.device, dtype=torch.bool),
                1,
            )
            encoded = self._apply_encoder(
                tokens_flat,
                key_padding_mask=key_padding_mask.reshape(b * a, token_count),
                causal_mask=causal_mask,
            )
            encoded = encoded.reshape(b, a, token_count, self.d_model)
            obs_idx = token_count - t_new + int(obs_positions[-1])
            obs_idx = max(0, min(token_count - 1, obs_idx))
            if self.use_z_as_query and z_for_query is not None:
                if self.query_proj is None:
                    self.query_proj = nn.Linear(
                        z_for_query.shape[-1], self.d_model, device=z_for_query.device
                    )
                query_proj = self.query_proj(z_for_query)
                if query_proj.dim() == 2:
                    query_proj = query_proj.unsqueeze(1).unsqueeze(2).expand(-1, a, 1, -1)
                elif query_proj.dim() == 3:
                    query_proj = query_proj.unsqueeze(2)
                query_proj = query_proj.transpose(1, 2).reshape(b * a, -1, self.d_model)
                kv = encoded.reshape(b * a, token_count, self.d_model)
                attn_out, _ = self.query_attn(query=query_proj, key=kv, value=kv)
                pooled = attn_out.reshape(b, a, -1, self.d_model).squeeze(2)
            else:
                pooled = encoded[:, :, obs_idx, :]
            output = self._project_output(pooled, training=False)
            if output.dim() == 4 and output.shape[1] == 1:
                output = output.squeeze(1)
            history = combined
            history_len = total_len

            # Strip the synthetic batch dim we added so shapes match the
            # tensordict's batch_size (agent-only, no leading batch dim).
            if added_batch_dim:
                if history.shape[0] == 1:
                    history = history.squeeze(0)
                if history_len.shape[0] == 1:
                    history_len = history_len.squeeze(0)

            tensordict.set(("next", *self.history_name), history.detach())
            tensordict.set(("next", *self.history_len_name), history_len.detach())

        expected_batch = tensordict.batch_size
        # The eval path inserts a synthetic batch dim of 1 when obs_emb is
        # unbatched (2-D).  Strip it so the output matches the tensordict.
        if not training and added_batch_dim and output.shape[0] == 1:
            output = output.squeeze(0)

        tensordict.set(self.out_key, output)
        return tensordict

    def _perform_checks(self):
        super()._perform_checks()
        if self.input_has_agent_dim:
            if self.output_has_agent_dim and self.output_leaf_spec.shape[-2] != self.n_agents:
                raise ValueError(
                    "Transformer output with agent dimension expects last but one dimension to match n_agents"
                )
        else:
            if not self.share_params:
                raise ValueError(
                    "Transformer without agent dimension currently expects shared parameters"
                )

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        spec = Composite(
            {
                f"_hidden_transformer_history_{model_index}": Unbounded(
                    shape=(self.max_seq_len, self.d_model)
                ),
                f"_hidden_transformer_len_{model_index}": Unbounded(shape=(), dtype=torch.long),
            }
        )
        return spec


@dataclass
class TransformerConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.transformer.Transformer`."""

    d_model: int = MISSING
    nhead: int = MISSING
    num_layers: int = MISSING
    dim_feedforward: int = MISSING
    dropout: float = 0.0
    max_seq_len: int = 32
    use_z_as_query: bool = True
    append_actions: bool = True
    norm_first: bool = True
    prepend_z_token: bool = True

    @staticmethod
    def associated_class():
        return Transformer

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        spec = Composite(
            {
                f"_hidden_transformer_history_{model_index}": Unbounded(
                    shape=(self.max_seq_len, self.d_model)
                ),
                f"_hidden_transformer_len_{model_index}": Unbounded(shape=(), dtype=torch.long),
            }
        )
        return spec
