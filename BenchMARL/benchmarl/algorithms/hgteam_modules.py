"""Shared modules and utilities for HGTeam algorithm variants (PPO, SAC, HAPPO).

This module contains:
  - ``reparameterize``: Gaussian reparameterization trick (shared primitive).
  - ``EmbeddingProcessor``: GNN embedding post-processor (split-z / stochastic).
  - ``HyperNetworkJoiner``: Per-agent hypernetwork weight generation.
  - ``merge_embedding_losses``: Merge auxiliary embedding losses into a loss output.
"""

import torch
from torch import nn
from tensordict import TensorDictBase


# ======================================================================
# Shared primitive
# ======================================================================

def reparameterize(
    embedding: torch.Tensor,
    log_var_min: float = -10.0,
    log_var_max: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split embedding into (mean, log_var), clamp, and sample via reparameterization.

    Args:
        embedding: ``(..., 2*latent_dim)`` tensor whose last-dim halves are
            ``[mean | log_var]``.
        log_var_min: Lower clamp for log-variance.
        log_var_max: Upper clamp for log-variance.

    Returns:
        ``(z, mean, log_var)`` where ``z = mean + std * eps``.
    """
    mean, log_var = embedding.chunk(2, dim=-1)
    log_var = torch.clamp(log_var, min=log_var_min, max=log_var_max)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mean + eps * std
    return z, mean, log_var


# ======================================================================
# EmbeddingProcessor
# ======================================================================

class EmbeddingProcessor(nn.Module):
    """Process GNN embeddings — handles optional split-z and stochastic sampling.

    When ``split_z=False`` (default / legacy), the GNN output is treated as a
    single embedding that serves as both token and query in the Transformer.
    Stochastic sampling is controlled by the ``stochastic`` flag.

    When ``split_z=True``, the GNN output is split into two independent
    vectors:
      * **z_token** (always deterministic) — prepended to the Transformer KV
        sequence so self-attention can attend to structural context.
      * **z_query** (optionally stochastic via ``stochastic_query``) — used as
        the cross-attention query vector.

    This eliminates gradient interference between the two roles and doubles
    the representational budget without doubling GNN backbone parameters.
    """

    def __init__(
        self,
        embedding_dim: int,
        stochastic: bool = False,
        split_z: bool = False,
        z_token_dim: int = 32,
        z_query_dim: int = 32,
        stochastic_query: bool = True,
    ):
        super().__init__()
        self.split_z = split_z
        self.stochastic = stochastic
        self.stochastic_query = stochastic_query
        self.z_token_dim = z_token_dim
        self.z_query_dim = z_query_dim

        if split_z:
            # GNN output layout: [z_token | z_query_mean (| z_query_logvar)]
            self.latent_dim = z_query_dim  # effective query latent dim
        elif stochastic:
            self.latent_dim = embedding_dim // 2
        else:
            self.latent_dim = embedding_dim

    def forward(self, embedding: torch.Tensor):
        # embedding: (..., n_agents, embedding_dim)
        if self.split_z:
            return self._forward_split(embedding)
        return self._forward_legacy(embedding)

    # -- Legacy (single-z) path ------------------------------------------
    def _forward_legacy(self, embedding: torch.Tensor):
        """Returns ``(z, mean, log_var)``."""
        if self.stochastic:
            z, mean, log_var = reparameterize(embedding)
        else:
            z = embedding
            mean = embedding  # Deterministic: mean is the value itself
            log_var = None
        return z, mean, log_var

    # -- Split-z path ----------------------------------------------------
    def _forward_split(self, embedding: torch.Tensor):
        """Returns ``(z_token, z_query, query_mean, query_logvar)``.

        * ``z_token`` is always deterministic (first ``z_token_dim`` dims).
        * ``z_query`` is reparameterized-sampled when ``stochastic_query``.
        * ``query_mean`` is the pre-noise mean (needed for VIB KL computation).
        """
        z_token = embedding[..., :self.z_token_dim]
        remainder = embedding[..., self.z_token_dim:]

        if self.stochastic_query:
            z_query, mean, log_var = reparameterize(remainder)
        else:
            z_query = remainder
            mean = remainder  # Deterministic: mean is the value itself
            log_var = None

        return z_token, z_query, mean, log_var


# ======================================================================
# HyperNetworkJoiner
# ======================================================================

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
        # Returns: (logits, z, log_var)

        if self.stochastic_embedding:
            z, mean, log_var = reparameterize(embedding)
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


# ======================================================================
# Loss helper
# ======================================================================

def merge_embedding_losses(
    algorithm,
    group: str,
    tensordict: TensorDictBase,
    out: TensorDictBase,
    target_loss_key: str,
) -> TensorDictBase:
    """Compute embedding auxiliary losses and merge them into *out*.

    Calls ``algorithm._compute_embedding_losses(group, tensordict)`` then
    adds every ``loss_*`` entry to *target_loss_key* (e.g. ``"loss_objective"``
    for PPO or ``"loss_actor"`` for SAC).

    Returns *out* for convenience.
    """
    embedding_losses = algorithm._compute_embedding_losses(group, tensordict)

    if not embedding_losses and algorithm.gnn_mode != "none":
        print(
            f"[DEBUG] No embedding losses computed for group '{group}' "
            f"with gnn_mode '{algorithm.gnn_mode}'."
        )

    for k, v in embedding_losses.items():
        out.set(k, v)
        if k.startswith("loss_") and target_loss_key in out.keys():
            out.set(target_loss_key, out.get(target_loss_key) + v)

    return out
