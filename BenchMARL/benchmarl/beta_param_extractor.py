"""Beta distribution parameter extractor for bounded [0,1] actions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaParamExtractor(nn.Module):
    """Extract alpha and beta parameters for Beta distribution.

    Takes raw logits and converts them to valid Beta distribution parameters
    using softplus to ensure both alpha > 0 and beta > 0.
    """

    def __init__(self, min_param: float = 1.0):
        """Initialize Beta parameter extractor.

        Args:
            min_param: Minimum value for alpha and beta parameters to ensure stability
        """
        super().__init__()
        self.min_param = min_param

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract alpha and beta parameters from logits.

        Args:
            logits: Raw logits tensor with shape (..., 2 * action_dim)
                   First half are raw alpha values, second half are raw beta values

        Returns:
            Tuple of (alpha, beta) parameters for Beta distribution
        """
        # Split logits into alpha and beta components
        action_dim = logits.shape[-1] // 2
        alpha_raw = logits[..., :action_dim]
        beta_raw = logits[..., action_dim:]

        # Apply softplus to ensure positive parameters
        # Add min_param to prevent parameters from being too close to 0
        alpha = F.softplus(alpha_raw) + self.min_param
        beta = F.softplus(beta_raw) + self.min_param

        return alpha, beta
