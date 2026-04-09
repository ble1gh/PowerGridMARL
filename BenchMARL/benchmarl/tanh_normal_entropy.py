"""TanhNormal distribution with semi-analytical entropy computation.

The standard TorchRL TanhNormal raises NotImplementedError for entropy(),
causing PPO to fall back to a single-sample Monte Carlo estimate which is
very noisy.  This subclass provides a much more accurate estimate by
computing the Normal entropy analytically and using a multi-sample MC
estimate only for the tanh Jacobian correction term.
"""

import math

import torch
from torchrl.modules.distributions import TanhNormal

_NORMAL_ENTROPY_CONST = 0.5 * (1.0 + math.log(2.0 * math.pi))


class TanhNormalWithEntropy(TanhNormal):
    """TanhNormal with semi-analytical entropy.

    H(Y) = H_Normal(sigma)
         + E_X[ sum_i log(1 - tanh^2(X_i)) ]   (MC, 16 samples)
         + sum_i log(range_i / 2)                (if affine transform present)

    The Normal entropy is exact; only the tanh Jacobian correction is
    estimated via Monte Carlo, which has much lower variance than
    estimating the full entropy from a single log_prob sample.
    """

    _entropy_mc_samples: int = 16

    def entropy(self) -> torch.Tensor:
        # --- 1. Analytical Normal entropy: 0.5*(1 + ln 2pi) + ln(sigma) per dim
        scale = self.scale  # (..., action_dim)
        normal_entropy = _NORMAL_ENTROPY_CONST + scale.log()  # per-dim
        normal_entropy = normal_entropy.sum(-1)  # sum over action dims

        # --- 2. MC estimate of tanh Jacobian: E[sum log(1 - tanh^2(x))]
        #     Sample from the *base* Normal (before any transforms).
        loc = self.loc
        x = loc + scale * torch.randn(
            (self._entropy_mc_samples,) + scale.shape,
            device=scale.device,
            dtype=scale.dtype,
        )
        # log(1 - tanh^2(x)) = 2*log(sech(x)) = 2*(log2 - x - softplus(-2x))
        # Use the numerically stable form: log(1 - tanh^2(x)) = -2*|x| - 2*softplus(-2*|x|) + 2*ln(2)
        # Simplest stable form via torch: log(1 - tanh(x)^2)
        # But for large |x|, tanh(x)^2 ≈ 1 and 1-tanh^2 underflows.
        # Stable: log(1-tanh^2(x)) = log(4) - 2x - 2*softplus(-2x)  [for any x]
        #       = 2*ln2 - 2*softplus(2x) + 2x - 2*softplus(-2x) + 2x
        # Actually simplest stable: log(4*exp(-2|x|)/(1+exp(-2|x|))^2)
        # = log4 - 2|x| - 2*log(1+exp(-2|x|))
        # Use: log(1 - tanh^2(x)) = 2*log(2) - 2*abs(x) - 2*torch.nn.functional.softplus(-2*abs(x))
        abs_x = x.abs()
        log_dtanh = (
            2.0 * math.log(2.0) - 2.0 * abs_x - 2.0 * torch.nn.functional.softplus(-2.0 * abs_x)
        )
        # Sum over action dims, mean over MC samples
        tanh_correction = log_dtanh.sum(-1).mean(0)

        # --- 3. Affine correction: log(range/2) per dim (if non-trivial bounds)
        affine_correction = torch.zeros_like(normal_entropy)
        if self.non_trivial_max or self.non_trivial_min:
            action_range = self.high - self.low  # (..., action_dim)
            affine_correction = (action_range / 2.0).log().sum(-1)

        return normal_entropy + tanh_correction + affine_correction
