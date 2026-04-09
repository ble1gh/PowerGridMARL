"""Independent Beta distribution with optional rescaling to arbitrary [low, high] bounds."""

import torch
from torch.distributions import Beta, Independent


class IndependentBeta(Independent):
    """Independent Beta distribution with optional affine rescaling.

    Samples natively in [0, 1] via Beta(alpha, beta), then rescales to
    [low, high] if bounds are provided.  When low=0 and high=1 (or bounds
    are omitted) the behaviour is identical to a plain Beta distribution.

    The log_prob accounts for the change-of-variable Jacobian so that
    gradient estimates remain correct.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        low: torch.Tensor | None = None,
        high: torch.Tensor | None = None,
        validate_args: bool | None = None,
    ):
        base_dist = Beta(alpha, beta, validate_args=validate_args)
        super().__init__(base_dist, 1)  # Independent over last dimension

        # Store bounds; default to [0, 1] (no rescaling)
        # Move to same device/dtype as alpha to avoid CPU/CUDA mismatch
        device = alpha.device
        dtype = alpha.dtype
        if low is None:
            low = torch.zeros_like(alpha)
        else:
            low = low.to(device=device, dtype=dtype)
        if high is None:
            high = torch.ones_like(alpha)
        else:
            high = high.to(device=device, dtype=dtype)
        self._low = low
        self._high = high
        self._range = high - low  # cached for speed

    # ------------------------------------------------------------------
    # Rescaling helpers
    # ------------------------------------------------------------------
    def _to_action(self, unit: torch.Tensor) -> torch.Tensor:
        """Map [0,1] -> [low, high]."""
        return self._low + self._range * unit

    def _to_unit(self, action: torch.Tensor) -> torch.Tensor:
        """Map [low, high] -> [0,1]."""
        return (action - self._low) / self._range

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def sample(self, sample_shape=torch.Size()):
        unit = super().sample(sample_shape)
        return self._to_action(unit)

    def rsample(self, sample_shape=torch.Size()):
        unit = super().rsample(sample_shape)
        return self._to_action(unit)

    def log_prob(self, value):
        unit = self._to_unit(value)
        # Clamp to prevent -inf from Beta.log_prob at exact 0 or 1 boundaries
        unit = unit.clamp(1e-6, 1 - 1e-6)
        # Jacobian correction: log |d(unit)/d(value)| = -sum log(range)
        lp = super().log_prob(unit)
        lp = lp - self._range.log().sum(-1)
        return lp

    def entropy(self):
        # Independent.entropy() returns Beta entropy summed over action dims,
        # but doesn't account for the affine rescaling to [low, high].
        # For Y = low + range * X:  H(Y) = H(X) + sum(log(range))
        return super().entropy() + self._range.log().sum(-1)

    @property
    def mean(self):
        return self._to_action(self.base_dist.mean)

    @property
    def variance(self):
        return self.base_dist.variance * (self._range**2)

    @property
    def support(self):
        return self.base_dist.support
