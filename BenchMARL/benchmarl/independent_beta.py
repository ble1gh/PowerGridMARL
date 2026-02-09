"""Independent Beta distribution for bounded [0,1] actions."""

import torch
from torch.distributions import Beta, Independent
from typing import Optional


class IndependentBeta(Independent):
    """Independent Beta distribution for bounded actions in [0,1].
    
    This wraps the standard Beta distribution to work with TorchRL's
    ProbabilisticActor interface.
    """
    
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, validate_args: Optional[bool] = None):
        """Initialize Independent Beta distribution.
        
        Args:
            alpha: Concentration parameter (> 0)
            beta: Concentration parameter (> 0)
            validate_args: Whether to validate arguments
        """
        base_dist = Beta(alpha, beta, validate_args=validate_args)
        super().__init__(base_dist, 1)  # Independent over last dimension
    
    @property
    def mean(self):
        """Mean of the Beta distribution: alpha / (alpha + beta)"""
        return self.base_dist.mean
    
    @property
    def variance(self):
        """Variance of the Beta distribution."""
        return self.base_dist.variance
    
    @property
    def support(self):
        """Support of the Beta distribution: [0, 1]"""
        return self.base_dist.support
