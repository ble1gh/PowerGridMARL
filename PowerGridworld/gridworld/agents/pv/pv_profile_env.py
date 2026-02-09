import os

import numpy as np
import pandas as pd

import gymnasium as gym

from gridworld.log import logger
from gridworld import ComponentEnv
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROFILE_DIR = os.path.join(THIS_DIR, "profiles")


class PVEnv(ComponentEnv):
    """Simple control model for pv that is driven by a static profile for the 
    maximum real power injection.  See the ComponentEnv class for the required
    API.
    
    The profile CSV is treated as a 24-hour curve.  On each step the current
    time of day (passed as ``current_time`` from :class:`MultiAgentEnv`) is
    used to interpolate the profile, so the PV output correctly tracks the
    time of day regardless of episode start/end times or control time-step.
    """

    def __init__(
        self,
        name: str,
        profile_csv: str,
        profile_path: str = None,
        scaling_factor: float = 1.,
        profile_noise_std: float = 0.0,
        rescale_spaces: bool = True,
        grid_aware: bool = False,
        max_episode_steps: int = None,
        **kwargs
    ):

        """
        Args:
            name:  Component name.

            profile_csv:  Relative path from ./profiles for CSV file containing 
                the maximum real power profile that the device can generate.
                We assume the first column of this file contains these values 
                and discard the rest.  The file is interpreted as a **full
                24-hour** curve starting at midnight, sampled at uniform
                intervals (e.g. 288 rows → 5-minute resolution).
                
            profile_path:  Full path to profile csv.  If provided, this overrides
                the profile_csv argument.

            scaling_factor:  Float to rescale the csv data by (e.g., rated capacity in kW).

            profile_noise_std:  Standard deviation of Gaussian noise added to the
                normalized profile data before scaling. Applied on each reset.

            rescale_spaces:  If True, rescale action/obs spaces to [-1, 1].

            grid_aware:  If True, add "min_voltage" to obs space (for examples
                where PV is rewarded for voltage support).
        """

        super().__init__(name=name, **kwargs)

        self.scaling_factor = scaling_factor
        self.profile_noise_std = profile_noise_std
        self.rescale_spaces = rescale_spaces
        self.grid_aware = grid_aware

        # Read csv file.  If a full path is provide, that overrides reference to 
        # names of csv files stored locally in the `profiles` directory.
        profile_csv = os.path.join(PROFILE_DIR, profile_csv)
        if profile_path is not None:
            profile_csv = profile_path
        self.profile_csv = profile_csv
        
        # Read the base profile data (normalized 0-1 values).
        # Interpret it as a 24-hour curve starting at midnight.
        self.base_data = pd.read_csv(self.profile_csv).values[:, 0].squeeze()
        n_points = len(self.base_data)

        # Build a corresponding array of fractional-hours-since-midnight for
        # each row, e.g. 288 rows → [0, 5/60, 10/60, …, 23.9167].
        self._profile_hours = np.linspace(0.0, 24.0, n_points, endpoint=False)
        
        # Apply noise and scaling to create working data
        # Noise is applied to normalized data, then scaled
        self._apply_noise_and_scale()

        # The current interpolated PV value (set by _lookup_time)
        self._current_pv = 0.0
        
        # Fallback index for standalone usage (no current_time provided).
        # When current_time is supplied, this is ignored.
        self._index = 0
        self._use_time_lookup = False  # set True once current_time is received
        self._episode_length = len(self.base_data)
        if max_episode_steps is not None:
            self._episode_length = min(max_episode_steps, self._episode_length)

        # Create the obs labels and bounds.
        self._obs_labels = ["real_power"]
        self._obs_labels += ["min_voltage"] if grid_aware else []

        obs_bounds = {
            "real_power": (-np.max(self.data), 0.),
            "min_voltage": (0.9, 1.1)
        }

        # Create the optionally rescaled gym spaces.
        self._observation_space = gym.spaces.Box(
            shape=(len(self.obs_labels),),
            low=np.array([v[0] for k, v in obs_bounds.items() if k in self.obs_labels]),
            high=np.array([v[1] for k, v in obs_bounds.items() if k in self.obs_labels]),
            dtype=np.float64)

        self.observation_space = maybe_rescale_box_space(
            self._observation_space, rescale=self.rescale_spaces)

        self._action_space = gym.spaces.Box(
            shape=(1,), low=0., high=1., dtype=np.float64)

        self.action_space = maybe_rescale_box_space(
            self._action_space, rescale=self.rescale_spaces)


    def _lookup_time(self, current_time):
        """Interpolate the 24-hour profile at the given time of day.

        Args:
            current_time: A ``pd.Timestamp`` (or anything with ``.hour``,
                ``.minute``, ``.second`` attributes).

        Sets ``self._current_pv`` to the interpolated profile value (in kW
        after scaling).  If *current_time* is ``None`` the value is set to 0.
        """
        if current_time is None:
            self._current_pv = 0.0
            return
        hour_of_day = (
            current_time.hour
            + current_time.minute / 60.0
            + current_time.second / 3600.0
        )
        # Linear interpolation with wrap-around (np.interp handles ascending x)
        self._current_pv = float(np.interp(hour_of_day, self._profile_hours, self.data))

    def get_obs(self, **kwargs):
        """Returns the maximum real power possible for the current time of day.

        If ``current_time`` is provided in *kwargs* (as a ``pd.Timestamp``),
        the profile is looked up by time of day.  Otherwise falls back to
        the index-based sequential profile for standalone usage.
        """
        if "current_time" in kwargs and kwargs["current_time"] is not None:
            self._lookup_time(kwargs["current_time"])
        elif not self._use_time_lookup:
            # Fallback: use sequential index into the profile array
            self._current_pv = self.data[min(self._index, len(self.data) - 1)]

        raw_obs = [-self._current_pv]
        if self.grid_aware:
            raw_obs = raw_obs + [kwargs["min_voltage"]]
        raw_obs = np.array(raw_obs)
        if self.rescale_spaces:
            obs = to_scaled(raw_obs, self._observation_space.low, self._observation_space.high)
        else:
            obs = raw_obs
        return obs, {"real_power": raw_obs[0]}


    def _apply_noise_and_scale(self):
        """Apply Gaussian noise to base profile and scale to get working data."""
        if self.profile_noise_std > 0:
            noise = np.random.normal(0, self.profile_noise_std, size=self.base_data.shape)
            noisy_data = self.base_data + noise
            # Clip to [0, 1] to keep valid capacity factors
            noisy_data = np.clip(noisy_data, 0, 1)
        else:
            noisy_data = self.base_data.copy()
        
        self.data = noisy_data * self.scaling_factor

    def is_terminal(self):
        """When driven by MultiAgentEnv (time-based), never self-terminate.
        In standalone mode, terminate at end of profile."""
        if self._use_time_lookup:
            return False
        return self._index >= (self._episode_length - 1)


    def step_reward(self, **kwargs):
        """Step reward is always zero."""
        return 0., {}


    def reset(self, **kwargs):
        """Re-apply noise and look up the initial PV value for the current time."""
        self._index = 0
        # Re-apply noise each episode for stochasticity
        self._apply_noise_and_scale()
        # Set initial PV value from time of day (if provided)
        if "current_time" in kwargs and kwargs["current_time"] is not None:
            self._use_time_lookup = True
            self._lookup_time(kwargs["current_time"])
        else:
            self._use_time_lookup = False
            self._current_pv = self.data[0]
        self.get_obs(**kwargs)


    def step(self, action, **kwargs):
        """Look up the PV profile for the current time of day and apply the
        curtailment action (0–1 fraction of max output)."""

        if self.rescale_spaces:
            action = to_raw(action, self._action_space.low, self._action_space.high)

        # Update PV value from time of day, then get observation
        obs, obs_meta = self.get_obs(**kwargs)
        self._real_power = np.float64((action * obs_meta["real_power"]).squeeze())
        
        # Advance fallback index for standalone (non-time-based) usage
        if not self._use_time_lookup:
            self._index += 1
        
        rew, _ = self.step_reward(**kwargs)

        return obs, rew, self.is_terminal(), obs_meta
