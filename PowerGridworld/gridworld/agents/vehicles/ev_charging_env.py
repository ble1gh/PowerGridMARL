import os
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pandas as pd

from gridworld import ComponentEnv
from gridworld.log import logger
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class EVChargingEnv(ComponentEnv):
    def __init__(
        self,
        num_vehicles: int = 100,
        minutes_per_step: int = 5,
        max_charge_rate_kw: float = 7.0,  # ~40. for fast charge
        max_episode_steps: int = None,
        unserved_penalty: float = 1.0,
        urgency_coef: float = 0.0,
        peak_penalty: float = 0.0,
        peak_threshold: float = 10.0,
        reward_scale: float = 1e5,
        name: str = None,
        randomize: bool = False,
        vehicle_csv: str = None,
        vehicle_multiplier: int = 1,
        rescale_spaces: bool = True,
        random_arrival: bool = False,
        arrival_probability: float = 0.05,
        min_charge_duration_min: int = 60,
        max_charge_duration_min: int = 240,
        start_time=None,
        end_time=None,
        control_timedelta=None,
        **kwargs,
    ):

        super().__init__(name=name)

        # Per-instance RNG (will be overwritten by MultiAgentEnv._set_seed)
        self.rng = np.random.RandomState()

        self.num_vehicles = num_vehicles
        self.max_charge_rate_kw = max_charge_rate_kw
        self.minutes_per_step = minutes_per_step
        self.randomize = randomize
        self.vehicle_multiplier = vehicle_multiplier
        self.rescale_spaces = rescale_spaces

        # Random arrival parameters
        self.random_arrival = random_arrival
        self.arrival_probability = arrival_probability
        self.min_charge_duration_min = min_charge_duration_min
        self.max_charge_duration_min = max_charge_duration_min
        self._is_connected = False
        self._has_arrived = False
        self._has_departed = False
        self._sampled_duration = 0

        # Reward parameters
        self.unserved_penalty = unserved_penalty
        self.urgency_coef = urgency_coef
        self.peak_penalty = peak_penalty
        self.peak_threshold = peak_threshold
        self.reward_scale = reward_scale

        # By default, we simulate a whole day but allow user to specify
        # fewer steps if desired.
        self.start_time = pd.Timestamp(start_time)
        self.end_time = pd.Timestamp(end_time)
        self.control_timedelta = control_timedelta
        total_seconds = (self.end_time - self.start_time).total_seconds()
        step_seconds = pd.Timedelta(seconds=self.control_timedelta).total_seconds()
        self.max_episode_steps = int(total_seconds // step_seconds) + 1

        # Create an array of simulation times in minutes, in the interval
        # (0, max_episode_steps * minutes_per_step).
        self.simulation_times = np.arange(
            0, self.max_episode_steps * minutes_per_step, minutes_per_step
        )

        # Attributes that will be initialized in reset.
        self.time_index = None  # time index
        self.time = None  # time in minutes
        self.df = None  # episode vehicle dataframe
        self.charging_vehicles = None  # charging vehicle list
        self.departed_vehicles = None  # vehicle list departed in last time step

        # Participation score: total remaining energy for all currently connected vehicles
        # Updates whenever the number of charging vehicles changes (connect or disconnect)
        self._participation_score = 0.0
        self._prev_num_charging = 0  # Track previous number of charging vehicles to detect changes

        # Read the source dataframe.
        vehicle_csv = vehicle_csv if vehicle_csv else os.path.join(THIS_DIR, "vehicles.csv")
        self._df = pd.read_csv(vehicle_csv)  # all vehicles
        self._df["energy_required_kwh"] *= self.vehicle_multiplier

        # Round the start/end times to the nearest step.
        self._df["start_time_min"] = self._round(self._df["start_time_min"])
        self._df["end_time_park_min"] = self._round(self._df["end_time_park_min"])

        # Comment out random arrival/departure times - use for original behavior
        # # Adjust "start_time_min" to be uniformly distributed in the first 2 hours.
        # self._df["start_time_min"] = np.random.uniform(
        #     0, 2 * 60, size=len(self._df)
        # ).astype(int)

        # # Adjust "end_time_park_min" to be randomly distributed in the last 2 hours.
        # self._df["end_time_park_min"] = np.random.uniform(
        #     (self.max_episode_steps * self.minutes_per_step) - (2 * 60),
        #     self.max_episode_steps * self.minutes_per_step,
        #     size=len(self._df)
        # ).astype(int)
        # self._df["end_time_park_min"] = np.minimum(self._df["end_time_park_min"], self.simulation_times[-1])

        # Set all vehicles to arrive at first time step and depart at last time step
        # self._df["start_time_min"] = self.simulation_times[0]  # First time step
        # self._df["end_time_park_min"] = self.simulation_times[-1]  # Last time step

        # Bounds on the observation space variables.
        obs_bounds = OrderedDict(
            {
                "time": (0, self.simulation_times[-1]),
                "time_remaining": (0, self.simulation_times[-1]),
                "real_power_consumed": (
                    0,
                    self.num_vehicles
                    * self.max_charge_rate_kw
                    * self.vehicle_multiplier
                    * (self.minutes_per_step / 60.0),
                ),
                "real_power_demand": (0, self.num_vehicles * self._df["energy_required_kwh"].max()),
                # "num_active_vehicles": (
                #     0, self.num_vehicles * self.vehicle_multiplier),
                # "mean_charge_rate_deficit": (
                #     0, self._df["energy_required_kwh"].max() / (self.minutes_per_step / 60.)),
                "real_energy_unserved": (
                    0,
                    self.num_vehicles * self._df["energy_required_kwh"].max(),
                ),
            }
        )

        # Construct the gym spaces.
        self._observation_space = gym.spaces.Box(
            low=np.array([x[0] for x in obs_bounds.values()]),
            high=np.array([x[1] for x in obs_bounds.values()]),
            shape=(len(obs_bounds),),
            dtype=np.float64,
        )
        self.observation_space = maybe_rescale_box_space(
            self._observation_space, rescale=self.rescale_spaces
        )

        # Fraction between 0 and 1 of max charge rate for all charging vehicles.
        self._action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
        self.action_space = maybe_rescale_box_space(self._action_space, rescale=self.rescale_spaces)

        # Use a dictionary to keep track of various state quantities.
        # Use the self._update(key, value) to ensure valid keys when updating state.
        self.state = OrderedDict({k: None for k in obs_bounds.keys()})

        # Use the state dict to create the observation labels.
        self._obs_labels = list(self.state.keys())

    @property
    def participation_score(self) -> float:
        """Returns the participation score (total remaining energy for connected vehicles).

        This value is recalculated whenever the number of charging vehicles changes
        (vehicle connects or disconnects). It represents the current 'task difficulty'
        for this agent and is used as a node feature in the hypernetwork GNN.
        """
        return self._participation_score

    @property
    def max_real_power(self) -> float:
        """Returns the maximum possible power draw if action were 1.0 (no curtailment).

        This is the sum of min(max_charge_per_step, energy_required) across all
        currently connected vehicles. Computed during each step before the
        charging loop modifies energy_required. Used for VPP reward calculation
        where the EV demand response contribution = max_real_power - real_power.
        """
        return self._max_real_power

    @property
    def is_connected(self) -> bool:
        """Whether the vehicle is currently connected for charging.

        In random_arrival mode, this reflects the stochastic arrival/departure state.
        In standard mode, returns True if any charging vehicles are present.
        Used by the multi-agent wrapper to set per-agent active masks.
        """
        if self.random_arrival:
            return self._is_connected
        return self.charging_vehicles is not None and len(self.charging_vehicles) > 0

    def get_obs(self, **kwargs) -> tuple[dict, dict]:
        "Returns an observation dict and metadata dict."
        # Ensure every state value is a plain float scalar so np.array()
        # never encounters inhomogeneous shapes.
        raw_obs = np.array([float(v) for v in self.state.values()])
        if self.rescale_spaces:
            obs = to_scaled(raw_obs, self._observation_space.low, self._observation_space.high)
        else:
            obs = raw_obs
        return obs.copy(), self.state.copy()

    def is_terminal(self) -> bool:
        """Returns True if max episode steps have been reached."""
        return self.time_index == self.max_episode_steps - 1

    def step_reward(self) -> tuple[float, dict]:
        """Return a non-zero reward here if you want to use RL.

        Components:
          - unserved_reward: penalty for energy remaining when a vehicle departs
          - urgency_reward: per-step shaping that grows as the charging deadline
            tightens, kicking in once the agent must charge at >= 100% of max
            rate every remaining step (deficit_ratio >= 1.0)
          - peak_reward: penalty for exceeding the peak power threshold
        """
        # ── Unserved penalty (on departure / episode end) ──
        unserved_reward = -self.unserved_penalty * self.state["real_energy_unserved"] ** 2

        # ── Urgency shaping (per-step, grows as deadline tightens) ──
        urgency_reward = 0.0
        deficit_ratio = 0.0
        if self.urgency_coef > 0:
            energy_remaining = self.state["real_power_demand"]
            time_remaining_min = self.state["time_remaining"]
            max_energy_per_step = (
                self.max_charge_rate_kw * self.vehicle_multiplier * (self.minutes_per_step / 60.0)
            )

            if energy_remaining > 0 and time_remaining_min > 0:
                steps_remaining = time_remaining_min / self.minutes_per_step
                energy_per_step_needed = energy_remaining / steps_remaining
                # 0 = fully charged, 1 = must charge at max rate every remaining step
                deficit_ratio = (
                    energy_per_step_needed / max_energy_per_step if max_energy_per_step > 0 else 0.0
                )
                # Quadratic penalty kicks in at deficit_ratio >= 1.0 (infeasible territory)
                urgency_reward = -self.urgency_coef * max(0.0, deficit_ratio - 1.0) ** 2
            elif energy_remaining > 0 and time_remaining_min <= 0:
                # Past deadline with energy still remaining — maximum urgency
                deficit_ratio = (
                    energy_remaining / max_energy_per_step if max_energy_per_step > 0 else 0.0
                )
                urgency_reward = -self.urgency_coef * (deficit_ratio**2)

        # ── Peak penalty ──
        peak_reward = (
            -self.peak_penalty
            * max(0, self.state["real_power_consumed"] - self.peak_threshold) ** 2
        )

        reward = unserved_reward + urgency_reward + peak_reward
        reward /= self.reward_scale

        # Scale components to match actual contribution to agent reward
        scaled_unserved = unserved_reward / self.reward_scale
        scaled_urgency = urgency_reward / self.reward_scale
        scaled_peak = peak_reward / self.reward_scale

        return reward, {
            "unserved_reward": scaled_unserved,
            "urgency_reward": scaled_urgency,
            "deficit_ratio": deficit_ratio,
            "peak_reward": scaled_peak,
        }

    def reset(self, **kwargs) -> tuple[dict, dict]:
        """Reset the initial conditions and run a single step of the simulation
        so that `get_obs` here can be used in the first control step."""

        self.time_index = 0
        self.time = self.simulation_times[self.time_index]
        self.charging_vehicles = []
        self.departed_vehicles = []

        # Select first N vehicles if not randomized, else shuffle rows of df.
        self.df = (
            self._df.sample(self.num_vehicles, random_state=self.rng).copy()
            if self.randomize
            else self._df[: self.num_vehicles].copy()
        )
        self.df = self.df.reset_index()  # index is now 0 to N-1

        # Randomly sample energy_required_kwh for each vehicle from the full distribution
        sampled_energies = (
            self._df["energy_required_kwh"]
            .sample(n=len(self.df), replace=True, random_state=self.rng)
            .values
        )
        self.df["energy_required_kwh"] = sampled_energies

        # Initialize real power.
        self._real_power = 0.0
        self._max_real_power = 0.0

        # Reset participation score tracking
        self._participation_score = 0.0
        self._prev_num_charging = 0

        # Random arrival: initialize vehicle as not yet arrived
        self._just_arrived_this_step = False
        if self.random_arrival:
            self._has_arrived = False
            self._has_departed = False
            self._is_connected = False
            # Sample a random charging duration (rounded to step boundary, min 1 step)
            raw_duration = self.rng.randint(
                self.min_charge_duration_min, self.max_charge_duration_min + 1
            )
            self._sampled_duration = max(self._round(raw_duration), self.minutes_per_step)
            # Prevent auto-arrival by setting times far beyond simulation end
            for i in range(len(self.df)):
                self.df.at[i, "start_time_min"] = self.simulation_times[-1] + 99999
                self.df.at[i, "end_time_park_min"] = self.simulation_times[-1] + 99999

        # Step the simulator one time without a control action.
        self.step()

        # Get the observation needed to solve the first control step.
        obs, _ = self.get_obs()

        return obs, {}

    def step(self, action: np.ndarray = None, **kwargs) -> tuple[np.ndarray, float, bool, dict]:

        logger.debug(f"Time index {self.time_index}/{self.max_episode_steps}")
        logger.debug(f"Action: {action}")

        # If no action is applied, use minimum.
        # TODO: Make sure you are scaling things correctly.
        action = action if action is not None else self._action_space.low
        if self.rescale_spaces:
            action = to_raw(action, self._action_space.low, self._action_space.high)

        action_kw = action[0] * self.max_charge_rate_kw * self.vehicle_multiplier
        action_kwh = action_kw * (self.minutes_per_step / 60.0)

        # Random arrival: check if the vehicle arrives this timestep
        time_until_end = self.simulation_times[-1] - self.time
        if (
            self.random_arrival
            and not self._has_arrived
            and not self._has_departed
            and time_until_end >= self.min_charge_duration_min
        ):
            if self.rng.random() < self.arrival_probability:
                self._has_arrived = True
                self._is_connected = True
                self._just_arrived_this_step = True
                # Set arrival time to now, departure based on sampled duration
                for i in range(len(self.df)):
                    self.df.at[i, "start_time_min"] = self.time
                    end_time = min(self.time + self._sampled_duration, self.simulation_times[-1])
                    self.df.at[i, "end_time_park_min"] = end_time

        # Get indexes of vehicles arriving and departing.
        start_idx = np.where(self.time >= np.floor(self.df["start_time_min"]))[0]
        end_idx = np.where(self.time <= np.floor(self.df["end_time_park_min"]))[0]

        # Get indexes of charging vehicles.
        charging_vehicles = list(set(list(start_idx)).intersection(set(list(end_idx))))
        charging_vehicles = [
            i for i in charging_vehicles if self.df.at[i, "energy_required_kwh"] > 0.0
        ]

        # Get vehicles that have left the station in the last time step.
        self.departed_vehicles = list(set(self.charging_vehicles) - set(charging_vehicles))

        # Random arrival: update connection status
        # Vehicle remains "connected" on its departure step so the reward flows through,
        # then disconnects on the following step.
        if self.random_arrival and self._has_arrived:
            if 0 in charging_vehicles or 0 in self.departed_vehicles:
                self._is_connected = True
            else:
                self._is_connected = False
                # Reset flags so the vehicle can arrive again on a future step
                self._has_arrived = False
                self._has_departed = False
                # Resample charging duration for next potential arrival
                raw_duration = self.rng.randint(
                    self.min_charge_duration_min, self.max_charge_duration_min + 1
                )
                self._sampled_duration = max(self._round(raw_duration), self.minutes_per_step)
                # Resample energy required from original data distribution
                for i in range(len(self.df)):
                    sampled_row = self._df.sample(1, random_state=self.rng).iloc[0]
                    self.df.at[i, "energy_required_kwh"] = sampled_row["energy_required_kwh"]
                # Reset times to far future to prevent auto-arrival
                for i in range(len(self.df)):
                    self.df.at[i, "start_time_min"] = self.simulation_times[-1] + 99999
                    self.df.at[i, "end_time_park_min"] = self.simulation_times[-1] + 99999

        # Participation score logic:
        # - Recalculate total remaining energy whenever the number of charging vehicles changes
        # - This captures both connection and disconnection events
        current_num_charging = len(charging_vehicles)
        if current_num_charging != self._prev_num_charging:
            # Number of vehicles changed - recalculate participation score
            if current_num_charging > 0:
                self._participation_score = sum(
                    self.df.at[i, "energy_required_kwh"] for i in charging_vehicles
                )
            else:
                self._participation_score = 0.0
            self._prev_num_charging = current_num_charging

        logger.debug(
            f"STEP, {self.time}, {self.time_index}, {charging_vehicles}, {self.departed_vehicles}"
        )

        # Aggregate quantities that are needed for obs space.
        real_power_consumed = 0.0
        real_power_demand = 0.0
        # min_energy_required = 0.
        # charge_rate_deficit = []  # charge rate missing to reach full charge

        # Compute max possible power draw at action=1.0 (before charging loop
        # modifies energy_required). Used for VPP reward calculation.
        max_action_kwh = (
            self.max_charge_rate_kw * self.vehicle_multiplier * (self.minutes_per_step / 60.0)
        )
        self._max_real_power = 0.0
        for i in charging_vehicles:
            er = self.df["energy_required_kwh"][i]
            if er > 0.0:
                self._max_real_power += min(max_action_kwh, er)

        for i in charging_vehicles:
            # Compute energy required to fully charge.
            energy_required_kwh = self.df["energy_required_kwh"][i]

            # If the vehicle does not require any more charging then skip it.
            if energy_required_kwh <= 0.0:
                continue

            # Apply action and update the vehicle data.
            charge_energy_kwh = min(action_kwh, energy_required_kwh)
            self.df.at[i, "energy_required_kwh"] -= charge_energy_kwh
            self.df.at[i, "energy_required_kwh"] = max(0.0, self.df.at[i, "energy_required_kwh"])
            real_power_consumed += charge_energy_kwh

            # Update energy required to fully charge.
            energy_required_kwh = self.df["energy_required_kwh"][i]

            # Update the aggregate variables
            real_power_demand += energy_required_kwh
            # min_energy_required = max(min_energy_required, energy_required_kwh)

            # What is the min energy this vehicle needs to reach full charge?
            # time_left_h = (self.df["end_time_park_min"][i] - self.time) / 60.
            # if time_left_h <= 0:
            #     continue
            # deficit = max(
            #     0, energy_required_kwh / time_left_h - self.max_charge_rate_kw)
            # charge_rate_deficit.append(deficit)

            # print(action_kwh, energy_required_kwh, real_power_consumed)

            logger.debug(f"{i}, {energy_required_kwh}, {action}")

        # Check done
        done = self.is_terminal()

        # Update time variables.
        if not done:
            self.time_index += 1
            self.time = self.simulation_times[self.time_index]
            self.charging_vehicles = charging_vehicles

        # Compute unmet charging demand as a *per-step* quantity:
        # - normal steps: only vehicles that departed this step
        # - terminal step: departed this step + vehicles still connected at episode end
        # This avoids carry-over accumulation that would over-penalize later departures.
        departed_unserved = 0.0
        for i in self.departed_vehicles:
            departed_unserved += self.df["energy_required_kwh"][i]

        terminal_unserved = 0.0
        if done:
            for i in charging_vehicles:
                terminal_unserved += self.df["energy_required_kwh"][i]

        self._update("real_energy_unserved", departed_unserved + terminal_unserved)

        # Update the state dict.
        self._update("time", self.time)
        # Time remaining until the soonest-departing connected vehicle leaves (minutes).
        # If no vehicles are connected, default to 0.
        if charging_vehicles:
            soonest_departure = min(self.df.at[i, "end_time_park_min"] for i in charging_vehicles)
            time_remaining = max(0.0, soonest_departure - self.time)
        else:
            time_remaining = 0.0
        self._update("time_remaining", time_remaining)
        self._update(
            "real_power_consumed", real_power_consumed
        )  # Already includes vehicle_multiplier from action
        self._update(
            "real_power_demand", real_power_demand
        )  # Already includes vehicle_multiplier from energy
        # self._update("num_active_vehicles", self.vehicle_multiplier * len(charging_vehicles))
        # self._update(
        #     "mean_charge_rate_deficit",
        #     0 if len(charge_rate_deficit) == 0 else np.mean(charge_rate_deficit))

        # On the arrival step the agent's action was based on the previous
        # observation (no vehicle), so any difference between max and actual
        # power is NOT intentional demand-response.  Set _max_real_power equal
        # to what was actually consumed so that curtailment = 0 for this step.
        if self._just_arrived_this_step:
            self._max_real_power = real_power_consumed
            self._just_arrived_this_step = False

        # Update the real power attribute needed for component envs.
        self._real_power = real_power_consumed  # Already includes vehicle_multiplier

        # Get the return values
        obs, meta = self.get_obs()
        rew, rew_meta = self.step_reward()

        meta.update(rew_meta)

        # Only report energy remaining for vehicles currently plugged in
        current_energy_remaining = self.df["energy_required_kwh"].copy()
        mask = np.ones(len(self.df), dtype=bool)
        if charging_vehicles:
            mask[charging_vehicles] = False
        current_energy_remaining.loc[mask] = 0.0

        meta["energy_remaining"] = current_energy_remaining.to_dict()
        meta["participation_score"] = self._participation_score

        return obs, rew, done, meta

    def _update(self, key, value):
        if key not in self.state:
            raise ValueError(f"Invalid state key {key}")
        self.state[key] = value

    def _round(self, x):
        """Round the value x down to the nearest time step interval."""
        return x - x % self.minutes_per_step
