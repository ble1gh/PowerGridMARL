#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""SMACv2 Variable-Composition Environment Wrapper.

Transforms SMACv2's single-group agent interface into per-unit-type groups
with active_mask support for HGTeam's variable-composition framework.

Base SMACv2Env: single "agents" group → all units lumped together.
This wrapper: per-type groups ("stalker", "zealot", "colossus") with
active_mask, participation_score, and per-capita reward broadcasting.
"""

import copy
import math
from collections import OrderedDict
from collections.abc import Callable

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.envs.transforms import RewardSum, Transform

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

# --- Protoss unit type names (ordered) ---
PROTOSS_TYPE_NAMES = ("stalker", "zealot", "colossus")


class VariableSmacv2Env(EnvBase):
    """Wraps an SMACv2-like base env to provide per-unit-type grouping.

    The base env provides a single "agents" group with all units.
    This wrapper splits into per-type groups ("stalker", "zealot", "colossus")
    and provides:
    - Per-type observations, action_masks, active_masks
    - Flat active_mask across all types
    - Per-type participation scores (1.0 for alive, 0.0 for dead/padded)
    - Per-capita reward broadcasting (D2)
    - Optional entity-decomposed observations (entity_obs=True)
    """

    # Observation layout for Protoss (obs_all_health, obs_own_pos, use_unit_ranges):
    # move_feats(4) | enemy_feats(n_enemies * entity_dim) | ally_feats(n_allies * entity_dim) | own_feats(7)
    MOVE_FEATS_DIM = 4
    OWN_FEATS_DIM = 7
    ENTITY_FEATS_DIM = 9  # [visible/shootable, dist, rx, ry, health, shield, t0, t1, t2]

    # Type-bit encoding for Protoss units (3 bits in entity features[-3:])
    _TYPE_BIT_MAP = {
        (True, False, False): "stalker",
        (False, True, False): "zealot",
        (False, False, True): "colossus",
    }

    def __init__(
        self,
        base_env: EnvBase,
        max_per_type: dict[str, int],
        device: DEVICE_TYPING = "cpu",
        entity_obs: bool = False,
    ):
        super().__init__(device=device, batch_size=torch.Size([]))
        self.base_env = base_env
        self.max_per_type = OrderedDict(max_per_type)
        self.type_names = list(self.max_per_type.keys())
        self.max_total = sum(self.max_per_type.values())
        self.entity_obs = entity_obs

        # Dimensions from base env
        base_obs = base_env.observation_spec
        self.n_units = base_obs["agents", "observation"].shape[-2]
        self._obs_dim = base_obs["agents", "observation"].shape[-1]

        act_entry = base_env.full_action_spec["agents", "action"]
        self._n_actions = getattr(act_entry, "n", None) or act_entry.shape[-1]

        self._state_dim = (
            base_obs["state"].shape[-1] if "state" in base_obs.keys() else None
        )
        self._episode_limit = getattr(base_env, "episode_limit", 200)

        # Entity obs dimensions (derived from n_units)
        self._n_enemies = self.n_units  # SMACv2: n_enemies == n_units (from config)
        self._n_allies = self.n_units - 1  # each agent sees n_units-1 allies
        # Max allies per type: the total ally count is n_units-1
        # We allocate max_per_type[t] - 1 for the agent's own type (it's an ally
        # to itself minus 1), and max_per_type[t] for other types.  But the
        # simplest correct approach is: max allies of type t = max_per_type[t]
        # (an agent can never see itself as an ally, but another slot of the
        # same type can).  The actual count varies per agent; padding handles it.
        self._max_allies_per_type = dict(self.max_per_type)  # copy

        # Per-episode type assignment (updated on reset)
        self._type_assignments: list[str] = []
        self._local_indices: list[int] = []
        self._type_counts: dict[str, int] = {}

        # Cached base env tensordict for step forwarding
        self._base_td: TensorDictBase | None = None

        self._build_specs()

    @property
    def group_map(self) -> dict[str, list[str]]:
        return {
            t: [f"{t}_{i}" for i in range(m)] for t, m in self.max_per_type.items()
        }

    @property
    def episode_limit(self) -> int:
        return self._episode_limit

    # ------------------------------------------------------------------ #
    #  Spec construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_specs(self):
        obs = Composite(device=self.device)
        act = Composite(device=self.device)
        rew = Composite(device=self.device)

        for tn, mc in self.max_per_type.items():
            per_agent_obs = dict(
                observation=Unbounded(shape=(self._obs_dim,)),
                action_mask=Unbounded(
                    shape=(self._n_actions,), dtype=torch.bool
                ),
                active_mask=Unbounded(shape=(), dtype=torch.bool),
            )
            if self.entity_obs:
                per_agent_obs["entity_enemy"] = Unbounded(
                    shape=(self._n_enemies, self.ENTITY_FEATS_DIM)
                )
                per_agent_obs["entity_self"] = Unbounded(
                    shape=(self.OWN_FEATS_DIM,)
                )
                per_agent_obs["move_feats"] = Unbounded(
                    shape=(self.MOVE_FEATS_DIM,)
                )
                for ally_tn in self.type_names:
                    per_agent_obs[f"entity_{ally_tn}_ally"] = Unbounded(
                        shape=(self._max_allies_per_type[ally_tn], self.ENTITY_FEATS_DIM)
                    )
            obs[tn] = Composite(
                per_agent_obs,
                device=self.device,
            ).expand(mc)
            act[tn] = Composite(
                action=Categorical(n=self._n_actions, shape=()),
                device=self.device,
            ).expand(mc)
            rew[tn] = Composite(
                reward=Unbounded(shape=(1,)),
                device=self.device,
            ).expand(mc)
            obs.set(
                f"{tn}_participation_score",
                Unbounded(shape=(mc, 1)),
            )

        obs.set(
            "active_mask",
            Unbounded(shape=(self.max_total,), dtype=torch.bool),
        )
        if self._state_dim is not None:
            obs.set("state", Unbounded(shape=(self._state_dim,)))
        obs["info"] = Composite(
            battle_won=Unbounded(shape=(1,), dtype=torch.bool),
            episode_limit=Unbounded(shape=(1,), dtype=torch.bool),
            device=self.device,
        )

        self.observation_spec = obs
        self.action_spec = act
        self.reward_spec = rew
        self.done_spec = Composite(
            done=Unbounded(shape=(1,), dtype=torch.bool),
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    #  Type detection & assignment                                        #
    # ------------------------------------------------------------------ #

    def _detect_and_assign_types(self, obs=None):
        """Detect unit types and assign agents to per-type group slots."""
        # Priority 1: base env exposes types directly (e.g. mock env)
        if hasattr(self.base_env, "unit_types"):
            labels = [self.type_names[t] for t in self.base_env.unit_types]

        # Priority 2: TorchRL SMACv2Env internals
        elif hasattr(self.base_env, "_env") and hasattr(
            self.base_env._env, "agents"
        ):
            labels = [
                self.type_names[
                    getattr(a, "unit_type_id", 0) % len(self.type_names)
                ]
                for a in self.base_env._env.agents
            ]

        # Priority 3: proportional fallback (weighted_teams [0.45, 0.45, 0.1])
        else:
            n = self.n_units
            ns = round(n * 0.45)
            nz = round(n * 0.45)
            nc = n - ns - nz
            labels = ["stalker"] * ns + ["zealot"] * nz + ["colossus"] * nc

        # Assign to group slots
        counts: dict[str, int] = {t: 0 for t in self.type_names}
        assignments: list[str] = []
        local_idxs: list[int] = []

        for label in labels:
            t = label if label in counts else self.type_names[0]
            li = counts[t]
            if li >= self.max_per_type[t]:
                # Overflow: find first type with space
                for alt in self.type_names:
                    if counts[alt] < self.max_per_type[alt]:
                        t, li = alt, counts[alt]
                        break
            assignments.append(t)
            local_idxs.append(li)
            counts[t] += 1

        self._type_assignments = assignments
        self._local_indices = local_idxs
        self._type_counts = counts

    # ------------------------------------------------------------------ #
    #  Data splitting / merging                                           #
    # ------------------------------------------------------------------ #

    def _split(self, flat: torch.Tensor, pad_value=0.0) -> dict[str, torch.Tensor]:
        """Split flat (n_units, ...) tensor into per-type padded tensors."""
        out = {}
        for tn in self.type_names:
            mc = self.max_per_type[tn]
            extra = flat.shape[1:] if flat.ndim > 1 else ()
            if flat.dtype == torch.bool:
                p = torch.full(
                    (mc, *extra), bool(pad_value), dtype=torch.bool, device=flat.device
                )
            else:
                p = torch.full(
                    (mc, *extra), pad_value, dtype=flat.dtype, device=flat.device
                )
            for i, (t, li) in enumerate(
                zip(self._type_assignments, self._local_indices)
            ):
                if t == tn:
                    p[li] = flat[i]
            out[tn] = p
        return out

    def _normalize_group_action(self, action: torch.Tensor, type_name: str) -> torch.Tensor:
        """Return a 1D per-slot action tensor for one agent type.

        The wrapper is single-env oriented, but TorchRL eval can preserve
        singleton rollout/env dims. Accept shapes such as ``(mc,)``,
        ``(mc, 1)``, ``(1, mc)``, and ``(1, mc, 1)``; reject true batched
        actions instead of guessing which environment to use.
        """
        action = action.to(device=self.device, dtype=torch.long)
        mc = self.max_per_type[type_name]

        while action.ndim > 1 and action.shape[-1] == 1:
            action = action.squeeze(-1)

        if mc == 1:
            if action.numel() != 1:
                raise RuntimeError(
                    f"Unsupported action shape for {type_name}: "
                    f"{tuple(action.shape)}; expected a single action slot"
                )
            return action.reshape(1)

        agent_axes = [dim for dim, size in enumerate(action.shape) if size == mc]
        if len(agent_axes) != 1:
            raise RuntimeError(
                f"Unsupported action shape for {type_name}: {tuple(action.shape)}; "
                f"expected exactly one agent axis of length {mc}"
            )

        agent_axis = agent_axes[0]
        non_agent_shape = [
            size for dim, size in enumerate(action.shape) if dim != agent_axis
        ]
        if any(size != 1 for size in non_agent_shape):
            raise RuntimeError(
                f"Unsupported batched action shape for {type_name}: {tuple(action.shape)}; "
                "only singleton non-agent dimensions are supported"
            )

        return action.movedim(agent_axis, 0).reshape(mc)

    def _merge_actions(self, td: TensorDictBase) -> torch.Tensor:
        """Merge per-type discrete actions back to flat (n_units,) tensor."""
        flat = torch.zeros(self.n_units, dtype=torch.long, device=self.device)
        actions_by_type = {}
        for tn in self.type_names:
            action = td.get((tn, "action"), None)
            if action is not None:
                actions_by_type[tn] = self._normalize_group_action(action, tn)

        for i, (t, li) in enumerate(
            zip(self._type_assignments, self._local_indices)
        ):
            a = actions_by_type.get(t, None)
            if a is not None:
                flat[i] = a[li]
        return flat

    def _team_reward_from_base(self, reward: torch.Tensor) -> torch.Tensor:
        """Extract the scalar team reward from supported SMAC reward layouts."""
        reward = reward.to(self.device)
        if reward.ndim == 0 or reward.numel() == 1:
            return reward.reshape(()).float()

        squeezed = reward
        while squeezed.ndim > 1 and squeezed.shape[-1] == 1:
            squeezed = squeezed.squeeze(-1)

        if squeezed.ndim == 1 and squeezed.shape[0] == self.n_units:
            return squeezed[0].float()

        raise RuntimeError(
            "Unsupported SMACv2 reward shape "
            f"{tuple(reward.shape)}; expected scalar, singleton, or per-agent "
            f"reward with first dimension {self.n_units}"
        )

    def _decompose_obs(
        self, obs_per_type: dict[str, torch.Tensor], active_t: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Decompose flat observations into entity tensors per agent group.

        For each agent in each type group, the flat obs is sliced into:
        - move_feats: (4,)
        - entity_enemy: (n_enemies, 9)
        - entity_{T}_ally: (max_allies_T, 9) for each type T
        - entity_self: (7,)

        Ally features are routed to per-type tensors by parsing the 3 type
        bits in each ally's features.  Out-of-range allies (all zeros) are
        left as zeros in their respective type tensor.
        """
        ed = self.ENTITY_FEATS_DIM
        result = {}

        for tn in self.type_names:
            mc = self.max_per_type[tn]
            flat = obs_per_type[tn]  # (mc, obs_dim)
            alive = active_t[tn]  # (mc,)

            # Slice flat obs
            move = flat[:, :self.MOVE_FEATS_DIM]  # (mc, 4)
            enemy_start = self.MOVE_FEATS_DIM
            enemy_end = enemy_start + self._n_enemies * ed
            enemy_flat = flat[:, enemy_start:enemy_end]  # (mc, n_enemies*9)
            enemy = enemy_flat.reshape(mc, self._n_enemies, ed)

            ally_start = enemy_end
            ally_end = ally_start + self._n_allies * ed
            ally_flat = flat[:, ally_start:ally_end]  # (mc, n_allies*9)
            ally = ally_flat.reshape(mc, self._n_allies, ed)

            own_start = ally_end
            own = flat[:, own_start:own_start + self.OWN_FEATS_DIM]  # (mc, 7)

            # Route allies to per-type tensors by parsing type bits
            ally_typed = {}
            for at in self.type_names:
                ally_typed[at] = torch.zeros(
                    mc, self._max_allies_per_type[at], ed,
                    device=flat.device, dtype=flat.dtype,
                )

            # For each agent slot, iterate over its ally features and route
            # by type bits.  Vectorized over the ally dimension.
            # Type bits are the last 3 features of each ally entity.
            type_bits = ally[:, :, -3:]  # (mc, n_allies, 3)
            ally_active = ally.abs().sum(-1) > 0  # (mc, n_allies)

            # Counters for slot assignment (per agent, per ally type)
            for at_idx, at in enumerate(self.type_names):
                # Build mask: ally is of this type if its type bit is set
                # type_bits[:, :, at_idx] > 0.5 AND ally is active
                is_type = (type_bits[:, :, at_idx] > 0.5) & ally_active  # (mc, n_allies)

                max_at = self._max_allies_per_type[at]
                # For each agent slot, scatter matching allies into the typed tensor
                for agent_i in range(mc):
                    if not alive[agent_i]:
                        continue
                    indices = is_type[agent_i].nonzero(as_tuple=True)[0]
                    n_fill = min(len(indices), max_at)
                    if n_fill > 0:
                        ally_typed[at][agent_i, :n_fill] = ally[agent_i, indices[:n_fill]]

            group_result = {
                "move_feats": move,
                "entity_enemy": enemy,
                "entity_self": own,
            }
            for at in self.type_names:
                group_result[f"entity_{at}_ally"] = ally_typed[at]

            result[tn] = group_result

        return result

    # ------------------------------------------------------------------ #
    #  Output tensordict construction                                     #
    # ------------------------------------------------------------------ #

    def _build_output(
        self, base_td: TensorDictBase, with_reward: bool = False
    ) -> TensorDict:
        """Convert base env output to per-type grouped format."""
        obs = base_td.get(("agents", "observation"))
        amask = base_td.get(("agents", "action_mask"))

        obs_t = self._split(obs, 0.0)
        am_t = self._split(amask, False)

        # Fix padded slots: only noop (action 0) valid
        for tn in self.type_names:
            actual = self._type_counts.get(tn, 0)
            mc = self.max_per_type[tn]
            if actual < mc:
                am_t[tn][actual:] = False
                am_t[tn][actual:, 0] = True  # noop only

        # Active mask: alive = any non-noop action valid AND within actual count
        active_t: dict[str, torch.Tensor] = {}
        for tn in self.type_names:
            alive = am_t[tn][:, 1:].any(dim=-1)
            alive[self._type_counts.get(tn, 0) :] = False
            active_t[tn] = alive
            # Zero observations for inactive agents
            obs_t[tn][~alive] = 0.0

        # Participation scores: 1.0 for alive, 0.0 for dead/padded
        participation = {
            tn: active_t[tn].float().unsqueeze(-1) for tn in self.type_names
        }

        # Flat active mask
        flat_mask = torch.cat([active_t[tn] for tn in self.type_names])

        # Build output TD
        td = TensorDict({}, batch_size=torch.Size([]), device=self.device)

        for tn in self.type_names:
            td.set((tn, "observation"), obs_t[tn])
            td.set((tn, "action_mask"), am_t[tn])
            td.set((tn, "active_mask"), active_t[tn])
            td.set(f"{tn}_participation_score", participation[tn])

        # Entity-decomposed observations
        if self.entity_obs:
            entity_data = self._decompose_obs(obs_t, active_t)
            for tn in self.type_names:
                for key, val in entity_data[tn].items():
                    td.set((tn, key), val)

        td.set("active_mask", flat_mask)

        # State
        state = base_td.get("state", default=None)
        if state is not None:
            td.set("state", state)

        # Info
        for k in ("battle_won", "episode_limit"):
            v = base_td.get(("info", k), default=None)
            if v is not None:
                td.set(("info", k), v)

        # Done / terminated
        for k in ("done", "terminated"):
            v = base_td.get(k, default=None)
            if v is not None:
                td.set(k, v)

        # Reward: per-capita broadcast to alive agents (D2)
        if with_reward:
            # Base SMACv2Env puts reward at root level, not under ("agents", "reward")
            r = base_td.get("reward", default=None)
            if r is None:
                r = base_td.get(("agents", "reward"), default=None)
            if r is not None:
                # In SMAC, all agents get the same team reward
                team_r = self._team_reward_from_base(r)
                n_alive = flat_mask.sum().clamp(min=1).float()
                per_capita = team_r / n_alive
                for tn in self.type_names:
                    mc = self.max_per_type[tn]
                    rew = torch.zeros(mc, 1, device=self.device)
                    rew[active_t[tn]] = per_capita
                    td.set((tn, "reward"), rew)
            else:
                # Fallback: zero reward (should not happen in normal operation)
                for tn in self.type_names:
                    mc = self.max_per_type[tn]
                    td.set((tn, "reward"), torch.zeros(mc, 1, device=self.device))

        return td

    # ------------------------------------------------------------------ #
    #  EnvBase API                                                        #
    # ------------------------------------------------------------------ #

    def _reset(self, tensordict=None, **kwargs):
        self._base_td = self.base_env.reset(tensordict)
        obs = self._base_td.get(("agents", "observation"))
        self._detect_and_assign_types(obs)
        return self._build_output(self._base_td, with_reward=False)

    def _step(self, tensordict):
        flat_actions = self._merge_actions(tensordict)

        # Build input for base env from cached state + new actions
        base_input = self._base_td.clone()
        base_input.set(("agents", "action"), flat_actions)

        # Step the base env (adds "next" subtree)
        base_stepped = self.base_env.step(base_input)
        base_next = base_stepped.get("next")

        # Cache next state for the following step
        self._base_td = base_next.clone()

        return self._build_output(base_next, with_reward=True)

    def _set_seed(self, seed):
        if hasattr(self.base_env, "_set_seed"):
            self.base_env._set_seed(seed)

    def render(self, *args, **kwargs):
        """Forward rendering to the wrapped SMACv2 env when available."""
        render_fn = getattr(self.base_env, "render", None)
        if callable(render_fn):
            return render_fn(*args, **kwargs)
        raise AttributeError(
            "VariableSmacv2Env base_env does not expose render(); "
            "disable experiment rendering or use a render-capable SMACv2 env."
        )


# ==================================================================== #
#  BenchMARL Task / TaskClass                                           #
# ==================================================================== #


class Smacv2VariableTask(Task):
    """Enum for SMACv2 variable-composition tasks."""

    PROTOSS_10_VS_10 = None
    PROTOSS_10_VS_11 = None
    PROTOSS_20_VS_20 = None
    PROTOSS_20_VS_23 = None

    @staticmethod
    def associated_class():
        return Smacv2VariableClass

    @staticmethod
    def env_name() -> str:
        return "smacv2_variable"


class Smacv2VariableClass(TaskClass):
    """TaskClass for SMACv2 with per-unit-type variable-composition grouping."""

    # Keys consumed by this wrapper (popped before passing to SMACv2Env)
    _WRAPPER_KEYS = frozenset(
        {"max_stalkers", "max_zealots", "max_colossi", "proximity_threshold", "entity_obs"}
    )

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)

        # Extract wrapper-specific params
        max_stalkers = config.pop("max_stalkers", 8)
        max_zealots = config.pop("max_zealots", 8)
        max_colossi = config.pop("max_colossi", 4)
        config.pop("proximity_threshold", None)
        entity_obs = config.pop("entity_obs", False)

        max_per_type = OrderedDict(
            [
                ("stalker", max_stalkers),
                ("zealot", max_zealots),
                ("colossus", max_colossi),
            ]
        )

        def _make_env():
            from torchrl.envs.libs.smacv2 import SMACv2Env

            base_env = SMACv2Env(
                categorical_actions=True, seed=seed, device=device, **config
            )
            return VariableSmacv2Env(
                base_env=base_env,
                max_per_type=max_per_type,
                device=device,
                entity_obs=entity_obs,
            )

        return _make_env

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        current = env
        for _ in range(8):
            if isinstance(current, VariableSmacv2Env):
                current = current.base_env
                continue
            render_fn = getattr(current, "render", None)
            if callable(render_fn):
                return True
            next_env = getattr(current, "base_env", None)
            if next_env is None or next_env is current:
                break
            current = next_env
        return False

    def max_steps(self, env: EnvBase) -> int:
        return env.episode_limit

    def group_map(self, env: EnvBase) -> dict[str, list[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Composite | None:
        obs = env.observation_spec.clone()
        if "state" in obs.keys():
            return Composite(state=obs["state"])
        return None

    def action_mask_spec(self, env: EnvBase) -> Composite | None:
        obs = env.observation_spec.clone()
        spec = Composite()
        for tn in env.type_names:
            group_spec = obs[tn].clone()
            # Keep only action_mask
            for k in list(group_spec.keys()):
                if k != "action_mask":
                    del group_spec[k]
            spec[tn] = group_spec
        return spec

    def observation_spec(self, env: EnvBase) -> Composite:
        obs = env.observation_spec.clone()
        # Remove keys handled by other spec methods
        if "info" in obs.keys():
            del obs["info"]
        if "state" in obs.keys():
            del obs["state"]
        # Remove action_mask from per-type composites (handled by action_mask_spec)
        for tn in env.type_names:
            if "action_mask" in obs[tn].keys():
                del obs[tn]["action_mask"]
        return obs

    def info_spec(self, env: EnvBase) -> Composite | None:
        obs = env.observation_spec.clone()
        if "info" in obs.keys():
            return Composite(info=obs["info"])
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    def get_reward_sum_transform(self, env: EnvBase) -> Transform:
        group_map = self.group_map(env)
        in_keys = [(group, "reward") for group in group_map.keys()]
        out_keys = [(group, "episode_reward") for group in group_map.keys()]
        return RewardSum(in_keys=in_keys, out_keys=out_keys)

    @staticmethod
    def log_info(batch: TensorDictBase) -> dict[str, float]:
        logs = {}

        # Win rate
        done = batch.get(("next", "done")).squeeze(-1)
        battle_won = batch.get(("next", "info", "battle_won"), default=None)
        if battle_won is not None:
            logs["collection/info/win_rate"] = (
                battle_won[done].to(torch.float).mean().item()
            )

        episode_limit = batch.get(("next", "info", "episode_limit"), default=None)
        if episode_limit is not None:
            logs["collection/info/episode_limit_rate"] = (
                episode_limit[done].to(torch.float).mean().item()
            )

        # Active agent stats
        if "active_mask" in batch.keys():
            mask = batch["active_mask"]
            active_counts = mask.float().sum(dim=-1)
            logs["counters/num_active_agents_mean"] = active_counts.mean().item()
            logs["counters/num_active_agents_min"] = active_counts.min().item()
            logs["counters/num_active_agents_max"] = active_counts.max().item()

        for tn in PROTOSS_TYPE_NAMES:
            try:
                type_mask = batch.get((tn, "active_mask"), default=None)
                if type_mask is not None:
                    type_counts = type_mask.float().sum(dim=-1)
                    logs[f"counters/num_active_{tn}_mean"] = (
                        type_counts.mean().item()
                    )
            except Exception:
                pass

        return logs

    @staticmethod
    def env_name() -> str:
        return "smacv2_variable"
