#!/usr/bin/env python3
"""Diagnostic: verify _sample_active_agents + _reset produce correct PV counts
using the REAL environment (not a simulation)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BenchMARL"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PowerGridworld"))

import torch
import yaml
from benchmarl.environments.PowerGridworldVariable.common import (
    PowerGridworldVariableClass,
)

# ---- Load VPP task config -------------------------------------------------
conf_path = os.path.join(
    os.path.dirname(__file__),
    "BenchMARL/benchmarl/conf/task/PowerGridworldVariable/evovernight13node_vpp.yaml",
)
with open(conf_path) as f:
    raw = yaml.safe_load(f)
raw.pop("defaults", None)

task = PowerGridworldVariableClass.__new__(PowerGridworldVariableClass)
task.config = raw

print("=== Raw config values ===")
print(f"min_PVs={raw.get('min_PVs')}, max_PVs={raw.get('max_PVs')}")
print(f"min_EVs={raw.get('min_EVs')}, max_EVs={raw.get('max_EVs')}")
print(f"min_Storage={raw.get('min_Storage')}, max_Storage={raw.get('max_Storage')}")
print(f"PV_busses={raw.get('PV_busses')}")
print(f"allow_multiple_agents_per_node={raw.get('allow_multiple_agents_per_node')}")

# ---- Create the real env ---------------------------------------------------
env_fn = task.get_env_fun(num_envs=1, continuous_actions=True, seed=42, device="cpu")
env = env_fn()

print("=== Agent registry ===")
print(f"Total agents: {len(env.agents)}")
for pfx in ("EV-", "PV-", "Storage-"):
    names = [a.name for a in env.agents if a.name.startswith(pfx)]
    print(f"  {pfx[:-1]:>7}: {len(names)}  {names}")

pv_in_idx = [(n, i) for n, i in env._agent_name_to_idx.items() if n.startswith("PV-")]
print(f"\nPV entries in _agent_name_to_idx: {len(pv_in_idx)}")

# ---- Test 1: _sample_active_agents only (no reset) -------------------------
print("\n=== Test 1: _sample_active_agents × 50 ===")
ev_c, pv_c, st_c = [], [], []
for _ in range(50):
    env._sample_active_agents()
    m = env.active_mask
    n_ev = sum(1 for n, i in env._agent_name_to_idx.items() if n.startswith("EV-") and m[i])
    n_pv = sum(1 for n, i in env._agent_name_to_idx.items() if n.startswith("PV-") and m[i])
    n_st = sum(1 for n, i in env._agent_name_to_idx.items() if n.startswith("Storage-") and m[i])
    ev_c.append(n_ev)
    pv_c.append(n_pv)
    st_c.append(n_st)

print(f"EV   min={min(ev_c)} max={max(ev_c)} mean={sum(ev_c) / len(ev_c):.1f}  (config 10-20)")
print(f"PV   min={min(pv_c)} max={max(pv_c)} mean={sum(pv_c) / len(pv_c):.1f}  (config 12-12)")
print(f"Stor min={min(st_c)} max={max(st_c)} mean={sum(st_c) / len(st_c):.1f}  (config 5-8)")
if any(c != 12 for c in pv_c):
    print("*** BUG: PV count != 12 in _sample_active_agents! ***")

# ---- Test 2: full _reset (applies random_arrival EV override) ---------------
print("\n=== Test 2: full _reset × 10 ===")
for trial in range(10):
    env._reset()
    m = env.active_mask
    n_ev = sum(1 for n, i in env._agent_name_to_idx.items() if n.startswith("EV-") and m[i])
    n_pv = sum(1 for n, i in env._agent_name_to_idx.items() if n.startswith("PV-") and m[i])
    n_st = sum(1 for n, i in env._agent_name_to_idx.items() if n.startswith("Storage-") and m[i])
    total = n_ev + n_pv + n_st
    print(f"  Reset {trial:2d}: EV={n_ev:2d}  PV={n_pv:2d}  Stor={n_st}  total={total}")

# ---- Test 3: simulate a few steps to see EV connection mask updates ---------
print("\n=== Test 3: step-by-step EV mask evolution (5 steps) ===")
td = env._reset()
for step in range(5):
    # Build dummy actions (0.5 for everyone)
    action_td = td.clone()
    for t in env.agent_types:
        n = len(env._type_agent_names[t])
        action_td.set((t, "action"), torch.full((n, 1), 0.5))
    td = env._step(action_td)
    m = env.active_mask
    n_ev = sum(1 for n, i in env._agent_name_to_idx.items() if n.startswith("EV-") and m[i])
    n_pv = sum(1 for n, i in env._agent_name_to_idx.items() if n.startswith("PV-") and m[i])
    n_st = sum(1 for n, i in env._agent_name_to_idx.items() if n.startswith("Storage-") and m[i])
    print(f"  Step {step}: EV={n_ev:2d}  PV={n_pv:2d}  Stor={n_st}  total={n_ev + n_pv + n_st}")

print("\nDone.")
