import sys
import json

class DummyEnv:
    def __init__(self, config):
        # Store config variables as attributes
        self.n_agents = config.get("n_agents", 1)
        self.start_time = config.get("start_time", "")
        self.end_time = config.get("end_time", "")
        self.control_timedelta = config.get("control_timedelta", 900)
        self.busses = config.get("busses", [])
        self.cls = config.get("cls", "OpenDSSSolver")
        self.feeder_file = config.get("feeder_file", "ieee_13_dss/IEEE13Nodeckt.dss")
        self.loadshape_file = config.get("loadshape_file", "ieee_13_dss/annual_hourly_load_profile.csv")
        self.system_load_rescale_factor = config.get("system_load_rescale_factor", 1)
        self.num_vehicles = config.get("num_vehicles", 1)
        self.minutes_per_step = config.get("minutes_per_step", 15)
        self.max_charge_rate_kw = config.get("max_charge_rate_kw", 10)
        self.peak_threshold = config.get("peak_threshold", 500)
        self.vehicle_multiplier = config.get("vehicle_multiplier", 1)
        self.rescale_spaces = config.get("rescale_spaces", False)
        self.unserved_penalty = config.get("unserved_penalty", 1)

    def reset(self):
        # Return a dummy observation
        return {"obs": [0] * self.n_agents}

    def step(self, action):
        # Return dummy observation, reward, done, info
        return {
            "obs": [0] * self.n_agents,
            "reward": [0.0] * self.n_agents,
            "done": False,
            "info": {}
        }

env = None

for line in sys.stdin:
    try:
        msg = json.loads(line)
    except Exception:
        continue  # skip malformed lines

    cmd = msg.get("cmd")
    if cmd == "init":
        env = DummyEnv(msg["config"])
        print(json.dumps({"status": "ok"}), flush=True)
    elif cmd == "reset" and env is not None:
        print(json.dumps(env.reset()), flush=True)
    elif cmd == "step" and env is not None:
        action = msg.get("action")
        print(json.dumps(env.step(action)), flush=True)
    elif cmd == "close":
        print(json.dumps({"status": "closed"}), flush=True)
        break