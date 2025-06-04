from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    start_time: str = MISSING
    end_time: str = MISSING
    control_timedelta: int = MISSING