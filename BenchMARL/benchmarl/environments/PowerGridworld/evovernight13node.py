from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    agents: list = MISSING
    start_time: str = MISSING
    end_time: str = MISSING
    control_timedelta: int = MISSING
    busses: list = MISSING
    cls: str = MISSING
    feeder_file: str = MISSING
    loadshape_file: str = MISSING
    system_load_rescale_factor: float = MISSING
    num_vehicles: int = MISSING
    minutes_per_step: int = MISSING
    max_charge_rate_kw: float = MISSING
    peak_threshold: float = MISSING
    vehicle_multiplier: float = MISSING
    rescale_spaces: bool = MISSING
    unserved_penalty: float = MISSING