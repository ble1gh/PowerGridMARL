from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    agents: list = MISSING
    EVs_per_node: int = MISSING
    PVs_per_node: int = MISSING
    Storage_per_node: int = MISSING
    start_time: str = MISSING
    end_time: str = MISSING
    control_timedelta: int = MISSING
    EV_busses: list = MISSING
    PV_busses: list = MISSING
    Storage_busses: list = MISSING
    cls: str = MISSING
    feeder_file: str = MISSING
    loadshape_file: str = MISSING
    system_load_rescale_factor: float = MISSING
    
    # EV agent parameters
    num_vehicles: int = MISSING
    minutes_per_step: int = MISSING
    max_charge_rate_kw: float = MISSING
    peak_threshold: float = MISSING
    vehicle_multiplier: float = MISSING
    rescale_spaces: bool = MISSING
    unserved_penalty: float = MISSING
    reward_scale: float = MISSING
    
    # PV agent parameters
    pv_profile_csv: str = MISSING
    pv_scaling_factor: float = MISSING
    pv_grid_aware: bool = MISSING
    
    # Energy Storage agent parameters
    storage_range_min: float = MISSING
    storage_range_max: float = MISSING
    initial_storage_mean: float = MISSING
    initial_storage_std: float = MISSING
    charge_efficiency: float = MISSING
    discharge_efficiency: float = MISSING
    max_power: float = MISSING