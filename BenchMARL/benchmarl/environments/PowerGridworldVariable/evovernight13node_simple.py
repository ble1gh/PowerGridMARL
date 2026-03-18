from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    agents: list = MISSING
    start_time: str = MISSING
    end_time: str = MISSING
    control_timedelta: int = MISSING
    
    # Variable agent configuration
    min_EVs: int = MISSING
    max_EVs: int = MISSING
    min_PVs: int = MISSING
    max_PVs: int = MISSING
    min_Storage: int = MISSING
    max_Storage: int = MISSING
    
    # Possible buses for each agent type
    EV_busses: list = MISSING
    PV_busses: list = MISSING
    Storage_busses: list = MISSING
    
    # Whether multiple agents can be at same node
    allow_multiple_agents_per_node: bool = MISSING
    
    # Grid Model Parameters
    cls: str = MISSING
    feeder_file: str = MISSING
    loadshape_file: str = MISSING
    system_load_rescale_factor: float = MISSING
    load_noise_std: float = MISSING  # DEPRECATED legacy noise param
    load_forecast_noise_std: float = MISSING  # Per-node forecast noise std
    load_actual_noise_std: float = MISSING  # Per-node actual noise std (on top of forecast)
    include_load_in_agent_obs: bool = MISSING  # If True, append normalized load coefficient to agent obs

    # EV agent parameters
    num_vehicles: int = MISSING
    minutes_per_step: int = MISSING
    max_charge_rate_kw: float = MISSING
    peak_threshold: float = MISSING
    vehicle_multiplier: float = MISSING
    rescale_spaces: bool = MISSING
    unserved_penalty: float = MISSING
    reward_scale: float = MISSING
    urgency_coef: float = MISSING

    # Random EV arrival parameters
    random_arrival: bool = MISSING
    arrival_probability: float = MISSING
    min_charge_duration_min: int = MISSING
    max_charge_duration_min: int = MISSING
    
    # Global reward penalty scaling parameters
    power_loss_penalty: float = MISSING
    voltage_penalty: float = MISSING
    cooperative_voltage: bool = MISSING
    load_2norm_penalty: float = MISSING
    tracking_reward_penalty: float = MISSING

    # PV agent parameters
    pv_profile_csv: str = MISSING
    pv_scaling_factor: float = MISSING
    pv_profile_noise_std: float = MISSING  # Std dev of Gaussian noise on PV profile
    pv_grid_aware: bool = MISSING
    min_pv_scaling_factor: float = MISSING
    max_pv_scaling_factor: float = MISSING
    
    # Energy Storage agent parameters
    storage_range_min: float = MISSING
    storage_range_max: float = MISSING
    initial_storage_mean: float = MISSING
    initial_storage_std: float = MISSING
    charge_efficiency: float = MISSING
    discharge_efficiency: float = MISSING
    max_power: float = MISSING

    # Signal tracking parameters
    signal_tracking: bool = MISSING
    track_total_load: bool = MISSING
    setpoint: float = MISSING

    # VPP (Virtual Power Plant) reward parameters
    vpp_reward: bool = MISSING
    vpp_setpoint: float = MISSING
    vpp_reward_penalty: float = MISSING