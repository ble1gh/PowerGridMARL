from datetime import datetime
import os
from typing import Tuple, Union, List, Dict

import numpy as np
import pandas as pd

from gridworld.log import logger
from gridworld.distribution_system.powerflow import PowerFlowSolver


DSS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')


class OpenDSSSolver(PowerFlowSolver):

    def __init__(
        self,
        feeder_file: str,
        loadshape_file: str,    
        system_load_rescale_factor: float = 1.0,
        load_noise_std: float = 0.0,
        load_forecast_noise_std: float = 0.0,
        load_actual_noise_std: float = 0.0,
        **kwargs
    ):
        """ An agent responsible for power flow computation given different 
        gen-load scenarios.

        Args:
            feeder_file: path to the dss file, relative to DSS_DATA_DIR.
            loadshape_file: path to the annual loadshape csv file, relative to 
                DSS_DATA_DIR.
            system_load_rescale_factor: scaling factor for base load
            load_noise_std: DEPRECATED — legacy single-noise parameter. If set
                and the new per-layer params are both 0, it is used as
                load_actual_noise_std for backward compatibility.
            load_forecast_noise_std: standard deviation of per-node Gaussian
                noise added to the CSV load coefficient to produce the
                *forecast* that agents / GNN observe.  Each load bus receives
                independent noise. Set to 0.0 for no forecast noise (default).
            load_actual_noise_std: standard deviation of per-node Gaussian noise
                added *on top of the forecast* to produce the *actual* load
                coefficient used in the power-flow solve.  Each load bus
                receives independent noise. Set to 0.0 for no actual noise
                (default).
        """

        super().__init__(**kwargs)

        import opendssdirect as dss
        self.dss = dss
        self.dss_data_path = os.path.join(DSS_DATA_DIR, feeder_file)
        self.dss.run_command("Redirect " + self.dss_data_path)

        self.system_load_rescale_factor = system_load_rescale_factor

        # ----- backward-compatible noise config -----
        # If the caller only provides the old load_noise_std and the new params
        # are at their defaults, treat load_noise_std as load_actual_noise_std.
        if load_noise_std > 0 and load_forecast_noise_std == 0 and load_actual_noise_std == 0:
            self.load_forecast_noise_std = 0.0
            self.load_actual_noise_std = load_noise_std
        else:
            self.load_forecast_noise_std = load_forecast_noise_std
            self.load_actual_noise_std = load_actual_noise_std
        # keep legacy attribute so nothing breaks downstream
        self.load_noise_std = load_noise_std

        load_profile_path = os.path.join(DSS_DATA_DIR, loadshape_file)
        self.annual_hourly_load_profile = np.genfromtxt(load_profile_path)
        if len(self.annual_hourly_load_profile) != 8760:
            print("Warning: The provided load shape file is not annual hourly ",
                "profile. Error might occur later")
             
        # Initialize class variable
        self.bus_voltages = {}
        self.load_bus_name, self.base_load = self._obtain_base_load_info()

        n_loads = len(self.load_bus_name)

        # Per-node load coefficient tracking (shape: (n_loads,) arrays)
        # base_load_coefficient: the raw CSV value for this hour (scalar, same for all)
        # forecast_load_coefficients: per-node forecast = base + forecast_noise  (what agents/GNN see)
        # actual_load_coefficients:  per-node actual  = forecast + actual_noise (used in power flow)
        self.base_load_coefficient = None
        self.forecast_load_coefficients = None   # shape (n_loads,)
        self.actual_load_coefficients = None      # shape (n_loads,)

        # Legacy scalar aliases (kept for backward compat with multiagent_env)
        self.normalized_load_coefficient = None
        self.actual_load_coefficient = None


    def _obtain_base_load_info(self) -> Tuple[list, np.ndarray]:
        """ Get base load info from the original OpenDSS project.

        Note, currently we only manipulate the PQ loads 
            (self.dss.Loads.Model() == 1).

        Returns:
          load_bus_name: A list of load bus names.
          base_load: Numpy array of (N, 2) dimension, where N is the system PQ 
            load number.
        """

        base_load = []
        load_bus_name = []
        load_buses_real = []

        ret = self.dss.Loads.First()
        while ret != 0:
            if self.dss.Loads.Model() == 1:
                base_load.append([self.dss.Loads.kW(), self.dss.Loads.kvar()])
                load_bus_name.append(self.dss.Loads.Name())
                # Handle bus name parsing (e.g. '634.1.2.3' -> '634')
                # Use CktElement interface to get the bus connection
                full_bus = self.dss.CktElement.BusNames()[0]
                load_buses_real.append(full_bus)
                
            ret = self.dss.Loads.Next()
        base_load = np.array(base_load)
        self.load_buses_real_names = load_buses_real

        return load_bus_name, base_load


    def get_bus_connectivity(self) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Extracts the adjacency data for different edge types.
        
        Returns:
            all_nodes: List of unique node names (e.g., '634.1', '671.2').
            edge_data: Dictionary mapping edge type to adjacency tensor (N, N, F).
                - 'line': F=3 [R, X, Length]
                - 'transformer': F=3 [R, X, Tap]
                - 'switch': F=1 [Status] (1=Closed, 0=Open)
        """
        
        # 1. Collect all raw node names from the circuit (BusName.Phase)
        # dss.Circuit.AllNodeNames() returns e.g., ['633.2', '634.1', ...]
        all_nodes = sorted(self.dss.Circuit.AllNodeNames())
        node_map = {name: i for i, name in enumerate(all_nodes)}
        n = len(all_nodes)
        
        # Initialize dense adjacency tensors for each type
        adj_line = np.zeros((n, n, 3), dtype=np.float32)       # R, X, Len
        adj_xfmr = np.zeros((n, n, 3), dtype=np.float32)       # R, X, Tap
        adj_switch = np.zeros((n, n, 1), dtype=np.float32)     # Status

        def add_edge_feat(adj, node1, node2, features):
            """Features: list or array matching adj's last dim"""
            if node1 in node_map and node2 in node_map:
                i, j = node_map[node1], node_map[node2]
                adj[i, j] = features
                adj[j, i] = features

        def parse_dss_bus_string(bus_str):
            parts = bus_str.split('.')
            bus_name = parts[0]
            phases = parts[1:]
            if phases:
                return [f"{bus_name}.{p}" for p in phases]
            existing_phases = [n.split('.')[-1] for n in all_nodes if n.startswith(f"{bus_name}.")]
            return [f"{bus_name}.{p}" for p in existing_phases]

        # 2. Iterate Lines
        flag = self.dss.Lines.First()
        while flag > 0:
            bus1_str = self.dss.Lines.Bus1()
            bus2_str = self.dss.Lines.Bus2()
            nodes1 = parse_dss_bus_string(bus1_str)
            nodes2 = parse_dss_bus_string(bus2_str)
            
            # Extract Line Features
            # We fetch the full impedance matrices (Ohms/unit length)
            # RMatrix/XMatrix return tuples. We reshape conceptually to (n_phases, n_phases)
            # and extract the diagonal element corresponding to the conductor K.
            n_phases = self.dss.Lines.Phases()
            r_matrix = self.dss.Lines.RMatrix()
            x_matrix = self.dss.Lines.XMatrix()
            length_val = self.dss.Lines.Length()
            
            # Switch Logic
            is_switch_def = self.dss.Lines.IsSwitch() or (length_val < 1e-6)
            
            if is_switch_def:
                # Switch Features: [Status]
                # 1.0 = Closed (Conducting), 0.0 = Open
                # Check if opened at either terminal (1 or 2) for all phases (0)
                is_open = self.dss.CktElement.IsOpen(1, 0) or self.dss.CktElement.IsOpen(2, 0)
                status = 0.0 if is_open else 1.0
                feat_vec = np.array([status], dtype=np.float32)
                
                # Switches are typically phase-to-phase specific (1-1, 2-2)
                # We do NOT cross-link switches unless specified, as they maintain isolation.
                min_len = min(len(nodes1), len(nodes2))
                for k in range(min_len):
                    add_edge_feat(adj_switch, nodes1[k], nodes2[k], feat_vec)
            else:
                # Line Features: [R(actual), X(actual), Length]
                # We allow Cross-Linking (Mutual Coupling) here.
                # Nodes on different phases are connected by mutual impedance.
                
                # Check matrix validity
                has_matrix = (len(r_matrix) == n_phases * n_phases)
                
                # Iterate all combinations of input/output phases
                for k1, node_src in enumerate(nodes1):
                    for k2, node_dst in enumerate(nodes2):
                        # Ensure we are within matrix bounds
                        if has_matrix and k1 < n_phases and k2 < n_phases:
                            # Flattened index: Row k1, Col k2
                            idx = k1 * n_phases + k2
                            r_val = r_matrix[idx] * length_val
                            x_val = x_matrix[idx] * length_val
                        elif k1 == k2:
                            # Fallback Diagonal (Positive Sequence)
                            r_val = self.dss.Lines.R1() * length_val
                            x_val = self.dss.Lines.X1() * length_val
                        else:
                            # Fallback Off-Diagonal (Mutual)
                            # If no matrix, assume 0 mutual coupling? Or approximation?
                            # Using 0 for safety if matrix missing.
                            r_val = 0.0
                            x_val = 0.0
                        
                        # Sparsify: Only add edge if there is non-trivial coupling (impedance is defined)
                        # Check for non-zero coupling (using small epsilon for float comparison)
                        if abs(r_val) > 1e-9 or abs(x_val) > 1e-9:
                            feat_vec = np.array([r_val, x_val, length_val], dtype=np.float32)
                            add_edge_feat(adj_line, node_src, node_dst, feat_vec)
                
            flag = self.dss.Lines.Next()

        # 3. Iterate Transformers
        flag = self.dss.Transformers.First()
        while flag > 0:
            bus_strs = self.dss.CktElement.BusNames()
            windings_nodes = [parse_dss_bus_string(b) for b in bus_strs]
            
            # Transformer Features: [R, X, Tap]
            self.dss.Transformers.Wdg(1)
            tap_val = self.dss.Transformers.Tap()
            r_val = self.dss.Transformers.R() * 0.01 
            x_val = self.dss.Transformers.Xhl() * 0.01
            
            feat_vec = np.array([r_val, x_val, tap_val], dtype=np.float32)
            
            if len(windings_nodes) >= 2:
                w1_nodes = windings_nodes[0]
                for w_other in windings_nodes[1:]:
                    min_len = min(len(w1_nodes), len(w_other))
                    for k in range(min_len):
                        add_edge_feat(adj_xfmr, w1_nodes[k], w_other[k], feat_vec)

            flag = self.dss.Transformers.Next()
        
        return all_nodes, {
            "line": adj_line,
            "transformer": adj_xfmr,
            "switch": adj_switch
        }

    def calculate_power_flow(
        self,
        p_controllable_consumed: dict = None,
        q_controllable_consumed: dict = None,
        current_time: str = None
    ) -> None:
        """ Calculate the power flow for the current time step.

        Args:
          p_controllable_consumed: dict of <bus, p>
          q_controllable_consumed: dict of <bus, q>
          current_time: string timestamp (convertible by pd.Timestamp)
        """

        # 1. Update the base load according the load shape file.

        current_time = pd.Timestamp(current_time)

        def get_hour_of_year(dt):
            """ Get hour of the year from pandas datetime object. Result is used 
            to retrieve load factor from the annual hourly load profile.
            """
            beginning_of_year = datetime(dt.year, 1, 1)
            return int((dt - beginning_of_year).total_seconds() // 3600)

        hour_of_year = get_hour_of_year(current_time)
        
        # ---- Per-node two-layer noise ----
        n_loads = len(self.load_bus_name)
        base_coeff = self.annual_hourly_load_profile[hour_of_year]  # scalar
        self.base_load_coefficient = base_coeff

        # Layer 1 — FORECAST noise (independent per load bus)
        # This is what agents / GNN observe.
        if self.load_forecast_noise_std > 0:
            forecast_noise = np.random.normal(0, self.load_forecast_noise_std, size=n_loads)
            self.forecast_load_coefficients = base_coeff + forecast_noise
        else:
            self.forecast_load_coefficients = np.full(n_loads, base_coeff)

        # Layer 2 — ACTUAL noise on top of forecast (independent per load bus)
        # This is what the power flow actually uses.
        if self.load_actual_noise_std > 0:
            actual_noise = np.random.normal(0, self.load_actual_noise_std, size=n_loads)
            self.actual_load_coefficients = self.forecast_load_coefficients + actual_noise
        else:
            self.actual_load_coefficients = self.forecast_load_coefficients.copy()

        # Legacy scalar aliases (mean across buses for backward compat)
        self.normalized_load_coefficient = float(np.mean(self.forecast_load_coefficients))
        self.actual_load_coefficient = float(np.mean(self.actual_load_coefficients))

        # Per-node actual coefficients broadcast over (n_loads, 2) base_load
        current_step_load = self.actual_load_coefficients[:, np.newaxis] * self.base_load * \
            self.system_load_rescale_factor

        # 2. Update the PQ from uncontrollable assets at this step.
        # TODO: get this using current_time and uncontrollable assets profile

        # 3. Update the PQ from controllable assets at this step and direct set 
        # the number in OpenDSS.
        if p_controllable_consumed is not None:
            for idx, load_bus_name in enumerate(self.load_bus_name):

                try:
                    controllable_p = p_controllable_consumed[load_bus_name]
                except KeyError:
                    controllable_p = 0.0

                try:
                    controllable_q = q_controllable_consumed[load_bus_name]
                except KeyError:
                    controllable_q = 0.0

                current_step_load[idx, 0] += controllable_p
                current_step_load[idx, 1] += controllable_q

        self._set_load(current_step_load)

        # 4. Calculate the power flow
        self.dss.run_command('Solve mode=snap')
        self._prepare_bus_voltages()
        self.losses = self.dss.Circuit.Losses()

    def get_losses(self) -> np.ndarray:
        """ Get the losses of the system.

        Returns: 
          losses: Numpy array of (N, 2)
        """
        # Note, the losses are already in kW and kvar.
        losses = np.array(self.losses)
        return losses

    def _set_load(self, current_step_load) -> None:
        """ Set current load to OpenDSS.

        Args:
          current_step_load: Numpy array of (N, 2) dimension, where N is the 
            system PQ load number.
        """

        ret = self.dss.Loads.First()
        load_idx = 0
        while ret != 0:
            if self.dss.Loads.Model() == 1:  # Note, currently we only manipulate the PQ loads.
                self.dss.Loads.kW(current_step_load[load_idx, 0])
                self.dss.Loads.kvar(current_step_load[load_idx, 1])
                load_idx += 1
            ret = self.dss.Loads.Next()


    def _prepare_bus_voltages(self) -> None:
        """ Parse OpenDSS voltages to bus_name -> voltage mapping.

        Returns: None
        """
        voltages = self.dss.Circuit.AllBusMagPu()
        voltage_bus_name = self.dss.Circuit.AllNodeNames()

        for idx, bus_name in enumerate(voltage_bus_name):
            self.bus_voltages[bus_name] = voltages[idx]


    def get_bus_voltages(self) -> dict:
        """Returns a dict of <bus, voltages> on the feeder."""
        return self.bus_voltages


    def get_bus_voltage_by_name(self, bus_name: any) -> Union[float, List[float]]:
        """Get bus voltage by name handling phases."""
        if bus_name in self.bus_voltages:
            return self.bus_voltages[bus_name]
            
        # Handle 3-phase buses (return list of voltages)
        multi_phase_voltages = []
        for phase in [".1", ".2", ".3"]:
            key = f"{bus_name}{phase}"
            if key in self.bus_voltages:
                multi_phase_voltages.append(self.bus_voltages[key])
        
        if multi_phase_voltages:
            if len(multi_phase_voltages) == 1:
                return multi_phase_voltages[0]
            return multi_phase_voltages
            
        # Handle letter phases
        if isinstance(bus_name, str) and bus_name[-1] in ['a','b','c']:
             conv = {'a':'.1', 'b':'.2', 'c':'.3'}
             new_name = bus_name[:-1] + conv[bus_name[-1]]
             if new_name in self.bus_voltages:
                 return self.bus_voltages[new_name]

        raise KeyError(f"Bus '{bus_name}' not found in voltage data.")

    def get_nodal_load_forecast(self, current_time: any = None) -> Dict[str, np.ndarray]:
        """Returns map of {bus_name: [kW, kvar]} for the current time step.

        Uses the per-node *forecast* load coefficients (which include per-bus
        forecast noise but NOT the additional actual-noise layer).  This is the
        view that agents / GNN should observe.

        If called before calculate_power_flow (forecast_load_coefficients is
        None), falls back to recomputing from the CSV for the given time.
        """
        if self.forecast_load_coefficients is not None:
            # Fast path: use the already-computed per-node forecast coefficients
            coeffs = self.forecast_load_coefficients  # shape (n_loads,)
        else:
            # Fallback: compute from scratch (before first power flow call)
            if current_time is None:
                raise ValueError("current_time must be provided when forecast "
                                 "coefficients have not been computed yet.")
            current_time = pd.Timestamp(current_time)
            beginning_of_year = datetime(current_time.year, 1, 1)
            hour_of_year = int((current_time - beginning_of_year).total_seconds() // 3600)
            if hour_of_year >= len(self.annual_hourly_load_profile):
                hour_of_year = hour_of_year % len(self.annual_hourly_load_profile)
            coeff = self.annual_hourly_load_profile[hour_of_year]
            coeffs = np.full(len(self.load_bus_name), coeff)

        # Per-node forecast loads: coeffs_i * base_load_i * rescale
        current_step_loads = coeffs[:, np.newaxis] * self.base_load * \
            self.system_load_rescale_factor

        # Map to buses
        bus_loads = {}
        for i, bus in enumerate(self.load_buses_real_names):
            if bus in bus_loads:
                bus_loads[bus] += current_step_loads[i]
            else:
                bus_loads[bus] = current_step_loads[i].copy()

        return bus_loads
        """Returns the voltages at the specified bus.  If the bus is multi-phase,
        returns a list of floats, otherwise a single float."""

        PHASE_MAP = {'a': '.1', 'b': '.2', 'c': '.3'}

        # Handle single-phase with letter notation (e.g., "634a")
        if bus_name[-1] in PHASE_MAP.keys():
            bus_name = bus_name.replace(bus_name[-1], PHASE_MAP[bus_name[-1]])
            if bus_name in self.bus_voltages:
                return self.bus_voltages[bus_name]
            else:
                raise KeyError(f"Bus '{bus_name}' not found in voltage data")
        else:
            # Check for all possible phases and collect them
            multi_phase_voltages = []
            for phase_ext in ['.1', '.2', '.3']:
                test_name = bus_name + phase_ext
                if test_name in self.bus_voltages:
                    multi_phase_voltages.append(self.bus_voltages[test_name])
            
            # If we found multiple phases, return them as a list
            if len(multi_phase_voltages) > 1:
                return multi_phase_voltages
            # If we found exactly one phase, return it as a single float
            elif len(multi_phase_voltages) == 1:
                return multi_phase_voltages[0]
            
            # If still no match, raise an error
            raise KeyError(f"Bus '{bus_name}' not found in voltage data. Available buses: {list(self.bus_voltages.keys())}")

def main():

    dss_config = {"feeder_file": "ieee_13_dss/IEEE13Nodeckt.dss",
                  "loadshape_file": "ieee_13_dss/annual_hourly_load_profile.csv"}

    odss = OpenDSSSolver(**dss_config)
    current_time = pd.Timestamp("01-01-2021 05:00:00")
    odss.calculate_power_flow(current_time=current_time)
    v = odss.get_bus_voltages()

    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(v)


if __name__ == '__main__':
    main()

