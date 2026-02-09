import os
import numpy as np
import pandas as pd
from pprint import pprint
import sys

# Ensure gridworld is in path
sys.path.append(os.getcwd())

from PowerGridworld.gridworld.distribution_system.opendss import OpenDSSSolver

def test_connectivity():
    print("Initializing OpenDSSSolver...")
    # Use the default IEEE 13 node test case
    dss_config = {
        "feeder_file": "ieee_13_dss/IEEE13Nodeckt.dss",
        "loadshape_file": "ieee_13_dss/annual_hourly_load_profile.csv"
    }

    try:
        odss = OpenDSSSolver(**dss_config)
    except Exception as e:
        print(f"Failed to initialize OpenDSSSolver: {e}")
        return

    print("\nExtracting bus connectivity...")
    all_nodes, adj_dict = odss.get_bus_connectivity()
    
    print(f"\nTotal Nodes Found: {len(all_nodes)}")
    print(f"Node List (first 10): {all_nodes[:10]}")
    
    print("\nAdjacency Data:")
    for edge_type, adj_tensor in adj_dict.items():
        print(f"  Type: {edge_type}")
        print(f"    Shape: {adj_tensor.shape}")
        
        # Count non-zero edges (since it's dense, count rows where sum > 0)
        # Note: adj_tensor is (N, N, F).
        # We check if any feature is non-zero
        nonzero_mask = np.any(adj_tensor != 0, axis=2)
        num_edges = np.sum(nonzero_mask) / 2 # Undirected, so divide by 2
        print(f"    Active Connections (Undirected): {int(num_edges)}")
        
        if int(num_edges) > 0:
            # Print a sample edge
            rows, cols = np.where(nonzero_mask)
            # Pick first unique pair
            for r, c in zip(rows, cols):
                if r < c: # Upper triangle
                    node_a = all_nodes[r]
                    node_b = all_nodes[c]
                    features = adj_tensor[r, c]
                    print(f"    Sample Edge: {node_a} <--> {node_b}")
                    print(f"      Features: {features}")
                    break
        print("-" * 30)

if __name__ == "__main__":
    test_connectivity()
