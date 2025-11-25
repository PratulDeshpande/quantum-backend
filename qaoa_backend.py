import json
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
from dotenv import load_dotenv

# --- SECURITY UPDATE ---
# Load secrets from the hidden .env file (for local dev)
load_dotenv()

# Get tokens from environment variables. 
# If not found, default to None (Simulation Mode).
IBM_TOKEN = os.getenv("IBM_TOKEN")
DWAVE_TOKEN = os.getenv("DWAVE_TOKEN")

# --- IMPORTS ---
app = Flask(__name__)
CORS(app)

# 1. Qiskit Setup (IBM)
QISKIT_AVAILABLE = False
try:
    from qiskit import transpile
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.applications import Tsp
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
    from qiskit.primitives import StatevectorSampler as LocalSampler
    QISKIT_AVAILABLE = True
except ImportError:
    print("⚠️ Qiskit not installed. IBM mode disabled.")

# 2. D-Wave Setup (Ocean)
DWAVE_AVAILABLE = False
try:
    from dwave.system import LeapHybridSampler
    import dimod
    DWAVE_AVAILABLE = True
except ImportError:
    print("⚠️ D-Wave Ocean not installed. D-Wave mode disabled.")

# --- HELPERS ---
def calculate_distance_matrix(locations):
    n = len(locations)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = locations[i]['lat'], locations[i]['lng']
                lat2, lon2 = locations[j]['lat'], locations[j]['lng']
                dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                matrix[i][j] = dist
    return matrix

# --- IBM WRAPPERS ---
class ISA_Transpiler_Wrapper:
    def __init__(self, backend, sampler):
        self.backend = backend
        self.sampler = sampler
    def run(self, pubs):
        transpiled_pubs = []
        for pub in pubs:
            if isinstance(pub, tuple):
                t_circuit = transpile(pub[0], self.backend, optimization_level=3)
                transpiled_pubs.append((t_circuit,) + pub[1:])
            else:
                transpiled_pubs.append(transpile(pub, self.backend, optimization_level=3))
        return self.sampler.run(transpiled_pubs)

def solve_with_ibm(dist_matrix, num_nodes):
    tsp = Tsp(dist_matrix)
    qp = tsp.to_quadratic_program()
    
    # Check for Cloud Token
    if IBM_TOKEN:
        print("☁️ Connecting to IBM Quantum Cloud...")
        try:
            try:
                service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
            except:
                service = QiskitRuntimeService(channel="ibm_cloud", token=IBM_TOKEN)

            backend = service.least_busy(operational=True, simulator=False)
            print(f"✅ IBM Target: {backend.name}")
            
            optimizer = COBYLA(maxiter=1)
            raw_sampler = IBMSampler(mode=backend)
            sampler = ISA_Transpiler_Wrapper(backend, raw_sampler)
            
            qaoa = QAOA(sampler, optimizer, reps=1)
            algorithm = MinimumEigenOptimizer(qaoa)
            result = algorithm.solve(qp)
            method = f"Real QPU ({backend.name})"
            
        except Exception as e:
            print(f"❌ IBM Cloud Error: {e}")
            raise e
    else:
        print("💻 Using Local Qiskit Simulator (No Token Found)...")
        optimizer = COBYLA(maxiter=100)
        sampler = LocalSampler()
        qaoa = QAOA(sampler, optimizer, reps=1)
        algorithm = MinimumEigenOptimizer(qaoa)
        result = algorithm.solve(qp)
        method = "Local Qiskit Simulator"

    x = tsp.interpret(result)
    return list(x), method, result.fval

# --- DWAVE SOLVER ---
def solve_with_dwave(dist_matrix):
    print("🌊 Connecting to D-Wave Leap...")
    # NOTE: D-Wave library automatically looks for DWAVE_API_TOKEN in os.environ
    # But passing it explicitly is safer for this setup.
    sampler = LeapHybridSampler(token=DWAVE_TOKEN)
    
    import dwave_networkx as dnx
    G = nx.from_numpy_array(dist_matrix)
    route = dnx.traveling_salesperson(G, sampler)
    
    energy = 0
    for i in range(len(route)-1):
        energy += dist_matrix[route[i]][route[i+1]]
    energy += dist_matrix[route[-1]][route[0]]
    
    return route, "D-Wave Quantum Annealer", energy

# --- MAIN ENDPOINT ---
@app.route('/solve', methods=['POST'])
def solve_tsp():
    print("\n--- New Optimization Request ---")
    data = request.json
    locations = data.get('locations', [])
    num_nodes = len(locations)
    dist_matrix = calculate_distance_matrix(locations)

    try:
        # PRIORITY 1: D-WAVE
        if DWAVE_AVAILABLE and DWAVE_TOKEN:
            try:
                route, method, energy = solve_with_dwave(dist_matrix)
                if 0 in route:
                     idx_0 = route.index(0)
                     route = route[idx_0:] + route[:idx_0]
                return jsonify({'route': route, 'method': method, 'energy': energy})
            except Exception as e:
                print(f"D-Wave Error: {e}")

        # PRIORITY 2: IBM / QISKIT
        if QISKIT_AVAILABLE and num_nodes <= 4:
            route, method, energy = solve_with_ibm(dist_matrix, num_nodes)
            if 0 in route:
                idx_0 = route.index(0)
                route = route[idx_0:] + route[:idx_0]
            return jsonify({'route': route, 'method': method, 'energy': energy})
        
        # PRIORITY 3: CLASSICAL FALLBACK
        else:
            print("⚠️ Classical Fallback (No tokens or too many nodes)")
            G = nx.from_numpy_array(dist_matrix)
            path = nx.approximation.greedy_tsp(G, source=0)
            if path[0] == path[-1]: path.pop()
            return jsonify({'route': path, 'method': 'Classical Greedy', 'energy': 0})

    except Exception as e:
        print(f"❌ Error: {e}")
        G = nx.from_numpy_array(dist_matrix)
        path = nx.approximation.greedy_tsp(G, source=0)
        if path[0] == path[-1]: path.pop()
        return jsonify({'route': path, 'method': 'Error Recovery', 'energy': 0})

if __name__ == '__main__':
    print("--- SECURE BACKEND ONLINE ---")
    app.run(port=5000, debug=True)