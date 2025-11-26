import json
import os
import gc
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
IBM_TOKEN = os.getenv("IBM_TOKEN")
DWAVE_TOKEN = os.getenv("DWAVE_TOKEN")

app = Flask(__name__)
# Allow CORS for your frontend domain specifically or all
CORS(app, resources={r"/*": {"origins": "*"}})

# --- IMPORTS & SETUP ---
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
    print("⚠️ Qiskit not installed.")

DWAVE_AVAILABLE = False
try:
    from dwave.system import LeapHybridSampler
    import dimod
    DWAVE_AVAILABLE = True
except ImportError:
    print("⚠️ D-Wave not installed.")

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

class ISA_Transpiler_Wrapper:
    def __init__(self, backend, sampler):
        self.backend = backend
        self.sampler = sampler
    def run(self, pubs):
        transpiled_pubs = []
        for pub in pubs:
            # MEMORY FIX: Reduced optimization_level from 3 to 1
            # Level 3 is too heavy for free tier RAM
            if isinstance(pub, tuple):
                t_circuit = transpile(pub[0], self.backend, optimization_level=1)
                transpiled_pubs.append((t_circuit,) + pub[1:])
            else:
                transpiled_pubs.append(transpile(pub, self.backend, optimization_level=1))
        return self.sampler.run(transpiled_pubs)

def get_ibm_service(token):
    try:
        return QiskitRuntimeService(channel="ibm_quantum", token=token)
    except:
        try:
             return QiskitRuntimeService(channel="ibm_cloud", token=token)
        except:
             return QiskitRuntimeService(channel="ibm_quantum")

# --- SOLVERS ---

def solve_with_dwave(dist_matrix):
    print("🌊 D-Wave Leap...")
    sampler = LeapHybridSampler(token=DWAVE_TOKEN)
    import dwave_networkx as dnx
    G = nx.from_numpy_array(dist_matrix)
    route = dnx.traveling_salesperson(G, sampler)
    return route, "D-Wave Quantum Annealer"

def solve_with_ibm_cloud(dist_matrix):
    print("☁️ IBM Cloud...")
    # MEMORY FIX: Explicit garbage collection before heavy tasks
    gc.collect()
    
    tsp = Tsp(dist_matrix)
    qp = tsp.to_quadratic_program()
    
    service = get_ibm_service(IBM_TOKEN)
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Target: {backend.name}")
    
    optimizer = COBYLA(maxiter=1) 
    raw_sampler = IBMSampler(mode=backend)
    sampler = ISA_Transpiler_Wrapper(backend, raw_sampler)
    
    qaoa = QAOA(sampler, optimizer, reps=1)
    algo = MinimumEigenOptimizer(qaoa)
    result = algo.solve(qp)
    
    x = tsp.interpret(result)
    gc.collect() # Clean up after
    return list(x), f"Real QPU ({backend.name})", result.fval

def solve_with_local_sim(dist_matrix):
    print("💻 Local Simulator...")
    gc.collect()
    tsp = Tsp(dist_matrix)
    qp = tsp.to_quadratic_program()
    
    optimizer = COBYLA(maxiter=20) 
    sampler = LocalSampler()
    qaoa = QAOA(sampler, optimizer, reps=1)
    algo = MinimumEigenOptimizer(qaoa)
    result = algo.solve(qp)
    
    x = tsp.interpret(result)
    gc.collect()
    return list(x), "Local Qiskit Simulator", result.fval

@app.route('/solve', methods=['POST'])
def solve_tsp():
    print("\n--- Request ---")
    try:
        data = request.json
        locations = data.get('locations', [])
        num_nodes = len(locations)
        
        if num_nodes < 2:
            return jsonify({'error': 'Need 2+ locations'}), 400

        dist_matrix = calculate_distance_matrix(locations)
        
        # --- STRATEGY SELECTION ---
        
        # 1. D-WAVE (Best for Production/Stability)
        if DWAVE_AVAILABLE and DWAVE_TOKEN:
            try:
                route, method = solve_with_dwave(dist_matrix)
                return format_response(route, method, 0, dist_matrix)
            except Exception as e:
                print(f"D-Wave Failed: {e}")

        # 2. IBM CLOUD (Real Hardware)
        # MEMORY FIX: We previously limited this to 3.
        # UPDATE: Change 3 to 4 to allow 4-node problems on Cloud.
        if QISKIT_AVAILABLE and IBM_TOKEN and num_nodes <= 4:  # <--- CHANGE THIS FROM 3 TO 4
            try:
                route, method, energy = solve_with_ibm_cloud(dist_matrix)
                return format_response(route, method, energy, dist_matrix)
            except Exception as e:
                print(f"IBM Cloud Failed: {e}")

        # 3. LOCAL SIMULATOR (The RAM Killer)
        # KEEP THIS AT 3. Your free server definitely cannot simulate 4 nodes locally.
        if QISKIT_AVAILABLE and num_nodes <= 3: 
            try:
                route, method, energy = solve_with_local_sim(dist_matrix)
                return format_response(route, method, energy, dist_matrix)
            except Exception as e:
                print(f"Sim Failed: {e}")

        # 4. CLASSICAL FALLBACK (Safe Mode)
        print("⚠️ Using Classical Fallback (Memory Safe)")
        G = nx.from_numpy_array(dist_matrix)
        route = nx.approximation.greedy_tsp(G, source=0)
        return format_response(route, "Classical Greedy (Memory Safe)", 0, dist_matrix)

    except Exception as e:
        print(f"❌ Critical: {e}")
        return jsonify({'error': str(e)}), 500

def format_response(route, method, energy, dist_matrix):
    # Ensure start at 0
    if 0 in route:
        idx_0 = route.index(0)
        route = route[idx_0:] + route[:idx_0]
    
    # Calc energy if 0
    if energy == 0:
        for i in range(len(route)-1):
            energy += dist_matrix[route[i]][route[i+1]]
        energy += dist_matrix[route[-1]][route[0]]

    print(f"✅ Success: {method}")
    return jsonify({'route': route, 'method': method, 'energy': energy})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)