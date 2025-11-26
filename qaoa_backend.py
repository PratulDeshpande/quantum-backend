import json
import os
import time
import threading
import uuid
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
CORS(app, resources={r"/*": {"origins": "*"}})

# --- IMPORTS (Safe Loading) ---
QISKIT_AVAILABLE = False
try:
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_optimization.applications import Tsp
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
    from qiskit.primitives import StatevectorSampler as LocalSampler
    QISKIT_AVAILABLE = True
except ImportError:
    print("⚠️ Qiskit not installed. (Cloud/Local Quantum disabled)")

DWAVE_AVAILABLE = False
try:
    from dwave.system import LeapHybridSampler
    import dimod
    import dwave_networkx as dnx
    DWAVE_AVAILABLE = True
except ImportError:
    print("⚠️ D-Wave not installed. (Cloud Annealer disabled)")

# --- JOB STORE ---
jobs_db = {}

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

# --- SOLVER FUNCTIONS ---

def run_dwave_solver(dist_matrix):
    if not (DWAVE_AVAILABLE and DWAVE_TOKEN):
        raise Exception("D-Wave Token missing or library not installed.")
    print("🌊 Executing on D-Wave Leap...")
    sampler = LeapHybridSampler(token=DWAVE_TOKEN)
    G = nx.from_numpy_array(dist_matrix)
    route = dnx.traveling_salesperson(G, sampler)
    return route, "D-Wave Quantum Annealer"

def run_ibm_cloud_solver(dist_matrix):
    if not (QISKIT_AVAILABLE and IBM_TOKEN):
        raise Exception("IBM Token missing or Qiskit not installed.")
    print("☁️ Executing on IBM Quantum Cloud...")
    
    tsp = Tsp(dist_matrix)
    qp = tsp.to_quadratic_program()
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    operator, offset = qubo.to_ising()
    
    # 1. Build Circuit
    ansatz = QAOAAnsatz(operator, reps=1)
    ansatz.measure_all() # <--- CRITICAL FIX: Adds measurements to register 'meas'
    
    # 2. Connect
    try:
        service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
    except:
        service = QiskitRuntimeService(channel="ibm_cloud", token=IBM_TOKEN)
        
    backend = service.least_busy(operational=True, simulator=False)
    
    # 3. Transpile & Run
    t_circuit = transpile(ansatz, backend, optimization_level=0)
    sampler = IBMSampler(mode=backend)
    
    pub = (t_circuit, [np.random.rand(ansatz.num_parameters)])
    job = sampler.run([pub])
    result = job.result()
    
    # 4. Decode
    # Now 'meas' exists because we called measure_all()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get)
    
    x = np.array([int(bit) for bit in best_bitstring])
    try:
        route = tsp.interpret(x)
    except:
        route = list(range(len(dist_matrix))) # Noise Fallback
        
    return list(route), f"Real QPU ({backend.name})"

def run_local_simulator(dist_matrix):
    if not QISKIT_AVAILABLE:
        raise Exception("Qiskit not installed.")
    print("💻 Executing on Local Qiskit Simulator...")
    
    tsp = Tsp(dist_matrix)
    qp = tsp.to_quadratic_program()
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    operator, offset = qubo.to_ising()
    
    ansatz = QAOAAnsatz(operator, reps=1)
    ansatz.measure_all() # <--- CRITICAL FIX: Adds measurements
    
    sampler = LocalSampler()
    t_circuit = transpile(ansatz, backend=None, optimization_level=0)
    
    job = sampler.run([(t_circuit, [np.random.rand(ansatz.num_parameters)])])
    result = job.result()
    
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get)
    
    x = np.array([int(bit) for bit in best_bitstring])
    route = tsp.interpret(x)
    return list(route), "Local Qiskit Simulator"

# --- MAIN WORKER LOGIC ---
def background_worker(job_id, locations, solver_mode):
    try:
        num_nodes = len(locations)
        dist_matrix = calculate_distance_matrix(locations)
        route = []
        method = ""
        
        print(f"Job {job_id}: Starting Mode [{solver_mode}]")

        # STRICT MODE SELECTION
        try:
            if solver_mode == 'cloud':
                try:
                    route, method = run_dwave_solver(dist_matrix)
                except Exception as dwave_err:
                    print(f"D-Wave skipped: {dwave_err}")
                    route, method = run_ibm_cloud_solver(dist_matrix)
                    
            elif solver_mode == 'local':
                route, method = run_local_simulator(dist_matrix)
                
            else:
                raise Exception(f"Unknown solver mode: {solver_mode}")

        except Exception as quantum_err:
            print(f"Quantum failed ({solver_mode}): {quantum_err}")
            print("⚠️ Falling back to Classical Greedy")
            G = nx.from_numpy_array(dist_matrix)
            route = nx.approximation.greedy_tsp(G, source=0)
            method = f"Classical Fallback (Quantum Failed)"

        # Calculate Energy & Format
        if 0 in route:
            idx_0 = route.index(0)
            route = route[idx_0:] + route[:idx_0]
            
        energy = 0
        for i in range(len(route)-1):
            energy += dist_matrix[route[i]][route[i+1]]
        energy += dist_matrix[route[-1]][route[0]]
        
        jobs_db[job_id] = {
            "status": "completed",
            "result": { "route": route, "method": method, "energy": energy }
        }
        print(f"Job {job_id} Finished via {method}")
        
    except Exception as e:
        print(f"Critical Job Error: {e}")
        jobs_db[job_id] = {"status": "failed", "error": str(e)}

# --- ENDPOINTS ---
@app.route('/solve', methods=['POST'])
def start_solve():
    data = request.json
    locations = data.get('locations', [])
    solver_mode = data.get('solver_mode', 'local') 
    
    if len(locations) < 2: return jsonify({'error': 'Need 2+ locations'}), 400

    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {"status": "pending"}
    
    thread = threading.Thread(target=background_worker, args=(job_id, locations, solver_mode))
    thread.start()
    
    return jsonify({"job_id": job_id, "status": "pending"})

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs_db.get(job_id)
    if not job: return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)