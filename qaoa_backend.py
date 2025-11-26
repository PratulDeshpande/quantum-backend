import json
import os
import gc
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

# --- JOB STORAGE (Simple In-Memory Database) ---
# Stores results: { "job_id": { "status": "pending" | "completed" | "failed", "data": ... } }
jobs_db = {}

# --- IMPORTS ---
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
    print("⚠️ Qiskit not installed.")

DWAVE_AVAILABLE = False
try:
    from dwave.system import LeapHybridSampler
    import dimod
    import dwave_networkx as dnx
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

# --- WORKER FUNCTION (Runs in Background) ---
def background_worker(job_id, locations):
    try:
        num_nodes = len(locations)
        dist_matrix = calculate_distance_matrix(locations)
        
        # STRATEGY 1: D-WAVE
        if DWAVE_AVAILABLE and DWAVE_TOKEN:
            try:
                print(f"Job {job_id}: Sending to D-Wave...")
                sampler = LeapHybridSampler(token=DWAVE_TOKEN)
                G = nx.from_numpy_array(dist_matrix)
                route = dnx.traveling_salesperson(G, sampler)
                save_result(job_id, route, "D-Wave Quantum Annealer", dist_matrix)
                return
            except Exception as e:
                print(f"D-Wave Failed: {e}")

        # STRATEGY 2: IBM ONE-SHOT
        if QISKIT_AVAILABLE and IBM_TOKEN and num_nodes <= 4:
            try:
                print(f"Job {job_id}: Sending to IBM Cloud...")
                tsp = Tsp(dist_matrix)
                qp = tsp.to_quadratic_program()
                
                # Fix: Convert constrained QP to unconstrained QUBO
                converter = QuadraticProgramToQubo()
                qubo = converter.convert(qp)
                operator, offset = qubo.to_ising()
                
                ansatz = QAOAAnsatz(operator, reps=1)
                
                service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
                backend = service.least_busy(operational=True, simulator=False)
                
                # Optimization level 0 saves backend RAM
                t_circuit = transpile(ansatz, backend, optimization_level=0)
                sampler = IBMSampler(mode=backend)
                
                # Run Job
                pub = (t_circuit, [np.random.rand(ansatz.num_parameters)])
                job = sampler.run([pub])
                result = job.result() # This blocks thread, but not main server!
                
                pub_result = result[0]
                counts = pub_result.data.meas.get_counts()
                best_bitstring = max(counts, key=counts.get)
                
                x = np.array([int(bit) for bit in best_bitstring])
                try:
                    route = tsp.interpret(x)
                except:
                    route = list(range(len(dist_matrix)))
                
                save_result(job_id, list(route), f"Real QPU ({backend.name})", dist_matrix)
                return
            except Exception as e:
                print(f"IBM Failed: {e}")

        # STRATEGY 3: FALLBACK
        print(f"Job {job_id}: Using Classical Fallback")
        G = nx.from_numpy_array(dist_matrix)
        route = nx.approximation.greedy_tsp(G, source=0)
        save_result(job_id, route, "Classical Greedy (Fallback)", dist_matrix)

    except Exception as e:
        print(f"Job {job_id} Failed: {e}")
        jobs_db[job_id] = {"status": "failed", "error": str(e)}

def save_result(job_id, route, method, dist_matrix):
    # Normalize route
    if 0 in route:
        idx_0 = route.index(0)
        route = route[idx_0:] + route[:idx_0]
    
    # Calculate energy
    energy = 0
    for i in range(len(route)-1):
        energy += dist_matrix[route[i]][route[i+1]]
    energy += dist_matrix[route[-1]][route[0]]
    
    jobs_db[job_id] = {
        "status": "completed",
        "result": {
            "route": route,
            "method": method,
            "energy": energy
        }
    }
    print(f"Job {job_id} Completed!")

# --- ENDPOINTS ---

@app.route('/solve', methods=['POST'])
def start_solve():
    data = request.json
    locations = data.get('locations', [])
    
    if len(locations) < 2: return jsonify({'error': 'Need 2+ locations'}), 400

    # Generate ID and start background thread
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {"status": "pending"}
    
    thread = threading.Thread(target=background_worker, args=(job_id, locations))
    thread.start()
    
    return jsonify({"job_id": job_id, "status": "pending"})

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs_db.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
        
    return jsonify(job)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)