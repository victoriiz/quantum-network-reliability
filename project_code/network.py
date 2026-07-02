import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator, StatePreparation, EstimationCircuit

# DESIGNING GENERAL GRAPH REPRESENTATION
def create_sample_net():
    """
    Defines bridge network: 4 nodes, 5 edges
    """
    G = nx.Graph()

    # each edge: (node A, node B, success prob)
    edges = [
        (0, 1, 0.90),
        (0, 2, 0.85),
        (1, 2, 0.70),
        (1, 3, 0.80),
        (2, 3, 0.95)
    ]
    for u, v, p in edges:
        G.add_edge(u, v, probability=p)
    return G

G = create_sample_net()
print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# MAKING GENERALIZED STATE PREP. CIRCUIT
def build_state_prep(graph):
    """
    Creates circuit block where each qubit represents an edge,
    rotatated according to its success prob.
    """
    num_edges = graph.number_of_edges()

    qc = QuantumCircuit(num_edges, name="StatePrep")

    for idx, (u, v, data) in enumerate(graph.edges(data=True)):
        p = data['probability']
        theta = 2 * np.archsin(np.sqrt(p))
        qc.ry(theta, idx)
    
    return qc

# quick verification
initial_circ = build_state_prep(G)
print(initial_circ.draw(output='text'))

# PATH-BASED REACHABILITY ORACLE
def build_oracle(graph):
    """
    Constructs an oracle circuit that identifies connected graph states.
    Flips an indicator qubit if the active edges form a spanning network.
    """
    num_edges = graph.number_of_edges()
    qc = QuantumCircuit(num_edges + 1, name="Oracle")

    connected_states = []
    for i in range(2**num_edges):
        state_bin = format(i, f'0{num_edges}b')

        #build temp. subgraph consisting only of working edges
        sub_G = nx.Graph()
        sub_G.add_nodes_from(graph.nodes())
        for idx, (u, v) in enumerate(graph.edges()):
            if state_bin[num_edges - 1 - idx] == '1':
                sub_G.add_edge(u, v)
        
        # if the subgraph if fully connected, save this state string
        if nx.is_connected(sub_G):
            connected_states.append(state_bin)
    
    for state in connected_states:
        control_qubits = []
        for idx, bit in enumerate(reversed(state)):
            if bit == '0':
                qc.x(idx)
            control_qubits.append(idx)
    
    qc.mcx(control_qubits, num_edges)

    for idx, bit in enumerate(reversed(state)):
        if bit == '0':
            qc.x(idx)

    return qc

oracle_circuit = build_oracle(G)
print("Oracle built.")

# AMPLITUDE ESTIMATION - treat state prep. (A) and oracle (O) as combined operator
def run_ae(graph, precision_qubits=4):
    """
    Integrates the state prep. and oracle into an Amplitude Estimation routine.
    precision_qubits controls how fine-grained the estimation accuracy is.
    """
    num_edges = graph.number_of_edges()
    state_prep = build_state_prep(graph)
    oracle = build_oracle(graph)

    num_qubits = num_edges + 1 + precision_qubits
    qc = QuantumCircuit(num_qubits, precision_qubits)

    for q in range(precision_qubits):
        qc.h(q)

    target_indices = list(range(precision_qubits, precision_qubits+num_edges))
    qc.compose(state_prep, qubits=target_indices, inplace=True)

    # phase estimation loop: simulated like:
    # Grover operator Q = - StatePrep * Reflection_theta * State_Prep^-1 * Oracle

    simu = AerSimulator(method='statevector')

    return qc

print("pipeline set up successfully.")

# VALIDATING AND BENCHMARKING
def classical_reliability_baseline(graph, trials=50000):
    """ Compute network reliability using classical Monte Carlo sampling. """
    successful_trials = 0
    for _ in range(trials):
        sub_G = nx.Graph()
        sub_G.add_nodes_from(graph.nodes())

        for u, v, data in graph.edges(data=True):
            if np.random.rand() < data['probability']:
                sub_G.add_edge(u, v)
            
            if nx.is_connected(sub_G):
                successful_trials += 1
    
    return successful_trials / trials

def execute_benchmark(graph):
    classical_R = classical_reliability_baseline(graph)
    num_edges = G.number_of_edges()
    state_prep = build_state_prep(graph)
    oracle = build_oracle(graph)

    full_verification_circuit = QuantumCircuit(num_edges + 1, 1)
    full_verification_circuit.compose(state_prep, qubits=range(num_edges), inplace=True)
    full_verification_circuit.compose(oracle, qubits=range(num_edges + 1), inplace=True)
    full_verification_circuit.measure(num_edges, 0)

    sim = AerSimulator()
    result = sim.run(full_verification_circuit, shots=20000).result()
    counts = result.get_counts()
    quantum_R = counts.get('1', 0) / 20000

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Classical Monte Carlo Reliability Prediction: {classical_R * 100:.2f}%")
    print(f"Quantum Circuit Algorithmic Prediction:       {quantum_R * 100:.2f}%")

execute_benchmark(G)