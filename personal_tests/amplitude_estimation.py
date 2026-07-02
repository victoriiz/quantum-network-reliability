import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def run_simulation(p_edge1, p_edge2):
    """
    Models a simple 2-node, 2-parallel-edge network reliability problem.
    Returns estimated reliability using a quantum circuit simulaton.
    """
    theta1 = 2 * np.arcsin(np.sqrt(p_edge1))
    theta2 = 2 * np.arcsin(np.sqrt(p_edge2))

    qc = QuantumCircuit(3, 1)

    # state prep
    qc.ry(theta1, 0)
    qc.ry(theta2, 1)
    qc.barrier()

    # reachability oracle
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 2)
    qc.x(2)

    # clean up edge qubits by restoring states
    qc.x(0)
    qc.x(1)
    qc.barrier()

    qc.measure(2, 0)
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)

    shots = 1000
    job = simulator.run(compiled, shots=shots)
    res = job.result()
    counts = res.get_counts()

    successes = counts.get('1',0)
    estimated_R = successes / shots

    return qc, estimated_R

def main():
    p1, p2 = 0.80, 0.50
    circ, quant_res = run_simulation(p1, p2)

    # classical baseline to compare to
    class_res = 1 - ((1-p1) * (1-p2))

    print(f"classical calculated reliability: {class_res*100:.2f}%")
    print(f"quantum simulated relaibility: {quant_res * 100:.2f}%")

    print("\nQuantum Circuit Schematic")
    print(circ.draw(output='text'))

main()

