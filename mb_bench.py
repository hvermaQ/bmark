#many body benchmark
#for systems with ground state energies known in the thermodynamic limit
# typically we need cases where the energy converges to a finite value in the thermodynamic limit
# random seeds yeild the error bars at individual points
#structure: MB_benchmark class.
#inputs: {nqbits}, {depths}, {nshots}, [noise param] = (infidelity, n_errors), rnds
#if n_errors  = -n, then n errors per gate
#selections: Anstaz, topology, compression, 2x hardwares, gateset 
#next version: noise type, model choice

from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN #Gates
from qat.core import Observable, Term, Batch #Hamiltonian
import numpy as np
#from qat.plugins import ScipyMinimizePlugin
#import pickle
from multiprocessing import Pool
import matplotlib.pyplot as plt


from qat.plugins import Junction
from qat.core import Result
from scipy.optimize import minimize
from qat.core.plugins import AbstractPlugin
from qat.fermion import SpinHamiltonian
from qat.qpus import get_default_qpu


class MB_benchmark:
    def __init__(self, nqbits, depths,  nshots, noise_params, rnds, ansatz, observe):
        self.nqbits = nqbits
        self.nshots = nshots
        self.noise_params = noise_params
        self.rnds = rnds
        self.depths = depths
        self.ansatz = ansatz
        self.error = 0 #should be the final result for metric in the thermodynamic limit
        self.resources = 0 #should be the final result for resources, given the inputs
        self.observe = observe

    def gen_circ_RYA(self, args):
        nqbt = args[0]
        ct = args[1]
        #Variational circuit can only be constructed using the program framework
        qprog = Program()
        qbits = qprog.qalloc(nqbt)
        #variational parameters used for generating gates (permutation of [odd/even, xx/yy/zz])
        #variational parameters used for generating gates
        angs = [qprog.new_var(float, 'a_%s'%i) for i in range(nqbt*(2*ct+1))]
        ang = iter(angs)
        #circuit
        for it in range(ct):
            for q_index in range(nqbt):
                RY(next(ang))(qbits[q_index])
            for q_index in range(nqbt):
                if not q_index%2 and q_index <= nqbt-2:
                    CSIGN(qbits[q_index],qbits[q_index+1])
            for q_index in range(nqbt):
                RY(next(ang))(qbits[q_index])
            for q_index in range(nqbt):
                if q_index%2 and q_index <= nqbt-2:
                    CSIGN(qbits[q_index],qbits[q_index+1])
            CSIGN(qbits[0],qbits[nqbt-1])
            if it==(ct-1):
                for q_index in range(nqbt):
                    RY(next(ang))(qbits[q_index])
        #circuit
        circuit = qprog.to_circ()
        return(circuit)

    def gen_circ_HVA(self, args):
        nqbt = args[0]
        ct = args[1]
        #Variational circuit can only be constructed using the program framework
        qprog = Program()
        qbits = qprog.qalloc(nqbt)
        #variational parameters used for generating gates (permutation of [odd/even, xx/yy/zz])
        ao = [qprog.new_var(float, 'ao_%s'%i) for i in range(ct)]
        bo = [qprog.new_var(float, 'bo_%s'%i) for i in range(ct)]
        co = [qprog.new_var(float, 'co_%s'%i) for i in range(ct)]
        ae = [qprog.new_var(float, 'ae_%s'%i) for i in range(ct)]
        be = [qprog.new_var(float, 'be_%s'%i) for i in range(ct)]
        ce = [qprog.new_var(float, 'ce_%s'%i) for i in range(ct)]
        for q_index in range(nqbt):
            X(qbits[q_index])
        for q_index in range(nqbt):
            if not q_index%2 and q_index <= nqbt-1:
                H(qbits[q_index])
        for q_index in range(nqbt):
            if not q_index%2 and q_index <= nqbt-2:
                CNOT(qbits[q_index],qbits[q_index+1])
        for it in range(ct):
            for q_index in range(nqbt): #odd Rzz
                if q_index%2 and q_index <= nqbt-2:
                    CNOT(qbits[q_index],qbits[q_index+1])
                    RZ(ao[it-1]/2)(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
            for q_index in range(nqbt): #odd Ryy
                if q_index%2 and q_index <= nqbt-2:
                    RZ(np.pi/2)(qbits[q_index])
                    RZ(np.pi/2)(qbits[q_index+1])
                    H(qbits[q_index])
                    H(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
                    RZ(bo[it-1]/2)(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
                    H(qbits[q_index])
                    H(qbits[q_index+1])
                    RZ(-np.pi/2)(qbits[q_index])
                    RZ(-np.pi/2)(qbits[q_index+1])
            for q_index in range(nqbt): #odd Rxx
                if q_index%2 and q_index <= nqbt-2:
                    H(qbits[q_index])
                    H(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
                    RZ(co[it-1]/2)(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
                    H(qbits[q_index])
                    H(qbits[q_index+1])
            for q_index in range(nqbt): #even Rzz
                if not q_index%2 and q_index <= nqbt-2:
                    CNOT(qbits[q_index],qbits[q_index+1])
                    RZ(ae[it-1]/2)(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
            for q_index in range(nqbt): #even Ryy
                if not q_index%2 and q_index <= nqbt-2:
                    RZ(np.pi/2)(qbits[q_index])
                    RZ(np.pi/2)(qbits[q_index+1])
                    H(qbits[q_index])
                    H(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
                    RZ(be[it-1]/2)(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
                    H(qbits[q_index])
                    H(qbits[q_index+1])
                    RZ(-np.pi/2)(qbits[q_index])
                    RZ(-np.pi/2)(qbits[q_index+1])
            for q_index in range(nqbt): #even Rxx
                if not q_index%2 and q_index <= nqbt-2:
                    H(qbits[q_index])
                    H(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
                    RZ(ce[it-1]/2)(qbits[q_index+1])
                    CNOT(qbits[q_index],qbits[q_index+1])
                    H(qbits[q_index])
                    H(qbits[q_index+1])
        #circuit
        circuit = qprog.to_circ()
        return(circuit)

    # Assuming stack.submit is a method, wrapper below avoiding any object instantitation
    # Wrapper function to handle multiple arguments for map
    #for a selected problem size and selected depth
    # i : dummy iterable
    def submit_job(self, sel_size, sel_dep, noise_params, ansatz):
        if ansatz == "RYA":
            circ = self.gen_circ_RYA(sel_size, sel_dep)
        elif ansatz == "HVA":
            circ = self.gen_circ_HVA(sel_size, sel_dep)
        F = noise_params[0]
        n_errs = noise_params[1]
        mat = self.observe.get_matrix()
        optimizer = Opto()
        qpu_ideal = get_default_qpu()
        stack = optimizer | GaussianNoise(F, n_errs, mat) | qpu_ideal  # noisy stack
        job = circ.to_job(observable=self.observe, nbshots=self.nshots)
        result = stack.submit(job).value
        return result
        #return the result
        return result
    
    # Simple program running wrapper: VQE stack
    # reverts to ideal run if F<0
    # F: infidelity, layers: relevant layer list
    def run_prog_map(layers, qubits, Fidelity, randoms):
        problem_set = [(nqb, dep) for nqb in nqbits for dep in depths]
        # Parallel run handled here
        with Pool() as p:  # Number of processes can be adjusted
            result_async = p.map_async(submit_job, [(layer, qubits, Fidelity, y) for layer in layers for y in range(randoms)])  # Passing arguments as a tuple
            #ress = p.map(submit_job, [(layers, qubits, Fidelity, y) for y in range(randoms)])
            # Get the results from the asynchronous map
            ress = result_async.get()  # This will block until all results are available
        # Now, we want to extract the minimum result for each layer
        min_results_per_layer = []
        # Iterate through each layer and find the minimum result for each layer
        for i, layer in enumerate(layers):
            layer_results = ress[i * randoms: (i + 1) * randoms]  # Extract the results for the current layer
            min_results_per_layer.append(np.min(layer_results))  # Find the minimum result for this layer
        return(min_results_per_layer)  # Return the minimum result

    def result(self):
        res = benchmark(nqbits, nshots, noise_param, rnds)
        self.value = res
        return self.value