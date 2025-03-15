#many body benchmark
#for systems with ground state energies known in the thermodynamic limit
# typically we need cases where the energy converges to a finite value in the thermodynamic limit
# random seeds yeild the error bars at individual points
#structure: MB_benchmark class.
#inputs: {nqbits}, {depths}, {nshots}, [noise param] = (infidelity, n_errors), rnds, observable key, ansatz key
#if n_errors  = -n, then n errors per gate, if n_errors >0, then n errors per circuit
#selections: observable, Anstaz, topology, compression, 2x hardwares, gateset 
#next version: noise type, noise model choice

from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN #Gates
from qat.core import Observable, Term, Batch #Hamiltonian
from qat.core import get_default_qpu
import numpy as np
#from qat.plugins import ScipyMinimizePlugin
#import pickle
from multiprocessing import Pool
import matplotlib.pyplot as plt

from qat.fermion import SpinHamiltonian
from qat.qpus import get_default_qpu
from opto_gauss import Opto, GaussianNoise

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

    def create_observable(self, nqbts):
        if self.observe == "Heisenberg":
            #Instantiation of Hamiltoniian
            heisen = Observable(nqbts)
            #Generation of Heisenberg Hamiltonian
            for q_reg in range(nqbts-1):
                heisen += Observable(nqbts, pauli_terms = [Term(1., typ, [q_reg,q_reg + 1]) for typ in ['XX','YY','ZZ']])  
            obs = heisen
        return obs

    def exact_Result(self, obs):
        obs_class = SpinHamiltonian(nqbits=obs.nbqbits, terms=obs.terms)
        obs_mat = obs_class.get_matrix()
        eigvals, _ = np.linalg.eigh(obs_mat)
        g_energy = eigvals[0]
        return g_energy

    # Assuming stack.submit is a method, wrapper below avoiding any object instantitation
    # Wrapper function to handle multiple arguments for map
    #for a selected problem size and selected depth
    # i : dummy iterable
    def submit_job(self, sel_size, sel_dep, noise_params, i):
        if self.ansatz == "RYA":
            circ = self.gen_circ_RYA(sel_size, sel_dep)
        elif self.ansatz == "HVA":
            circ = self.gen_circ_HVA(sel_size, sel_dep)
        F = noise_params[0]
        n_errs = noise_params[1]
        obse = self.create_observable(sel_size)
        obse_class = SpinHamiltonian(nqbits=obse.nbqbits, terms=obse.terms)
        obs_mat = obse_class.get_matrix()
        optimizer = Opto()
        qpu_ideal = get_default_qpu()
        stack = optimizer | GaussianNoise(F, n_errs, obs_mat) | qpu_ideal  # noisy stack
        job = circ.to_job(observable=self.observe, nbshots=self.nshots)
        result = stack.submit(job).value
        return result
    
    # Run parallel jobs for n selected sizes and corresponding depths with rnds random seeds
    def run_parallel_jobs(self):
        # Ensure sizes and depths are of the same length
        if len(self.nqbits) != len(self.depths):
            raise ValueError("Sizes and depths must have the same length")

        problem_set = list(zip(self.nqbits, self.depths))  # Elementwise tuples
        # Parallel run handled here
        with Pool() as pool:  # Number of processes can be adjusted
            result_async = pool.map_async(
                self.submit_job,
                [(size, depth, self.noise_params, rnd) for (size, depth) in problem_set for rnd in range(self.rnds)]
            )
            # Get the results from the asynchronous map
            results = result_async.get()  # This will block until all results are available

        # Now, we want to extract the minimum result for each size and depth
        min_results_per_problem = []
        variance_results_per_problem = []
        for i, (size, depth) in enumerate(problem_set):
            problem_results = results[i * self.rnds: (i + 1) * self.rnds]  # Extract the results for the current problem
            min_results_per_problem.append((size, depth, np.min(problem_results)))  # Find the minimum result for this problem
            variance_results_per_problem.append((size, depth, np.var(problem_results)))  # Calculate the variance for this problem

        return (min_results_per_problem, variance_results_per_problem)  # Return the minimum results
    
    def result(self):
        res = benchmark(nqbits, nshots, noise_param, rnds)
        self.value = res
        return self.value