#many body benchmark
#for systems with ground state energies known in the thermodynamic limit
# typically we need cases where the energy converges to a finite value in the thermodynamic limit
# random seeds yeild the error bars at individual points
#structure: MB_benchmark class.
#inputs: {nqbits}, {depths}, {nshots}, [noise param] = (infidelity, n_errors), rnds, observable key, ansatz key, hardware_key
#if n_errors  = -n, then n errors per gate, if n_errors >0, then n errors per circuit
#selections: observable, Anstaz, topology, compression, 2x hardwares, gateset 
#next version: noise type, noise model choice
#default hardware is non so algorithmic efficiency and resources are calculated, does not call hardware routine
#algo_Resources = shots * number of pauli strings * number of gates * number of iterations

from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN #Gates
from qat.core import Observable, Term, Batch #Hamiltonian
import numpy as np
#from qat.plugins import ScipyMinimizePlugin
#import pickle
from multiprocessing import Pool
import matplotlib.pyplot as plt

from qat.fermion import SpinHamiltonian
from qat.qpus import get_default_qpu
from opto_gauss import Opto, GaussianNoise

import seaborn as sns
from scipy import stats
from tqdm import tqdm

class MB_benchmark:
    def __init__(self, nqbits, depths,  nshots, noise_params, rnds, ansatz, observe, thermal_value, thermal_size, hardware=None):
        self.nqbits = nqbits
        self.nshots = nshots
        self.noise_params = noise_params
        self.rnds = rnds
        self.depths = depths
        self.ansatz = ansatz
        self.projected_value = 0 #should be the final result for the gs in the thermodynamic limit
        self.projected_results = None #should be the wrapped final result in ther thermodynamic limit
        self.error = 0 #should be the final result for metric i.e. error in the thermodynamic limit
        self.error_bars = 0 #should be the final result for error bars in the thermodynamic limit
        self.algo_resources = 0 #should be the final result for resources, given the inputs
        self.observe = observe #observable key "Heisenberg"
        self.algo_efficiency = 0 #should be the final result for algorithm efficiency
        self.hardware_resources = 0 #should be the final result for hardware resources
        self.hardware_efficiency = 0 #should be the final result for hardware efficiency
        self.hardware = hardware
        self.thermodynamic_limit = thermal_value #exact value at thermodynamic limit
        self.thermal_size = thermal_size #effective thermodynamic limit i.e. problem size say 1000
        self.gates = None
        self.sim_results = 0 #averages at quoted problem sizes
        self.sim_variance = 0 #variance at quoted problem sizes
        self.optimizer = Opto()
        self.qpu_ideal = get_default_qpu()
        self.gateset()
        self.bench_run()

    #gateset for counting algorithmic resources
    def gateset(self):
        #gateset for counting gates to introduce noise through Gaussian noise plugin
        one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
        two_qb_gateset = ['CNOT', 'CSIGN']  
        self.gates = one_qb_gateset + two_qb_gateset
        print("Gates!")

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
    def submit_job(self, sel_size, sel_dep, i):
        if self.ansatz == "RYA":
            circ = self.gen_circ_RYA(sel_size, sel_dep)
        elif self.ansatz == "HVA":
            circ = self.gen_circ_HVA(sel_size, sel_dep)
        F = self.noise_params[0]
        n_errs = self.noise_params[1]
        obse = self.create_observable(sel_size)
        obse_class = SpinHamiltonian(nqbits=obse.nbqbits, terms=obse.terms)
        pauls = len(obse.terms)
        obs_mat = obse_class.get_matrix()
        stack = self.optimizer | GaussianNoise(F, n_errs, obs_mat) | self.qpu_ideal  # noisy stack
        job = circ.to_job(observable=self.observe, nbshots=self.nshots)
        gate_size = sum([job.circuit.count(yt) for yt in self.gates()])
        result = stack.submit(job)
        value = result.value
        iterations = result.meta_data("n_steps")
        return (value, iterations, gate_size, pauls)
    
    # Run parallel jobs for n selected sizes and corresponding depths with rnds random seeds
    def run_parallel_jobs(self):
        print("Parallelizing")
        # Ensure sizes and depths are of the same length
        if len(self.nqbits) != len(self.depths):
            raise ValueError("Sizes and depths must have the same length")

        problem_set = list(zip(self.nqbits, self.depths))  # Elementwise tuples
        # Parallel run handled here
        with Pool() as pool:  # Number of processes can be adjusted
            result_async = pool.map_async(
                self.submit_job,
                [(size, depth, rnd) for (size, depth) in problem_set for rnd in range(self.rnds)]
            )
            # Get the results from the asynchronous map
            results = result_async.get()  # This will block until all results are available
        print("Parallel done")
        # Now, we want to extract the minimum and variance result for each size
        avg_results_per_size = []
        variance_results_per_size = []
        n_pauls = []
        n_gates = []
        n_iterations = []
        for size in self.nqbits:
            size_results = [results[i] for i, (s, _) in enumerate(problem_set) if s == size]
            values = [result[0] for result in size_results]
            iterations = [result[1] for result in size_results]
            pauls = [result[3] for result in size_results]
            gates = [result[2] for result in size_results]

            avg_results_per_size.append((size, np.avg(values)))  # Find the average of result for this size
            variance_results_per_size.append((size, np.var(values)))  # Calculate the variance for this size
            n_pauls.append((size, np.mean(pauls)))  # Average number of pauli terms
            n_gates.append((size, np.mean(gates)))  # Average number of gates
            n_iterations.append((size, np.mean(iterations))) #average iterations
        return (avg_results_per_size, variance_results_per_size, n_pauls, n_gates, n_iterations)  # Return the results
    

    def random_walk_extrapolation(self, problem_sizes, values, errors, target_size, 
                                num_walks=100000, confidence_levels=[0.6827, 0.9545, 0.9973]):
        """
        Implement random walk method for error extrapolation to larger problem sizes
        
        Parameters:
        -----------
        problem_sizes : array-like
            The problem sizes for which we have data
        values : array-like
            The corresponding metric values
        errors : array-like
            The errors/uncertainties associated with each value
        target_size : float
            The problem size to extrapolate to
        num_walks : int, optional
            Number of random walks to simulate (default: 100,000)
        confidence_levels : list, optional
            Confidence levels for interval calculation
            
        Returns:
        --------
        dict
            Dictionary containing extrapolated value, error, confidence intervals, 
            and simulated trajectories
        """
        # Sort data by problem size
        sorted_indices = np.argsort(problem_sizes)
        problem_sizes = np.array(problem_sizes)[sorted_indices]
        values = np.array(values)[sorted_indices]
        errors = np.array(errors)[sorted_indices]
        
        # Linear model for metric values
        value_slope, value_intercept, _, _, _ = stats.linregress(problem_sizes, values)
        extrapolated_value = value_slope * target_size + value_intercept
        
        # Model how errors scale with problem size (power law)
        log_sizes = np.log(problem_sizes)
        log_errors = np.log(errors)
        error_slope, error_intercept, _, _, _ = stats.linregress(log_sizes, log_errors)
        extrapolated_error = np.exp(error_intercept) * (target_size ** error_slope)
        
        # Calculate differences between consecutive values for bounding
        value_diffs = np.diff(values)
        
        # Initialize array for walk results
        all_walks = np.zeros(num_walks)
        
        # Perform random walks
        for i in tqdm(range(num_walks), desc="Simulating random walks"):
            # Start each walk from the last observed point
            current_size = problem_sizes[-1]
            current_value = values[-1]
            
            # Steps to reach target (adaptive based on size difference)
            steps_needed = max(3, min(20, int(np.log2(target_size / current_size) * 3) + 1))
            size_ratio = (target_size / current_size) ** (1/steps_needed)
            
            # Generate sequence of sizes to evaluate
            eval_sizes = [current_size * (size_ratio ** j) for j in range(1, steps_needed + 1)]
            
            # Walk through sizes
            for next_size in eval_sizes:
                # Get most recent difference as baseline for bounds
                recent_diff = abs(value_diffs[-1] if len(value_diffs) > 0 else values[-1] * 0.1)
                
                # Scale bounds based on problem size
                scaling_factor = (current_size / next_size) ** 0.5  # Square root scaling
                bounded_diff = recent_diff * scaling_factor
                
                # Random step within bounds
                step = np.random.uniform(-bounded_diff, bounded_diff)
                next_value = current_value + step
                
                # Update for next iteration
                current_value = next_value
                current_size = next_size
                
                # Stop if we've reached or passed target
                if current_size >= target_size:
                    break
            
            # Store final value
            all_walks[i] = current_value
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for level in confidence_levels:
            alpha = (1 - level) / 2
            lower = np.percentile(all_walks, alpha * 100)
            upper = np.percentile(all_walks, (1 - alpha) * 100)
            confidence_intervals[level] = (lower, upper)
        
        # Return results
        return {
            'extrapolated_value': extrapolated_value,
            'extrapolated_error': extrapolated_error,
            'random_walk_median': np.median(all_walks),
            'confidence_intervals': confidence_intervals,
            'all_walks': all_walks
        }

    def benchmark(self):
        print("Benchmarking Main")
        avg_results, variance_results, avg_pauls, avg_gates, avg_iterations = self.run_parallel_jobs()
        # Extract the minimum result and variance for each size
        sizes, values = zip(*avg_results)
        _, errors = zip(*variance_results)
        
        self.sim_results = values
        self.sim_variance = errors

        # Perform random walk extrapolation
        extrapolation_results = self.random_walk_extrapolation(sizes, values, errors, target_size=self.thermal_size)
        self.projected_results = extrapolation_results
        self.projected_value = extrapolation_results['extrapolated_value']
        self.error = np.abs(self.projected_value - self.thermodynamic_limit)
        self.error_bars = extrapolation_results['extrapolated_error']
        
        # Calculate algorithmic resources
        for i in range(len(self.nqbits)):
            self.algo_resources += avg_pauls[i] * avg_gates[i] * avg_iterations[i] * self.nshots

        #return self.projected_value, self.error, self.algo_resources

    def hardware_resource(self, algo_resources):
        #insert algorithmic resources to hardware resources conversion
        #assume same energy consumption for single and two qubit gates
        h_w0 =  6e9  # Frequency [Hz]  (Ghz ranges)
        gam = 1  # Gamma [kHz]
        t_1qb = 25* 10**(-9) #single qubit gate duration in nanosecs
        A_db = 50 #attenuation in DB
        A = 10**(A_db/10) #absolute attenuation
        T_qb = 6e-3  # Qubit Temperature [K]   (6e-3, 10)
        T_ext = 300 #external temperature in K
        E_1qb = h_w0 * (np.pi*np.pi)/(4*gam*t_1qb)
        #total heat evacuated    
        E_cool = (T_ext - T_qb) * A * E_1qb * algo_resources / T_qb
        return E_cool

    def plot_results(self):
        problem_sizes = self.nqbits
        values = self.sim_results
        errors = self.sim_variance
        target_size = self.thermal_size
        results = self.projected_results

        """
        Plot extrapolation results with confidence intervals
        """
        plt.figure(figsize=(12, 8))
        
        # Plot data points with error bars
        plt.errorbar(problem_sizes, values, yerr=errors, fmt='o', color='blue', 
                    label='Data with error bars', markersize=8, capsize=5)
        
        # Plot linear trend
        x_range = np.linspace(min(problem_sizes), target_size * 1.1, 100)
        value_slope, value_intercept, _, _, _ = stats.linregress(problem_sizes, values)
        plt.plot(x_range, value_slope * x_range + value_intercept, 'b--', 
                label='Linear extrapolation')
        
        # Mark extrapolated point
        plt.plot(target_size, results['extrapolated_value'], 'bs', markersize=8)
        
        # Random walk result with error bar
        plt.errorbar([target_size], [results['random_walk_median']], 
                    yerr=[results['extrapolated_error']], fmt='ro', markersize=10, 
                    capsize=5, label=f'Random walk estimate with error')
        
        # Add confidence intervals
        for level, (lower, upper) in results['confidence_intervals'].items():
            alpha = 0.2 + 0.1 * list(results['confidence_intervals'].keys()).index(level)
            plt.fill_between([target_size-0.05*target_size, target_size+0.05*target_size], 
                            [lower, lower], [upper, upper], alpha=alpha, color='red',
                            label=f'{level*100:.1f}% CI')
        
        # Add vertical line at target size
        plt.axvline(x=target_size, color='k', linestyle='--', alpha=0.5,
                    label=f'Target size: {target_size}')
        
        # Add histogram inset
        ax_inset = plt.axes([0.6, 0.2, 0.25, 0.25])
        sns.histplot(results['all_walks'], kde=True, ax=ax_inset)
        ax_inset.set_title('Distribution of estimates')
        ax_inset.axvline(x=results['random_walk_median'], color='r', linestyle='--')
        
        plt.xlabel('Problem Size')
        plt.ylabel('Value')
        plt.title('Random Walk Error Extrapolation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('extrapolation_results.pdf', bbox_inches='tight')
        plt.show()

    def bench_run(self):
        print("STart benchmarking")
        self.benchmark()
        print("Done numerical")
        self.plot_results()
        print("Done plotting")
        if self.hardware is not None:
            self.hardware_resources = self.hardware_resource(self.algo_resources)
            self.hardware_efficiency = self.error / self.hardware_resources
            print("Hardware efficiency = %f" %self.algo_efficiency)
        self.algo_efficiency = self.error / self.algo_resources
        print("Algorithmic efficiency = %f" %self.algo_efficiency)
        print("Projected value = %f" %self.projected_value)     
        print("Error using the exact value = %f" %self.error)