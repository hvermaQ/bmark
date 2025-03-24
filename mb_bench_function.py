# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:54:14 2025

@author: cqtv201

features to add:
    1. test for the noiseless case before proceeding to optimization
    2. samples on the next epsilon curve drawn near the last one
"""

from qat.core import Observable, Term, Batch #Hamiltonian
from qat.qpus import get_default_qpu
from qat.fermion import SpinHamiltonian
import numpy as np
from multiprocessing import Pool, cpu_count
from circ_gen import gen_circ_RYA, gen_circ_HVA
import matplotlib.pyplot as plt

from opto_gauss_mod import Opto, GaussianNoise

import seaborn as sns
from scipy import stats
from tqdm import tqdm

plt.rcParams.update({'font.size': 16})  # Set global font size

optimizer = Opto()
qpu = get_default_qpu()

#gateset for counting algorithmic resources
def gateset():
    #gateset for counting gates to introduce noise through Gaussian noise plugin
    one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    two_qb_gateset = ['CNOT', 'CSIGN']  
    gates = one_qb_gateset + two_qb_gateset
    return gates

def create_observable(type, nqbts):
    if type == "Heisenberg":
        #Instantiation of Hamiltoniian
        heisen = Observable(nqbts)
        #Generation of Heisenberg Hamiltonian
        for q_reg in range(nqbts-1):
            heisen += Observable(nqbts, pauli_terms = [Term(1., typ, [q_reg,q_reg + 1]) for typ in ['XX','YY','ZZ']])  
        obs = heisen
    return obs

def exact_Result(obs):
    obs_class = SpinHamiltonian(nqbits=obs.nbqbits, terms=obs.terms)
    obs_mat = obs_class.get_matrix()
    eigvals, _ = np.linalg.eigh(obs_mat)
    g_energy = eigvals[0]
    return g_energy

def submit_job_wrapper(args):
    """Wrapper function for parallel job submission that recreates required objects"""
    nqbts, dep, rnd, ans, ham, n_params = args
    #everything based on i
    if ans == "RYA":
        circuit = gen_circ_RYA((nqbts, dep))
    elif ans == "HVA":
        circuit = gen_circ_HVA((nqbts, dep))
    obss = create_observable(ham, nqbts)
    job = circuit.to_job(observable=obss, nbshots=0)
    obs_mat = obss.to_matrix().A
    # Create fresh instances in worker process
    stack = optimizer | GaussianNoise(n_params[0], n_params[1], obs_mat) | qpu
    result = stack.submit(job)
    print(result.meta_data["n_steps"])
    return (nqbts, result.value, result.meta_data["n_steps"])

def run_parallel_jobs(problem_set, rnds, ansatz, observe, noise_params):
    print("Parallelizing")
    # Prepare arguments for parallel processing
    job_args = []
    for i in range(len(problem_set)):
        for rnd in range(rnds):
            # Only pass serializable data
            job_args.append((
                problem_set[i][0], 
                problem_set[i][1],
                rnd,
                ansatz,
                observe,
                noise_params
            ))
    
    num_processes = min(cpu_count(), len(job_args))
    print(f"Using {num_processes} processes")
    
    #with Pool(processes=num_processes, initializer=MB_benchmark.init_worker) as pool:
    with Pool(processes=num_processes) as pool:
        # Use imap with tqdm for progress tracking
        result_async = pool.map_async(submit_job_wrapper, job_args)
        results = result_async.get()
            
    print("Parallel done")
    
    # Process results
    avg_results_per_size = []
    variance_results_per_size = []
    n_iterations = []
    
    # Group results by problem size
    for i in range(len(problem_set)):
        size_results = [(val, iters) for nqb, val, iters in results if  nqb == problem_set[i][0]]
        values = [r[0] for r in size_results]
        iterations = [int(r[1]) for r in size_results]
        
        avg_results_per_size.append(np.mean(values))
        variance_results_per_size.append(np.var(values))
        n_iterations.append(np.sum(iterations))

    return (avg_results_per_size, variance_results_per_size, n_iterations)

def run_serial_jobs(problem_set, rnds, ansatz, observe, noise_params):
    print("Running jobs serially")
    #MB_benchmark.init_worker()
    
    results = []
    for i in range(len(problem_set)):
        for rnd in range(rnds):
            # Only pass serializable data
            args = (
                problem_set[i][0], 
                problem_set[i][1],
                rnd,
                ansatz,
                observe,
                noise_params
            )
            # Use existing static submit_job_wrapper
            result = submit_job_wrapper(args)
            results.append(result)
    
    print("Processing complete")
    
    # Process results
    avg_results_per_size = []
    variance_results_per_size = []
    n_iterations = []
    
    # Group results by problem size
    for i in range(len(problem_set)):
        size_results = [(val, iters) for nqb, val, iters in results if nqb == problem_set[i][0]]
        print(size_results)
        values = [r[0] for r in size_results]
        iterations = [int(r[1]) for r in size_results]
        
        avg_results_per_size.append(np.mean(values))
        variance_results_per_size.append(np.var(values))
        n_iterations.append(np.sum(iterations))
    return (avg_results_per_size, variance_results_per_size, n_iterations)


def random_walk_extrapolation(problem_sizes, values, errors, target_size, 
                            num_walks=1000, confidence_levels=[0.6827, 0.9545, 0.9973]):
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

def hardware_resource(algo_resources):
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

def plot_results(problem_sizes, values, errors, results, target_size):
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
    #plt.savefig('extrapolation_results.pdf', bbox_inches='tight')
    plt.show()

def benchmark(nqbits, depths, rnds, ansatz, observe, noise_params, nshots, thermal_size, thermodynamic_limit, hw):
    print("Benchmarking Main")
    problem_set = list(zip(nqbits, depths))
    sim_results, sim_variance, sim_iterations = run_parallel_jobs(problem_set, rnds, ansatz, observe, noise_params)
    #self.run_serial_jobs() #for testing
    # Perform random walk extrapolation
    projected_results = random_walk_extrapolation(nqbits, sim_results, sim_variance, target_size=thermal_size)
    projected_value = projected_results['extrapolated_value']
    error = np.abs(projected_value - thermodynamic_limit)
    error_bars = projected_results['extrapolated_error']
    # Calculate algorithmic resources
    algo_resources = 0
    for i in range(len(problem_set)):
        obss = create_observable(observe, nqbits[i])
        if ansatz == "RYA":
            circuit = gen_circ_RYA((nqbits[i], depths[i]))
        elif ansatz == "HVA":
            circuit = gen_circ_HVA((nqbits[i], depths[i]))
        pauls = len(obss.terms)
        gates_count = sum([circuit.count(yt) for yt in gateset()])
        algo_resources += pauls * gates_count * sim_iterations[i] * nshots
    algo_eff = error / algo_resources
    if hw != None:
        hw_res = hardware_resource(algo_resources)
        hw_eff = error / hw_res
        print("Hardware resources = %f" %hw_res)
        print("Hardware efficiency = %f" %hw_eff)
    else:
        hw_res = None
        hw_eff = None
    print("Algorithmic resources = %f" %algo_resources)
    print("Algorithmic efficiency = %f" %algo_eff)
    print("Projected value = %f" %projected_value)     
    print("Error using the exact value = %f" %error)
    plot_results(nqbits, sim_results, sim_variance, projected_results, thermal_size)
    return (sim_results, sim_variance, projected_results, error, error_bars)

#benchmark(nqbits, depths, rnds, ansatz, observe, noise_params, nshots, thermal_size, thermodynamic_limit, hw):
#code to generate result. Especially required for using multiprocessing correctly
if __name__ == '__main__':
    nqbits = [3, 4]
    deps = [2, 3]
    rnd = 4 #random seeds
    noise_p = [0.0001, -1]
    scale = -0.443147 #per site using bethe ansatz for isotropic heisenberg chain
    therm_size = 100
    a, b, c, d, e = benchmark(nqbits, deps, rnd, 'RYA', 'Heisenberg', noise_p, 1000, therm_size, therm_size*scale, 'supercond')
    print("Completed")