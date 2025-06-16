# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:54:14 2025

@author: cqtv201
"""
#compare the performance of RYA and HVA
#plots: metric vs efficiency
#error vs problem size
#change the plotting routine entirely, to use as postprocessing


from qat.core import Observable, Term, Batch #Hamiltonian
from qat.qpus import get_default_qpu
from qat.fermion import SpinHamiltonian
import numpy as np
from multiprocessing import Pool, cpu_count
from circ_gen import gen_circ_RYA, gen_circ_HVA

from numpy.polynomial.polynomial import Polynomial

from opto_gauss_mod import Opto, GaussianNoise 

import seaborn as sns
from scipy import stats
from scipy.constants import hbar
from tqdm import tqdm
from plotter import plot_analysis

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
        #print(size_results)
        values = [r[0] for r in size_results]
        iterations = [int(r[1]) for r in size_results]
        
        avg_results_per_size.append(np.mean(values))
        variance_results_per_size.append(np.var(values))
        n_iterations.append(np.sum(iterations))
    return (avg_results_per_size, variance_results_per_size, n_iterations)

def linear_extrapolation(problem_sizes, values, errors, target_size):
    """
    Perform linear regression to estimate the value at the target size,
    incorporating error propagation.

    Parameters:
        problem_sizes (list): List of problem sizes.
        values (list): Corresponding values for the problem sizes.
        errors (list): Errors (standard deviations) associated with the values.
        target_size (int): The target problem size for extrapolation.

    Returns:
        dict: Extrapolated value, propagated error, and regression coefficients.
    """
    # Convert inputs to numpy arrays
    problem_sizes = np.array(problem_sizes)
    values = np.array(values)
    errors = np.array(errors)

    # Convert errors to weights (inverse of variance)
    weights = 1 / (errors ** 2)

    # Perform weighted linear regression
    W = np.diag(weights)
    X = np.vstack([problem_sizes, np.ones(len(problem_sizes))]).T  # Design matrix
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ values)  # Regression coefficients

    # Extract slope and intercept
    slope, intercept = beta

    # Extrapolate the value at the target size
    extrapolated_value = slope * target_size + intercept

    # Propagate the error using the covariance matrix
    cov_matrix = np.linalg.inv(X.T @ W @ X)
    target_vector = np.array([target_size, 1])
    extrapolated_error = np.sqrt(target_vector @ cov_matrix @ target_vector.T)

    return {
        'extrapolated_value': extrapolated_value,
        'extrapolated_error': extrapolated_error,
        'regression_coefficients': beta,
    }

def hardware_resource(algo_resources, scale):
    #algo_resources should be provided either in linear or log scale
    #insert algorithmic resources to hardware resources conversion
    #assume same energy consumption for single and two qubit gates
    h_w0 =  6e9  # Frequency [Hz]  (Ghz ranges)
    gam = 1  # Gamma [kHz]
    t_1qb = 25* 10**(-9) #single qubit gate duration in nanosecs
    A_db = 10 #attenuation in DB
    A = 10**(A_db/10) #absolute attenuation
    T_qb = 6e-3  # Qubit Temperature [K]   (6e-3, 10)
    T_ext = 300 #external temperature in K
    E_1qb = hbar*h_w0 * (np.pi*np.pi)/(4*gam*t_1qb)
    #total heat evacuated   
    if isinstance(algo_resources, (list, np.ndarray)):
        E_cool = []
        for res in algo_resources:
            if scale == 'log':
                #log_algo_resources = np.log10(res)
                E_cool.append(np.log10((T_ext - T_qb) * A * E_1qb / T_qb) + res)
            elif scale == 'linear': 
                E_cool.append((T_ext - T_qb) * A * E_1qb * res / T_qb)
    else :
        if scale == 'log':
            #log_algo_resources = np.log10(algo_resources)
            E_cool = np.log10((T_ext - T_qb) * A * E_1qb / T_qb) + algo_resources
        elif scale == 'linear': 
            E_cool = (T_ext - T_qb) * A * E_1qb * algo_resources / T_qb
    return E_cool

def benchmark(nqbits, depths, rnds, ansatz, observe, noise_params, nshots, known_size, hw):
    print("Benchmarking Main")
    problem_set = list(zip(nqbits, depths))
    print(problem_set)
    #parallel jobs routine for problem set
    sim_results, sim_variance, sim_iterations = run_parallel_jobs(problem_set, rnds, ansatz, observe, noise_params)

    # Perform linear regression and extrapolation
    projected_results = linear_extrapolation(nqbits, sim_results, sim_variance, target_size=known_size)
    projected_value = projected_results['extrapolated_value']

    # Calculate the known result
    obss = create_observable(observe, known_size)
    obs_class = SpinHamiltonian(nqbits=obss.nbqbits, terms=obss.terms)
    obs_mat = obs_class.get_matrix()
    eigvals, _ = np.linalg.eigh(obs_mat)
    known_res = eigvals[0]

    error = np.abs(projected_value - known_res) #absolute errors at projected value
    #error_bars = projected_results['extrapolated_error'] #this is propagated error, not same as absolute error at projected value

    # Calculate algorithmic resources
    # also errors at sampled problem sizes 
    algo_res = 0
    log_algo_res_list = []
    abs_errors = []
    for i in range(len(problem_set)):
        obss = create_observable(observe, nqbits[i])
        #absolute errors
        obs_class = SpinHamiltonian(nqbits=obss.nbqbits, terms=obss.terms)
        obs_mat = obs_class.get_matrix()
        eigvals, _ = np.linalg.eigh(obs_mat)
        abs_errors.append(np.abs(eigvals[0]- sim_results[i])) #absolute error at sampled sizes
        #algorithmic resources
        if ansatz == "RYA":
            circuit = gen_circ_RYA((nqbits[i], depths[i]))
        elif ansatz == "HVA":
            circuit = gen_circ_HVA((nqbits[i], depths[i]))
        pauls = len(obss.terms)
        gates_count = sum([circuit.count(yt) for yt in gateset()])
        resource = pauls * gates_count * sim_iterations[i] * nshots
        resources_log = np.log10(pauls) + np.log10(gates_count) + np.log10(sim_iterations[i]) + np.log10(nshots)
        algo_res += resource
        log_algo_res_list.append(resources_log)
    #print(algo_res_list)

    log_algo_res = np.log10(algo_res)  # Convert to log scale for better readability
    #log_algo_res_list = np.log10(algo_res_list)  # Convert to log scale for better readability
    log_error = np.log10(error)
    #algo_eff = error / algo_resources
    log_algo_eff = log_error - log_algo_res
    #log form for hardware resources is good, since algo resources is very large
    #this is for the target problem size, not the sampled problem sizes
    if hw is not None:
        log_hw_res = hardware_resource(log_algo_res, 'log')
        log_hw_res_list = hardware_resource(log_algo_res_list, 'log')
        log_hw_eff = log_error - log_hw_res

    else:
        hw_res = None
        hw_eff = None

    bmark = {}
    bmark['metrics'] = [sim_results, sim_variance, abs_errors, projected_results, error]
    bmark['resources'] = [log_algo_res, log_hw_res, log_algo_res_list, log_hw_res_list]
    bmark['efficiency'] = [log_algo_eff, log_hw_eff]
    bmark['given_params'] = [nqbits, depths, known_size]
    print("Benchmarking main complete")
    return bmark


#benchmark(nqbits, depths, rnds, ansatz, observe, noise_params, nshots, thermal_size, thermodynamic_limit, hw):
#code to generate result. Especially required for using multiprocessing correctly
if __name__ == '__main__':
    print("Starting Benchmarking")
    #nqbits = [3, 4, 5, 6, 7]
    #deps = [3, 4, 5, 6, 7] #ensure the depths are same in the number as qubits
    nqbits = [3, 4, 5]
    deps = [3, 4, 5]
    rnd = 4 #random seeds
    noise_p = [0.0001, -1]
    therm_size = 6 #size of the system to be used for extrapolation
    bmark_RYA = benchmark(nqbits, deps, rnd, 'RYA', 'Heisenberg', noise_p, 1000, therm_size, 'supercond')
    #plot individual results
    #problem_sizes, values, errors, results, target_size
    #(nqbits, sim_results, sim_variance, projected_results, known_size)
    #plot_results(nqbits, bmark_RYA['metrics'][0], bmark_RYA['metrics'][1], bmark_RYA['metrics'][3], therm_size)
    bmark_HVA = benchmark(nqbits, deps, rnd, 'HVA', 'Heisenberg', noise_p, 1000, therm_size, 'supercond')
    #plot_results(nqbits, bmark_HVA['metrics'][0], bmark_HVA['metrics'][1], bmark_HVA['metrics'][3], therm_size)
    #plot comparative results using MNR
    bmarks = {'RYA': bmark_RYA, 'HVA': bmark_HVA}
    plot_analysis(bmarks)
    print("Completed Benhcmarking")