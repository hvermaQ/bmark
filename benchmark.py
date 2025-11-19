# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:54:14 2025

@author: cqtv201

Modified with improved weighted regression, error handling, and consistency
(Original comments retained)

Extended for full energetic benchmarking:
- Incremental & cumulative energy accounting (quantum + classical)
- Slide-22 style plots:
    (1) Linear extrapolation vs problem size n (with dual x-axis: n & depth)
    (2) VQE energy vs cumulative total energy (markers = problem size)
- Slide-23 style plots:
    (3) HW efficiency vs metric (pointwise)
    (4) ALGO efficiency vs metric (pointwise)
    (5) HW efficiency vs HW energy cost per point
    (6) ALGO efficiency vs ALGO resource cost per point
    (7) Quantum vs classical energy (per point, dual y-axis)
    (8) Metric and Err vs cumulative algorithmic resources (dual y-axis)
- QPE success probability vs total cumulative energy

ENERGY MODEL (AGREED):
- AlgoResources_i = #Pauli_i × #gates_i × nshots × n_iterations_i
- E_q_i           = hardware_resource(AlgoResources_i)
- classical_energy_consumption(depth, eff, ansatz) returns:
      (E_c_per_iter, FLOPs_per_iter)
  so:
      FLOPs_i = FLOPs_per_iter × n_iterations_i
      E_c_i   = E_c_per_iter × n_iterations_i
- QPE success probability:
      P_QPE = cos^2(Err / 2)
"""

from qat.core import Observable, Term  # Hamiltonian
from qat.qpus import get_default_qpu
from qat.fermion import SpinHamiltonian
import numpy as np
from multiprocessing import Pool, cpu_count
from circ_gen import gen_circ_RYA, gen_circ_HVA
from opto_gauss_mod import Opto, GaussianNoise
import os, time
from qat.plugins import ScipyMinimizePlugin

from extrapolation import linear_extrapolation
from energy_models import (
    hardware_resource,
    classical_energy_consumption,
)
from plotting import (
    plot_linear_extrapolation_sizes,
    plot_energy_vs_cumulative_energy,
    plot_hw_efficiency_vs_metric,
    plot_algo_efficiency_vs_metric,
    plot_hw_efficiency_vs_hw_energy,
    plot_algo_efficiency_vs_algo_resources,
    plot_hw_quantum_vs_classical,
    plot_metric_and_err_vs_algo_resources,
    plot_qpe_success_vs_total_energy,
)


#optimizer = Opto()

optimizer = ScipyMinimizePlugin(method="COBYLA",
                            tol=1e-6,
                            options={"maxiter": 50000},
#                                  x0=np.zeros(nqbt)
                                )
qpu = get_default_qpu()


# ===========================================================
#  gateset for counting algorithmic resources
# ===========================================================
def gateset():
    # gateset for counting gates to introduce noise through Gaussian noise plugin
    one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    two_qb_gateset = ['CNOT', 'CSIGN']
    gates = one_qb_gateset + two_qb_gateset
    return gates


# ===========================================================
#  Observable / Hamiltonian creation
# ===========================================================
def create_observable(model, nqbts):
    if model == "Heisenberg":
        # Instantiation of Hamiltonian
        heisen = Observable(nqbts)
        # Generation of Heisenberg Hamiltonian
        for q_reg in range(nqbts - 1):
            heisen += Observable(
                nqbts,
                pauli_terms=[
                    Term(1., typ, [q_reg, q_reg + 1])
                    for typ in ['XX', 'YY', 'ZZ']
                ]
            )
        obs = heisen
    else:
        raise ValueError(f"Unknown model type: {model}")
    return obs


# ===========================================================
#  Exact diagonalization for reference ground-state energy
# ===========================================================
def exact_Result(obs):
    obs_class = SpinHamiltonian(nqbits=obs.nbqbits, terms=obs.terms)
    obs_mat = obs_class.get_matrix()
    eigvals, _ = np.linalg.eigh(obs_mat)
    g_energy = eigvals[0]
    return g_energy


# ===========================================================
#  Wrapper for parallel job submission
# ===========================================================
def submit_job_wrapper(args):
    """Wrapper function for parallel job submission that recreates required objects"""
    nqbts, dep, rnd, ans, ham, n_params = args

    # Everything based on i
    if ans == "RYA":
        circuit = gen_circ_RYA((nqbts, dep))
    elif ans == "HVA":
        circuit = gen_circ_HVA((nqbts, dep))
    else:
        raise ValueError("Unsupported ansatz type")

    # Create observable
    obss = create_observable(ham, nqbts)
    job = circuit.to_job(observable=obss, nbshots=0)
    obs_mat = obss.to_matrix().A

    # Create fresh instances in worker process
    stack = optimizer | GaussianNoise(n_params[0], n_params[1], obs_mat) | qpu
    result = stack.submit(job)

    #print(result.meta_data["n_steps"])
    print(len(eval(result.meta_data["optimization_trace"])))
    #return (nqbts, result.value, result.meta_data["n_steps"])
    return (nqbts, result.value, len(eval(result.meta_data["optimization_trace"])))


# ===========================================================
#  Parallel execution of jobs
# ===========================================================
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

    with Pool(processes=num_processes) as pool:
        result_async = pool.map_async(submit_job_wrapper, job_args)
        results = result_async.get()

    print("Parallel done")

    avg_results_per_size = []
    std_results_per_size = []
    n_iterations = []

    # Group results by problem size
    for i in range(len(problem_set)):
        size_results = [
            (val, iters)
            for nqb, val, iters in results
            if nqb == problem_set[i][0]
        ]
        values = np.array([r[0] for r in size_results], dtype=float)
        iterations = [int(r[1]) for r in size_results]

        avg_results_per_size.append(np.mean(values))
        std = np.std(values, ddof=1) if len(values) > 1 else 1e-12
        std_results_per_size.append(max(std, 1e-12))
        n_iterations.append(np.sum(iterations))

    return (np.asarray(avg_results_per_size, dtype=float),
            np.asarray(std_results_per_size, dtype=float),
            np.asarray(n_iterations, dtype=int))


# ===========================================================
#  Main benchmark function
# ===========================================================
def benchmark(nqbits, depths, rnds, ansatz, observe, noise_params,
              nshots, known_size, hw,
              efficiency_flops_per_watt=1e10):
    """
    Extended benchmark including classical optimization energy consumption.

    Tracks:
    - Per-size VQE energies and local errors
    - Algorithmic resources (point and cumulative)
    - Quantum and classical energies (point and cumulative)
    - Projection to target size n* using linear extrapolation
    - QPE success probabilities derived from local errors via P_QPE = cos^2(Err/2)

    ENERGY MODEL:
    - AlgoResources_i = #Pauli_i × #gates_i × nshots × n_iterations_i
    - E_q_i           = hardware_resource(AlgoResources_i)
    - classical_energy_consumption(depth, eff, ansatz) returns:
          (E_c_per_iter, FLOPs_per_iter)
      so:
          FLOPs_i = FLOPs_per_iter × n_iterations_i
          E_c_i   = E_c_per_iter × n_iterations_i
    """
    print("Benchmarking Main")
    problem_set = list(zip(nqbits, depths))
    nqbits = np.asarray(nqbits, dtype=int)
    depths = np.asarray(depths, dtype=int)

    sim_results, sim_std, sim_iterations = run_parallel_jobs(
        problem_set, rnds, ansatz, observe, noise_params
    )

    # --------------------------
    # Per-size exact energies & errors (local, no extrapolation)
    # --------------------------
    exact_per_size = []
    for n in nqbits:
        obs_n = create_observable(observe, int(n))
        exact_per_size.append(exact_Result(obs_n))
    exact_per_size = np.asarray(exact_per_size, dtype=float)

    local_errors = np.abs(sim_results - exact_per_size)
    metric_per_point = np.clip(1.0 - local_errors, 0.0, 1.0)

    # QPE success probability per point: P_QPE = cos^2(Err/2)
    P_qpe_per_point = np.cos(local_errors / 2.0) ** 2

    # --------------------------
    # Projection to known_size (Slide-22 linear extrapolation)
    # --------------------------
    projected_results = linear_extrapolation(
        nqbits, sim_results, sim_std, target_size=known_size
    )
    projected_value = projected_results['extrapolated_value']

    obss_star = create_observable(observe, known_size)
    obs_class_star = SpinHamiltonian(
        nqbits=obss_star.nbqbits, terms=obss_star.terms
    )
    obs_mat_star = obs_class_star.get_matrix()
    eigvals_star, _ = np.linalg.eigh(obs_mat_star)
    known_res_star = float(eigvals_star[0])

    projected_error = float(np.abs(projected_value - known_res_star))
    projected_error_bar = float(projected_results['extrapolated_error'])
    P_qpe_star = float(np.cos(projected_error / 2.0) ** 2)

    # -------------------------------------------------------
    # Algorithmic resources & energies (pointwise and cumulative)
    # -------------------------------------------------------
    algo_resources_point = []   # total gate executions per size
    E_q_point = []              # quantum energy per size
    E_c_point = []              # classical energy per size
    classical_flops_total = 0.0

    for i in range(len(problem_set)):
        n_i = int(nqbits[i])
        d_i = int(depths[i])
        iters_i = int(sim_iterations[i])

        # Hamiltonian (for #Pauli terms)
        obss_i = create_observable(observe, n_i)
        paulis = len(obss_i.terms)

        # Circuit & gate count
        if ansatz == "RYA":
            circuit = gen_circ_RYA((n_i, d_i))
        elif ansatz == "HVA":
            circuit = gen_circ_HVA((n_i, d_i))
        else:
            raise ValueError("Unsupported ansatz")

        gcount = sum(circuit.count(g) for g in gateset())

        # ------------------------------
        # Quantum-side algorithmic resources
        # R_i = #Pauli × #gates × nshots × #iterations
        # ------------------------------
        R_i = paulis * gcount * nshots * iters_i
        algo_resources_point.append(R_i)

        # Quantum energy
        E_q_i = hardware_resource(R_i) if hw is not None else 0.0
        E_q_point.append(E_q_i)

        # ------------------------------
        # Classical FLOPs and energy
        # E_c_per_iter, FLOPs_per_iter from fits
        # ------------------------------
        E_c_per_iter, FLOPs_per_iter = classical_energy_consumption(
            d_i,
            efficiency_flops_per_watt,
            ansatz,
        )

        FLOPs_i = FLOPs_per_iter * iters_i
        E_c_i = E_c_per_iter * iters_i

        classical_flops_total += FLOPs_i
        E_c_point.append(E_c_i)

    algo_resources_point = np.asarray(algo_resources_point, dtype=float)
    E_q_point = np.asarray(E_q_point, dtype=float)
    E_c_point = np.asarray(E_c_point, dtype=float)

    E_point_total = E_q_point + E_c_point
    E_cum_total = np.cumsum(E_point_total)
    algo_resources_cum = np.cumsum(algo_resources_point)

    total_quantum_energy = float(np.sum(E_q_point))
    total_classical_energy = float(np.sum(E_c_point))
    total_energy = float(np.sum(E_point_total))
    algo_resources_total = float(np.sum(algo_resources_point))

    print("\n=== Energy Accounting (incremental & cumulative, NEW MODEL) ===")
    print(f"Total quantum energy     = {total_quantum_energy:.6e} J")
    print(f"Total classical energy   = {total_classical_energy:.6e} J")
    print(f" → Total energy          = {total_energy:.6e} J")
    print(f"Total classical FLOPs    = {classical_flops_total:.3e}")
    print(f"Total algorithmic res.   = {algo_resources_total:.3e}\n")

    # -------------------------------------------------------
    # Plots
    # -------------------------------------------------------
    plot_linear_extrapolation_sizes(
        problem_sizes=nqbits,
        energies=sim_results,
        errors=sim_std,
        projected_results=projected_results,
        n_star=known_size,
        exact_E_star=known_res_star,
        depths=depths,
        P_qpe_star=P_qpe_star,
        P_qpe_points=P_qpe_per_point,
    )

    plot_energy_vs_cumulative_energy(
        problem_sizes=nqbits,
        energies=sim_results,
        cumulative_energies=E_cum_total,
        P_qpe=P_qpe_per_point,
    )

    plot_hw_efficiency_vs_metric(
        problem_sizes=nqbits,
        metric=metric_per_point,
        hw_energy_point=E_q_point,
        P_qpe=P_qpe_per_point,
    )

    plot_algo_efficiency_vs_metric(
        problem_sizes=nqbits,
        metric=metric_per_point,
        algo_resources_point=algo_resources_point,
        P_qpe=P_qpe_per_point,
    )


    plot_hw_efficiency_vs_hw_energy(
        problem_sizes=nqbits,
        metric=metric_per_point,
        hw_energy_point=E_q_point,
        P_qpe=P_qpe_per_point,
    )

    plot_algo_efficiency_vs_algo_resources(
        problem_sizes=nqbits,
        metric=metric_per_point,
        algo_resources_point=algo_resources_point,
        P_qpe=P_qpe_per_point,
    )


    plot_hw_quantum_vs_classical(
        problem_sizes=nqbits,
        E_q_point=E_q_point,
        E_c_point=E_c_point,
        P_qpe=P_qpe_per_point,
    )
    plot_metric_and_err_vs_algo_resources(
        problem_sizes=nqbits,
        metric=metric_per_point,
        errors=local_errors,
        algo_resources_cum=algo_resources_cum,
        P_qpe=P_qpe_per_point,
    )

    plot_qpe_success_vs_total_energy(
        problem_sizes=nqbits,
        P_qpe=P_qpe_per_point,
        cumulative_energy=E_cum_total,
        P_qpe_star=P_qpe_star,
    )

    # ===========================================================
    # SAVE ALL NUMERICAL RESULTS (NPZ ONLY) FOR RYA vs HVA ANALYSIS
    # ===========================================================
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "benchmark_results"
    os.makedirs(save_dir, exist_ok=True)

    tag = f"{ansatz}_nshots{nshots}_noise{noise_params[0]}_{timestamp}".replace(".", "p")

    np.savez(
        f"{save_dir}/{tag}.npz",
        ansatz=ansatz,
        model=observe,  # ← optional but recommended
        nqbits=nqbits,
        depths=depths,
        sim_results=sim_results,
        sim_std=sim_std,
        exact_per_size=exact_per_size,
        local_errors=local_errors,
        metric_per_point=metric_per_point,
        algo_resources_point=algo_resources_point,
        algo_resources_cum=algo_resources_cum,
        E_q_point=E_q_point,
        E_c_point=E_c_point,
        E_point_total=E_point_total,
        E_cum_total=E_cum_total,
        projected_value=projected_results["extrapolated_value"],
        projected_error=projected_results["extrapolated_error"],
        regression_coefficients=projected_results["regression_coefficients"],  # optional
        P_qpe_per_point=P_qpe_per_point,
        P_qpe_star=P_qpe_star,
        total_quantum_energy=total_quantum_energy,
        total_classical_energy=total_classical_energy,
        total_energy=total_energy,
    )

    print(f"[saved] NPZ results → {save_dir}/{tag}.npz\n")

    return {
        "nqbits": nqbits,
        "depths": depths,
        "sim_results": sim_results,
        "sim_std": sim_std,
        "exact_per_size": exact_per_size,
        "local_errors": local_errors,
        "metric_per_point": metric_per_point,
        "projected_results": projected_results,
        "projected_error": projected_error,
        "projected_error_bar": projected_error_bar,
        "algo_resources_point": algo_resources_point,
        "algo_resources_cum": algo_resources_cum,
        "E_q_point": E_q_point,
        "E_c_point": E_c_point,
        "E_point_total": E_point_total,
        "E_cum_total": E_cum_total,
        "total_quantum_energy": total_quantum_energy,
        "total_classical_energy": total_classical_energy,
        "total_energy": total_energy,
        "P_qpe_per_point": P_qpe_per_point,
        "P_qpe_star": P_qpe_star,
    }


if __name__ == '__main__':
    nqbits = [4, 5, 6, 7]
    deps = [4, 5, 6, 7]
    rnd = 5
    noise_p = [0.0001, -1]
    therm_size = 8

    results = benchmark(
        nqbits, deps, rnd, 'RYA', 'Heisenberg',
        noise_p, 1000, therm_size, 'supercond',
        efficiency_flops_per_watt=1e10,
    )
    print("Completed")
