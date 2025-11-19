# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:54:14 2025

@author: cqtv201

Modified with improved weighted regression, error handling, and consistency
(Original comments retained)

Extended for full energetic benchmarking:
- Incremental & cumulative energy accounting (quantum + classical)
- Per-depth plots and cumulative frontier plots
- Energetic Benchmark (Slide 22): Error vs Total Cumulative Energy (log-x)
- Energetic Standard (Slide 23): Efficiency vs Metric (linear)
- Metric/Resource and Metric/Energy diagnostics
- Incremental efficiency (ΔMetric / ΔEnergy)
- Dual-format figure export (.pdf + .png), scientific style

NEW:
- QPE success probability from VQE error (NISQ→LSQ bridge)
- Success vs energy and success-per-joule frontier plots

UPDATED (this version):
- n = [4, 5, 6, 7] only (n = 3 excluded)
- Slide 22 extrapolation plot (n vs energy) + Slide 22-style plot vs cumulative energy
- Hardware (HW) and algorithmic (ALGO) efficiency vs metric and vs energy/resources
- HW-quantum and HW-classical energies on dual y-axis vs problem size
- Metric (𝓜 = 1 − Err) and Err vs algorithmic resources on dual y-axis
- QPE success probability vs total cumulative energy
"""

from qat.core import Observable, Term  # Hamiltonian
from qat.qpus import get_default_qpu
from qat.fermion import SpinHamiltonian
import numpy as np
from multiprocessing import Pool, cpu_count
from circ_gen import gen_circ_RYA, gen_circ_HVA
import matplotlib.pyplot as plt
from opto_gauss_mod import Opto, GaussianNoise
from scipy.constants import hbar
import math

# --- Scientific plotting style (paper-ready) ---
plt.rcParams.update({
    'font.size': 16,
    'figure.dpi': 120,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
})

optimizer = Opto()
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
                pauli_terms=[Term(1., typ, [q_reg, q_reg + 1])
                             for typ in ['XX', 'YY', 'ZZ']]
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

    print(result.meta_data["n_steps"])
    return (nqbts, result.value, result.meta_data["n_steps"])


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
        # Use asynchronous mapping for parallel execution
        result_async = pool.map_async(submit_job_wrapper, job_args)
        results = result_async.get()

    print("Parallel done")

    # Process results: compute mean, std (not variance) and total iterations
    avg_results_per_size = []
    std_results_per_size = []
    n_iterations = []

    # Group results by problem size
    for i in range(len(problem_set)):
        size_results = [(val, iters)
                        for nqb, val, iters in results if nqb == problem_set[i][0]]
        values = np.array([r[0] for r in size_results], dtype=float)
        iterations = [int(r[1]) for r in size_results]

        avg_results_per_size.append(np.mean(values))
        std = np.std(values, ddof=1) if len(values) > 1 else 1e-12
        std_results_per_size.append(max(std, 1e-12))
        n_iterations.append(np.sum(iterations))

    return (avg_results_per_size, std_results_per_size, n_iterations)


# ===========================================================
#  Weighted linear regression with error propagation
# ===========================================================
def linear_extrapolation(problem_sizes, values, errors, target_size):
    """
    Perform weighted linear regression to estimate the value at the target size,
    incorporating proper error propagation.
    """
    x = np.asarray(problem_sizes, dtype=float)
    y = np.asarray(values, dtype=float)
    sigma = np.maximum(np.asarray(errors, dtype=float), 1e-12)

    # Convert errors to weights (inverse variance)
    w = 1.0 / (sigma ** 2)
    X = np.vstack([x, np.ones_like(x)]).T

    # Weighted normal equations
    WX = X * w[:, None]
    XT_W_X = X.T @ WX
    XT_W_y = (X.T * w) @ y

    try:
        beta = np.linalg.solve(XT_W_X, XT_W_y)
        cov_unscaled = np.linalg.inv(XT_W_X)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XT_W_X) @ XT_W_y
        cov_unscaled = np.linalg.pinv(XT_W_X)

    m, b = beta

    # Residual variance scaling
    y_hat = m * x + b
    dof = max(len(x) - 2, 1)
    wrss = np.sum(w * (y - y_hat) ** 2)
    s2 = wrss / dof
    cov_beta = cov_unscaled * s2

    # Extrapolate to target size
    x_star = float(target_size)
    v = np.array([x_star, 1.0])
    y_star = float(m * x_star + b)
    y_star_std = float(np.sqrt(v @ cov_beta @ v))

    return {
        'extrapolated_value': y_star,
        'extrapolated_error': y_star_std,
        'regression_coefficients': np.array([m, b]),
        'residual_variance': s2,
    }


# ===========================================================
#  Hardware resource model
# ===========================================================
def hardware_resource(algo_resources):
    # insert algorithmic resources to hardware resources conversion
    # assume same energy consumption for single and two qubit gates
    h_w0 = 6e9  # Frequency [Hz]  (GHz ranges)
    gam = 1  # Gamma [kHz]
    t_1qb = 25 * 10 ** (-9)  # single qubit gate duration in nanosecs
    A_db = 20  # attenuation in DB
    A = 10 ** (A_db / 10)  # absolute attenuation
    T_qb = 15e-3  # Qubit Temperature [K]   (6e-3, 10)
    T_ext = 300  # external temperature in K
    E_1qb = hbar * h_w0 * (np.pi * np.pi) / (4 * gam * t_1qb)
    # total heat evacuated
    E_cool = (T_ext - T_qb) * A * E_1qb * algo_resources / T_qb
    return float(E_cool)


# ======================================================================
#  Classical energy consumption model (from VQE–QPU fits) for HVA only
# ======================================================================
def classical_energy_consumption(depth, efficiency_flops_per_watt):
    """
    Compute classical optimization energy consumption for a given circuit depth
    **per optimizer iteration**.

    Parameters
    ----------
    depth : int or float
        Ansatz depth (number of layers)
    efficiency_flops_per_watt : float
        Hardware efficiency in FLOP/(s·W), i.e. FLOPs per joule.

    Returns
    -------
    energy_J_per_iter : float
        Classical energy consumption in joules for one optimizer iteration.
    flops_opt_per_iter : float
        FLOP count attributable to optimization overhead for one iteration.
    """
    # --- Baseline QPU-only fit (linear) ---
    flops_qpu = 18426.225 * depth + 19210.299

    # --- Full noisy VQE fit (exponential form) ---
    flops_vqe = 96621.711 * np.exp(0.262 * depth) - 102781.482

    # --- Optimization-only FLOPs per iteration ---
    flops_opt_per_iter = max(flops_vqe - flops_qpu, 0.0)

    # --- Convert FLOPs to joules using efficiency ---
    energy_J_per_iter = flops_opt_per_iter / efficiency_flops_per_watt
    return float(energy_J_per_iter), float(flops_opt_per_iter)

# ===========================================================
#  QPE success probability from VQE error (NISQ→LSQ bridge)
# ===========================================================
def qpe_success_from_vqe_error(
    epsilon,      # energy error (|E_calc - E0|) in same units as gap_delta
    gap_delta,    # spectral gap Δ = E1 - E0 (positive)
    t_bits,       # QPE precision bits
    s_bits=2,     # extra "padding" bits
    p_eff=0.0,    # effective error per controlled-U
    T_total=0.0,  # total QPE circuit duration (s)
    T2=float("inf")  # coherence time (s)
):
    """Return dict with QPE success terms for a *scalar* epsilon."""
    if gap_delta <= 0:
        raise ValueError("gap_delta must be positive")
    # Variational bound → fidelity lower bound
    F_min = max(0.0, 1.0 - float(epsilon) / float(gap_delta))
    # Ideal phase success for t precision with s padding
    if s_bits >= 1:
        P_phase = 1.0 - 2.0 ** (-(s_bits - 1))
    else:
        P_phase = 0.0
    # Controlled-U count
    M_ctrlU = max(0, (1 << (int(t_bits) + int(max(s_bits, 0)))) - 1)
    # Noise factors
    noise_gate_fac = math.exp(-float(p_eff) * M_ctrlU) if p_eff > 0 else 1.0
    noise_deph_fac = math.exp(-float(T_total) / float(T2)) if (T2 not in (0, float("inf")) and T_total > 0) else 1.0
    P_success = F_min * P_phase * noise_gate_fac * noise_deph_fac
    return {
        "F_min": F_min,
        "P_phase": P_phase,
        "M_ctrlU": M_ctrlU,
        "noise_gate_fac": noise_gate_fac,
        "noise_deph_fac": noise_deph_fac,
        "P_success": max(0.0, min(1.0, P_success)),
    }


# ===========================================================
#  Plot helpers (all save .pdf + .png)
# ===========================================================
def _savefig(base):
    plt.tight_layout()
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.savefig(f"{base}.png", bbox_inches="tight")
    plt.show()


def _annotate_metric_and_qpe(deltaE):
    """
    Annotate plots where the metric 𝓜 appears on an axis,
    and connect it to QPE success scaling with ΔE.
    """
    txt = (
        r"$\mathcal{M} = 1 - \mathrm{Err}$" + "\n" +
        r"$P_{\mathrm{QPE}}$ from VQE error and $\Delta E$" + "\n" +
        r"$\Delta E = %.3f$" % deltaE
    )
    ax = plt.gca()
    ax.annotate(
        txt,
        xy=(0.03, 0.05),
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8)
    )


# ===========================================================
#  Slide 22: linear extrapolation plot (n vs energy)
# ===========================================================
def plot_slide22_extrapolation(problem_sizes, values, std_errors,
                               results, target_size, exact_target):
    ps = np.asarray(problem_sizes, dtype=float)
    vals = np.asarray(values, dtype=float)
    errs = np.asarray(std_errors, dtype=float)

    plt.figure(figsize=(8, 6))
    # Data with error bars
    plt.errorbar(ps, vals, yerr=errs, fmt='o', color='C0',
                 capsize=5, label="VQE energies $E_{\\mathrm{VQE}}(n)$",
                 markersize=8, mec='k')
    for x, y, n in zip(ps, vals, problem_sizes):
        plt.text(x, y, f"{n}", fontsize=10, ha='left', va='bottom')

    # Linear fit
    slope, intercept = results['regression_coefficients']
    x_range = np.linspace(min(ps), target_size * 1.1, 200)
    y_fit = slope * x_range + intercept
    plt.plot(x_range, y_fit, 'r--', label="Weighted linear fit")

    # Extrapolated point
    y_star = results['extrapolated_value']
    y_star_err = results['extrapolated_error']
    plt.errorbar([target_size], [y_star], yerr=[y_star_err],
                 fmt='o', color='C3', markersize=9, mec='k',
                 capsize=5,
                 label=f"Extrapolated at $n^*={target_size}$")

    # Exact energy at n*
    plt.axhline(exact_target, color='gray', linestyle=':',
                label=f"Exact $E_0(n^*={target_size})$")

    plt.xlabel("Problem size $n$")
    plt.ylabel("Energy $E_{\\mathrm{VQE}}(n)$")
    plt.title("Linear extrapolation to $n^*$ (Slide 22)")
    plt.legend()
    _savefig("slide22_linear_extrapolation")


# ===========================================================
#  Slide 22-style: energy vs cumulative total energy (markers = n)
# ===========================================================
def plot_slide22_vs_energy(cumulative_energies, values, sizes, exact_target, target_size):
    e_cum = np.asarray(cumulative_energies, dtype=float)
    vals = np.asarray(values, dtype=float)
    sizes = np.asarray(sizes, dtype=int)

    plt.figure(figsize=(8, 6))
    plt.scatter(e_cum, vals, s=80, c='C1', edgecolors='k',
                label="$E_{\\mathrm{VQE}}(n)$")
    for x, y, n in zip(e_cum, vals, sizes):
        plt.text(x, y, f"{n}", fontsize=10, ha='left', va='bottom')

    plt.xscale('log')
    plt.axhline(exact_target, color='gray', linestyle=':',
                label=f"Exact $E_0(n^*={target_size})$")
    plt.xlabel("Cumulative total energy $E_{\\mathrm{tot}}^{(\\mathrm{cum})}$ (J)")
    plt.ylabel("Energy $E_{\\mathrm{VQE}}(n)$")
    plt.title("Slide 22-style: VQE energy vs cumulative total energy")
    plt.legend()
    _savefig("slide22_energy_vs_cum_energy")


# ===========================================================
#  HW & ALGO efficiency / energy / metric plots
# ===========================================================
def plot_hw_eff_vs_metric(metric, e_hw_cum, sizes, deltaE):
    metric = np.asarray(metric, dtype=float)
    e_hw_cum = np.asarray(e_hw_cum, dtype=float)
    sizes = np.asarray(sizes, dtype=int)

    eta_hw = np.divide(metric, e_hw_cum,
                       out=np.zeros_like(metric),
                       where=e_hw_cum > 0)

    plt.figure(figsize=(8, 6))
    plt.plot(metric, eta_hw, 'o-', color='C2')
    for x, y, n in zip(metric, eta_hw, sizes):
        plt.text(x, y, f"{n}", fontsize=10, ha='left', va='bottom')

    plt.xlabel(r"Metric $\mathcal{M} = 1 - \mathrm{Err}$")
    plt.ylabel(r"Hardware efficiency $\eta_{\mathrm{HW}} = \mathcal{M} / E_{\mathrm{HW}}^{(\mathrm{cum})}$ (1/J)")
    plt.title("Hardware efficiency vs metric")
    _annotate_metric_and_qpe(deltaE)
    _savefig("hw_eff_vs_metric")


def plot_algo_eff_vs_metric(metric, algo_res_cum, sizes, deltaE):
    metric = np.asarray(metric, dtype=float)
    algo_res_cum = np.asarray(algo_res_cum, dtype=float)
    sizes = np.asarray(sizes, dtype=int)

    eta_algo = np.divide(metric, algo_res_cum,
                         out=np.zeros_like(metric),
                         where=algo_res_cum > 0)

    plt.figure(figsize=(8, 6))
    plt.plot(metric, eta_algo, 'o-', color='C4')
    for x, y, n in zip(metric, eta_algo, sizes):
        plt.text(x, y, f"{n}", fontsize=10, ha='left', va='bottom')

    plt.xlabel(r"Metric $\mathcal{M} = 1 - \mathrm{Err}$")
    plt.ylabel(r"Algorithmic efficiency $\eta_{\mathrm{ALGO}} = \mathcal{M} / R_{\mathrm{ALGO}}^{(\mathrm{cum})}$")
    plt.title("Algorithmic efficiency vs metric")
    _annotate_metric_and_qpe(deltaE)
    _savefig("algo_eff_vs_metric")


def plot_hw_eff_vs_hw_energy(e_hw_cum, metric, sizes):
    e_hw_cum = np.asarray(e_hw_cum, dtype=float)
    metric = np.asarray(metric, dtype=float)
    sizes = np.asarray(sizes, dtype=int)

    eta_hw = np.divide(metric, e_hw_cum,
                       out=np.zeros_like(metric),
                       where=e_hw_cum > 0)

    plt.figure(figsize=(8, 6))
    plt.plot(e_hw_cum, eta_hw, 'o-', color='C2')
    for x, y, n in zip(e_hw_cum, eta_hw, sizes):
        plt.text(x, y, f"{n}", fontsize=10, ha='left', va='bottom')

    plt.xlabel(r"Cumulative hardware quantum energy $E_{\mathrm{HW}}^{(\mathrm{cum})}$ (J)")
    plt.ylabel(r"Hardware efficiency $\eta_{\mathrm{HW}}$ (1/J)")
    plt.title("Hardware efficiency vs hardware energy cost")
    _savefig("hw_eff_vs_hw_energy")


def plot_algo_eff_vs_algo_energy(algo_res_cum, metric, sizes):
    algo_res_cum = np.asarray(algo_res_cum, dtype=float)
    metric = np.asarray(metric, dtype=float)
    sizes = np.asarray(sizes, dtype=int)

    eta_algo = np.divide(metric, algo_res_cum,
                         out=np.zeros_like(metric),
                         where=algo_res_cum > 0)

    plt.figure(figsize=(8, 6))
    plt.plot(algo_res_cum, eta_algo, 'o-', color='C4')
    for x, y, n in zip(algo_res_cum, eta_algo, sizes):
        plt.text(x, y, f"{n}", fontsize=10, ha='left', va='bottom')

    plt.xlabel(r"Cumulative algorithmic resources $R_{\mathrm{ALGO}}^{(\mathrm{cum})}$")
    plt.ylabel(r"Algorithmic efficiency $\eta_{\mathrm{ALGO}}$")
    plt.title("Algorithmic efficiency vs algorithmic resource cost")
    _savefig("algo_eff_vs_algo_energy")


# ===========================================================
#  HW-quantum vs HW-classical on dual y-axis (per size)
# ===========================================================
def plot_hw_quantum_vs_classical(nqbits, e_q_inc, e_c_inc):
    nqbits = np.asarray(nqbits, dtype=int)
    e_q_inc = np.asarray(e_q_inc, dtype=float)
    e_c_inc = np.asarray(e_c_inc, dtype=float)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.plot(nqbits, e_q_inc, 'o-', color='C0',
             label="Quantum hardware energy (incremental)")
    ax1.set_xlabel("Problem size $n$")
    ax1.set_ylabel("Quantum energy (incremental) [J]", color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')

    ax2 = ax1.twinx()
    ax2.plot(nqbits, e_c_inc, 's--', color='C3',
             label="Classical optimization energy (incremental)")
    ax2.set_ylabel("Classical energy (incremental) [J]", color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')

    fig.suptitle("Quantum vs classical incremental energy per problem size")
    fig.tight_layout()
    fig.savefig("hw_quantum_vs_classical.pdf", bbox_inches="tight")
    fig.savefig("hw_quantum_vs_classical.png", bbox_inches="tight")
    plt.show()


# ===========================================================
#  Metric vs algorithmic resources (dual y-axis with Err)
# ===========================================================
def plot_metric_and_err_vs_algo_resources(algo_res_cum, metric, Err, sizes, deltaE):
    algo_res_cum = np.asarray(algo_res_cum, dtype=float)
    metric = np.asarray(metric, dtype=float)
    Err = np.asarray(Err, dtype=float)
    sizes = np.asarray(sizes, dtype=int)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.plot(algo_res_cum, metric, 'o-', color='C2', label=r"$\mathcal{M}$")
    for x, y, n in zip(algo_res_cum, metric, sizes):
        ax1.text(x, y, f"{n}", fontsize=10, ha='left', va='bottom')
    ax1.set_xlabel(r"Cumulative algorithmic resources $R_{\mathrm{ALGO}}^{(\mathrm{cum})}$")
    ax1.set_ylabel(r"Metric $\mathcal{M} = 1 - \mathrm{Err}$", color='C2')
    ax1.tick_params(axis='y', labelcolor='C2')

    ax2 = ax1.twinx()
    ax2.plot(algo_res_cum, Err, 's--', color='C1', label=r"$\mathrm{Err}$")
    ax2.set_ylabel(r"$\mathrm{Err}$", color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    _annotate_metric_and_qpe(deltaE)
    fig.suptitle("Metric and Err vs cumulative algorithmic resources")
    fig.tight_layout()
    fig.savefig("metric_and_err_vs_algo_resources.pdf", bbox_inches="tight")
    fig.savefig("metric_and_err_vs_algo_resources.png", bbox_inches="tight")
    plt.show()


# ===========================================================
#  QPE success probability vs total cumulative energy
# ===========================================================
def plot_qpe_success_vs_energy(e_total_cum, p_succ, sizes, deltaE):
    e_total_cum = np.asarray(e_total_cum, dtype=float)
    p_succ = np.asarray(p_succ, dtype=float)
    sizes = np.asarray(sizes, dtype=int)

    plt.figure(figsize=(8, 6))
    plt.plot(e_total_cum, p_succ, 'o-', color='C5')
    for x, y, n in zip(e_total_cum, p_succ, sizes):
        plt.text(x, y, f"{n}", fontsize=10, ha='left', va='bottom')

    plt.xscale('log')
    plt.ylim(0, 1.05)
    plt.xlabel("Cumulative total energy $E_{\\mathrm{tot}}^{(\\mathrm{cum})}$ (J)")
    plt.ylabel("QPE success probability $P_{\\mathrm{QPE}}$")
    plt.title("QPE success probability vs total cumulative energy")
    _annotate_metric_and_qpe(deltaE)
    _savefig("qpe_success_vs_energy")


# ===========================================================
#  Main benchmark function
# ===========================================================
def benchmark(nqbits, depths, rnds, ansatz, observe, noise_params,
              nshots, known_size, hw,
              efficiency_flops_per_watt=1e12,
              # --- QPE mapping parameters (NEW) ---
              t_bits=6,           # QPE precision bits
              s_bits=2,           # padding bits
              p_eff=0.0,          # per-controlled-U error
              T_total=0.0,        # QPE circuit duration (s)
              T2=float("inf")):   # coherence time (s)
    """
    Extended benchmark including classical optimization energy consumption.
    Now tracks per-depth incremental energies and cumulative totals,
    computes per-size errors (not only projected), AND maps errors to QPE success.

    In this version:
    - n = [4, 5, 6, 7]
    - linear extrapolation to known_size = n* (Slide 22)
    - cumulative energy usage per size
    - HW and ALGO efficiencies vs metric and vs energy/resources
    - Dual-axis plots for HW classical/quantum and Metric/Err vs resources
    - QPE success probability using ΔE = spectral gap at n*
    """
    print("Benchmarking Main")
    problem_set = list(zip(nqbits, depths))
    sim_results, sim_std, sim_iterations = run_parallel_jobs(problem_set, rnds, ansatz, observe, noise_params)

    # --------------------------
    # Per-size exact energies & errors (local)
    # --------------------------
    exact_per_size = []
    for n in nqbits:
        obs_n = create_observable(observe, n)
        exact_per_size.append(exact_Result(obs_n))
    exact_per_size = np.asarray(exact_per_size, dtype=float)

    sim_results = np.asarray(sim_results, dtype=float)
    local_errors = np.abs(sim_results - exact_per_size)  # per-point Err

    # --------------------------
    # Projection to known_size (as before)
    # --------------------------
    projected_results = linear_extrapolation(nqbits, sim_results, sim_std, target_size=known_size)
    projected_value = projected_results['extrapolated_value']
    # Exact at n* and spectral gap ΔE
    obss = create_observable(observe, known_size)
    obs_class = SpinHamiltonian(nqbits=obss.nbqbits, terms=obss.terms)
    obs_mat = obs_class.get_matrix()
    eigvals, _ = np.linalg.eigh(obs_mat)
    eigvals = np.sort(eigvals)
    known_res = float(eigvals[0])
    gap_delta = float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else 1.0
    projected_error = float(np.abs(projected_value - known_res))
    projected_error_bar = float(projected_results['extrapolated_error'])

    print("\n=== Linear Extrapolation Summary (Slide 22) ===")
    print(f"Problem sizes n: {nqbits}")
    print(f"Exact energies E0(n): {exact_per_size}")
    print(f"VQE mean energies:   {sim_results}")
    print(f"Std dev (per n):     {np.asarray(sim_std, dtype=float)}")
    print(f"Extrapolated E_VQE(n*={known_size}) = {projected_value:.6f} ± {projected_error_bar:.6f}")
    print(f"Exact E0(n*={known_size})          = {known_res:.6f}")
    print(f"Err(n*={known_size})               = {projected_error:.6e}")
    print(f"Spectral gap ΔE(n*={known_size})   = {gap_delta:.6e}")

    # -------------------------------------------------------
    # Algorithmic resources & per-depth incremental energies
    # -------------------------------------------------------
    algo_resources_per_depth = []
    gate_counts = []
    e_q_incremental = []
    e_c_incremental = []
    classical_flops_total = 0.0

    for i in range(len(problem_set)):
        n_i, d_i = nqbits[i], depths[i]
        obss_i = create_observable(observe, n_i)

        # Build circuit and count gates
        if ansatz == "RYA":
            circuit = gen_circ_RYA((n_i, d_i))
        elif ansatz == "HVA":
            circuit = gen_circ_HVA((n_i, d_i))
        else:
            raise ValueError("Unsupported ansatz")

        gcount = sum(circuit.count(g) for g in gateset())
        gate_counts.append(gcount)
        pauls = len(obss_i.terms)

        # Per-depth algorithmic resources (for THIS data point)
        res_i = pauls * gcount * sim_iterations[i] * nshots
        algo_resources_per_depth.append(res_i)

        # Per-depth quantum energy (incremental)
        e_q_i = hardware_resource(res_i) if hw is not None else 0.0
        e_q_incremental.append(e_q_i)

        # Per-depth classical energy (incremental)
        e_c_i, f_c_i = classical_energy_consumption(d_i, efficiency_flops_per_watt)
        e_c_incremental.append(e_c_i)
        classical_flops_total += f_c_i

    algo_resources_per_depth = np.asarray(algo_resources_per_depth, dtype=float)
    e_q_incremental = np.asarray(e_q_incremental, dtype=float)
    e_c_incremental = np.asarray(e_c_incremental, dtype=float)

    # Cumulative energies and resources
    e_q_cum = np.cumsum(e_q_incremental)
    e_c_cum = np.cumsum(e_c_incremental)
    e_total_cum = e_q_cum + e_c_cum
    algo_resources_cum = np.cumsum(algo_resources_per_depth)

    total_quantum_energy = float(np.sum(e_q_incremental))
    total_classical_energy = float(np.sum(e_c_incremental))
    total_energy = float(np.sum(e_q_incremental + e_c_incremental))
    algo_resources_total = float(np.sum(algo_resources_per_depth))

    print("\n=== Energy Accounting (incremental & cumulative) ===")
    print(f"Total quantum energy     = {total_quantum_energy:.6e} J")
    print(f"Total classical energy   = {total_classical_energy:.6e} J")
    print(f" → Total energy          = {total_energy:.6e} J")
    print(f"Total classical FLOPs    = {classical_flops_total:.3e}")
    print(f"Total algorithmic res.   = {algo_resources_total:.3e}\n")

    # -------------------------------------------------------
    # Derived metrics & efficiencies
    # -------------------------------------------------------
    metric_per_point = np.clip(1.0 - local_errors, 0.0, 1.0)  # 𝓜 = 1 - Err

    # QPE success per point based on ΔE(n*)
    qpe_list = []
    for eps in local_errors:
        qpe = qpe_success_from_vqe_error(
            epsilon=float(eps),
            gap_delta=float(gap_delta),
            t_bits=int(t_bits),
            s_bits=int(s_bits),
            p_eff=float(p_eff),
            T_total=float(T_total),
            T2=float(T2)
        )
        qpe_list.append(qpe)
    qpe_success = np.array([q["P_success"] for q in qpe_list], dtype=float)

    print("=== Per-point summary ===")
    for n, evqe, e_exact, err, m, eq, ec, r, pq in zip(
            nqbits, sim_results, exact_per_size, local_errors, metric_per_point,
            e_q_cum, e_c_cum, algo_resources_cum, qpe_success):
        print(f"n = {n:2d} | EVQE = {evqe: .6f} | E0 = {e_exact: .6f} | "
              f"Err = {err: .3e} | M = {m: .3f} | "
              f"E_Q_hw(cum) = {eq: .3e} J | E_class(cum) = {ec: .3e} J | "
              f"R_ALGO(cum) = {r: .3e} | P_QPE = {pq: .3f}")

    # -------------------------------------------------------
    # Plots (exactly the requested set)
    # -------------------------------------------------------
    sizes = np.array(nqbits, dtype=int)

    # 1) Slide 22: linear extrapolation plot (n vs energy)
    plot_slide22_extrapolation(nqbits, sim_results, sim_std,
                               projected_results, known_size, known_res)

    # 2) Slide 22-style plot: energy vs cumulative total energy
    plot_slide22_vs_energy(e_total_cum, sim_results, sizes, known_res, known_size)

    # 3) HW efficiency vs metric (no n* in data, only measured sizes)
    plot_hw_eff_vs_metric(metric_per_point, e_q_cum, sizes, gap_delta)

    # 4) ALGO efficiency vs metric
    plot_algo_eff_vs_metric(metric_per_point, algo_resources_cum, sizes, gap_delta)

    # 5) HW efficiency vs HW energy cost
    plot_hw_eff_vs_hw_energy(e_q_cum, metric_per_point, sizes)

    # 6) ALGO efficiency vs ALGO energy cost
    plot_algo_eff_vs_algo_energy(algo_resources_cum, metric_per_point, sizes)

    # 7) HW-quantum and HW-classical on dual y-axis vs n
    plot_hw_quantum_vs_classical(nqbits, e_q_incremental, e_c_incremental)

    # 8) Metric (𝓜 = 1 − Err) vs algorithmic resources with Err on dual y-axis
    plot_metric_and_err_vs_algo_resources(algo_resources_cum, metric_per_point,
                                          local_errors, sizes, gap_delta)

    # 9) QPE success probability vs total cumulative energy
    plot_qpe_success_vs_energy(e_total_cum, qpe_success, sizes, gap_delta)

    # -------------------------------------------------------
    # Return everything useful for post-processing (incl. QPE)
    # -------------------------------------------------------
    return {
        "sim_results": sim_results,
        "sim_std": np.asarray(sim_std, dtype=float),
        "projected_results": projected_results,
        "projected_error": float(projected_error),
        "projected_error_bar": float(projected_error_bar),
        "local_errors": local_errors,
        "metric_per_point": metric_per_point,
        "algo_resources_per_depth": algo_resources_per_depth,
        "algo_resources_cum": algo_resources_cum,
        "e_q_incremental": e_q_incremental,
        "e_c_incremental": e_c_incremental,
        "e_q_cum": e_q_cum,
        "e_c_cum": e_c_cum,
        "e_total_cum": e_total_cum,
        "total_quantum_energy": total_quantum_energy,
        "total_classical_energy": total_classical_energy,
        "total_energy": total_energy,
        # QPE outputs
        "qpe_success": qpe_success,
        "qpe_constants": {
            "gap_delta": float(gap_delta),
            "t_bits": int(t_bits),
            "s_bits": int(s_bits),
            "p_eff": float(p_eff),
            "T_total": float(T_total),
            "T2": float(T2),
        }
    }


# ===========================================================
#  Entry point for multiprocessing
# ===========================================================
if __name__ == '__main__':
    # benchmark(nqbits, depths, rnds, ansatz, observe, noise_params, nshots, thermal_size, thermodynamic_limit, hw):
    # code to generate result. Especially required for using multiprocessing correctly

    # n from 4 to 7 only (3 removed entirely)
    nqbits = [4, 5, 6, 7]
    deps = [4, 5, 6, 7]  # ensure the depths are same in the number as qubits
    rnd = 5  # random seeds
    noise_p = [0.0001, -1]
    therm_size = 8  # size of the system to be used for extrapolation (n*)

    # Example QPE settings (tune as needed)
    results = benchmark(
        nqbits, deps, rnd, 'HVA', 'Heisenberg',
        noise_p, 1000, therm_size, 'supercond',
        efficiency_flops_per_watt=1e10,
        t_bits=6, s_bits=2,
        p_eff=0.0, T_total=0.0, T2=float("inf")
    )
    print("Completed")
