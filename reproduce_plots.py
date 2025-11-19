import numpy as np
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

def load_npz(path):
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}

def main(npz_file):
    data = load_npz(npz_file)

    # Unpack data
    nqbits = data["nqbits"]
    depths = data["depths"]
    sim_results = data["sim_results"]
    sim_std = data["sim_std"]
    exact_per_size = data["exact_per_size"]
    local_errors = data["local_errors"]
    metric_per_point = data["metric_per_point"]
    algo_resources_point = data["algo_resources_point"]
    algo_resources_cum = data["algo_resources_cum"]
    E_q_point = data["E_q_point"]
    E_c_point = data["E_c_point"]
    E_point_total = data["E_point_total"]
    E_cum_total = data["E_cum_total"]
    projected_value = data["projected_value"]
    projected_error = data["projected_error"]
    regression_coefficients = data["regression_coefficients"]
    P_qpe_per_point = data["P_qpe_per_point"]
    P_qpe_star = data["P_qpe_star"]
    total_quantum_energy = data["total_quantum_energy"]
    total_classical_energy = data["total_classical_energy"]
    total_energy = data["total_energy"]

    # For Slide-22 style plot
    projected_results = {
        "extrapolated_value": projected_value,
        "extrapolated_error": projected_error,
    }
    if regression_coefficients is not None:
        projected_results["regression_coefficients"] = regression_coefficients

    n_star = 8
    exact_E_star = -13.499730394751566

    # 1. Linear extrapolation vs problem size
    plot_linear_extrapolation_sizes(
        problem_sizes=nqbits,
        energies=sim_results,
        errors=sim_std,
        projected_results=projected_results,
        n_star=n_star,
        exact_E_star=exact_E_star,
        depths=depths,
        P_qpe_star=P_qpe_star,
        P_qpe_points=P_qpe_per_point,
    )
    
    return data

"""
    # 2. VQE energy vs cumulative total energy
    plot_energy_vs_cumulative_energy(
        problem_sizes=nqbits,
        energies=sim_results,
        cumulative_energies=E_cum_total,
        P_qpe=P_qpe_per_point,
    )

    # 3. HW efficiency vs metric (pointwise)
    plot_hw_efficiency_vs_metric(
        problem_sizes=nqbits,
        metric=metric_per_point,
        hw_energy_point=E_q_point,
        P_qpe=P_qpe_per_point,
    )

    # 4. ALGO efficiency vs metric (pointwise)
    plot_algo_efficiency_vs_metric(
        problem_sizes=nqbits,
        metric=metric_per_point,
        algo_resources_point=algo_resources_point,
        P_qpe=P_qpe_per_point,
    )

    # 5. HW efficiency vs HW energy cost per point
    plot_hw_efficiency_vs_hw_energy(
        problem_sizes=nqbits,
        metric=metric_per_point,
        hw_energy_point=E_q_point,
        P_qpe=P_qpe_per_point,
    )

    # 6. ALGO efficiency vs ALGO resource cost per point
    plot_algo_efficiency_vs_algo_resources(
        problem_sizes=nqbits,
        metric=metric_per_point,
        algo_resources_point=algo_resources_point,
        P_qpe=P_qpe_per_point,
    )

    # 7. Quantum vs classical energy (per point, dual y-axis)
    plot_hw_quantum_vs_classical(
        problem_sizes=nqbits,
        E_q_point=E_q_point,
        E_c_point=E_c_point,
        P_qpe=P_qpe_per_point,
    )

    # 8. Metric and Err vs cumulative algorithmic resources (dual y-axis)
    plot_metric_and_err_vs_algo_resources(
        problem_sizes=nqbits,
        metric=metric_per_point,
        errors=local_errors,
        algo_resources_cum=algo_resources_cum,
        P_qpe=P_qpe_per_point,
    )

    # 9. QPE success probability vs total cumulative energy
    plot_qpe_success_vs_total_energy(
        problem_sizes=nqbits,
        P_qpe=P_qpe_per_point,
        cumulative_energy=E_cum_total,
        P_qpe_star=P_qpe_star,
    )
"""
if __name__ == "__main__":
    # Set your NPZ file path here
    npz_file = "benchmark_results/RYA_best_results.npz"
    dattaa = main(npz_file)