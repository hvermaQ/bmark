import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_linear_extrapolation_sizes

# --- User: set your .npz filenames here ---
npz_file_rya = "benchmark_results/RYA_results.npz"  # Replace with actual filename
npz_file_hva = "benchmark_results/HVA_results.npz"  # Replace with actual filename

# --- Load data ---
def load_npz(path):
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}

data_rya = load_npz(npz_file_rya)
data_hva = load_npz(npz_file_hva)

# --- Extract relevant data for slide 22 plot ---
def get_slide22_args(data):
    return dict(
        problem_sizes=data["nqbits"],
        energies=data["sim_results"],
        errors=data["sim_std"],
        projected_results={
            "regression_coefficients": data["regression_coefficients"],
            "extrapolated_value": data["projected_value"],
            "extrapolated_error": data["projected_error"],
        },
        n_star=8,
        exact_E_star=-13.499730394751566,
        depths=data["depths"],
        P_qpe_star=data["P_qpe_star"],
        P_qpe_points=data["P_qpe_per_point"],
    )

args_rya = get_slide22_args(data_rya)
args_hva = get_slide22_args(data_hva)

# --- Plot side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

plt.sca(axes[0])
plot_linear_extrapolation_sizes(**args_rya)
plt.title("RYA")

plt.sca(axes[1])
plot_linear_extrapolation_sizes(**args_hva)
plt.title("HVA")

plt.tight_layout()
plt.show()
