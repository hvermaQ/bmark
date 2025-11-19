import numpy as np
import matplotlib.pyplot as plt
from plotting import _annotate_pqpe_points

# --- User: set your .npz filenames here ---
npz_file_rya = "benchmark_results/RYA_best_results.npz"
npz_file_hva = "benchmark_results/HVA_best_results.npz"

def load_npz(path):
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}

data_rya = load_npz(npz_file_rya)
data_hva = load_npz(npz_file_hva)

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

# --- Plot both RYA and HVA on the same axes ---
fig, ax = plt.subplots(figsize=(15, 10))


def plot_extrapolation(ax, args, color, label, band_color, marker):
    # Unpack
    problem_sizes = np.asarray(args["problem_sizes"], dtype=float)
    energies = np.asarray(args["energies"], dtype=float)
    errors = np.asarray(args["errors"], dtype=float)
    depths = np.asarray(args["depths"], dtype=float)
    projected_results = args["projected_results"]
    n_star = args["n_star"]
    exact_E_star = args["exact_E_star"]
    m, b = projected_results["regression_coefficients"]
    E_star = projected_results["extrapolated_value"]
    sigma_E_star = projected_results.get("extrapolated_error", None)
    n_bootstrap = 10000

    rng = np.random.default_rng()
    n_points = len(problem_sizes)
    x_fit = np.linspace(min(problem_sizes), max(max(problem_sizes), n_star), 200)
    m_samples = []
    b_samples = []
    sigma = np.maximum(errors, 1e-12)
    w = 1.0 / (sigma ** 2)
    A = np.vstack([problem_sizes, np.ones_like(problem_sizes)]).T
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_points, size=n_points)
        x_boot = problem_sizes[idx]
        y_boot = energies[idx]
        w_boot = w[idx]
        A_boot = np.vstack([x_boot, np.ones_like(x_boot)]).T
        W_boot = np.diag(w_boot)
        ATA = A_boot.T @ W_boot @ A_boot
        ATy = A_boot.T @ W_boot @ y_boot
        try:
            m_b = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            continue
        m_samples.append(m_b[0])
        b_samples.append(m_b[1])
    m_samples = np.asarray(m_samples)
    b_samples = np.asarray(b_samples)
    if len(m_samples) == 0:
        y_fit_mean = m * x_fit + b
        y_fit_low = y_fit_mean
        y_fit_high = y_fit_mean
    else:
        y_fit_all = np.outer(m_samples, x_fit) + b_samples[:, None]
        y_fit_mean = np.mean(y_fit_all, axis=0)
        y_fit_std = np.std(y_fit_all, axis=0, ddof=1)
        y_fit_low = y_fit_mean - y_fit_std
        y_fit_high = y_fit_mean + y_fit_std
    y_fit_nominal = m * x_fit + b

    # Data with error bars
    ax.errorbar(
        problem_sizes,
        energies,
        yerr=errors,
        fmt=marker,
        color=color,
        ecolor=color,
        capsize=4,
        linestyle="none",
        label=f"{label} data",
        zorder=3,
    )
    # Scatter
    ax.scatter(
        problem_sizes,
        energies,
        c=color,
        s=80,
        edgecolors="black",
        zorder=4,
    )
    # Bootstrap band
    ax.fill_between(
        x_fit,
        y_fit_low,
        y_fit_high,
        color=band_color,
        alpha=0.18,
        label=f"{label} bootstrap $1\\sigma$ band",
        zorder=1,
    )
    # Nominal fit
    ax.plot(x_fit, y_fit_nominal, "--", color=band_color, label=f"{label} weighted fit", zorder=2)
    # Extrapolated point
    ax.scatter(
        [n_star],
        [E_star],
        s=120,
        color=band_color,
        edgecolors="black",
        zorder=5,
        label=rf"{label} extrapolated at $n^*={int(n_star)}$",
    )
    return depths, exact_E_star

# Plot RYA
depths_rya, exact_E_star = plot_extrapolation(
    ax, args_rya, color="tab:blue", label="RYA", band_color="tab:cyan", marker="o"
)
# Plot HVA
depths_hva, _ = plot_extrapolation(
    ax, args_hva, color="tab:orange", label="HVA", band_color="tab:red", marker="s"
)

# Exact line at E_0(n*) (only once)
ax.axhline(
    exact_E_star,
    color="gray",
    linestyle="dotted",
    linewidth=1.5,
    label=rf"Exact $E_0(n^*={int(args_rya['n_star'])})$",
    zorder=0,
)

ax.set_xlabel(r"Problem size $n$")
all_problem_sizes = np.unique(np.concatenate([args_rya["problem_sizes"], args_hva["problem_sizes"]]))
ax.set_xticks(all_problem_sizes)
ax.set_xticklabels([str(int(n)) for n in all_problem_sizes])
ax.set_ylabel(r"Energy $E_{\mathrm{VQE}}(n)$")
#ax.set_title("Linear extrapolation to $n^*$ (Slide 22 style): RYA vs HVA")

# Top twin x-axis for depth (RYA shown, HVA depths can be added if needed)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(args_rya["problem_sizes"])
ax2.set_xticklabels([str(int(d)) for d in depths_rya])
ax2.set_xlabel(r"$N_{layers}$", labelpad=30)

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
