import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------------------------------------
# --- Load NPZs ---
# -----------------------------------------------------------
npz_file_rya = "benchmark_results/RYA_best_results.npz"
npz_file_hva = "benchmark_results/HVA_best_results.npz"


def load_npz(path):
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


data_rya = load_npz(npz_file_rya)
data_hva = load_npz(npz_file_hva)


# -----------------------------------------------------------
# --- Helpers for extrapolation plot ---
# -----------------------------------------------------------
def get_slide22_args(data):
    return dict(
        problem_sizes=data["nqbits"],
        energies=data["sim_results"],
        errors=data["sim_std"],
        n_star=8,
        exact_E_star=-13.499730394751566,
        depths=data["depths"],
    )


args_rya = get_slide22_args(data_rya)
args_hva = get_slide22_args(data_hva)


# Linear model
def model(n, m, b):
    return m * n + b


# SciPy weighted linear fit
def scipy_fit(problem_sizes, energies, errors):
    popt, pcov = curve_fit(
        model,
        problem_sizes,
        energies,
        sigma=errors,
        absolute_sigma=True,
        maxfev=10000,
    )
    m, b = popt
    return m, b, pcov


# Confidence band
def confidence_band(x_eval, m, b, cov):
    X = np.vstack([x_eval, np.ones_like(x_eval)]).T
    y_mean = model(x_eval, m, b)
    y_var = np.einsum("ij,jk,ik->i", X, cov, X)
    y_std = np.sqrt(y_var)
    return y_mean, y_mean - y_std, y_mean + y_std


# Full plot for extrapolation
def plot_extrapolation(ax, args, color, label, band_color, marker):
    problem_sizes = np.asarray(args["problem_sizes"], float)
    energies = np.asarray(args["energies"], float)
    errors = np.asarray(args["errors"], float)
    depths = np.asarray(args["depths"], float)

    n_star = float(args["n_star"])
    exact_E_star = args["exact_E_star"]

    # Fit
    m, b, cov = scipy_fit(problem_sizes, energies, errors)

    # Extrapolated value
    E_star = model(n_star, m, b)
    sigma_E_star = np.sqrt([n_star, 1.0] @ cov @ [n_star, 1.0])

    # Fit line
    x_fit = np.linspace(problem_sizes.min(), n_star, 200)
    y_mean, y_low, y_high = confidence_band(x_fit, m, b, cov)

    # Plot data
    ax.errorbar(
        problem_sizes,
        energies,
        yerr=errors,
        fmt=marker,
        color=color,
        ecolor=color,
        capsize=5,
        markersize=12,
        label=f"{label} data",
        zorder=5,
    )
    ax.scatter(problem_sizes, energies, s=130, c=color,
               edgecolors="black", zorder=6)

    # Band
    ax.fill_between(
        x_fit, y_low, y_high,
        color=band_color, alpha=0.20,
        label="_nolegend_",
        zorder=1,
    )

    # Weighted fit
    ax.plot(
        x_fit, y_mean, "--",
        linewidth=2.5,
        color=band_color,
        label=f"{label} weighted fit",
        zorder=2,
    )

    # Extrapolated point
    ax.scatter(
        [n_star], [E_star],
        s=200, color=band_color,
        edgecolors="black",
        zorder=7,
        label=rf"{label} extrapolated at $n^* = {int(n_star)}$",
    )

    return depths, exact_E_star


# -----------------------------------------------------------
# --- GLOBAL STYLING ---
# -----------------------------------------------------------
plt.rcParams.update({
    "font.size": 22,
    "axes.labelsize": 28,
    "axes.titlesize": 30,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,
})


# -----------------------------------------------------------
# --- PLOT 1: Extrapolation ---
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 10))

depths_rya, exact_E_star = plot_extrapolation(
    ax, args_rya, "tab:blue", "RYA", "tab:cyan", "o"
)
depths_hva, _ = plot_extrapolation(
    ax, args_hva, "tab:orange", "HVA", "tab:red", "s"
)

# Exact ground state line
ax.axhline(
    exact_E_star,
    color="gray",
    linestyle="dotted",
    linewidth=2.0,
    label=rf"Exact $E_0(n^* = 8)$",
)

# X-axis ticks include n* = 8
all_n = np.unique(
    np.concatenate([args_rya["problem_sizes"], args_hva["problem_sizes"], [args_rya["n_star"]]])
)
ax.set_xticks(all_n)
ax.set_xticklabels([str(int(n)) for n in all_n])
ax.set_xlabel(r"Problem size $n$")
ax.set_ylabel(r"Energy $E_{\mathrm{VQE}}(n)$")

# Top axis: depths
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(args_rya["problem_sizes"])
ax2.set_xticklabels([str(int(d)) for d in depths_rya])
ax2.set_xlabel(r"$N_{\mathrm{layers}}$", fontsize=26, labelpad=20)

ax.legend(loc="center left", bbox_to_anchor=(1, 0.50))
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# --- PLOT 2: Cumulative Energy vs Problem Size (RYA & HVA)
#        + dual x-axis (depth)
# -----------------------------------------------------------

# Extract sizes, energies, and depths
n_rya  = np.atleast_1d(data_rya["nqbits"])
E_rya  = np.atleast_1d(data_rya["E_cum_total"])
d_rya  = np.atleast_1d(data_rya["depths"])

n_hva  = np.atleast_1d(data_hva["nqbits"])
E_hva  = np.atleast_1d(data_hva["E_cum_total"])
d_hva  = np.atleast_1d(data_hva["depths"])

# Sort by problem size
order_rya = np.argsort(n_rya)
order_hva = np.argsort(n_hva)

n_rya, E_rya, d_rya = n_rya[order_rya], E_rya[order_rya], d_rya[order_rya]
n_hva, E_hva, d_hva = n_hva[order_hva], E_hva[order_hva], d_hva[order_hva]

fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.25

# RYA bars
ax.bar(
    n_rya - bar_width/2,
    E_rya,
    width=bar_width,
    alpha=0.60,
    color="tab:cyan",
    edgecolor="black",
    label="RYA",
)

# HVA bars
ax.bar(
    n_hva + bar_width/2,
    E_hva,
    width=bar_width,
    alpha=0.60,
    color="tab:red",
    edgecolor="black",
    label="HVA",
)

# Primary X-axis
ax.set_xlabel(r"Problem Size $n$", fontsize=24)
ax.set_ylabel("Cumulative Energy Consumption (J)", fontsize=24)
ax.set_xticks(n_rya)
ax.set_xticklabels([str(int(n)) for n in n_rya], fontsize=24)
ax.tick_params(axis="y", labelsize=22)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=24)

# Secondary X-axis (depth)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())  # match primary axis
ax2.set_xticks(n_rya)
ax2.set_xticklabels([str(int(d)) for d in d_rya], fontsize=24)
ax2.set_xlabel(r"$N_{\mathrm{layers}}$", fontsize=24, labelpad=12)

plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# --- PLOT 3: Quantum vs Classical Energy Breakdown
#        + dual x-axis (depth)
# -----------------------------------------------------------

def get_energy_breakdown(data):
    n  = np.atleast_1d(data["nqbits"])
    Eq = np.atleast_1d(data["E_q_point"])
    Ec = np.atleast_1d(data["E_c_point"])
    d  = np.atleast_1d(data["depths"])
    return n, Eq, Ec, d

# Extract RYA and HVA data
n_rya, Eq_rya, Ec_rya, d_rya = get_energy_breakdown(data_rya)
n_hva, Eq_hva, Ec_hva, d_hva = get_energy_breakdown(data_hva)

# Sort by n
order_rya = np.argsort(n_rya)
order_hva = np.argsort(n_hva)

n_rya, Eq_rya, Ec_rya, d_rya = (
    n_rya[order_rya],
    Eq_rya[order_rya],
    Ec_rya[order_rya],
    d_rya[order_rya],
)
n_hva, Eq_hva, Ec_hva, d_hva = (
    n_hva[order_hva],
    Eq_hva[order_hva],
    Ec_hva[order_hva],
    d_hva[order_hva],
)

fig, ax1 = plt.subplots(figsize=(14, 7))

# ---- Quantum (left axis)
ax1.set_xlabel("Problem size $n$", fontsize=24)
ax1.set_ylabel("Quantum energy $E_q$ (J)", fontsize=24, color="tab:blue")

ax1.plot(n_rya, Eq_rya, "-o", color="tab:blue", linewidth=3, markersize=10, label="RYA Quantum")
ax1.plot(n_hva, Eq_hva, "-s", color="tab:blue", linewidth=3, markersize=10, label="HVA Quantum")

ax1.tick_params(axis="y", labelsize=22)
ax1.tick_params(axis="x", labelsize=22)
ax1.grid(True, alpha=0.3)

# ---- Classical (right axis)
ax2 = ax1.twinx()
ax2.set_ylabel("Classical energy $E_c$ (J)", fontsize=24, color="tab:orange")

ax2.plot(n_rya, Ec_rya, "--o", color="tab:orange", linewidth=3, markersize=10, label="RYA Classical")
ax2.plot(n_hva, Ec_hva, "--s", color="tab:orange", linewidth=3, markersize=10, label="HVA Classical")

ax2.tick_params(axis="y", labelsize=22)

# ---- Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=24, loc="upper left")

# ---- Secondary x-axis for depth
ax3 = ax1.twiny()
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks(n_rya)
ax3.set_xticklabels([str(int(d)) for d in d_rya], fontsize=24)
ax3.set_xlabel(r"$N_{layers}$", fontsize=24, labelpad=10)

#plt.tight_layout()
plt.savefig("quantum_classical_energy_breakdown_rya_hva.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# --- PLOT: Algorithmic Efficiency vs Metric (RYA + HVA)
# -----------------------------------------------------------

def get_efficiency_data(data):
    n = np.atleast_1d(data["nqbits"])
    M = np.atleast_1d(data["metric_per_point"])   # Metric M = 1 - Err
    R = np.atleast_1d(data["E_point_total"])      # Energy consumption (resource)
    eff = M / R                                   # Efficiency
    return n, M, eff


# Extract for RYA and HVA
n_rya, M_rya, eff_rya = get_efficiency_data(data_rya)
n_hva, M_hva, eff_hva = get_efficiency_data(data_hva)

# Sort (optional, for smooth lines)
ord_rya = np.argsort(M_rya)
ord_hva = np.argsort(M_hva)

n_rya, M_rya, eff_rya = n_rya[ord_rya], M_rya[ord_rya], eff_rya[ord_rya]
n_hva, M_hva, eff_hva  = n_hva[ord_hva], M_hva[ord_hva], eff_hva[ord_hva]

plt.figure(figsize=(14, 9))

# -----------------------------------------------------------
# RYA curve
# -----------------------------------------------------------
plt.plot(
    M_rya, eff_rya,
    "-o", color="tab:blue", linewidth=3, markersize=10,
    label="RYA"
)

for x, y, n in zip(M_rya, eff_rya, n_rya):
    plt.annotate(f"{n}", (x, y),
                 xytext=(6, 6), textcoords="offset points",
                 fontsize=24)

# -----------------------------------------------------------
# HVA curve
# -----------------------------------------------------------
plt.plot(
    M_hva, eff_hva,
    "-s", color="tab:red", linewidth=3, markersize=10,
    label="HVA"
)

for x, y, n in zip(M_hva, eff_hva, n_hva):
    plt.annotate(f"{n}", (x, y),
                 xytext=(6, 6), textcoords="offset points",
                 fontsize=24)

# -----------------------------------------------------------
# Axes
# -----------------------------------------------------------
plt.yscale("log")  # <-- Logarithmic y-axis

plt.xlabel("Metric  $\mathcal{M} = 1 - \\mathrm{Err}$", fontsize=28)
plt.ylabel("Efficiency", fontsize=24)

plt.ticklabel_format(axis="x", style="plain")
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.grid(True, alpha=0.3, which="both")
plt.legend(fontsize=24)

#plt.title("Algorithmic Efficiency vs Metric\nRYA vs HVA",
#          fontsize=32, pad=20)

# Flip x-axis
plt.gca().invert_xaxis()

plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# --- PLOT: Efficiency vs Problem Size (RYA + HVA)
# -----------------------------------------------------------

def get_efficiency_data(data):
    n = np.atleast_1d(data["nqbits"])
    M = np.atleast_1d(data["metric_per_point"])   # Metric M
    R = np.atleast_1d(data["E_point_total"])      # Resource
    eff = M / R                                   # Efficiency
    return n, M, eff


# Extract for RYA and HVA
n_rya, M_rya, eff_rya = get_efficiency_data(data_rya)
n_hva, M_hva, eff_hva = get_efficiency_data(data_hva)

# Sort by problem size
order_rya = np.argsort(n_rya)
order_hva = np.argsort(n_hva)

n_rya, M_rya, eff_rya = n_rya[order_rya], M_rya[order_rya], eff_rya[order_rya]
n_hva, M_hva, eff_hva = n_hva[order_hva], M_hva[order_hva], eff_hva[order_hva]

plt.figure(figsize=(14, 9))

# -----------------------------------------------------------
# RYA curve
# -----------------------------------------------------------
plt.plot(
    n_rya, eff_rya,
    "--o", color="tab:blue", linewidth=3, markersize=10,
    label="RYA"
)

# annotate with metric values
for x, y, M in zip(n_rya, eff_rya, M_rya):
    plt.annotate(f"{M:.2f}", (x, y),
                 xytext=(6, 6), textcoords="offset points",
                 fontsize=24)

# -----------------------------------------------------------
# HVA curve
# -----------------------------------------------------------
plt.plot(
    n_hva, eff_hva,
    "-s", color="tab:red", linewidth=3, markersize=10,
    label="HVA"
)

# annotate with metric values
for x, y, M in zip(n_hva, eff_hva, M_hva):
    plt.annotate(f"{M:.2f}", (x, y),
                 xytext=(6, 6), textcoords="offset points",
                 fontsize=24)

# -----------------------------------------------------------
# Axes / formatting
# -----------------------------------------------------------
plt.yscale("log")

plt.xlabel("Problem size $n$", fontsize=24)
plt.ylabel(r"$Efficiency = \frac{Metric}{Resource}$", fontsize=24)

plt.xticks(n_rya, fontsize=20)
plt.yticks(fontsize=20)

plt.grid(True, alpha=0.3, which="both")
plt.legend(fontsize=22)

#plt.title("Efficiency vs Problem Size\nRYA vs HVA", fontsize=32, pad=20)
plt.savefig("efficiency_vs_problem_size_rya_hva.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


# -----------------------------------------
# Build combined plot
# -----------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 10))

# === LEFT AXIS: VQE ENERGY (primary plot) ===
depths_rya, exact_E_star = plot_extrapolation(
    ax1, args_rya, "tab:blue", "RYA", "tab:cyan", "o"
)
depths_hva, _ = plot_extrapolation(
    ax1, args_hva, "tab:orange", "HVA", "tab:red", "s"
)

ax1.axhline(
    exact_E_star,
    linestyle="dotted", linewidth=2.0, color="gray",
    label=r"Exact $E_{\mathrm{gs}}~at~n^* = 8$"
)

ax1.set_xlabel("Problem size $n$", fontsize=26)
ax1.set_ylabel(r"Energy $E_{\mathrm{VQE}}(n)$", fontsize=26)

all_n = np.unique(np.concatenate([n_rya, n_hva, [args_rya["n_star"]]]))
ax1.set_xticks(all_n)
ax1.set_xticklabels([str(int(n)) for n in all_n], fontsize=22)
ax1.tick_params(axis="y", labelsize=22)

# === TOP X-AXIS: DEPTHS ===
ax_top = ax1.twiny()
ax_top.set_xlim(ax1.get_xlim())
ax_top.set_xticks(n_rya)
ax_top.set_xticklabels([str(int(d)) for d in d_rya], fontsize=22)
ax_top.set_xlabel(r"$N_{\mathrm{layers}}$", fontsize=24, labelpad=15)

# -----------------------------------------
# === RIGHT AXIS: CUMULATIVE ENERGY (bars) ===
# -----------------------------------------
ax2 = ax1.twinx()
ax2.set_ylabel("Cumulative Energy Consumption (J)", fontsize=26, color="black")
ax2.tick_params(axis="y", labelsize=22)

# Bars in background: extremely transparent + behind other artists
bar_width = 0.35

ax2.bar(
    n_rya - bar_width/2,
    E_rya,
    width=bar_width,
    color="tab:cyan",
    alpha=0.18,
    edgecolor="none",
    zorder=0,
    label="RYA cumulative energy"
)

ax2.bar(
    n_hva + bar_width/2,
    E_hva,
    width=bar_width,
    color="tab:red",
    alpha=0.18,
    edgecolor="none",
    zorder=0,
    label="HVA cumulative energy"
)

# Keep bars behind everything
for bar in ax2.containers:
    for patch in bar:
        patch.set_zorder(0)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    fontsize=20,
    ncols=2,
    loc="center left",
    bbox_to_anchor=(0,1.3)
)
plt.savefig("combined_rya_hva_extrapolation_cumulative_energy.png", dpi=300, bbox_inches='tight')
#plt.tight_layout()
plt.show()
