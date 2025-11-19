# plotting.py
"""
Plotting utilities for energetic VQE benchmark.

Implements:
1. Slide-22 style plots:
   - Linear extrapolation vs problem size n (dual x-axis: n and depth)
   - VQE energy vs cumulative total energy

2. Slide-23 style plots:
   - Hardware efficiency vs metric (pointwise)
   - Algorithmic efficiency vs metric (pointwise)
   - Hardware efficiency vs hardware energy cost per point
   - Algorithmic efficiency vs algorithmic resource cost per point
   - Quantum vs classical energy vs problem size (dual y-axis)
   - Metric and Err vs cumulative algorithmic resources (dual y-axis)

3. QPE bridging:
   - QPE success probability vs total cumulative energy

Conventions:
- Problem size n is always encoded by color (no numeric labels near markers).
- Each point is annotated with its QPE success probability
  P_QPE = cos^2(Err/2) using LaTeX:  P_{QPE} = 0.87
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext

# --- Scientific plotting style (paper-ready) ---
plt.rcParams.update({
    "font.size": 16,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})


# ===========================================================
# Helpers
# ===========================================================
def _savefig(base):
    """Save current figure as PNG, then close."""
    #plt.tight_layout()
    #plt.savefig(f"{base}.png", bbox_inches="tight")
    #plt.show()
    plt.close()


def _annotate_pqpe_points(ax, x, y, P_qpe, y_offset_frac=0.03):
    """
    Annotate each point with P_QPE in LaTeX near (x_i, y_i).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : array-like
        Data coordinates of the points.
    P_qpe : array-like
        QPE success probabilities for each point.
    y_offset_frac : float
        Vertical offset as a fraction of the y-range.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    P_qpe = np.asarray(P_qpe, dtype=float)

    if len(x) == 0:
        return

    y_min, y_max = ax.get_ylim()
    dy = (y_max - y_min) * y_offset_frac

    for xi, yi, Pi in zip(x, y, P_qpe):
        y_annot = yi + dy
        # If label would go out of bounds, flip below
        if y_annot > y_max:
            y_annot = yi - dy
            va = "top"
        else:
            va = "bottom"

        ax.text(
            xi,
            y_annot,
            rf"$P_{{\mathrm{{QPE}}}}={Pi:.2f}$",
            fontsize=11,
            ha="center",
            va=va,
        )


def _set_log10_xaxis(ax):
    """Set a log10 x-axis with 10^k-style ticks."""
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(LogFormatterMathtext())


def _set_sci_xaxis(ax):
    """
    Force scientific notation on x-axis using 10^k (not 1e12),
    for linear axes.
    """
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))


# ===========================================================
# (1) Slide-22: linear extrapolation vs problem size n
# ===========================================================
def plot_linear_extrapolation_sizes(
    problem_sizes,
    energies,
    errors,
    projected_results,
    n_star,
    exact_E_star,
    depths,
    P_qpe_star,
    P_qpe_points,
    n_bootstrap=1000,
):
    """
    Slide-22 style:
    - x (bottom): problem size n (integers)
    - x (top): depth (user-specified list, integers)
    - y: VQE energy E_VQE(n)
    - weighted linear fit
    - bootstrap 1σ band for fit
    - extrapolated value at n*
    - horizontal line at exact E_0(n*)
    - P_QPE annotations at each size and at extrapolated point
    - Confidence interval for extrapolated P_QPE shown in text.

    Parameters
    ----------
    projected_results : dict
        Must contain:
        - "regression_coefficients" -> (m, b)
        - "extrapolated_value" -> E_star
        - "extrapolated_error" -> sigma_E_star (1σ)
    """
    problem_sizes = np.asarray(problem_sizes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    errors = np.asarray(errors, dtype=float)
    depths = np.asarray(depths, dtype=float)
    P_qpe_points = np.asarray(P_qpe_points, dtype=float)

    # Regression coefficients from extrapolation step
    m, b = projected_results["regression_coefficients"]
    E_star = projected_results["extrapolated_value"]
    sigma_E_star = projected_results.get("extrapolated_error", None)

    # --- Bootstrap band for the regression ---
    rng = np.random.default_rng()
    n_points = len(problem_sizes)
    x_fit = np.linspace(min(problem_sizes), max(max(problem_sizes), n_star), 200)

    m_samples = []
    b_samples = []

    # Precompute weights (inverse variance)
    # Avoid zero errors by lower-bounding.
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
        # Numerical guard
        try:
            m_b = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            continue

        m_samples.append(m_b[0])
        b_samples.append(m_b[1])

    m_samples = np.asarray(m_samples)
    b_samples = np.asarray(b_samples)

    # If bootstrap failed for some reason, fall back to single fit
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

    # Nominal best-fit line from (m, b)
    y_fit_nominal = m * x_fit + b

    fig, ax = plt.subplots(figsize=(8, 6))

    # Data with vertical error bars
    ax.errorbar(
        problem_sizes,
        energies,
        yerr=errors,
        fmt="o",
        color="tab:blue",
        ecolor="tab:blue",
        capsize=4,
        linestyle="none",
        label="VQE data",
    )

    # Re-plot as scatter (single color, n encoded in label text only)
    sc = ax.scatter(
        problem_sizes,
        energies,
        c="tab:blue",
        s=80,
        edgecolors="black",
        zorder=3,
    )

    # Bootstrap band
    ax.fill_between(
        x_fit,
        y_fit_low,
        y_fit_high,
        color="tab:red",
        alpha=0.2,
        label="Bootstrap $1\\sigma$ band",
        zorder=1,
    )

    # Nominal linear fit
    ax.plot(x_fit, y_fit_nominal, "r--", label="Weighted linear fit", zorder=2)

    # Extrapolated point at n*
    ax.scatter(
        [n_star],
        [E_star],
        s=120,
        color="red",
        edgecolors="black",
        zorder=4,
        label=rf"Extrapolated at $n^*={int(n_star)}$",
    )

    # Exact line at E_0(n*)
    ax.axhline(
        exact_E_star,
        color="gray",
        linestyle="dotted",
        linewidth=1.5,
        label=rf"Exact $E_0(n^*={int(n_star)})$",
    )

    ax.set_xlabel(r"Problem size $n$")
    # Force integer ticks for x-axis
    ax.set_xticks(problem_sizes)
    ax.set_xticklabels([str(int(n)) for n in problem_sizes])
    ax.set_ylabel(r"Energy $E_{\mathrm{VQE}}(n)$")
    ax.set_title("Linear extrapolation to $n^*$ (Slide 22)")

    # Top twin x-axis for depth
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(problem_sizes)
    ax2.set_xticklabels([str(int(d)) for d in depths])
    ax2.set_xlabel("Depth (layers)")

    # P_QPE annotations at each data point
    _annotate_pqpe_points(ax, problem_sizes, energies, P_qpe_points)

    # --- P_QPE at extrapolated point with confidence interval ---
    # (Removed P_QPE(n^*) text annotation as requested)

    # Also annotate P_QPE directly at the extrapolated marker
    _annotate_pqpe_points(ax, [n_star], [E_star], [P_qpe_star])

    ax.legend(loc="best")
    plt.show()
    plt.savefig("try_extra.png", bbox_inches="tight")
    #_savefig("slide22_linear_extrapolation")


# ===========================================================
# (2) Slide-22-like: VQE energy vs cumulative total energy
# ===========================================================
def plot_energy_vs_cumulative_energy(
    problem_sizes,
    energies,
    cumulative_energies,
    P_qpe,
):
    """
    Slide-22 variant:
    x: cumulative total energy (quantum + classical) [log10]
    y: VQE energy E_VQE(n)

    - Problem sizes encoded as color.
    - Each point annotated with P_QPE in LaTeX.
    """
    problem_sizes = np.asarray(problem_sizes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    cumulative_energies = np.asarray(cumulative_energies, dtype=float)
    P_qpe = np.asarray(P_qpe, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        cumulative_energies,
        energies,
        c=problem_sizes,
        cmap="viridis",
        s=80,
        edgecolors="black",
    )

    _set_log10_xaxis(ax)
    ax.set_xlabel(r"Cumulative total energy $E_{\mathrm{tot}}^{(\mathrm{cum})}$ (J)")
    ax.set_ylabel(r"Energy $E_{\mathrm{VQE}}(n)$")
    ax.set_title("VQE energy vs cumulative total energy (Slide 22 variant)")

    # Annotate P_QPE at each point
    _annotate_pqpe_points(ax, cumulative_energies, energies, P_qpe)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Problem size $n$")

    _savefig("slide22_energy_vs_cum_energy")


# ===========================================================
# (3) HW efficiency vs metric (pointwise)
# ===========================================================
def plot_hw_efficiency_vs_metric(
    problem_sizes,
    metric,
    hw_energy_point,
    P_qpe,
):
    """
    Hardware efficiency vs metric (Slide-23 style):

    x: metric  M = 1 - Err
    y: hardware efficiency M / E_q (per point)

    - markers encode problem size via color
    - Each point annotated with P_QPE
    """
    problem_sizes = np.asarray(problem_sizes, dtype=float)
    metric = np.asarray(metric, dtype=float)
    hw_energy_point = np.asarray(hw_energy_point, dtype=float)
    P_qpe = np.asarray(P_qpe, dtype=float)

    hw_eff = np.divide(
        metric,
        hw_energy_point,
        out=np.zeros_like(metric),
        where=hw_energy_point > 0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        metric,
        hw_eff,
        c=problem_sizes,
        cmap="viridis",
        s=80,
        edgecolors="black",
    )

    ax.set_xlabel(r"Metric $\mathcal{M} = 1 - \mathrm{Err}$")
    ax.set_ylabel(r"HW efficiency $\mathcal{M} / E_q$ (1/J)")
    ax.set_title("Hardware efficiency vs metric")

    _annotate_pqpe_points(ax, metric, hw_eff, P_qpe)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Problem size $n$")

    _savefig("hw_eff_vs_metric")


# ===========================================================
# (4) ALGO efficiency vs metric (pointwise)
# ===========================================================
def plot_algo_efficiency_vs_metric(
    problem_sizes,
    metric,
    algo_resources_point,
    P_qpe,
):
    """
    Algorithmic efficiency vs metric (Slide-23 style):

    x: metric  M = 1 - Err
    y: algo efficiency M / R (per point)

    where R = gates × iterations × shots × #Pauli.
    """
    problem_sizes = np.asarray(problem_sizes, dtype=float)
    metric = np.asarray(metric, dtype=float)
    algo_resources_point = np.asarray(algo_resources_point, dtype=float)
    P_qpe = np.asarray(P_qpe, dtype=float)

    algo_eff = np.divide(
        metric,
        algo_resources_point,
        out=np.zeros_like(metric),
        where=algo_resources_point > 0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        metric,
        algo_eff,
        c=problem_sizes,
        cmap="viridis",
        s=80,
        edgecolors="black",
    )

    ax.set_xlabel(r"Metric $\mathcal{M} = 1 - \mathrm{Err}$")
    ax.set_ylabel(r"Algo efficiency $\mathcal{M} / \mathcal{R}$ (1/resource)")
    ax.set_title("Algorithmic efficiency vs metric")

    _annotate_pqpe_points(ax, metric, algo_eff, P_qpe)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Problem size $n$")

    _savefig("algo_eff_vs_metric")


# ===========================================================
# (5) HW efficiency vs HW energy cost per point
# ===========================================================
def plot_hw_efficiency_vs_hw_energy(
    problem_sizes,
    metric,
    hw_energy_point,
    P_qpe,
):
    """
    Hardware efficiency vs HW energy cost (per point):

    x: hardware energy E_q (J) [log10]
    y: HW efficiency M / E_q

    - markers encode problem size via color
    - Each point annotated with P_QPE
    """
    problem_sizes = np.asarray(problem_sizes, dtype=float)
    metric = np.asarray(metric, dtype=float)
    hw_energy_point = np.asarray(hw_energy_point, dtype=float)
    P_qpe = np.asarray(P_qpe, dtype=float)

    hw_eff = np.divide(
        metric,
        hw_energy_point,
        out=np.zeros_like(metric),
        where=hw_energy_point > 0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        hw_energy_point,
        hw_eff,
        c=problem_sizes,
        cmap="viridis",
        s=80,
        edgecolors="black",
    )

    _set_log10_xaxis(ax)
    ax.set_xlabel(r"Hardware energy cost $E_q$ (J)")
    ax.set_ylabel(r"HW efficiency $\mathcal{M} / E_q$ (1/J)")
    ax.set_title("Hardware efficiency vs hardware energy cost")

    _annotate_pqpe_points(ax, hw_energy_point, hw_eff, P_qpe)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Problem size $n$")

    _savefig("hw_eff_vs_hw_energy")


# ===========================================================
# (6) ALGO efficiency vs ALGO resource cost per point
# ===========================================================
def plot_algo_efficiency_vs_algo_resources(
    problem_sizes,
    metric,
    algo_resources_point,
    P_qpe,
):
    """
    Algorithmic efficiency vs algorithmic resource cost (per point):

    x: algorithmic resources R (gates × iterations × shots × #Pauli) [log10]
    y: algo efficiency M / R
    """
    problem_sizes = np.asarray(problem_sizes, dtype=float)
    metric = np.asarray(metric, dtype=float)
    algo_resources_point = np.asarray(algo_resources_point, dtype=float)
    P_qpe = np.asarray(P_qpe, dtype=float)

    algo_eff = np.divide(
        metric,
        algo_resources_point,
        out=np.zeros_like(metric),
        where=algo_resources_point > 0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        algo_resources_point,
        algo_eff,
        c=problem_sizes,
        cmap="viridis",
        s=80,
        edgecolors="black",
    )

    _set_log10_xaxis(ax)
    ax.set_xlabel(
        r"Algorithmic resources $\mathcal{R}$ "
        r"(gates × iterations × shots × \#Pauli)"
    )
    ax.set_ylabel(r"Algo efficiency $\mathcal{M} / \mathcal{R}$ (1/resource)")
    ax.set_title("Algorithmic efficiency vs algorithmic resource cost")

    _annotate_pqpe_points(ax, algo_resources_point, algo_eff, P_qpe)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Problem size $n$")

    _savefig("algo_eff_vs_algo_energy")


# ===========================================================
# (7) HW quantum vs classical energy per problem size
# ===========================================================
def plot_hw_quantum_vs_classical(
    problem_sizes,
    E_q_point,
    E_c_point,
    P_qpe,
):
    """
    Quantum vs classical incremental energy per problem size (dual y-axis):

    x: problem size n (integers)
    y1: quantum energy (J)
    y2: classical energy (J)

    - P_QPE annotated at each point on the quantum curve.
    """
    problem_sizes = np.asarray(problem_sizes, dtype=float)
    E_q_point = np.asarray(E_q_point, dtype=float)
    E_c_point = np.asarray(E_c_point, dtype=float)
    P_qpe = np.asarray(P_qpe, dtype=float)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color_q = "tab:blue"
    color_c = "tab:orange"

    ax1.set_xlabel(r"Problem size $n$")
    ax1.set_ylabel(r"Quantum energy $E_q$ (J)", color=color_q)
    l1 = ax1.plot(
        problem_sizes, E_q_point, "o-",
        color=color_q, label="Quantum",
    )
    ax1.tick_params(axis="y", labelcolor=color_q)

    # annotate P_QPE at each quantum point
    _annotate_pqpe_points(ax1, problem_sizes, E_q_point, P_qpe)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Classical energy $E_c$ (J)", color=color_c)
    l2 = ax2.plot(
        problem_sizes, E_c_point, "s--",
        color=color_c, label="Classical",
    )
    ax2.tick_params(axis="y", labelcolor=color_c)

    ax1.set_title("Quantum vs classical energy per problem size")
    # Force integer ticks for x-axis
    ax1.set_xticks(problem_sizes)
    ax1.set_xticklabels([str(int(n)) for n in problem_sizes])

    # Combined legend
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    _savefig("hw_quantum_vs_classical")


# ===========================================================
# (8) Metric and Err vs cumulative algorithmic resources
# ===========================================================
def plot_metric_and_err_vs_algo_resources(
    problem_sizes,
    metric,
    errors,
    algo_resources_cum,
    P_qpe,
):
    """
    Metric and Err vs cumulative algorithmic resources (Slide-23 diagnostic):

    x: cumulative algorithmic resources (sum over sizes), with sci 10^k notation
    y1 (left): metric M = 1 - Err
    y2 (right): Err

    - Metric and Error share the same [0, 1] linear scale and gridlines.
    - Error ticks at 0, 0.2, ..., 1.0.
    - P_QPE annotated at each point (on metric curve).
    """
    problem_sizes = np.asarray(problem_sizes, dtype=float)
    metric = np.asarray(metric, dtype=float)
    # Always compute errors as 1 - metric to ensure alignment
    metric = np.asarray(metric, dtype=float)
    errors = 1.0 - metric
    algo_resources_cum = np.asarray(algo_resources_cum, dtype=float)
    P_qpe = np.asarray(P_qpe, dtype=float)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color_m = "tab:blue"
    color_e = "tab:red"

    # Metric vs cumulative resources
    l1 = ax1.plot(
        algo_resources_cum,
        metric,
        "o-",
        color=color_m,
        label=r"Metric $\mathcal{M} = 1 - \mathrm{Err}$",
    )
    ax1.set_xlabel(
        r"Cumulative algorithmic resources $\mathcal{R}^{(\mathrm{cum})}$"
    )
    ax1.set_ylabel(r"Metric $\mathcal{M}$", color=color_m)
    ax1.tick_params(axis="y", labelcolor=color_m)

    # Enforce [0,1] range and ticks
    ax1.set_ylim(0.0, 1.0)
    ax1.set_yticks(np.linspace(0.0, 1.0, 6))

    # Error vs cumulative resources (second axis)
    ax2 = ax1.twinx()
    l2 = ax2.plot(
        algo_resources_cum,
        errors,
        "s--",
        color=color_e,
        label=r"Error $\mathrm{Err} = 1 - \mathcal{M}$",
    )
    ax2.set_ylabel(r"Error $\mathrm{Err} = 1 - \mathcal{M}$", color=color_e)
    ax2.tick_params(axis="y", labelcolor=color_e)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks(np.linspace(0.0, 1.0, 6))

    # Shared gridlines come from ax1 because both y-lims/ticks are identical
    _set_sci_xaxis(ax1)

    ax1.set_title("Metric and error vs cumulative algorithmic resources")

    # Combined legend (no P_QPE entry)
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    # Annotate P_QPE at each metric point
    _annotate_pqpe_points(ax1, algo_resources_cum, metric, P_qpe)

    _savefig("metric_and_err_vs_algo_resources")


# ===========================================================
# (9) QPE success probability vs total cumulative energy
# ===========================================================
def plot_qpe_success_vs_total_energy(
    problem_sizes,
    P_qpe,
    cumulative_energy,
    P_qpe_star=None,
):
    """
    QPE success probability vs total cumulative energy:

    x: cumulative total energy (J), log10 scale with 10^k ticks
    y: P_QPE (from VQE errors via P_QPE = cos^2(Err/2))

    - Threshold line at P_QPE = 0.5
    - P_QPE annotated at each point.
    - If P_qpe_star is provided, also plot an extrapolated point
      at slightly shifted energy for visibility.
    """
    problem_sizes = np.asarray(problem_sizes, dtype=float)
    P_qpe = np.asarray(P_qpe, dtype=float)
    cumulative_energy = np.asarray(cumulative_energy, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        cumulative_energy,
        P_qpe,
        c=problem_sizes,
        cmap="viridis",
        s=80,
        edgecolors="black",
    )

    _set_log10_xaxis(ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(
        r"Cumulative total energy $E_{\mathrm{tot}}^{(\mathrm{cum})}$ (J)"
    )
    ax.set_ylabel(r"QPE success probability $P_{\mathrm{QPE}}$")
    ax.set_title("QPE success probability vs cumulative total energy")

    # Threshold line at P_QPE = 0.5
    ax.axhline(0.5, color="gray", linestyle="dotted", linewidth=1.5)

    # Annotate P_QPE at each point
    _annotate_pqpe_points(ax, cumulative_energy, P_qpe, P_qpe)

    # Optional extrapolated point
    if P_qpe_star is not None:
        # Place at 10% higher energy than last cumulative value for visibility
        energy_star = cumulative_energy[-1] * 1.1
        ax.scatter(
            [energy_star],
            [P_qpe_star],
            s=120,
            color="red",
            edgecolors="black",
            zorder=4,
            label=r"Extrapolated $P_{\mathrm{QPE}}(n^*)$",
        )
        _annotate_pqpe_points(ax, [energy_star], [P_qpe_star], [P_qpe_star])

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Problem size $n$")

    _savefig("qpe_success_vs_energy")
