import numpy as np
import matplotlib.pyplot as plt

# --- User: set your .npz filenames here ---
npz_file_rya = "benchmark_results/RYA_best_results.npz"  # Replace with actual filename
npz_file_hva = "benchmark_results/HVA_best_results.npz"  # Replace with actual filename

# --- Load data ---
def load_npz(path):
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}

data_rya = load_npz(npz_file_rya)
data_hva = load_npz(npz_file_hva)

# --- Extract relevant data ---
def get_slide22_data(data):
    # Ensure all arrays are float64 and sorted by n
    n = np.array(data["nqbits"], dtype=float)
    E = np.array(data["sim_results"], dtype=float)
    Eerr = np.abs(np.array(data["sim_std"], dtype=float))  # Ensure non-negative error bars
    sort_idx = np.argsort(n)
    n = n[sort_idx]
    E = E[sort_idx]
    Eerr = Eerr[sort_idx]
    return dict(
        n=n,
        E=E,
        Eerr=Eerr,
        n_star=int(n[-1]),
        E_exact_star=float(data["exact_per_size"][-1]),
    )

rya = get_slide22_data(data_rya)
hva = get_slide22_data(data_hva)

# --- Linear regression (weighted) ---

# Use scipy curve_fit for weighted linear regression
from scipy.optimize import curve_fit

def lin_model(x, m, b):
    return m * x + b

# RYA fit
popt_rya, _ = curve_fit(lin_model, rya['n'], rya['E'], sigma=rya['Eerr'], absolute_sigma=True)
m_rya, b_rya = popt_rya

# HVA fit
popt_hva, _ = curve_fit(lin_model, hva['n'], hva['E'], sigma=hva['Eerr'], absolute_sigma=True)
m_hva, b_hva = popt_hva

# --- Extrapolate to n* ---
E_star_rya = m_rya * rya['n_star'] + b_rya
E_star_hva = m_hva * hva['n_star'] + b_hva

# --- Plot ---

# Set global font size
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 15})
fig, ax = plt.subplots(figsize=(10, 7))

# Plot RYA data and fit
ax.errorbar(rya['n'], rya['E'], yerr=rya['Eerr'], fmt='o', color='tab:blue', label='RYA data', capsize=4)
xfit_rya = np.linspace(min(rya['n']), max(rya['n']), 200)
ax.plot(xfit_rya, m_rya*xfit_rya + b_rya, 'b--', label='RYA fit')
ax.scatter([rya['n_star']], [E_star_rya], color='blue', s=100, edgecolors='black', zorder=5, label=r"RYA extrapolated $n^*$")

# Plot HVA data and fit
ax.errorbar(hva['n'], hva['E'], yerr=hva['Eerr'], fmt='s', color='tab:orange', label='HVA data', capsize=4)
xfit_hva = np.linspace(min(hva['n']), max(hva['n']), 200)
ax.plot(xfit_hva, m_hva*xfit_hva + b_hva, 'orange', linestyle='--', label='HVA fit')
ax.scatter([hva['n_star']], [E_star_hva], color='orange', s=100, edgecolors='black', zorder=5, label=r"HVA extrapolated $n^*$")

# Only one exact line at n* (choose e.g. RYA)
ax.axhline(rya['E_exact_star'], color='black', linestyle=':', label=r"Exact $E_0(n^*)$")

# Axis formatting
all_n = np.unique(np.concatenate([rya['n'], hva['n']]))
ax.set_xlabel(r"Problem size $n$")
ax.set_ylabel(r"Energy $E_{\mathrm{VQE}}(n)$")
ax.set_xticks(all_n)
ax.set_xticklabels([str(int(x)) for x in all_n])
ax.set_title("Linear extrapolation comparison: RYA vs HVA (fresh fit)")
ax.legend(loc='best')
#plt.tight_layout()
plt.show()