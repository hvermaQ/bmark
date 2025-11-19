# energy_models.py
"""
Energy and success-probability models for VQE/QPE benchmark.

Includes:
- hardware_resource: converts algorithmic resources to quantum hardware energy
- classical_energy_consumption: FLOP & energy model per optimizer iteration
- qpe_success_from_vqe_error: map VQE energy error to QPE success probability
"""

import numpy as np
from scipy.constants import hbar
import math


# ===========================================================
#  Hardware resource model
# ===========================================================
def hardware_resource(algo_resources):
    """
    Insert algorithmic resources to hardware resources conversion.
    Assume same energy consumption for single and two qubit gates.
    """
    # Frequency [Hz]  (GHz ranges)
    h_w0 = 6e9

    # Gamma [kHz]
    gam = 1

    # Single-qubit gate duration [s] (25 ns)
    t_1qb = 25e-9

    # Attenuation in dB (changed to 20 dB as per discussion)
    A_db = 20
    A = 10 ** (A_db / 10)  # absolute attenuation

    # Qubit temperature [K] (changed to 15 mK as per discussion)
    T_qb = 15e-3

    # External temperature [K]
    T_ext = 300.0

    # Energy per 1qb gate at qubit temperature
    E_1qb = hbar * h_w0 * (np.pi * np.pi) / (4 * gam * t_1qb)

    # Total heat evacuated
    E_cool = (T_ext - T_qb) * A * E_1qb * algo_resources / T_qb
    return float(E_cool)


# ======================================================================
#  Classical energy consumption model (from VQE–QPU fits) for HVA only
# ======================================================================
def classical_energy_consumption(depth, efficiency_flops_per_watt, ansatz):
    """
    Compute classical optimization energy consumption for a given circuit depth
    **per optimizer iteration**, with ansatz-specific FLOP models.

    Parameters
    ----------
    depth : int or float
        Ansatz depth (number of layers)
    efficiency_flops_per_watt : float
        Hardware efficiency in FLOP/(s·W), i.e. FLOPs per joule.
    ansatz : str
        Ansatz type, either "HVA" or "RYA"

    Returns
    -------
    energy_J_per_iter : float
        Classical energy consumption in joules for one optimizer iteration.
    flops_opt_per_iter : float
        FLOP count attributable to optimization overhead for one iteration.
    """
    if ansatz.upper() == "HVA":
        # --- Baseline QPU-only fit (linear) ---
        flops_qpu = 18426.225 * depth + 19210.299
        # --- Full noisy VQE fit (exponential form) ---
        flops_vqe = 96621.711 * np.exp(0.262 * depth) - 102781.482
    elif ansatz.upper() == "RYA":
        flops_qpu = 200096.333 * depth + 13997.734
        flops_vqe = 345574.410 * np.exp(0.279 * depth) - 497233.218
    else:
        raise ValueError("Unknown ansatz type. Use 'HVA' or 'RYA'.")

    flops_opt_per_iter = max(flops_vqe - flops_qpu, 0.0)
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
    """
    Map a VQE energy error to a lower bound on QPE success probability.

    Returns dict with:
        - F_min        : variational fidelity lower bound
        - P_phase      : ideal phase success probability
        - M_ctrlU      : number of controlled-U applications
        - noise_gate_fac, noise_deph_fac
        - P_success    : combined success probability (clipped to [0, 1])
    """
    if gap_delta <= 0:
        raise ValueError("gap_delta must be positive")

    eps = float(epsilon)
    gap = float(gap_delta)

    # Variational bound → fidelity lower bound
    F_min = max(0.0, 1.0 - eps / gap)

    # Ideal phase success for t precision with s padding
    if s_bits >= 1:
        P_phase = 1.0 - 2.0 ** (-(s_bits - 1))
    else:
        P_phase = 0.0

    # Controlled-U count
    t_bits = int(t_bits)
    s_bits = int(max(s_bits, 0))
    M_ctrlU = max(0, (1 << (t_bits + s_bits)) - 1)

    # Noise factors
    noise_gate_fac = math.exp(-float(p_eff) * M_ctrlU) if p_eff > 0 else 1.0
    noise_deph_fac = (
        math.exp(-float(T_total) / float(T2))
        if (T2 not in (0, float("inf")) and T_total > 0)
        else 1.0
    )

    P_success = F_min * P_phase * noise_gate_fac * noise_deph_fac
    P_success = max(0.0, min(1.0, P_success))

    return {
        "F_min": F_min,
        "P_phase": P_phase,
        "M_ctrlU": M_ctrlU,
        "noise_gate_fac": noise_gate_fac,
        "noise_deph_fac": noise_deph_fac,
        "P_success": P_success,
    }
