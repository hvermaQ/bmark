# extrapolation.py
"""
Weighted linear regression and extrapolation helper.

Implements the same linear_extrapolation routine as in the original script,
with proper error propagation.
"""

import numpy as np


# ===========================================================
#  Weighted linear regression with error propagation
# ===========================================================
def linear_extrapolation(problem_sizes, values, errors, target_size):
    """
    Perform weighted linear regression to estimate the value at the target size,
    incorporating proper error propagation.

    Parameters
    ----------
    problem_sizes : list or array
        List of problem sizes (e.g. n = 4, 5, 6, 7).
    values : list or array
        Corresponding VQE energies for the problem sizes.
    errors : list or array
        Standard deviations associated with the values.
    target_size : int or float
        Target problem size for extrapolation (n*).

    Returns
    -------
    dict with keys:
        extrapolated_value
        extrapolated_error
        regression_coefficients (m, b)
        residual_variance
    """
    from scipy.optimize import curve_fit

    x = np.asarray(problem_sizes, dtype=float)
    y = np.asarray(values, dtype=float)
    sigma = np.maximum(np.asarray(errors, dtype=float), 1e-12)

    def lin_model(x, m, b):
        return m * x + b

    popt, pcov = curve_fit(lin_model, x, y, sigma=sigma, absolute_sigma=True)
    m, b = popt
    # Standard errors for m, b
    m_err, b_err = np.sqrt(np.diag(pcov))

    # Residual variance
    y_hat = lin_model(x, m, b)
    dof = max(len(x) - 2, 1)
    wrss = np.sum(((y - y_hat) / sigma) ** 2)
    s2 = wrss / dof

    # Extrapolate to target size
    x_star = float(target_size)
    y_star = float(lin_model(x_star, m, b))
    # Error propagation for y_star
    v = np.array([x_star, 1.0])
    y_star_std = float(np.sqrt(v @ pcov @ v))

    return {
        'extrapolated_value': y_star,
        'extrapolated_error': y_star_std,
        'regression_coefficients': np.array([m, b]),
        'residual_variance': s2,
    }
