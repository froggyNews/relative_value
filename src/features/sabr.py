import numpy as np
from scipy.optimize import brentq
from .utils import suppress_runtime_warnings



def _hagan_implied_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """Approximate Black implied volatility under the SABR model."""
    with suppress_runtime_warnings():
        if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
            return np.nan
        
        # Check for invalid parameters
        if abs(rho) >= 1 or nu < 0 or not (0 <= beta <= 1):
            return np.nan

        if np.isclose(F, K):
            term1 = alpha / (F ** (1 - beta))
            term2 = 1 + (
                ((1 - beta) ** 2 / 24) * (alpha ** 2 / (F ** (2 - 2 * beta)))
                + (rho * beta * nu * alpha / (4 * F ** (1 - beta)))
                + ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
            ) * T
            return term1 * term2

        FK_beta = (F * K) ** ((1 - beta) / 2)
        logFK = np.log(F / K)
        z = (nu / alpha) * FK_beta * logFK
        
        # Handle the z/x_z term carefully
        if np.isclose(z, 0, atol=1e-7):
            # When z ≈ 0, limit of z/x_z = 1
            term2 = 1.0
        else:
            # Check if the log argument is valid
            sqrt_term = np.sqrt(1 - 2 * rho * z + z * z)
            log_arg = (sqrt_term + z - rho) / (1 - rho)
            
            if log_arg <= 0 or (1 - rho) == 0:
                return np.nan
                
            x_z = np.log(log_arg)
            
            if np.isclose(x_z, 0, atol=1e-12):
                # Avoid division by zero
                term2 = 1.0
            else:
                term2 = z / x_z
        
        term1 = alpha / (FK_beta * (1 + ((1 - beta) ** 2 / 24) * (logFK ** 2) + ((1 - beta) ** 4 / 1920) * (logFK ** 4)))
        term3 = 1 + (
            ((1 - beta) ** 2 / 24) * (alpha ** 2 / (FK_beta ** 2))
            + (rho * beta * nu * alpha / (4 * FK_beta))
            + ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
        ) * T
        
        result = term1 * term2 * term3
        
        # Final sanity check
        if not np.isfinite(result) or result <= 0:
            return np.nan
            
        return result




def _solve_sabr_alpha(sigma: float, F: float, K: float, T: float, beta: float, rho: float, nu: float) -> float:
    """Calibrate alpha for a single observation using Hagan's formula."""
    if np.any(np.isnan([sigma, F, K, T])) or sigma <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan
    
    # Check for invalid parameter ranges
    if abs(rho) >= 1 or nu < 0 or not (0 <= beta <= 1):
        return np.nan

