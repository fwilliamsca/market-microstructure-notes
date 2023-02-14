"""
hawkes.py
---------
Univariate Hawkes Process Calibration via Maximum Likelihood Estimation (MLE).
Used to model self-exciting order arrival processes in High-Frequency Trading.

Mathematical Formulation:
    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

Where:
    - mu: Background intensity (exogenous events)
    - alpha: Excitation parameter (endogenous events, branching ratio)
    - beta: Decay rate of influence

Author: F.Williams
Date: 2023-02-10
License: MIT
"""

import numpy as np
from scipy.optimize import minimize
import sys
import time

class HawkesCalibration:
    def __init__(self, decay_beta=1.0):
        """
        Initialize the Hawkes Process calibrator.
        :param decay_beta: Fixed decay parameter to stabilize MLE optimization.
        """
        self.beta = decay_beta
        self.params = None # [mu, alpha]
        self.n_events = 0

    def _recursive_log_likelihood(self, params, arrival_times):
        """
        Calculates the log-likelihood using the O(N) recursive formulation.
        Standard implementation is O(N^2), which is too slow for HFT data.
        """
        mu, alpha = params
        
        # Constraint enforcement: Intensities must be positive
        if mu <= 1e-6 or alpha < 0:
            return 1e9  # Large penalty

        t_max = arrival_times[-1]
        n = len(arrival_times)
        
        # 1. Integral term: - integral_0^T lambda(t) dt
        # Analytical solution for exponential kernel
        term1 = mu * t_max
        term2 = (alpha / self.beta) * np.sum(1.0 - np.exp(-self.beta * (t_max - arrival_times)))
        integral_term = term1 + term2

        # 2. Sum of log intensities: sum(log(lambda(t_i)))
        log_lambda_sum = 0.0
        
        # Recursive variable R(k)
        # R(k) = exp(-beta * (t_k - t_{k-1})) * (1 + R(k-1))
        R = 0.0 
        
        # First event
        log_lambda_sum += np.log(mu)

        for i in range(1, n):
            dt = arrival_times[i] - arrival_times[i-1]
            R = np.exp(-self.beta * dt) * (1.0 + R)
            
            intensity = mu + alpha * R
            
            # Sanity check for numerical stability
            if intensity <= 1e-9:
                return 1e9
            
            log_lambda_sum += np.log(intensity)

        # Log-Likelihood = Sum - Integral
        # Return negative because scipy.minimize finds the minimum
        return -(log_lambda_sum - integral_term)

    def fit(self, arrival_times):
        """
        Calibrates mu and alpha parameters given a sequence of timestamps.
        """
        self.n_events = len(arrival_times)
        print(f"[*] Starting MLE calibration for {self.n_events} events...")
        print(f"    Fixed Beta (Decay): {self.beta}")

        start_time = time.time()

        # Initial guesses: [mu, alpha]
        x0 = [0.5, 0.2]
        
        # Bounds: mu > 0, 0 <= alpha < beta (stationarity condition)
        # If alpha > beta, the process explodes.
        bounds = ((1e-5, None), (1e-5, self.beta - 0.01))

        res = minimize(
            self._recursive_log_likelihood, 
            x0, 
            args=(arrival_times,), 
            method='L-BFGS-B', 
            bounds=bounds,
            tol=1e-5
        )

        elapsed = time.time() - start_time

        if res.success:
            self.params = res.x
            print(f"    [CONVERGED] in {elapsed:.4f}s")
            print(f"    mu (Baseline): {res.x[0]:.4f}")
            print(f"    alpha (Excitation): {res.x[1]:.4f}")
            print(f"    Branching Ratio (n): {res.x[1]/self.beta:.4f}")
            return True
        else:
            print(f"    [FAILED] {res.message}")
            return False

if __name__ == "__main__":
    # Unit Test / Simulation
    np.random.seed(42)
    print("Generating synthetic self-exciting data...")
    
    # Simple cluster generation
    N = 3000
    T = 1000
    base = np.random.uniform(0, T, N//2)
    clusters = base + np.random.exponential(0.05, N//2)
    events = np.sort(np.concatenate([base, clusters]))
    events = events[events < T] # Truncate

    model = HawkesCalibration(decay_beta=5.0)
    model.fit(events)