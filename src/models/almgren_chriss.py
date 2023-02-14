"""
almgren_chriss.py
-----------------
Optimal Execution Engine based on Almgren-Chriss (2000).
Calculates the efficient frontier for liquidating large positions.

Cost Function:
    E[Cost] = 0.5 * gamma * X^2 + eta * sum(v_k^2 * tau)
    Var[Cost] = 0.5 * sigma^2 * X * T (simplified)

Objective:
    min E[Cost] + lambda * Var[Cost]

Author: F.Williams
Date: 2023-02-12
"""

import numpy as np
import pandas as pd

class OptimalExecution:
    def __init__(self, sigma, eta, gamma, risk_aversion):
        """
        :param sigma: Daily volatility of the asset
        :param eta: Temporary price impact coefficient (slippage)
        :param gamma: Permanent price impact coefficient (information leakage)
        :param risk_aversion: Lambda parameter (Trade urgency)
        """
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lamb = risk_aversion

    def compute_trajectory(self, X_total, T, N):
        """
        Derives the optimal trading schedule.
        
        :param X_total: Total quantity to sell
        :param T: Time horizon (e.g., 1 day)
        :param N: Number of trading intervals
        """
        tau = T / N
        
        # Calculate Urgency Parameter (Kappa)
        # kappa^2 = (lambda * sigma^2) / eta
        if self.eta < 1e-9:
             raise ValueError("Eta (temporary impact) cannot be zero.")

        term = (self.lamb * (self.sigma**2)) / self.eta
        kappa = np.sqrt(term)
        
        # Limit case: Risk Neutral (lambda -> 0) => TWAP (Linear)
        if kappa < 1e-6:
            return np.linspace(X_total, 0, N+1)

        # Hyperbolic solution for risk-averse trader
        # x(t) = sinh(kappa * (T - t)) / sinh(kappa * T) * X
        t_grid = np.linspace(0, T, N+1)
        schedule = (np.sinh(kappa * (T - t_grid)) / np.sinh(kappa * T)) * X_total
        
        return schedule

    def estimate_transaction_cost(self, trajectory, dt=1.0):
        """
        Post-trade analysis of expected implementation shortfall.
        """
        trades = -np.diff(trajectory) # Amount sold per interval
        
        # Permanent Impact Cost: 0.5 * gamma * X^2
        perm_cost = 0.5 * self.gamma * (trajectory[0]**2)
        
        # Temporary Impact Cost: eta * sum(v^2) * dt
        # v = trade / dt
        velocities = trades / dt
        temp_cost = self.eta * np.sum(velocities**2) * dt
        
        return perm_cost + temp_cost

if __name__ == "__main__":
    # Simulation
    print("[*] Optimal Execution Strategy Simulation")
    
    # Market Params
    SIGMA = 0.5   # High Volatility
    ETA = 0.05    # Liquid Market
    GAMMA = 0.01  # Low Info Leakage
    
    total_shares = 1000000
    intervals = 50
    
    # 1. Compare Strategies
    strategies = {
        "Risk Neutral (TWAP)": 1e-9,
        "Normal Urgency": 1e-5,
        "Panic Liquidation": 1e-3
    }
    
    for name, l_val in strategies.items():
        engine = OptimalExecution(SIGMA, ETA, GAMMA, risk_aversion=l_val)
        traj = engine.compute_trajectory(total_shares, 1.0, intervals)
        cost = engine.estimate_transaction_cost(traj)
        
        print(f"  Strategy: {name:20} | Lambda: {l_val:.1e} | Exp. Cost: ${cost:,.2f}")