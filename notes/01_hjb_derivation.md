# Stochastic Control in Optimal Execution

**Date:** 2023-02-12
**Reference:** Cartea, Jaimungal, Penalva (2015)

## 1. Problem Setup

We assume the mid-price $S_t$ follows an arithmetic Brownian motion:

$$dS_t = \sigma dW_t$$

The trader controls the execution speed $\nu_t$. The inventory $q_t$ evolves as:

$$dq_t = - \nu_t dt$$

The cash process $X_t$ evolves as:

$$dX_t = (S_t - k \nu_t) \nu_t dt$$

where $k$ is the temporary impact parameter.

## 2. The Value Function

We define the value function $H(t, S, q)$ maximizing the terminal wealth minus a running inventory penalty (risk):

$$H(t, S, q) = \sup_{\nu} \mathbb{E}_{t, S, q} \left[ X_T + q_T(S_T - \alpha q_T) - \phi \int_t^T q_u^2 du \right]$$

## 3. The HJB Equation

Using the standard dynamic programming principle, we derive the HJB equation:

$$(\partial_t + \frac{1}{2}\sigma^2 \partial_{SS})H + \sup_{\nu} \{ (S - k\nu)\nu \partial_x H - \nu \partial_q H - \phi q^2 \} = 0$$

Using the ansatz $H(t, S, q) = x + qS - h(t, q)$, we reduce this to a system of ODEs.

### 4. Optimal Control

The optimal trading speed $\nu^*$ is given by:

$$\nu^*_t = \sqrt{\frac{\phi}{k}} q_t$$

This implies that execution speed is proportional to the remaining inventory, leading to an **exponential decay** of inventory over time, unlike the linear decay in Almgren-Chriss (risk-neutral case).