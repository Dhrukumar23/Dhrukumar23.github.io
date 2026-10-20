"""
Option Pricing & Risk Toolkit
-----------------------------
Portfolio project by Dhrukumar Valand

1. Fair value of European options with the Black-Scholes model
2. Greeks (Delta, Gamma, Vega, Theta, Rho)
3. Monte Carlo pricing (geometric Brownian motion) and convergence check
4. Implied volatility via Newton-Raphson
5. Portfolio risk: 1-day 99% Value at Risk and Expected Shortfall
   (historical-style simulation and parametric)
6. PRIIPs-style performance scenarios (stress, unfavourable, moderate,
   favourable) for a holding period

All market data is simulated (no real client data).
Run:  python option_risk_toolkit.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

RNG = np.random.default_rng(42)


# ---------- 1. Black-Scholes ----------
def bs_price(S, K, T, r, sigma, kind="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------- 2. Greeks ----------
def bs_greeks(S, K, T, r, sigma, kind="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf = norm.pdf(d1)
    if kind == "call":
        delta = norm.cdf(d1)
        theta = -S * pdf * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -S * pdf * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T)
    return {"Delta": delta, "Gamma": gamma, "Vega (per 1%)": vega / 100,
            "Theta (per day)": theta / 365, "Rho (per 1%)": rho / 100}


# ---------- 3. Monte Carlo ----------
def mc_price(S, K, T, r, sigma, n, kind="call"):
    z = RNG.standard_normal(n // 2)
    z = np.concatenate([z, -z])                      # antithetic variates
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    disc = np.exp(-r * T) * payoff
    return disc.mean(), 1.96 * disc.std(ddof=1) / np.sqrt(n)


# ---------- 4. Implied volatility ----------
def implied_vol(price, S, K, T, r, kind="call", guess=0.2):
    sigma = guess
    for _ in range(100):
        diff = bs_price(S, K, T, r, sigma, kind) - price
        if abs(diff) < 1e-8:
            break
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        sigma -= diff / (S * norm.pdf(d1) * np.sqrt(T))
    return sigma


# ---------- 5. Portfolio VaR / ES ----------
def portfolio_var(weights, mu, vol, corr, value, days=1000, alpha=0.99):
    cov = np.outer(vol, vol) * corr
    rets = RNG.multivariate_normal(mu, cov, size=days)        # simulated daily returns
    pnl = value * rets @ weights
    var_hist = -np.quantile(pnl, 1 - alpha)
    es_hist = -pnl[pnl <= -var_hist].mean()
    port_sd = np.sqrt(weights @ cov @ weights)
    var_param = value * (norm.ppf(alpha) * port_sd - weights @ mu)
    return var_hist, es_hist, var_param


# ---------- 6. PRIIPs-style scenarios ----------
def priips_scenarios(S0, mu, sigma, years, invest=10_000, paths=100_000):
    z = RNG.standard_normal(paths)
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * years + sigma * np.sqrt(years) * z)
    end_values = invest * ST / S0
    pct = {"Stress (1st pct)": 1, "Unfavourable (10th pct)": 10,
           "Moderate (50th pct)": 50, "Favourable (90th pct)": 90}
    rows = []
    for name, p in pct.items():
        v = np.percentile(end_values, p)
        rows.append([name, round(v, 0), f"{((v / invest) ** (1 / years) - 1) * 100:.1f}%"])
    return pd.DataFrame(rows, columns=["Scenario", "Value after holding period (EUR)",
                                       "Average return p.a."])


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 105, 1.0, 0.03, 0.25
    print("=== 1. Black-Scholes fair value (S=100, K=105, T=1y, r=3%, vol=25%) ===")
    c, p = bs_price(S, K, T, r, sigma, "call"), bs_price(S, K, T, r, sigma, "put")
    print(f"Call: {c:.4f}   Put: {p:.4f}")
    print(f"Put-call parity check (should be 0): {c - p - (S - K * np.exp(-r * T)):.2e}")

    print("\n=== 2. Greeks (call) ===")
    for k, v in bs_greeks(S, K, T, r, sigma).items():
        print(f"{k:16s} {v: .4f}")

    print("\n=== 3. Monte Carlo vs. Black-Scholes (call) ===")
    for n in (1_000, 10_000, 100_000, 1_000_000):
        m, ci = mc_price(S, K, T, r, sigma, n)
        print(f"n={n:>9,}  MC={m:.4f} ± {ci:.4f}   error vs BS={m - c:+.4f}")

    print("\n=== 4. Implied volatility ===")
    market_price = 9.50
    print(f"Market call price {market_price} -> implied vol {implied_vol(market_price, S, K, T, r):.2%}")

    print("\n=== 5. Portfolio risk (EUR 1m, 3 assets: equity / bond / commodity) ===")
    w = np.array([0.5, 0.3, 0.2])
    mu = np.array([0.0004, 0.0001, 0.0002])
    vol = np.array([0.015, 0.004, 0.020])
    corr = np.array([[1, -0.2, 0.3], [-0.2, 1, 0.0], [0.3, 0.0, 1]])
    vh, es, vp = portfolio_var(w, mu, vol, corr, 1_000_000)
    print(f"1-day 99% VaR (simulation): EUR {vh:,.0f}")
    print(f"1-day 99% Expected Shortfall: EUR {es:,.0f}")
    print(f"1-day 99% VaR (parametric):  EUR {vp:,.0f}")

    print("\n=== 6. PRIIPs-style scenarios (EUR 10,000, 5 years, mu=5%, vol=18%) ===")
    print(priips_scenarios(100, 0.05, 0.18, 5).to_string(index=False))
