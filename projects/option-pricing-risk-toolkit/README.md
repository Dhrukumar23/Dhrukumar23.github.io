# Option Pricing & Risk Toolkit

Python implementation of core valuation and risk methods. **All market data is simulated.**

| Module | Result (example: S=100, K=105, T=1y, r=3%, vol=25%) |
|---|---|
| Black-Scholes fair value | Call 9.1218, Put 11.0186 (put-call parity verified to 1e-14) |
| Greeks | Delta 0.52, Gamma 0.016, Vega 0.40 per 1%, Theta −0.017/day, Rho 0.43 per 1% |
| Monte Carlo (GBM, antithetic variates) | 9.1256 ± 0.032 with 1 million paths |
| Implied volatility (Newton-Raphson) | Market price 9.50 → 25.95% |
| Portfolio risk (EUR 1m, 3 correlated assets) | 1-day 99% VaR EUR 21,351 (simulation) vs. EUR 21,564 (parametric); Expected Shortfall EUR 24,482 |
| PRIIPs-style scenarios (EUR 10,000, 5 years) | Stress 4,592 · Unfavourable 7,058 · Moderate 11,820 · Favourable 19,787 |

## Run
```bash
pip install -r requirements.txt
python option_risk_toolkit.py
```
Full output: [`output.txt`](output.txt)

*Author: Dhrukumar Valand*
