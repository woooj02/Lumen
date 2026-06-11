"""Generate a realistic sample dataset (telco-style customer churn) so the app
demos well out-of-the-box and screenshots tell a story. Intentionally seeds
some missing values, duplicates, and outliers to exercise the cleaning tools.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Credit card", "Bank transfer", "Electronic check", "Mailed check"]
REGIONS = ["North", "South", "East", "West"]
INTERNET = ["DSL", "Fiber optic", "No"]


def make_customers(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 72, n)
    monthly = np.round(rng.normal(70, 30, n).clip(18, 200), 2)
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n, p=[0.35, 0.45, 0.20])
    tickets = rng.poisson(1.2, n)
    satisfaction = rng.integers(1, 6, n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])

    # Total charges roughly = tenure * monthly with noise.
    total = np.round(tenure * monthly * rng.uniform(0.9, 1.1, n), 2)

    # Churn probability driven by real signal so models find structure.
    logit = (
        -2.4
        + 1.6 * (contract == "Month-to-month")
        + 0.9 * (internet == "Fiber optic")
        - 0.035 * tenure
        + 0.012 * monthly
        + 0.45 * tickets
        - 0.35 * satisfaction
        + 0.4 * senior
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "age": rng.integers(18, 80, n),
        "gender": rng.choice(["Male", "Female"], n),
        "senior_citizen": senior,
        "region": rng.choice(REGIONS, n),
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": rng.choice(PAYMENTS, n),
        "monthly_charges": monthly,
        "total_charges": total,
        "support_tickets": tickets,
        "satisfaction_score": satisfaction,
        "churn": np.where(churn == 1, "Yes", "No"),
    })

    # Inject realistic mess: missing values, a few outliers, duplicate rows.
    miss_idx = rng.choice(n, size=int(n * 0.04), replace=False)
    df.loc[miss_idx, "total_charges"] = np.nan
    miss_idx2 = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[miss_idx2, "satisfaction_score"] = np.nan
    out_idx = rng.choice(n, size=8, replace=False)
    df.loc[out_idx, "monthly_charges"] = rng.uniform(500, 900, len(out_idx)).round(2)
    df = pd.concat([df, df.sample(12, random_state=seed)], ignore_index=True)

    return df


if __name__ == "__main__":
    make_customers().to_csv("data/sample_customers.csv", index=False)
    print("Wrote data/sample_customers.csv")
