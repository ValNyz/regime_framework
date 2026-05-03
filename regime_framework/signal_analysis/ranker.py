"""Combined ranker: merges lift + MI + conditional accuracy into a single table."""
from __future__ import annotations

import pandas as pd

from .lift import compute_lift_table
from .mutual_info import compute_mutual_info_table


_LIFT_COLS = [
    "signal", "n_triggers", "p_bull_base", "p_bull_cond", "lift_bull",
    "p_bear_base", "p_bear_cond", "lift_bear", "max_abs_lift", "deviation",
]


def __empty_lift() -> pd.DataFrame:
    return pd.DataFrame(columns=_LIFT_COLS)


def rank_signals(
    X: pd.DataFrame,
    y: pd.Series,
    binary_threshold: float = 0.5,
    min_triggers: int = 30,
) -> pd.DataFrame:
    """Combined ranking by predictive power.

    Returns one row per feature with all metrics merged. Sorted by combined
    score = z-score(max_abs_lift) + z-score(mi).
    """
    try:
        lift = compute_lift_table(X, y, binary_threshold=binary_threshold, min_triggers=min_triggers)
    except Exception as e:
        print(f"  WARN: lift table failed ({e}); empty lift.")
        lift = __empty_lift()
    try:
        mi = compute_mutual_info_table(X, y, discrete_features=False)
    except Exception as e:
        print(f"  WARN: MI table failed ({e}); zeros.")
        mi = pd.DataFrame({"signal": list(X.columns), "mi": [0.0] * X.shape[1]})

    if lift.empty:
        # all features continuous — fall back to MI only
        out = mi.copy()
        out["max_abs_lift"] = float("nan")
        out["lift_bull"] = float("nan")
        out["lift_bear"] = float("nan")
        out["n_triggers"] = 0
        out["combined_score"] = (out["mi"] - out["mi"].mean()) / (out["mi"].std() + 1e-12)
    else:
        merged = lift.merge(mi, on="signal", how="outer")
        # Fill NaN in numeric columns introduced by outer merge
        for c in ("n_triggers", "max_abs_lift", "lift_bull", "lift_bear",
                  "p_bull_cond", "p_bear_cond", "deviation", "mi"):
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
        z_lift = (merged["max_abs_lift"] - merged["max_abs_lift"].mean()) / (merged["max_abs_lift"].std() + 1e-12)
        z_mi = (merged["mi"] - merged["mi"].mean()) / (merged["mi"].std() + 1e-12)
        merged["combined_score"] = z_lift + z_mi
        out = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)

    return out
