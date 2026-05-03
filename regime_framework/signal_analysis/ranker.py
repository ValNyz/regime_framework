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

    # Outer merge: every feature appears once. Binary features have lift_*; continuous have mi only.
    merged = lift.merge(mi, on="signal", how="outer")
    for c in ("n_triggers", "max_abs_lift", "lift_bull", "lift_bear",
              "p_bull_cond", "p_bear_cond", "deviation", "mi"):
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    # Mark which features have a meaningful lift entry
    has_lift = merged["n_triggers"] > 0 if "n_triggers" in merged.columns else pd.Series(False, index=merged.index)
    merged["feature_kind"] = ["binary" if h else "continuous" for h in has_lift]

    # Combined score:
    # - For binary signals: z(MI) + z(max_abs_lift). Lift is a real, additional signal.
    # - For continuous signals: z(MI) only. Lift is structurally zero — don't penalize.
    z_mi = (merged["mi"] - merged["mi"].mean()) / (merged["mi"].std() + 1e-12)
    if has_lift.any():
        sub = merged.loc[has_lift, "max_abs_lift"]
        mu, sd = float(sub.mean()), float(sub.std() + 1e-12)
        z_lift = (merged["max_abs_lift"] - mu) / sd
        z_lift = z_lift.where(has_lift, 0.0)
    else:
        z_lift = pd.Series(0.0, index=merged.index)

    merged["combined_score"] = z_mi + z_lift
    out = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return out
