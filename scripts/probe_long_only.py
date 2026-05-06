"""Probe — verifie que long_only=True met le strategy_log_ret a STRICTEMENT 0
sur les bars bear, sur les vraies donnees BTC du user. Genere aussi un PNG
qui superpose les equity curves des differents scenarios pour inspection.

Usage:
    python scripts/probe_long_only.py [path/to/closes.feather]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regime_framework.evaluation.metrics import (
    synth_equity_curve,
    _strategy_log_returns,
)


def dd(eq: np.ndarray) -> float:
    return float((eq / np.maximum.accumulate(eq)).min() - 1.0)


def main(feather: Path, out_path: Path) -> None:
    df = pd.read_feather(feather)
    closes = df["close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["date"]) if "date" in df.columns else pd.RangeIndex(len(closes))
    print(f"Loaded {len(closes)} bars from {feather}")
    print(f"  range {closes.min():.0f}..{closes.max():.0f}  last={closes[-1]:.0f}")

    log_ret = np.diff(np.log(closes))
    thr = np.percentile(log_ret, 30)
    n = len(closes)

    # Regime-block labels — 500-bar contiguous blocks alternating bull/bear so
    # the flat segments are visually obvious (vs. bar-by-bar labels where each
    # flat is 1 bar = invisible at chart resolution).
    block = 500
    block_id = np.arange(n) // block
    regime_alt   = np.where(block_id % 2 == 0, "bull", "bear")  # 50% bull / 50% bear blocks
    regime_first = np.where(np.arange(n) < n // 2, "bull", "bear")  # bull premiere moitie, bear deuxieme

    scenarios = {
        "ALL BEAR (spot long_only)":         ("bear", None),
        "ALL BULL (= BH)":                   ("bull", None),
        "REGIME-ALT (500-bar blocks)":       ("regime_alt", regime_alt),
        "REGIME-FIRST-HALF-BULL":            ("regime_first", regime_first),
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, closes, color="black", linewidth=0.8, alpha=0.4, label="close (BTC)")

    print("\n--- Long-only spot results ---")
    for label, (kind, payload) in scenarios.items():
        if kind == "bear":
            lbl = np.array(["bear"] * len(closes))
        elif kind == "bull":
            lbl = np.array(["bull"] * len(closes))
        elif kind in ("regime_alt", "regime_first"):
            lbl = payload
        else:
            continue
        eq, gain = synth_equity_curve(closes, lbl, long_only=True)
        # Rebase to closes[0] for visual comparability with the price line.
        eq_r = eq * (closes[0] / eq[0]) if eq[0] else eq
        d = dd(eq)
        bull_frac = float((lbl == "bull").mean())
        ax.plot(dates, eq_r, linewidth=1.5, alpha=0.9,
                label=f"{label}  gain={gain:+.1%}  DD={d:+.1%}  bull%={bull_frac:.0%}")
        print(f"  {label:35s}  gain={gain:+.2%}  DD={d:+.2%}  bull%={bull_frac:.0%}")

    # Hard invariant: bear-only en spot => strategy_log_ret strictement nul partout
    ret = _strategy_log_returns(closes, np.array(["bear"] * len(closes)), long_only=True)
    inv = float(np.abs(ret).max())
    print(f"\nINVARIANT  bear-only spot  max(|ret|) = {inv:.2e}   (must be 0)")

    ax.set_yscale("log")
    ax.set_ylabel("price / equity (log, rebased a closes[0])")
    ax.set_title(
        f"Probe long_only spot — {len(closes)} bars  "
        f"(BH gain={(closes[-1]/closes[0]-1):+.1%}, BH DD={dd(closes):+.1%})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    default_feather = Path(
        "/home/nyzam/Documents/Valentin/hyperliquid_data/user_data/data/hyperliquid/BTC_USDC-1h.feather"
    )
    feather = Path(sys.argv[1]) if len(sys.argv) > 1 else default_feather
    out_path = Path("plots/probe_long_only.png")
    main(feather, out_path)
