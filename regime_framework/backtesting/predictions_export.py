"""Persist a predictor's stitched OOS predictions to feather + manifest.

The stitching logic mirrors `evaluation/runner.py:1240-1253` — we concatenate
each fold's predictions by date, dedupe overlapping bars (rolling CV with
step < test_window can produce overlap), and emit a 4-column feather:
[date, label, close, confidence]. Labels are kept as strings ("bull"/"bear"/"")
— the freqtrade strategy does string comparison on them at runtime. The
`confidence` column is the per-bar max class probability (1.0 when the
predictor lacks predict_proba); the strategy consumes it via the
`cfg.backtest.proba_threshold` filter to discard weak signals.

A sidecar JSON manifest records cache invalidation hints + framework metrics
so the side-by-side report can compare without re-running.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _cfg_hash(cfg) -> str:
    """Stable short hash of the config fields that affect predictions.

    Includes target/timeframe/market_type/cv split + label method + features
    + predictors families. Excludes plot/output/seed (don't change predictions).
    """
    payload = {
        "target": cfg.target,
        "venue": cfg.venue,
        "quote": cfg.quote,
        "settle": cfg.settle,
        "timeframe": cfg.timeframe,
        "market_type": cfg.market_type,
        "label_method": cfg.label.method,
        "L_range": list(cfg.label.L_range or []),
        "horizon": cfg.label.horizon,
        "split_mode": cfg.split.cv_mode,
        "train_window": cfg.split.train_window_bars,
        "test_window": cfg.split.test_window_bars,
        "retrain_every": cfg.split.retrain_every,
        "families": sorted(cfg.predictors.families or []),
        "n_bags_classical": cfg.predictors.n_bags_classical,
        "n_bags_rl": cfg.predictors.n_bags_rl,
        "n_bags_per_predictor": dict(sorted((cfg.predictors.n_bags_per_predictor or {}).items())),
        "lifetime_per_predictor": dict(sorted((cfg.predictors.lifetime_per_predictor or {}).items())),
        "use_external": cfg.features.use_external,
        "use_funding": cfg.features.use_funding,
        "use_trading_signals": cfg.features.use_trading_signals,
        "cross": [(s.target, s.market_type, s.quote) for s in (cfg.cross or [])],
        "seed": cfg.seed,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def dump_stitched_predictions(
    all_fold_preds: list[dict],
    df: pd.DataFrame,
    predictor_name: str,
    out_path: Path,
    manifest_path: Path,
    stitched_metrics: dict,
    cfg,
) -> tuple[int, str, str]:
    """Concatenate `predictor_name`'s per-fold predictions and write to disk.

    Returns (n_unique_bars, oos_first_iso, oos_last_iso).

    Raises ValueError when no fold contains the requested predictor.
    """
    preds_list, dates_list, closes_list, confs_list = [], [], [], []
    for fp in all_fold_preds:
        if predictor_name not in fp["predictions"]:
            continue
        pred_arr = np.asarray(fp["predictions"][predictor_name])
        preds_list.append(pred_arr)
        dates_list.append(np.asarray(fp["d_te"].values))
        if "close" in df.columns:
            cl = df.loc[fp["test_index"], "close"].to_numpy(dtype=np.float64)
            closes_list.append(cl)
        else:
            closes_list.append(np.full(len(fp["d_te"]), np.nan, dtype=np.float64))
        # Confidence is optional (older runs / predictors without predict_proba):
        # default to 1.0 so the downstream threshold filter is a no-op.
        conf = (fp.get("confidences") or {}).get(predictor_name)
        if conf is None:
            confs_list.append(np.ones(len(pred_arr), dtype=np.float64))
        else:
            confs_list.append(np.asarray(conf, dtype=np.float64))

    if not preds_list:
        raise ValueError(
            f"No fold contains predictions for {predictor_name!r}. "
            f"Available: {sorted({k for fp in all_fold_preds for k in fp['predictions']})}"
        )

    preds_concat = np.concatenate(preds_list)
    dates_concat = pd.to_datetime(np.concatenate(dates_list), utc=True)
    closes_concat = np.concatenate(closes_list)
    confs_concat = np.concatenate(confs_list)

    out = pd.DataFrame({
        "date": dates_concat,
        "label": preds_concat.astype(str),
        "close": closes_concat,
        "confidence": confs_concat,
    })
    # Rolling CV with step < test_window produces overlapping OOS windows.
    # Keep the LAST occurrence (= freshest model's prediction at that bar)
    # which matches what stitched_metrics in the framework already used.
    n_concat = len(out)
    out = out.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    n_unique = len(out)
    if n_unique < n_concat * 0.95:
        # Heads up: a lot of duplicates suggests the CV step is much smaller
        # than the test window. Still works, but worth flagging.
        print(
            f"  WARN: {n_concat - n_unique} duplicate dates dropped "
            f"({100 * (1 - n_unique / n_concat):.0f}% of stitched bars). "
            f"Rolling CV step likely << test window."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(out_path)

    oos_first = out["date"].iloc[0].isoformat()
    oos_last = out["date"].iloc[-1].isoformat()

    manifest = {
        "predictor": predictor_name,
        "oos_first": oos_first,
        "oos_last": oos_last,
        "n_bars": n_unique,
        "n_bull": int((out["label"] == "bull").sum()),
        "n_bear": int((out["label"] == "bear").sum()),
        "n_other": int(((out["label"] != "bull") & (out["label"] != "bear")).sum()),
        "stitched_metrics": stitched_metrics,
        "cfg_hash": _cfg_hash(cfg),
        "schema": "[date, label, close, confidence]",
        # Threshold value used to compute `stitched_metrics`. Used by the
        # backtest CLI to flag mismatches between manifest metrics and the
        # CLI-overridden threshold — if they differ, the side-by-side
        # framework column is stale relative to the freqtrade run.
        "proba_threshold_used": float(getattr(cfg.backtest, "proba_threshold", 0.0) or 0.0),
        "confidence_stats": {
            "min": float(out["confidence"].min()),
            "max": float(out["confidence"].max()),
            "mean": float(out["confidence"].mean()),
            # Quick scan of how aggressive a filter would be: fraction of
            # bars that would survive each threshold. Saved so we can pick
            # `proba_threshold` without rerunning the framework.
            "frac_above_0.5": float((out["confidence"] >= 0.5).mean()),
            "frac_above_0.55": float((out["confidence"] >= 0.55).mean()),
            "frac_above_0.6": float((out["confidence"] >= 0.6).mean()),
            "frac_above_0.65": float((out["confidence"] >= 0.65).mean()),
            "frac_above_0.7": float((out["confidence"] >= 0.7).mean()),
        },
    }
    manifest_path = Path(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return n_unique, oos_first, oos_last
