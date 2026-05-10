"""regime_framework CLI — typer + rich.

Commands:
  run [PRESET]        Run the full benchmark on a preset
  signals [PRESET]    Just signal analysis (lift + MI on labels)
  pretrained [PRESET] Just the pretrained models (zero-shot)
  presets             List available presets
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import CONFIG_DIR, RunConfig

app = typer.Typer(
    name="regime",
    help="Modular benchmark of time-series regime classification approaches.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    preset: str = typer.Argument(..., help="Preset name (e.g. btc_binance_1h)"),
    families: list[str] | None = typer.Option(
        None,
        "--family", "-f",
        help="Predictor families: classical, rule_based, deep_nets, transformer, "
             "pretrained, rl. Accepts repeated flags (-f classical -f rl) OR "
             "comma-separated (-f classical,rl). When unset, the preset's "
             "predictors.families wins (CLI overrides preset only if -f passed).",
    ),
    pretrained: list[str] | None = typer.Option(
        None, "--pretrained", "-p",
        help="Override pretrained models list. If unset, uses preset config.",
    ),
    skip_pretrained: bool = typer.Option(
        False, "--skip-pretrained", help="Skip all foundation models (faster).",
    ),
    cv_folds: int = typer.Option(
        0, "--cv-folds", "-k",
        help="Number of CV folds. 0 = single split (default). In 'rolling' mode, "
             "caps at this many folds — the LATEST N consecutive folds are kept "
             "(window + step stay fixed; fold_id reflects original chronological "
             "position in the data).",
    ),
    cv_mode: str = typer.Option(
        "walk_forward", "--cv-mode",
        help="CV mode: walk_forward (expanding), leave_one_out, both, or "
             "rolling (fixed train + sliding test window).",
    ),
    min_train_fraction: float = typer.Option(
        0.40, "--min-train-fraction",
        help="Walk-forward only: fold-0 train size (fraction of total).",
    ),
    train_window_bars: int = typer.Option(
        0, "--train-window-bars",
        help="Rolling mode: training window size in bars. Defaults to 4380 "
             "(6mo at 1h timeframe) when unset.",
    ),
    test_window_bars: int = typer.Option(
        0, "--test-window-bars",
        help="Rolling mode: test window size in bars. Defaults to 730 "
             "(1mo at 1h timeframe) when unset.",
    ),
    step_bars: int = typer.Option(
        0, "--step-bars",
        help="Rolling mode: slide step in bars. 0 = same as test window "
             "(non-overlapping consecutive tests).",
    ),
    feature_importance: bool = typer.Option(
        None, "--feature-importance/--no-feature-importance",
        help="Compute + display feature importance for the best classical "
             "predictor (per fold + end of CV). When unset, the preset's "
             "predictors.feature_importance wins. CLI overrides preset only "
             "when the flag is explicitly passed.",
    ),
    plots: bool = typer.Option(
        None, "--plots/--no-plots",
        help="Master plot switch. --no-plots kills ALL plot output (label, "
             "prediction, fold, multi, stitched). Defaults to preset value.",
    ),
    fold_plots: bool = typer.Option(
        None, "--fold-plots/--no-fold-plots",
        help="Per-fold plots only (best-predictor + multi-overlay per CV "
             "fold). --no-fold-plots keeps once-per-run summary plots "
             "(label, stitched OOS) but drops the per-fold ones — useful "
             "when running with 50+ rolling folds.",
    ),
    rank_by: str = typer.Option(
        None, "--rank-by",
        help="Sort the CV summary table and pick the 'best' predictor for "
             "the stitched OOS plot by: 'kappa' (default — classification "
             "skill), 'gain' (compounded synth_gain across folds), or "
             "'vs_bh' (gain_total minus B&H). Use 'gain' or 'vs_bh' when "
             "you care about money rather than per-bar agreement.",
    ),
):
    """Run the full benchmark on a preset config."""
    cfg = RunConfig.from_preset(preset)
    # CLI families override preset only when -f explicitly passed.
    if families:
        flat_families: list[str] = []
        for f in families:
            flat_families.extend([x.strip() for x in str(f).split(",") if x.strip()])
        valid = {"classical", "rule_based", "deep_nets", "transformer", "pretrained", "rl"}
        bad = [f for f in flat_families if f not in valid]
        if bad:
            raise typer.BadParameter(
                f"Unknown family/families: {bad}. Valid: {sorted(valid)}"
            )
        cfg.predictors.families = flat_families
    if skip_pretrained and "pretrained" in cfg.predictors.families:
        cfg.predictors.families.remove("pretrained")
    if pretrained:
        cfg.predictors.pretrained_models = list(pretrained)
    valid_modes = ("walk_forward", "leave_one_out", "both", "rolling")
    if cv_mode == "rolling":
        # Rolling mode is always CV; cv_folds derives from window sizes.
        cfg.split.cv_mode = "rolling"
        cfg.split.cv_folds = max(1, cv_folds)  # >0 just to enter the CV path
        cfg.split.train_window_bars = train_window_bars or cfg.split.train_window_bars or 4380
        cfg.split.test_window_bars = test_window_bars or cfg.split.test_window_bars or 730
        cfg.split.step_bars = step_bars or cfg.split.step_bars
    elif cv_folds > 0:
        if cv_mode not in valid_modes:
            raise typer.BadParameter(
                f"--cv-mode must be one of {valid_modes} (got {cv_mode!r})"
            )
        cfg.split.cv_folds = cv_folds
        cfg.split.cv_mode = cv_mode
        cfg.split.min_train_fraction = float(min_train_fraction)

    # CLI flags override preset only when explicitly passed (None = unset).
    if feature_importance is not None:
        cfg.predictors.feature_importance = bool(feature_importance)
    if plots is not None:
        cfg.plots.enabled = bool(plots)
    if fold_plots is not None:
        cfg.plots.per_fold = bool(fold_plots)
    if rank_by is not None:
        if rank_by not in ("kappa", "dir_kappa", "gain", "vs_bh"):
            raise typer.BadParameter(
                f"--rank-by must be one of: kappa, dir_kappa, gain, vs_bh "
                f"(got {rank_by!r})"
            )
        cfg.predictors.rank_by = rank_by

    from .evaluation.runner import BenchmarkRunner
    runner = BenchmarkRunner(cfg)
    runner.run()


@app.command()
def signals(preset: str = typer.Argument(..., help="Preset name")):
    """Run only the signal analysis (lift + MI)."""
    cfg = RunConfig.from_preset(preset)
    # Force signal analysis only — disable all predictors
    cfg.predictors.families = []
    from .evaluation.runner import BenchmarkRunner
    runner = BenchmarkRunner(cfg)
    runner.run()


@app.command()
def pretrained(
    preset: str = typer.Argument(..., help="Preset name"),
    zero_shot_only: bool = typer.Option(False, "--zero-shot-only", help="Skip fine-tuned head"),
):
    """Run only pretrained foundation models."""
    cfg = RunConfig.from_preset(preset)
    cfg.predictors.families = ["pretrained"]
    if zero_shot_only:
        cfg.predictors.pretrained_modes = ["zero_shot"]
    from .evaluation.runner import BenchmarkRunner
    runner = BenchmarkRunner(cfg)
    runner.run()


def _load_cfg(preset_or_yaml: str) -> RunConfig:
    """Try preset name first, then fall back to yaml path."""
    p = Path(preset_or_yaml)
    if p.suffix in (".yaml", ".yml") or p.exists():
        return RunConfig.from_yaml(p)
    return RunConfig.from_preset(preset_or_yaml)


def _default_user_data_dir(cfg: RunConfig) -> Path:
    """Default <repo_root>/user_data when not overridden."""
    if cfg.backtest.user_data_dir is not None:
        return Path(cfg.backtest.user_data_dir)
    return Path(__file__).resolve().parents[1] / "user_data"


def _default_datadir(cfg: RunConfig) -> Path:
    """Default datadir = <ohlcv_path>.parent.parent (matches Freqtrade layout)."""
    if cfg.backtest.datadir is not None:
        return Path(cfg.backtest.datadir)
    return Path(cfg.paths.ohlcv).parent.parent


def _safe_class_name(predictor_name: str) -> str:
    """Make a Python-class-safe identifier from a predictor display name."""
    out = "".join(c if c.isalnum() else "_" for c in predictor_name)
    if out and out[0].isdigit():
        out = "_" + out
    return out or "Regime"


@app.command()
def backtest(
    preset_or_yaml: str = typer.Argument(..., help="Preset name OR yaml path"),
    predictor: str = typer.Option(
        None, "--predictor",
        help="Predictor whose stitched OOS series to backtest. Default: top-1 by rank_by.",
    ),
    timerange: str = typer.Option(
        None, "--timerange",
        help="Freqtrade timerange (YYYYMMDD-YYYYMMDD). Default: stitched OOS span.",
    ),
    user_data_dir: Path = typer.Option(
        None, "--user-data-dir",
        help="Isolated freqtrade user_data dir. Default: <repo>/user_data.",
    ),
    datadir: Path = typer.Option(
        None, "--datadir",
        help="Freqtrade --datadir (where OHLCV feathers live). Default: cfg.paths.ohlcv.parent.parent.",
    ),
    mode: str = typer.Option(
        "rolling", "--mode",
        help="CV mode whose stitched predictions to backtest (rolling, walk_forward, ...).",
    ),
    force_rebuild: bool = typer.Option(
        False, "--force-rebuild",
        help="Re-run the framework even if the predictions feather exists.",
    ),
    export_strategy_only: bool = typer.Option(
        False, "--export-strategy-only",
        help="Generate strategy.py + freqtrade config.json but don't invoke freqtrade.",
    ),
    cv_folds: int = typer.Option(
        0, "--cv-folds", "-k",
        help="Cap the number of CV folds. 0 = use cfg.split.cv_folds. "
             "In rolling mode, keeps the LATEST N folds (most recent OOS span).",
    ),
    train_window_bars: int = typer.Option(
        0, "--train-window-bars",
        help="Override cfg.split.train_window_bars (rolling mode).",
    ),
    test_window_bars: int = typer.Option(
        0, "--test-window-bars",
        help="Override cfg.split.test_window_bars (rolling mode).",
    ),
    step_bars: int = typer.Option(
        0, "--step-bars",
        help="Override cfg.split.step_bars (rolling). 0 = same as test_window.",
    ),
    data_timerange: str = typer.Option(
        None, "--data-timerange",
        help="Slice the loaded OHLCV to YYYYMMDD-YYYYMMDD before CV. Use to "
             "test the framework on a specific historical period (e.g. bull).",
    ),
):
    """Backtest a predictor's stitched OOS predictions in freqtrade.

    1. Run the framework (or load cached predictions feather).
    2. Generate a self-contained freqtrade IStrategy + minimal config.json.
    3. Spawn `freqtrade backtesting` and parse the resulting metrics.
    4. Print a side-by-side report (framework stitched vs freqtrade).
    """
    import json as _json

    cfg = _load_cfg(preset_or_yaml)
    # Apply CLI overrides on the CV split BEFORE the framework runs so that
    # the predictions feather is computed against the chosen window/period.
    if cv_folds > 0:
        cfg.split.cv_folds = cv_folds
    if train_window_bars > 0:
        cfg.split.train_window_bars = train_window_bars
    if test_window_bars > 0:
        cfg.split.test_window_bars = test_window_bars
    if step_bars > 0:
        cfg.split.step_bars = step_bars
    if data_timerange:
        # Stash the timerange on cfg.split so BenchmarkRunner can apply it
        # right after load_ohlcv. Format: YYYYMMDD-YYYYMMDD (open right end).
        cfg.split.data_timerange = data_timerange
    udd = (user_data_dir or _default_user_data_dir(cfg)).expanduser().resolve()
    dd = (datadir or _default_datadir(cfg)).expanduser().resolve()
    strategies_dir = udd / "strategies"
    configs_dir = udd / "configs"
    results_dir = udd / "backtest_results"
    for d in (strategies_dir, configs_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Predictions feather + manifest are keyed by (target, tf, predictor_name).
    # When `predictor` is None we use a wildcard `auto` token; the resolved
    # name is written into the manifest at first run and re-read on cache hit.
    pred_token = predictor or "auto"
    pred_path = strategies_dir / f"regime_pred_{cfg.target}_{cfg.timeframe}_{pred_token}.feather"
    manifest_path = pred_path.with_suffix(".json")

    if force_rebuild or not pred_path.exists() or not manifest_path.exists():
        from .evaluation.runner import BenchmarkRunner
        from .backtesting.predictions_export import dump_stitched_predictions

        runner = BenchmarkRunner(cfg)
        result = runner.run()
        all_fold_preds_by_mode = result.get("all_fold_preds_by_mode") or {}
        stitched_metrics_by_mode = result.get("stitched_metrics_by_mode") or {}
        best_predictor_by_mode = result.get("best_predictor_by_mode") or {}

        if mode not in all_fold_preds_by_mode:
            available = sorted(all_fold_preds_by_mode.keys())
            console.print(
                f"[red]No CV results for mode={mode}.[/red] Available: {available}"
            )
            raise typer.Exit(1)

        chosen = predictor or best_predictor_by_mode.get(mode)
        if chosen is None:
            console.print("[red]No predictor available; check that families produced output.[/red]")
            raise typer.Exit(1)

        # If the user used the wildcard token, rewrite the path to include
        # the resolved name so the cache key is precise.
        if predictor is None:
            pred_path = strategies_dir / f"regime_pred_{cfg.target}_{cfg.timeframe}_{chosen}.feather"
            manifest_path = pred_path.with_suffix(".json")

        n_unique, oos_first, oos_last = dump_stitched_predictions(
            all_fold_preds=all_fold_preds_by_mode[mode],
            df=runner.df,
            predictor_name=chosen,
            out_path=pred_path,
            manifest_path=manifest_path,
            stitched_metrics=stitched_metrics_by_mode.get(mode, {}).get(chosen, {}),
            cfg=cfg,
        )
        console.print(
            f"  [green]Stitched predictions dumped:[/green] {pred_path} "
            f"({n_unique} bars, {oos_first} → {oos_last})"
        )
    else:
        manifest = _json.loads(manifest_path.read_text())
        chosen = predictor or manifest.get("predictor")
        if chosen is None:
            console.print("[red]Manifest missing predictor field; re-run with --force-rebuild.[/red]")
            raise typer.Exit(1)
        console.print(f"  [cyan]Using cached predictions:[/cyan] {pred_path}")

    # Render the strategy file
    from .backtesting.strategy_template import write_strategy_file
    class_name = f"RegimeStrategy_{cfg.target}_{cfg.timeframe}_{_safe_class_name(chosen)}"
    can_short = (cfg.market_type == "futures")
    strategy_path = write_strategy_file(
        out_dir=strategies_dir,
        class_name=class_name,
        timeframe=cfg.timeframe,
        can_short=can_short,
        predictions_path=pred_path,
        predictor_name=chosen,
        target=cfg.target,
        quote=cfg.quote,
        settle=cfg.settle,
        market_type=cfg.market_type,
    )
    console.print(f"  [green]Strategy written:[/green] {strategy_path}")

    # Build + write the freqtrade config
    from .backtesting.freqtrade_runner import (
        build_freqtrade_config, write_freqtrade_config,
        run_backtest, parse_backtest_result,
    )
    ft_config = build_freqtrade_config(
        cfg, predictor_name=chosen, datadir=dd, user_data_dir=udd,
    )
    config_path = configs_dir / f"freqtrade_{cfg.target}_{cfg.timeframe}_{_safe_class_name(chosen)}.json"
    write_freqtrade_config(ft_config, config_path)
    console.print(f"  [green]Freqtrade config written:[/green] {config_path}")

    if export_strategy_only:
        console.print("[yellow]--export-strategy-only:[/yellow] skipping freqtrade subprocess.")
        return

    # Determine timerange.
    # Priority: CLI --timerange > cfg.backtest.timerange > derived from FEATHER.
    # We read from the feather itself (source of truth) rather than the
    # manifest: a previous bug had stale manifests survive cache hits, which
    # produced a freqtrade --timerange far narrower than the actual feather
    # span (= silent loss of most trades). The feather can't lie — if a label
    # row exists at date X, X belongs in the OOS span.
    if not timerange:
        timerange = cfg.backtest.timerange or ""
    if not timerange:
        import pandas as _pd
        _feather = _pd.read_feather(pred_path)
        _dates = _pd.to_datetime(_feather["date"], utc=True)
        oos_first = _dates.min().strftime("%Y%m%d")
        oos_last = _dates.max().strftime("%Y%m%d")
        timerange = f"{oos_first}-{oos_last}"
        # Reconcile manifest if it disagrees — silent stale manifests are
        # what bit us before; surface the divergence loudly.
        try:
            _mf = _json.loads(manifest_path.read_text())
            _mf_first = (_mf.get("oos_first") or "").replace("-", "")[:8]
            _mf_last = (_mf.get("oos_last") or "").replace("-", "")[:8]
            if (_mf_first, _mf_last) != (oos_first, oos_last):
                console.print(
                    f"  [yellow]WARN[/yellow] manifest OOS span "
                    f"({_mf_first}-{_mf_last}) disagrees with feather "
                    f"({oos_first}-{oos_last}); using feather."
                )
                _mf["oos_first"] = _dates.min().isoformat()
                _mf["oos_last"] = _dates.max().isoformat()
                manifest_path.write_text(_json.dumps(_mf, indent=2))
        except (FileNotFoundError, KeyError, ValueError):
            pass
    console.print(f"  [cyan]Freqtrade --timerange:[/cyan] {timerange}")

    export_path = results_dir / f"regime_{cfg.target}_{_safe_class_name(chosen)}.json"
    try:
        run_backtest(
            strategy_class=class_name,
            strategies_dir=strategies_dir,
            config_path=config_path,
            user_data_dir=udd,
            datadir=dd,
            timerange=timerange,
            export_path=export_path,
            breakdown=cfg.backtest.breakdown,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)

    ft_metrics = parse_backtest_result(export_path, class_name)

    # Side-by-side report (framework stitched OOS vs freqtrade)
    fw_metrics = _json.loads(manifest_path.read_text()).get("stitched_metrics") or {}
    # Backfill n_trades from the predictions feather when the manifest pre-dates
    # the count_trades commit. Cheap (single np.diff over <10k rows).
    if fw_metrics.get("n_trades") is None and pred_path.exists():
        try:
            import pandas as _pd
            from .evaluation.metrics import count_trades as _count_trades
            _df = _pd.read_feather(pred_path)
            fw_metrics["n_trades"] = _count_trades(
                _df["label"].to_numpy(),
                long_only=(cfg.market_type == "spot"),
            )
        except Exception:
            pass
    from .backtesting.report import format_side_by_side, format_breakdown
    table = format_side_by_side(fw_metrics, ft_metrics, cfg.backtest.divergence_warn_pct)
    console.print(table)

    # Per-period breakdown (freqtrade-only — no framework counterpart yet)
    breakdown_table = format_breakdown(
        ft_metrics.get("periodic_breakdown") or {},
        unit=cfg.backtest.breakdown,
        starting_balance=ft_metrics.get("starting_balance") or cfg.backtest.dry_run_wallet,
    )
    if breakdown_table is not None:
        console.print(breakdown_table)


@app.command()
def webserver(
    preset_or_yaml: str = typer.Argument(..., help="Same preset/YAML you used for `backtest`"),
    predictor: str = typer.Option(
        None, "--predictor",
        help="Predictor whose strategy + config to load. Default: derived from cached manifest.",
    ),
    user_data_dir: Path = typer.Option(
        None, "--user-data-dir",
        help="Same dir as `backtest` (where strategies/, configs/, backtest_results/ live).",
    ),
    datadir: Path = typer.Option(
        None, "--datadir",
        help="Where OHLCV feathers live. Default: cfg.paths.ohlcv.parent.parent.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address. 0.0.0.0 for remote access."),
    port: int = typer.Option(8080, "--port", help="UI port."),
):
    """Launch freqtrade webserver UI to view backtest charts.

    Reads the config previously generated by `regime-run backtest`. Auto-enables
    api_server (required by the UI) and passes --datadir / --user-data-dir
    explicitly so the chart view can locate OHLCV + backtest_results.

    After launch, open http://<host>:<port>; login with the credentials in
    the config (default freqtrade / regime_run).
    """
    import json as _json
    import subprocess

    cfg = _load_cfg(preset_or_yaml)
    udd = (user_data_dir or _default_user_data_dir(cfg)).expanduser().resolve()
    dd = (datadir or _default_datadir(cfg)).expanduser().resolve()

    # Resolve which config file to use. Honor --predictor; else look up the
    # most recently-modified freqtrade_*.json in <udd>/configs/.
    configs_dir = udd / "configs"
    if predictor:
        config_path = configs_dir / f"freqtrade_{cfg.target}_{cfg.timeframe}_{_safe_class_name(predictor)}.json"
    else:
        candidates = sorted(
            configs_dir.glob(f"freqtrade_{cfg.target}_{cfg.timeframe}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            console.print(
                f"[red]No freqtrade config found in {configs_dir}. "
                f"Run `regime-run backtest {preset_or_yaml}` first.[/red]"
            )
            raise typer.Exit(1)
        config_path = candidates[0]

    if not config_path.exists():
        console.print(
            f"[red]Config not found: {config_path}\n"
            f"Run `regime-run backtest {preset_or_yaml} --predictor {predictor or 'Ensemble-Conf'}` first.[/red]"
        )
        raise typer.Exit(1)

    # Auto-enable api_server (webserver UI requires it).
    config_data = _json.loads(config_path.read_text())
    api = config_data.setdefault("api_server", {})
    changed = False
    if not api.get("enabled"):
        api["enabled"] = True
        changed = True
    if api.get("listen_ip_address") != host:
        api["listen_ip_address"] = host
        changed = True
    if api.get("listen_port") != port:
        api["listen_port"] = port
        changed = True
    api.setdefault("username", "freqtrade")
    api.setdefault("password", "regime_run")
    api.setdefault("jwt_secret_key", "regime-run-replace-me")
    api.setdefault("ws_token", "regime-run-ws-replace-me")
    api.setdefault("CORS_origins", [])
    if changed:
        config_path.write_text(_json.dumps(config_data, indent=2))
        console.print(f"[cyan]Updated api_server in {config_path.name}[/cyan]")

    cmd = [
        "freqtrade", "webserver",
        "--config", str(config_path.resolve()),
        "--user-data-dir", str(udd),
        "--datadir", str(dd),
    ]
    console.print(f"  $ {' '.join(cmd)}")
    console.print(
        f"[green]UI: http://{host}:{port}  "
        f"({api['username']} / {api['password']})[/green]"
    )
    console.print(
        f"[dim]In the UI: Backtest tab → 'Backtest Results' → load a "
        f"zip from {udd / 'backtest_results'}[/dim]"
    )
    try:
        subprocess.run(cmd, shell=False, check=False)
    except FileNotFoundError:
        console.print("[red]freqtrade not found on PATH. Install: pip install freqtrade[/red]")
        raise typer.Exit(2)


@app.command()
def presets():
    """List available presets."""
    preset_dir = CONFIG_DIR / "presets"
    if not preset_dir.exists():
        console.print(f"[yellow]No presets directory found at {preset_dir}[/yellow]")
        return
    table = Table(title=f"Available presets in {preset_dir}")
    table.add_column("name", style="cyan")
    table.add_column("file")
    for p in sorted(preset_dir.glob("*.yaml")):
        table.add_row(p.stem, p.name)
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
