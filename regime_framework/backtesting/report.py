"""Side-by-side comparison: framework's stitched OOS vs freqtrade backtest.

The framework's metrics come from `stitched_metrics` written into the
predictions manifest (gain, sharpe, max_dd computed on the concatenated
OOS series). Freqtrade's metrics come from parse_backtest_result.

Divergence is reported as relative gap on each metric. Threshold default
(BacktestConfig.divergence_warn_pct = 10%) flags rows where the two
mesures disagree enough to warrant investigation — typically slippage,
funding, or order-type modeling.
"""
from __future__ import annotations

from rich.table import Table


def format_breakdown(breakdown: dict, unit: str = "month") -> Table | None:
    """Build a 4-column rich.Table of per-period freqtrade results.

    `breakdown` shape (freqtrade 2026): either a dict {unit: list[dict]}
    where each dict has keys like {date, profit_abs, profit_pct,
    wins, draws, loses}, or directly a list[dict] when only one unit was
    requested. We try both and degrade gracefully on schema variants.
    """
    if not breakdown:
        return None
    rows: list[dict] = []
    if isinstance(breakdown, dict):
        # Prefer the requested unit; fall back to anything non-empty.
        for key in (unit, f"{unit}ly", f"{unit}s"):
            v = breakdown.get(key)
            if v:
                rows = v if isinstance(v, list) else [v]
                break
        if not rows:
            for k, v in breakdown.items():
                if v:
                    rows = v if isinstance(v, list) else [v]
                    unit = k
                    break
    elif isinstance(breakdown, list):
        rows = breakdown

    if not rows:
        return None

    table = Table(title=f"Freqtrade backtest — {unit}ly breakdown", show_lines=False)
    table.add_column(unit, style="cyan", no_wrap=True)
    table.add_column("profit (abs)", justify="right")
    table.add_column("profit %", justify="right")
    table.add_column("trades", justify="right")
    table.add_column("wins/draws/loses", justify="right")

    for row in rows:
        date = str(row.get("date") or row.get("period") or "?")[:10]
        prof_abs = row.get("profit_abs") or row.get("rel_profit") or row.get("profit")
        prof_pct = row.get("profit_pct") or row.get("profit_total_pct") or row.get("profit_percentage")
        trades = row.get("trade_count") or row.get("trades") or row.get("total_trades")
        wins = row.get("wins")
        draws = row.get("draws")
        loses = row.get("loses") or row.get("losses")
        wdl = " / ".join(
            str(int(x)) if x is not None else "?" for x in (wins, draws, loses)
        )
        prof_abs_str = f"{float(prof_abs):+8.2f}" if prof_abs is not None else "    --  "
        prof_pct_str = f"{float(prof_pct) * 100:+6.2f}%" if prof_pct is not None else "  --  "
        trades_str = str(int(trades)) if trades is not None else "--"
        table.add_row(date, prof_abs_str, prof_pct_str, trades_str, wdl)

    return table


def _fmt_pct(v) -> str:
    if v is None:
        return "  --  "
    return f"{float(v) * 100:+6.2f}%" if abs(float(v)) <= 5 else f"{float(v):+6.2f}"


def _fmt_num(v, fmt: str = "+.2f") -> str:
    if v is None:
        return "  --  "
    return format(float(v), fmt)


def _divergence(framework_v, freqtrade_v) -> float | None:
    """Relative divergence in percentage points; None if either side is missing."""
    if framework_v is None or freqtrade_v is None:
        return None
    fa = abs(float(framework_v))
    if fa < 1e-9:
        return None
    return abs(float(framework_v) - float(freqtrade_v)) / fa * 100.0


def format_side_by_side(
    framework_metrics: dict,
    freqtrade_metrics: dict,
    divergence_warn_pct: float = 10.0,
) -> Table:
    """Build a 4-column rich.Table: metric, framework, freqtrade, divergence."""
    table = Table(
        title="Backtest comparison: framework stitched OOS vs freqtrade",
        show_lines=False,
    )
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("framework", justify="right")
    table.add_column("freqtrade", justify="right")
    table.add_column("Δ %", justify="right")

    rows = [
        ("gain_total",  "gain",         "profit_total_pct", _fmt_pct, _fmt_pct),
        ("Sharpe",      "sharpe",       "sharpe",           _fmt_num, _fmt_num),
        ("max_drawdown","max_dd",       "max_drawdown_pct", _fmt_pct, _fmt_pct),
        ("Calmar",      "calmar",       None,               _fmt_num, lambda v: "  --  "),
        ("trades",      None,           "total_trades",     lambda v: "  --  ", lambda v: _fmt_num(v, ".0f")),
    ]

    for label, fw_key, ft_key, fw_fmt, ft_fmt in rows:
        fw = framework_metrics.get(fw_key) if fw_key else None
        ft = freqtrade_metrics.get(ft_key) if ft_key else None
        div = _divergence(fw, ft) if (fw_key and ft_key) else None
        warn = (div is not None) and (div > divergence_warn_pct)
        div_str = "  --  " if div is None else f"{div:+5.1f}%"
        if warn:
            div_str = f"[yellow]{div_str}[/yellow]"
        table.add_row(label, fw_fmt(fw), ft_fmt(ft), div_str)

    return table
