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
