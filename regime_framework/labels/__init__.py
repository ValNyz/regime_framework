from .base import BaseLabeller
from .trend_scan import TrendScanLabeller
from .triple_barrier import TripleBarrierLabeller
from .drawdown import DrawdownLabeller


def get_labeller(method: str, **kwargs) -> BaseLabeller:
    if method == "trend_scan":
        return TrendScanLabeller(**kwargs)
    if method == "triple_barrier":
        return TripleBarrierLabeller(**kwargs)
    if method == "drawdown":
        return DrawdownLabeller(**kwargs)
    raise ValueError(f"Unknown labelling method: {method}")


__all__ = [
    "BaseLabeller",
    "TrendScanLabeller",
    "TripleBarrierLabeller",
    "DrawdownLabeller",
    "get_labeller",
]
