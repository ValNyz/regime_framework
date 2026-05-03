from .base import BaseLabeller
from .trend_scan import TrendScanLabeller
from .triple_barrier import TripleBarrierLabeller
from .drawdown import DrawdownLabeller


def get_labeller(method: str, **kwargs) -> BaseLabeller:
    """Each labeller has its own kwargs — filter to avoid 'unexpected keyword' errors."""
    import inspect
    if method == "trend_scan":
        cls = TrendScanLabeller
    elif method == "triple_barrier":
        cls = TripleBarrierLabeller
    elif method == "drawdown":
        cls = DrawdownLabeller
    else:
        raise ValueError(f"Unknown labelling method: {method}")
    accepted = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


__all__ = [
    "BaseLabeller",
    "TrendScanLabeller",
    "TripleBarrierLabeller",
    "DrawdownLabeller",
    "get_labeller",
]
