from .gdelt import collect_gdelt
from .rss import collect_rss
from .metaculus import collect_metaculus
from .polymarket import collect_polymarket
from .acled import collect_acled

__all__ = [
    "collect_gdelt",
    "collect_rss",
    "collect_metaculus",
    "collect_polymarket",
    "collect_acled",
]
