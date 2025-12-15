"""
Web Routes pour Ploutos Dashboard
"""

try:
    from .watchlists import watchlists_bp
except ImportError:
    watchlists_bp = None

try:
    from .live_trading import live_bp
except ImportError:
    live_bp = None

__all__ = ['watchlists_bp', 'live_bp']
