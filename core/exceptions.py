# core/exceptions.py
"""Custom exception hierarchy for Ploutos trading system."""


class PloutosError(Exception):
    """Base exception for all Ploutos errors."""

    pass


class DataFetchError(PloutosError):
    """Raised when data fetching fails (network, API, timeout)."""

    pass


class ConfigValidationError(PloutosError):
    """Raised when YAML config is invalid (types, ranges, constraints)."""

    pass


class TrainingError(PloutosError):
    """Raised when training fails (env creation, model loading, etc.)."""

    pass


class InsufficientDataError(PloutosError):
    """Raised when not enough data is available for a walk-forward fold."""

    pass
