import pytest

from core.exceptions import (
    ConfigValidationError,
    DataFetchError,
    InsufficientDataError,
    TrainingError,
)


def test_exceptions():
    with pytest.raises(DataFetchError):
        raise DataFetchError("No data")
    with pytest.raises(ConfigValidationError):
        raise ConfigValidationError("Bad config")
    with pytest.raises(TrainingError):
        raise TrainingError("Fail")
    with pytest.raises(InsufficientDataError):
        raise InsufficientDataError("Too little")
