import pytest
from core.exceptions import PloutosError, DataFetchError, ConfigValidationError, TrainingError, InsufficientDataError

def test_exceptions_inheritance():
    assert issubclass(DataFetchError, PloutosError)
    assert issubclass(ConfigValidationError, PloutosError)
    assert issubclass(TrainingError, PloutosError)
    assert issubclass(InsufficientDataError, PloutosError)

def test_exceptions_raise():
    with pytest.raises(DataFetchError, match="Network timeout"):
        raise DataFetchError("Network timeout")

    with pytest.raises(ConfigValidationError, match="Invalid type"):
        raise ConfigValidationError("Invalid type")

    with pytest.raises(TrainingError, match="Model failed to converge"):
        raise TrainingError("Model failed to converge")

    with pytest.raises(InsufficientDataError, match="Not enough bars"):
        raise InsufficientDataError("Not enough bars")
