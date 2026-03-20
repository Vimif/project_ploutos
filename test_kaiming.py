import torch
import math
from unittest.mock import patch

# Mock kaiming
try:
    import torch.nn.init

    _original_kaiming = torch.nn.init.kaiming_uniform_

    def _safe_kaiming(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
        try:
            return _original_kaiming(tensor, a, mode, nonlinearity, generator)
        except TypeError:
            print("TypeError caught, falling back to math.sqrt")
            # Fallback for the TypeError: isinstance() arg 2 must be a type...
            with torch.no_grad():
                fan = torch.nn.init._calculate_correct_fan(tensor, mode)
                gain = torch.nn.init.calculate_gain(nonlinearity, a)
                std = float(gain) / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                return tensor.uniform_(-float(bound), float(bound))

    torch.nn.init.kaiming_uniform_ = _safe_kaiming
except ImportError:
    pass

import torch.nn as nn
linear = nn.Linear(10, 10)
print(linear)
