import numpy as np
import pandas as pd
from typing import Union
try:
    print(isinstance(np.array([1,2]), Union[pd.Series, np.ndarray]))
except Exception as e:
    print(e)
