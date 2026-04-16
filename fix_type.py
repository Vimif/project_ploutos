import sys

with open("core/transaction_costs.py", "rb") as f:
    content = f.read()

# Replace Union[...] with ... | ...
content = content.replace(b"from typing import Dict, Tuple, Union\r\n", b"from typing import Dict, Tuple\r\n")
content = content.replace(b"from typing import Dict, Tuple, Union\n", b"from typing import Dict, Tuple\n")
content = content.replace(b"Union[pd.Series, np.ndarray]", b"pd.Series | np.ndarray")

with open("core/transaction_costs.py", "wb") as f:
    f.write(content)

print("Fixed type hints.")
