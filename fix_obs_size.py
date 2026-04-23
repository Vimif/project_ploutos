import os

filepath = 'core/observation_builder.py'
with open(filepath, 'rb') as f:
    content = f.read()

search = b'''        self.obs_size = int(
            self.n_assets * self.n_features
            + self.n_macro_features
            + self.n_assets  # position percentages
            + self.n_assets  # unrealized PnL per position
            + 3  # cash_pct, total_return, drawdown
            + 3  # recent returns: 1-step, 5-step, 20-step
        )'''

replace = b'''        self.obs_size = (
            self.n_assets * self.n_features
            + self.n_macro_features
            + self.n_assets  # position percentages
            + self.n_assets  # unrealized PnL per position
            + 3  # cash_pct, total_return, drawdown
            + 3  # recent returns: 1-step, 5-step, 20-step
        )'''

search = search.replace(b'\n', b'\r\n')
replace = replace.replace(b'\n', b'\r\n')

if search not in content:
    search = search.replace(b'\r\n', b'\n')
    replace = replace.replace(b'\r\n', b'\n')

content = content.replace(search, replace)

with open(filepath, 'wb') as f:
    f.write(content)
