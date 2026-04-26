import re

with open("core/observation_builder.py", "rb") as f:
    content = f.read()

old_build = b"""    def build(
        self,
        current_step: int,
        portfolio: Dict[str, float],
        prices: Dict[str, float],
        equity: float,
        balance: float,
        initial_balance: float,
        peak_value: float,
        entry_prices: Optional[Dict[str, float]] = None,
        portfolio_value_history: Optional[deque] = None,
    ) -> np.ndarray:
        \"\"\"Build observation vector for current step.

        Args:
            current_step: Current timestep index.
            portfolio: Dict of ticker -> quantity held.
            prices: Dict of ticker -> current price.
            equity: Current total portfolio value.
            balance: Current cash balance.
            initial_balance: Starting balance.
            peak_value: Historical peak equity.
            entry_prices: Dict of ticker -> entry price for open positions.
            portfolio_value_history: Recent equity history for return calculation.

        Returns:
            Flat numpy observation vector.
        \"\"\"
        clip = OBSERVATION_CLIP_RANGE
        obs_parts = []

        # Technical features per ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                features = np.zeros(self.n_features, dtype=np.float32)
            else:
                features = features_array[current_step]
            features = np.nan_to_num(features, nan=0.0, posinf=clip, neginf=-clip)
            features = np.clip(features, -clip, clip)
            obs_parts.append(features)

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                macro_features = self.macro_array[current_step]
            else:
                macro_features = np.zeros(self.n_macro_features, dtype=np.float32)
            macro_features = np.nan_to_num(macro_features, nan=0.0, posinf=clip, neginf=-clip)
            macro_features = np.clip(macro_features, -clip, clip)
            obs_parts.append(macro_features)

        # Position percentages
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0
            obs_parts.append([np.clip(position_pct, 0, 1)])

        # Unrealized PnL per position
        if entry_prices is None:
            entry_prices = {}
        for ticker in self.tickers:
            entry = entry_prices.get(ticker, 0.0)
            qty = portfolio.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            if entry > 0 and qty > 0 and price > 0:
                unrealized_pnl = (price - entry) / entry
            else:
                unrealized_pnl = 0.0
            obs_parts.append([np.clip(unrealized_pnl, -1.0, 5.0)])

        # Portfolio state
        cash_pct = np.clip(balance / (equity + EQUITY_EPSILON), 0, 1)
        total_return = np.clip((equity - initial_balance) / initial_balance, -1, 5)
        drawdown = np.clip((peak_value - equity) / (peak_value + EQUITY_EPSILON), 0, 1)
        obs_parts.append([cash_pct, total_return, drawdown])

        # Recent portfolio returns (1-step, 5-step, 20-step)
        hist = list(portfolio_value_history) if portfolio_value_history else []

        def _recent_return(lookback):
            if len(hist) > lookback and hist[-lookback - 1] > 0:
                return (hist[-1] - hist[-lookback - 1]) / hist[-lookback - 1]
            return 0.0

        ret_1 = np.clip(_recent_return(1), -0.5, 0.5)
        ret_5 = np.clip(_recent_return(5), -0.5, 0.5)
        ret_20 = np.clip(_recent_return(20), -0.5, 0.5)
        obs_parts.append([ret_1, ret_5, ret_20])

        obs = np.concatenate([np.array(p).flatten() for p in obs_parts])
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        obs = np.clip(obs, -clip, clip)

        return obs.astype(np.float32)"""

new_build = b"""    def build(
        self,
        current_step: int,
        portfolio: Dict[str, float],
        prices: Dict[str, float],
        equity: float,
        balance: float,
        initial_balance: float,
        peak_value: float,
        entry_prices: Optional[Dict[str, float]] = None,
        portfolio_value_history: Optional[deque] = None,
    ) -> np.ndarray:
        \"\"\"Build observation vector for current step.

        Args:
            current_step: Current timestep index.
            portfolio: Dict of ticker -> quantity held.
            prices: Dict of ticker -> current price.
            equity: Current total portfolio value.
            balance: Current cash balance.
            initial_balance: Starting balance.
            peak_value: Historical peak equity.
            entry_prices: Dict of ticker -> entry price for open positions.
            portfolio_value_history: Recent equity history for return calculation.

        Returns:
            Flat numpy observation vector.
        \"\"\"
        clip = OBSERVATION_CLIP_RANGE
        obs = np.zeros(self.obs_size, dtype=np.float32)
        idx = 0

        # Technical features per ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                obs[idx : idx + self.n_features] = 0.0
            else:
                obs[idx : idx + self.n_features] = features_array[current_step]
            idx += self.n_features

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                obs[idx : idx + self.n_macro_features] = self.macro_array[current_step]
            else:
                obs[idx : idx + self.n_macro_features] = 0.0
            idx += self.n_macro_features

        # Position percentages
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0
            obs[idx] = min(max(position_pct, 0.0), 1.0)
            idx += 1

        # Unrealized PnL per position
        if entry_prices is None:
            entry_prices = {}
        for ticker in self.tickers:
            entry = entry_prices.get(ticker, 0.0)
            qty = portfolio.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            if entry > 0 and qty > 0 and price > 0:
                unrealized_pnl = (price - entry) / entry
            else:
                unrealized_pnl = 0.0
            obs[idx] = min(max(unrealized_pnl, -1.0), 5.0)
            idx += 1

        # Portfolio state
        obs[idx] = min(max(balance / (equity + EQUITY_EPSILON), 0.0), 1.0)
        obs[idx + 1] = min(max((equity - initial_balance) / initial_balance, -1.0), 5.0)
        obs[idx + 2] = min(max((peak_value - equity) / (peak_value + EQUITY_EPSILON), 0.0), 1.0)
        idx += 3

        # Recent portfolio returns (1-step, 5-step, 20-step)
        hist = list(portfolio_value_history) if portfolio_value_history else []

        def _recent_return(lookback):
            if len(hist) > lookback and hist[-lookback - 1] > 0:
                return (hist[-1] - hist[-lookback - 1]) / hist[-lookback - 1]
            return 0.0

        obs[idx] = min(max(_recent_return(1), -0.5), 0.5)
        obs[idx + 1] = min(max(_recent_return(5), -0.5), 0.5)
        obs[idx + 2] = min(max(_recent_return(20), -0.5), 0.5)

        np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(obs, -clip, clip, out=obs)

        return obs"""

old_build_crlf = old_build.replace(b'\n', b'\r\n')
new_build_crlf = new_build.replace(b'\n', b'\r\n')

if old_build in content:
    content = content.replace(old_build, new_build)
elif old_build_crlf in content:
    content = content.replace(old_build_crlf, new_build_crlf)
else:
    print("WARNING: Could not find exactly old_build in the file.")

with open("core/observation_builder.py", "wb") as f:
    f.write(content)
