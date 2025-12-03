CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,
    quantity DECIMAL(10,4),
    price DECIMAL(10,2),
    amount DECIMAL(12,2),
    reason TEXT,
    portfolio_value DECIMAL(12,2)
);

CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(10,4),
    avg_entry_price DECIMAL(10,2),
    current_price DECIMAL(10,2),
    market_value DECIMAL(12,2),
    unrealized_pl DECIMAL(12,2),
    unrealized_plpc DECIMAL(8,4)
);

CREATE TABLE daily_summary (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    portfolio_value DECIMAL(12,2),
    cash DECIMAL(12,2),
    total_pl DECIMAL(12,2),
    positions_count INTEGER,
    trades_count INTEGER
);