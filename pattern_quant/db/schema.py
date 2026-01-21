"""TimescaleDB Schema for AI PatternQuant

This module defines the database schema for the pattern quantification system.
Includes hypertables for time-series data and regular tables for trading records.

Requirements: 14.1, 14.2, 8.1, 9.4
"""

# Stock prices hypertable - time-series data for OHLCV
CREATE_STOCK_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS stock_prices (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open        DECIMAL(12, 4),
    high        DECIMAL(12, 4),
    low         DECIMAL(12, 4),
    close       DECIMAL(12, 4),
    volume      BIGINT,
    PRIMARY KEY (time, symbol)
);
"""

# Convert stock_prices to TimescaleDB hypertable
CREATE_STOCK_PRICES_HYPERTABLE = """
SELECT create_hypertable('stock_prices', 'time', if_not_exists => TRUE);
"""

# Create index for symbol lookups
CREATE_STOCK_PRICES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices (symbol, time DESC);
"""

# Trade signals table
CREATE_TRADE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS trade_signals (
    id                      SERIAL PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    pattern_type            TEXT NOT NULL,
    match_score             DECIMAL(5, 2),
    breakout_price          DECIMAL(12, 4),
    stop_loss_price         DECIMAL(12, 4),
    status                  TEXT NOT NULL,
    expected_profit_ratio   DECIMAL(8, 4),
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);
"""

# Create index for trade signals lookups
CREATE_TRADE_SIGNALS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_trade_signals_symbol ON trade_signals (symbol);
CREATE INDEX IF NOT EXISTS idx_trade_signals_status ON trade_signals (status);
CREATE INDEX IF NOT EXISTS idx_trade_signals_created ON trade_signals (created_at DESC);
"""

# Positions table
CREATE_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS positions (
    id                      SERIAL PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    quantity                INTEGER NOT NULL,
    entry_price             DECIMAL(12, 4) NOT NULL,
    entry_time              TIMESTAMPTZ NOT NULL,
    sector                  TEXT,
    stop_loss_price         DECIMAL(12, 4),
    trailing_stop_active    BOOLEAN DEFAULT FALSE,
    trailing_stop_price     DECIMAL(12, 4),
    current_price           DECIMAL(12, 4),
    exit_price              DECIMAL(12, 4),
    exit_time               TIMESTAMPTZ,
    exit_reason             TEXT,
    pnl                     DECIMAL(12, 4),
    status                  TEXT DEFAULT 'open'
);
"""

# Create index for positions lookups
CREATE_POSITIONS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions (symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status);
CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions (entry_time DESC);
"""

# Backtest results table
CREATE_BACKTEST_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS backtest_results (
    id              SERIAL PRIMARY KEY,
    parameters      JSONB NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    total_trades    INTEGER,
    win_rate        DECIMAL(5, 2),
    total_return    DECIMAL(8, 4),
    max_drawdown    DECIMAL(8, 4),
    sharpe_ratio    DECIMAL(6, 4),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

# Create index for backtest results
CREATE_BACKTEST_RESULTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_backtest_results_created ON backtest_results (created_at DESC);
"""

# Factor configs table - stores indicator configurations per symbol
# Requirements: 8.1, 9.4
CREATE_FACTOR_CONFIGS_TABLE = """
CREATE TABLE IF NOT EXISTS factor_configs (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL UNIQUE,
    config_json     JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

# Create index for factor configs lookups
CREATE_FACTOR_CONFIGS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_factor_configs_symbol ON factor_configs (symbol);
"""

# Tuning results table - stores auto-tuning optimization results
# Requirements: 9.4
CREATE_TUNING_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS tuning_results (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    best_config     JSONB NOT NULL,
    win_rate        DECIMAL(5, 4),
    total_return    DECIMAL(8, 4),
    max_drawdown    DECIMAL(8, 4),
    sharpe_ratio    DECIMAL(6, 4),
    total_trades    INTEGER,
    backtest_start  DATE NOT NULL,
    backtest_end    DATE NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

# Create index for tuning results lookups
CREATE_TUNING_RESULTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_tuning_results_symbol ON tuning_results (symbol);
CREATE INDEX IF NOT EXISTS idx_tuning_results_created ON tuning_results (created_at DESC);
"""

# Indicator correlations table - stores correlation between indicators and win rate
# Requirements: 13.1, 13.2
CREATE_INDICATOR_CORRELATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS indicator_correlations (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    indicator_name  TEXT NOT NULL,
    correlation     DECIMAL(5, 4),
    sample_size     INTEGER,
    calculated_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, indicator_name)
);
"""

# Create index for indicator correlations lookups
CREATE_INDICATOR_CORRELATIONS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_indicator_correlations_symbol ON indicator_correlations (symbol);
"""

# Complete schema SQL for initialization
SCHEMA_SQL = f"""
-- AI PatternQuant Database Schema
-- TimescaleDB required

-- Stock prices hypertable
{CREATE_STOCK_PRICES_TABLE}

-- Convert to hypertable (TimescaleDB specific)
{CREATE_STOCK_PRICES_HYPERTABLE}

-- Stock prices indexes
{CREATE_STOCK_PRICES_INDEX}

-- Trade signals table
{CREATE_TRADE_SIGNALS_TABLE}

-- Trade signals indexes
{CREATE_TRADE_SIGNALS_INDEX}

-- Positions table
{CREATE_POSITIONS_TABLE}

-- Positions indexes
{CREATE_POSITIONS_INDEX}

-- Backtest results table
{CREATE_BACKTEST_RESULTS_TABLE}

-- Backtest results indexes
{CREATE_BACKTEST_RESULTS_INDEX}

-- Factor configs table
{CREATE_FACTOR_CONFIGS_TABLE}

-- Factor configs indexes
{CREATE_FACTOR_CONFIGS_INDEX}

-- Tuning results table
{CREATE_TUNING_RESULTS_TABLE}

-- Tuning results indexes
{CREATE_TUNING_RESULTS_INDEX}

-- Indicator correlations table
{CREATE_INDICATOR_CORRELATIONS_TABLE}

-- Indicator correlations indexes
{CREATE_INDICATOR_CORRELATIONS_INDEX}
"""


def get_schema_statements():
    """Get individual schema statements for step-by-step execution.
    
    Returns:
        List of SQL statements to execute in order
    """
    return [
        CREATE_STOCK_PRICES_TABLE,
        CREATE_STOCK_PRICES_HYPERTABLE,
        CREATE_STOCK_PRICES_INDEX,
        CREATE_TRADE_SIGNALS_TABLE,
        CREATE_TRADE_SIGNALS_INDEX,
        CREATE_POSITIONS_TABLE,
        CREATE_POSITIONS_INDEX,
        CREATE_BACKTEST_RESULTS_TABLE,
        CREATE_BACKTEST_RESULTS_INDEX,
        CREATE_FACTOR_CONFIGS_TABLE,
        CREATE_FACTOR_CONFIGS_INDEX,
        CREATE_TUNING_RESULTS_TABLE,
        CREATE_TUNING_RESULTS_INDEX,
        CREATE_INDICATOR_CORRELATIONS_TABLE,
        CREATE_INDICATOR_CORRELATIONS_INDEX,
    ]
