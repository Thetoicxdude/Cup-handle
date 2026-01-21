"""SQLite State Schema for Dashboard Persistence

This module defines the SQLite schema for persisting dashboard state,
enabling multi-user state sharing and simulation persistence across page refreshes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class TradeRecord:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    strategy_type: str

@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_date: datetime
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss_price: float
    strategy_type: str
    pattern_details: Optional[Dict[str, Any]] = None


# Live simulations table - tracks active simulation sessions
CREATE_LIVE_SIMULATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS live_simulations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'running',  -- running, stopped, paused
    symbols             TEXT NOT NULL,  -- JSON array of symbols
    parameters          TEXT NOT NULL,  -- JSON of strategy parameters
    update_interval     INTEGER DEFAULT 300,
    initial_capital     REAL DEFAULT 1000000,
    created_at          TEXT DEFAULT (datetime('now', 'localtime')),
    last_heartbeat      TEXT DEFAULT (datetime('now', 'localtime')),
    stopped_at          TEXT
);
"""

# Simulation state table - detailed state snapshots
CREATE_SIMULATION_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS simulation_state (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_id       INTEGER NOT NULL,
    capital             REAL NOT NULL,
    positions           TEXT,  -- JSON of current positions
    trades              TEXT,  -- JSON of trade history
    logs                TEXT,  -- JSON of log entries
    active_signals      TEXT,  -- JSON of active signals
    evolution_history   TEXT,  -- JSON of evolution updates
    updated_at          TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (simulation_id) REFERENCES live_simulations(id) ON DELETE CASCADE
);
"""

# App settings table - global persistent settings
CREATE_APP_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS app_settings (
    key                 TEXT PRIMARY KEY,
    value               TEXT NOT NULL,
    updated_at          TEXT DEFAULT (datetime('now', 'localtime'))
);
"""

# Create indexes
CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_simulations_status ON live_simulations(status);
CREATE INDEX IF NOT EXISTS idx_simulations_heartbeat ON live_simulations(last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_state_simulation_id ON simulation_state(simulation_id);
"""

# Full schema for initialization
STATE_SCHEMA_SQL = f"""
-- Dashboard State Persistence Schema (SQLite)

{CREATE_LIVE_SIMULATIONS_TABLE}

{CREATE_SIMULATION_STATE_TABLE}

{CREATE_APP_SETTINGS_TABLE}

{CREATE_INDEXES}
"""


def get_state_schema_statements():
    """Get individual schema statements for step-by-step execution.
    
    Returns:
        List of SQL statements to execute in order
    """
    return [
        CREATE_LIVE_SIMULATIONS_TABLE,
        CREATE_SIMULATION_STATE_TABLE,
        CREATE_APP_SETTINGS_TABLE,
        CREATE_INDEXES,
    ]
