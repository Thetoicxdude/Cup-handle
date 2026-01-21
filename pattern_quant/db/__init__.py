"""Database layer for AI PatternQuant

This module provides database schema and access layer for TimescaleDB and SQLite.
"""

from pattern_quant.db.schema import (
    SCHEMA_SQL,
    CREATE_STOCK_PRICES_TABLE,
    CREATE_TRADE_SIGNALS_TABLE,
    CREATE_POSITIONS_TABLE,
    CREATE_BACKTEST_RESULTS_TABLE,
)
from pattern_quant.db.repository import DatabaseRepository
from pattern_quant.db.state_manager import (
    StateManager,
    SimulationInfo,
    SimulationState,
    get_state_manager,
)

__all__ = [
    "SCHEMA_SQL",
    "CREATE_STOCK_PRICES_TABLE",
    "CREATE_TRADE_SIGNALS_TABLE",
    "CREATE_POSITIONS_TABLE",
    "CREATE_BACKTEST_RESULTS_TABLE",
    "DatabaseRepository",
    "StateManager",
    "SimulationInfo",
    "SimulationState",
    "get_state_manager",
]
