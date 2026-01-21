"""Database Repository for AI PatternQuant

This module provides the database access layer for CRUD operations
on stock prices, trade signals, positions, and backtest results.

Requirements: 14.1, 14.2, 14.3, 8.1, 8.2, 8.3
"""

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple

from pattern_quant.core.models import (
    OHLCV,
    TradeSignal,
    Position,
    SignalStatus,
    ExitReason,
)
from pattern_quant.db.schema import get_schema_statements

logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Database access layer for AI PatternQuant.
    
    Provides CRUD operations for:
    - OHLCV stock price data
    - Trade signals
    - Positions
    - Backtest results
    
    Attributes:
        connection: Database connection object (psycopg2 or compatible)
    """
    
    def __init__(self, connection):
        """Initialize the repository with a database connection.
        
        Args:
            connection: A psycopg2-compatible database connection
        """
        self.connection = connection
    
    def initialize_schema(self) -> bool:
        """Initialize the database schema.
        
        Creates all required tables and indexes if they don't exist.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            for statement in get_schema_statements():
                cursor.execute(statement)
            self.connection.commit()
            cursor.close()
            logger.info("Database schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            self.connection.rollback()
            return False
    
    # ==================== OHLCV Operations ====================
    
    def insert_ohlcv(self, ohlcv: OHLCV) -> bool:
        """Insert a single OHLCV record.
        
        Args:
            ohlcv: OHLCV data to insert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO stock_prices (time, symbol, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (time, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                (ohlcv.time, ohlcv.symbol, ohlcv.open, ohlcv.high,
                 ohlcv.low, ohlcv.close, ohlcv.volume)
            )
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to insert OHLCV: {e}")
            self.connection.rollback()
            return False
    
    def insert_ohlcv_batch(self, ohlcv_list: List[OHLCV]) -> Tuple[int, int]:
        """Insert multiple OHLCV records in batch.
        
        Args:
            ohlcv_list: List of OHLCV data to insert
            
        Returns:
            Tuple of (success_count, failure_count)
        """
        if not ohlcv_list:
            return (0, 0)
        
        success_count = 0
        failure_count = 0
        
        try:
            cursor = self.connection.cursor()
            for ohlcv in ohlcv_list:
                try:
                    cursor.execute(
                        """
                        INSERT INTO stock_prices (time, symbol, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (time, symbol) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                        """,
                        (ohlcv.time, ohlcv.symbol, ohlcv.open, ohlcv.high,
                         ohlcv.low, ohlcv.close, ohlcv.volume)
                    )
                    success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to insert OHLCV for {ohlcv.symbol}: {e}")
                    failure_count += 1
            
            self.connection.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            self.connection.rollback()
            failure_count = len(ohlcv_list)
            success_count = 0
        
        return (success_count, failure_count)

    
    def get_ohlcv_by_symbol(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Get OHLCV data for a symbol within a time range.
        
        Args:
            symbol: Stock symbol
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            limit: Maximum number of records to return
            
        Returns:
            List of OHLCV records ordered by time ascending
        """
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT time, symbol, open, high, low, close, volume FROM stock_prices WHERE symbol = %s"
            params = [symbol]
            
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
            
            query += " ORDER BY time ASC"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
            
            return [
                OHLCV(
                    time=row[0],
                    symbol=row[1],
                    open=float(row[2]) if row[2] else 0.0,
                    high=float(row[3]) if row[3] else 0.0,
                    low=float(row[4]) if row[4] else 0.0,
                    close=float(row[5]) if row[5] else 0.0,
                    volume=int(row[6]) if row[6] else 0
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            return []
    
    def get_latest_ohlcv(self, symbol: str) -> Optional[OHLCV]:
        """Get the most recent OHLCV record for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest OHLCV record or None if not found
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT time, symbol, open, high, low, close, volume
                FROM stock_prices
                WHERE symbol = %s
                ORDER BY time DESC
                LIMIT 1
                """,
                (symbol,)
            )
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return OHLCV(
                    time=row[0],
                    symbol=row[1],
                    open=float(row[2]) if row[2] else 0.0,
                    high=float(row[3]) if row[3] else 0.0,
                    low=float(row[4]) if row[4] else 0.0,
                    close=float(row[5]) if row[5] else 0.0,
                    volume=int(row[6]) if row[6] else 0
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get latest OHLCV for {symbol}: {e}")
            return None
    
    def delete_ohlcv(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """Delete OHLCV records for a symbol within a time range.
        
        Args:
            symbol: Stock symbol
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            
        Returns:
            Number of records deleted
        """
        try:
            cursor = self.connection.cursor()
            
            query = "DELETE FROM stock_prices WHERE symbol = %s"
            params = [symbol]
            
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
            
            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            self.connection.commit()
            cursor.close()
            
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete OHLCV for {symbol}: {e}")
            self.connection.rollback()
            return 0
    
    # ==================== Trade Signal Operations ====================
    
    def insert_trade_signal(self, signal: TradeSignal) -> Optional[int]:
        """Insert a trade signal.
        
        Args:
            signal: TradeSignal to insert
            
        Returns:
            ID of inserted record or None if failed
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO trade_signals 
                (symbol, pattern_type, match_score, breakout_price, stop_loss_price, 
                 status, expected_profit_ratio, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (signal.symbol, signal.pattern_type, signal.match_score,
                 signal.breakout_price, signal.stop_loss_price,
                 signal.status.value, signal.expected_profit_ratio, signal.created_at)
            )
            signal_id = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            return signal_id
        except Exception as e:
            logger.error(f"Failed to insert trade signal: {e}")
            self.connection.rollback()
            return None

    
    def get_trade_signal_by_id(self, signal_id: int) -> Optional[TradeSignal]:
        """Get a trade signal by ID.
        
        Args:
            signal_id: ID of the trade signal
            
        Returns:
            TradeSignal or None if not found
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT symbol, pattern_type, match_score, breakout_price, stop_loss_price,
                       status, expected_profit_ratio, created_at
                FROM trade_signals
                WHERE id = %s
                """,
                (signal_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return TradeSignal(
                    symbol=row[0],
                    pattern_type=row[1],
                    match_score=float(row[2]) if row[2] else 0.0,
                    breakout_price=float(row[3]) if row[3] else 0.0,
                    stop_loss_price=float(row[4]) if row[4] else 0.0,
                    status=SignalStatus(row[5]),
                    expected_profit_ratio=float(row[6]) if row[6] else 0.0,
                    created_at=row[7]
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get trade signal {signal_id}: {e}")
            return None
    
    def get_trade_signals_by_status(
        self,
        status: SignalStatus,
        limit: Optional[int] = None
    ) -> List[TradeSignal]:
        """Get trade signals by status.
        
        Args:
            status: Signal status to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of TradeSignal records
        """
        try:
            cursor = self.connection.cursor()
            
            query = """
                SELECT symbol, pattern_type, match_score, breakout_price, stop_loss_price,
                       status, expected_profit_ratio, created_at
                FROM trade_signals
                WHERE status = %s
                ORDER BY created_at DESC
            """
            params = [status.value]
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
            
            return [
                TradeSignal(
                    symbol=row[0],
                    pattern_type=row[1],
                    match_score=float(row[2]) if row[2] else 0.0,
                    breakout_price=float(row[3]) if row[3] else 0.0,
                    stop_loss_price=float(row[4]) if row[4] else 0.0,
                    status=SignalStatus(row[5]),
                    expected_profit_ratio=float(row[6]) if row[6] else 0.0,
                    created_at=row[7]
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get trade signals by status {status}: {e}")
            return []
    
    def update_trade_signal_status(
        self,
        signal_id: int,
        new_status: SignalStatus
    ) -> bool:
        """Update the status of a trade signal.
        
        Args:
            signal_id: ID of the trade signal
            new_status: New status to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE trade_signals
                SET status = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (new_status.value, signal_id)
            )
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to update trade signal status: {e}")
            self.connection.rollback()
            return False
    
    # ==================== Position Operations ====================
    
    def insert_position(self, position: Position) -> Optional[int]:
        """Insert a new position.
        
        Args:
            position: Position to insert
            
        Returns:
            ID of inserted record or None if failed
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO positions 
                (symbol, quantity, entry_price, entry_time, sector, stop_loss_price,
                 trailing_stop_active, trailing_stop_price, current_price, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'open')
                RETURNING id
                """,
                (position.symbol, position.quantity, position.entry_price,
                 position.entry_time, position.sector, position.stop_loss_price,
                 position.trailing_stop_active, position.trailing_stop_price,
                 position.current_price)
            )
            position_id = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            return position_id
        except Exception as e:
            logger.error(f"Failed to insert position: {e}")
            self.connection.rollback()
            return None

    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions.
        
        Returns:
            List of position dictionaries with id included
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT id, symbol, quantity, entry_price, entry_time, sector,
                       stop_loss_price, trailing_stop_active, trailing_stop_price,
                       current_price
                FROM positions
                WHERE status = 'open'
                ORDER BY entry_time DESC
                """
            )
            rows = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    'id': row[0],
                    'position': Position(
                        symbol=row[1],
                        quantity=row[2],
                        entry_price=float(row[3]) if row[3] else 0.0,
                        current_price=float(row[9]) if row[9] else 0.0,
                        sector=row[5] or 'Unknown',
                        entry_time=row[4],
                        stop_loss_price=float(row[6]) if row[6] else 0.0,
                        trailing_stop_active=row[7] or False,
                        trailing_stop_price=float(row[8]) if row[8] else 0.0
                    )
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []
    
    def get_positions_by_sector(self, sector: str) -> List[Dict[str, Any]]:
        """Get open positions by sector.
        
        Args:
            sector: Sector to filter by
            
        Returns:
            List of position dictionaries with id included
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT id, symbol, quantity, entry_price, entry_time, sector,
                       stop_loss_price, trailing_stop_active, trailing_stop_price,
                       current_price
                FROM positions
                WHERE status = 'open' AND sector = %s
                ORDER BY entry_time DESC
                """,
                (sector,)
            )
            rows = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    'id': row[0],
                    'position': Position(
                        symbol=row[1],
                        quantity=row[2],
                        entry_price=float(row[3]) if row[3] else 0.0,
                        current_price=float(row[9]) if row[9] else 0.0,
                        sector=row[5] or 'Unknown',
                        entry_time=row[4],
                        stop_loss_price=float(row[6]) if row[6] else 0.0,
                        trailing_stop_active=row[7] or False,
                        trailing_stop_price=float(row[8]) if row[8] else 0.0
                    )
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get positions by sector {sector}: {e}")
            return []
    
    def update_position_price(
        self,
        position_id: int,
        current_price: float
    ) -> bool:
        """Update the current price of a position.
        
        Args:
            position_id: ID of the position
            current_price: New current price
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE positions
                SET current_price = %s
                WHERE id = %s
                """,
                (current_price, position_id)
            )
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to update position price: {e}")
            self.connection.rollback()
            return False
    
    def update_position_trailing_stop(
        self,
        position_id: int,
        trailing_stop_active: bool,
        trailing_stop_price: float
    ) -> bool:
        """Update trailing stop settings for a position.
        
        Args:
            position_id: ID of the position
            trailing_stop_active: Whether trailing stop is active
            trailing_stop_price: Trailing stop price
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE positions
                SET trailing_stop_active = %s, trailing_stop_price = %s
                WHERE id = %s
                """,
                (trailing_stop_active, trailing_stop_price, position_id)
            )
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to update trailing stop: {e}")
            self.connection.rollback()
            return False
    
    def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: ExitReason,
        pnl: float
    ) -> bool:
        """Close a position.
        
        Args:
            position_id: ID of the position
            exit_price: Exit price
            exit_reason: Reason for exit
            pnl: Profit/loss amount
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE positions
                SET exit_price = %s, exit_time = NOW(), exit_reason = %s,
                    pnl = %s, status = 'closed'
                WHERE id = %s
                """,
                (exit_price, exit_reason.value, pnl, position_id)
            )
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            self.connection.rollback()
            return False

    
    # ==================== Backtest Results Operations ====================
    
    def insert_backtest_result(
        self,
        parameters: Dict[str, Any],
        start_date: date,
        end_date: date,
        total_trades: int,
        win_rate: float,
        total_return: float,
        max_drawdown: float,
        sharpe_ratio: float
    ) -> Optional[int]:
        """Insert a backtest result.
        
        Args:
            parameters: Backtest parameters as JSON
            start_date: Start date of backtest period
            end_date: End date of backtest period
            total_trades: Total number of trades
            win_rate: Win rate percentage
            total_return: Total return percentage
            max_drawdown: Maximum drawdown percentage
            sharpe_ratio: Sharpe ratio
            
        Returns:
            ID of inserted record or None if failed
        """
        try:
            import json
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO backtest_results 
                (parameters, start_date, end_date, total_trades, win_rate,
                 total_return, max_drawdown, sharpe_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (json.dumps(parameters), start_date, end_date, total_trades,
                 win_rate, total_return, max_drawdown, sharpe_ratio)
            )
            result_id = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            return result_id
        except Exception as e:
            logger.error(f"Failed to insert backtest result: {e}")
            self.connection.rollback()
            return None
    
    def get_backtest_results(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get backtest results.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of backtest result dictionaries
        """
        try:
            import json
            cursor = self.connection.cursor()
            
            query = """
                SELECT id, parameters, start_date, end_date, total_trades,
                       win_rate, total_return, max_drawdown, sharpe_ratio, created_at
                FROM backtest_results
                ORDER BY created_at DESC
            """
            
            if limit:
                query += " LIMIT %s"
                cursor.execute(query, (limit,))
            else:
                cursor.execute(query)
            
            rows = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    'id': row[0],
                    'parameters': row[1] if isinstance(row[1], dict) else json.loads(row[1]) if row[1] else {},
                    'start_date': row[2],
                    'end_date': row[3],
                    'total_trades': row[4],
                    'win_rate': float(row[5]) if row[5] else 0.0,
                    'total_return': float(row[6]) if row[6] else 0.0,
                    'max_drawdown': float(row[7]) if row[7] else 0.0,
                    'sharpe_ratio': float(row[8]) if row[8] else 0.0,
                    'created_at': row[9]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get backtest results: {e}")
            return []
    
    # ==================== Utility Methods ====================
    
    def get_symbols_with_data(self) -> List[str]:
        """Get list of symbols that have OHLCV data.
        
        Returns:
            List of unique symbols
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol")
            rows = cursor.fetchall()
            cursor.close()
            return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []
    
    def get_data_date_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """Get the date range of available data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (earliest_time, latest_time) or None if no data
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT MIN(time), MAX(time)
                FROM stock_prices
                WHERE symbol = %s
                """,
                (symbol,)
            )
            row = cursor.fetchone()
            cursor.close()
            
            if row and row[0] and row[1]:
                return (row[0], row[1])
            return None
        except Exception as e:
            logger.error(f"Failed to get date range for {symbol}: {e}")
            return None
    
    def count_ohlcv_records(self, symbol: Optional[str] = None) -> int:
        """Count OHLCV records.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Number of records
        """
        try:
            cursor = self.connection.cursor()
            
            if symbol:
                cursor.execute(
                    "SELECT COUNT(*) FROM stock_prices WHERE symbol = %s",
                    (symbol,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM stock_prices")
            
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Exception as e:
            logger.error(f"Failed to count OHLCV records: {e}")
            return 0

    # ==================== Factor Config Operations ====================
    # Requirements: 8.1, 8.2, 8.3
    
    def insert_factor_config(
        self,
        symbol: str,
        config_json: Dict[str, Any]
    ) -> Optional[int]:
        """Insert a new factor configuration.
        
        Args:
            symbol: Stock symbol
            config_json: Configuration as JSON-serializable dict
            
        Returns:
            ID of inserted record or None if failed
        """
        try:
            import json
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO factor_configs (symbol, config_json)
                VALUES (%s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    config_json = EXCLUDED.config_json,
                    updated_at = NOW()
                RETURNING id
                """,
                (symbol, json.dumps(config_json))
            )
            config_id = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            return config_id
        except Exception as e:
            logger.error(f"Failed to insert factor config for {symbol}: {e}")
            self.connection.rollback()
            return None
    
    def get_factor_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get factor configuration for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Configuration dict or None if not found
        """
        try:
            import json
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT config_json, created_at, updated_at
                FROM factor_configs
                WHERE symbol = %s
                """,
                (symbol,)
            )
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                config_data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                return {
                    'config': config_data,
                    'created_at': row[1],
                    'updated_at': row[2]
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get factor config for {symbol}: {e}")
            return None
    
    def update_factor_config(
        self,
        symbol: str,
        config_json: Dict[str, Any]
    ) -> bool:
        """Update factor configuration for a symbol.
        
        Args:
            symbol: Stock symbol
            config_json: New configuration as JSON-serializable dict
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE factor_configs
                SET config_json = %s, updated_at = NOW()
                WHERE symbol = %s
                """,
                (json.dumps(config_json), symbol)
            )
            updated = cursor.rowcount > 0
            self.connection.commit()
            cursor.close()
            return updated
        except Exception as e:
            logger.error(f"Failed to update factor config for {symbol}: {e}")
            self.connection.rollback()
            return False
    
    def delete_factor_config(self, symbol: str) -> bool:
        """Delete factor configuration for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "DELETE FROM factor_configs WHERE symbol = %s",
                (symbol,)
            )
            deleted = cursor.rowcount > 0
            self.connection.commit()
            cursor.close()
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete factor config for {symbol}: {e}")
            self.connection.rollback()
            return False
    
    def list_factor_config_symbols(self) -> List[str]:
        """List all symbols with custom factor configurations.
        
        Returns:
            List of symbols
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT symbol FROM factor_configs ORDER BY symbol"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Failed to list factor config symbols: {e}")
            return []
    
    # ==================== Tuning Results Operations ====================
    # Requirements: 9.4
    
    def insert_tuning_result(
        self,
        symbol: str,
        best_config: Dict[str, Any],
        win_rate: float,
        total_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
        total_trades: int,
        backtest_start: date,
        backtest_end: date
    ) -> Optional[int]:
        """Insert a tuning result.
        
        Args:
            symbol: Stock symbol
            best_config: Best configuration as JSON-serializable dict
            win_rate: Win rate (0-1)
            total_return: Total return percentage
            max_drawdown: Maximum drawdown percentage
            sharpe_ratio: Sharpe ratio
            total_trades: Total number of trades
            backtest_start: Start date of backtest period
            backtest_end: End date of backtest period
            
        Returns:
            ID of inserted record or None if failed
        """
        try:
            import json
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO tuning_results 
                (symbol, best_config, win_rate, total_return, max_drawdown,
                 sharpe_ratio, total_trades, backtest_start, backtest_end)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (symbol, json.dumps(best_config), win_rate, total_return,
                 max_drawdown, sharpe_ratio, total_trades, backtest_start, backtest_end)
            )
            result_id = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            return result_id
        except Exception as e:
            logger.error(f"Failed to insert tuning result for {symbol}: {e}")
            self.connection.rollback()
            return None
    
    def get_tuning_results_by_symbol(
        self,
        symbol: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get tuning results for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of records to return
            
        Returns:
            List of tuning result dictionaries
        """
        try:
            import json
            cursor = self.connection.cursor()
            
            query = """
                SELECT id, best_config, win_rate, total_return, max_drawdown,
                       sharpe_ratio, total_trades, backtest_start, backtest_end, created_at
                FROM tuning_results
                WHERE symbol = %s
                ORDER BY created_at DESC
            """
            params = [symbol]
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    'id': row[0],
                    'best_config': row[1] if isinstance(row[1], dict) else json.loads(row[1]) if row[1] else {},
                    'win_rate': float(row[2]) if row[2] else 0.0,
                    'total_return': float(row[3]) if row[3] else 0.0,
                    'max_drawdown': float(row[4]) if row[4] else 0.0,
                    'sharpe_ratio': float(row[5]) if row[5] else 0.0,
                    'total_trades': row[6] or 0,
                    'backtest_start': row[7],
                    'backtest_end': row[8],
                    'created_at': row[9]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get tuning results for {symbol}: {e}")
            return []
    
    def get_latest_tuning_result(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent tuning result for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuning result dict or None if not found
        """
        results = self.get_tuning_results_by_symbol(symbol, limit=1)
        return results[0] if results else None
    
    # ==================== Indicator Correlations Operations ====================
    # Requirements: 13.1, 13.2
    
    def upsert_indicator_correlation(
        self,
        symbol: str,
        indicator_name: str,
        correlation: float,
        sample_size: int
    ) -> bool:
        """Insert or update an indicator correlation.
        
        Args:
            symbol: Stock symbol
            indicator_name: Name of the indicator (rsi, volume, macd, ema, bollinger)
            correlation: Correlation coefficient (-1 to 1)
            sample_size: Number of samples used for calculation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO indicator_correlations 
                (symbol, indicator_name, correlation, sample_size, calculated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (symbol, indicator_name) DO UPDATE SET
                    correlation = EXCLUDED.correlation,
                    sample_size = EXCLUDED.sample_size,
                    calculated_at = NOW()
                """,
                (symbol, indicator_name, correlation, sample_size)
            )
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to upsert indicator correlation: {e}")
            self.connection.rollback()
            return False
    
    def get_indicator_correlations(self, symbol: str) -> Dict[str, float]:
        """Get all indicator correlations for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict mapping indicator name to correlation coefficient
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT indicator_name, correlation
                FROM indicator_correlations
                WHERE symbol = %s
                """,
                (symbol,)
            )
            rows = cursor.fetchall()
            cursor.close()
            
            return {row[0]: float(row[1]) if row[1] else 0.0 for row in rows}
        except Exception as e:
            logger.error(f"Failed to get indicator correlations for {symbol}: {e}")
            return {}
    
    def save_indicator_correlations_batch(
        self,
        symbol: str,
        correlations: Dict[str, float],
        sample_size: int
    ) -> bool:
        """Save multiple indicator correlations at once.
        
        Args:
            symbol: Stock symbol
            correlations: Dict mapping indicator name to correlation coefficient
            sample_size: Number of samples used for calculation
            
        Returns:
            True if all successful, False otherwise
        """
        try:
            for indicator_name, correlation in correlations.items():
                if not self.upsert_indicator_correlation(
                    symbol, indicator_name, correlation, sample_size
                ):
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to save indicator correlations batch: {e}")
            return False
