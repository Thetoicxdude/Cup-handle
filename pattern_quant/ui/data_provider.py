"""Data Provider for Dashboard UI

This module provides the database-backed data provider for the dashboard.
Connects to the repository layer to fetch real data.

Requirements: 11.1, 11.2, 11.3
"""

from typing import List, Optional
from datetime import datetime

from pattern_quant.core.models import (
    TradeSignal,
    Position,
    SignalStatus,
)
from pattern_quant.ui.dashboard import PortfolioMetrics, DataProvider
from pattern_quant.db.repository import DatabaseRepository


class RepositoryDataProvider:
    """Database-backed data provider for the dashboard.
    
    Fetches real data from the database repository.
    
    Attributes:
        repository: Database repository instance
        total_capital: Total capital for exposure calculations
    """
    
    def __init__(
        self,
        repository: DatabaseRepository,
        total_capital: float = 1000000.0
    ):
        """Initialize the data provider.
        
        Args:
            repository: Database repository instance
            total_capital: Total capital amount
        """
        self.repository = repository
        self.total_capital = total_capital
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get current portfolio metrics from database.
        
        Calculates:
        - Daily P&L from open positions
        - Total market value of positions
        - Available cash
        - Exposure ratio
        
        Returns:
            PortfolioMetrics with current values
        """
        positions_data = self.repository.get_open_positions()
        
        total_market_value = 0.0
        daily_pnl = 0.0
        
        for pos_dict in positions_data:
            pos = pos_dict['position']
            market_value = pos.current_price * pos.quantity
            total_market_value += market_value
            
            # Calculate P&L
            entry_value = pos.entry_price * pos.quantity
            daily_pnl += (market_value - entry_value)
        
        available_cash = self.total_capital - total_market_value
        exposure_ratio = total_market_value / self.total_capital if self.total_capital > 0 else 0.0
        
        return PortfolioMetrics(
            daily_pnl=daily_pnl,
            total_market_value=total_market_value,
            available_cash=available_cash,
            exposure_ratio=exposure_ratio,
            total_capital=self.total_capital
        )
    
    def get_active_signals(self, limit: int = 20) -> List[TradeSignal]:
        """Get active trade signals from database.
        
        Fetches signals with status WAITING_BREAKOUT or TRIGGERED.
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            List of active TradeSignal objects
        """
        signals = []
        
        # Get waiting breakout signals
        waiting_signals = self.repository.get_trade_signals_by_status(
            SignalStatus.WAITING_BREAKOUT,
            limit=limit
        )
        signals.extend(waiting_signals)
        
        # Get triggered signals
        triggered_signals = self.repository.get_trade_signals_by_status(
            SignalStatus.TRIGGERED,
            limit=limit
        )
        signals.extend(triggered_signals)
        
        # Sort by created_at descending and limit
        signals.sort(key=lambda s: s.created_at, reverse=True)
        return signals[:limit]
    
    def get_open_positions(self) -> List[Position]:
        """Get open positions from database.
        
        Returns:
            List of open Position objects
        """
        positions_data = self.repository.get_open_positions()
        return [pos_dict['position'] for pos_dict in positions_data]
