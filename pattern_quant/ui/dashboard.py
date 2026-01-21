"""Dashboard UI for AI PatternQuant

This module provides the main dashboard interface using Streamlit.
Displays core metrics (P&L, market value, cash, exposure) and real-time signal flow.
Includes strategy status indicators for dual-engine mode.

Requirements: 11.1, 11.2, 11.3, 10.1, 10.2, 10.3, 10.4
"""

import streamlit as st
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from enum import Enum

from pattern_quant.core.models import (
    TradeSignal,
    Position,
    SignalStatus,
)
from pattern_quant.strategy.models import MarketState


@dataclass
class PortfolioMetrics:
    """Portfolio core metrics for dashboard display.
    
    Attributes:
        daily_pnl: 當日損益
        total_market_value: 持倉總市值
        available_cash: 可用現金
        exposure_ratio: 曝險比例 (0-1)
        total_capital: 總資金
    """
    daily_pnl: float
    total_market_value: float
    available_cash: float
    exposure_ratio: float
    total_capital: float


@dataclass
class StrategyStatus:
    """Strategy status for a symbol.
    
    Attributes:
        symbol: 股票代碼
        market_state: 市場狀態 (TREND, RANGE, NOISE)
        adx_value: ADX 值
        bbw_value: BBW 值
        allocation_weight: 資金權重
        active_strategy: 當前執行的策略類型
    """
    symbol: str
    market_state: MarketState
    adx_value: float
    bbw_value: float
    allocation_weight: float
    active_strategy: str  # "trend", "mean_reversion", "none"


class DataProvider(Protocol):
    """Protocol for dashboard data providers.
    
    Allows dependency injection for testing and different data sources.
    """
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get current portfolio metrics."""
        ...
    
    def get_active_signals(self, limit: int = 20) -> List[TradeSignal]:
        """Get active trade signals."""
        ...
    
    def get_open_positions(self) -> List[Position]:
        """Get open positions."""
        ...
    
    def get_strategy_statuses(self) -> List[StrategyStatus]:
        """Get strategy status for all monitored symbols."""
        ...


class MockDataProvider:
    """Mock data provider for demo/testing purposes."""
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        return PortfolioMetrics(
            daily_pnl=0.0,
            total_market_value=0.0,
            available_cash=1000000.0,
            exposure_ratio=0.0,
            total_capital=1000000.0
        )
    
    def get_active_signals(self, limit: int = 20) -> List[TradeSignal]:
        return []
    
    def get_open_positions(self) -> List[Position]:
        return []
    
    def get_strategy_statuses(self) -> List[StrategyStatus]:
        """Return mock strategy statuses for demo."""
        return []



class Dashboard:
    """Main dashboard UI component.
    
    Provides the primary interface for monitoring the trading system,
    including core metrics and real-time signal flow.
    
    Attributes:
        data_provider: Data source for dashboard metrics and signals
        page_title: Title displayed in browser tab
        refresh_interval: Auto-refresh interval in seconds (0 = disabled)
    """
    
    def __init__(
        self,
        data_provider: Optional[DataProvider] = None,
        page_title: str = "AI PatternQuant Dashboard",
        refresh_interval: int = 30
    ):
        """Initialize the dashboard.
        
        Args:
            data_provider: Data provider instance (uses MockDataProvider if None)
            page_title: Browser tab title
            refresh_interval: Auto-refresh interval in seconds
        """
        self.data_provider = data_provider or MockDataProvider()
        self.page_title = page_title
        self.refresh_interval = refresh_interval
    
    def configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.page_title,
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_header(self) -> None:
        """Render the dashboard header."""
        st.title("📈 AI PatternQuant")
        st.markdown("幾何特徵量化交易系統")
        st.divider()
    
    def render_core_metrics(self, metrics: PortfolioMetrics) -> None:
        """Render the core metrics section.
        
        Displays: 當日損益、持倉總市值、可用現金、曝險比例
        
        Args:
            metrics: Portfolio metrics to display
        """
        st.subheader("📊 核心數據")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pnl_color = "green" if metrics.daily_pnl >= 0 else "red"
            pnl_sign = "+" if metrics.daily_pnl >= 0 else ""
            st.metric(
                label="當日損益",
                value=f"${metrics.daily_pnl:,.2f}",
                delta=f"{pnl_sign}{metrics.daily_pnl / metrics.total_capital * 100:.2f}%" if metrics.total_capital > 0 else "0%"
            )
        
        with col2:
            st.metric(
                label="持倉總市值",
                value=f"${metrics.total_market_value:,.2f}"
            )
        
        with col3:
            st.metric(
                label="可用現金",
                value=f"${metrics.available_cash:,.2f}"
            )
        
        with col4:
            exposure_pct = metrics.exposure_ratio * 100
            st.metric(
                label="曝險比例",
                value=f"{exposure_pct:.1f}%"
            )
        
        st.divider()

    def render_signal_flow(self, signals: List[TradeSignal]) -> None:
        """Render the real-time signal flow section.
        
        Displays: 股票代碼、型態名稱、吻合度、預期獲利比、狀態
        Implements real-time status updates via session state tracking.
        
        Args:
            signals: List of trade signals to display
        
        Requirements: 11.2, 11.3
        """
        st.subheader("🔔 即時訊號流")
        
        # Initialize session state for tracking signal changes
        if 'previous_signals' not in st.session_state:
            st.session_state.previous_signals = {}
        
        if not signals:
            st.info("目前沒有活躍訊號")
            return
        
        # Track status changes for notifications
        current_signals = {s.symbol: s.status for s in signals}
        status_changes = []
        
        for symbol, status in current_signals.items():
            prev_status = st.session_state.previous_signals.get(symbol)
            if prev_status and prev_status != status:
                status_changes.append((symbol, prev_status, status))
        
        # Update session state
        st.session_state.previous_signals = current_signals
        
        # Show status change notifications
        if status_changes:
            for symbol, old_status, new_status in status_changes:
                st.toast(
                    f"📢 {symbol}: {self._get_status_text(old_status)} → {self._get_status_text(new_status)}",
                    icon="🔔"
                )
        
        # Signal count by status
        status_counts = {}
        for signal in signals:
            status_text = self._get_status_text(signal.status)
            status_counts[status_text] = status_counts.get(status_text, 0) + 1
        
        # Display status summary
        status_cols = st.columns(len(status_counts) if status_counts else 1)
        for i, (status_text, count) in enumerate(status_counts.items()):
            with status_cols[i]:
                st.metric(label=status_text, value=count)
        
        st.divider()
        
        # Create signal data for display
        signal_data = []
        for signal in signals:
            status_emoji = self._get_status_emoji(signal.status)
            signal_data.append({
                "代碼": signal.symbol,
                "型態": signal.pattern_type,
                "吻合度": f"{signal.match_score:.1f}%",
                "預期獲利比": f"{signal.expected_profit_ratio:.2f}",
                "突破價": f"${signal.breakout_price:.2f}",
                "止損價": f"${signal.stop_loss_price:.2f}",
                "狀態": f"{status_emoji} {self._get_status_text(signal.status)}",
                "建立時間": signal.created_at.strftime("%Y-%m-%d %H:%M")
            })
        
        st.dataframe(
            signal_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "吻合度": st.column_config.ProgressColumn(
                    "吻合度",
                    help="型態吻合分數",
                    format="%s",
                    min_value=0,
                    max_value=100,
                ),
            }
        )
    
    def _get_status_emoji(self, status: SignalStatus) -> str:
        """Get emoji for signal status."""
        status_emojis = {
            SignalStatus.WAITING_BREAKOUT: "⏳",
            SignalStatus.TRIGGERED: "🎯",
            SignalStatus.EXECUTED: "✅",
            SignalStatus.CANCELLED: "❌"
        }
        return status_emojis.get(status, "❓")
    
    def _get_status_text(self, status: SignalStatus) -> str:
        """Get display text for signal status."""
        status_texts = {
            SignalStatus.WAITING_BREAKOUT: "等待突破",
            SignalStatus.TRIGGERED: "已觸發",
            SignalStatus.EXECUTED: "已執行",
            SignalStatus.CANCELLED: "已取消"
        }
        return status_texts.get(status, "未知")
    
    def render_positions_summary(self, positions: List[Position]) -> None:
        """Render a summary of open positions.
        
        Args:
            positions: List of open positions
        """
        st.subheader("💼 持倉概覽")
        
        if not positions:
            st.info("目前沒有持倉")
            return
        
        position_data = []
        for pos in positions:
            pnl = (pos.current_price - pos.entry_price) * pos.quantity
            pnl_pct = ((pos.current_price / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else 0
            
            position_data.append({
                "代碼": pos.symbol,
                "數量": pos.quantity,
                "進場價": f"${pos.entry_price:.2f}",
                "現價": f"${pos.current_price:.2f}",
                "損益": f"${pnl:,.2f}",
                "損益%": f"{pnl_pct:+.2f}%",
                "板塊": pos.sector,
                "移動止盈": "✅" if pos.trailing_stop_active else "❌"
            })
        
        st.dataframe(
            position_data,
            use_container_width=True,
            hide_index=True
        )

    def render_strategy_status_indicators(self, statuses: List[StrategyStatus]) -> None:
        """Render strategy status indicators for each symbol.
        
        Displays market state indicators:
        - 🟢 TREND MODE: 正在尋找突破
        - 🔵 RANGE MODE: 正在高拋低吸
        - ⚪ NEUTRAL: 觀望
        
        Args:
            statuses: List of strategy statuses for monitored symbols
            
        Requirements: 10.1, 10.2, 10.3, 10.4
        """
        st.subheader("🎯 策略狀態指示燈")
        
        if not statuses:
            st.info("目前沒有監控中的標的")
            return
        
        # 狀態統計
        state_counts = {
            MarketState.TREND: 0,
            MarketState.RANGE: 0,
            MarketState.NOISE: 0
        }
        for status in statuses:
            state_counts[status.market_state] = state_counts.get(status.market_state, 0) + 1
        
        # 顯示狀態統計
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 趨勢模式", state_counts[MarketState.TREND])
        with col2:
            st.metric("🔵 震盪模式", state_counts[MarketState.RANGE])
        with col3:
            st.metric("⚪ 觀望模式", state_counts[MarketState.NOISE])
        
        st.divider()
        
        # 建立狀態表格
        status_data = []
        for status in statuses:
            indicator = self._get_market_state_indicator(status.market_state)
            mode_text = self._get_market_state_text(status.market_state)
            
            status_data.append({
                "代碼": status.symbol,
                "狀態": f"{indicator} {mode_text}",
                "ADX": f"{status.adx_value:.1f}",
                "BBW": f"{status.bbw_value:.4f}",
                "資金權重": f"{status.allocation_weight * 100:.0f}%",
                "執行策略": self._get_strategy_text(status.active_strategy)
            })
        
        st.dataframe(
            status_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ADX": st.column_config.ProgressColumn(
                    "ADX",
                    help="平均趨向指標 (0-100)",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "資金權重": st.column_config.ProgressColumn(
                    "資金權重",
                    help="當前狀態的資金使用比例",
                    format="%s",
                    min_value=0,
                    max_value=100,
                ),
            }
        )
    
    def _get_market_state_indicator(self, state: MarketState) -> str:
        """Get emoji indicator for market state.
        
        Requirements: 10.1, 10.2, 10.3
        """
        indicators = {
            MarketState.TREND: "🟢",
            MarketState.RANGE: "🔵",
            MarketState.NOISE: "⚪"
        }
        return indicators.get(state, "❓")
    
    def _get_market_state_text(self, state: MarketState) -> str:
        """Get display text for market state.
        
        Requirements: 10.1, 10.2, 10.3
        """
        texts = {
            MarketState.TREND: "TREND MODE（正在尋找突破）",
            MarketState.RANGE: "RANGE MODE（正在高拋低吸）",
            MarketState.NOISE: "NEUTRAL（觀望）"
        }
        return texts.get(state, "未知狀態")
    
    def _get_strategy_text(self, strategy: str) -> str:
        """Get display text for active strategy."""
        strategy_texts = {
            "trend": "📈 趨勢突破",
            "mean_reversion": "📊 均值回歸",
            "none": "⏸️ 暫停"
        }
        return strategy_texts.get(strategy, "❓ 未知")

    def render_sidebar(self) -> None:
        """Render the sidebar with controls and info."""
        with st.sidebar:
            st.header("⚙️ 控制面板")
            
            # Refresh button
            if st.button("🔄 重新整理", use_container_width=True):
                st.rerun()
            
            st.divider()
            
            # Auto-refresh settings
            st.subheader("⏱️ 自動更新設定")
            
            # Initialize session state for auto-refresh
            if 'auto_refresh_enabled' not in st.session_state:
                st.session_state.auto_refresh_enabled = self.refresh_interval > 0
            
            auto_refresh = st.checkbox(
                "啟用自動更新",
                value=st.session_state.auto_refresh_enabled,
                help="自動定期更新儀表板數據"
            )
            st.session_state.auto_refresh_enabled = auto_refresh
            
            if auto_refresh:
                refresh_seconds = st.slider(
                    "更新間隔（秒）",
                    min_value=5,
                    max_value=120,
                    value=self.refresh_interval,
                    step=5
                )
                st.info(f"每 {refresh_seconds} 秒自動更新")
                # Store for use in render()
                st.session_state.refresh_seconds = refresh_seconds
            
            st.divider()
            
            # System status
            st.subheader("📡 系統狀態")
            st.success("系統運行中")
            st.caption(f"最後更新: {datetime.now().strftime('%H:%M:%S')}")
            
            # Signal filter options
            st.divider()
            st.subheader("🔍 訊號篩選")
            
            if 'signal_filter' not in st.session_state:
                st.session_state.signal_filter = "全部"
            
            filter_options = ["全部", "等待突破", "已觸發", "已執行", "已取消"]
            st.session_state.signal_filter = st.selectbox(
                "狀態篩選",
                options=filter_options,
                index=filter_options.index(st.session_state.signal_filter)
            )
    
    def render(self) -> None:
        """Render the complete dashboard.
        
        Main entry point for displaying the dashboard.
        Fetches data and renders all components.
        Implements real-time updates via auto-refresh.
        
        Requirements: 11.1, 11.2, 11.3, 10.1, 10.2, 10.3, 10.4
        """
        self.configure_page()
        self.render_header()
        self.render_sidebar()
        
        # Fetch data
        metrics = self.data_provider.get_portfolio_metrics()
        signals = self.data_provider.get_active_signals()
        positions = self.data_provider.get_open_positions()
        
        # Fetch strategy statuses if available
        strategy_statuses = []
        if hasattr(self.data_provider, 'get_strategy_statuses'):
            strategy_statuses = self.data_provider.get_strategy_statuses()
        
        # Apply signal filter if set
        if hasattr(st.session_state, 'signal_filter') and st.session_state.signal_filter != "全部":
            filter_map = {
                "等待突破": SignalStatus.WAITING_BREAKOUT,
                "已觸發": SignalStatus.TRIGGERED,
                "已執行": SignalStatus.EXECUTED,
                "已取消": SignalStatus.CANCELLED
            }
            target_status = filter_map.get(st.session_state.signal_filter)
            if target_status:
                signals = [s for s in signals if s.status == target_status]
        
        # Render main content
        self.render_core_metrics(metrics)
        
        # Strategy status indicators (Requirements 10.1, 10.2, 10.3, 10.4)
        if strategy_statuses:
            self.render_strategy_status_indicators(strategy_statuses)
            st.divider()
        
        # Two-column layout for signals and positions
        col1, col2 = st.columns([3, 2])
        
        with col1:
            self.render_signal_flow(signals)
        
        with col2:
            self.render_positions_summary(positions)
        
        # Auto-refresh using session state settings
        auto_refresh_enabled = getattr(st.session_state, 'auto_refresh_enabled', False)
        refresh_seconds = getattr(st.session_state, 'refresh_seconds', self.refresh_interval)
        
        if auto_refresh_enabled and refresh_seconds > 0:
            import time
            time.sleep(refresh_seconds)
            st.rerun()


def run_dashboard(data_provider: Optional[DataProvider] = None) -> None:
    """Run the dashboard application.
    
    Convenience function to start the dashboard.
    
    Args:
        data_provider: Optional data provider instance
    """
    dashboard = Dashboard(data_provider=data_provider)
    dashboard.render()


# Entry point for running with: streamlit run pattern_quant/ui/dashboard.py
if __name__ == "__main__":
    run_dashboard()
