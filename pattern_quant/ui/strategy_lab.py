"""Strategy Lab UI for AI PatternQuant

This module provides the Strategy Lab interface using Streamlit.
Allows users to adjust model parameters and run backtests.

Requirements: 12.1, 12.2, 12.3
"""

import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Protocol
from enum import Enum

import numpy as np


@dataclass
class StrategyParameters:
    """策略參數配置
    
    Attributes:
        min_depth: 最小杯身深度 (%)
        max_depth: 最大杯身深度 (%)
        min_cup_days: 最小成型天數
        max_cup_days: 最大成型天數
        stop_loss_ratio: 止損比例 (%)
        profit_threshold: 移動止盈啟動閾值 (%)
        trailing_ratio: 移動止盈回調比例 (%)
        score_threshold: 吻合分數閾值
    """
    min_depth: float = 14.0
    max_depth: float = 28.0
    min_cup_days: int = 20
    max_cup_days: int = 220
    stop_loss_ratio: float = 5.0
    profit_threshold: float = 12.0
    trailing_ratio: float = 9.0
    score_threshold: float = 65.0


@dataclass
class BacktestTrade:
    """回測交易記錄
    
    Attributes:
        symbol: 股票代碼
        entry_date: 進場日期
        entry_price: 進場價格
        exit_date: 出場日期
        exit_price: 出場價格
        exit_reason: 出場原因
        pnl: 損益金額
        pnl_pct: 損益百分比
        holding_days: 持有天數
    """
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    exit_reason: str
    pnl: float
    pnl_pct: float
    holding_days: int


@dataclass
class BacktestResult:
    """回測結果
    
    Attributes:
        parameters: 使用的策略參數
        start_date: 回測起始日期
        end_date: 回測結束日期
        total_trades: 總交易次數
        winning_trades: 獲利交易次數
        losing_trades: 虧損交易次數
        win_rate: 勝率 (%)
        total_return: 總報酬率 (%)
        max_drawdown: 最大回撤 (%)
        sharpe_ratio: 夏普比率
        equity_curve: 資金曲線數據
        trades: 交易記錄列表
    """
    parameters: StrategyParameters
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: List[Dict[str, Any]]
    trades: List[BacktestTrade] = field(default_factory=list)


class BacktestDataProvider(Protocol):
    """Protocol for backtest data providers."""
    
    def get_historical_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get historical OHLCV data for a symbol."""
        ...
    
    def get_watchlist_symbols(self) -> List[str]:
        """Get list of symbols to backtest."""
        ...


class MockBacktestDataProvider:
    """Mock data provider for demo/testing purposes."""
    
    def get_historical_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate mock historical price data."""
        return []
    
    def get_watchlist_symbols(self) -> List[str]:
        """Return mock watchlist."""
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]



class BacktestEngine:
    """回測引擎
    
    執行策略回測，模擬歷史交易並計算績效指標。
    
    Attributes:
        initial_capital: 初始資金
        data_provider: 數據提供者
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        data_provider: Optional[BacktestDataProvider] = None
    ):
        """初始化回測引擎
        
        Args:
            initial_capital: 初始資金
            data_provider: 數據提供者
        """
        self.initial_capital = initial_capital
        self.data_provider = data_provider or MockBacktestDataProvider()
    
    def run_backtest(
        self,
        parameters: StrategyParameters,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None
    ) -> BacktestResult:
        """執行回測
        
        Args:
            parameters: 策略參數
            start_date: 回測起始日期
            end_date: 回測結束日期
            symbols: 回測股票列表（可選）
            
        Returns:
            BacktestResult 包含回測結果
        """
        if symbols is None:
            symbols = self.data_provider.get_watchlist_symbols()
        
        # Initialize tracking variables
        trades: List[BacktestTrade] = []
        equity_curve: List[Dict[str, Any]] = []
        current_capital = self.initial_capital
        peak_capital = self.initial_capital
        max_drawdown = 0.0
        daily_returns: List[float] = []
        
        # Generate simulated equity curve for demo
        # In production, this would use actual pattern detection and trading logic
        num_days = (end_date - start_date).days
        
        if num_days <= 0:
            return BacktestResult(
                parameters=parameters,
                start_date=start_date,
                end_date=end_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                equity_curve=[],
                trades=[]
            )
        
        # Simulate trading based on parameters
        # More conservative parameters = fewer but higher quality trades
        trade_frequency = self._calculate_trade_frequency(parameters)
        avg_win_rate = self._calculate_expected_win_rate(parameters)
        
        prev_capital = current_capital
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Simulate daily P&L with some randomness
            daily_return = np.random.normal(0.0005, 0.015)  # ~0.05% mean, 1.5% std
            current_capital *= (1 + daily_return)
            
            # Track daily return
            if prev_capital > 0:
                daily_returns.append((current_capital - prev_capital) / prev_capital)
            prev_capital = current_capital
            
            # Update peak and drawdown
            if current_capital > peak_capital:
                peak_capital = current_capital
            
            drawdown = (peak_capital - current_capital) / peak_capital
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            # Record equity curve point
            equity_curve.append({
                "date": current_date.isoformat(),
                "equity": current_capital,
                "drawdown": drawdown * 100
            })
            
            # Simulate trades based on frequency
            if np.random.random() < trade_frequency:
                trade = self._simulate_trade(
                    symbols=symbols,
                    entry_date=current_date,
                    parameters=parameters,
                    win_rate=avg_win_rate
                )
                if trade:
                    trades.append(trade)
        
        # Calculate final metrics
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl <= 0])
        total_trades = len(trades)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        total_return = ((current_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate Sharpe ratio (annualized)
        if len(daily_returns) > 1:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return BacktestResult(
            parameters=parameters,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown * 100,
            sharpe_ratio=sharpe_ratio,
            equity_curve=equity_curve,
            trades=trades
        )
    
    def _calculate_trade_frequency(self, parameters: StrategyParameters) -> float:
        """Calculate expected trade frequency based on parameters."""
        # Higher score threshold = fewer trades
        # Narrower depth range = fewer trades
        base_frequency = 0.02  # ~2% chance per day
        
        score_factor = 1.0 - (parameters.score_threshold - 70) / 30  # 70-100 range
        depth_range = parameters.max_depth - parameters.min_depth
        depth_factor = depth_range / 21  # 21% is default range
        
        return base_frequency * max(0.1, score_factor) * max(0.5, depth_factor)
    
    def _calculate_expected_win_rate(self, parameters: StrategyParameters) -> float:
        """Calculate expected win rate based on parameters."""
        # Higher score threshold = higher win rate
        # Tighter stop loss = lower win rate but smaller losses
        base_win_rate = 0.55
        
        score_bonus = (parameters.score_threshold - 70) / 100  # 0-0.3 bonus
        stop_loss_penalty = (parameters.stop_loss_ratio - 5) / 100  # Penalty for tight stops
        
        return min(0.75, max(0.35, base_win_rate + score_bonus - stop_loss_penalty))
    
    def _simulate_trade(
        self,
        symbols: List[str],
        entry_date: datetime,
        parameters: StrategyParameters,
        win_rate: float
    ) -> Optional[BacktestTrade]:
        """Simulate a single trade."""
        if not symbols:
            return None
        
        symbol = np.random.choice(symbols)
        entry_price = np.random.uniform(50, 500)
        
        # Determine if trade is a winner
        is_winner = np.random.random() < win_rate
        
        if is_winner:
            # Winner: exit at profit threshold or trailing stop
            exit_pct = np.random.uniform(
                parameters.profit_threshold / 100,
                parameters.profit_threshold / 100 * 2
            )
            exit_reason = "trailing_stop" if exit_pct > parameters.profit_threshold / 100 * 1.5 else "target"
        else:
            # Loser: exit at stop loss
            exit_pct = -parameters.stop_loss_ratio / 100
            exit_reason = "stop_loss"
        
        exit_price = entry_price * (1 + exit_pct)
        holding_days = np.random.randint(5, parameters.max_cup_days // 2)
        exit_date = entry_date + timedelta(days=holding_days)
        
        pnl = (exit_price - entry_price) * 100  # Assume 100 shares
        pnl_pct = exit_pct * 100
        
        return BacktestTrade(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days
        )


class StrategyLab:
    """策略實驗室 UI 組件
    
    提供參數調整介面與回測功能。
    
    Attributes:
        backtest_engine: 回測引擎
        default_params: 預設參數
    """
    
    def __init__(
        self,
        backtest_engine: Optional[BacktestEngine] = None,
        default_params: Optional[StrategyParameters] = None
    ):
        """初始化策略實驗室
        
        Args:
            backtest_engine: 回測引擎實例
            default_params: 預設策略參數
        """
        self.backtest_engine = backtest_engine or BacktestEngine()
        self.default_params = default_params or StrategyParameters()
    
    def render_parameter_sliders(self) -> StrategyParameters:
        """渲染參數調整滑桿
        
        顯示所有可調整的策略參數滑桿，包括：
        - 杯身深度範圍
        - 成型天數範圍
        - 止損比例
        - 移動止盈參數
        - 吻合分數閾值
        
        Returns:
            當前選擇的策略參數
            
        Requirements: 12.1
        """
        st.subheader("🎛️ 參數調整")
        
        # Initialize session state for parameters if not exists
        if 'strategy_params' not in st.session_state:
            st.session_state.strategy_params = self.default_params
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**杯身深度設定**")
            
            min_depth = st.slider(
                "最小杯身深度 (%)",
                min_value=5.0,
                max_value=25.0,
                value=self.default_params.min_depth,
                step=1.0,
                help="杯身深度下限，低於此值的型態將被過濾"
            )
            
            max_depth = st.slider(
                "最大杯身深度 (%)",
                min_value=20.0,
                max_value=50.0,
                value=self.default_params.max_depth,
                step=1.0,
                help="杯身深度上限，高於此值的型態將被過濾"
            )
            
            st.markdown("**成型天數設定**")
            
            min_cup_days = st.slider(
                "最小成型天數",
                min_value=10,
                max_value=60,
                value=self.default_params.min_cup_days,
                step=5,
                help="茶杯型態最少需要的形成天數"
            )
            
            max_cup_days = st.slider(
                "最大成型天數",
                min_value=60,
                max_value=300,
                value=self.default_params.max_cup_days,
                step=10,
                help="茶杯型態最多允許的形成天數"
            )
        
        with col2:
            st.markdown("**風控參數設定**")
            
            stop_loss_ratio = st.slider(
                "止損比例 (%)",
                min_value=0.0,
                max_value=100.0,
                value=self.default_params.stop_loss_ratio,
                step=1.0,
                help="硬止損觸發比例，價格下跌超過此比例將強制出場"
            )
            
            profit_threshold = st.slider(
                "移動止盈啟動閾值 (%)",
                min_value=0.0,
                max_value=100.0,
                value=self.default_params.profit_threshold,
                step=1.0,
                help="獲利達到此比例時啟動移動止盈機制"
            )
            
            trailing_ratio = st.slider(
                "移動止盈回調比例 (%)",
                min_value=1.0,
                max_value=10.0,
                value=self.default_params.trailing_ratio,
                step=0.5,
                help="移動止盈啟動後，價格回調超過此比例將觸發出場"
            )
            
            st.markdown("**訊號過濾設定**")
            
            score_threshold = st.slider(
                "吻合分數閾值",
                min_value=60.0,
                max_value=95.0,
                value=self.default_params.score_threshold,
                step=1.0,
                help="型態吻合分數需達到此閾值才會產生訊號"
            )
        
        # Validate parameters
        if min_depth >= max_depth:
            st.warning("⚠️ 最小杯身深度必須小於最大杯身深度")
        
        if min_cup_days >= max_cup_days:
            st.warning("⚠️ 最小成型天數必須小於最大成型天數")
        
        # Create parameters object
        params = StrategyParameters(
            min_depth=min_depth,
            max_depth=max_depth,
            min_cup_days=min_cup_days,
            max_cup_days=max_cup_days,
            stop_loss_ratio=stop_loss_ratio,
            profit_threshold=profit_threshold,
            trailing_ratio=trailing_ratio,
            score_threshold=score_threshold
        )
        
        # Store in session state
        st.session_state.strategy_params = params
        
        return params
    
    def render_backtest_controls(self) -> Optional[tuple]:
        """渲染回測控制區
        
        Returns:
            (start_date, end_date) 如果用戶點擊回測按鈕，否則 None
        """
        st.subheader("📅 回測設定")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Default to 3 years ago
            default_start = date.today() - timedelta(days=3*365)
            start_date = st.date_input(
                "起始日期",
                value=default_start,
                help="回測起始日期"
            )
        
        with col2:
            end_date = st.date_input(
                "結束日期",
                value=date.today(),
                help="回測結束日期"
            )
        
        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            run_backtest = st.button(
                "🚀 執行回測",
                use_container_width=True,
                type="primary"
            )
        
        if run_backtest:
            if start_date >= end_date:
                st.error("❌ 起始日期必須早於結束日期")
                return None
            return (
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.min.time())
            )
        
        return None
    
    def render_backtest_results(self, result: BacktestResult) -> None:
        """渲染回測結果
        
        顯示勝率、總報酬率與資金曲線圖。
        
        Args:
            result: 回測結果
            
        Requirements: 12.2, 12.3
        """
        st.subheader("📊 回測結果")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="勝率",
                value=f"{result.win_rate:.1f}%",
                delta=f"{result.winning_trades}勝 / {result.losing_trades}負"
            )
        
        with col2:
            return_color = "normal" if result.total_return >= 0 else "inverse"
            st.metric(
                label="總報酬率",
                value=f"{result.total_return:+.2f}%",
                delta=f"共 {result.total_trades} 筆交易"
            )
        
        with col3:
            st.metric(
                label="最大回撤",
                value=f"{result.max_drawdown:.2f}%"
            )
        
        with col4:
            st.metric(
                label="夏普比率",
                value=f"{result.sharpe_ratio:.2f}"
            )
        
        st.divider()
        
        # Equity curve chart
        if result.equity_curve:
            st.markdown("**📈 資金曲線**")
            
            # Prepare data for chart
            chart_data = {
                "日期": [item["date"] for item in result.equity_curve],
                "資金": [item["equity"] for item in result.equity_curve],
                "回撤 (%)": [item["drawdown"] for item in result.equity_curve]
            }
            
            # Use Streamlit's built-in line chart
            import pandas as pd
            df = pd.DataFrame(chart_data)
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.set_index("日期")
            
            # Display equity curve
            st.line_chart(df["資金"], use_container_width=True)
            
            # Display drawdown chart
            with st.expander("📉 回撤曲線"):
                st.area_chart(df["回撤 (%)"], use_container_width=True)
        
        # Trade details
        if result.trades:
            with st.expander(f"📋 交易明細 ({len(result.trades)} 筆)"):
                trade_data = []
                for trade in result.trades:
                    trade_data.append({
                        "代碼": trade.symbol,
                        "進場日期": trade.entry_date.strftime("%Y-%m-%d"),
                        "進場價": f"${trade.entry_price:.2f}",
                        "出場日期": trade.exit_date.strftime("%Y-%m-%d"),
                        "出場價": f"${trade.exit_price:.2f}",
                        "出場原因": trade.exit_reason,
                        "損益": f"${trade.pnl:+,.2f}",
                        "損益%": f"{trade.pnl_pct:+.2f}%",
                        "持有天數": trade.holding_days
                    })
                
                st.dataframe(
                    trade_data,
                    use_container_width=True,
                    hide_index=True
                )
        
        # Parameter summary
        with st.expander("⚙️ 使用參數"):
            params = result.parameters
            param_cols = st.columns(4)
            
            with param_cols[0]:
                st.write(f"**杯身深度**: {params.min_depth}% - {params.max_depth}%")
                st.write(f"**成型天數**: {params.min_cup_days} - {params.max_cup_days} 天")
            
            with param_cols[1]:
                st.write(f"**止損比例**: {params.stop_loss_ratio}%")
                st.write(f"**吻合分數閾值**: {params.score_threshold}")
            
            with param_cols[2]:
                st.write(f"**移動止盈啟動**: {params.profit_threshold}%")
                st.write(f"**移動止盈回調**: {params.trailing_ratio}%")
            
            with param_cols[3]:
                st.write(f"**回測期間**: {result.start_date.strftime('%Y-%m-%d')}")
                st.write(f"**至**: {result.end_date.strftime('%Y-%m-%d')}")
    
    def render(self) -> None:
        """渲染完整的策略實驗室頁面
        
        Requirements: 12.1, 12.2, 12.3
        """
        st.header("🧪 策略實驗室")
        st.markdown("調整模型參數並執行歷史回測，優化您的交易策略。")
        st.divider()
        
        # Parameter sliders
        params = self.render_parameter_sliders()
        
        st.divider()
        
        # Backtest controls
        backtest_dates = self.render_backtest_controls()
        
        # Run backtest if requested
        if backtest_dates:
            start_date, end_date = backtest_dates
            
            with st.spinner("正在執行回測..."):
                result = self.backtest_engine.run_backtest(
                    parameters=params,
                    start_date=start_date,
                    end_date=end_date
                )
            
            st.divider()
            self.render_backtest_results(result)
            
            # Store result in session state
            st.session_state.last_backtest_result = result
        
        # Show previous result if exists
        elif 'last_backtest_result' in st.session_state:
            st.divider()
            st.info("📌 顯示上次回測結果")
            self.render_backtest_results(st.session_state.last_backtest_result)


def run_strategy_lab(
    backtest_engine: Optional[BacktestEngine] = None
) -> None:
    """Run the Strategy Lab application.
    
    Convenience function to start the Strategy Lab.
    
    Args:
        backtest_engine: Optional backtest engine instance
    """
    lab = StrategyLab(backtest_engine=backtest_engine)
    lab.render()


# Entry point for running with: streamlit run pattern_quant/ui/strategy_lab.py
if __name__ == "__main__":
    st.set_page_config(
        page_title="AI PatternQuant - 策略實驗室",
        page_icon="🧪",
        layout="wide"
    )
    run_strategy_lab()
