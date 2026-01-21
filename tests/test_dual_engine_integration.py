"""
Property-Based Tests for Dual Engine Integration

This module contains property-based tests using Hypothesis to verify
the correctness of the DualEngineStrategy and SignalOptimizer integration.

Properties tested:
- Property 14: 雙引擎與因子權重整合

Validates: Requirements 13.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from pattern_quant.strategy import (
    DualEngineStrategy,
    DualEngineAnalysisResult,
    DualEngineConfig,
    MarketState,
    MarketStateClassifier,
)
from pattern_quant.optimization.signal_optimizer import SignalOptimizer, OptimizedSignal
from pattern_quant.optimization.indicator_pool import IndicatorPool
from pattern_quant.optimization.factor_config import FactorConfigManager


# =============================================================================
# Custom Strategies for generating valid test data
# =============================================================================

@st.composite
def valid_ohlcv_series(draw, min_length=50, max_length=80):
    """
    Generate a valid OHLCV price series.
    
    Ensures:
    - high >= close >= low for each bar
    - high >= open >= low for each bar
    - Prices are positive
    - Volumes are positive integers
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    base_price = draw(st.floats(min_value=50.0, max_value=200.0))
    
    highs = []
    lows = []
    closes = []
    volumes = []
    
    current_price = base_price
    
    for _ in range(length):
        # Generate price change
        change_pct = draw(st.floats(min_value=-0.02, max_value=0.02))
        volatility = draw(st.floats(min_value=0.01, max_value=0.02))
        
        # Calculate close
        close = current_price * (1 + change_pct)
        close = max(1.0, close)
        
        # High is above close, low is below close
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        low = max(0.5, low)
        
        # Volume
        volume = draw(st.integers(min_value=100000, max_value=10000000))
        
        highs.append(high)
        lows.append(low)
        closes.append(close)
        volumes.append(volume)
        
        current_price = close
    
    return highs, lows, closes, volumes


@st.composite
def valid_dual_engine_config(draw):
    """Generate a valid DualEngineConfig."""
    return DualEngineConfig(
        enabled=True,
        adx_trend_threshold=draw(st.floats(min_value=22.0, max_value=30.0)),
        adx_range_threshold=draw(st.floats(min_value=15.0, max_value=22.0)),
        trend_allocation=draw(st.floats(min_value=0.8, max_value=1.0)),
        range_allocation=draw(st.floats(min_value=0.4, max_value=0.7)),
        noise_allocation=draw(st.floats(min_value=0.0, max_value=0.2)),
        trend_score_threshold=draw(st.floats(min_value=70.0, max_value=90.0)),
        trend_risk_reward=draw(st.floats(min_value=2.0, max_value=4.0)),
        reversion_rsi_oversold=draw(st.floats(min_value=25.0, max_value=35.0)),
        bbw_stability_threshold=draw(st.floats(min_value=0.05, max_value=0.15)),
    )


# =============================================================================
# Property 14: 雙引擎與因子權重整合
# Feature: dual-engine-strategy, Property 14: 雙引擎與因子權重整合
# Validates: Requirements 13.5
# =============================================================================

class TestDualEngineFactorWeightIntegration:
    """
    Property 14: 雙引擎與因子權重整合
    
    *For any* 同時啟用雙引擎模式和因子權重的情況，最終訊號分數必須整合兩者的計算結果。
    
    Integration rules:
    1. Factor weights calculate base score from pattern + indicators
    2. Dual engine adjusts score based on market state
    3. Allocation weight is applied to final score
    4. Final score = adjusted_score * allocation_weight
    """
    
    @given(
        ohlcv_data=valid_ohlcv_series(min_length=50, max_length=80),
        pattern_score=st.floats(min_value=60.0, max_value=95.0),
        config=valid_dual_engine_config(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dual_engine_integration_produces_valid_result(
        self, ohlcv_data, pattern_score, config
    ):
        """
        Feature: dual-engine-strategy, Property 14: 雙引擎與因子權重整合
        Validates: Requirements 13.5
        
        When dual engine mode and factor weights are both enabled,
        the integration must produce a valid OptimizedSignal with all fields populated.
        """
        highs, lows, closes, volumes = ohlcv_data
        
        # Setup
        indicator_pool = IndicatorPool()
        config_manager = FactorConfigManager()
        optimizer = SignalOptimizer(indicator_pool, config_manager)
        
        entry_price = closes[-1]
        
        # Execute integration
        result = optimizer.optimize_with_dual_engine(
            symbol="TEST",
            pattern_score=pattern_score,
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=entry_price,
            dual_engine_config=config,
        )
        
        # Verify result structure
        assert isinstance(result, OptimizedSignal)
        assert result.symbol == "TEST"
        assert result.pattern_score == pattern_score
        assert result.dual_engine_enabled == True
        assert result.market_state in ["trend", "range", "noise"]
        assert 0.0 <= result.allocation_weight <= 1.0
        assert result.dual_engine_adjusted_score is not None
    
    @given(
        ohlcv_data=valid_ohlcv_series(min_length=50, max_length=80),
        pattern_score=st.floats(min_value=60.0, max_value=95.0),
        config=valid_dual_engine_config(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_final_score_equals_adjusted_times_allocation(
        self, ohlcv_data, pattern_score, config
    ):
        """
        Feature: dual-engine-strategy, Property 14: 雙引擎與因子權重整合
        Validates: Requirements 13.5
        
        Final score must equal adjusted_score * allocation_weight.
        """
        highs, lows, closes, volumes = ohlcv_data
        
        # Setup
        indicator_pool = IndicatorPool()
        config_manager = FactorConfigManager()
        optimizer = SignalOptimizer(indicator_pool, config_manager)
        
        entry_price = closes[-1]
        
        # Execute integration
        result = optimizer.optimize_with_dual_engine(
            symbol="TEST",
            pattern_score=pattern_score,
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=entry_price,
            dual_engine_config=config,
        )
        
        # Verify: final_score = adjusted_score * allocation_weight
        expected_final = result.dual_engine_adjusted_score * result.allocation_weight
        assert abs(result.final_score - expected_final) < 1e-10, \
            f"Final score {result.final_score} != adjusted {result.dual_engine_adjusted_score} * " \
            f"allocation {result.allocation_weight} = {expected_final}"
    
    @given(
        ohlcv_data=valid_ohlcv_series(min_length=50, max_length=80),
        pattern_score=st.floats(min_value=60.0, max_value=95.0),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_disabled_dual_engine_returns_base_signal(
        self, ohlcv_data, pattern_score
    ):
        """
        Feature: dual-engine-strategy, Property 14: 雙引擎與因子權重整合
        Validates: Requirements 13.5
        
        When dual engine is disabled, the result should be the base factor weight signal.
        """
        highs, lows, closes, volumes = ohlcv_data
        
        # Setup with disabled dual engine
        disabled_config = DualEngineConfig(enabled=False)
        
        indicator_pool = IndicatorPool()
        config_manager = FactorConfigManager()
        optimizer = SignalOptimizer(indicator_pool, config_manager)
        
        entry_price = closes[-1]
        
        # Get base signal
        base_signal = optimizer.optimize(
            symbol="TEST",
            pattern_score=pattern_score,
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=entry_price,
        )
        
        # Get integrated signal with disabled dual engine
        integrated_signal = optimizer.optimize_with_dual_engine(
            symbol="TEST",
            pattern_score=pattern_score,
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=entry_price,
            dual_engine_config=disabled_config,
        )
        
        # Verify: disabled dual engine should return base signal
        assert integrated_signal.final_score == base_signal.final_score
        assert integrated_signal.dual_engine_enabled == False
    
    @given(
        ohlcv_data=valid_ohlcv_series(min_length=50, max_length=80),
        pattern_score=st.floats(min_value=60.0, max_value=95.0),
        config=valid_dual_engine_config(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_market_state_affects_score_adjustment(
        self, ohlcv_data, pattern_score, config
    ):
        """
        Feature: dual-engine-strategy, Property 14: 雙引擎與因子權重整合
        Validates: Requirements 13.5
        
        Market state must affect the score adjustment:
        - TREND: +10% bonus
        - RANGE: no change
        - NOISE: -20% penalty
        """
        highs, lows, closes, volumes = ohlcv_data
        
        # Setup
        indicator_pool = IndicatorPool()
        config_manager = FactorConfigManager()
        optimizer = SignalOptimizer(indicator_pool, config_manager)
        
        entry_price = closes[-1]
        
        # Get base signal first
        base_signal = optimizer.optimize(
            symbol="TEST",
            pattern_score=pattern_score,
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=entry_price,
        )
        
        # Get integrated signal
        result = optimizer.optimize_with_dual_engine(
            symbol="TEST",
            pattern_score=pattern_score,
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=entry_price,
            dual_engine_config=config,
        )
        
        # Verify adjustment based on market state
        base_score = base_signal.final_score
        adjusted_score = result.dual_engine_adjusted_score
        
        if result.market_state == "trend":
            # TREND: +10% bonus
            expected_adjusted = base_score * 1.10
            assert abs(adjusted_score - expected_adjusted) < 1e-10, \
                f"TREND adjustment: {adjusted_score} != {expected_adjusted}"
        elif result.market_state == "range":
            # RANGE: no change
            expected_adjusted = base_score
            assert abs(adjusted_score - expected_adjusted) < 1e-10, \
                f"RANGE adjustment: {adjusted_score} != {expected_adjusted}"
        elif result.market_state == "noise":
            # NOISE: -20% penalty
            expected_adjusted = base_score * 0.80
            assert abs(adjusted_score - expected_adjusted) < 1e-10, \
                f"NOISE adjustment: {adjusted_score} != {expected_adjusted}"
    
    @given(
        ohlcv_data=valid_ohlcv_series(min_length=50, max_length=80),
        pattern_score=st.floats(min_value=60.0, max_value=95.0),
        config=valid_dual_engine_config(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_details_include_dual_engine_adjustment(
        self, ohlcv_data, pattern_score, config
    ):
        """
        Feature: dual-engine-strategy, Property 14: 雙引擎與因子權重整合
        Validates: Requirements 13.5
        
        The result details must include a dual_engine adjustment entry.
        """
        highs, lows, closes, volumes = ohlcv_data
        
        # Setup
        indicator_pool = IndicatorPool()
        config_manager = FactorConfigManager()
        optimizer = SignalOptimizer(indicator_pool, config_manager)
        
        entry_price = closes[-1]
        
        # Execute integration
        result = optimizer.optimize_with_dual_engine(
            symbol="TEST",
            pattern_score=pattern_score,
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=entry_price,
            dual_engine_config=config,
        )
        
        # Verify dual_engine detail exists
        dual_engine_details = [d for d in result.details if d.source == "dual_engine"]
        assert len(dual_engine_details) == 1, \
            f"Expected 1 dual_engine detail, got {len(dual_engine_details)}"
        
        detail = dual_engine_details[0]
        assert "雙引擎調整" in detail.reason
        assert "市場狀態" in detail.reason
        assert "資金權重" in detail.reason


# =============================================================================
# Integration Tests for DualEngineStrategy
# =============================================================================

class TestDualEngineStrategyIntegration:
    """Integration tests for DualEngineStrategy class."""
    
    @given(
        ohlcv_data=valid_ohlcv_series(min_length=50, max_length=80),
        config=valid_dual_engine_config(),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_dual_engine_analyze_produces_valid_result(
        self, ohlcv_data, config
    ):
        """
        Test that DualEngineStrategy.analyze() produces valid results.
        """
        highs, lows, closes, volumes = ohlcv_data
        
        # Setup
        strategy = DualEngineStrategy(config=config)
        
        # Execute
        result = strategy.analyze(
            symbol="TEST",
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            total_capital=100000.0,
        )
        
        # Verify result structure
        assert isinstance(result, DualEngineAnalysisResult)
        assert result.symbol == "TEST"
        assert result.market_state is not None
        assert result.resolved_signal is not None
        assert 0.0 <= result.allocation_weight <= 1.0
    
    @given(ohlcv_data=valid_ohlcv_series(min_length=50, max_length=80))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_disabled_dual_engine_returns_empty_result(self, ohlcv_data):
        """
        Test that disabled DualEngineStrategy returns empty result.
        """
        highs, lows, closes, volumes = ohlcv_data
        
        # Setup with disabled config
        config = DualEngineConfig(enabled=False)
        strategy = DualEngineStrategy(config=config)
        
        # Execute
        result = strategy.analyze(
            symbol="TEST",
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
        )
        
        # Verify disabled result
        assert result.market_state.state == MarketState.NOISE
        assert result.resolved_signal.signal is None
        assert result.allocation_weight == 0.0
    
    def test_dual_engine_config_update(self):
        """Test that config update reinitializes components."""
        # Initial config
        config1 = DualEngineConfig(
            enabled=True,
            adx_trend_threshold=25.0,
        )
        strategy = DualEngineStrategy(config=config1)
        
        assert strategy.market_classifier.trend_threshold == 25.0
        
        # Update config
        config2 = DualEngineConfig(
            enabled=True,
            adx_trend_threshold=30.0,
        )
        strategy.update_config(config2)
        
        assert strategy.market_classifier.trend_threshold == 30.0


# =============================================================================
# Unit Tests for edge cases
# =============================================================================

class TestDualEngineIntegrationEdgeCases:
    """Unit tests for edge cases in dual engine integration."""
    
    def test_integrate_dual_engine_result_method(self):
        """Test the integrate_dual_engine_result method."""
        # Setup
        indicator_pool = IndicatorPool()
        config_manager = FactorConfigManager()
        optimizer = SignalOptimizer(indicator_pool, config_manager)
        
        # Create a mock base signal
        from pattern_quant.optimization.signal_optimizer import ScoreDetail, SignalStrength
        base_signal = OptimizedSignal(
            symbol="TEST",
            pattern_score=75.0,
            final_score=80.0,
            strength=SignalStrength.WATCH,
            details=[
                ScoreDetail(source="pattern", raw_value=75.0, score_change=75.0, reason="型態識別分數"),
            ],
        )
        
        # Create a mock dual engine result
        from pattern_quant.strategy.models import ADXResult, BBWResult, MarketStateResult
        from pattern_quant.strategy.signal_resolver import ResolvedSignal, ConflictType
        
        market_state = MarketStateResult(
            state=MarketState.TREND,
            allocation_weight=1.0,
            adx_result=ADXResult(adx=30.0, plus_di=40.0, minus_di=20.0),
            bbw_result=BBWResult(bandwidth=0.1, avg_bandwidth=0.1, is_squeeze=False, change_rate=0.05),
            confidence=0.8,
        )
        
        resolved_signal = ResolvedSignal(
            signal=None,
            strategy_type="none",
            conflict_type=ConflictType.NONE,
            resolution_reason="Test",
        )
        
        dual_engine_result = DualEngineAnalysisResult(
            symbol="TEST",
            market_state=market_state,
            resolved_signal=resolved_signal,
            allocation_weight=1.0,
        )
        
        # Execute integration
        result = optimizer.integrate_dual_engine_result(base_signal, dual_engine_result)
        
        # Verify
        assert result.dual_engine_enabled == True
        assert result.market_state == "trend"
        assert result.allocation_weight == 1.0
        # TREND state should give +10% bonus
        expected_adjusted = 80.0 * 1.10
        assert abs(result.dual_engine_adjusted_score - expected_adjusted) < 1e-10
    
    def test_noise_state_reduces_score(self):
        """Test that NOISE state reduces the final score."""
        # Setup
        indicator_pool = IndicatorPool()
        config_manager = FactorConfigManager()
        optimizer = SignalOptimizer(indicator_pool, config_manager)
        
        # Generate simple price data
        closes = [100.0 + i * 0.1 for i in range(60)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        volumes = [1000000] * 60
        
        # Config that will likely produce NOISE state (ADX in 20-25 range)
        config = DualEngineConfig(
            enabled=True,
            adx_trend_threshold=25.0,
            adx_range_threshold=20.0,
        )
        
        # Get base signal
        base_signal = optimizer.optimize(
            symbol="TEST",
            pattern_score=75.0,
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=closes[-1],
        )
        
        # Get integrated signal with explicit NOISE state
        from pattern_quant.strategy.models import MarketState as MS
        result = optimizer.optimize_with_dual_engine(
            symbol="TEST",
            pattern_score=75.0,
            prices=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=closes[-1],
            dual_engine_config=config,
            market_state=MS.NOISE,
            allocation_weight=0.0,
        )
        
        # NOISE state with 0% allocation should result in 0 final score
        assert result.final_score == 0.0
        assert result.market_state == "noise"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Property 12: 回測報告結構完整性
# Feature: dual-engine-strategy, Property 12: 回測報告結構完整性
# Validates: Requirements 11.1, 11.2, 11.3, 11.4
# =============================================================================

from pattern_quant.ui.strategy_lab_enhanced import (
    EnhancedBacktestTrade,
    StrategyPerformance,
    DualEngineBacktestReport,
    EnhancedBacktestResult,
    RealDataBacktestEngine,
    StrategyParameters,
)
from datetime import datetime, timedelta


@st.composite
def valid_backtest_trade(draw, strategy_type: str = "pattern"):
    """Generate a valid EnhancedBacktestTrade."""
    entry_date = datetime(2024, 1, 1) + timedelta(days=draw(st.integers(0, 100)))
    holding_days = draw(st.integers(1, 30))
    exit_date = entry_date + timedelta(days=holding_days)
    
    entry_price = draw(st.floats(min_value=10.0, max_value=500.0))
    pnl_pct = draw(st.floats(min_value=-20.0, max_value=50.0))
    exit_price = entry_price * (1 + pnl_pct / 100)
    shares = draw(st.integers(min_value=10, max_value=1000))
    pnl = (exit_price - entry_price) * shares
    
    return EnhancedBacktestTrade(
        symbol=draw(st.sampled_from(["AAPL", "GOOGL", "MSFT", "AMZN", "META"])),
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=draw(st.sampled_from(["stop_loss", "trailing_stop", "target", "max_holding"])),
        pnl=pnl,
        pnl_pct=pnl_pct,
        holding_days=holding_days,
        strategy_type=strategy_type,
        market_state=draw(st.sampled_from(["trend", "range", "noise", None])),
    )


@st.composite
def valid_trade_list(draw, min_trades=5, max_trades=20):
    """Generate a list of valid trades with mixed strategy types."""
    num_trades = draw(st.integers(min_value=min_trades, max_value=max_trades))
    trades = []
    
    for _ in range(num_trades):
        strategy_type = draw(st.sampled_from(["trend", "mean_reversion", "pattern"]))
        trade = draw(valid_backtest_trade(strategy_type=strategy_type))
        trades.append(trade)
    
    return trades


class TestBacktestReportStructure:
    """
    Property 12: 回測報告結構完整性
    
    *For any* 完成的回測，報告必須包含：
    - 總體績效（Sharpe Ratio、Total Return）
    - 趨勢策略績效（勝率、平均獲利、交易次數、最大回撤）
    - 震盪策略績效（勝率、平均獲利、交易次數、最大回撤）
    
    Validates: Requirements 11.1, 11.2, 11.3, 11.4
    """
    
    @given(trades=valid_trade_list(min_trades=5, max_trades=20))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dual_engine_report_contains_overall_performance(self, trades):
        """
        Feature: dual-engine-strategy, Property 12: 回測報告結構完整性
        Validates: Requirements 11.1
        
        The report must contain overall performance metrics:
        - Total Return
        - Sharpe Ratio
        - Max Drawdown
        - Total Trades
        """
        # Setup
        engine = RealDataBacktestEngine()
        
        # Generate report
        total_return = sum(t.pnl_pct for t in trades)
        sharpe_ratio = 1.5  # Mock value
        max_drawdown = max(abs(t.pnl_pct) for t in trades if t.pnl_pct < 0) if any(t.pnl_pct < 0 for t in trades) else 0.0
        
        report = engine._generate_dual_engine_report(
            trades=trades,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )
        
        # Verify overall performance fields exist
        assert report is not None, "Report should not be None"
        assert hasattr(report, 'total_return'), "Report must have total_return"
        assert hasattr(report, 'sharpe_ratio'), "Report must have sharpe_ratio"
        assert hasattr(report, 'max_drawdown'), "Report must have max_drawdown"
        assert hasattr(report, 'total_trades'), "Report must have total_trades"
        
        # Verify values
        assert report.total_return == total_return
        assert report.sharpe_ratio == sharpe_ratio
        assert report.max_drawdown == max_drawdown
        assert report.total_trades == len(trades)
    
    @given(trades=valid_trade_list(min_trades=5, max_trades=20))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dual_engine_report_contains_trend_strategy_performance(self, trades):
        """
        Feature: dual-engine-strategy, Property 12: 回測報告結構完整性
        Validates: Requirements 11.2
        
        The report must contain trend strategy performance:
        - Win Rate
        - Average Profit
        - Trade Count
        - Max Drawdown
        """
        # Setup
        engine = RealDataBacktestEngine()
        
        # Generate report
        report = engine._generate_dual_engine_report(
            trades=trades,
            total_return=10.0,
            sharpe_ratio=1.5,
            max_drawdown=5.0
        )
        
        # Verify trend performance exists
        assert report is not None
        assert hasattr(report, 'trend_performance'), "Report must have trend_performance"
        
        trend_perf = report.trend_performance
        assert isinstance(trend_perf, StrategyPerformance)
        
        # Verify required fields
        assert hasattr(trend_perf, 'strategy_name')
        assert hasattr(trend_perf, 'total_trades')
        assert hasattr(trend_perf, 'winning_trades')
        assert hasattr(trend_perf, 'losing_trades')
        assert hasattr(trend_perf, 'win_rate')
        assert hasattr(trend_perf, 'avg_profit')
        assert hasattr(trend_perf, 'max_drawdown')
        
        # Verify trade count matches
        trend_trades = [t for t in trades if t.strategy_type == "trend"]
        assert trend_perf.total_trades == len(trend_trades)
    
    @given(trades=valid_trade_list(min_trades=5, max_trades=20))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dual_engine_report_contains_reversion_strategy_performance(self, trades):
        """
        Feature: dual-engine-strategy, Property 12: 回測報告結構完整性
        Validates: Requirements 11.3
        
        The report must contain mean reversion strategy performance:
        - Win Rate
        - Average Profit
        - Trade Count
        - Max Drawdown
        """
        # Setup
        engine = RealDataBacktestEngine()
        
        # Generate report
        report = engine._generate_dual_engine_report(
            trades=trades,
            total_return=10.0,
            sharpe_ratio=1.5,
            max_drawdown=5.0
        )
        
        # Verify reversion performance exists
        assert report is not None
        assert hasattr(report, 'reversion_performance'), "Report must have reversion_performance"
        
        reversion_perf = report.reversion_performance
        assert isinstance(reversion_perf, StrategyPerformance)
        
        # Verify required fields
        assert hasattr(reversion_perf, 'strategy_name')
        assert hasattr(reversion_perf, 'total_trades')
        assert hasattr(reversion_perf, 'winning_trades')
        assert hasattr(reversion_perf, 'losing_trades')
        assert hasattr(reversion_perf, 'win_rate')
        assert hasattr(reversion_perf, 'avg_profit')
        assert hasattr(reversion_perf, 'max_drawdown')
        
        # Verify trade count matches
        reversion_trades = [t for t in trades if t.strategy_type == "mean_reversion"]
        assert reversion_perf.total_trades == len(reversion_trades)
    
    @given(trades=valid_trade_list(min_trades=5, max_trades=20))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_strategy_performance_win_rate_calculation(self, trades):
        """
        Feature: dual-engine-strategy, Property 12: 回測報告結構完整性
        Validates: Requirements 11.4
        
        Win rate must be correctly calculated as:
        win_rate = (winning_trades / total_trades) * 100
        """
        # Setup
        engine = RealDataBacktestEngine()
        
        # Generate report
        report = engine._generate_dual_engine_report(
            trades=trades,
            total_return=10.0,
            sharpe_ratio=1.5,
            max_drawdown=5.0
        )
        
        # Verify win rate calculation for each strategy
        for perf in [report.trend_performance, report.reversion_performance]:
            if perf.total_trades > 0:
                expected_win_rate = (perf.winning_trades / perf.total_trades) * 100
                assert abs(perf.win_rate - expected_win_rate) < 1e-10, \
                    f"Win rate {perf.win_rate} != expected {expected_win_rate}"
            else:
                assert perf.win_rate == 0.0
    
    @given(trades=valid_trade_list(min_trades=5, max_trades=20))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_strategy_performance_trade_count_consistency(self, trades):
        """
        Feature: dual-engine-strategy, Property 12: 回測報告結構完整性
        Validates: Requirements 11.4
        
        Trade counts must be consistent:
        winning_trades + losing_trades == total_trades
        """
        # Setup
        engine = RealDataBacktestEngine()
        
        # Generate report
        report = engine._generate_dual_engine_report(
            trades=trades,
            total_return=10.0,
            sharpe_ratio=1.5,
            max_drawdown=5.0
        )
        
        # Verify trade count consistency for each strategy
        for perf in [report.trend_performance, report.reversion_performance]:
            assert perf.winning_trades + perf.losing_trades == perf.total_trades, \
                f"Trade count inconsistent: {perf.winning_trades} + {perf.losing_trades} != {perf.total_trades}"
    
    @given(trades=valid_trade_list(min_trades=5, max_trades=20))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_strategy_performance_profit_factor_calculation(self, trades):
        """
        Feature: dual-engine-strategy, Property 12: 回測報告結構完整性
        Validates: Requirements 11.4
        
        Profit factor must be correctly calculated as:
        profit_factor = total_profit / total_loss (if total_loss > 0)
        """
        # Setup
        engine = RealDataBacktestEngine()
        
        # Generate report
        report = engine._generate_dual_engine_report(
            trades=trades,
            total_return=10.0,
            sharpe_ratio=1.5,
            max_drawdown=5.0
        )
        
        # Verify profit factor is non-negative
        for perf in [report.trend_performance, report.reversion_performance]:
            assert perf.profit_factor >= 0.0, \
                f"Profit factor must be non-negative, got {perf.profit_factor}"
    
    def test_empty_trades_returns_valid_report(self):
        """
        Feature: dual-engine-strategy, Property 12: 回測報告結構完整性
        Validates: Requirements 11.1, 11.2, 11.3, 11.4
        
        Empty trade list should return None (no report).
        """
        # Setup
        engine = RealDataBacktestEngine()
        
        # Generate report with empty trades
        report = engine._generate_dual_engine_report(
            trades=[],
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0
        )
        
        # Empty trades should return None
        assert report is None
    
    def test_single_strategy_trades_report(self):
        """
        Feature: dual-engine-strategy, Property 12: 回測報告結構完整性
        Validates: Requirements 11.2, 11.3
        
        Report with only one strategy type should still have both performance sections.
        """
        # Setup
        engine = RealDataBacktestEngine()
        
        # Create trades with only trend strategy
        trades = [
            EnhancedBacktestTrade(
                symbol="AAPL",
                entry_date=datetime(2024, 1, 1),
                entry_price=100.0,
                exit_date=datetime(2024, 1, 10),
                exit_price=110.0,
                exit_reason="target",
                pnl=1000.0,
                pnl_pct=10.0,
                holding_days=9,
                strategy_type="trend",
            ),
            EnhancedBacktestTrade(
                symbol="GOOGL",
                entry_date=datetime(2024, 1, 15),
                entry_price=150.0,
                exit_date=datetime(2024, 1, 20),
                exit_price=145.0,
                exit_reason="stop_loss",
                pnl=-500.0,
                pnl_pct=-3.33,
                holding_days=5,
                strategy_type="trend",
            ),
        ]
        
        # Generate report
        report = engine._generate_dual_engine_report(
            trades=trades,
            total_return=6.67,
            sharpe_ratio=1.0,
            max_drawdown=3.33
        )
        
        # Verify both performance sections exist
        assert report is not None
        assert report.trend_performance.total_trades == 2
        assert report.reversion_performance.total_trades == 0
        assert report.reversion_performance.win_rate == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
