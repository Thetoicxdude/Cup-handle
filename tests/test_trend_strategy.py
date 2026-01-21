"""
Property-Based Tests for Trend Strategy

This module contains property-based tests using Hypothesis to verify
the correctness of the TrendStrategy implementation.

Properties tested:
- Property 4: 趨勢策略進場條件
- Property 5: 趨勢策略移動止損

Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from dataclasses import dataclass
from datetime import datetime

from pattern_quant.strategy.trend_strategy import TrendStrategy
from pattern_quant.strategy.models import MarketState, TrendSignal
from pattern_quant.core.models import (
    PatternResult,
    CupPattern,
    HandlePattern,
    MatchScore,
    Position,
)


# =============================================================================
# Custom Strategies for generating valid test data
# =============================================================================

@st.composite
def valid_cup_pattern(draw):
    """Generate a valid CupPattern."""
    left_peak_price = draw(st.floats(min_value=50.0, max_value=200.0))
    # Right peak should be close to left peak (within 5% tolerance)
    right_peak_price = left_peak_price * draw(st.floats(min_value=0.95, max_value=1.05))
    # Bottom should be 12-33% below peaks
    depth_ratio = draw(st.floats(min_value=0.12, max_value=0.33))
    bottom_price = left_peak_price * (1 - depth_ratio)
    
    return CupPattern(
        left_peak_index=draw(st.integers(min_value=0, max_value=50)),
        left_peak_price=left_peak_price,
        right_peak_index=draw(st.integers(min_value=80, max_value=150)),
        right_peak_price=right_peak_price,
        bottom_index=draw(st.integers(min_value=51, max_value=79)),
        bottom_price=bottom_price,
        r_squared=draw(st.floats(min_value=0.8, max_value=1.0)),
        depth_ratio=depth_ratio,
        symmetry_score=draw(st.floats(min_value=0.7, max_value=1.0))
    )


@st.composite
def valid_handle_pattern(draw, cup: CupPattern):
    """Generate a valid HandlePattern based on a cup."""
    # Handle lowest price should be above cup bottom but below right peak
    min_handle_price = cup.bottom_price
    max_handle_price = cup.right_peak_price * 0.95
    
    if min_handle_price >= max_handle_price:
        min_handle_price = cup.right_peak_price * 0.7
        max_handle_price = cup.right_peak_price * 0.95
    
    return HandlePattern(
        start_index=cup.right_peak_index + 1,
        end_index=cup.right_peak_index + draw(st.integers(min_value=5, max_value=25)),
        lowest_price=draw(st.floats(min_value=min_handle_price, max_value=max_handle_price)),
        volume_slope=draw(st.floats(min_value=-0.5, max_value=0.0))
    )


@st.composite
def valid_match_score(draw, min_score=0.0, max_score=100.0):
    """Generate a valid MatchScore."""
    total = draw(st.floats(min_value=min_score, max_value=max_score))
    return MatchScore(
        total_score=total,
        r_squared_score=draw(st.floats(min_value=0.0, max_value=100.0)),
        symmetry_score=draw(st.floats(min_value=0.0, max_value=100.0)),
        volume_score=draw(st.floats(min_value=0.0, max_value=100.0)),
        depth_score=draw(st.floats(min_value=0.0, max_value=100.0))
    )


@st.composite
def valid_pattern_result_for_entry(draw, symbol="TEST"):
    """Generate a valid PatternResult that could trigger entry."""
    cup = draw(valid_cup_pattern())
    handle = draw(valid_handle_pattern(cup))
    score = draw(valid_match_score(min_score=80.0, max_score=100.0))
    
    return PatternResult(
        symbol=symbol,
        pattern_type="cup_and_handle",
        cup=cup,
        handle=handle,
        score=score,
        is_valid=True,
        rejection_reason=None
    )


@st.composite
def invalid_pattern_result(draw, symbol="TEST"):
    """Generate an invalid PatternResult."""
    return PatternResult(
        symbol=symbol,
        pattern_type="cup_and_handle",
        cup=None,
        handle=None,
        score=None,
        is_valid=False,
        rejection_reason=draw(st.sampled_from([
            "Not in uptrend",
            "No valid cup pattern found",
            "No valid handle pattern found",
            "Insufficient data"
        ]))
    )


@st.composite
def valid_position(draw, symbol="TEST"):
    """Generate a valid Position."""
    entry_price = draw(st.floats(min_value=50.0, max_value=200.0))
    stop_loss_price = entry_price * draw(st.floats(min_value=0.85, max_value=0.95))
    current_price = entry_price * draw(st.floats(min_value=0.9, max_value=1.5))
    trailing_active = draw(st.booleans())
    trailing_price = entry_price * draw(st.floats(min_value=1.0, max_value=1.3)) if trailing_active else 0.0
    
    return Position(
        symbol=symbol,
        quantity=draw(st.integers(min_value=1, max_value=1000)),
        entry_price=entry_price,
        current_price=current_price,
        sector=draw(st.sampled_from(["Technology", "Healthcare", "Finance"])),
        entry_time=datetime.now(),
        stop_loss_price=stop_loss_price,
        trailing_stop_active=trailing_active,
        trailing_stop_price=trailing_price
    )


# =============================================================================
# Property 4: 趨勢策略進場條件
# Feature: dual-engine-strategy, Property 4: 趨勢策略進場條件
# Validates: Requirements 4.1, 4.2, 4.3
# =============================================================================

class TestTrendStrategyEntryProperty:
    """
    Property 4: 趨勢策略進場條件
    
    *For any* 市場狀態為 Trend 且型態分數 > 80 的情況，當價格突破頸線時，
    必須生成買入信號並設定止損於型態支撐位。
    """
    
    @given(pattern_result=valid_pattern_result_for_entry())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_entry_signal_generated_when_conditions_met(self, pattern_result):
        """
        Feature: dual-engine-strategy, Property 4: 趨勢策略進場條件
        Validates: Requirements 4.1, 4.2, 4.3
        
        For any valid pattern with score > 80 in TREND state, when price breaks neckline,
        a buy signal must be generated with stop loss at pattern support.
        """
        strategy = TrendStrategy(score_threshold=80.0)
        
        # Price that breaks the neckline (right peak + buffer)
        neckline = pattern_result.cup.right_peak_price
        breakout_price = neckline * 1.01  # Above neckline + buffer
        
        signal = strategy.check_entry(
            pattern_result=pattern_result,
            current_price=breakout_price,
            market_state=MarketState.TREND
        )
        
        # Signal must be generated
        assert signal is not None, \
            f"Expected signal for score={pattern_result.score.total_score}, " \
            f"price={breakout_price}, neckline={neckline}"
        
        # Signal type must be breakout_long
        assert signal.signal_type == "breakout_long"
        
        # Entry price must match current price
        assert signal.entry_price == breakout_price
        
        # Neckline price must be set
        assert signal.neckline_price == neckline
        
        # Stop loss must be at pattern support (handle lowest or cup bottom)
        expected_stop = pattern_result.handle.lowest_price
        assert signal.stop_loss_price == expected_stop, \
            f"Stop loss {signal.stop_loss_price} != expected {expected_stop}"
        
        # Pattern score must be recorded
        assert signal.pattern_score == pattern_result.score.total_score
    
    @given(pattern_result=valid_pattern_result_for_entry())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_entry_when_not_trend_state(self, pattern_result):
        """
        Feature: dual-engine-strategy, Property 4: 趨勢策略進場條件
        Validates: Requirements 4.1
        
        For any valid pattern, no signal should be generated when market state is not TREND.
        """
        strategy = TrendStrategy(score_threshold=80.0)
        
        neckline = pattern_result.cup.right_peak_price
        breakout_price = neckline * 1.01
        
        # Test with RANGE state
        signal_range = strategy.check_entry(
            pattern_result=pattern_result,
            current_price=breakout_price,
            market_state=MarketState.RANGE
        )
        assert signal_range is None, "Should not generate signal in RANGE state"
        
        # Test with NOISE state
        signal_noise = strategy.check_entry(
            pattern_result=pattern_result,
            current_price=breakout_price,
            market_state=MarketState.NOISE
        )
        assert signal_noise is None, "Should not generate signal in NOISE state"
    
    @given(pattern_result=valid_pattern_result_for_entry())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_entry_when_price_below_neckline(self, pattern_result):
        """
        Feature: dual-engine-strategy, Property 4: 趨勢策略進場條件
        Validates: Requirements 4.2
        
        For any valid pattern, no signal should be generated when price is below neckline.
        """
        strategy = TrendStrategy(score_threshold=80.0, breakout_buffer=0.005)
        
        neckline = pattern_result.cup.right_peak_price
        # Price below neckline + buffer
        below_breakout_price = neckline * 1.002  # Below the 0.5% buffer
        
        signal = strategy.check_entry(
            pattern_result=pattern_result,
            current_price=below_breakout_price,
            market_state=MarketState.TREND
        )
        
        assert signal is None, \
            f"Should not generate signal when price {below_breakout_price} " \
            f"is below breakout level {neckline * 1.005}"
    
    @given(pattern_result=invalid_pattern_result())
    @settings(max_examples=100)
    def test_no_entry_when_pattern_invalid(self, pattern_result):
        """
        Feature: dual-engine-strategy, Property 4: 趨勢策略進場條件
        Validates: Requirements 4.1
        
        For any invalid pattern, no signal should be generated.
        """
        strategy = TrendStrategy(score_threshold=80.0)
        
        signal = strategy.check_entry(
            pattern_result=pattern_result,
            current_price=100.0,
            market_state=MarketState.TREND
        )
        
        assert signal is None, "Should not generate signal for invalid pattern"
    
    @given(
        cup=valid_cup_pattern(),
        score_value=st.floats(min_value=0.0, max_value=79.9)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_entry_when_score_below_threshold(self, cup, score_value):
        """
        Feature: dual-engine-strategy, Property 4: 趨勢策略進場條件
        Validates: Requirements 4.1
        
        For any pattern with score < 80, no signal should be generated.
        """
        strategy = TrendStrategy(score_threshold=80.0)
        
        # Create pattern with low score
        handle = HandlePattern(
            start_index=cup.right_peak_index + 1,
            end_index=cup.right_peak_index + 10,
            lowest_price=cup.right_peak_price * 0.9,
            volume_slope=-0.1
        )
        score = MatchScore(
            total_score=score_value,
            r_squared_score=50.0,
            symmetry_score=50.0,
            volume_score=50.0,
            depth_score=50.0
        )
        pattern_result = PatternResult(
            symbol="TEST",
            pattern_type="cup_and_handle",
            cup=cup,
            handle=handle,
            score=score,
            is_valid=True,
            rejection_reason=None
        )
        
        neckline = cup.right_peak_price
        breakout_price = neckline * 1.01
        
        signal = strategy.check_entry(
            pattern_result=pattern_result,
            current_price=breakout_price,
            market_state=MarketState.TREND
        )
        
        assert signal is None, \
            f"Should not generate signal for score {score_value} < 80"


# =============================================================================
# Property 5: 趨勢策略移動止損
# Feature: dual-engine-strategy, Property 5: 趨勢策略移動止損
# Validates: Requirements 4.4
# =============================================================================

class TestTrendStrategyTrailingStopProperty:
    """
    Property 5: 趨勢策略移動止損
    
    *For any* 持倉獲利達到風險回報比 1:3 的情況，移動止損必須被啟動。
    """
    
    @given(
        entry_price=st.floats(min_value=50.0, max_value=200.0),
        risk_pct=st.floats(min_value=0.05, max_value=0.15)
    )
    @settings(max_examples=100)
    def test_trailing_stop_activates_at_target_rr(self, entry_price, risk_pct):
        """
        Feature: dual-engine-strategy, Property 5: 趨勢策略移動止損
        Validates: Requirements 4.4
        
        For any position where profit reaches 1:3 risk-reward ratio,
        trailing stop must be activated.
        """
        strategy = TrendStrategy(risk_reward_target=3.0)
        
        # Calculate stop loss and target price
        stop_loss_price = entry_price * (1 - risk_pct)
        risk = entry_price - stop_loss_price
        
        # Price slightly above 1:3 risk-reward to avoid floating-point precision issues
        # Adding a small epsilon (0.1%) ensures we're clearly above the threshold
        target_profit = risk * 3.01  # Slightly above 3.0 to handle floating-point precision
        target_price = entry_price + target_profit
        
        should_activate = strategy.should_activate_trailing_stop(
            entry_price=entry_price,
            current_price=target_price,
            stop_loss_price=stop_loss_price
        )
        
        assert should_activate is True, \
            f"Trailing stop should activate at 1:3 RR. " \
            f"Entry={entry_price}, Stop={stop_loss_price}, Current={target_price}"
    
    @given(
        entry_price=st.floats(min_value=50.0, max_value=200.0),
        risk_pct=st.floats(min_value=0.05, max_value=0.15),
        profit_ratio=st.floats(min_value=0.0, max_value=2.9)
    )
    @settings(max_examples=100)
    def test_trailing_stop_not_activated_below_target(self, entry_price, risk_pct, profit_ratio):
        """
        Feature: dual-engine-strategy, Property 5: 趨勢策略移動止損
        Validates: Requirements 4.4
        
        For any position where profit is below 1:3 risk-reward ratio,
        trailing stop must NOT be activated.
        """
        strategy = TrendStrategy(risk_reward_target=3.0)
        
        stop_loss_price = entry_price * (1 - risk_pct)
        risk = entry_price - stop_loss_price
        
        # Price below 1:3 risk-reward
        profit = risk * profit_ratio
        current_price = entry_price + profit
        
        should_activate = strategy.should_activate_trailing_stop(
            entry_price=entry_price,
            current_price=current_price,
            stop_loss_price=stop_loss_price
        )
        
        assert should_activate is False, \
            f"Trailing stop should NOT activate below 1:3 RR. " \
            f"Entry={entry_price}, Stop={stop_loss_price}, Current={current_price}, " \
            f"Profit ratio={profit_ratio}"
    
    @given(
        entry_price=st.floats(min_value=50.0, max_value=200.0),
        risk_pct=st.floats(min_value=0.05, max_value=0.15),
        profit_multiplier=st.floats(min_value=3.0, max_value=10.0)
    )
    @settings(max_examples=100)
    def test_trailing_stop_price_calculation(self, entry_price, risk_pct, profit_multiplier):
        """
        Feature: dual-engine-strategy, Property 5: 趨勢策略移動止損
        Validates: Requirements 4.4
        
        For any position with trailing stop activated, the trailing stop price
        must be calculated correctly and only move upward.
        """
        strategy = TrendStrategy(
            risk_reward_target=3.0,
            trailing_activation=0.10  # 10% trailing
        )
        
        stop_loss_price = entry_price * (1 - risk_pct)
        risk = entry_price - stop_loss_price
        
        # Price above 1:3 risk-reward
        profit = risk * profit_multiplier
        current_price = entry_price + profit
        
        trailing_stop = strategy.calculate_trailing_stop_price(
            entry_price=entry_price,
            current_price=current_price,
            current_trailing_stop=None
        )
        
        # Trailing stop should be current_price * (1 - 0.10) or entry_price, whichever is higher
        expected_trailing = max(current_price * 0.90, entry_price)
        
        assert abs(trailing_stop - expected_trailing) < 0.01, \
            f"Trailing stop {trailing_stop} != expected {expected_trailing}"
    
    @given(
        entry_price=st.floats(min_value=50.0, max_value=200.0),
        price_increases=st.lists(
            st.floats(min_value=1.01, max_value=1.05),
            min_size=3,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_trailing_stop_only_moves_up(self, entry_price, price_increases):
        """
        Feature: dual-engine-strategy, Property 5: 趨勢策略移動止損
        Validates: Requirements 4.4
        
        Trailing stop must only move upward, never downward.
        """
        strategy = TrendStrategy(trailing_activation=0.10)
        
        current_trailing = None
        current_price = entry_price
        
        for multiplier in price_increases:
            new_price = current_price * multiplier
            
            new_trailing = strategy.calculate_trailing_stop_price(
                entry_price=entry_price,
                current_price=new_price,
                current_trailing_stop=current_trailing
            )
            
            if current_trailing is not None:
                assert new_trailing >= current_trailing, \
                    f"Trailing stop moved down: {current_trailing} -> {new_trailing}"
            
            current_trailing = new_trailing
            current_price = new_price


# =============================================================================
# Unit Tests for edge cases and exit conditions
# =============================================================================

class TestTrendStrategyExitConditions:
    """Unit tests for exit conditions (Requirements 4.5)."""
    
    def test_exit_on_stop_loss_hit(self):
        """Test exit when price hits stop loss."""
        strategy = TrendStrategy()
        
        position = Position(
            symbol="TEST",
            quantity=100,
            entry_price=100.0,
            current_price=90.0,
            sector="Technology",
            entry_time=datetime.now(),
            stop_loss_price=92.0,
            trailing_stop_active=False,
            trailing_stop_price=0.0
        )
        
        exit_reason = strategy.check_exit(
            position=position,
            current_price=91.0,  # Below stop loss
            entry_price=100.0,
            stop_loss_price=92.0
        )
        
        assert exit_reason == "stop_loss"
    
    def test_exit_on_trailing_stop_hit(self):
        """Test exit when price hits trailing stop."""
        strategy = TrendStrategy()
        
        position = Position(
            symbol="TEST",
            quantity=100,
            entry_price=100.0,
            current_price=125.0,
            sector="Technology",
            entry_time=datetime.now(),
            stop_loss_price=92.0,
            trailing_stop_active=True,
            trailing_stop_price=120.0
        )
        
        exit_reason = strategy.check_exit(
            position=position,
            current_price=119.0,  # Below trailing stop
            entry_price=100.0,
            stop_loss_price=92.0
        )
        
        assert exit_reason == "trailing_stop"
    
    def test_no_exit_when_price_above_stops(self):
        """Test no exit when price is above all stop levels."""
        strategy = TrendStrategy()
        
        position = Position(
            symbol="TEST",
            quantity=100,
            entry_price=100.0,
            current_price=130.0,
            sector="Technology",
            entry_time=datetime.now(),
            stop_loss_price=92.0,
            trailing_stop_active=True,
            trailing_stop_price=120.0
        )
        
        exit_reason = strategy.check_exit(
            position=position,
            current_price=125.0,  # Above both stops
            entry_price=100.0,
            stop_loss_price=92.0
        )
        
        assert exit_reason is None
    
    def test_strategy_from_config(self):
        """Test creating strategy from DualEngineConfig."""
        from pattern_quant.strategy.models import DualEngineConfig
        
        config = DualEngineConfig(
            trend_score_threshold=85.0,
            trend_risk_reward=4.0,
            trend_trailing_activation=0.15
        )
        
        strategy = TrendStrategy.from_config(config)
        
        assert strategy.score_threshold == 85.0
        assert strategy.risk_reward_target == 4.0
        assert strategy.trailing_activation == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
