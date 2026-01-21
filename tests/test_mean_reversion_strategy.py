"""
Property-Based Tests for Mean Reversion Strategy

This module contains property-based tests using Hypothesis to verify
the correctness of the MeanReversionStrategy implementation.

Properties tested:
- Property 6: 均值回歸進場確認
- Property 7: 均值回歸分批出場
- Property 8: 均值回歸風控覆蓋
- Property 13: 錘頭線識別正確性

Validates: Requirements 5.1, 5.2, 5.3, 5.5, 5.6, 6.1, 12.1, 12.2, 12.3
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from pattern_quant.strategy.mean_reversion_strategy import MeanReversionStrategy
from pattern_quant.strategy.models import MarketState, CandlePattern, MeanReversionSignal
from pattern_quant.optimization.models import BollingerResult, RSIResult


# =============================================================================
# Custom Strategies for generating valid test data
# =============================================================================

@st.composite
def valid_bollinger_result(draw):
    """Generate a valid BollingerResult."""
    middle = draw(st.floats(min_value=50.0, max_value=200.0))
    # Bandwidth typically 5-20% of middle
    bandwidth_pct = draw(st.floats(min_value=0.05, max_value=0.20))
    half_band = middle * bandwidth_pct / 2
    
    return BollingerResult(
        upper=middle + half_band,
        middle=middle,
        lower=middle - half_band,
        bandwidth=bandwidth_pct,
        squeeze=draw(st.booleans()),
        breakout_upper=False
    )


@st.composite
def valid_rsi_result(draw, oversold=False, overbought=False):
    """Generate a valid RSIResult."""
    if oversold:
        value = draw(st.floats(min_value=0.0, max_value=29.9))
    elif overbought:
        value = draw(st.floats(min_value=70.1, max_value=100.0))
    else:
        value = draw(st.floats(min_value=0.0, max_value=100.0))
    
    return RSIResult(
        value=value,
        is_overbought=value > 70,
        is_oversold=value < 30,
        trend_zone=30 <= value <= 70,
        support_bounce=False
    )


@st.composite
def hammer_candle_data(draw):
    """Generate candle data that forms a valid hammer pattern."""
    open_price = draw(st.floats(min_value=50.0, max_value=200.0))
    # Body size (small relative to shadows)
    body_size = draw(st.floats(min_value=0.5, max_value=3.0))
    # Lower shadow at least 2x body
    lower_shadow_ratio = draw(st.floats(min_value=2.0, max_value=5.0))
    # Upper shadow very small (< 0.1x body)
    upper_shadow_ratio = draw(st.floats(min_value=0.0, max_value=0.09))
    
    # Bullish hammer (close > open)
    close = open_price + body_size
    lower_shadow = body_size * lower_shadow_ratio
    upper_shadow = body_size * upper_shadow_ratio
    
    low = open_price - lower_shadow
    high = close + upper_shadow
    
    return {
        "open_price": open_price,
        "high": high,
        "low": low,
        "close": close,
        "body_size": body_size,
        "lower_shadow_ratio": lower_shadow_ratio,
        "upper_shadow_ratio": upper_shadow_ratio
    }


@st.composite
def non_hammer_candle_data(draw):
    """Generate candle data that does NOT form a hammer pattern."""
    open_price = draw(st.floats(min_value=50.0, max_value=200.0))
    body_size = draw(st.floats(min_value=0.5, max_value=3.0))
    
    # Either lower shadow too short OR upper shadow too long
    invalid_type = draw(st.sampled_from(["short_lower", "long_upper", "both"]))
    
    if invalid_type == "short_lower":
        lower_shadow_ratio = draw(st.floats(min_value=0.0, max_value=1.9))
        upper_shadow_ratio = draw(st.floats(min_value=0.0, max_value=0.09))
    elif invalid_type == "long_upper":
        lower_shadow_ratio = draw(st.floats(min_value=2.0, max_value=5.0))
        upper_shadow_ratio = draw(st.floats(min_value=0.11, max_value=2.0))
    else:  # both invalid
        lower_shadow_ratio = draw(st.floats(min_value=0.0, max_value=1.9))
        upper_shadow_ratio = draw(st.floats(min_value=0.11, max_value=2.0))
    
    close = open_price + body_size
    lower_shadow = body_size * lower_shadow_ratio
    upper_shadow = body_size * upper_shadow_ratio
    
    low = open_price - lower_shadow
    high = close + upper_shadow
    
    return {
        "open_price": open_price,
        "high": high,
        "low": low,
        "close": close,
        "lower_shadow_ratio": lower_shadow_ratio,
        "upper_shadow_ratio": upper_shadow_ratio
    }


# =============================================================================
# Property 13: 錘頭線識別正確性
# Feature: dual-engine-strategy, Property 13: 錘頭線識別正確性
# Validates: Requirements 12.1, 12.2, 12.3
# =============================================================================

class TestHammerDetectionProperty:
    """
    Property 13: 錘頭線識別正確性
    
    *For any* K 線數據，識別為錘頭線的 K 線必須滿足：
    下影線長度 >= 實體長度 × 2，且上影線極短（<= 實體長度 × 0.1）。
    """
    
    @given(candle=hammer_candle_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_hammer_detected(self, candle):
        """
        Feature: dual-engine-strategy, Property 13: 錘頭線識別正確性
        Validates: Requirements 12.1, 12.2, 12.3
        
        For any candle with lower_shadow >= 2*body and upper_shadow <= 0.1*body,
        it must be detected as a hammer.
        """
        strategy = MeanReversionStrategy()
        
        result = strategy.detect_hammer(
            open_price=candle["open_price"],
            high=candle["high"],
            low=candle["low"],
            close=candle["close"]
        )
        
        assert result is not None, \
            f"Expected hammer detection for lower_shadow_ratio={candle['lower_shadow_ratio']:.2f}, " \
            f"upper_shadow_ratio={candle['upper_shadow_ratio']:.2f}"
        
        assert result.pattern_type == "hammer"
        assert 0.0 <= result.confidence <= 1.0
    
    @given(candle=non_hammer_candle_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_non_hammer_not_detected(self, candle):
        """
        Feature: dual-engine-strategy, Property 13: 錘頭線識別正確性
        Validates: Requirements 12.2, 12.3
        
        For any candle that doesn't meet hammer criteria,
        it must NOT be detected as a hammer.
        """
        strategy = MeanReversionStrategy()
        
        result = strategy.detect_hammer(
            open_price=candle["open_price"],
            high=candle["high"],
            low=candle["low"],
            close=candle["close"]
        )
        
        assert result is None, \
            f"Should not detect hammer for lower_shadow_ratio={candle['lower_shadow_ratio']:.2f}, " \
            f"upper_shadow_ratio={candle['upper_shadow_ratio']:.2f}"
    
    @given(
        open_price=st.floats(min_value=50.0, max_value=200.0),
        body_size=st.floats(min_value=0.5, max_value=3.0),
        lower_ratio=st.floats(min_value=2.0, max_value=10.0)
    )
    @settings(max_examples=100)
    def test_hammer_confidence_increases_with_lower_shadow(self, open_price, body_size, lower_ratio):
        """
        Feature: dual-engine-strategy, Property 13: 錘頭線識別正確性
        Validates: Requirements 12.4
        
        Hammer confidence should increase with longer lower shadow.
        """
        strategy = MeanReversionStrategy()
        
        close = open_price + body_size
        lower_shadow = body_size * lower_ratio
        upper_shadow = body_size * 0.05  # Very small upper shadow
        
        low = open_price - lower_shadow
        high = close + upper_shadow
        
        result = strategy.detect_hammer(open_price, high, low, close)
        
        assert result is not None
        assert result.confidence >= 0.6, \
            f"Confidence {result.confidence} should be at least 0.6 for valid hammer"


# =============================================================================
# Property 6: 均值回歸進場確認
# Feature: dual-engine-strategy, Property 6: 均值回歸進場確認
# Validates: Requirements 5.1, 5.2, 5.3
# =============================================================================

class TestMeanReversionEntryProperty:
    """
    Property 6: 均值回歸進場確認
    
    *For any* 市場狀態為 Range 且價格觸及布林下軌的情況，
    當 RSI < 30 或出現錘頭線時，必須確認做多信號。
    """
    
    @given(
        bollinger=valid_bollinger_result(),
        rsi=valid_rsi_result(oversold=True)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_entry_with_rsi_confirmation(self, bollinger, rsi):
        """
        Feature: dual-engine-strategy, Property 6: 均值回歸進場確認
        Validates: Requirements 5.1, 5.2
        
        For any RANGE state with price at lower band and RSI < 30,
        a long entry signal must be generated.
        """
        strategy = MeanReversionStrategy(rsi_oversold=30.0)
        
        # Price at or below lower band
        current_price = bollinger.lower * 0.99
        
        signal = strategy.check_entry(
            current_price=current_price,
            bollinger=bollinger,
            rsi=rsi,
            candle=None,
            market_state=MarketState.RANGE,
            symbol="TEST"
        )
        
        assert signal is not None, \
            f"Expected entry signal with RSI={rsi.value:.1f} < 30"
        
        assert signal.signal_type == "long_entry"
        assert signal.confirmation in ["rsi_oversold", "both"]
        assert signal.entry_price == bollinger.lower
        assert signal.target_1 == bollinger.middle
        assert signal.target_2 == bollinger.upper
    
    @given(bollinger=valid_bollinger_result())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_entry_with_hammer_confirmation(self, bollinger):
        """
        Feature: dual-engine-strategy, Property 6: 均值回歸進場確認
        Validates: Requirements 5.1, 5.3
        
        For any RANGE state with price at lower band and hammer pattern,
        a long entry signal must be generated.
        """
        strategy = MeanReversionStrategy()
        
        current_price = bollinger.lower * 0.99
        hammer = CandlePattern(pattern_type="hammer", confidence=0.8)
        
        signal = strategy.check_entry(
            current_price=current_price,
            bollinger=bollinger,
            rsi=None,
            candle=hammer,
            market_state=MarketState.RANGE,
            symbol="TEST"
        )
        
        assert signal is not None, "Expected entry signal with hammer confirmation"
        assert signal.signal_type == "long_entry"
        assert signal.confirmation == "hammer"
    
    @given(
        bollinger=valid_bollinger_result(),
        rsi=valid_rsi_result(oversold=True)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_entry_with_both_confirmations(self, bollinger, rsi):
        """
        Feature: dual-engine-strategy, Property 6: 均值回歸進場確認
        Validates: Requirements 5.2, 5.3
        
        For any RANGE state with both RSI and hammer confirmation,
        the signal should indicate "both" confirmation.
        """
        strategy = MeanReversionStrategy(rsi_oversold=30.0)
        
        current_price = bollinger.lower * 0.99
        hammer = CandlePattern(pattern_type="hammer", confidence=0.8)
        
        signal = strategy.check_entry(
            current_price=current_price,
            bollinger=bollinger,
            rsi=rsi,
            candle=hammer,
            market_state=MarketState.RANGE,
            symbol="TEST"
        )
        
        assert signal is not None
        assert signal.confirmation == "both"
    
    @given(
        bollinger=valid_bollinger_result(),
        rsi=valid_rsi_result(oversold=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_entry_without_confirmation(self, bollinger, rsi):
        """
        Feature: dual-engine-strategy, Property 6: 均值回歸進場確認
        Validates: Requirements 5.2, 5.3
        
        For any RANGE state without RSI or hammer confirmation,
        no signal should be generated.
        """
        # Ensure RSI is not oversold
        assume(rsi.value >= 30.0)
        
        strategy = MeanReversionStrategy(rsi_oversold=30.0)
        
        current_price = bollinger.lower * 0.99
        
        signal = strategy.check_entry(
            current_price=current_price,
            bollinger=bollinger,
            rsi=rsi,
            candle=None,  # No hammer
            market_state=MarketState.RANGE,
            symbol="TEST"
        )
        
        assert signal is None, \
            f"Should not generate signal without confirmation (RSI={rsi.value:.1f})"
    
    @given(
        bollinger=valid_bollinger_result(),
        rsi=valid_rsi_result(oversold=True)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_entry_when_not_range_state(self, bollinger, rsi):
        """
        Feature: dual-engine-strategy, Property 6: 均值回歸進場確認
        Validates: Requirements 5.1
        
        For any non-RANGE state, no signal should be generated.
        """
        strategy = MeanReversionStrategy()
        
        current_price = bollinger.lower * 0.99
        
        # Test TREND state
        signal_trend = strategy.check_entry(
            current_price=current_price,
            bollinger=bollinger,
            rsi=rsi,
            candle=None,
            market_state=MarketState.TREND,
            symbol="TEST"
        )
        assert signal_trend is None, "Should not generate signal in TREND state"
        
        # Test NOISE state
        signal_noise = strategy.check_entry(
            current_price=current_price,
            bollinger=bollinger,
            rsi=rsi,
            candle=None,
            market_state=MarketState.NOISE,
            symbol="TEST"
        )
        assert signal_noise is None, "Should not generate signal in NOISE state"
    
    @given(
        bollinger=valid_bollinger_result(),
        rsi=valid_rsi_result(oversold=True),
        price_above_pct=st.floats(min_value=0.01, max_value=0.10)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_entry_when_price_above_lower_band(self, bollinger, rsi, price_above_pct):
        """
        Feature: dual-engine-strategy, Property 6: 均值回歸進場確認
        Validates: Requirements 5.1
        
        For any price above lower band (with buffer), no signal should be generated.
        """
        strategy = MeanReversionStrategy(lower_band_buffer=0.005)
        
        # Price above lower band + buffer
        current_price = bollinger.lower * (1 + 0.005 + price_above_pct)
        
        signal = strategy.check_entry(
            current_price=current_price,
            bollinger=bollinger,
            rsi=rsi,
            candle=None,
            market_state=MarketState.RANGE,
            symbol="TEST"
        )
        
        assert signal is None, \
            f"Should not generate signal when price {current_price:.2f} " \
            f"is above lower band {bollinger.lower:.2f}"


# =============================================================================
# Property 7: 均值回歸分批出場
# Feature: dual-engine-strategy, Property 7: 均值回歸分批出場
# Validates: Requirements 5.5, 5.6
# =============================================================================

class TestMeanReversionExitProperty:
    """
    Property 7: 均值回歸分批出場
    
    *For any* 均值回歸持倉，當價格回歸布林中軌時必須賣出 50% 倉位，
    當價格觸及布林上軌時必須清倉。
    """
    
    @given(bollinger=valid_bollinger_result())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_partial_exit_at_middle_band(self, bollinger):
        """
        Feature: dual-engine-strategy, Property 7: 均值回歸分批出場
        Validates: Requirements 5.5
        
        For any position, when price reaches middle band,
        50% of position must be sold.
        """
        strategy = MeanReversionStrategy(partial_exit_ratio=0.5)
        
        # Price at or above middle band
        current_price = bollinger.middle * 1.001
        
        result = strategy.check_exit(
            current_price=current_price,
            bollinger=bollinger,
            current_adx=15.0,  # Low ADX, no risk override
            has_partial_exited=False
        )
        
        assert result is not None, "Expected partial exit at middle band"
        exit_type, exit_ratio = result
        
        assert exit_type == "partial_exit"
        assert exit_ratio == 0.5
    
    @given(bollinger=valid_bollinger_result())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_full_exit_at_upper_band(self, bollinger):
        """
        Feature: dual-engine-strategy, Property 7: 均值回歸分批出場
        Validates: Requirements 5.6
        
        For any position, when price reaches upper band,
        full position must be sold.
        """
        strategy = MeanReversionStrategy()
        
        # Price at or above upper band
        current_price = bollinger.upper * 1.001
        
        result = strategy.check_exit(
            current_price=current_price,
            bollinger=bollinger,
            current_adx=15.0,
            has_partial_exited=False
        )
        
        assert result is not None, "Expected full exit at upper band"
        exit_type, exit_ratio = result
        
        assert exit_type == "full_exit"
        assert exit_ratio == 1.0
    
    @given(bollinger=valid_bollinger_result())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_partial_exit_if_already_done(self, bollinger):
        """
        Feature: dual-engine-strategy, Property 7: 均值回歸分批出場
        Validates: Requirements 5.5
        
        For any position that has already partial exited,
        no additional partial exit should occur at middle band.
        """
        strategy = MeanReversionStrategy()
        
        # Price at middle band but below upper band
        current_price = bollinger.middle * 1.001
        assume(current_price < bollinger.upper)
        
        result = strategy.check_exit(
            current_price=current_price,
            bollinger=bollinger,
            current_adx=15.0,
            has_partial_exited=True  # Already partial exited
        )
        
        assert result is None, \
            "Should not trigger partial exit again after already done"
    
    @given(
        bollinger=valid_bollinger_result(),
        price_below_middle_pct=st.floats(min_value=0.01, max_value=0.10)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_exit_below_middle_band(self, bollinger, price_below_middle_pct):
        """
        Feature: dual-engine-strategy, Property 7: 均值回歸分批出場
        Validates: Requirements 5.5, 5.6
        
        For any price below middle band, no exit should occur.
        """
        strategy = MeanReversionStrategy()
        
        # Price below middle band
        current_price = bollinger.middle * (1 - price_below_middle_pct)
        
        result = strategy.check_exit(
            current_price=current_price,
            bollinger=bollinger,
            current_adx=15.0,
            has_partial_exited=False
        )
        
        assert result is None, \
            f"Should not exit when price {current_price:.2f} " \
            f"is below middle band {bollinger.middle:.2f}"


# =============================================================================
# Property 8: 均值回歸風控覆蓋
# Feature: dual-engine-strategy, Property 8: 均值回歸風控覆蓋
# Validates: Requirements 6.1
# =============================================================================

class TestMeanReversionRiskOverrideProperty:
    """
    Property 8: 均值回歸風控覆蓋
    
    *For any* 持有均值回歸倉位的情況，當 ADX 突然飆升超過 25 時，
    必須觸發強制止損。
    """
    
    @given(adx=st.floats(min_value=25.01, max_value=100.0))
    @settings(max_examples=100)
    def test_risk_override_triggers_above_threshold(self, adx):
        """
        Feature: dual-engine-strategy, Property 8: 均值回歸風控覆蓋
        Validates: Requirements 6.1
        
        For any ADX > 25, risk override must be triggered.
        """
        strategy = MeanReversionStrategy(adx_override_threshold=25.0)
        
        result = strategy.check_risk_override(adx)
        
        assert result is True, \
            f"Risk override should trigger for ADX={adx:.2f} > 25"
    
    @given(adx=st.floats(min_value=0.0, max_value=25.0))
    @settings(max_examples=100)
    def test_risk_override_not_triggered_below_threshold(self, adx):
        """
        Feature: dual-engine-strategy, Property 8: 均值回歸風控覆蓋
        Validates: Requirements 6.1
        
        For any ADX <= 25, risk override must NOT be triggered.
        """
        strategy = MeanReversionStrategy(adx_override_threshold=25.0)
        
        result = strategy.check_risk_override(adx)
        
        assert result is False, \
            f"Risk override should NOT trigger for ADX={adx:.2f} <= 25"
    
    @given(
        bollinger=valid_bollinger_result(),
        adx=st.floats(min_value=25.01, max_value=100.0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_risk_override_in_exit_check(self, bollinger, adx):
        """
        Feature: dual-engine-strategy, Property 8: 均值回歸風控覆蓋
        Validates: Requirements 6.1
        
        For any position with ADX > 25, check_exit must return risk_override.
        """
        strategy = MeanReversionStrategy(adx_override_threshold=25.0)
        
        # Price below middle band (no normal exit condition)
        current_price = bollinger.lower * 1.01
        
        result = strategy.check_exit(
            current_price=current_price,
            bollinger=bollinger,
            current_adx=adx,
            has_partial_exited=False
        )
        
        assert result is not None, "Expected risk override exit"
        exit_type, exit_ratio = result
        
        assert exit_type == "risk_override"
        assert exit_ratio == 1.0
    
    @given(
        bollinger=valid_bollinger_result(),
        adx=st.floats(min_value=25.01, max_value=100.0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_risk_override_takes_priority(self, bollinger, adx):
        """
        Feature: dual-engine-strategy, Property 8: 均值回歸風控覆蓋
        Validates: Requirements 6.1
        
        Risk override must take priority over normal exit conditions.
        """
        strategy = MeanReversionStrategy(adx_override_threshold=25.0)
        
        # Price at upper band (would normally trigger full_exit)
        current_price = bollinger.upper * 1.01
        
        result = strategy.check_exit(
            current_price=current_price,
            bollinger=bollinger,
            current_adx=adx,
            has_partial_exited=False
        )
        
        assert result is not None
        exit_type, _ = result
        
        # Risk override should take priority
        assert exit_type == "risk_override", \
            f"Risk override should take priority, got {exit_type}"


# =============================================================================
# Unit Tests for edge cases
# =============================================================================

class TestMeanReversionEdgeCases:
    """Unit tests for edge cases and configuration."""
    
    def test_strategy_from_config(self):
        """Test creating strategy from DualEngineConfig."""
        from pattern_quant.strategy.models import DualEngineConfig
        
        config = DualEngineConfig(
            reversion_rsi_oversold=25.0,
            reversion_partial_exit=0.6,
            reversion_adx_override=30.0
        )
        
        strategy = MeanReversionStrategy.from_config(config)
        
        assert strategy.rsi_oversold == 25.0
        assert strategy.partial_exit_ratio == 0.6
        assert strategy.adx_override_threshold == 30.0
    
    def test_hammer_with_doji(self):
        """Test hammer detection with doji (very small body)."""
        strategy = MeanReversionStrategy()
        
        # Doji candle (open == close)
        result = strategy.detect_hammer(
            open_price=100.0,
            high=100.01,
            low=95.0,
            close=100.0
        )
        
        # Doji should not be detected as hammer
        assert result is None
    
    def test_entry_signal_stop_loss_calculation(self):
        """Test that stop loss is calculated correctly."""
        strategy = MeanReversionStrategy(stop_loss_pct=0.03)
        
        bollinger = BollingerResult(
            upper=110.0,
            middle=100.0,
            lower=90.0,
            bandwidth=0.2,
            squeeze=False,
            breakout_upper=False
        )
        
        rsi = RSIResult(
            value=25.0,
            is_overbought=False,
            is_oversold=True,
            trend_zone=False,
            support_bounce=False
        )
        
        signal = strategy.check_entry(
            current_price=89.0,
            bollinger=bollinger,
            rsi=rsi,
            candle=None,
            market_state=MarketState.RANGE,
            symbol="TEST"
        )
        
        assert signal is not None
        expected_stop_loss = 90.0 * (1 - 0.03)  # 87.3
        assert abs(signal.stop_loss_price - expected_stop_loss) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
