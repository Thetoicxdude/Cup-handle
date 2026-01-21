"""
Property-Based Tests for Market State Classifier

This module contains property-based tests using Hypothesis to verify
the correctness of the MarketStateClassifier implementation.

Properties tested:
- Property 1: ADX 計算正確性
- Property 2: BBW 計算公式正確性
- Property 3: 市場狀態分類一致性

Validates: Requirements 1.1, 1.2, 1.3, 2.2, 2.4, 3.1, 3.2, 3.3, 3.5, 3.6, 3.7
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from pattern_quant.strategy import (
    MarketStateClassifier,
    MarketState,
    ADXResult,
    BBWResult,
    MarketStateResult,
)


# =============================================================================
# Custom Strategies for generating valid price data
# =============================================================================

@st.composite
def valid_price_series(draw, min_length=30, max_length=60):
    """
    Generate a valid OHLC price series.
    
    Ensures:
    - high >= close >= low for each bar
    - high >= open >= low for each bar
    - Prices are positive
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    base_price = draw(st.floats(min_value=50.0, max_value=200.0))
    
    highs = []
    lows = []
    closes = []
    
    current_price = base_price
    
    for _ in range(length):
        # Generate price change - use simpler generation
        change_pct = draw(st.floats(min_value=-0.02, max_value=0.02))
        volatility = draw(st.floats(min_value=0.01, max_value=0.02))
        
        # Calculate close
        close = current_price * (1 + change_pct)
        close = max(1.0, close)  # Ensure positive
        
        # High is above close, low is below close
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        low = max(0.5, low)  # Ensure low > 0
        
        highs.append(high)
        lows.append(low)
        closes.append(close)
        
        current_price = close
    
    return highs, lows, closes


@st.composite
def valid_close_prices(draw, min_length=25, max_length=60):
    """Generate a valid close price series."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    base_price = draw(st.floats(min_value=50.0, max_value=200.0))
    
    prices = []
    current = base_price
    
    for _ in range(length):
        change = draw(st.floats(min_value=-0.02, max_value=0.02))
        current = current * (1 + change)
        current = max(1.0, current)
        prices.append(current)
    
    return prices


# =============================================================================
# Property 1: ADX 計算正確性
# Feature: dual-engine-strategy, Property 1: ADX 計算正確性
# Validates: Requirements 1.1, 1.2, 1.3
# =============================================================================

class TestADXCalculationProperty:
    """
    Property 1: ADX 計算正確性
    
    *For any* 有效的價格序列（包含高、低、收盤價），計算出的 ADX 值必須在 0 到 100 之間（含），
    且同時返回有效的 +DI 和 -DI 值。
    """
    
    @given(price_data=valid_price_series(min_length=30, max_length=60))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_adx_value_range(self, price_data):
        """
        Feature: dual-engine-strategy, Property 1: ADX 計算正確性
        Validates: Requirements 1.1, 1.2, 1.3
        
        For any valid price series, ADX must be in [0, 100] and +DI/-DI must be valid.
        """
        highs, lows, closes = price_data
        classifier = MarketStateClassifier()
        
        result = classifier.calculate_adx(highs, lows, closes)
        
        if result is not None:
            # ADX must be in [0, 100]
            assert 0 <= result.adx <= 100, f"ADX {result.adx} out of range [0, 100]"
            
            # +DI must be in [0, 100]
            assert 0 <= result.plus_di <= 100, f"+DI {result.plus_di} out of range [0, 100]"
            
            # -DI must be in [0, 100]
            assert 0 <= result.minus_di <= 100, f"-DI {result.minus_di} out of range [0, 100]"
    
    @given(
        highs=st.lists(st.floats(min_value=1.0, max_value=100.0), min_size=10, max_size=13),
        lows=st.lists(st.floats(min_value=0.5, max_value=99.0), min_size=10, max_size=13),
        closes=st.lists(st.floats(min_value=0.5, max_value=100.0), min_size=10, max_size=13)
    )
    @settings(max_examples=100)
    def test_adx_insufficient_data_returns_none(self, highs, lows, closes):
        """
        Feature: dual-engine-strategy, Property 1: ADX 計算正確性
        Validates: Requirements 1.1 (data requirement)
        
        For any price series with less than 15 data points, ADX should return None.
        """
        # Ensure all lists have same length
        min_len = min(len(highs), len(lows), len(closes))
        highs = highs[:min_len]
        lows = lows[:min_len]
        closes = closes[:min_len]
        
        classifier = MarketStateClassifier(adx_period=14)
        result = classifier.calculate_adx(highs, lows, closes)
        
        # With less than period + 1 data points, should return None
        if min_len < 15:
            assert result is None, f"Expected None for {min_len} data points, got {result}"


# =============================================================================
# Property 2: BBW 計算公式正確性
# Feature: dual-engine-strategy, Property 2: BBW 計算公式正確性
# Validates: Requirements 2.2, 2.4
# =============================================================================

class TestBBWCalculationProperty:
    """
    Property 2: BBW 計算公式正確性
    
    *For any* 有效的價格序列，計算出的 BBW 必須等於 (上軌 - 下軌) / 中軌，
    且當 BBW < 歷史平均 × 0.5 時，壓縮標記必須為 True。
    """
    
    @given(prices=valid_close_prices(min_length=25, max_length=60))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_bbw_formula_correctness(self, prices):
        """
        Feature: dual-engine-strategy, Property 2: BBW 計算公式正確性
        Validates: Requirements 2.2
        
        For any valid price series, BBW = (upper - lower) / middle.
        """
        classifier = MarketStateClassifier(bb_period=20, bb_std=2.0)
        result = classifier.calculate_bbw(prices)
        
        if result is not None:
            # Manually calculate expected BBW
            recent = prices[-20:]
            middle = sum(recent) / 20
            
            if middle > 0:
                variance = sum((p - middle) ** 2 for p in recent) / 20
                std = variance ** 0.5
                upper = middle + 2.0 * std
                lower = middle - 2.0 * std
                expected_bbw = (upper - lower) / middle
                
                # BBW should match the formula
                assert abs(result.bandwidth - expected_bbw) < 1e-10, \
                    f"BBW {result.bandwidth} != expected {expected_bbw}"
    
    @given(prices=valid_close_prices(min_length=50, max_length=60))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_bbw_squeeze_detection(self, prices):
        """
        Feature: dual-engine-strategy, Property 2: BBW 計算公式正確性
        Validates: Requirements 2.4
        
        When BBW < avg_bandwidth * 0.5, is_squeeze must be True.
        """
        classifier = MarketStateClassifier(bb_period=20, bb_std=2.0)
        result = classifier.calculate_bbw(prices)
        
        if result is not None:
            # Check squeeze detection logic
            expected_squeeze = result.bandwidth < (result.avg_bandwidth * 0.5)
            assert result.is_squeeze == expected_squeeze, \
                f"Squeeze detection mismatch: bandwidth={result.bandwidth}, " \
                f"avg={result.avg_bandwidth}, is_squeeze={result.is_squeeze}, " \
                f"expected={expected_squeeze}"
    
    @given(prices=st.lists(st.floats(min_value=1.0, max_value=100.0), min_size=5, max_size=19))
    @settings(max_examples=100)
    def test_bbw_insufficient_data_returns_none(self, prices):
        """
        Feature: dual-engine-strategy, Property 2: BBW 計算公式正確性
        Validates: Requirements 2.1 (data requirement)
        
        For any price series with less than bb_period data points, BBW should return None.
        """
        classifier = MarketStateClassifier(bb_period=20)
        result = classifier.calculate_bbw(prices)
        
        assert result is None, f"Expected None for {len(prices)} data points, got {result}"


# =============================================================================
# Property 3: 市場狀態分類一致性
# Feature: dual-engine-strategy, Property 3: 市場狀態分類一致性
# Validates: Requirements 3.1, 3.2, 3.3, 3.5, 3.6, 3.7
# =============================================================================

class TestMarketStateClassificationProperty:
    """
    Property 3: 市場狀態分類一致性
    
    *For any* ADX 值和 BBW 結果，市場狀態判定必須符合以下規則：
    - ADX > 25 → Trend 狀態，資金權重 100%
    - ADX < 20 且 BBW 穩定 → Range 狀態，資金權重 60%
    - 20 ≤ ADX ≤ 25 → Noise 狀態，資金權重 0%
    """
    
    @given(price_data=valid_price_series(min_length=50, max_length=60))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_market_state_classification_consistency(self, price_data):
        """
        Feature: dual-engine-strategy, Property 3: 市場狀態分類一致性
        Validates: Requirements 3.1, 3.2, 3.3, 3.5, 3.6, 3.7
        
        Market state classification must be consistent with ADX and BBW values.
        """
        highs, lows, closes = price_data
        classifier = MarketStateClassifier(
            trend_threshold=25.0,
            range_threshold=20.0,
            bbw_stability_threshold=0.10
        )
        
        result = classifier.classify(highs, lows, closes)
        
        adx = result.adx_result.adx
        bbw_stable = result.bbw_result.change_rate < 0.10
        
        # Verify state classification rules
        if adx > 25.0:
            # Requirement 3.1: ADX > 25 → Trend
            assert result.state == MarketState.TREND, \
                f"ADX={adx} > 25 should be TREND, got {result.state}"
            # Requirement 3.5: Trend → 100% allocation
            assert result.allocation_weight == 1.0, \
                f"TREND should have 100% allocation, got {result.allocation_weight}"
        
        elif adx < 20.0 and bbw_stable:
            # Requirement 3.2: ADX < 20 and BBW stable → Range
            assert result.state == MarketState.RANGE, \
                f"ADX={adx} < 20 and BBW stable should be RANGE, got {result.state}"
            # Requirement 3.6: Range → 60% allocation
            assert result.allocation_weight == 0.6, \
                f"RANGE should have 60% allocation, got {result.allocation_weight}"
        
        elif 20.0 <= adx <= 25.0:
            # Requirement 3.3: 20 ≤ ADX ≤ 25 → Noise
            assert result.state == MarketState.NOISE, \
                f"ADX={adx} in [20, 25] should be NOISE, got {result.state}"
            # Requirement 3.7: Noise → 0% allocation
            assert result.allocation_weight == 0.0, \
                f"NOISE should have 0% allocation, got {result.allocation_weight}"
        
        elif adx < 20.0 and not bbw_stable:
            # ADX < 20 but BBW not stable → Noise
            assert result.state == MarketState.NOISE, \
                f"ADX={adx} < 20 but BBW unstable should be NOISE, got {result.state}"
            assert result.allocation_weight == 0.0, \
                f"NOISE should have 0% allocation, got {result.allocation_weight}"
    
    @given(price_data=valid_price_series(min_length=50, max_length=60))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_allocation_weight_values(self, price_data):
        """
        Feature: dual-engine-strategy, Property 3: 市場狀態分類一致性
        Validates: Requirements 3.5, 3.6, 3.7
        
        Allocation weights must be exactly 1.0, 0.6, or 0.0 based on state.
        """
        highs, lows, closes = price_data
        classifier = MarketStateClassifier()
        
        result = classifier.classify(highs, lows, closes)
        
        # Verify allocation weight matches state
        if result.state == MarketState.TREND:
            assert result.allocation_weight == 1.0
        elif result.state == MarketState.RANGE:
            assert result.allocation_weight == 0.6
        elif result.state == MarketState.NOISE:
            assert result.allocation_weight == 0.0
    
    @given(price_data=valid_price_series(min_length=50, max_length=60))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_result_contains_all_required_fields(self, price_data):
        """
        Feature: dual-engine-strategy, Property 3: 市場狀態分類一致性
        Validates: Requirements 3.4
        
        MarketStateResult must contain all required fields.
        """
        highs, lows, closes = price_data
        classifier = MarketStateClassifier()
        
        result = classifier.classify(highs, lows, closes)
        
        # Verify result structure
        assert isinstance(result, MarketStateResult)
        assert isinstance(result.state, MarketState)
        assert isinstance(result.allocation_weight, float)
        assert isinstance(result.adx_result, ADXResult)
        assert isinstance(result.bbw_result, BBWResult)
        assert isinstance(result.confidence, float)
        
        # Confidence should be in [0, 1]
        assert 0 <= result.confidence <= 1.0


# =============================================================================
# Unit Tests for edge cases
# =============================================================================

class TestMarketClassifierEdgeCases:
    """Unit tests for edge cases and specific scenarios."""
    
    def test_classifier_from_config(self):
        """Test creating classifier from DualEngineConfig."""
        from pattern_quant.strategy import DualEngineConfig
        
        config = DualEngineConfig(
            adx_trend_threshold=30.0,
            adx_range_threshold=15.0,
            bbw_stability_threshold=0.15
        )
        
        classifier = MarketStateClassifier.from_config(config)
        
        assert classifier.trend_threshold == 30.0
        assert classifier.range_threshold == 15.0
        assert classifier.bbw_stability_threshold == 0.15
    
    def test_constant_prices_bbw(self):
        """Test BBW with constant prices (zero volatility)."""
        classifier = MarketStateClassifier(bb_period=20)
        prices = [100.0] * 30
        
        result = classifier.calculate_bbw(prices)
        
        assert result is not None
        # With constant prices, std = 0, so BBW = 0
        assert result.bandwidth == 0.0
    
    def test_strong_uptrend_adx(self):
        """Test ADX with strong uptrend."""
        classifier = MarketStateClassifier(adx_period=14)
        
        # Strong uptrend
        closes = [100.0 + i * 2 for i in range(50)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        
        result = classifier.calculate_adx(highs, lows, closes)
        
        assert result is not None
        # Strong trend should have high ADX
        assert result.adx > 0
        # Uptrend should have +DI > -DI
        assert result.plus_di > result.minus_di


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
