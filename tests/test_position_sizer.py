"""
Property-Based Tests for Position Sizer

This module contains property-based tests using Hypothesis to verify
the correctness of the ATRPositionSizer and FixedFractionalSizer implementations.

Properties tested:
- Property 9: 趨勢策略倉位計算
- Property 10: 震盪策略倉位計算

Validates: Requirements 7.1, 7.2, 7.3, 8.1, 8.2, 8.3
"""

import pytest
import math
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from pattern_quant.strategy.position_sizer import (
    ATRPositionSizer,
    FixedFractionalSizer,
    PositionSizeResult,
)


# =============================================================================
# Property 9: 趨勢策略倉位計算
# Feature: dual-engine-strategy, Property 9: 趨勢策略倉位計算
# Validates: Requirements 7.1, 7.2, 7.3
# =============================================================================

class TestATRPositionSizerProperty:
    """
    Property 9: 趨勢策略倉位計算
    
    *For any* 總資金、進場價格和 ATR 值，計算出的倉位必須確保單筆交易風險
    不超過總資金的 1%。
    """
    
    @given(
        total_capital=st.floats(min_value=10000.0, max_value=10000000.0),
        entry_price=st.floats(min_value=1.0, max_value=1000.0),
        atr=st.floats(min_value=0.1, max_value=50.0),
        atr_multiplier=st.floats(min_value=1.0, max_value=5.0),
        risk_per_trade=st.floats(min_value=0.001, max_value=0.05)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_atr_position_risk_constraint(
        self, total_capital, entry_price, atr, atr_multiplier, risk_per_trade
    ):
        """
        Feature: dual-engine-strategy, Property 9: 趨勢策略倉位計算
        Validates: Requirements 7.1, 7.2, 7.3
        
        For any valid inputs, the calculated position risk must not exceed
        the configured risk_per_trade ratio of total capital.
        """
        sizer = ATRPositionSizer(risk_per_trade=risk_per_trade)
        result = sizer.calculate(total_capital, entry_price, atr, atr_multiplier)
        
        # Maximum allowed risk
        max_risk = total_capital * risk_per_trade
        
        # Actual risk should not exceed max risk
        # Allow small tolerance for floating point
        assert result.risk_amount <= max_risk + 1e-6, \
            f"Risk {result.risk_amount} exceeds max {max_risk}"
        
        # Verify result structure
        assert isinstance(result, PositionSizeResult)
        assert result.sizing_method == "atr_volatility"
        assert result.shares >= 0
        assert result.position_value >= 0
        assert result.risk_amount >= 0
    
    @given(
        total_capital=st.floats(min_value=10000.0, max_value=10000000.0),
        entry_price=st.floats(min_value=1.0, max_value=1000.0),
        atr=st.floats(min_value=0.1, max_value=50.0),
        atr_multiplier=st.floats(min_value=1.0, max_value=5.0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_atr_position_default_1_percent_risk(
        self, total_capital, entry_price, atr, atr_multiplier
    ):
        """
        Feature: dual-engine-strategy, Property 9: 趨勢策略倉位計算
        Validates: Requirements 7.2
        
        With default settings, risk must not exceed 1% of total capital.
        """
        sizer = ATRPositionSizer()  # Default 1% risk
        result = sizer.calculate(total_capital, entry_price, atr, atr_multiplier)
        
        # Maximum allowed risk (1%)
        max_risk = total_capital * 0.01
        
        # Actual risk should not exceed 1%
        assert result.risk_amount <= max_risk + 1e-6, \
            f"Risk {result.risk_amount} exceeds 1% max {max_risk}"
    
    @given(
        total_capital=st.floats(min_value=10000.0, max_value=10000000.0),
        entry_price=st.floats(min_value=1.0, max_value=1000.0),
        atr=st.floats(min_value=0.1, max_value=50.0),
        atr_multiplier=st.floats(min_value=1.0, max_value=5.0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_atr_position_formula_correctness(
        self, total_capital, entry_price, atr, atr_multiplier
    ):
        """
        Feature: dual-engine-strategy, Property 9: 趨勢策略倉位計算
        Validates: Requirements 7.1
        
        Position size should follow the formula:
        shares = floor((capital * risk_ratio) / (atr * multiplier))
        """
        risk_per_trade = 0.01
        sizer = ATRPositionSizer(risk_per_trade=risk_per_trade)
        result = sizer.calculate(total_capital, entry_price, atr, atr_multiplier)
        
        # Calculate expected shares
        max_risk = total_capital * risk_per_trade
        risk_per_share = atr * atr_multiplier
        expected_shares = math.floor(max_risk / risk_per_share)
        expected_shares = max(0, expected_shares)
        
        assert result.shares == expected_shares, \
            f"Shares {result.shares} != expected {expected_shares}"
        
        # Verify position value
        expected_value = expected_shares * entry_price
        assert abs(result.position_value - expected_value) < 1e-6, \
            f"Position value {result.position_value} != expected {expected_value}"
    
    @given(
        total_capital=st.floats(min_value=10000.0, max_value=10000000.0),
        entry_price=st.floats(min_value=1.0, max_value=1000.0),
        atr=st.floats(min_value=0.1, max_value=50.0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_atr_position_returns_shares_and_risk(
        self, total_capital, entry_price, atr
    ):
        """
        Feature: dual-engine-strategy, Property 9: 趨勢策略倉位計算
        Validates: Requirements 7.3
        
        Result must include recommended shares and corresponding risk amount.
        """
        sizer = ATRPositionSizer()
        result = sizer.calculate(total_capital, entry_price, atr)
        
        # Must return shares
        assert isinstance(result.shares, int)
        assert result.shares >= 0
        
        # Must return risk amount
        assert isinstance(result.risk_amount, float)
        assert result.risk_amount >= 0
        
        # Risk amount should be consistent with shares and ATR
        if result.shares > 0:
            expected_risk = result.shares * atr * 2.0  # default multiplier
            assert abs(result.risk_amount - expected_risk) < 1e-6


# =============================================================================
# Property 10: 震盪策略倉位計算
# Feature: dual-engine-strategy, Property 10: 震盪策略倉位計算
# Validates: Requirements 8.1, 8.2, 8.3
# =============================================================================

class TestFixedFractionalSizerProperty:
    """
    Property 10: 震盪策略倉位計算
    
    *For any* 總資金和進場價格，計算出的倉位市值必須等於總資金的 5%（或配置的比例）。
    """
    
    @given(
        total_capital=st.floats(min_value=10000.0, max_value=10000000.0),
        entry_price=st.floats(min_value=1.0, max_value=1000.0),
        position_ratio=st.floats(min_value=0.01, max_value=0.20)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_fixed_fractional_position_ratio(
        self, total_capital, entry_price, position_ratio
    ):
        """
        Feature: dual-engine-strategy, Property 10: 震盪策略倉位計算
        Validates: Requirements 8.1, 8.2
        
        For any valid inputs, the position value should be approximately
        equal to total_capital * position_ratio (within one share tolerance).
        """
        sizer = FixedFractionalSizer(position_ratio=position_ratio)
        result = sizer.calculate(total_capital, entry_price)
        
        # Target position value
        target_value = total_capital * position_ratio
        
        # Due to floor operation, actual value may be less than target
        # but should not exceed target
        assert result.position_value <= target_value + 1e-6, \
            f"Position value {result.position_value} exceeds target {target_value}"
        
        # Should be within one share of target
        one_share_tolerance = entry_price
        assert result.position_value >= target_value - one_share_tolerance, \
            f"Position value {result.position_value} too far from target {target_value}"
        
        # Verify result structure
        assert isinstance(result, PositionSizeResult)
        assert result.sizing_method == "fixed_fractional"
        assert result.shares >= 0
        assert result.position_value >= 0
    
    @given(
        total_capital=st.floats(min_value=10000.0, max_value=10000000.0),
        entry_price=st.floats(min_value=1.0, max_value=1000.0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_fixed_fractional_default_5_percent(
        self, total_capital, entry_price
    ):
        """
        Feature: dual-engine-strategy, Property 10: 震盪策略倉位計算
        Validates: Requirements 8.2
        
        With default settings, position value should be approximately 5% of capital.
        """
        sizer = FixedFractionalSizer()  # Default 5%
        result = sizer.calculate(total_capital, entry_price)
        
        # Target position value (5%)
        target_value = total_capital * 0.05
        
        # Should not exceed target
        assert result.position_value <= target_value + 1e-6, \
            f"Position value {result.position_value} exceeds 5% target {target_value}"
        
        # Should be within one share of target
        one_share_tolerance = entry_price
        assert result.position_value >= target_value - one_share_tolerance, \
            f"Position value {result.position_value} too far from 5% target {target_value}"
    
    @given(
        total_capital=st.floats(min_value=10000.0, max_value=10000000.0),
        entry_price=st.floats(min_value=1.0, max_value=1000.0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_fixed_fractional_formula_correctness(
        self, total_capital, entry_price
    ):
        """
        Feature: dual-engine-strategy, Property 10: 震盪策略倉位計算
        Validates: Requirements 8.1
        
        Position size should follow the formula:
        shares = floor((capital * position_ratio) / entry_price)
        """
        position_ratio = 0.05
        sizer = FixedFractionalSizer(position_ratio=position_ratio)
        result = sizer.calculate(total_capital, entry_price)
        
        # Calculate expected shares
        target_value = total_capital * position_ratio
        expected_shares = math.floor(target_value / entry_price)
        expected_shares = max(0, expected_shares)
        
        assert result.shares == expected_shares, \
            f"Shares {result.shares} != expected {expected_shares}"
        
        # Verify position value
        expected_value = expected_shares * entry_price
        assert abs(result.position_value - expected_value) < 1e-6, \
            f"Position value {result.position_value} != expected {expected_value}"
    
    @given(
        total_capital=st.floats(min_value=10000.0, max_value=10000000.0),
        entry_price=st.floats(min_value=1.0, max_value=1000.0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_fixed_fractional_returns_shares_and_amount(
        self, total_capital, entry_price
    ):
        """
        Feature: dual-engine-strategy, Property 10: 震盪策略倉位計算
        Validates: Requirements 8.3
        
        Result must include recommended shares and corresponding investment amount.
        """
        sizer = FixedFractionalSizer()
        result = sizer.calculate(total_capital, entry_price)
        
        # Must return shares
        assert isinstance(result.shares, int)
        assert result.shares >= 0
        
        # Must return position value (investment amount)
        assert isinstance(result.position_value, float)
        assert result.position_value >= 0
        
        # Position value should equal shares * entry_price
        if result.shares > 0:
            expected_value = result.shares * entry_price
            assert abs(result.position_value - expected_value) < 1e-6


# =============================================================================
# Unit Tests for edge cases and validation
# =============================================================================

class TestPositionSizerEdgeCases:
    """Unit tests for edge cases and input validation."""
    
    def test_atr_sizer_invalid_risk_ratio(self):
        """Test ATRPositionSizer rejects invalid risk ratio."""
        with pytest.raises(ValueError):
            ATRPositionSizer(risk_per_trade=0)
        
        with pytest.raises(ValueError):
            ATRPositionSizer(risk_per_trade=-0.01)
        
        with pytest.raises(ValueError):
            ATRPositionSizer(risk_per_trade=1.5)
    
    def test_fixed_fractional_invalid_position_ratio(self):
        """Test FixedFractionalSizer rejects invalid position ratio."""
        with pytest.raises(ValueError):
            FixedFractionalSizer(position_ratio=0)
        
        with pytest.raises(ValueError):
            FixedFractionalSizer(position_ratio=-0.05)
        
        with pytest.raises(ValueError):
            FixedFractionalSizer(position_ratio=1.5)
    
    def test_atr_sizer_zero_capital(self):
        """Test ATRPositionSizer with zero capital."""
        sizer = ATRPositionSizer()
        result = sizer.calculate(0, 100.0, 5.0)
        
        assert result.shares == 0
        assert result.position_value == 0.0
        assert result.risk_amount == 0.0
    
    def test_atr_sizer_zero_entry_price(self):
        """Test ATRPositionSizer with zero entry price."""
        sizer = ATRPositionSizer()
        result = sizer.calculate(100000.0, 0, 5.0)
        
        assert result.shares == 0
        assert result.position_value == 0.0
        assert result.risk_amount == 0.0
    
    def test_atr_sizer_zero_atr(self):
        """Test ATRPositionSizer with zero ATR."""
        sizer = ATRPositionSizer()
        result = sizer.calculate(100000.0, 100.0, 0)
        
        assert result.shares == 0
        assert result.position_value == 0.0
        assert result.risk_amount == 0.0
    
    def test_fixed_fractional_zero_capital(self):
        """Test FixedFractionalSizer with zero capital."""
        sizer = FixedFractionalSizer()
        result = sizer.calculate(0, 100.0)
        
        assert result.shares == 0
        assert result.position_value == 0.0
    
    def test_fixed_fractional_zero_entry_price(self):
        """Test FixedFractionalSizer with zero entry price."""
        sizer = FixedFractionalSizer()
        result = sizer.calculate(100000.0, 0)
        
        assert result.shares == 0
        assert result.position_value == 0.0
    
    def test_atr_sizer_negative_inputs(self):
        """Test ATRPositionSizer with negative inputs."""
        sizer = ATRPositionSizer()
        
        # Negative capital
        result = sizer.calculate(-100000.0, 100.0, 5.0)
        assert result.shares == 0
        
        # Negative entry price
        result = sizer.calculate(100000.0, -100.0, 5.0)
        assert result.shares == 0
        
        # Negative ATR
        result = sizer.calculate(100000.0, 100.0, -5.0)
        assert result.shares == 0
    
    def test_fixed_fractional_negative_inputs(self):
        """Test FixedFractionalSizer with negative inputs."""
        sizer = FixedFractionalSizer()
        
        # Negative capital
        result = sizer.calculate(-100000.0, 100.0)
        assert result.shares == 0
        
        # Negative entry price
        result = sizer.calculate(100000.0, -100.0)
        assert result.shares == 0
    
    def test_atr_sizer_specific_example(self):
        """Test ATRPositionSizer with specific values."""
        # Capital: 100,000, Entry: 50, ATR: 2, Multiplier: 2
        # Max risk: 100,000 * 0.01 = 1,000
        # Risk per share: 2 * 2 = 4
        # Shares: floor(1000 / 4) = 250
        sizer = ATRPositionSizer(risk_per_trade=0.01)
        result = sizer.calculate(100000.0, 50.0, 2.0, 2.0)
        
        assert result.shares == 250
        assert result.position_value == 250 * 50.0
        assert result.risk_amount == 250 * 4.0
    
    def test_fixed_fractional_specific_example(self):
        """Test FixedFractionalSizer with specific values."""
        # Capital: 100,000, Entry: 50, Ratio: 5%
        # Target value: 100,000 * 0.05 = 5,000
        # Shares: floor(5000 / 50) = 100
        sizer = FixedFractionalSizer(position_ratio=0.05)
        result = sizer.calculate(100000.0, 50.0)
        
        assert result.shares == 100
        assert result.position_value == 100 * 50.0
        assert result.risk_amount == 100 * 50.0  # For fixed fractional, risk = position value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
