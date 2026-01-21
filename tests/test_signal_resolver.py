"""
Property-Based Tests for Signal Resolver

This module contains property-based tests using Hypothesis to verify
the correctness of the SignalResolver implementation.

Properties tested:
- Property 11: 信號衝突解決

Validates: Requirements 9.1, 9.2, 9.3
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from pattern_quant.strategy.signal_resolver import (
    SignalResolver,
    ConflictType,
    ResolvedSignal,
)
from pattern_quant.strategy.models import (
    TrendSignal,
    MeanReversionSignal,
)


# =============================================================================
# Custom Strategies for generating valid signals
# =============================================================================

@st.composite
def valid_trend_signal(draw):
    """Generate a valid TrendSignal."""
    symbol = draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=5))
    entry_price = draw(st.floats(min_value=10.0, max_value=500.0))
    stop_loss_price = entry_price * draw(st.floats(min_value=0.90, max_value=0.98))
    neckline_price = entry_price * draw(st.floats(min_value=0.95, max_value=1.0))
    pattern_score = draw(st.floats(min_value=80.0, max_value=100.0))
    risk_reward_ratio = draw(st.floats(min_value=2.0, max_value=5.0))
    
    return TrendSignal(
        symbol=symbol,
        signal_type="breakout_long",
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        neckline_price=neckline_price,
        pattern_score=pattern_score,
        risk_reward_ratio=risk_reward_ratio
    )


@st.composite
def valid_reversion_signal(draw):
    """Generate a valid MeanReversionSignal."""
    symbol = draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=5))
    entry_price = draw(st.floats(min_value=10.0, max_value=500.0))
    target_1 = entry_price * draw(st.floats(min_value=1.02, max_value=1.05))
    target_2 = target_1 * draw(st.floats(min_value=1.02, max_value=1.05))
    stop_loss_price = entry_price * draw(st.floats(min_value=0.95, max_value=0.99))
    confirmation = draw(st.sampled_from(["rsi_oversold", "hammer", "both"]))
    
    return MeanReversionSignal(
        symbol=symbol,
        signal_type="long_entry",
        entry_price=entry_price,
        target_1=target_1,
        target_2=target_2,
        stop_loss_price=stop_loss_price,
        confirmation=confirmation
    )


# =============================================================================
# Property 11: 信號衝突解決
# Feature: dual-engine-strategy, Property 11: 信號衝突解決
# Validates: Requirements 9.1, 9.2, 9.3
# =============================================================================

class TestSignalConflictResolutionProperty:
    """
    Property 11: 信號衝突解決
    
    *For any* 同時存在趨勢信號和回歸信號的情況，必須執行「趨勢優先」原則。
    當價格跌到布林下軌但 ADX 暴衝時，必須判定為「向下突破」並禁止回歸買入。
    """
    
    @given(
        trend_signal=valid_trend_signal(),
        reversion_signal=valid_reversion_signal(),
        current_adx=st.floats(min_value=0.0, max_value=100.0),
        adx_change_rate=st.floats(min_value=-0.5, max_value=0.5)
    )
    @settings(max_examples=100)
    def test_trend_trumps_reversion(
        self, trend_signal, reversion_signal, current_adx, adx_change_rate
    ):
        """
        Feature: dual-engine-strategy, Property 11: 信號衝突解決
        Validates: Requirements 9.1
        
        For any situation where both trend and reversion signals exist,
        the trend signal must be prioritized (Trend Trumps Range).
        """
        resolver = SignalResolver()
        
        result = resolver.resolve(
            trend_signal=trend_signal,
            reversion_signal=reversion_signal,
            current_adx=current_adx,
            adx_change_rate=adx_change_rate
        )
        
        # When both signals exist, trend must be prioritized
        assert result.signal == trend_signal, \
            "When both signals exist, trend signal must be prioritized"
        assert result.strategy_type == "trend", \
            "Strategy type must be 'trend' when trend signal is prioritized"
        assert result.conflict_type == ConflictType.TREND_VS_REVERSION, \
            "Conflict type must be TREND_VS_REVERSION"
    
    @given(
        reversion_signal=valid_reversion_signal(),
        current_adx=st.floats(min_value=26.0, max_value=100.0),
        adx_change_rate=st.floats(min_value=0.21, max_value=0.5)
    )
    @settings(max_examples=100)
    def test_downward_breakout_blocks_reversion(
        self, reversion_signal, current_adx, adx_change_rate
    ):
        """
        Feature: dual-engine-strategy, Property 11: 信號衝突解決
        Validates: Requirements 9.2, 9.3
        
        When price hits lower band but ADX is surging (downward breakout),
        reversion buy must be blocked.
        """
        resolver = SignalResolver(
            adx_surge_threshold=25.0,
            adx_change_rate_threshold=0.20
        )
        
        result = resolver.resolve(
            trend_signal=None,
            reversion_signal=reversion_signal,
            current_adx=current_adx,
            adx_change_rate=adx_change_rate
        )
        
        # Downward breakout must block reversion signal
        assert result.signal is None, \
            f"Downward breakout (ADX={current_adx}, rate={adx_change_rate}) must block reversion"
        assert result.strategy_type == "none", \
            "Strategy type must be 'none' when reversion is blocked"
        assert result.conflict_type == ConflictType.DOWNWARD_BREAKOUT, \
            "Conflict type must be DOWNWARD_BREAKOUT"
    
    @given(
        reversion_signal=valid_reversion_signal(),
        current_adx=st.floats(min_value=0.0, max_value=24.9),
        adx_change_rate=st.floats(min_value=-0.5, max_value=0.19)
    )
    @settings(max_examples=100)
    def test_reversion_allowed_when_no_breakout(
        self, reversion_signal, current_adx, adx_change_rate
    ):
        """
        Feature: dual-engine-strategy, Property 11: 信號衝突解決
        Validates: Requirements 9.2, 9.3
        
        When ADX is not surging (no downward breakout), reversion signal
        should be allowed.
        """
        resolver = SignalResolver(
            adx_surge_threshold=25.0,
            adx_change_rate_threshold=0.20
        )
        
        result = resolver.resolve(
            trend_signal=None,
            reversion_signal=reversion_signal,
            current_adx=current_adx,
            adx_change_rate=adx_change_rate
        )
        
        # No breakout, reversion should be allowed
        assert result.signal == reversion_signal, \
            f"Without breakout (ADX={current_adx}, rate={adx_change_rate}), reversion should be allowed"
        assert result.strategy_type == "mean_reversion", \
            "Strategy type must be 'mean_reversion'"
        assert result.conflict_type == ConflictType.NONE, \
            "Conflict type must be NONE"
    
    @given(
        trend_signal=valid_trend_signal(),
        current_adx=st.floats(min_value=0.0, max_value=100.0),
        adx_change_rate=st.floats(min_value=-0.5, max_value=0.5)
    )
    @settings(max_examples=100)
    def test_trend_only_signal_adopted(
        self, trend_signal, current_adx, adx_change_rate
    ):
        """
        Feature: dual-engine-strategy, Property 11: 信號衝突解決
        Validates: Requirements 9.1
        
        When only trend signal exists, it should be adopted directly.
        """
        resolver = SignalResolver()
        
        result = resolver.resolve(
            trend_signal=trend_signal,
            reversion_signal=None,
            current_adx=current_adx,
            adx_change_rate=adx_change_rate
        )
        
        assert result.signal == trend_signal, \
            "When only trend signal exists, it must be adopted"
        assert result.strategy_type == "trend", \
            "Strategy type must be 'trend'"
        assert result.conflict_type == ConflictType.NONE, \
            "Conflict type must be NONE when no conflict"
    
    @given(
        current_adx=st.floats(min_value=0.0, max_value=100.0),
        adx_change_rate=st.floats(min_value=-0.5, max_value=0.5)
    )
    @settings(max_examples=100)
    def test_no_signal_returns_none(self, current_adx, adx_change_rate):
        """
        Feature: dual-engine-strategy, Property 11: 信號衝突解決
        Validates: Requirements 9.1
        
        When no signals exist, result should be None.
        """
        resolver = SignalResolver()
        
        result = resolver.resolve(
            trend_signal=None,
            reversion_signal=None,
            current_adx=current_adx,
            adx_change_rate=adx_change_rate
        )
        
        assert result.signal is None, \
            "When no signals exist, result signal must be None"
        assert result.strategy_type == "none", \
            "Strategy type must be 'none'"
        assert result.conflict_type == ConflictType.NONE, \
            "Conflict type must be NONE"


# =============================================================================
# Unit Tests for edge cases
# =============================================================================

class TestSignalResolverEdgeCases:
    """Unit tests for edge cases and specific scenarios."""
    
    def test_downward_breakout_boundary_adx(self):
        """Test downward breakout at exact ADX threshold boundary."""
        resolver = SignalResolver(
            adx_surge_threshold=25.0,
            adx_change_rate_threshold=0.20
        )
        
        reversion_signal = MeanReversionSignal(
            symbol="TEST",
            signal_type="long_entry",
            entry_price=100.0,
            target_1=105.0,
            target_2=110.0,
            stop_loss_price=97.0,
            confirmation="rsi_oversold"
        )
        
        # Exactly at threshold - should NOT be blocked (need to exceed)
        result = resolver.resolve(
            trend_signal=None,
            reversion_signal=reversion_signal,
            current_adx=25.0,  # Exactly at threshold
            adx_change_rate=0.20  # Exactly at threshold
        )
        
        # At exact threshold, should allow (need to exceed)
        assert result.signal == reversion_signal
        assert result.conflict_type == ConflictType.NONE
    
    def test_downward_breakout_just_above_threshold(self):
        """Test downward breakout just above threshold."""
        resolver = SignalResolver(
            adx_surge_threshold=25.0,
            adx_change_rate_threshold=0.20
        )
        
        reversion_signal = MeanReversionSignal(
            symbol="TEST",
            signal_type="long_entry",
            entry_price=100.0,
            target_1=105.0,
            target_2=110.0,
            stop_loss_price=97.0,
            confirmation="hammer"
        )
        
        # Just above threshold - should be blocked
        result = resolver.resolve(
            trend_signal=None,
            reversion_signal=reversion_signal,
            current_adx=25.1,  # Just above threshold
            adx_change_rate=0.21  # Just above threshold
        )
        
        assert result.signal is None
        assert result.conflict_type == ConflictType.DOWNWARD_BREAKOUT
    
    def test_high_adx_but_low_change_rate(self):
        """Test high ADX but low change rate - not a breakout."""
        resolver = SignalResolver(
            adx_surge_threshold=25.0,
            adx_change_rate_threshold=0.20
        )
        
        reversion_signal = MeanReversionSignal(
            symbol="TEST",
            signal_type="long_entry",
            entry_price=100.0,
            target_1=105.0,
            target_2=110.0,
            stop_loss_price=97.0,
            confirmation="both"
        )
        
        # High ADX but low change rate - not a breakout
        result = resolver.resolve(
            trend_signal=None,
            reversion_signal=reversion_signal,
            current_adx=30.0,  # High ADX
            adx_change_rate=0.10  # Low change rate
        )
        
        # Should allow reversion (ADX is high but not surging)
        assert result.signal == reversion_signal
        assert result.conflict_type == ConflictType.NONE
    
    def test_low_adx_but_high_change_rate(self):
        """Test low ADX but high change rate - not a breakout."""
        resolver = SignalResolver(
            adx_surge_threshold=25.0,
            adx_change_rate_threshold=0.20
        )
        
        reversion_signal = MeanReversionSignal(
            symbol="TEST",
            signal_type="long_entry",
            entry_price=100.0,
            target_1=105.0,
            target_2=110.0,
            stop_loss_price=97.0,
            confirmation="rsi_oversold"
        )
        
        # Low ADX but high change rate - not a breakout yet
        result = resolver.resolve(
            trend_signal=None,
            reversion_signal=reversion_signal,
            current_adx=20.0,  # Low ADX
            adx_change_rate=0.30  # High change rate
        )
        
        # Should allow reversion (ADX not yet at surge level)
        assert result.signal == reversion_signal
        assert result.conflict_type == ConflictType.NONE
    
    def test_resolution_reason_contains_details(self):
        """Test that resolution reason contains useful details."""
        resolver = SignalResolver()
        
        trend_signal = TrendSignal(
            symbol="AAPL",
            signal_type="breakout_long",
            entry_price=150.0,
            stop_loss_price=145.0,
            neckline_price=148.0,
            pattern_score=85.0,
            risk_reward_ratio=3.0
        )
        
        reversion_signal = MeanReversionSignal(
            symbol="AAPL",
            signal_type="long_entry",
            entry_price=145.0,
            target_1=150.0,
            target_2=155.0,
            stop_loss_price=142.0,
            confirmation="rsi_oversold"
        )
        
        result = resolver.resolve(
            trend_signal=trend_signal,
            reversion_signal=reversion_signal,
            current_adx=30.0,
            adx_change_rate=0.15
        )
        
        assert "趨勢優先" in result.resolution_reason or "trend" in result.resolution_reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
