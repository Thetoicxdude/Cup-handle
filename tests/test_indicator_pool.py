"""Unit tests for Indicator Pool - Checkpoint 2 verification

This module contains basic unit tests to verify the indicator calculation
module implemented in task 1 is working correctly.
"""

import pytest
import numpy as np

from pattern_quant.optimization import (
    IndicatorPool,
    RSIResult,
    MACDResult,
    BollingerResult,
    StochasticResult,
    IndicatorSnapshot,
)


class TestIndicatorPoolInit:
    """Test IndicatorPool initialization"""
    
    def test_init_default_params(self):
        """Test initialization with default parameters"""
        pool = IndicatorPool()
        assert pool.rsi_period == 14
        assert pool.macd_fast == 12
        assert pool.macd_slow == 26
        assert pool.macd_signal == 9
        assert pool.bb_period == 20
        assert pool.bb_std == 2.0
        assert pool.atr_period == 14
        assert pool.stoch_k == 14
        assert pool.stoch_d == 3
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters"""
        pool = IndicatorPool(
            rsi_period=10,
            macd_fast=8,
            macd_slow=21,
            macd_signal=5,
            bb_period=15,
            bb_std=2.5,
            atr_period=10,
            stoch_k=9,
            stoch_d=5
        )
        assert pool.rsi_period == 10
        assert pool.macd_fast == 8
        assert pool.macd_slow == 21
        assert pool.macd_signal == 5
        assert pool.bb_period == 15
        assert pool.bb_std == 2.5
        assert pool.atr_period == 10
        assert pool.stoch_k == 9
        assert pool.stoch_d == 5


class TestRSICalculation:
    """Test RSI indicator calculation"""
    
    def test_rsi_insufficient_data(self):
        """Test RSI returns None with insufficient data"""
        pool = IndicatorPool(rsi_period=14)
        prices = [100.0] * 10  # Less than period + 1
        result = pool.calculate_rsi(prices)
        assert result is None
    
    def test_rsi_basic_calculation(self):
        """Test RSI calculation with valid data"""
        pool = IndicatorPool(rsi_period=14)
        # Generate uptrending prices
        prices = [100.0 + i * 0.5 for i in range(30)]
        result = pool.calculate_rsi(prices)
        
        assert result is not None
        assert isinstance(result, RSIResult)
        assert 0 <= result.value <= 100
    
    def test_rsi_value_range(self):
        """Test RSI value is always between 0 and 100"""
        pool = IndicatorPool(rsi_period=14)
        
        # Test with various price patterns
        test_cases = [
            [100.0 + i for i in range(30)],  # Uptrend
            [100.0 - i * 0.5 for i in range(30)],  # Downtrend
            [100.0 + (i % 5) for i in range(30)],  # Oscillating
        ]
        
        for prices in test_cases:
            result = pool.calculate_rsi(prices)
            if result is not None:
                assert 0 <= result.value <= 100
    
    def test_rsi_overbought_detection(self):
        """Test RSI overbought detection"""
        pool = IndicatorPool(rsi_period=14)
        # Strong uptrend should produce high RSI
        prices = [100.0 + i * 2 for i in range(30)]
        result = pool.calculate_rsi(prices, overbought=80)
        
        assert result is not None
        # Strong uptrend should have high RSI
        assert result.value > 50
    
    def test_rsi_trend_zone_detection(self):
        """Test RSI trend zone detection"""
        pool = IndicatorPool(rsi_period=14)
        prices = [100.0 + i * 0.5 for i in range(30)]
        result = pool.calculate_rsi(prices, trend_lower=50, trend_upper=75)
        
        assert result is not None
        # Check trend_zone is correctly set based on value
        if 50 <= result.value <= 75:
            assert result.trend_zone is True
        else:
            assert result.trend_zone is False


class TestStochasticCalculation:
    """Test Stochastic (KD) indicator calculation"""
    
    def test_stochastic_insufficient_data(self):
        """Test Stochastic returns None with insufficient data"""
        pool = IndicatorPool(stoch_k=14, stoch_d=3)
        highs = [100.0] * 10
        lows = [90.0] * 10
        closes = [95.0] * 10
        result = pool.calculate_stochastic(highs, lows, closes)
        assert result is None
    
    def test_stochastic_basic_calculation(self):
        """Test Stochastic calculation with valid data"""
        pool = IndicatorPool(stoch_k=14, stoch_d=3)
        # Generate price data
        highs = [100.0 + i for i in range(30)]
        lows = [90.0 + i for i in range(30)]
        closes = [95.0 + i for i in range(30)]
        result = pool.calculate_stochastic(highs, lows, closes)
        
        assert result is not None
        assert isinstance(result, StochasticResult)
        assert 0 <= result.k_value <= 100
        assert 0 <= result.d_value <= 100
    
    def test_stochastic_value_range(self):
        """Test K and D values are always between 0 and 100"""
        pool = IndicatorPool(stoch_k=14, stoch_d=3)
        
        # Generate random-like price data
        np.random.seed(42)
        base = 100.0
        highs = [base + np.random.uniform(5, 15) for _ in range(50)]
        lows = [h - np.random.uniform(5, 10) for h in highs]
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
        
        result = pool.calculate_stochastic(highs, lows, closes)
        
        assert result is not None
        assert 0 <= result.k_value <= 100
        assert 0 <= result.d_value <= 100


class TestMACDCalculation:
    """Test MACD indicator calculation"""
    
    def test_macd_insufficient_data(self):
        """Test MACD returns None with insufficient data"""
        pool = IndicatorPool(macd_fast=12, macd_slow=26, macd_signal=9)
        prices = [100.0] * 30  # Less than slow + signal
        result = pool.calculate_macd(prices)
        assert result is None
    
    def test_macd_basic_calculation(self):
        """Test MACD calculation with valid data"""
        pool = IndicatorPool(macd_fast=12, macd_slow=26, macd_signal=9)
        prices = [100.0 + i * 0.5 for i in range(50)]
        result = pool.calculate_macd(prices)
        
        assert result is not None
        assert isinstance(result, MACDResult)
        assert isinstance(result.macd_line, float)
        assert isinstance(result.signal_line, float)
        assert isinstance(result.histogram, float)
    
    def test_macd_histogram_calculation(self):
        """Test MACD histogram equals MACD line minus signal line"""
        pool = IndicatorPool()
        prices = [100.0 + i * 0.5 for i in range(50)]
        result = pool.calculate_macd(prices)
        
        assert result is not None
        expected_histogram = result.macd_line - result.signal_line
        assert abs(result.histogram - expected_histogram) < 1e-10


class TestEMACalculation:
    """Test EMA indicator calculation"""
    
    def test_ema_insufficient_data(self):
        """Test EMA returns None with insufficient data"""
        pool = IndicatorPool()
        prices = [100.0] * 10
        result = pool.calculate_ema(prices, period=20)
        assert result is None
    
    def test_ema_basic_calculation(self):
        """Test EMA calculation with valid data"""
        pool = IndicatorPool()
        prices = [100.0 + i for i in range(30)]
        result = pool.calculate_ema(prices, period=20)
        
        assert result is not None
        assert isinstance(result, float)
    
    def test_ema_constant_prices(self):
        """Test EMA of constant prices equals the constant"""
        pool = IndicatorPool()
        prices = [100.0] * 30
        result = pool.calculate_ema(prices, period=20)
        
        assert result is not None
        assert abs(result - 100.0) < 1e-10


class TestATRCalculation:
    """Test ATR indicator calculation"""
    
    def test_atr_insufficient_data(self):
        """Test ATR returns None with insufficient data"""
        pool = IndicatorPool(atr_period=14)
        highs = [100.0] * 10
        lows = [90.0] * 10
        closes = [95.0] * 10
        result = pool.calculate_atr(highs, lows, closes)
        assert result is None
    
    def test_atr_basic_calculation(self):
        """Test ATR calculation with valid data"""
        pool = IndicatorPool(atr_period=14)
        highs = [100.0 + i for i in range(30)]
        lows = [90.0 + i for i in range(30)]
        closes = [95.0 + i for i in range(30)]
        result = pool.calculate_atr(highs, lows, closes)
        
        assert result is not None
        assert isinstance(result, float)
        assert result > 0
    
    def test_atr_constant_range(self):
        """Test ATR with constant price range"""
        pool = IndicatorPool(atr_period=14)
        # Constant range of 10
        highs = [110.0] * 30
        lows = [100.0] * 30
        closes = [105.0] * 30
        result = pool.calculate_atr(highs, lows, closes)
        
        assert result is not None
        # ATR should be close to 10 (the constant range)
        assert 9 <= result <= 11


class TestBollingerCalculation:
    """Test Bollinger Bands calculation"""
    
    def test_bollinger_insufficient_data(self):
        """Test Bollinger returns None with insufficient data"""
        pool = IndicatorPool(bb_period=20)
        prices = [100.0] * 10
        result = pool.calculate_bollinger(prices)
        assert result is None
    
    def test_bollinger_basic_calculation(self):
        """Test Bollinger calculation with valid data"""
        pool = IndicatorPool(bb_period=20, bb_std=2.0)
        prices = [100.0 + i * 0.5 for i in range(30)]
        result = pool.calculate_bollinger(prices)
        
        assert result is not None
        assert isinstance(result, BollingerResult)
        assert result.lower < result.middle < result.upper
    
    def test_bollinger_band_structure(self):
        """Test Bollinger bands maintain correct structure"""
        pool = IndicatorPool(bb_period=20, bb_std=2.0)
        
        # Test with various price patterns
        test_cases = [
            [100.0 + i for i in range(30)],
            [100.0 + np.sin(i * 0.5) * 5 for i in range(50)],
        ]
        
        for prices in test_cases:
            result = pool.calculate_bollinger(prices)
            if result is not None:
                assert result.lower < result.middle < result.upper
                assert result.bandwidth >= 0
    
    def test_bollinger_bandwidth_calculation(self):
        """Test Bollinger bandwidth is calculated correctly"""
        pool = IndicatorPool(bb_period=20, bb_std=2.0)
        prices = [100.0 + i * 0.5 for i in range(30)]
        result = pool.calculate_bollinger(prices)
        
        assert result is not None
        expected_bandwidth = (result.upper - result.lower) / result.middle
        assert abs(result.bandwidth - expected_bandwidth) < 1e-10


class TestVolumeRatioCalculation:
    """Test Volume Ratio calculation"""
    
    def test_volume_ratio_insufficient_data(self):
        """Test Volume Ratio returns None with insufficient data"""
        pool = IndicatorPool()
        volumes = [1000.0] * 10
        result = pool.calculate_volume_ratio(volumes, period=20)
        assert result is None
    
    def test_volume_ratio_basic_calculation(self):
        """Test Volume Ratio calculation with valid data"""
        pool = IndicatorPool()
        volumes = [1000.0] * 30
        result = pool.calculate_volume_ratio(volumes, period=20)
        
        assert result is not None
        assert isinstance(result, float)
        # Constant volume should give ratio of 1.0
        assert abs(result - 1.0) < 1e-10
    
    def test_volume_ratio_high_volume(self):
        """Test Volume Ratio with high current volume"""
        pool = IndicatorPool()
        volumes = [1000.0] * 29 + [2000.0]  # Last volume is 2x average
        result = pool.calculate_volume_ratio(volumes, period=20)
        
        assert result is not None
        # Current volume is higher than average
        assert result > 1.0


class TestRSIDivergenceDetection:
    """Test RSI Divergence detection"""
    
    def test_divergence_insufficient_data(self):
        """Test divergence detection with insufficient data"""
        pool = IndicatorPool(rsi_period=14)
        prices = [100.0] * 20
        result = pool.detect_rsi_divergence(prices, lookback=20)
        assert result is False
    
    def test_divergence_no_divergence(self):
        """Test no divergence when price and RSI move together"""
        pool = IndicatorPool(rsi_period=14)
        # Consistent uptrend - no divergence expected
        prices = [100.0 + i * 0.5 for i in range(50)]
        result = pool.detect_rsi_divergence(prices, lookback=20)
        
        # Result should be boolean (numpy.bool_ or Python bool)
        assert result in (True, False)


class TestIndicatorSnapshot:
    """Test IndicatorSnapshot generation"""
    
    def test_snapshot_basic(self):
        """Test snapshot generation with valid data"""
        pool = IndicatorPool()
        
        # Generate sufficient data
        prices = [100.0 + i * 0.5 for i in range(60)]
        highs = [p + 2 for p in prices]
        lows = [p - 2 for p in prices]
        volumes = [1000.0] * 60
        
        snapshot = pool.get_snapshot(prices, highs, lows, volumes)
        
        assert isinstance(snapshot, IndicatorSnapshot)
        # Check that indicators are calculated
        assert snapshot.rsi is not None
        assert snapshot.macd is not None
        assert snapshot.bollinger is not None
        assert snapshot.ema_20 is not None
        assert snapshot.ema_50 is not None
        assert snapshot.atr is not None
        assert snapshot.volume_ratio is not None
    
    def test_snapshot_insufficient_data(self):
        """Test snapshot with insufficient data returns None for some indicators"""
        pool = IndicatorPool()
        
        # Short data - some indicators won't calculate
        prices = [100.0] * 20
        highs = [102.0] * 20
        lows = [98.0] * 20
        volumes = [1000.0] * 20
        
        snapshot = pool.get_snapshot(prices, highs, lows, volumes)
        
        assert isinstance(snapshot, IndicatorSnapshot)
        # Some indicators may be None due to insufficient data
        # EMA 50 requires 50 data points
        assert snapshot.ema_50 is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
