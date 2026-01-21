"""Trend Filter for AI PatternQuant

This module implements the TrendFilter class for confirming long-term
trend direction using moving averages.
"""

from typing import List


class TrendFilter:
    """趨勢過濾器
    
    使用移動平均線確認長期趨勢方向。
    當短期均線高於長期均線時，判定為上升趨勢。
    
    Attributes:
        short_period: 短期移動平均週期（預設 50 日）
        long_period: 長期移動平均週期（預設 200 日）
    """
    
    def __init__(self, short_period: int = 50, long_period: int = 200):
        """初始化趨勢過濾器
        
        Args:
            short_period: 短期移動平均週期，預設 50
            long_period: 長期移動平均週期，預設 200
        
        Raises:
            ValueError: 如果週期不是正整數或短期週期大於等於長期週期
        """
        if short_period < 1:
            raise ValueError("short_period must be positive")
        if long_period < 1:
            raise ValueError("long_period must be positive")
        if short_period >= long_period:
            raise ValueError("short_period must be less than long_period")
        
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate_ma(self, prices: List[float], period: int) -> float:
        """計算移動平均線
        
        計算最近 n 個價格的算術平均值。
        
        Args:
            prices: 收盤價序列
            period: 移動平均週期
            
        Returns:
            移動平均值
            
        Raises:
            ValueError: 如果價格序列長度小於週期
        """
        if len(prices) < period:
            raise ValueError(f"prices length ({len(prices)}) must be >= period ({period})")
        
        # Take the last 'period' prices and calculate arithmetic mean
        recent_prices = prices[-period:]
        return sum(recent_prices) / period
    
    def is_uptrend(self, prices: List[float]) -> bool:
        """判斷是否處於上升趨勢
        
        當短期均線（50日）高於長期均線（200日）時，判定為上升趨勢。
        
        Args:
            prices: 收盤價序列，長度必須至少為 long_period
            
        Returns:
            True 如果短期均線 > 長期均線，否則 False
            
        Raises:
            ValueError: 如果價格序列長度小於 long_period
        """
        if len(prices) < self.long_period:
            raise ValueError(
                f"prices length ({len(prices)}) must be >= long_period ({self.long_period})"
            )
        
        short_ma = self.calculate_ma(prices, self.short_period)
        long_ma = self.calculate_ma(prices, self.long_period)
        
        return short_ma > long_ma
