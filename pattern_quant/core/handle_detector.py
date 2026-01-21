"""Handle Pattern Detector for AI PatternQuant

This module implements the HandlePatternDetector class for identifying
handle patterns following a cup pattern in price series.
"""

from typing import List, Optional

import numpy as np

from .models import CupPattern, HandlePattern


class HandlePatternDetector:
    """柄部型態檢測器
    
    識別茶杯右峰後的柄部型態，驗證深度與成交量萎縮條件。
    
    Attributes:
        max_handle_depth: 柄部最大深度（相對於杯身上半部）
        max_handle_days: 柄部最大天數
    """
    
    def __init__(
        self,
        max_handle_depth: float = 0.5,
        max_handle_days: int = 25
    ):
        """初始化柄部型態檢測器
        
        Args:
            max_handle_depth: 柄部最大深度，相對於杯身深度的比例 (預設 0.5)
                             柄部最低點不得低於杯身上半部
            max_handle_days: 柄部最大天數 (預設 25)
        """
        self.max_handle_depth = max_handle_depth
        self.max_handle_days = max_handle_days

    def calculate_volume_slope(self, volumes: List[float]) -> float:
        """計算成交量的線性回歸斜率
        
        使用最小二乘法進行線性回歸，計算成交量隨時間的變化斜率。
        負斜率表示成交量萎縮趨勢。
        
        Args:
            volumes: 成交量序列
            
        Returns:
            線性回歸斜率值，負值表示萎縮趨勢
            
        Raises:
            ValueError: 如果成交量序列少於 2 個點
        """
        if len(volumes) < 2:
            raise ValueError("Need at least 2 points for linear regression")
        
        # Create x values (time indices)
        x = np.arange(len(volumes), dtype=float)
        y = np.array(volumes, dtype=float)
        
        # Linear regression: y = slope * x + intercept
        # Using numpy.polyfit with degree 1
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        
        return float(slope)

    def detect(
        self,
        prices: List[float],
        volumes: List[float],
        cup: CupPattern
    ) -> Optional[HandlePattern]:
        """檢測柄部型態
        
        從茶杯右峰之後的價格序列中尋找符合條件的柄部型態。
        檢測流程：
        1. 確認右峰之後有足夠的數據點
        2. 找出柄部區間（右峰後到當前）
        3. 驗證柄部深度不低於杯身上半部
        4. 驗證成交量呈萎縮趨勢（斜率為負）
        
        Args:
            prices: 收盤價序列
            volumes: 成交量序列
            cup: 已識別的茶杯型態
            
        Returns:
            識別到的柄部型態，若無則返回 None
        """
        if len(prices) != len(volumes):
            return None
        
        # Handle starts after the right peak
        handle_start = cup.right_peak_index + 1
        
        # Check if there's enough data after the right peak
        if handle_start >= len(prices):
            return None
        
        # Determine handle end (either max_handle_days or end of data)
        handle_end = min(handle_start + self.max_handle_days, len(prices) - 1)
        
        # Need at least 2 days for handle
        if handle_end <= handle_start:
            return None
        
        # Extract handle price and volume segments
        handle_prices = prices[handle_start:handle_end + 1]
        handle_volumes = volumes[handle_start:handle_end + 1]
        
        if len(handle_prices) < 2:
            return None
        
        # Find the lowest price in the handle
        lowest_price = min(handle_prices)
        
        # Calculate the cup's upper half threshold
        # 杯身上半部 = (左峰價格 + 杯底價格) / 2
        cup_upper_half = (cup.left_peak_price + cup.bottom_price) / 2
        
        # Requirement 3.1 & 3.3: Handle lowest point must not be below cup's upper half
        if lowest_price < cup_upper_half:
            return None
        
        # Requirement 3.2: Calculate volume slope (must be negative for shrinking volume)
        try:
            volume_slope = self.calculate_volume_slope(handle_volumes)
        except ValueError:
            return None
        
        # Volume must show shrinking trend (negative slope)
        if volume_slope >= 0:
            return None
        
        return HandlePattern(
            start_index=handle_start,
            end_index=handle_end,
            lowest_price=lowest_price,
            volume_slope=volume_slope
        )
