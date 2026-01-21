"""Local Extrema Detector for AI PatternQuant

This module implements the LocalExtremaDetector class for identifying
local peaks and troughs in price series.
"""

from typing import List

from .models import Extremum


class LocalExtremaDetector:
    """局部極值檢測器
    
    從價格序列中識別波峰（局部最大值）與波谷（局部最小值）。
    使用可配置的窗口大小來過濾雜訊。
    
    Attributes:
        window_size: 用於判定局部極值的窗口大小，必須為正奇數
    """
    
    def __init__(self, window_size: int = 5):
        """初始化極值檢測器
        
        Args:
            window_size: 用於判定局部極值的窗口大小，必須為正奇數。
                        預設為 5，表示檢查前後各 2 個點。
        
        Raises:
            ValueError: 如果 window_size 不是正奇數
        """
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        
        self.window_size = window_size
        self._half_window = window_size // 2
    
    def detect(self, prices: List[float]) -> List[Extremum]:
        """檢測價格序列中的所有局部極值
        
        識別所有波峰與波谷，並按索引位置排序返回。
        波峰與波谷會交替出現（不會連續兩個波峰或兩個波谷）。
        
        Args:
            prices: 收盤價序列
            
        Returns:
            按索引排序的極值列表
        """
        if len(prices) < self.window_size:
            return []
        
        peaks = self.find_peaks(prices)
        troughs = self.find_troughs(prices)
        
        # Merge and sort by index
        all_extrema = peaks + troughs
        all_extrema.sort(key=lambda x: x.index)
        
        # Filter to ensure alternating peaks and troughs
        return self._filter_alternating(all_extrema)
    
    def find_peaks(self, prices: List[float]) -> List[Extremum]:
        """僅返回波峰
        
        波峰定義為在窗口範圍內價格最高的點。
        
        Args:
            prices: 收盤價序列
            
        Returns:
            波峰列表
        """
        if len(prices) < self.window_size:
            return []
        
        peaks = []
        
        for i in range(self._half_window, len(prices) - self._half_window):
            window_start = i - self._half_window
            window_end = i + self._half_window + 1
            window = prices[window_start:window_end]
            
            current_price = prices[i]
            
            # Check if current point is strictly greater than all other points in window
            is_peak = True
            for j, price in enumerate(window):
                if j != self._half_window:  # Skip the center point
                    if current_price <= price:
                        is_peak = False
                        break
            
            if is_peak:
                peaks.append(Extremum(index=i, price=current_price, is_peak=True))
        
        return peaks
    
    def find_troughs(self, prices: List[float]) -> List[Extremum]:
        """僅返回波谷
        
        波谷定義為在窗口範圍內價格最低的點。
        
        Args:
            prices: 收盤價序列
            
        Returns:
            波谷列表
        """
        if len(prices) < self.window_size:
            return []
        
        troughs = []
        
        for i in range(self._half_window, len(prices) - self._half_window):
            window_start = i - self._half_window
            window_end = i + self._half_window + 1
            window = prices[window_start:window_end]
            
            current_price = prices[i]
            
            # Check if current point is strictly less than all other points in window
            is_trough = True
            for j, price in enumerate(window):
                if j != self._half_window:  # Skip the center point
                    if current_price >= price:
                        is_trough = False
                        break
            
            if is_trough:
                troughs.append(Extremum(index=i, price=current_price, is_peak=False))
        
        return troughs
    
    def _filter_alternating(self, extrema: List[Extremum]) -> List[Extremum]:
        """過濾極值列表，確保波峰與波谷交替出現
        
        當連續出現相同類型的極值時，保留最極端的那個：
        - 連續波峰：保留價格最高的
        - 連續波谷：保留價格最低的
        
        Args:
            extrema: 按索引排序的極值列表
            
        Returns:
            過濾後的交替極值列表
        """
        if len(extrema) <= 1:
            return extrema
        
        result = []
        i = 0
        
        while i < len(extrema):
            current = extrema[i]
            
            # Find all consecutive extrema of the same type
            same_type_group = [current]
            j = i + 1
            while j < len(extrema) and extrema[j].is_peak == current.is_peak:
                same_type_group.append(extrema[j])
                j += 1
            
            # Keep the most extreme one from the group
            if current.is_peak:
                # For peaks, keep the highest
                best = max(same_type_group, key=lambda x: x.price)
            else:
                # For troughs, keep the lowest
                best = min(same_type_group, key=lambda x: x.price)
            
            result.append(best)
            i = j
        
        return result
