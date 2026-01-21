"""Cup Pattern Detector for AI PatternQuant

This module implements the CupPatternDetector class for identifying
cup patterns in price series using mathematical curve fitting.
"""

from typing import List, Optional, Tuple

import numpy as np

from .models import CupPattern, Extremum


class CupPatternDetector:
    """茶杯型態檢測器
    
    識別價格序列中的茶杯型態，使用二次函數擬合驗證杯底形狀。
    
    Attributes:
        peak_tolerance: 左右峰高度容差 (預設 5%)
        min_depth: 最小杯身深度 (預設 12%)
        max_depth: 最大杯身深度 (預設 33%)
        min_r_squared: 最小擬合度
        min_cup_days: 最小成型天數
        max_cup_days: 最大成型天數
    """
    
    def __init__(
        self,
        peak_tolerance: float = 0.05,
        min_depth: float = 0.12,
        max_depth: float = 0.33,
        min_r_squared: float = 0.8,
        min_cup_days: int = 30,
        max_cup_days: int = 150
    ):
        """初始化茶杯型態檢測器
        
        Args:
            peak_tolerance: 左右峰高度容差 (預設 5%)
            min_depth: 最小杯身深度 (預設 12%)
            max_depth: 最大杯身深度 (預設 33%)
            min_r_squared: 最小擬合度 (預設 0.8)
            min_cup_days: 最小成型天數 (預設 30)
            max_cup_days: 最大成型天數 (預設 150)
        """
        self.peak_tolerance = peak_tolerance
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_r_squared = min_r_squared
        self.min_cup_days = min_cup_days
        self.max_cup_days = max_cup_days

    def fit_parabola(
        self,
        prices: List[float],
        start_idx: int,
        end_idx: int
    ) -> Tuple[float, float, float, float]:
        """對杯底區間進行二次函數擬合
        
        使用 numpy.polyfit 進行二次多項式擬合 y = ax² + bx + c，
        並計算 R² 值來評估擬合度。
        
        Args:
            prices: 完整的收盤價序列
            start_idx: 杯底區間起始索引
            end_idx: 杯底區間結束索引（包含）
            
        Returns:
            (a, b, c, r_squared) - y = ax² + bx + c 的係數與 R² 值
            
        Raises:
            ValueError: 如果區間無效或數據點不足
        """
        if start_idx < 0 or end_idx >= len(prices) or start_idx >= end_idx:
            raise ValueError("Invalid index range")
        
        # Extract the price segment for fitting
        segment = prices[start_idx:end_idx + 1]
        
        if len(segment) < 3:
            raise ValueError("Need at least 3 points for parabola fitting")
        
        # Create x values (normalized to avoid numerical issues)
        x = np.arange(len(segment), dtype=float)
        y = np.array(segment, dtype=float)
        
        # Fit quadratic polynomial: y = ax² + bx + c
        coefficients = np.polyfit(x, y, 2)
        a, b, c = coefficients
        
        # Calculate R² (coefficient of determination)
        y_pred = np.polyval(coefficients, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # Handle edge case where all y values are the same
        if ss_tot == 0:
            r_squared = 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        return float(a), float(b), float(c), float(r_squared)

    def is_v_shaped(
        self,
        prices: List[float],
        left_peak_idx: int,
        bottom_idx: int,
        right_peak_idx: int
    ) -> bool:
        """判斷杯底是否為 V 型（應拒絕）
        
        V 型底部的特徵是價格快速下跌後快速反彈，缺乏圓弧形的緩慢過渡。
        使用以下方法判斷：
        1. 計算左半部和右半部的價格變異係數
        2. 如果變異係數過低（價格變化過於線性），則判定為 V 型
        3. 或者檢查底部區域的寬度是否過窄
        
        Args:
            prices: 完整的收盤價序列
            left_peak_idx: 左峰索引
            bottom_idx: 杯底索引
            right_peak_idx: 右峰索引
            
        Returns:
            True 如果是 V 型底部（應拒絕），False 如果是圓弧形底部
        """
        if bottom_idx <= left_peak_idx or bottom_idx >= right_peak_idx:
            return True  # Invalid structure, treat as V-shaped
        
        # Get the cup segment
        cup_segment = prices[left_peak_idx:right_peak_idx + 1]
        if len(cup_segment) < 5:
            return True  # Too short, likely V-shaped
        
        # Calculate the relative position of the bottom within the cup
        cup_length = right_peak_idx - left_peak_idx
        bottom_relative_pos = (bottom_idx - left_peak_idx) / cup_length
        
        # V-shaped patterns have very sharp bottoms
        # Check if the bottom is too narrow (concentrated in a small region)
        left_segment = prices[left_peak_idx:bottom_idx + 1]
        right_segment = prices[bottom_idx:right_peak_idx + 1]
        
        if len(left_segment) < 2 or len(right_segment) < 2:
            return True
        
        # Calculate the "sharpness" of the bottom
        # For V-shape: prices drop quickly and rise quickly
        # For U-shape: prices have a gradual transition at the bottom
        
        # Method: Check the curvature around the bottom
        # A V-shape will have high variance in the rate of change
        bottom_region_start = max(left_peak_idx, bottom_idx - 3)
        bottom_region_end = min(right_peak_idx, bottom_idx + 3)
        bottom_region = prices[bottom_region_start:bottom_region_end + 1]
        
        if len(bottom_region) < 3:
            return True
        
        # Calculate second derivative approximation at bottom
        # V-shape has very high curvature (sharp point)
        # U-shape has lower, more gradual curvature
        
        # Use the ratio of bottom region length to total cup length
        bottom_region_ratio = len(bottom_region) / cup_length
        
        # If the bottom region is too small relative to the cup, it's V-shaped
        if bottom_region_ratio < 0.1:
            return True
        
        # Check if the bottom is too "pointy" by comparing slopes
        left_slope = (prices[bottom_idx] - prices[left_peak_idx]) / (bottom_idx - left_peak_idx)
        right_slope = (prices[right_peak_idx] - prices[bottom_idx]) / (right_peak_idx - bottom_idx)
        
        # For V-shape, slopes are steep and nearly symmetric
        # For U-shape, the transition is more gradual
        avg_slope_magnitude = (abs(left_slope) + abs(right_slope)) / 2
        
        # Normalize by price range
        price_range = max(prices[left_peak_idx], prices[right_peak_idx]) - prices[bottom_idx]
        if price_range == 0:
            return True
        
        normalized_slope = avg_slope_magnitude * cup_length / price_range
        
        # High normalized slope indicates V-shape
        # Threshold determined empirically
        if normalized_slope > 2.5:
            return True
        
        return False

    def detect(
        self,
        prices: List[float],
        extrema: List[Extremum]
    ) -> Optional[CupPattern]:
        """檢測茶杯型態
        
        從價格序列和已識別的極值中尋找符合條件的茶杯型態。
        檢測流程：
        1. 找出所有可能的左峰-右峰配對
        2. 驗證左右峰高度對稱性
        3. 找出杯底並驗證深度
        4. 進行二次函數擬合驗證圓弧形
        5. 排除 V 型底部
        
        Args:
            prices: 收盤價序列
            extrema: 已識別的極值列表
            
        Returns:
            識別到的茶杯型態，若無則返回 None
        """
        if len(prices) < self.min_cup_days:
            return None
        
        # Filter peaks from extrema
        peaks = [e for e in extrema if e.is_peak]
        
        if len(peaks) < 2:
            return None
        
        best_pattern = None
        best_r_squared = 0.0
        
        # Try all possible left-right peak combinations
        for i, left_peak in enumerate(peaks):
            for right_peak in peaks[i + 1:]:
                pattern = self._try_detect_cup(prices, left_peak, right_peak)
                if pattern is not None and pattern.r_squared > best_r_squared:
                    best_pattern = pattern
                    best_r_squared = pattern.r_squared
        
        return best_pattern
    
    def _try_detect_cup(
        self,
        prices: List[float],
        left_peak: Extremum,
        right_peak: Extremum
    ) -> Optional[CupPattern]:
        """嘗試從指定的左右峰檢測茶杯型態
        
        Args:
            prices: 收盤價序列
            left_peak: 左峰極值
            right_peak: 右峰極值
            
        Returns:
            識別到的茶杯型態，若不符合條件則返回 None
        """
        # Check cup duration
        cup_days = right_peak.index - left_peak.index
        if cup_days < self.min_cup_days or cup_days > self.max_cup_days:
            return None
        
        # Check peak symmetry: |P_left - P_right| / P_left < tolerance
        price_diff_ratio = abs(left_peak.price - right_peak.price) / left_peak.price
        if price_diff_ratio > self.peak_tolerance:
            return None
        
        # Find the bottom (minimum price between peaks)
        cup_segment = prices[left_peak.index:right_peak.index + 1]
        bottom_local_idx = int(np.argmin(cup_segment))
        bottom_idx = left_peak.index + bottom_local_idx
        bottom_price = prices[bottom_idx]
        
        # Check depth: (P_left - Min_cup) / P_left should be in [min_depth, max_depth]
        depth_ratio = (left_peak.price - bottom_price) / left_peak.price
        if depth_ratio < self.min_depth or depth_ratio > self.max_depth:
            return None
        
        # Check for V-shape (should reject)
        if self.is_v_shaped(prices, left_peak.index, bottom_idx, right_peak.index):
            return None
        
        # Fit parabola to cup bottom
        try:
            a, b, c, r_squared = self.fit_parabola(
                prices, left_peak.index, right_peak.index
            )
        except ValueError:
            return None
        
        # Check parabola opens upward (a > 0) and good fit (R² > threshold)
        if a <= 0 or r_squared < self.min_r_squared:
            return None
        
        # Calculate symmetry score (1.0 = perfect symmetry)
        symmetry_score = 1.0 - price_diff_ratio / self.peak_tolerance
        symmetry_score = max(0.0, min(1.0, symmetry_score))
        
        return CupPattern(
            left_peak_index=left_peak.index,
            left_peak_price=left_peak.price,
            right_peak_index=right_peak.index,
            right_peak_price=right_peak.price,
            bottom_index=bottom_idx,
            bottom_price=bottom_price,
            r_squared=r_squared,
            depth_ratio=depth_ratio,
            symmetry_score=symmetry_score
        )
