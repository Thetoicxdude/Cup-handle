"""Score Calculator for AI PatternQuant

This module implements the ScoreCalculator class for computing
pattern match scores based on cup and handle pattern characteristics.
"""

from .models import CupPattern, HandlePattern, MatchScore


class ScoreCalculator:
    """吻合分數計算器
    
    計算型態的綜合吻合分數，考慮杯底擬合度、左右峰對稱性、
    柄部成交量萎縮程度與杯身深度合理性。
    
    Attributes:
        r_squared_weight: 擬合度分數權重 (預設 0.3)
        symmetry_weight: 對稱性分數權重 (預設 0.25)
        volume_weight: 成交量萎縮分數權重 (預設 0.25)
        depth_weight: 深度合理性分數權重 (預設 0.2)
    """
    
    def __init__(
        self,
        r_squared_weight: float = 0.3,
        symmetry_weight: float = 0.25,
        volume_weight: float = 0.25,
        depth_weight: float = 0.2
    ):
        """初始化吻合分數計算器
        
        Args:
            r_squared_weight: 擬合度分數權重 (預設 0.3)
            symmetry_weight: 對稱性分數權重 (預設 0.25)
            volume_weight: 成交量萎縮分數權重 (預設 0.25)
            depth_weight: 深度合理性分數權重 (預設 0.2)
            
        Raises:
            ValueError: 如果權重總和不為 1.0（允許小誤差）
        """
        total_weight = r_squared_weight + symmetry_weight + volume_weight + depth_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight}"
            )
        
        self.r_squared_weight = r_squared_weight
        self.symmetry_weight = symmetry_weight
        self.volume_weight = volume_weight
        self.depth_weight = depth_weight

    def _calculate_r_squared_score(self, r_squared: float) -> float:
        """計算擬合度分數
        
        將 R² 值轉換為 0-100 分數。
        R² 範圍通常在 0.8-1.0 之間（低於 0.8 的型態已被過濾）。
        
        Args:
            r_squared: 二次函數擬合的 R² 值
            
        Returns:
            擬合度分數 (0-100)
        """
        # R² is already between 0 and 1, scale to 0-100
        # Higher R² = better fit = higher score
        score = r_squared * 100.0
        return max(0.0, min(100.0, score))

    def _calculate_symmetry_score(self, symmetry: float) -> float:
        """計算對稱性分數
        
        將對稱性值轉換為 0-100 分數。
        CupPattern 的 symmetry_score 已經是 0-1 範圍。
        
        Args:
            symmetry: 左右峰對稱性值 (0-1)
            
        Returns:
            對稱性分數 (0-100)
        """
        # symmetry_score from CupPattern is already 0-1
        score = symmetry * 100.0
        return max(0.0, min(100.0, score))

    def _calculate_volume_score(self, volume_slope: float) -> float:
        """計算成交量萎縮分數
        
        將成交量斜率轉換為 0-100 分數。
        負斜率表示萎縮趨勢，斜率越負分數越高。
        
        Args:
            volume_slope: 成交量線性回歸斜率（負值表示萎縮）
            
        Returns:
            成交量萎縮分數 (0-100)
        """
        # volume_slope is negative for shrinking volume
        # More negative = better shrinkage = higher score
        if volume_slope >= 0:
            return 0.0
        
        # Normalize the slope to a score
        # Use a sigmoid-like transformation for smooth scoring
        # Typical volume slopes might range from -1000 to 0
        # We want to map this to 0-100
        
        # Simple approach: use absolute value with diminishing returns
        abs_slope = abs(volume_slope)
        
        # Score increases with more negative slope, with diminishing returns
        # Using a formula that gives ~50 at moderate shrinkage, ~90+ at strong shrinkage
        score = 100.0 * (1.0 - 1.0 / (1.0 + abs_slope / 100.0))
        
        return max(0.0, min(100.0, score))

    def _calculate_depth_score(self, depth_ratio: float) -> float:
        """計算深度合理性分數
        
        將杯身深度比例轉換為 0-100 分數。
        理想深度在 12%-33% 範圍內，中間值（約 20-25%）得分最高。
        
        Args:
            depth_ratio: 杯身深度比例
            
        Returns:
            深度合理性分數 (0-100)
        """
        # Ideal depth is around 20-25% (middle of 12%-33% range)
        ideal_depth = 0.225  # (0.12 + 0.33) / 2
        min_depth = 0.12
        max_depth = 0.33
        
        # Check if depth is within valid range
        if depth_ratio < min_depth or depth_ratio > max_depth:
            return 0.0
        
        # Calculate distance from ideal depth
        distance_from_ideal = abs(depth_ratio - ideal_depth)
        max_distance = max(ideal_depth - min_depth, max_depth - ideal_depth)
        
        # Score decreases as we move away from ideal
        # Perfect score at ideal depth, lower scores at extremes
        score = 100.0 * (1.0 - distance_from_ideal / max_distance)
        
        return max(0.0, min(100.0, score))

    def calculate(
        self,
        cup: CupPattern,
        handle: HandlePattern
    ) -> MatchScore:
        """計算綜合吻合分數
        
        根據茶杯型態和柄部型態的特徵，計算各項子分數並加權平均
        得出綜合吻合分數。
        
        Args:
            cup: 已識別的茶杯型態
            handle: 已識別的柄部型態
            
        Returns:
            包含總分與各子分數的 MatchScore 物件
        """
        # Calculate individual scores
        r_squared_score = self._calculate_r_squared_score(cup.r_squared)
        symmetry_score = self._calculate_symmetry_score(cup.symmetry_score)
        volume_score = self._calculate_volume_score(handle.volume_slope)
        depth_score = self._calculate_depth_score(cup.depth_ratio)
        
        # Calculate weighted total score
        total_score = (
            r_squared_score * self.r_squared_weight +
            symmetry_score * self.symmetry_weight +
            volume_score * self.volume_weight +
            depth_score * self.depth_weight
        )
        
        # Ensure total score is within bounds
        total_score = max(0.0, min(100.0, total_score))
        
        return MatchScore(
            total_score=total_score,
            r_squared_score=r_squared_score,
            symmetry_score=symmetry_score,
            volume_score=volume_score,
            depth_score=depth_score
        )
