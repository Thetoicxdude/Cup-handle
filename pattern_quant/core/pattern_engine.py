"""Pattern Engine for AI PatternQuant

This module implements the PatternEngine class that integrates all
pattern detection components to perform complete pattern recognition.
"""

from typing import List, Optional

from .models import (
    CupPattern,
    HandlePattern,
    MatchScore,
    PatternResult,
    OHLCV,
)
from .trend_filter import TrendFilter
from .extrema_detector import LocalExtremaDetector
from .cup_detector import CupPatternDetector
from .handle_detector import HandlePatternDetector
from .score_calculator import ScoreCalculator


class PatternEngine:
    """型態識別整合引擎
    
    整合 TrendFilter, LocalExtremaDetector, CupPatternDetector,
    HandlePatternDetector, ScoreCalculator 執行完整型態識別流程。
    
    Attributes:
        trend_filter: 趨勢過濾器
        extrema_detector: 局部極值檢測器
        cup_detector: 茶杯型態檢測器
        handle_detector: 柄部型態檢測器
        score_calculator: 吻合分數計算器
    """
    
    def __init__(
        self,
        # TrendFilter parameters
        short_period: int = 50,
        long_period: int = 200,
        # LocalExtremaDetector parameters
        window_size: int = 5,
        # CupPatternDetector parameters
        peak_tolerance: float = 0.05,
        min_depth: float = 0.12,
        max_depth: float = 0.33,
        min_r_squared: float = 0.8,
        min_cup_days: int = 30,
        max_cup_days: int = 150,
        # HandlePatternDetector parameters
        max_handle_depth: float = 0.5,
        max_handle_days: int = 25,
        # ScoreCalculator parameters
        r_squared_weight: float = 0.3,
        symmetry_weight: float = 0.25,
        volume_weight: float = 0.25,
        depth_weight: float = 0.2,
    ):
        """初始化型態識別引擎
        
        Args:
            short_period: 短期移動平均週期 (預設 50)
            long_period: 長期移動平均週期 (預設 200)
            window_size: 極值檢測窗口大小 (預設 5)
            peak_tolerance: 左右峰高度容差 (預設 5%)
            min_depth: 最小杯身深度 (預設 12%)
            max_depth: 最大杯身深度 (預設 33%)
            min_r_squared: 最小擬合度 (預設 0.8)
            min_cup_days: 最小成型天數 (預設 30)
            max_cup_days: 最大成型天數 (預設 150)
            max_handle_depth: 柄部最大深度 (預設 0.5)
            max_handle_days: 柄部最大天數 (預設 25)
            r_squared_weight: 擬合度分數權重 (預設 0.3)
            symmetry_weight: 對稱性分數權重 (預設 0.25)
            volume_weight: 成交量萎縮分數權重 (預設 0.25)
            depth_weight: 深度合理性分數權重 (預設 0.2)
        """
        self.trend_filter = TrendFilter(
            short_period=short_period,
            long_period=long_period
        )
        
        self.extrema_detector = LocalExtremaDetector(
            window_size=window_size
        )
        
        self.cup_detector = CupPatternDetector(
            peak_tolerance=peak_tolerance,
            min_depth=min_depth,
            max_depth=max_depth,
            min_r_squared=min_r_squared,
            min_cup_days=min_cup_days,
            max_cup_days=max_cup_days
        )
        
        self.handle_detector = HandlePatternDetector(
            max_handle_depth=max_handle_depth,
            max_handle_days=max_handle_days
        )
        
        self.score_calculator = ScoreCalculator(
            r_squared_weight=r_squared_weight,
            symmetry_weight=symmetry_weight,
            volume_weight=volume_weight,
            depth_weight=depth_weight
        )

    def analyze_symbol(
        self,
        symbol: str,
        prices: List[float],
        volumes: List[float]
    ) -> PatternResult:
        """執行完整型態識別流程
        
        對指定股票執行完整的 Cup and Handle 型態識別流程：
        1. 趨勢過濾：確認處於上升趨勢
        2. 極值檢測：識別波峰與波谷
        3. 茶杯型態檢測：識別茶杯型態
        4. 柄部型態檢測：識別柄部型態
        5. 分數計算：計算綜合吻合分數
        
        Args:
            symbol: 股票代碼
            prices: 收盤價序列
            volumes: 成交量序列
            
        Returns:
            PatternResult 包含識別結果與相關資訊
        """
        # Validate input
        if len(prices) != len(volumes):
            return PatternResult(
                symbol=symbol,
                pattern_type="cup_and_handle",
                cup=None,
                handle=None,
                score=None,
                is_valid=False,
                rejection_reason="Price and volume series length mismatch"
            )
        
        if len(prices) < self.trend_filter.long_period:
            return PatternResult(
                symbol=symbol,
                pattern_type="cup_and_handle",
                cup=None,
                handle=None,
                score=None,
                is_valid=False,
                rejection_reason=f"Insufficient data: need at least {self.trend_filter.long_period} data points"
            )
        
        # Step 1: Trend Filter
        try:
            is_uptrend = self.trend_filter.is_uptrend(prices)
        except ValueError as e:
            return PatternResult(
                symbol=symbol,
                pattern_type="cup_and_handle",
                cup=None,
                handle=None,
                score=None,
                is_valid=False,
                rejection_reason=f"Trend filter error: {str(e)}"
            )
        
        if not is_uptrend:
            return PatternResult(
                symbol=symbol,
                pattern_type="cup_and_handle",
                cup=None,
                handle=None,
                score=None,
                is_valid=False,
                rejection_reason="Not in uptrend (50-day MA below 200-day MA)"
            )
        
        # Step 2: Detect Local Extrema
        extrema = self.extrema_detector.detect(prices)
        
        if len(extrema) < 2:
            return PatternResult(
                symbol=symbol,
                pattern_type="cup_and_handle",
                cup=None,
                handle=None,
                score=None,
                is_valid=False,
                rejection_reason="Insufficient extrema detected"
            )
        
        # Step 3: Detect Cup Pattern
        cup = self.cup_detector.detect(prices, extrema)
        
        if cup is None:
            return PatternResult(
                symbol=symbol,
                pattern_type="cup_and_handle",
                cup=None,
                handle=None,
                score=None,
                is_valid=False,
                rejection_reason="No valid cup pattern found"
            )
        
        # Step 4: Detect Handle Pattern
        handle = self.handle_detector.detect(prices, volumes, cup)
        
        if handle is None:
            return PatternResult(
                symbol=symbol,
                pattern_type="cup_and_handle",
                cup=cup,
                handle=None,
                score=None,
                is_valid=False,
                rejection_reason="No valid handle pattern found"
            )
        
        # Step 5: Calculate Match Score
        score = self.score_calculator.calculate(cup, handle)
        
        return PatternResult(
            symbol=symbol,
            pattern_type="cup_and_handle",
            cup=cup,
            handle=handle,
            score=score,
            is_valid=True,
            rejection_reason=None
        )

    def analyze_ohlcv(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV]
    ) -> PatternResult:
        """使用 OHLCV 數據執行型態識別
        
        從 OHLCV 數據中提取收盤價與成交量，執行完整型態識別流程。
        
        Args:
            symbol: 股票代碼
            ohlcv_data: OHLCV K 線數據列表
            
        Returns:
            PatternResult 包含識別結果與相關資訊
        """
        if not ohlcv_data:
            return PatternResult(
                symbol=symbol,
                pattern_type="cup_and_handle",
                cup=None,
                handle=None,
                score=None,
                is_valid=False,
                rejection_reason="Empty OHLCV data"
            )
        
        # Extract prices and volumes from OHLCV data
        prices = [candle.close for candle in ohlcv_data]
        volumes = [float(candle.volume) for candle in ohlcv_data]
        
        return self.analyze_symbol(symbol, prices, volumes)
