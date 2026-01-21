"""Signal Generator - 訊號生成器

根據型態識別結果生成交易訊號。
整合訊號優化器 (Signal Optimizer) 進行多指標評分。
"""

from datetime import datetime
from typing import Optional, List

from pattern_quant.core.models import (
    CupPattern,
    HandlePattern,
    MatchScore,
    TradeSignal,
    SignalStatus,
)


class SignalGenerator:
    """訊號生成器
    
    根據型態識別結果與市場條件生成交易訊號。
    可選擇性整合訊號優化器進行多指標評分。
    
    Attributes:
        score_threshold: 吻合分數閾值，超過此值才考慮生成訊號
        breakout_buffer: 突破緩衝比例 (預設 0.5%)
        volume_multiplier: 成交量倍數要求 (預設 1.5 倍)
        signal_optimizer: 訊號優化器（可選）
        use_optimizer: 是否使用訊號優化器
    """
    
    def __init__(
        self,
        score_threshold: float = 80.0,
        breakout_buffer: float = 0.005,
        volume_multiplier: float = 1.5,
        signal_optimizer=None,
        use_optimizer: bool = True
    ):
        """初始化訊號生成器
        
        Args:
            score_threshold: 吻合分數閾值 (0-100)，預設 80.0
            breakout_buffer: 突破緩衝比例，預設 0.005 (0.5%)
            volume_multiplier: 成交量倍數要求，預設 1.5
            signal_optimizer: 訊號優化器實例（可選）
            use_optimizer: 是否使用訊號優化器，預設 True
        """
        self.score_threshold = score_threshold
        self.breakout_buffer = breakout_buffer
        self.volume_multiplier = volume_multiplier
        self.signal_optimizer = signal_optimizer
        self.use_optimizer = use_optimizer and signal_optimizer is not None
    
    def check_breakout(
        self,
        current_price: float,
        breakout_price: float,
        current_volume: float,
        avg_volume: float
    ) -> bool:
        """檢查是否滿足突破條件
        
        突破條件：
        1. 當前價格 >= 突破價位
        2. 當前成交量 >= 10日均量 × 成交量倍數
        
        Args:
            current_price: 當前價格
            breakout_price: 突破價位 (右杯緣 + 緩衝)
            current_volume: 當前成交量
            avg_volume: 10日平均成交量
            
        Returns:
            True 如果滿足突破條件，否則 False
        """
        # 檢查價格突破
        price_breakout = current_price >= breakout_price
        
        # 檢查成交量放大 (Requirements 7.2)
        volume_confirmation = current_volume >= avg_volume * self.volume_multiplier
        
        return price_breakout and volume_confirmation
    
    def generate(
        self,
        symbol: str,
        current_price: float,
        current_volume: float,
        avg_volume_10d: float,
        cup: CupPattern,
        handle: HandlePattern,
        score: MatchScore,
        prices: Optional[List[float]] = None,
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None
    ) -> Optional[TradeSignal]:
        """生成交易訊號
        
        根據型態識別結果與市場條件生成交易訊號。
        若啟用訊號優化器，會整合多指標評分。
        
        訊號生成條件 (Requirements 7.1, 7.2):
        1. Match_Score 超過閾值
        2. 價格突破右杯緣 + 0.5%
        3. 成交量 >= 10日均量 × 1.5
        
        Args:
            symbol: 股票代碼
            current_price: 當前價格
            current_volume: 當前成交量
            avg_volume_10d: 過去10日平均成交量
            cup: 茶杯型態
            handle: 柄部型態
            score: 吻合分數
            prices: 收盤價序列（用於訊號優化器）
            highs: 最高價序列（用於訊號優化器）
            lows: 最低價序列（用於訊號優化器）
            volumes: 成交量序列（用於訊號優化器）
            
        Returns:
            交易訊號，若條件不滿足則返回 None
        """
        # 計算突破價位 (右杯緣 + 0.5%)
        breakout_price = cup.right_peak_price * (1 + self.breakout_buffer)
        
        # 計算止損價位 (使用柄部最低點)
        stop_loss_price = handle.lowest_price
        
        # 使用訊號優化器進行多指標評分 (Requirements 7.1, 7.2, 7.3, 7.4)
        final_score = score.total_score
        optimized_signal = None
        signal_strength = None
        
        if self.use_optimizer and self.signal_optimizer is not None:
            if prices and highs and lows and volumes:
                try:
                    optimized_signal = self.signal_optimizer.optimize(
                        symbol=symbol,
                        pattern_score=score.total_score,
                        prices=prices,
                        highs=highs,
                        lows=lows,
                        volumes=volumes,
                        entry_price=current_price
                    )
                    final_score = optimized_signal.final_score
                    signal_strength = optimized_signal.strength
                    
                    # 使用優化器的 ATR 止損（如果有）
                    if optimized_signal.recommended_stop_loss > 0:
                        stop_loss_price = optimized_signal.recommended_stop_loss
                        
                except Exception:
                    # 優化器失敗時回退到原始分數
                    pass
        
        # 檢查吻合分數是否超過閾值 (Requirements 7.1)
        # 使用優化後的分數進行判斷
        if final_score < self.score_threshold:
            return None
        
        # 檢查訊號強度（如果使用優化器）
        # Requirements 7.3, 7.4: 根據訊號強度決定是否生成訊號
        if signal_strength is not None:
            from pattern_quant.optimization.signal_optimizer import SignalStrength
            if signal_strength == SignalStrength.SKIP:
                return None
        
        # 檢查突破條件
        if not self.check_breakout(
            current_price, breakout_price, current_volume, avg_volume_10d
        ):
            return None
        
        # 計算預期獲利比 (使用杯身深度作為目標)
        # 目標價位 = 突破價位 + 杯身深度
        cup_depth = cup.left_peak_price - cup.bottom_price
        target_price = breakout_price + cup_depth
        risk = current_price - stop_loss_price
        
        if risk > 0:
            expected_profit_ratio = (target_price - current_price) / risk
        else:
            expected_profit_ratio = 0.0
        
        # 生成交易訊號
        return TradeSignal(
            symbol=symbol,
            pattern_type="cup_and_handle",
            match_score=final_score,  # 使用優化後的分數
            breakout_price=breakout_price,
            stop_loss_price=stop_loss_price,
            status=SignalStatus.TRIGGERED,
            created_at=datetime.now(),
            expected_profit_ratio=expected_profit_ratio
        )
    
    def generate_with_optimization(
        self,
        symbol: str,
        current_price: float,
        current_volume: float,
        avg_volume_10d: float,
        cup: CupPattern,
        handle: HandlePattern,
        score: MatchScore,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float]
    ) -> Optional[TradeSignal]:
        """生成交易訊號（強制使用訊號優化器）
        
        此方法確保使用訊號優化器進行多指標評分。
        
        Args:
            symbol: 股票代碼
            current_price: 當前價格
            current_volume: 當前成交量
            avg_volume_10d: 過去10日平均成交量
            cup: 茶杯型態
            handle: 柄部型態
            score: 吻合分數
            prices: 收盤價序列
            highs: 最高價序列
            lows: 最低價序列
            volumes: 成交量序列
            
        Returns:
            交易訊號，若條件不滿足則返回 None
        """
        return self.generate(
            symbol=symbol,
            current_price=current_price,
            current_volume=current_volume,
            avg_volume_10d=avg_volume_10d,
            cup=cup,
            handle=handle,
            score=score,
            prices=prices,
            highs=highs,
            lows=lows,
            volumes=volumes
        )


def create_signal_generator_with_optimizer(
    score_threshold: float = 80.0,
    breakout_buffer: float = 0.005,
    volume_multiplier: float = 1.5
) -> SignalGenerator:
    """建立整合訊號優化器的訊號生成器
    
    此工廠函數會自動建立並整合 SignalOptimizer。
    
    Args:
        score_threshold: 吻合分數閾值 (0-100)，預設 80.0
        breakout_buffer: 突破緩衝比例，預設 0.005 (0.5%)
        volume_multiplier: 成交量倍數要求，預設 1.5
        
    Returns:
        整合訊號優化器的 SignalGenerator 實例
    """
    try:
        from pattern_quant.optimization.indicator_pool import IndicatorPool
        from pattern_quant.optimization.factor_config import FactorConfigManager
        from pattern_quant.optimization.signal_optimizer import SignalOptimizer
        
        indicator_pool = IndicatorPool()
        config_manager = FactorConfigManager()
        signal_optimizer = SignalOptimizer(indicator_pool, config_manager)
        
        return SignalGenerator(
            score_threshold=score_threshold,
            breakout_buffer=breakout_buffer,
            volume_multiplier=volume_multiplier,
            signal_optimizer=signal_optimizer,
            use_optimizer=True
        )
    except ImportError:
        # 如果優化模組不可用，返回基本的 SignalGenerator
        return SignalGenerator(
            score_threshold=score_threshold,
            breakout_buffer=breakout_buffer,
            volume_multiplier=volume_multiplier,
            signal_optimizer=None,
            use_optimizer=False
        )
