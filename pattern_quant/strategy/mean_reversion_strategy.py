"""
均值回歸策略 (Mean Reversion Strategy)

在震盪市場中執行布林均值回歸交易，整合 RSI 與 K 線型態確認。

Requirements:
- 5.1: 市場狀態為 Range 且價格觸及布林下軌時準備做多
- 5.2: 價格觸及布林下軌且 RSI < 30 時確認做多訊號
- 5.3: 價格觸及布林下軌且出現錘頭線時確認做多訊號
- 5.4: 做多訊號確認時以限價單掛在下軌價格
- 5.5: 持倉價格回歸布林中軌時賣出 50% 倉位
- 5.6: 持倉價格觸及布林上軌時清倉
- 6.1: 持有均值回歸倉位時 ADX 飆升超過 25 則強制止損
- 12.1: 識別錘頭線型態
- 12.2: 驗證下影線長度至少為實體的 2 倍
- 12.3: 驗證上影線極短或不存在
- 12.4: 返回訊號類型與可信度分數
"""

import logging
from typing import Optional, Tuple

from pattern_quant.strategy.models import (
    MarketState,
    CandlePattern,
    MeanReversionSignal,
    DualEngineConfig,
)
from pattern_quant.optimization.models import BollingerResult, RSIResult

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """
    均值回歸策略
    
    在震盪市場中執行布林均值回歸交易：
    1. 價格觸及布林下軌時準備做多
    2. RSI < 30 或出現錘頭線時確認進場
    3. 價格回歸中軌時賣出 50%
    4. 價格觸及上軌時清倉
    5. ADX 飆升時強制止損（風控覆蓋）
    """
    
    def __init__(
        self,
        rsi_oversold: float = 30.0,
        partial_exit_ratio: float = 0.5,
        adx_override_threshold: float = 25.0,
        lower_band_buffer: float = 0.005,
        stop_loss_pct: float = 0.03
    ):
        """
        初始化均值回歸策略
        
        Args:
            rsi_oversold: RSI 超賣閾值（預設 30）
            partial_exit_ratio: 部分出場比例（預設 50%）
            adx_override_threshold: ADX 風控覆蓋閾值（預設 25）
            lower_band_buffer: 下軌觸及緩衝比例（預設 0.5%）
            stop_loss_pct: 止損比例（預設 3%）
        """
        self.rsi_oversold = rsi_oversold
        self.partial_exit_ratio = partial_exit_ratio
        self.adx_override_threshold = adx_override_threshold
        self.lower_band_buffer = lower_band_buffer
        self.stop_loss_pct = stop_loss_pct
    
    @classmethod
    def from_config(cls, config: DualEngineConfig) -> "MeanReversionStrategy":
        """
        從配置物件建立策略
        
        Args:
            config: 雙引擎策略配置
            
        Returns:
            配置好的策略實例
        """
        return cls(
            rsi_oversold=config.reversion_rsi_oversold,
            partial_exit_ratio=config.reversion_partial_exit,
            adx_override_threshold=config.reversion_adx_override,
        )
    
    def detect_hammer(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float
    ) -> Optional[CandlePattern]:
        """
        識別錘頭線型態
        
        錘頭線條件：
        1. 下影線長度 >= 實體長度 × 2
        2. 上影線極短（<= 實體長度 × 0.1）
        
        Args:
            open_price: 開盤價
            high: 最高價
            low: 最低價
            close: 收盤價
            
        Returns:
            CandlePattern 若識別為錘頭線，否則 None
            
        Requirements:
            12.1: 識別錘頭線型態
            12.2: 驗證下影線長度至少為實體的 2 倍
            12.3: 驗證上影線極短或不存在
            12.4: 返回訊號類型與可信度分數
        """
        # 計算實體長度（絕對值）
        body = abs(close - open_price)
        
        # 處理十字線情況（實體極小）
        if body < 0.0001:
            # 十字線不是錘頭線
            return None
        
        # 計算上下影線
        body_top = max(open_price, close)
        body_bottom = min(open_price, close)
        
        upper_shadow = high - body_top
        lower_shadow = body_bottom - low
        
        # 條件 1: 下影線長度 >= 實體長度 × 2
        if lower_shadow < body * 2 - 0.0001:  # 使用小容差處理浮點數精度
            return None
        
        # 條件 2: 上影線極短（<= 實體長度 × 0.1）
        if upper_shadow > body * 0.1:
            return None
        
        # 計算可信度分數
        # 下影線越長，可信度越高
        lower_shadow_ratio = lower_shadow / body if body > 0 else 0
        # 上影線越短，可信度越高
        upper_shadow_ratio = upper_shadow / body if body > 0 else 0
        
        # 可信度計算：基礎分 0.6，下影線比例每增加 1 倍加 0.1，最高 1.0
        # 上影線會造成懲罰
        base_confidence = 0.6
        lower_bonus = min((lower_shadow_ratio - 2) * 0.1, 0.35)
        upper_penalty = upper_shadow_ratio * 0.3
        
        confidence = min(max(base_confidence + lower_bonus - upper_penalty, 0.6), 1.0)
        
        logger.debug(
            f"錘頭線識別: 實體={body:.4f}, 下影線={lower_shadow:.4f}, "
            f"上影線={upper_shadow:.4f}, 可信度={confidence:.2f}"
        )
        
        return CandlePattern(
            pattern_type="hammer",
            confidence=confidence
        )
    
    def check_entry(
        self,
        current_price: float,
        bollinger: BollingerResult,
        rsi: Optional[RSIResult],
        candle: Optional[CandlePattern],
        market_state: MarketState,
        symbol: str = "UNKNOWN"
    ) -> Optional[MeanReversionSignal]:
        """
        檢查是否符合進場條件
        
        進場條件：
        1. 市場狀態為 RANGE
        2. 價格觸及布林下軌（含緩衝）
        3. RSI < 30 或出現錘頭線（至少一個確認）
        
        Args:
            current_price: 當前價格
            bollinger: 布林通道計算結果
            rsi: RSI 計算結果（可選）
            candle: K 線型態（可選）
            market_state: 市場狀態
            symbol: 股票代碼
            
        Returns:
            MeanReversionSignal 若符合進場條件，否則 None
            
        Requirements:
            5.1: 市場狀態為 Range 且價格觸及布林下軌時準備做多
            5.2: 價格觸及布林下軌且 RSI < 30 時確認做多訊號
            5.3: 價格觸及布林下軌且出現錘頭線時確認做多訊號
            5.4: 做多訊號確認時以限價單掛在下軌價格
        """
        # 條件 1: 市場狀態必須為 RANGE
        if market_state != MarketState.RANGE:
            logger.debug(f"進場檢查失敗: 市場狀態為 {market_state.value}，非 RANGE")
            return None
        
        # 條件 2: 價格必須觸及布林下軌（含緩衝）
        lower_band_with_buffer = bollinger.lower * (1 + self.lower_band_buffer)
        if current_price > lower_band_with_buffer:
            logger.debug(
                f"進場檢查失敗: 價格 {current_price:.2f} 未觸及下軌 "
                f"{bollinger.lower:.2f} (緩衝後 {lower_band_with_buffer:.2f})"
            )
            return None
        
        # 條件 3: 需要至少一個確認訊號
        rsi_confirmed = rsi is not None and rsi.value < self.rsi_oversold
        hammer_confirmed = candle is not None and candle.pattern_type == "hammer"
        
        if not rsi_confirmed and not hammer_confirmed:
            logger.debug(
                f"進場檢查失敗: 無確認訊號 "
                f"(RSI={rsi.value if rsi else 'N/A'}, 錘頭線={hammer_confirmed})"
            )
            return None
        
        # 決定確認方式
        if rsi_confirmed and hammer_confirmed:
            confirmation = "both"
        elif rsi_confirmed:
            confirmation = "rsi_oversold"
        else:
            confirmation = "hammer"
        
        # 計算止損價格（下軌下方 stop_loss_pct）
        stop_loss_price = bollinger.lower * (1 - self.stop_loss_pct)
        
        logger.info(
            f"均值回歸進場信號: {symbol} "
            f"價格={current_price:.2f}, 下軌={bollinger.lower:.2f}, "
            f"中軌={bollinger.middle:.2f}, 上軌={bollinger.upper:.2f}, "
            f"確認方式={confirmation}"
        )
        
        return MeanReversionSignal(
            symbol=symbol,
            signal_type="long_entry",
            entry_price=bollinger.lower,  # 限價單掛在下軌
            target_1=bollinger.middle,    # 第一目標：中軌
            target_2=bollinger.upper,     # 第二目標：上軌
            stop_loss_price=stop_loss_price,
            confirmation=confirmation
        )
    
    def check_exit(
        self,
        current_price: float,
        bollinger: BollingerResult,
        current_adx: float,
        has_partial_exited: bool = False
    ) -> Optional[Tuple[str, float]]:
        """
        檢查是否應該出場
        
        出場條件：
        1. 價格回歸布林中軌 → 賣出 50%（部分出場）
        2. 價格觸及布林上軌 → 清倉（全部出場）
        3. ADX 飆升超過閾值 → 風控覆蓋強制出場
        
        Args:
            current_price: 當前價格
            bollinger: 布林通道計算結果
            current_adx: 當前 ADX 值
            has_partial_exited: 是否已執行部分出場
            
        Returns:
            (exit_type, exit_ratio) 或 None
            - exit_type: "partial_exit", "full_exit", "risk_override"
            - exit_ratio: 出場比例 (0-1)
            
        Requirements:
            5.5: 持倉價格回歸布林中軌時賣出 50% 倉位
            5.6: 持倉價格觸及布林上軌時清倉
            6.1: ADX 飆升超過 25 時強制止損
        """
        # 優先檢查風控覆蓋
        if self.check_risk_override(current_adx):
            logger.warning(
                f"風控覆蓋觸發: ADX={current_adx:.2f} > {self.adx_override_threshold}"
            )
            return ("risk_override", 1.0)
        
        # 檢查是否觸及上軌（全部出場）
        if current_price >= bollinger.upper:
            logger.info(
                f"均值回歸全部出場: 價格 {current_price:.2f} >= 上軌 {bollinger.upper:.2f}"
            )
            return ("full_exit", 1.0)
        
        # 檢查是否回歸中軌（部分出場）
        if not has_partial_exited and current_price >= bollinger.middle:
            logger.info(
                f"均值回歸部分出場: 價格 {current_price:.2f} >= 中軌 {bollinger.middle:.2f}"
            )
            return ("partial_exit", self.partial_exit_ratio)
        
        return None
    
    def check_risk_override(self, current_adx: float) -> bool:
        """
        檢查是否觸發風控覆蓋
        
        當 ADX 飆升超過閾值時，表示市場可能從震盪轉為趨勢，
        應立即強制止損以避免逆勢持倉。
        
        Args:
            current_adx: 當前 ADX 值
            
        Returns:
            True 如果應該強制止損
            
        Requirements:
            6.1: 持有均值回歸倉位時 ADX 飆升超過 25 則強制止損
        """
        return current_adx > self.adx_override_threshold

