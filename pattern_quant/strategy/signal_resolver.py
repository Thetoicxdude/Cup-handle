"""
信號解決器 (Signal Resolver)

處理趨勢策略與均值回歸策略之間的信號衝突。

Requirements:
- 9.1: 同一標的同時出現趨勢信號與回歸信號時執行「趨勢優先」原則
- 9.2: 價格跌到布林下軌但 ADX 開始暴衝時判定為「向下突破」
- 9.3: 判定為向下突破時禁止執行回歸買入
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from pattern_quant.strategy.models import (
    TrendSignal,
    MeanReversionSignal,
)

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """衝突類型
    
    Attributes:
        NONE: 無衝突
        TREND_VS_REVERSION: 趨勢信號與回歸信號同時存在
        DOWNWARD_BREAKOUT: 向下突破情境（價格跌到下軌但 ADX 暴衝）
    """
    NONE = "none"
    TREND_VS_REVERSION = "trend_vs_reversion"
    DOWNWARD_BREAKOUT = "downward_breakout"


@dataclass
class ResolvedSignal:
    """解決後的信號
    
    Attributes:
        signal: 最終採用的信號（TrendSignal、MeanReversionSignal 或 None）
        strategy_type: 策略類型 ("trend", "mean_reversion", "none")
        conflict_type: 衝突類型
        resolution_reason: 解決原因說明
    """
    signal: Union[TrendSignal, MeanReversionSignal, None]
    strategy_type: str
    conflict_type: ConflictType
    resolution_reason: str


class SignalResolver:
    """
    信號解決器
    
    處理策略信號衝突，核心原則：
    1. 趨勢優先 (Trend Trumps Range)
    2. 向下突破時禁止回歸買入
    
    Requirements:
        9.1: 同一標的同時出現趨勢信號與回歸信號時執行「趨勢優先」原則
        9.2: 價格跌到布林下軌但 ADX 開始暴衝時判定為「向下突破」
        9.3: 判定為向下突破時禁止執行回歸買入
    """
    
    def __init__(
        self,
        adx_surge_threshold: float = 25.0,
        adx_change_rate_threshold: float = 0.20
    ):
        """
        初始化信號解決器
        
        Args:
            adx_surge_threshold: ADX 暴衝閾值（預設 25）
            adx_change_rate_threshold: ADX 變化率閾值（預設 20%）
        """
        self.adx_surge_threshold = adx_surge_threshold
        self.adx_change_rate_threshold = adx_change_rate_threshold
    
    def resolve(
        self,
        trend_signal: Optional[TrendSignal],
        reversion_signal: Optional[MeanReversionSignal],
        current_adx: float,
        adx_change_rate: float
    ) -> ResolvedSignal:
        """
        解決信號衝突
        
        原則：趨勢優先 (Trend Trumps Range)
        
        解決邏輯：
        1. 若無任何信號，返回空結果
        2. 若只有趨勢信號，直接採用
        3. 若只有回歸信號，檢查是否為向下突破情境
        4. 若兩者同時存在，執行趨勢優先原則
        
        Args:
            trend_signal: 趨勢策略信號（可選）
            reversion_signal: 均值回歸策略信號（可選）
            current_adx: 當前 ADX 值
            adx_change_rate: ADX 變化率（正值表示上升）
            
        Returns:
            ResolvedSignal 包含最終信號與解決原因
            
        Requirements:
            9.1: 同一標的同時出現趨勢信號與回歸信號時執行「趨勢優先」原則
            9.2: 價格跌到布林下軌但 ADX 開始暴衝時判定為「向下突破」
            9.3: 判定為向下突破時禁止執行回歸買入
        """
        # 情況 1: 無任何信號
        if trend_signal is None and reversion_signal is None:
            logger.debug("信號解決: 無任何信號")
            return ResolvedSignal(
                signal=None,
                strategy_type="none",
                conflict_type=ConflictType.NONE,
                resolution_reason="無交易信號"
            )
        
        # 情況 2: 只有趨勢信號
        if trend_signal is not None and reversion_signal is None:
            logger.debug(f"信號解決: 採用趨勢信號 ({trend_signal.symbol})")
            return ResolvedSignal(
                signal=trend_signal,
                strategy_type="trend",
                conflict_type=ConflictType.NONE,
                resolution_reason="僅有趨勢信號，直接採用"
            )
        
        # 情況 3: 只有回歸信號 - 需檢查向下突破情境
        if trend_signal is None and reversion_signal is not None:
            # 檢查是否為向下突破情境
            is_downward_breakout = self._is_downward_breakout(
                current_adx, adx_change_rate
            )
            
            if is_downward_breakout:
                logger.warning(
                    f"信號解決: 向下突破情境，禁止回歸買入 "
                    f"(ADX={current_adx:.2f}, 變化率={adx_change_rate:.2%})"
                )
                return ResolvedSignal(
                    signal=None,
                    strategy_type="none",
                    conflict_type=ConflictType.DOWNWARD_BREAKOUT,
                    resolution_reason=(
                        f"向下突破情境：ADX={current_adx:.2f} 暴衝中 "
                        f"(變化率={adx_change_rate:.2%})，禁止回歸買入"
                    )
                )
            
            logger.debug(f"信號解決: 採用回歸信號 ({reversion_signal.symbol})")
            return ResolvedSignal(
                signal=reversion_signal,
                strategy_type="mean_reversion",
                conflict_type=ConflictType.NONE,
                resolution_reason="僅有回歸信號，無向下突破風險，採用回歸策略"
            )
        
        # 情況 4: 兩者同時存在 - 執行趨勢優先原則
        logger.info(
            f"信號解決: 趨勢與回歸信號衝突，執行趨勢優先原則 "
            f"(趨勢: {trend_signal.symbol}, 回歸: {reversion_signal.symbol})"
        )
        return ResolvedSignal(
            signal=trend_signal,
            strategy_type="trend",
            conflict_type=ConflictType.TREND_VS_REVERSION,
            resolution_reason="趨勢與回歸信號衝突，執行趨勢優先原則"
        )
    
    def _is_downward_breakout(
        self,
        current_adx: float,
        adx_change_rate: float
    ) -> bool:
        """
        判斷是否為向下突破情境
        
        向下突破條件：
        1. ADX 超過暴衝閾值（表示趨勢正在形成）
        2. ADX 變化率為正且超過閾值（表示 ADX 正在快速上升）
        
        Args:
            current_adx: 當前 ADX 值
            adx_change_rate: ADX 變化率
            
        Returns:
            True 若判定為向下突破情境
            
        Requirements:
            9.2: 價格跌到布林下軌但 ADX 開始暴衝時判定為「向下突破」
        """
        # 條件 1: ADX 超過暴衝閾值
        adx_surging = current_adx > self.adx_surge_threshold
        
        # 條件 2: ADX 變化率為正且超過閾值
        adx_rising_fast = adx_change_rate > self.adx_change_rate_threshold
        
        is_breakout = adx_surging and adx_rising_fast
        
        if is_breakout:
            logger.debug(
                f"向下突破判定: ADX={current_adx:.2f} > {self.adx_surge_threshold}, "
                f"變化率={adx_change_rate:.2%} > {self.adx_change_rate_threshold:.2%}"
            )
        
        return is_breakout
