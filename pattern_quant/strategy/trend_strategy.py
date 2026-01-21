"""
趨勢策略 (Trend Strategy)

在趨勢市場中執行型態突破交易，整合 PatternEngine 進行型態識別。

Requirements:
- 4.1: 市場狀態為 Trend 且型態 Score > 80 時啟動突破監控
- 4.2: 價格突破頸線時生成買入訊號
- 4.3: 買入執行後設定止損於型態支撐位
- 4.4: 持倉獲利達到風險回報比 1:3 時啟動移動止損
- 4.5: 價格跌破型態支撐位時執行止損出場
"""

import logging
from dataclasses import dataclass
from typing import Optional, List

from pattern_quant.core.models import PatternResult, Position
from pattern_quant.core.pattern_engine import PatternEngine
from pattern_quant.strategy.models import (
    MarketState,
    TrendSignal,
    DualEngineConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TrendStrategyConfig:
    """趨勢策略配置"""
    score_threshold: float = 80.0  # 型態分數閾值
    risk_reward_target: float = 3.0  # 目標風險回報比
    trailing_activation: float = 0.10  # 移動止損啟動獲利比例 (10%)
    breakout_buffer: float = 0.005  # 突破緩衝 (0.5%)


class TrendStrategy:
    """
    趨勢策略
    
    在趨勢市場中執行型態突破交易：
    1. 監控型態分數 > 80 的標的
    2. 價格突破頸線時生成買入信號
    3. 設定止損於型態支撐位
    4. 獲利達 1:3 風險回報比時啟動移動止損
    """
    
    def __init__(
        self,
        score_threshold: float = 80.0,
        risk_reward_target: float = 3.0,
        trailing_activation: float = 0.10,
        breakout_buffer: float = 0.005,
        pattern_engine: Optional[PatternEngine] = None
    ):
        """
        初始化趨勢策略
        
        Args:
            score_threshold: 型態分數閾值（預設 80）
            risk_reward_target: 目標風險回報比（預設 3.0）
            trailing_activation: 移動止損啟動獲利比例（預設 10%）
            breakout_buffer: 突破緩衝比例（預設 0.5%）
            pattern_engine: 型態識別引擎（可選，若未提供則使用預設配置）
        """
        self.score_threshold = score_threshold
        self.risk_reward_target = risk_reward_target
        self.trailing_activation = trailing_activation
        self.breakout_buffer = breakout_buffer
        self.pattern_engine = pattern_engine or PatternEngine()
    
    @classmethod
    def from_config(cls, config: DualEngineConfig) -> "TrendStrategy":
        """
        從配置物件建立策略
        
        Args:
            config: 雙引擎策略配置
            
        Returns:
            配置好的策略實例
        """
        return cls(
            score_threshold=config.trend_score_threshold,
            risk_reward_target=config.trend_risk_reward,
            trailing_activation=config.trend_trailing_activation,
        )
    
    def check_entry(
        self,
        pattern_result: PatternResult,
        current_price: float,
        market_state: MarketState
    ) -> Optional[TrendSignal]:
        """
        檢查是否符合進場條件
        
        進場條件：
        1. 市場狀態為 TREND
        2. 型態有效且分數 > score_threshold
        3. 價格突破頸線（右杯緣 + 緩衝）
        
        Args:
            pattern_result: 型態識別結果
            current_price: 當前價格
            market_state: 市場狀態
            
        Returns:
            TrendSignal 若符合進場條件，否則 None
            
        Requirements:
            4.1: 市場狀態為 Trend 且型態 Score > 80 時啟動突破監控
            4.2: 價格突破頸線時生成買入訊號
            4.3: 買入執行後設定止損於型態支撐位
        """
        # 條件 1: 市場狀態必須為 TREND
        if market_state != MarketState.TREND:
            logger.debug(f"進場檢查失敗: 市場狀態為 {market_state.value}，非 TREND")
            return None
        
        # 條件 2: 型態必須有效
        if not pattern_result.is_valid:
            logger.debug(f"進場檢查失敗: 型態無效 - {pattern_result.rejection_reason}")
            return None
        
        # 條件 3: 型態分數必須超過閾值
        if pattern_result.score is None:
            logger.debug("進場檢查失敗: 型態分數為 None")
            return None
        
        pattern_score = pattern_result.score.total_score
        if pattern_score < self.score_threshold:
            logger.debug(f"進場檢查失敗: 型態分數 {pattern_score:.1f} < 閾值 {self.score_threshold}")
            return None
        
        # 取得頸線價格（右杯緣價格）
        if pattern_result.cup is None:
            logger.debug("進場檢查失敗: 無茶杯型態")
            return None
        
        neckline_price = pattern_result.cup.right_peak_price
        breakout_price = neckline_price * (1 + self.breakout_buffer)
        
        # 條件 4: 價格必須突破頸線
        if current_price < breakout_price:
            logger.debug(f"進場檢查失敗: 價格 {current_price:.2f} 未突破頸線 {breakout_price:.2f}")
            return None
        
        # 計算止損價格（型態支撐位 = 杯底或柄部最低點）
        stop_loss_price = self._calculate_stop_loss(pattern_result)
        
        # 計算風險回報比
        risk = current_price - stop_loss_price
        if risk <= 0:
            logger.warning("風險計算異常: 止損價格高於或等於進場價格")
            return None
        
        # 預期獲利 = 風險 × 目標風險回報比
        expected_profit = risk * self.risk_reward_target
        risk_reward_ratio = self.risk_reward_target
        
        logger.info(
            f"趨勢策略進場信號: {pattern_result.symbol} "
            f"價格={current_price:.2f}, 頸線={neckline_price:.2f}, "
            f"止損={stop_loss_price:.2f}, 風險回報比={risk_reward_ratio:.1f}"
        )
        
        return TrendSignal(
            symbol=pattern_result.symbol,
            signal_type="breakout_long",
            entry_price=current_price,
            stop_loss_price=stop_loss_price,
            neckline_price=neckline_price,
            pattern_score=pattern_score,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def check_exit(
        self,
        position: Position,
        current_price: float,
        entry_price: float,
        stop_loss_price: float
    ) -> Optional[str]:
        """
        檢查是否應該出場
        
        出場條件：
        1. 價格跌破止損價格 → 止損出場
        2. 移動止損已啟動且價格跌破移動止損價格 → 移動止損出場
        
        Args:
            position: 持倉資訊
            current_price: 當前價格
            entry_price: 進場價格
            stop_loss_price: 原始止損價格
            
        Returns:
            出場原因字串，若不應出場則返回 None
            
        Requirements:
            4.4: 持倉獲利達到風險回報比 1:3 時啟動移動止損
            4.5: 價格跌破型態支撐位時執行止損出場
        """
        # 檢查是否跌破原始止損
        if current_price <= stop_loss_price:
            logger.info(
                f"趨勢策略止損出場: {position.symbol} "
                f"價格={current_price:.2f} <= 止損={stop_loss_price:.2f}"
            )
            return "stop_loss"
        
        # 檢查移動止損
        if position.trailing_stop_active:
            if current_price <= position.trailing_stop_price:
                logger.info(
                    f"趨勢策略移動止損出場: {position.symbol} "
                    f"價格={current_price:.2f} <= 移動止損={position.trailing_stop_price:.2f}"
                )
                return "trailing_stop"
        
        return None
    
    def should_activate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        stop_loss_price: float
    ) -> bool:
        """
        檢查是否應該啟動移動止損
        
        當獲利達到風險回報比 1:3 時啟動移動止損。
        
        Args:
            entry_price: 進場價格
            current_price: 當前價格
            stop_loss_price: 止損價格
            
        Returns:
            True 若應該啟動移動止損
            
        Requirements:
            4.4: 持倉獲利達到風險回報比 1:3 時啟動移動止損
        """
        # 計算風險（進場價格到止損的距離）
        risk = entry_price - stop_loss_price
        if risk <= 0:
            return False
        
        # 計算當前獲利
        profit = current_price - entry_price
        
        # 計算當前風險回報比
        current_rr = profit / risk if risk > 0 else 0
        
        # 當風險回報比達到目標時啟動移動止損
        should_activate = current_rr >= self.risk_reward_target
        
        if should_activate:
            logger.debug(
                f"移動止損啟動條件達成: 風險回報比 {current_rr:.2f} >= {self.risk_reward_target}"
            )
        
        return should_activate
    
    def calculate_trailing_stop_price(
        self,
        entry_price: float,
        current_price: float,
        current_trailing_stop: Optional[float] = None
    ) -> float:
        """
        計算移動止損價格
        
        移動止損價格 = 當前價格 × (1 - trailing_activation)
        只會向上調整，不會向下調整。
        
        Args:
            entry_price: 進場價格
            current_price: 當前價格
            current_trailing_stop: 當前移動止損價格（可選）
            
        Returns:
            新的移動止損價格
        """
        # 計算新的移動止損價格
        new_trailing_stop = current_price * (1 - self.trailing_activation)
        
        # 確保移動止損至少在進場價格之上（保護獲利）
        new_trailing_stop = max(new_trailing_stop, entry_price)
        
        # 移動止損只會向上調整
        if current_trailing_stop is not None:
            new_trailing_stop = max(new_trailing_stop, current_trailing_stop)
        
        return new_trailing_stop
    
    def _calculate_stop_loss(self, pattern_result: PatternResult) -> float:
        """
        計算止損價格
        
        止損價格設定於型態支撐位：
        - 若有柄部，使用柄部最低點
        - 否則使用杯底價格
        
        Args:
            pattern_result: 型態識別結果
            
        Returns:
            止損價格
        """
        # 優先使用柄部最低點
        if pattern_result.handle is not None:
            return pattern_result.handle.lowest_price
        
        # 否則使用杯底價格
        if pattern_result.cup is not None:
            return pattern_result.cup.bottom_price
        
        # 異常情況：無法計算止損
        logger.warning("無法計算止損價格: 無茶杯或柄部型態")
        return 0.0
    
    def analyze(
        self,
        symbol: str,
        prices: List[float],
        volumes: List[float],
        current_price: float,
        market_state: MarketState
    ) -> Optional[TrendSignal]:
        """
        執行完整的趨勢策略分析
        
        整合 PatternEngine 進行型態識別，並檢查進場條件。
        
        Args:
            symbol: 股票代碼
            prices: 收盤價序列
            volumes: 成交量序列
            current_price: 當前價格
            market_state: 市場狀態
            
        Returns:
            TrendSignal 若符合進場條件，否則 None
        """
        # 使用 PatternEngine 進行型態識別
        pattern_result = self.pattern_engine.analyze_symbol(symbol, prices, volumes)
        
        # 檢查進場條件
        return self.check_entry(pattern_result, current_price, market_state)
