"""
雙引擎策略主類別 (Dual Engine Strategy)

整合市場狀態分類器、趨勢策略、均值回歸策略、信號解決器與倉位計算器，
提供統一的 analyze() 介面進行市場分析與交易信號生成。

Requirements:
- 3.4: 市場狀態判定完成時返回狀態名稱與對應的資金權重
- 4.1: 市場狀態為 Trend 且型態 Score > 80 時啟動突破監控
- 5.1: 市場狀態為 Range 且價格觸及布林下軌時準備做多
- 9.1: 同一標的同時出現趨勢信號與回歸信號時執行「趨勢優先」原則
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

from pattern_quant.core.models import PatternResult
from pattern_quant.core.pattern_engine import PatternEngine
from pattern_quant.optimization.models import BollingerResult, RSIResult, IndicatorSnapshot
from pattern_quant.optimization.indicator_pool import IndicatorPool

from pattern_quant.strategy.models import (
    MarketState,
    MarketStateResult,
    TrendSignal,
    MeanReversionSignal,
    CandlePattern,
    DualEngineConfig,
)
from pattern_quant.strategy.market_classifier import MarketStateClassifier
from pattern_quant.strategy.trend_strategy import TrendStrategy
from pattern_quant.strategy.mean_reversion_strategy import MeanReversionStrategy
from pattern_quant.strategy.signal_resolver import SignalResolver, ResolvedSignal, ConflictType
from pattern_quant.strategy.position_sizer import (
    ATRPositionSizer,
    FixedFractionalSizer,
    PositionSizeResult,
)

logger = logging.getLogger(__name__)


@dataclass
class DualEngineAnalysisResult:
    """雙引擎分析結果
    
    Attributes:
        symbol: 股票代碼
        market_state: 市場狀態判定結果
        resolved_signal: 解決後的交易信號
        position_size: 倉位計算結果（若有信號）
        trend_signal: 原始趨勢策略信號（若有）
        reversion_signal: 原始均值回歸信號（若有）
        indicators: 指標快照
        allocation_weight: 資金權重
        adx_change_rate: ADX 變化率
    """
    symbol: str
    market_state: MarketStateResult
    resolved_signal: ResolvedSignal
    position_size: Optional[PositionSizeResult] = None
    trend_signal: Optional[TrendSignal] = None
    reversion_signal: Optional[MeanReversionSignal] = None
    indicators: Optional[IndicatorSnapshot] = None
    allocation_weight: float = 0.0
    adx_change_rate: float = 0.0


class DualEngineStrategy:
    """
    雙引擎策略主類別
    
    整合所有策略組件，提供統一的分析介面：
    1. 市場狀態分類 (MarketStateClassifier)
    2. 趨勢策略 (TrendStrategy)
    3. 均值回歸策略 (MeanReversionStrategy)
    4. 信號解決器 (SignalResolver)
    5. 倉位計算器 (ATRPositionSizer, FixedFractionalSizer)
    
    Requirements:
        3.4: 市場狀態判定完成時返回狀態名稱與對應的資金權重
        4.1: 市場狀態為 Trend 且型態 Score > 80 時啟動突破監控
        5.1: 市場狀態為 Range 且價格觸及布林下軌時準備做多
        9.1: 同一標的同時出現趨勢信號與回歸信號時執行「趨勢優先」原則
    """
    
    def __init__(
        self,
        config: Optional[DualEngineConfig] = None,
        pattern_engine: Optional[PatternEngine] = None,
        indicator_pool: Optional[IndicatorPool] = None,
    ):
        """
        初始化雙引擎策略
        
        Args:
            config: 雙引擎配置，若未提供則使用預設配置
            pattern_engine: 型態識別引擎，若未提供則建立新實例
            indicator_pool: 指標計算庫，若未提供則建立新實例
        """
        self.config = config or DualEngineConfig()
        self.pattern_engine = pattern_engine or PatternEngine()
        self.indicator_pool = indicator_pool or IndicatorPool()
        
        # 初始化各組件
        self._init_components()
        
        # 追蹤前一次 ADX 值（用於計算變化率）
        self._prev_adx: Dict[str, float] = {}
    
    def _init_components(self) -> None:
        """根據配置初始化各策略組件"""
        # 市場狀態分類器
        self.market_classifier = MarketStateClassifier(
            trend_threshold=self.config.adx_trend_threshold,
            range_threshold=self.config.adx_range_threshold,
            bbw_stability_threshold=self.config.bbw_stability_threshold,
        )
        
        # 趨勢策略
        self.trend_strategy = TrendStrategy(
            score_threshold=self.config.trend_score_threshold,
            risk_reward_target=self.config.trend_risk_reward,
            trailing_activation=self.config.trend_trailing_activation,
            pattern_engine=self.pattern_engine,
        )
        
        # 均值回歸策略
        self.mean_reversion_strategy = MeanReversionStrategy(
            rsi_oversold=self.config.reversion_rsi_oversold,
            partial_exit_ratio=self.config.reversion_partial_exit,
            adx_override_threshold=self.config.reversion_adx_override,
        )
        
        # 信號解決器
        self.signal_resolver = SignalResolver(
            adx_surge_threshold=self.config.adx_trend_threshold,
        )
        
        # 倉位計算器
        self.atr_position_sizer = ATRPositionSizer(
            risk_per_trade=self.config.trend_risk_per_trade,
        )
        self.fixed_fractional_sizer = FixedFractionalSizer(
            position_ratio=self.config.reversion_position_ratio,
        )
    
    def update_config(self, config: DualEngineConfig) -> None:
        """
        更新配置並重新初始化組件
        
        Args:
            config: 新的雙引擎配置
        """
        self.config = config
        self._init_components()
    
    def analyze(
        self,
        symbol: str,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        total_capital: float = 100000.0,
        ohlc_for_candle: Optional[Dict[str, float]] = None,
    ) -> DualEngineAnalysisResult:
        """
        執行完整的雙引擎策略分析
        
        分析流程：
        1. 計算市場狀態（ADX、BBW）
        2. 根據市場狀態執行對應策略
        3. 解決可能的信號衝突
        4. 計算建議倉位
        
        Args:
            symbol: 股票代碼
            prices: 收盤價序列
            highs: 最高價序列
            lows: 最低價序列
            volumes: 成交量序列
            total_capital: 總資金（用於倉位計算）
            ohlc_for_candle: 最新 K 線數據（用於錘頭線識別）
                - open: 開盤價
                - high: 最高價
                - low: 最低價
                - close: 收盤價
            
        Returns:
            DualEngineAnalysisResult 包含完整分析結果
            
        Requirements:
            3.4: 市場狀態判定完成時返回狀態名稱與對應的資金權重
            4.1: 市場狀態為 Trend 且型態 Score > 80 時啟動突破監控
            5.1: 市場狀態為 Range 且價格觸及布林下軌時準備做多
            9.1: 同一標的同時出現趨勢信號與回歸信號時執行「趨勢優先」原則
        """
        if not self.config.enabled:
            # 雙引擎模式未啟用，返回空結果
            return self._create_disabled_result(symbol)
        
        current_price = prices[-1] if prices else 0.0
        
        # Step 1: 計算市場狀態
        market_state_result = self.market_classifier.classify(highs, lows, prices)
        
        # 計算 ADX 變化率
        current_adx = market_state_result.adx_result.adx
        adx_change_rate = self._calculate_adx_change_rate(symbol, current_adx)
        
        # 取得資金權重
        allocation_weight = self._get_allocation_weight(market_state_result.state)
        
        # Step 2: 計算指標快照
        indicators = self.indicator_pool.get_snapshot(prices, highs, lows, volumes)
        
        # Step 3: 執行趨勢策略（若市場狀態為 TREND）
        trend_signal = self._check_trend_strategy(
            symbol, prices, volumes, current_price, market_state_result.state
        )
        
        # Step 4: 執行均值回歸策略（若市場狀態為 RANGE）
        reversion_signal = self._check_mean_reversion_strategy(
            symbol, current_price, indicators, market_state_result.state, ohlc_for_candle
        )
        
        # Step 5: 解決信號衝突
        resolved_signal = self.signal_resolver.resolve(
            trend_signal=trend_signal,
            reversion_signal=reversion_signal,
            current_adx=current_adx,
            adx_change_rate=adx_change_rate,
        )
        
        # Step 6: 計算倉位
        position_size = self._calculate_position_size(
            resolved_signal=resolved_signal,
            total_capital=total_capital,
            allocation_weight=allocation_weight,
            atr=indicators.atr,
        )
        
        logger.info(
            f"雙引擎分析完成: {symbol} "
            f"狀態={market_state_result.state.value}, "
            f"ADX={current_adx:.2f}, "
            f"策略={resolved_signal.strategy_type}, "
            f"權重={allocation_weight:.0%}"
        )
        
        return DualEngineAnalysisResult(
            symbol=symbol,
            market_state=market_state_result,
            resolved_signal=resolved_signal,
            position_size=position_size,
            trend_signal=trend_signal,
            reversion_signal=reversion_signal,
            indicators=indicators,
            allocation_weight=allocation_weight,
            adx_change_rate=adx_change_rate,
        )
    
    def _create_disabled_result(self, symbol: str) -> DualEngineAnalysisResult:
        """建立雙引擎未啟用時的空結果"""
        from pattern_quant.strategy.models import ADXResult, BBWResult
        
        empty_market_state = MarketStateResult(
            state=MarketState.NOISE,
            allocation_weight=0.0,
            adx_result=ADXResult(adx=0.0, plus_di=0.0, minus_di=0.0),
            bbw_result=BBWResult(bandwidth=0.0, avg_bandwidth=0.0, is_squeeze=False, change_rate=0.0),
            confidence=0.0,
        )
        
        empty_resolved = ResolvedSignal(
            signal=None,
            strategy_type="none",
            conflict_type=ConflictType.NONE,
            resolution_reason="雙引擎模式未啟用",
        )
        
        return DualEngineAnalysisResult(
            symbol=symbol,
            market_state=empty_market_state,
            resolved_signal=empty_resolved,
        )
    
    def _calculate_adx_change_rate(self, symbol: str, current_adx: float) -> float:
        """計算 ADX 變化率"""
        prev_adx = self._prev_adx.get(symbol)
        self._prev_adx[symbol] = current_adx
        
        if prev_adx is None or prev_adx == 0:
            return 0.0
        
        return (current_adx - prev_adx) / prev_adx
    
    def _get_allocation_weight(self, state: MarketState) -> float:
        """根據市場狀態取得資金權重"""
        if state == MarketState.TREND:
            return self.config.trend_allocation
        elif state == MarketState.RANGE:
            return self.config.range_allocation
        else:
            return self.config.noise_allocation
    
    def _check_trend_strategy(
        self,
        symbol: str,
        prices: List[float],
        volumes: List[float],
        current_price: float,
        market_state: MarketState,
    ) -> Optional[TrendSignal]:
        """執行趨勢策略檢查"""
        if market_state != MarketState.TREND:
            return None
        
        # 使用 PatternEngine 進行型態識別
        pattern_result = self.pattern_engine.analyze_symbol(symbol, prices, volumes)
        
        # 檢查進場條件
        return self.trend_strategy.check_entry(
            pattern_result=pattern_result,
            current_price=current_price,
            market_state=market_state,
        )
    
    def _check_mean_reversion_strategy(
        self,
        symbol: str,
        current_price: float,
        indicators: IndicatorSnapshot,
        market_state: MarketState,
        ohlc_for_candle: Optional[Dict[str, float]] = None,
    ) -> Optional[MeanReversionSignal]:
        """執行均值回歸策略檢查"""
        if market_state != MarketState.RANGE:
            return None
        
        if indicators.bollinger is None:
            return None
        
        # 識別錘頭線（若有 K 線數據）
        candle_pattern = None
        if ohlc_for_candle:
            candle_pattern = self.mean_reversion_strategy.detect_hammer(
                open_price=ohlc_for_candle.get("open", current_price),
                high=ohlc_for_candle.get("high", current_price),
                low=ohlc_for_candle.get("low", current_price),
                close=ohlc_for_candle.get("close", current_price),
            )
        
        # 構建 RSI 結果
        rsi_result = indicators.rsi
        
        # 檢查進場條件
        return self.mean_reversion_strategy.check_entry(
            current_price=current_price,
            bollinger=indicators.bollinger,
            rsi=rsi_result,
            candle=candle_pattern,
            market_state=market_state,
            symbol=symbol,
        )
    
    def _calculate_position_size(
        self,
        resolved_signal: ResolvedSignal,
        total_capital: float,
        allocation_weight: float,
        atr: Optional[float],
    ) -> Optional[PositionSizeResult]:
        """計算倉位大小"""
        if resolved_signal.signal is None:
            return None
        
        # 調整後的資金
        adjusted_capital = total_capital * allocation_weight
        
        if resolved_signal.strategy_type == "trend":
            # 趨勢策略使用 ATR 倉位計算
            signal = resolved_signal.signal
            if not isinstance(signal, TrendSignal):
                return None
            
            if atr is None or atr <= 0:
                return None
            
            return self.atr_position_sizer.calculate(
                total_capital=adjusted_capital,
                entry_price=signal.entry_price,
                atr=atr,
            )
        
        elif resolved_signal.strategy_type == "mean_reversion":
            # 均值回歸策略使用固定金額倉位計算
            signal = resolved_signal.signal
            if not isinstance(signal, MeanReversionSignal):
                return None
            
            return self.fixed_fractional_sizer.calculate(
                total_capital=adjusted_capital,
                entry_price=signal.entry_price,
            )
        
        return None
    
    def get_market_state(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
    ) -> MarketStateResult:
        """
        僅取得市場狀態（不執行完整分析）
        
        Args:
            highs: 最高價序列
            lows: 最低價序列
            closes: 收盤價序列
            
        Returns:
            MarketStateResult 市場狀態判定結果
        """
        return self.market_classifier.classify(highs, lows, closes)
    
    def is_enabled(self) -> bool:
        """檢查雙引擎模式是否啟用"""
        return self.config.enabled
