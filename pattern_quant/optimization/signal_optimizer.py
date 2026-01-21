"""
訊號優化器 (Signal Optimizer)

核心評分引擎，整合型態分數與技術指標分數，計算最終交易評分。
支援雙引擎模式與因子權重的整合計算。

Requirements:
- 13.5: 雙引擎模式與因子權重同時啟用時整合計算最終訊號分數
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING

from enum import Enum

from .models import (
    RSIResult,
    MACDResult,
    BollingerResult,
    IndicatorSnapshot,
)
from .indicator_pool import IndicatorPool
from .factor_config import (
    FactorConfig,
    FactorConfigManager,
    RSIConfig,
    VolumeConfig,
    MACDConfig,
    EMAConfig,
    BollingerConfig,
)

if TYPE_CHECKING:
    from pattern_quant.strategy.models import DualEngineConfig, MarketState
    from pattern_quant.strategy.dual_engine import DualEngineAnalysisResult


class SignalStrength(Enum):
    """訊號強度列舉"""
    STRONG_BUY = "strong_buy"
    WATCH = "watch"
    SKIP = "skip"


@dataclass
class ScoreDetail:
    """分數明細"""
    source: str           # 分數來源 (pattern, rsi, volume, etc.)
    raw_value: float      # 原始指標值
    score_change: float   # 分數變化
    reason: str           # 原因說明


@dataclass
class OptimizedSignal:
    """優化後的訊號"""
    symbol: str
    pattern_score: float          # 原始型態分數
    final_score: float            # 最終分數
    strength: SignalStrength      # 訊號強度
    details: List[ScoreDetail] = field(default_factory=list)  # 分數明細
    recommended_stop_loss: float = 0.0  # 建議止損價
    indicators: Optional[IndicatorSnapshot] = None  # 指標快照
    # 雙引擎整合欄位
    dual_engine_enabled: bool = False  # 是否啟用雙引擎模式
    market_state: Optional[str] = None  # 市場狀態 (trend, range, noise)
    allocation_weight: float = 1.0  # 資金權重
    dual_engine_adjusted_score: Optional[float] = None  # 雙引擎調整後分數


class SignalOptimizer:
    """
    訊號優化器
    
    整合型態識別分數與多種技術指標，計算最終交易評分。
    """
    
    def __init__(
        self,
        indicator_pool: IndicatorPool,
        config_manager: FactorConfigManager,
    ):
        """
        初始化訊號優化器
        
        Args:
            indicator_pool: 指標計算庫
            config_manager: 因子配置管理器
        """
        self.indicator_pool = indicator_pool
        self.config_manager = config_manager

    def _calculate_rsi_score(
        self,
        rsi: RSIResult,
        config: RSIConfig,
        has_divergence: bool = False,
    ) -> Tuple[float, List[ScoreDetail]]:
        """
        計算 RSI 相關分數
        
        評分邏輯：
        1. RSI 在趨勢區間 (40-70)：健康動能，加分
        2. RSI 超買 (>=80)：風險較高，扣分
        3. RSI 在超賣反彈區 (30-40)：潛在買入機會，加分
        4. RSI 極度超賣 (<30)：可能繼續下跌，不加分
        5. RSI 從支撐區反彈：確認反轉，加分
        6. RSI 頂背離：趨勢可能反轉，扣分
        
        Args:
            rsi: RSI 計算結果
            config: RSI 配置
            has_divergence: 是否存在 RSI 背離
            
        Returns:
            (總分數變化, 分數明細列表)
        """
        if not config.enabled:
            return 0.0, []
        
        total_score = 0.0
        details: List[ScoreDetail] = []
        
        # 趨勢區間加分 (健康的上升動能)
        if rsi.trend_zone:
            score = config.trend_zone_bonus * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="rsi",
                raw_value=rsi.value,
                score_change=score,
                reason=f"RSI 在趨勢區間 ({config.trend_lower}-{config.trend_upper})",
            ))
        
        # 超買扣分 (風險較高)
        if rsi.is_overbought:
            score = config.overbought_penalty * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="rsi",
                raw_value=rsi.value,
                score_change=score,
                reason=f"RSI 超買 (>= {config.overbought})",
            ))
        
        # 超賣反彈區加分 (RSI 在 30-40 之間，潛在買入機會)
        # 注意：這裡改為加分而非扣分
        if config.oversold < rsi.value < config.trend_lower:
            # 給予小幅加分，因為這可能是超賣反彈的好時機
            score = config.support_bounce_bonus * 0.5 * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="rsi",
                raw_value=rsi.value,
                score_change=score,
                reason=f"RSI 在超賣反彈區 ({config.oversold}-{config.trend_lower})",
            ))
        
        # 極度超賣時輕微扣分 (RSI < 30，可能繼續下跌)
        if rsi.is_oversold:
            score = config.weak_penalty * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="rsi",
                raw_value=rsi.value,
                score_change=score,
                reason=f"RSI 極度超賣 (<= {config.oversold})",
            ))
        
        # 支撐反彈加分 (確認反轉)
        if rsi.support_bounce:
            score = config.support_bounce_bonus * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="rsi",
                raw_value=rsi.value,
                score_change=score,
                reason="RSI 從支撐區反彈",
            ))
        
        # 背離扣分 (趨勢可能反轉)
        if config.check_divergence and has_divergence:
            score = config.divergence_penalty * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="rsi",
                raw_value=rsi.value,
                score_change=score,
                reason="RSI 頂背離",
            ))
        
        return total_score, details


    def _calculate_volume_score(
        self,
        volume_ratio: float,
        config: VolumeConfig,
    ) -> Tuple[float, List[ScoreDetail]]:
        """
        計算成交量相關分數
        
        Args:
            volume_ratio: 成交量比率 (當前量 / 均量)
            config: 成交量配置
            
        Returns:
            (總分數變化, 分數明細列表)
        """
        if not config.enabled:
            return 0.0, []
        
        total_score = 0.0
        details: List[ScoreDetail] = []
        
        # 放量加分
        if volume_ratio >= config.high_volume_threshold:
            score = config.high_volume_bonus * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="volume",
                raw_value=volume_ratio,
                score_change=score,
                reason=f"放量 (>= {config.high_volume_threshold}x 均量)",
            ))
        # 無量扣分
        elif volume_ratio < 1.0:
            score = config.low_volume_penalty * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="volume",
                raw_value=volume_ratio,
                score_change=score,
                reason="成交量低於均量",
            ))
        
        return total_score, details

    def _calculate_macd_score(
        self,
        macd: MACDResult,
        config: MACDConfig,
    ) -> Tuple[float, List[ScoreDetail]]:
        """
        計算 MACD 相關分數
        
        Args:
            macd: MACD 計算結果
            config: MACD 配置
            
        Returns:
            (總分數變化, 分數明細列表)
        """
        if not config.enabled:
            return 0.0, []
        
        total_score = 0.0
        details: List[ScoreDetail] = []
        
        # 零軸之上加分
        if macd.above_zero:
            score = config.above_zero_bonus * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="macd",
                raw_value=macd.macd_line,
                score_change=score,
                reason="MACD 在零軸之上且柱狀圖為正",
            ))
        # 零軸之下扣分
        elif macd.macd_line < 0:
            score = config.below_zero_penalty * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="macd",
                raw_value=macd.macd_line,
                score_change=score,
                reason="MACD 在零軸之下",
            ))
        
        # 黃金交叉加分
        if macd.golden_cross:
            score = config.golden_cross_bonus * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="macd",
                raw_value=macd.macd_line,
                score_change=score,
                reason="MACD 黃金交叉",
            ))
        
        return total_score, details


    def _calculate_ema_score(
        self,
        current_price: float,
        ema_20: Optional[float],
        ema_50: Optional[float],
        config: EMAConfig,
    ) -> Tuple[float, List[ScoreDetail]]:
        """
        計算均線相關分數
        
        Args:
            current_price: 當前價格
            ema_20: 20 日 EMA
            ema_50: 50 日 EMA
            config: 均線配置
            
        Returns:
            (總分數變化, 分數明細列表)
        """
        if not config.enabled:
            return 0.0, []
        
        total_score = 0.0
        details: List[ScoreDetail] = []
        
        # 價格在 EMA20 之上加分
        if ema_20 is not None:
            if current_price > ema_20:
                score = config.above_ema20_bonus * config.weight
                total_score += score
                details.append(ScoreDetail(
                    source="ema",
                    raw_value=ema_20,
                    score_change=score,
                    reason="價格在 EMA20 之上",
                ))
            else:
                # 價格在 EMA20 之下扣分
                score = config.below_ema20_penalty * config.weight
                total_score += score
                details.append(ScoreDetail(
                    source="ema",
                    raw_value=ema_20,
                    score_change=score,
                    reason="價格在 EMA20 之下",
                ))
        
        # 價格在 EMA50 之上額外加分
        if ema_50 is not None and current_price > ema_50:
            score = config.above_ema50_bonus * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="ema",
                raw_value=ema_50,
                score_change=score,
                reason="價格在 EMA50 之上",
            ))
        
        return total_score, details

    def _calculate_bollinger_score(
        self,
        bollinger: BollingerResult,
        rsi: Optional[RSIResult],
        config: BollingerConfig,
    ) -> Tuple[float, List[ScoreDetail]]:
        """
        計算布林通道相關分數
        
        Args:
            bollinger: 布林通道計算結果
            rsi: RSI 計算結果（用於組合判定）
            config: 布林通道配置
            
        Returns:
            (總分數變化, 分數明細列表)
        """
        if not config.enabled:
            return 0.0, []
        
        total_score = 0.0
        details: List[ScoreDetail] = []
        
        # 波動率壓縮 + 突破上軌加分
        if bollinger.squeeze and bollinger.breakout_upper:
            score = config.squeeze_breakout_bonus * config.weight
            total_score += score
            details.append(ScoreDetail(
                source="bollinger",
                raw_value=bollinger.bandwidth,
                score_change=score,
                reason="波動率壓縮後突破上軌",
            ))
        
        # 波動率壓縮 + RSI 突破 50 組合加分
        if bollinger.squeeze and rsi is not None:
            if rsi.value > 50 and rsi.trend_zone:
                score = config.squeeze_rsi_combo_bonus * config.weight
                total_score += score
                details.append(ScoreDetail(
                    source="bollinger",
                    raw_value=bollinger.bandwidth,
                    score_change=score,
                    reason="波動率壓縮 + RSI 突破 50",
                ))
        
        return total_score, details


    def _calculate_atr_stop_loss(
        self,
        entry_price: float,
        atr: float,
        multiplier: float,
    ) -> float:
        """
        計算 ATR 動態止損價
        
        Args:
            entry_price: 進場價格
            atr: ATR 值
            multiplier: ATR 倍數
            
        Returns:
            止損價位
        """
        return entry_price - (atr * multiplier)

    def _determine_strength(
        self,
        final_score: float,
        buy_threshold: float,
        watch_threshold: float,
    ) -> SignalStrength:
        """
        根據分數決定訊號強度
        
        Args:
            final_score: 最終分數
            buy_threshold: 買入閾值
            watch_threshold: 觀望閾值
            
        Returns:
            訊號強度
        """
        if final_score >= buy_threshold:
            return SignalStrength.STRONG_BUY
        elif final_score >= watch_threshold:
            return SignalStrength.WATCH
        else:
            return SignalStrength.SKIP

    def optimize(
        self,
        symbol: str,
        pattern_score: float,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        entry_price: float,
    ) -> OptimizedSignal:
        """
        優化交易訊號
        
        Args:
            symbol: 股票代碼
            pattern_score: 型態識別分數 (0-100)
            prices: 收盤價序列
            highs: 最高價序列
            lows: 最低價序列
            volumes: 成交量序列
            entry_price: 預計進場價格
            
        Returns:
            優化後的訊號，包含最終分數與明細
        """
        # 載入配置
        config = self.config_manager.get_config(symbol)
        
        # 計算所有指標
        indicators = self.indicator_pool.get_snapshot(prices, highs, lows, volumes)
        
        # 初始化分數與明細
        final_score = pattern_score
        all_details: List[ScoreDetail] = [
            ScoreDetail(
                source="pattern",
                raw_value=pattern_score,
                score_change=pattern_score,
                reason="型態識別分數",
            )
        ]
        
        # RSI 評分
        if indicators.rsi is not None:
            rsi_score, rsi_details = self._calculate_rsi_score(
                indicators.rsi,
                config.rsi,
                has_divergence=indicators.rsi_divergence or False,
            )
            final_score += rsi_score
            all_details.extend(rsi_details)
        
        # 成交量評分
        if indicators.volume_ratio is not None:
            vol_score, vol_details = self._calculate_volume_score(
                indicators.volume_ratio,
                config.volume,
            )
            final_score += vol_score
            all_details.extend(vol_details)
        
        # MACD 評分
        if indicators.macd is not None:
            macd_score, macd_details = self._calculate_macd_score(
                indicators.macd,
                config.macd,
            )
            final_score += macd_score
            all_details.extend(macd_details)
        
        # 均線評分
        current_price = prices[-1] if prices else entry_price
        ema_score, ema_details = self._calculate_ema_score(
            current_price,
            indicators.ema_20,
            indicators.ema_50,
            config.ema,
        )
        final_score += ema_score
        all_details.extend(ema_details)
        
        # 布林通道評分
        if indicators.bollinger is not None:
            bb_score, bb_details = self._calculate_bollinger_score(
                indicators.bollinger,
                indicators.rsi,
                config.bollinger,
            )
            final_score += bb_score
            all_details.extend(bb_details)
        
        # 計算止損價
        stop_loss = 0.0
        if config.use_atr_stop_loss and indicators.atr is not None:
            stop_loss = self._calculate_atr_stop_loss(
                entry_price,
                indicators.atr,
                config.atr_multiplier,
            )
        
        # 決定訊號強度
        strength = self._determine_strength(
            final_score,
            config.buy_threshold,
            config.watch_threshold,
        )
        
        return OptimizedSignal(
            symbol=symbol,
            pattern_score=pattern_score,
            final_score=final_score,
            strength=strength,
            details=all_details,
            recommended_stop_loss=stop_loss,
            indicators=indicators,
        )


    def optimize_with_precomputed(
        self,
        symbol: str,
        pattern_score: float,
        entry_price: float,
        precomputed_indicators: Dict[str, Any],
        index: int,
    ) -> OptimizedSignal:
        """
        使用預計算的指標優化交易訊號（快速版本）
        
        這個方法避免重複計算指標，適合批量回測使用。
        
        Args:
            symbol: 股票代碼
            pattern_score: 型態識別分數 (0-100)
            entry_price: 預計進場價格
            precomputed_indicators: 預計算的指標字典
            index: 當前數據索引
            
        Returns:
            優化後的訊號
        """
        # 載入配置
        config = self.config_manager.get_config(symbol)
        
        # 從預計算結果中取得指標值
        rsi_value = self._get_precomputed_value(precomputed_indicators, 'rsi', index)
        macd_data = self._get_precomputed_value(precomputed_indicators, 'macd', index)
        bb_data = self._get_precomputed_value(precomputed_indicators, 'bollinger', index)
        atr_value = self._get_precomputed_value(precomputed_indicators, 'atr', index)
        volume_ratio = self._get_precomputed_value(precomputed_indicators, 'volume_ratio', index)
        ema_20 = self._get_precomputed_ema(precomputed_indicators, 'ema_20', index, 20)
        ema_50 = self._get_precomputed_ema(precomputed_indicators, 'ema_50', index, 50)
        
        # 初始化分數與明細
        final_score = pattern_score
        all_details: List[ScoreDetail] = [
            ScoreDetail(
                source="pattern",
                raw_value=pattern_score,
                score_change=pattern_score,
                reason="型態識別分數",
            )
        ]
        
        # RSI 評分
        if rsi_value is not None and config.rsi.enabled:
            # 構建簡化的 RSI 結果
            rsi_result = RSIResult(
                value=rsi_value,
                is_overbought=rsi_value >= config.rsi.overbought,
                is_oversold=rsi_value <= config.rsi.oversold,
                trend_zone=config.rsi.trend_lower <= rsi_value <= config.rsi.trend_upper,
                support_bounce=False,  # 簡化處理
            )
            rsi_score, rsi_details = self._calculate_rsi_score(
                rsi_result,
                config.rsi,
                has_divergence=False,  # 簡化處理
            )
            final_score += rsi_score
            all_details.extend(rsi_details)
        
        # 成交量評分
        if volume_ratio is not None:
            vol_score, vol_details = self._calculate_volume_score(
                volume_ratio,
                config.volume,
            )
            final_score += vol_score
            all_details.extend(vol_details)
        
        # MACD 評分
        if macd_data is not None and isinstance(macd_data, dict) and config.macd.enabled:
            macd_result = MACDResult(
                macd_line=macd_data.get('macd_line', 0),
                signal_line=macd_data.get('signal_line', 0),
                histogram=macd_data.get('histogram', 0),
                above_zero=macd_data.get('macd_line', 0) > 0 and macd_data.get('histogram', 0) > 0,
                golden_cross=False,  # 簡化處理
            )
            macd_score, macd_details = self._calculate_macd_score(
                macd_result,
                config.macd,
            )
            final_score += macd_score
            all_details.extend(macd_details)
        
        # 均線評分
        ema_score, ema_details = self._calculate_ema_score(
            entry_price,
            ema_20,
            ema_50,
            config.ema,
        )
        final_score += ema_score
        all_details.extend(ema_details)
        
        # 布林通道評分
        if bb_data is not None and isinstance(bb_data, dict) and config.bollinger.enabled:
            bb_result = BollingerResult(
                upper=bb_data.get('upper', 0),
                middle=bb_data.get('middle', 0),
                lower=bb_data.get('lower', 0),
                bandwidth=bb_data.get('bandwidth', 0),
                squeeze=bb_data.get('bandwidth', 1) < 0.1,  # 簡化判定
                breakout_upper=entry_price > bb_data.get('upper', float('inf')),
            )
            # 構建簡化的 RSI 結果用於組合判定
            simple_rsi = None
            if rsi_value is not None:
                simple_rsi = RSIResult(
                    value=rsi_value,
                    is_overbought=False,
                    is_oversold=False,
                    trend_zone=config.rsi.trend_lower <= rsi_value <= config.rsi.trend_upper,
                    support_bounce=False,
                )
            bb_score, bb_details = self._calculate_bollinger_score(
                bb_result,
                simple_rsi,
                config.bollinger,
            )
            final_score += bb_score
            all_details.extend(bb_details)
        
        # 計算止損價
        stop_loss = 0.0
        if config.use_atr_stop_loss and atr_value is not None:
            stop_loss = self._calculate_atr_stop_loss(
                entry_price,
                atr_value,
                config.atr_multiplier,
            )
        
        # 決定訊號強度
        strength = self._determine_strength(
            final_score,
            config.buy_threshold,
            config.watch_threshold,
        )
        
        return OptimizedSignal(
            symbol=symbol,
            pattern_score=pattern_score,
            final_score=final_score,
            strength=strength,
            details=all_details,
            recommended_stop_loss=stop_loss,
            indicators=None,  # 不保存完整指標快照以節省記憶體
        )
    
    def _get_precomputed_value(
        self,
        precomputed: Dict[str, Any],
        indicator_name: str,
        index: int,
    ) -> Optional[Any]:
        """從預計算結果中取得指定索引的指標值"""
        if indicator_name not in precomputed:
            return None
        
        data = precomputed[indicator_name]
        
        if isinstance(data, dict):
            indices = data.get('indices')
            values = data.get('values')
            
            if indices is None or values is None:
                # 可能是 MACD 或 Bollinger 等複合指標
                if 'macd_line' in data:
                    # MACD
                    indices = data.get('indices')
                    if indices is not None and len(indices) > 0:
                        import numpy as np
                        pos = np.searchsorted(indices, index)
                        if pos < len(indices) and indices[pos] == index:
                            return {
                                'macd_line': float(data['macd_line'][pos]),
                                'signal_line': float(data['signal_line'][pos]),
                                'histogram': float(data['histogram'][pos]),
                            }
                elif 'upper' in data:
                    # Bollinger
                    indices = data.get('indices')
                    if indices is not None and len(indices) > 0:
                        import numpy as np
                        pos = np.searchsorted(indices, index)
                        if pos < len(indices) and indices[pos] == index:
                            return {
                                'upper': float(data['upper'][pos]),
                                'middle': float(data['middle'][pos]),
                                'lower': float(data['lower'][pos]),
                                'bandwidth': float(data['bandwidth'][pos]),
                            }
                return None
            
            # 標準格式：有 indices 和 values
            import numpy as np
            if len(indices) == 0:
                return None
            
            pos = np.searchsorted(indices, index)
            if pos < len(indices) and indices[pos] == index:
                return float(values[pos])
        
        return None
    
    def _get_precomputed_ema(
        self,
        precomputed: Dict[str, Any],
        ema_name: str,
        index: int,
        period: int,
    ) -> Optional[float]:
        """從預計算結果中取得 EMA 值"""
        if ema_name not in precomputed:
            return None
        
        ema_arr = precomputed[ema_name]
        
        if ema_arr is None or len(ema_arr) == 0:
            return None
        
        # EMA 陣列從 period-1 開始有效
        ema_index = index - period + 1
        if 0 <= ema_index < len(ema_arr):
            return float(ema_arr[ema_index])
        
        return None


    def optimize_with_dual_engine(
        self,
        symbol: str,
        pattern_score: float,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        entry_price: float,
        dual_engine_config: "DualEngineConfig",
        market_state: Optional["MarketState"] = None,
        allocation_weight: float = 1.0,
    ) -> OptimizedSignal:
        """
        整合雙引擎模式與因子權重計算最終訊號分數
        
        當雙引擎模式與因子權重同時啟用時，此方法整合兩者的計算結果：
        1. 先使用因子權重計算基礎分數
        2. 根據市場狀態調整分數權重
        3. 應用資金權重到最終分數
        
        Args:
            symbol: 股票代碼
            pattern_score: 型態識別分數 (0-100)
            prices: 收盤價序列
            highs: 最高價序列
            lows: 最低價序列
            volumes: 成交量序列
            entry_price: 預計進場價格
            dual_engine_config: 雙引擎配置
            market_state: 市場狀態（可選，若未提供則自動計算）
            allocation_weight: 資金權重 (0-1)
            
        Returns:
            整合後的優化訊號
            
        Requirements:
            13.5: 雙引擎模式與因子權重同時啟用時整合計算最終訊號分數
        """
        # 先使用標準因子權重計算
        base_signal = self.optimize(
            symbol=symbol,
            pattern_score=pattern_score,
            prices=prices,
            highs=highs,
            lows=lows,
            volumes=volumes,
            entry_price=entry_price,
        )
        
        # 若雙引擎未啟用，直接返回基礎訊號
        if not dual_engine_config.enabled:
            return base_signal
        
        # 計算市場狀態（若未提供）
        if market_state is None:
            from pattern_quant.strategy.market_classifier import MarketStateClassifier
            classifier = MarketStateClassifier(
                trend_threshold=dual_engine_config.adx_trend_threshold,
                range_threshold=dual_engine_config.adx_range_threshold,
                bbw_stability_threshold=dual_engine_config.bbw_stability_threshold,
            )
            market_result = classifier.classify(highs, lows, prices)
            market_state = market_result.state
            allocation_weight = market_result.allocation_weight
        
        # 根據市場狀態調整分數
        adjusted_score = self._apply_market_state_adjustment(
            base_score=base_signal.final_score,
            market_state=market_state,
            dual_engine_config=dual_engine_config,
        )
        
        # 應用資金權重
        final_adjusted_score = adjusted_score * allocation_weight
        
        # 添加雙引擎調整明細
        market_state_str = market_state.value if market_state else "unknown"
        dual_engine_details = base_signal.details.copy()
        dual_engine_details.append(ScoreDetail(
            source="dual_engine",
            raw_value=allocation_weight,
            score_change=final_adjusted_score - base_signal.final_score,
            reason=f"雙引擎調整: 市場狀態={market_state_str}, 資金權重={allocation_weight:.0%}",
        ))
        
        # 重新決定訊號強度
        config = self.config_manager.get_config(symbol)
        strength = self._determine_strength(
            final_adjusted_score,
            config.buy_threshold,
            config.watch_threshold,
        )
        
        return OptimizedSignal(
            symbol=symbol,
            pattern_score=pattern_score,
            final_score=final_adjusted_score,
            strength=strength,
            details=dual_engine_details,
            recommended_stop_loss=base_signal.recommended_stop_loss,
            indicators=base_signal.indicators,
            dual_engine_enabled=True,
            market_state=market_state_str,
            allocation_weight=allocation_weight,
            dual_engine_adjusted_score=adjusted_score,
        )
    
    def _apply_market_state_adjustment(
        self,
        base_score: float,
        market_state: "MarketState",
        dual_engine_config: "DualEngineConfig",
    ) -> float:
        """
        根據市場狀態調整分數
        
        調整邏輯：
        - TREND 狀態：型態突破更有效，給予加分
        - RANGE 狀態：均值回歸更有效，維持原分數
        - NOISE 狀態：不確定性高，給予扣分
        
        Args:
            base_score: 基礎分數
            market_state: 市場狀態
            dual_engine_config: 雙引擎配置
            
        Returns:
            調整後的分數
        """
        from pattern_quant.strategy.models import MarketState
        
        if market_state == MarketState.TREND:
            # 趨勢狀態：型態突破更有效，加分 10%
            return base_score * 1.10
        elif market_state == MarketState.RANGE:
            # 震盪狀態：維持原分數
            return base_score
        else:
            # 混沌狀態：不確定性高，扣分 20%
            return base_score * 0.80
    
    def integrate_dual_engine_result(
        self,
        base_signal: OptimizedSignal,
        dual_engine_result: "DualEngineAnalysisResult",
    ) -> OptimizedSignal:
        """
        將雙引擎分析結果整合到優化訊號中
        
        此方法用於將已完成的雙引擎分析結果與因子權重訊號整合。
        
        Args:
            base_signal: 因子權重計算的基礎訊號
            dual_engine_result: 雙引擎分析結果
            
        Returns:
            整合後的優化訊號
            
        Requirements:
            13.5: 雙引擎模式與因子權重同時啟用時整合計算最終訊號分數
        """
        market_state = dual_engine_result.market_state.state
        allocation_weight = dual_engine_result.allocation_weight
        
        # 根據市場狀態調整分數
        from pattern_quant.strategy.models import DualEngineConfig
        adjusted_score = self._apply_market_state_adjustment(
            base_score=base_signal.final_score,
            market_state=market_state,
            dual_engine_config=DualEngineConfig(),  # 使用預設配置進行調整
        )
        
        # 應用資金權重
        final_adjusted_score = adjusted_score * allocation_weight
        
        # 添加雙引擎調整明細
        market_state_str = market_state.value
        dual_engine_details = base_signal.details.copy()
        dual_engine_details.append(ScoreDetail(
            source="dual_engine",
            raw_value=allocation_weight,
            score_change=final_adjusted_score - base_signal.final_score,
            reason=f"雙引擎整合: 市場狀態={market_state_str}, 資金權重={allocation_weight:.0%}",
        ))
        
        # 如果雙引擎有產生信號，添加額外明細
        if dual_engine_result.resolved_signal.signal is not None:
            strategy_type = dual_engine_result.resolved_signal.strategy_type
            dual_engine_details.append(ScoreDetail(
                source="dual_engine_signal",
                raw_value=1.0,
                score_change=0.0,
                reason=f"雙引擎策略信號: {strategy_type}",
            ))
        
        # 重新決定訊號強度
        config = self.config_manager.get_config(base_signal.symbol)
        strength = self._determine_strength(
            final_adjusted_score,
            config.buy_threshold,
            config.watch_threshold,
        )
        
        return OptimizedSignal(
            symbol=base_signal.symbol,
            pattern_score=base_signal.pattern_score,
            final_score=final_adjusted_score,
            strength=strength,
            details=dual_engine_details,
            recommended_stop_loss=base_signal.recommended_stop_loss,
            indicators=base_signal.indicators,
            dual_engine_enabled=True,
            market_state=market_state_str,
            allocation_weight=allocation_weight,
            dual_engine_adjusted_score=adjusted_score,
        )
