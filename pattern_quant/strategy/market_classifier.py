"""
市場狀態分類器 (Market State Classifier)

根據 ADX（趨勢強度）與 BBW（波動率）自動判定市場狀態，
用於雙引擎策略的策略切換決策。

Requirements:
- 1.1, 1.2, 1.3: ADX 指標計算
- 2.1, 2.2, 2.3, 2.4: BBW 指標計算
- 3.1, 3.2, 3.3: 市場狀態分類
"""

import logging
from typing import List, Optional

from pattern_quant.strategy.models import (
    MarketState,
    ADXResult,
    BBWResult,
    MarketStateResult,
    DualEngineConfig,
)

logger = logging.getLogger(__name__)


class MarketStateClassifier:
    """
    市場狀態分類器
    
    根據 ADX 與 BBW 指標判定市場狀態：
    - TREND: 強趨勢 (ADX > trend_threshold)
    - RANGE: 震盪盤整 (ADX < range_threshold 且 BBW 穩定)
    - NOISE: 混沌轉換 (range_threshold ≤ ADX ≤ trend_threshold)
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        trend_threshold: float = 25.0,
        range_threshold: float = 20.0,
        bbw_stability_threshold: float = 0.10
    ):
        """
        初始化市場狀態分類器
        
        Args:
            adx_period: ADX 計算週期（預設 14 日）
            bb_period: 布林通道週期（預設 20 日）
            bb_std: 布林通道標準差倍數（預設 2.0）
            trend_threshold: 趨勢判定閾值 (ADX > 此值 = Trend)
            range_threshold: 震盪判定閾值 (ADX < 此值 = Range)
            bbw_stability_threshold: BBW 穩定性閾值（變化率 < 此值視為穩定）
        """
        self.adx_period = adx_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.bbw_stability_threshold = bbw_stability_threshold
    
    @classmethod
    def from_config(cls, config: DualEngineConfig) -> "MarketStateClassifier":
        """
        從配置物件建立分類器
        
        Args:
            config: 雙引擎策略配置
            
        Returns:
            配置好的分類器實例
        """
        return cls(
            trend_threshold=config.adx_trend_threshold,
            range_threshold=config.adx_range_threshold,
            bbw_stability_threshold=config.bbw_stability_threshold,
        )
    
    def calculate_adx(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float]
    ) -> Optional[ADXResult]:
        """
        計算 ADX 指標（14 日）
        
        ADX (Average Directional Index) 衡量趨勢強度，無關方向。
        同時計算 +DI 和 -DI 用於判斷趨勢方向。
        
        Args:
            highs: 最高價序列
            lows: 最低價序列
            closes: 收盤價序列
            
        Returns:
            ADXResult 包含 ADX、+DI、-DI 值，若數據不足則返回 None
            
        Requirements:
            1.1: 計算 14 日 ADX 值
            1.2: 同時返回 +DI 與 -DI 值
            1.3: 返回 0-100 範圍內的數值
        """
        n = len(closes)
        
        # 需要至少 adx_period + 1 個數據點
        if n < self.adx_period + 1 or len(highs) != n or len(lows) != n:
            logger.warning(f"ADX 計算數據不足: 需要 {self.adx_period + 1} 個數據點，實際 {n} 個")
            return None
        
        # 計算 True Range (TR), +DM, -DM
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(1, n):
            high = highs[i]
            low = lows[i]
            prev_high = highs[i - 1]
            prev_low = lows[i - 1]
            prev_close = closes[i - 1]
            
            # True Range = max(H-L, |H-PC|, |L-PC|)
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_list.append(tr)
            
            # +DM = H - PH if (H - PH) > (PL - L) and (H - PH) > 0, else 0
            # -DM = PL - L if (PL - L) > (H - PH) and (PL - L) > 0, else 0
            up_move = high - prev_high
            down_move = prev_low - low
            
            if up_move > down_move and up_move > 0:
                plus_dm = up_move
            else:
                plus_dm = 0.0
            
            if down_move > up_move and down_move > 0:
                minus_dm = down_move
            else:
                minus_dm = 0.0
            
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
        
        # 使用 Wilder's Smoothing (類似 EMA)
        period = self.adx_period
        
        if len(tr_list) < period:
            logger.warning(f"ADX 計算數據不足: TR 序列長度 {len(tr_list)} < {period}")
            return None
        
        # 初始化：使用前 period 個值的總和
        atr = sum(tr_list[:period])
        plus_dm_smooth = sum(plus_dm_list[:period])
        minus_dm_smooth = sum(minus_dm_list[:period])
        
        # Wilder's Smoothing
        for i in range(period, len(tr_list)):
            atr = atr - (atr / period) + tr_list[i]
            plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period) + plus_dm_list[i]
            minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period) + minus_dm_list[i]
        
        # 計算 +DI 和 -DI
        if atr == 0:
            logger.warning("ATR 為零，無法計算 DI")
            return None
        
        plus_di = 100.0 * plus_dm_smooth / atr
        minus_di = 100.0 * minus_dm_smooth / atr
        
        # 計算 DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0.0
        else:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum
        
        # 計算 ADX（DX 的平滑平均）
        # 簡化版：使用當前 DX 作為 ADX（完整版需要 DX 序列的平滑）
        # 為了更準確，我們需要計算 DX 序列並平滑
        dx_list = []
        
        # 重新計算以獲取 DX 序列
        atr_temp = sum(tr_list[:period])
        plus_dm_temp = sum(plus_dm_list[:period])
        minus_dm_temp = sum(minus_dm_list[:period])
        
        for i in range(period, len(tr_list)):
            atr_temp = atr_temp - (atr_temp / period) + tr_list[i]
            plus_dm_temp = plus_dm_temp - (plus_dm_temp / period) + plus_dm_list[i]
            minus_dm_temp = minus_dm_temp - (minus_dm_temp / period) + minus_dm_list[i]
            
            if atr_temp > 0:
                pdi = 100.0 * plus_dm_temp / atr_temp
                mdi = 100.0 * minus_dm_temp / atr_temp
                di_sum_temp = pdi + mdi
                if di_sum_temp > 0:
                    dx_temp = 100.0 * abs(pdi - mdi) / di_sum_temp
                else:
                    dx_temp = 0.0
                dx_list.append(dx_temp)
        
        if len(dx_list) < period:
            # 數據不足以計算平滑 ADX，使用最後一個 DX
            adx = dx
        else:
            # 使用 Wilder's Smoothing 計算 ADX
            adx = sum(dx_list[:period]) / period
            for i in range(period, len(dx_list)):
                adx = (adx * (period - 1) + dx_list[i]) / period
        
        # 確保值在 0-100 範圍內
        adx = max(0.0, min(100.0, adx))
        plus_di = max(0.0, min(100.0, plus_di))
        minus_di = max(0.0, min(100.0, minus_di))
        
        return ADXResult(
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di
        )
    
    def calculate_bbw(
        self,
        prices: List[float]
    ) -> Optional[BBWResult]:
        """
        計算布林帶寬 (BBW)
        
        BBW = (上軌 - 下軌) / 中軌
        用於衡量波動率的壓縮與擴張狀態。
        
        Args:
            prices: 收盤價序列
            
        Returns:
            BBWResult 包含帶寬、平均帶寬、壓縮標記、變化率，
            若數據不足則返回 None
            
        Requirements:
            2.1: 計算 20 日布林通道（標準差 2.0）
            2.2: 使用公式 (上軌 - 下軌) / 中軌
            2.3: 返回帶寬數值與歷史 20 日平均帶寬
            2.4: BBW < 歷史平均 × 0.5 時標記為壓縮狀態
        """
        n = len(prices)
        
        # 需要至少 bb_period 個數據點來計算布林通道
        # 需要額外的數據來計算歷史平均帶寬
        min_required = self.bb_period * 2  # 需要足夠數據計算歷史平均
        
        if n < self.bb_period:
            logger.warning(f"BBW 計算數據不足: 需要至少 {self.bb_period} 個數據點，實際 {n} 個")
            return None
        
        # 計算當前布林通道
        recent_prices = prices[-self.bb_period:]
        middle = sum(recent_prices) / self.bb_period
        
        # 計算標準差
        variance = sum((p - middle) ** 2 for p in recent_prices) / self.bb_period
        std = variance ** 0.5
        
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        
        # 計算當前帶寬
        if middle == 0:
            logger.warning("布林中軌為零，無法計算 BBW")
            return None
        
        bandwidth = (upper - lower) / middle
        
        # 計算歷史帶寬序列（用於計算平均帶寬和變化率）
        bbw_history = []
        
        for i in range(self.bb_period, n + 1):
            window = prices[i - self.bb_period:i]
            mid = sum(window) / self.bb_period
            if mid == 0:
                continue
            var = sum((p - mid) ** 2 for p in window) / self.bb_period
            s = var ** 0.5
            u = mid + self.bb_std * s
            l = mid - self.bb_std * s
            bw = (u - l) / mid
            bbw_history.append(bw)
        
        # 計算歷史平均帶寬（最近 20 個帶寬值）
        if len(bbw_history) >= self.bb_period:
            avg_bandwidth = sum(bbw_history[-self.bb_period:]) / self.bb_period
        elif len(bbw_history) > 0:
            avg_bandwidth = sum(bbw_history) / len(bbw_history)
        else:
            avg_bandwidth = bandwidth
        
        # 計算變化率
        if len(bbw_history) >= 2:
            prev_bandwidth = bbw_history[-2]
            if prev_bandwidth != 0:
                change_rate = abs(bandwidth - prev_bandwidth) / prev_bandwidth
            else:
                change_rate = 0.0
        else:
            change_rate = 0.0
        
        # 判斷是否處於壓縮狀態
        is_squeeze = bandwidth < (avg_bandwidth * 0.5)
        
        return BBWResult(
            bandwidth=bandwidth,
            avg_bandwidth=avg_bandwidth,
            is_squeeze=is_squeeze,
            change_rate=change_rate
        )
    
    def classify(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float]
    ) -> MarketStateResult:
        """
        判定市場狀態
        
        根據 ADX 和 BBW 指標判定市場處於哪種狀態：
        - TREND: ADX > trend_threshold (預設 25)
        - RANGE: ADX < range_threshold (預設 20) 且 BBW 穩定
        - NOISE: 其他情況
        
        Args:
            highs: 最高價序列
            lows: 最低價序列
            closes: 收盤價序列
            
        Returns:
            MarketStateResult 包含狀態、資金權重、ADX/BBW 詳情、信心度
            
        Requirements:
            3.1: ADX > 25 → Trend 狀態
            3.2: ADX < 20 且 BBW 穩定 → Range 狀態
            3.3: 20 ≤ ADX ≤ 25 → Noise 狀態
        """
        # 計算 ADX
        adx_result = self.calculate_adx(highs, lows, closes)
        
        # 計算 BBW
        bbw_result = self.calculate_bbw(closes)
        
        # 處理計算失敗的情況
        if adx_result is None:
            # ADX 計算失敗，使用預設值並返回 NOISE 狀態
            adx_result = ADXResult(adx=22.5, plus_di=50.0, minus_di=50.0)
            logger.warning("ADX 計算失敗，使用預設 NOISE 狀態")
        
        if bbw_result is None:
            # BBW 計算失敗，使用預設值
            bbw_result = BBWResult(
                bandwidth=0.1,
                avg_bandwidth=0.1,
                is_squeeze=False,
                change_rate=0.0
            )
            logger.warning("BBW 計算失敗，使用預設值")
        
        # 判定市場狀態
        adx = adx_result.adx
        bbw_stable = bbw_result.change_rate < self.bbw_stability_threshold
        
        if adx > self.trend_threshold:
            # 強趨勢狀態
            state = MarketState.TREND
            allocation_weight = 1.0  # 100%
            # 信心度基於 ADX 超過閾值的程度
            confidence = min(1.0, (adx - self.trend_threshold) / 25.0 + 0.5)
        elif adx < self.range_threshold and bbw_stable:
            # 震盪盤整狀態
            state = MarketState.RANGE
            allocation_weight = 0.6  # 60%
            # 信心度基於 ADX 低於閾值的程度和 BBW 穩定性
            confidence = min(1.0, (self.range_threshold - adx) / 20.0 + 0.5)
        else:
            # 混沌轉換狀態
            state = MarketState.NOISE
            allocation_weight = 0.0  # 0%
            # 信心度較低，因為處於不確定區間
            confidence = 0.3
        
        return MarketStateResult(
            state=state,
            allocation_weight=allocation_weight,
            adx_result=adx_result,
            bbw_result=bbw_result,
            confidence=confidence
        )
