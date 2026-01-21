"""
指標庫 (Indicator Pool)

負責計算所有技術指標，提供統一的指標存取介面。
"""

from typing import List, Optional
import numpy as np

from .models import (
    RSIResult,
    MACDResult,
    BollingerResult,
    StochasticResult,
    IndicatorSnapshot,
)


class IndicatorPool:
    """技術指標計算庫"""

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        stoch_k: int = 14,
        stoch_d: int = 3,
    ):
        """
        初始化指標庫，設定各指標的計算參數
        
        Args:
            rsi_period: RSI 計算週期
            macd_fast: MACD 快線週期
            macd_slow: MACD 慢線週期
            macd_signal: MACD 訊號線週期
            bb_period: 布林通道週期
            bb_std: 布林通道標準差倍數
            atr_period: ATR 計算週期
            stoch_k: KD 指標 K 值週期
            stoch_d: KD 指標 D 值週期
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        
        # 儲存前一次 MACD 結果用於黃金交叉偵測
        self._prev_macd_line: Optional[float] = None
        self._prev_signal_line: Optional[float] = None


    def calculate_rsi(
        self,
        prices: List[float],
        period: Optional[int] = None,
        overbought: float = 80,
        oversold: float = 30,
        trend_lower: float = 50,
        trend_upper: float = 75,
    ) -> Optional[RSIResult]:
        """
        計算 RSI 指標
        
        Args:
            prices: 收盤價序列
            period: RSI 週期（預設使用初始化時設定的值）
            overbought: 超買閾值
            oversold: 超賣閾值
            trend_lower: 趨勢區間下限
            trend_upper: 趨勢區間上限
            
        Returns:
            RSIResult 或 None（若數據不足）
        """
        if period is None:
            period = self.rsi_period
            
        # 過濾 NaN 值
        prices_arr = np.array(prices, dtype=float)
        prices_arr = prices_arr[~np.isnan(prices_arr)]
        
        if len(prices_arr) < period + 1:
            return None
        
        # 計算價格變化
        deltas = np.diff(prices_arr)
        
        # 分離漲跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 使用 Wilder's smoothing method (EMA)
        avg_gain = self._wilder_smoothing(gains, period)
        avg_loss = self._wilder_smoothing(losses, period)
        
        if avg_loss == 0:
            rsi_value = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100 - (100 / (1 + rs))
        
        # 確保 RSI 在 0-100 範圍內
        rsi_value = max(0.0, min(100.0, rsi_value))
        
        # 判定狀態
        is_overbought = rsi_value >= overbought
        is_oversold = rsi_value <= oversold
        trend_zone = trend_lower <= rsi_value <= trend_upper
        
        # 支撐反彈判定：RSI 從 40-50 區間向上
        support_bounce = False
        if len(prices_arr) >= period + 2:
            prev_rsi = self._calculate_rsi_value(prices_arr[:-1], period)
            if prev_rsi is not None:
                support_bounce = (40 <= prev_rsi <= 50) and (rsi_value > prev_rsi)
        
        return RSIResult(
            value=rsi_value,
            is_overbought=is_overbought,
            is_oversold=is_oversold,
            trend_zone=trend_zone,
            support_bounce=support_bounce,
        )

    def _wilder_smoothing(self, values: np.ndarray, period: int) -> float:
        """Wilder's smoothing method"""
        if len(values) < period:
            return 0.0
        
        # 初始 SMA
        avg = np.mean(values[:period])
        
        # 後續使用 EMA 平滑
        alpha = 1.0 / period
        for i in range(period, len(values)):
            avg = alpha * values[i] + (1 - alpha) * avg
        
        return avg

    def _calculate_rsi_value(self, prices: np.ndarray, period: int) -> Optional[float]:
        """計算單一 RSI 值（內部輔助方法）"""
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = self._wilder_smoothing(gains, period)
        avg_loss = self._wilder_smoothing(losses, period)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


    def calculate_stochastic(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        k_period: Optional[int] = None,
        d_period: Optional[int] = None,
        overbought: float = 80,
        oversold: float = 20,
    ) -> Optional[StochasticResult]:
        """
        計算 Stochastic (KD) 指標
        
        Args:
            highs: 最高價序列
            lows: 最低價序列
            closes: 收盤價序列
            k_period: K 值週期
            d_period: D 值週期
            overbought: 超買閾值
            oversold: 超賣閾值
            
        Returns:
            StochasticResult 或 None（若數據不足）
        """
        if k_period is None:
            k_period = self.stoch_k
        if d_period is None:
            d_period = self.stoch_d
        
        highs_arr = np.array(highs, dtype=float)
        lows_arr = np.array(lows, dtype=float)
        closes_arr = np.array(closes, dtype=float)
        
        # 過濾 NaN
        valid_mask = ~(np.isnan(highs_arr) | np.isnan(lows_arr) | np.isnan(closes_arr))
        highs_arr = highs_arr[valid_mask]
        lows_arr = lows_arr[valid_mask]
        closes_arr = closes_arr[valid_mask]
        
        if len(closes_arr) < k_period + d_period - 1:
            return None
        
        # 計算 %K 值序列
        k_values = []
        for i in range(k_period - 1, len(closes_arr)):
            highest_high = np.max(highs_arr[i - k_period + 1:i + 1])
            lowest_low = np.min(lows_arr[i - k_period + 1:i + 1])
            
            if highest_high == lowest_low:
                k_values.append(50.0)  # 避免除以零
            else:
                k = ((closes_arr[i] - lowest_low) / (highest_high - lowest_low)) * 100
                k_values.append(max(0.0, min(100.0, k)))
        
        if len(k_values) < d_period:
            return None
        
        # 計算 %D 值（K 值的 SMA）
        d_value = np.mean(k_values[-d_period:])
        k_value = k_values[-1]
        
        # 確保值在 0-100 範圍內
        k_value = max(0.0, min(100.0, k_value))
        d_value = max(0.0, min(100.0, d_value))
        
        return StochasticResult(
            k_value=k_value,
            d_value=d_value,
            is_overbought=k_value >= overbought,
            is_oversold=k_value <= oversold,
        )


    def calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """
        計算指數移動平均線
        
        Args:
            prices: 價格序列
            period: EMA 週期
            
        Returns:
            EMA 值或 None（若數據不足）
        """
        prices_arr = np.array(prices, dtype=float)
        prices_arr = prices_arr[~np.isnan(prices_arr)]
        
        if len(prices_arr) < period:
            return None
        
        # EMA 計算：使用標準 EMA 公式
        multiplier = 2.0 / (period + 1)
        
        # 初始值使用 SMA
        ema = np.mean(prices_arr[:period])
        
        # 後續使用 EMA 公式
        for price in prices_arr[period:]:
            ema = (price - ema) * multiplier + ema
        
        return float(ema)

    def calculate_macd(
        self,
        prices: List[float],
        fast_period: Optional[int] = None,
        slow_period: Optional[int] = None,
        signal_period: Optional[int] = None,
    ) -> Optional[MACDResult]:
        """
        計算 MACD 指標
        
        Args:
            prices: 收盤價序列
            fast_period: 快線週期
            slow_period: 慢線週期
            signal_period: 訊號線週期
            
        Returns:
            MACDResult 或 None（若數據不足）
        """
        if fast_period is None:
            fast_period = self.macd_fast
        if slow_period is None:
            slow_period = self.macd_slow
        if signal_period is None:
            signal_period = self.macd_signal
        
        prices_arr = np.array(prices, dtype=float)
        prices_arr = prices_arr[~np.isnan(prices_arr)]
        
        min_length = slow_period + signal_period
        if len(prices_arr) < min_length:
            return None
        
        # 計算快線和慢線 EMA
        fast_ema = self._calculate_ema_series(prices_arr, fast_period)
        slow_ema = self._calculate_ema_series(prices_arr, slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None
        
        # MACD 線 = 快線 EMA - 慢線 EMA
        # 對齊長度（從慢線開始的位置）
        start_idx = slow_period - fast_period
        macd_line_series = fast_ema[start_idx:] - slow_ema
        
        if len(macd_line_series) < signal_period:
            return None
        
        # 訊號線 = MACD 線的 EMA
        signal_line_series = self._calculate_ema_series(macd_line_series, signal_period)
        
        if signal_line_series is None or len(signal_line_series) == 0:
            return None
        
        macd_line = float(macd_line_series[-1])
        signal_line = float(signal_line_series[-1])
        histogram = macd_line - signal_line
        
        # 黃金交叉偵測
        golden_cross = False
        if len(macd_line_series) >= 2 and len(signal_line_series) >= 2:
            prev_macd = macd_line_series[-2]
            # 訊號線序列比 MACD 線短 signal_period-1
            signal_offset = signal_period - 1
            if len(macd_line_series) > signal_offset + 1:
                prev_signal = signal_line_series[-2] if len(signal_line_series) >= 2 else signal_line
                # 黃金交叉：MACD 從下方穿越訊號線
                golden_cross = (prev_macd <= prev_signal) and (macd_line > signal_line)
        
        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            above_zero=macd_line > 0 and histogram > 0,
            golden_cross=golden_cross,
        )

    def _calculate_ema_series(self, prices: np.ndarray, period: int) -> Optional[np.ndarray]:
        """計算 EMA 序列"""
        if len(prices) < period:
            return None
        
        multiplier = 2.0 / (period + 1)
        ema_series = np.zeros(len(prices) - period + 1)
        
        # 初始值使用 SMA
        ema_series[0] = np.mean(prices[:period])
        
        # 後續使用 EMA 公式
        for i in range(1, len(ema_series)):
            ema_series[i] = (prices[period + i - 1] - ema_series[i - 1]) * multiplier + ema_series[i - 1]
        
        return ema_series


    def calculate_atr(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: Optional[int] = None,
    ) -> Optional[float]:
        """
        計算 ATR (Average True Range)
        
        Args:
            highs: 最高價序列
            lows: 最低價序列
            closes: 收盤價序列
            period: ATR 週期
            
        Returns:
            ATR 值或 None（若數據不足）
        """
        if period is None:
            period = self.atr_period
        
        highs_arr = np.array(highs, dtype=float)
        lows_arr = np.array(lows, dtype=float)
        closes_arr = np.array(closes, dtype=float)
        
        # 過濾 NaN
        valid_mask = ~(np.isnan(highs_arr) | np.isnan(lows_arr) | np.isnan(closes_arr))
        highs_arr = highs_arr[valid_mask]
        lows_arr = lows_arr[valid_mask]
        closes_arr = closes_arr[valid_mask]
        
        if len(closes_arr) < period + 1:
            return None
        
        # 計算 True Range
        true_ranges = []
        for i in range(1, len(closes_arr)):
            high_low = highs_arr[i] - lows_arr[i]
            high_close = abs(highs_arr[i] - closes_arr[i - 1])
            low_close = abs(lows_arr[i] - closes_arr[i - 1])
            true_ranges.append(max(high_low, high_close, low_close))
        
        if len(true_ranges) < period:
            return None
        
        # 使用 Wilder's smoothing 計算 ATR
        tr_arr = np.array(true_ranges)
        atr = self._wilder_smoothing(tr_arr, period)
        
        return float(atr) if atr > 0 else None


    def calculate_bollinger(
        self,
        prices: List[float],
        period: Optional[int] = None,
        std_dev: Optional[float] = None,
        squeeze_threshold: float = 0.5,
    ) -> Optional[BollingerResult]:
        """
        計算布林通道
        
        Args:
            prices: 收盤價序列
            period: 布林通道週期
            std_dev: 標準差倍數
            squeeze_threshold: 壓縮判定閾值（帶寬 < 歷史平均 × 此值）
            
        Returns:
            BollingerResult 或 None（若數據不足）
        """
        if period is None:
            period = self.bb_period
        if std_dev is None:
            std_dev = self.bb_std
        
        prices_arr = np.array(prices, dtype=float)
        prices_arr = prices_arr[~np.isnan(prices_arr)]
        
        if len(prices_arr) < period:
            return None
        
        # 計算中軌（SMA）
        middle = np.mean(prices_arr[-period:])
        
        # 計算標準差
        std = np.std(prices_arr[-period:], ddof=0)
        
        # 計算上下軌
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        # 計算帶寬
        if middle == 0:
            bandwidth = 0.0
        else:
            bandwidth = (upper - lower) / middle
        
        # 計算歷史帶寬以判定壓縮
        squeeze = False
        if len(prices_arr) >= period * 2:
            historical_bandwidths = []
            for i in range(period, len(prices_arr) + 1):
                window = prices_arr[i - period:i]
                m = np.mean(window)
                s = np.std(window, ddof=0)
                if m > 0:
                    bw = (2 * std_dev * s) / m
                    historical_bandwidths.append(bw)
            
            if historical_bandwidths:
                avg_bandwidth = np.mean(historical_bandwidths[-period:])
                squeeze = bandwidth < (avg_bandwidth * squeeze_threshold)
        
        # 判定是否突破上軌
        current_price = prices_arr[-1]
        breakout_upper = current_price > upper
        
        return BollingerResult(
            upper=float(upper),
            middle=float(middle),
            lower=float(lower),
            bandwidth=float(bandwidth),
            squeeze=squeeze,
            breakout_upper=breakout_upper,
        )


    def calculate_volume_ratio(
        self,
        volumes: List[float],
        period: int = 20,
    ) -> Optional[float]:
        """
        計算成交量比率（當前量 / N日均量）
        
        Args:
            volumes: 成交量序列
            period: 均量計算週期
            
        Returns:
            成交量比率或 None（若數據不足）
        """
        volumes_arr = np.array(volumes, dtype=float)
        volumes_arr = volumes_arr[~np.isnan(volumes_arr)]
        
        if len(volumes_arr) < period:
            return None
        
        avg_volume = np.mean(volumes_arr[-period:])
        
        if avg_volume == 0:
            return None
        
        current_volume = volumes_arr[-1]
        return float(current_volume / avg_volume)

    def detect_rsi_divergence(
        self,
        prices: List[float],
        rsi_values: Optional[List[float]] = None,
        lookback: int = 20,
    ) -> bool:
        """
        偵測 RSI 頂背離
        
        Args:
            prices: 價格序列
            rsi_values: RSI 值序列（若為 None 則自動計算）
            lookback: 回溯期間
            
        Returns:
            True 如果價格創新高但 RSI 未創新高
        """
        prices_arr = np.array(prices, dtype=float)
        prices_arr = prices_arr[~np.isnan(prices_arr)]
        
        if len(prices_arr) < lookback + self.rsi_period:
            return False
        
        # 如果沒有提供 RSI 值，自動計算
        if rsi_values is None:
            rsi_values = []
            for i in range(self.rsi_period + 1, len(prices_arr) + 1):
                result = self.calculate_rsi(prices_arr[:i].tolist())
                if result:
                    rsi_values.append(result.value)
        
        if len(rsi_values) < lookback:
            return False
        
        rsi_arr = np.array(rsi_values[-lookback:])
        price_window = prices_arr[-lookback:]
        
        # 找出價格高點
        current_price = price_window[-1]
        max_price_idx = np.argmax(price_window[:-1])
        max_price = price_window[max_price_idx]
        
        # 找出對應的 RSI 高點
        current_rsi = rsi_arr[-1]
        max_rsi = rsi_arr[max_price_idx]
        
        # 頂背離：價格創新高但 RSI 未創新高
        price_new_high = current_price >= max_price
        rsi_not_new_high = current_rsi < max_rsi
        
        return price_new_high and rsi_not_new_high

    def get_snapshot(
        self,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
    ) -> IndicatorSnapshot:
        """
        計算並返回所有指標的快照
        
        Args:
            prices: 收盤價序列
            highs: 最高價序列
            lows: 最低價序列
            volumes: 成交量序列
            
        Returns:
            包含所有指標狀態的快照
        """
        # RSI
        rsi = self.calculate_rsi(prices)
        
        # MACD
        macd = self.calculate_macd(prices)
        
        # Bollinger Bands
        bollinger = self.calculate_bollinger(prices)
        
        # Stochastic
        stochastic = self.calculate_stochastic(highs, lows, prices)
        
        # EMA
        ema_20 = self.calculate_ema(prices, 20)
        ema_50 = self.calculate_ema(prices, 50)
        
        # ATR
        atr = self.calculate_atr(highs, lows, prices)
        
        # Volume Ratio
        volume_ratio = self.calculate_volume_ratio(volumes)
        
        # RSI Divergence
        rsi_divergence = self.detect_rsi_divergence(prices)
        
        return IndicatorSnapshot(
            rsi=rsi,
            macd=macd,
            bollinger=bollinger,
            stochastic=stochastic,
            ema_20=ema_20,
            ema_50=ema_50,
            atr=atr,
            volume_ratio=volume_ratio,
            rsi_divergence=rsi_divergence,
        )
