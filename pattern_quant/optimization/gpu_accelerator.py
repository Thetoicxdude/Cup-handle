"""
GPU 加速模組 (GPU Accelerator)

使用 CuPy 進行 GPU 加速的技術指標計算。
如果 GPU 不可用，自動回退到 NumPy CPU 計算。
"""

from typing import List, Optional, Tuple, Union
import numpy as np

# 嘗試導入 CuPy，如果失敗則使用 NumPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp  # 使用 CuPy
except ImportError:
    GPU_AVAILABLE = False
    xp = np  # 回退到 NumPy


def is_gpu_available() -> bool:
    """檢查 GPU 是否可用"""
    if not GPU_AVAILABLE:
        return False
    try:
        # 嘗試創建一個小陣列來測試 GPU
        test = cp.array([1, 2, 3])
        _ = test.sum()
        return True
    except Exception:
        return False


def to_device(arr: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
    """將 NumPy 陣列轉移到 GPU（如果可用）"""
    if GPU_AVAILABLE and is_gpu_available():
        return cp.asarray(arr)
    return arr


def to_host(arr: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
    """將陣列從 GPU 轉回 CPU"""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr


class GPUIndicatorPool:
    """
    GPU 加速的技術指標計算庫
    
    使用向量化操作和 GPU 並行計算來加速指標計算。
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        use_gpu: bool = True,
    ):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        
        # 決定使用 GPU 還是 CPU
        self.use_gpu = use_gpu and is_gpu_available()
        self._xp = cp if self.use_gpu else np
    
    def _to_array(self, data: List[float]) -> Union[np.ndarray, 'cp.ndarray']:
        """將列表轉換為適當的陣列類型"""
        arr = np.array(data, dtype=np.float64)
        if self.use_gpu:
            return cp.asarray(arr)
        return arr
    
    def _to_numpy(self, arr: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """將陣列轉回 NumPy"""
        if self.use_gpu and hasattr(arr, 'get'):
            return arr.get()
        return arr
    
    def calculate_ema_vectorized(
        self,
        prices: Union[np.ndarray, 'cp.ndarray'],
        period: int,
    ) -> Union[np.ndarray, 'cp.ndarray']:
        """
        向量化 EMA 計算
        
        使用遞迴公式的向量化版本
        """
        xp = self._xp
        n = len(prices)
        
        if n < period:
            return xp.array([])
        
        alpha = 2.0 / (period + 1)
        
        # 初始化 EMA 陣列
        ema = xp.zeros(n - period + 1, dtype=xp.float64)
        
        # 第一個值是 SMA
        ema[0] = xp.mean(prices[:period])
        
        # 使用向量化的方式計算後續 EMA
        # EMA[i] = alpha * price[i] + (1-alpha) * EMA[i-1]
        for i in range(1, len(ema)):
            ema[i] = alpha * prices[period + i - 1] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def calculate_rsi_batch(
        self,
        prices: List[float],
        period: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量計算 RSI 序列
        
        返回完整的 RSI 序列，而不是單一值。
        這樣可以避免重複計算。
        
        Returns:
            (rsi_values, valid_indices) - RSI 值陣列和對應的有效索引
        """
        if period is None:
            period = self.rsi_period
        
        xp = self._xp
        prices_arr = self._to_array(prices)
        
        n = len(prices_arr)
        if n < period + 1:
            return np.array([]), np.array([])
        
        # 計算價格變化
        deltas = xp.diff(prices_arr)
        
        # 分離漲跌
        gains = xp.where(deltas > 0, deltas, 0)
        losses = xp.where(deltas < 0, -deltas, 0)
        
        # 計算滾動平均（使用 Wilder's smoothing）
        alpha = 1.0 / period
        
        # 初始化
        avg_gains = xp.zeros(n - 1, dtype=xp.float64)
        avg_losses = xp.zeros(n - 1, dtype=xp.float64)
        
        # 第一個值是 SMA
        if len(gains) >= period:
            avg_gains[period - 1] = xp.mean(gains[:period])
            avg_losses[period - 1] = xp.mean(losses[:period])
            
            # 後續使用 EMA
            for i in range(period, len(gains)):
                avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i - 1]
                avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i - 1]
        
        # 計算 RSI
        # 避免除以零
        avg_losses_safe = xp.where(avg_losses == 0, 1e-10, avg_losses)
        rs = avg_gains / avg_losses_safe
        rsi = 100 - (100 / (1 + rs))
        
        # 處理 avg_loss == 0 的情況
        rsi = xp.where(avg_losses == 0, 100.0, rsi)
        
        # 只返回有效的 RSI 值（從 period 開始）
        valid_rsi = rsi[period - 1:]
        valid_indices = xp.arange(period, n)
        
        return self._to_numpy(valid_rsi), self._to_numpy(valid_indices)
    
    def calculate_macd_batch(
        self,
        prices: List[float],
        fast_period: Optional[int] = None,
        slow_period: Optional[int] = None,
        signal_period: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        批量計算 MACD 序列
        
        Returns:
            (macd_line, signal_line, histogram, valid_indices)
        """
        if fast_period is None:
            fast_period = self.macd_fast
        if slow_period is None:
            slow_period = self.macd_slow
        if signal_period is None:
            signal_period = self.macd_signal
        
        prices_arr = self._to_array(prices)
        n = len(prices_arr)
        
        min_length = slow_period + signal_period
        if n < min_length:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 計算快線和慢線 EMA
        fast_ema = self.calculate_ema_vectorized(prices_arr, fast_period)
        slow_ema = self.calculate_ema_vectorized(prices_arr, slow_period)
        
        if len(fast_ema) == 0 or len(slow_ema) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 對齊長度
        start_idx = slow_period - fast_period
        macd_line = fast_ema[start_idx:] - slow_ema
        
        if len(macd_line) < signal_period:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 計算訊號線
        signal_line = self.calculate_ema_vectorized(macd_line, signal_period)
        
        # 對齊 MACD 線和訊號線
        macd_aligned = macd_line[signal_period - 1:]
        histogram = macd_aligned - signal_line
        
        # 計算有效索引
        valid_start = slow_period + signal_period - 1
        valid_indices = self._xp.arange(valid_start, n)
        
        return (
            self._to_numpy(macd_aligned),
            self._to_numpy(signal_line),
            self._to_numpy(histogram),
            self._to_numpy(valid_indices),
        )
    
    def calculate_bollinger_batch(
        self,
        prices: List[float],
        period: Optional[int] = None,
        std_dev: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        批量計算布林通道序列
        
        Returns:
            (upper, middle, lower, bandwidth, valid_indices)
        """
        if period is None:
            period = self.bb_period
        if std_dev is None:
            std_dev = self.bb_std
        
        xp = self._xp
        prices_arr = self._to_array(prices)
        n = len(prices_arr)
        
        if n < period:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        # 計算滾動 SMA 和標準差
        result_len = n - period + 1
        middle = xp.zeros(result_len, dtype=xp.float64)
        std_arr = xp.zeros(result_len, dtype=xp.float64)
        
        for i in range(result_len):
            window = prices_arr[i:i + period]
            middle[i] = xp.mean(window)
            std_arr[i] = xp.std(window)
        
        upper = middle + std_dev * std_arr
        lower = middle - std_dev * std_arr
        
        # 計算帶寬
        bandwidth = xp.where(middle > 0, (upper - lower) / middle, 0)
        
        valid_indices = xp.arange(period - 1, n)
        
        return (
            self._to_numpy(upper),
            self._to_numpy(middle),
            self._to_numpy(lower),
            self._to_numpy(bandwidth),
            self._to_numpy(valid_indices),
        )
    
    def calculate_atr_batch(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量計算 ATR 序列
        
        Returns:
            (atr_values, valid_indices)
        """
        if period is None:
            period = self.atr_period
        
        xp = self._xp
        highs_arr = self._to_array(highs)
        lows_arr = self._to_array(lows)
        closes_arr = self._to_array(closes)
        
        n = len(closes_arr)
        if n < period + 1:
            return np.array([]), np.array([])
        
        # 計算 True Range（向量化）
        high_low = highs_arr[1:] - lows_arr[1:]
        high_close = xp.abs(highs_arr[1:] - closes_arr[:-1])
        low_close = xp.abs(lows_arr[1:] - closes_arr[:-1])
        
        true_range = xp.maximum(xp.maximum(high_low, high_close), low_close)
        
        # 計算 ATR（使用 Wilder's smoothing）
        alpha = 1.0 / period
        atr_len = len(true_range) - period + 1
        
        if atr_len <= 0:
            return np.array([]), np.array([])
        
        atr = xp.zeros(atr_len, dtype=xp.float64)
        atr[0] = xp.mean(true_range[:period])
        
        for i in range(1, atr_len):
            atr[i] = alpha * true_range[period + i - 1] + (1 - alpha) * atr[i - 1]
        
        valid_indices = xp.arange(period, n)
        
        return self._to_numpy(atr), self._to_numpy(valid_indices)
    
    def calculate_volume_ratio_batch(
        self,
        volumes: List[float],
        period: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量計算成交量比率序列
        
        Returns:
            (volume_ratios, valid_indices)
        """
        xp = self._xp
        volumes_arr = self._to_array(volumes)
        n = len(volumes_arr)
        
        if n < period:
            return np.array([]), np.array([])
        
        result_len = n - period + 1
        ratios = xp.zeros(result_len, dtype=xp.float64)
        
        for i in range(result_len):
            avg_vol = xp.mean(volumes_arr[i:i + period])
            if avg_vol > 0:
                ratios[i] = volumes_arr[i + period - 1] / avg_vol
            else:
                ratios[i] = 1.0
        
        valid_indices = xp.arange(period - 1, n)
        
        return self._to_numpy(ratios), self._to_numpy(valid_indices)


class BatchBacktester:
    """
    批量回測器
    
    預先計算所有指標，然後進行向量化回測。
    """
    
    def __init__(self, gpu_pool: Optional[GPUIndicatorPool] = None):
        self.gpu_pool = gpu_pool or GPUIndicatorPool()
        self._indicator_cache = {}
    
    def precompute_indicators(
        self,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
    ) -> dict:
        """
        預先計算所有指標
        
        這樣在回測時就不需要重複計算。
        """
        cache = {}
        
        # RSI
        rsi_values, rsi_indices = self.gpu_pool.calculate_rsi_batch(prices)
        cache['rsi'] = {'values': rsi_values, 'indices': rsi_indices}
        
        # MACD
        macd, signal, hist, macd_indices = self.gpu_pool.calculate_macd_batch(prices)
        cache['macd'] = {
            'macd_line': macd,
            'signal_line': signal,
            'histogram': hist,
            'indices': macd_indices,
        }
        
        # Bollinger
        upper, middle, lower, bandwidth, bb_indices = self.gpu_pool.calculate_bollinger_batch(prices)
        cache['bollinger'] = {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': bandwidth,
            'indices': bb_indices,
        }
        
        # ATR
        atr_values, atr_indices = self.gpu_pool.calculate_atr_batch(highs, lows, prices)
        cache['atr'] = {'values': atr_values, 'indices': atr_indices}
        
        # Volume Ratio
        vol_ratios, vol_indices = self.gpu_pool.calculate_volume_ratio_batch(volumes)
        cache['volume_ratio'] = {'values': vol_ratios, 'indices': vol_indices}
        
        # EMA
        prices_arr = np.array(prices, dtype=np.float64)
        ema_20 = self.gpu_pool.calculate_ema_vectorized(
            self.gpu_pool._to_array(prices), 20
        )
        ema_50 = self.gpu_pool.calculate_ema_vectorized(
            self.gpu_pool._to_array(prices), 50
        )
        cache['ema_20'] = self.gpu_pool._to_numpy(ema_20)
        cache['ema_50'] = self.gpu_pool._to_numpy(ema_50)
        
        self._indicator_cache = cache
        return cache
    
    def get_indicator_at_index(self, indicator_name: str, index: int) -> Optional[float]:
        """
        取得特定索引位置的指標值
        """
        if indicator_name not in self._indicator_cache:
            return None
        
        cache = self._indicator_cache[indicator_name]
        
        if 'indices' in cache:
            indices = cache['indices']
            if len(indices) == 0:
                return None
            
            # 找到對應的位置
            pos = np.searchsorted(indices, index)
            if pos < len(indices) and indices[pos] == index:
                if 'values' in cache:
                    return float(cache['values'][pos])
                elif 'macd_line' in cache:
                    return {
                        'macd_line': float(cache['macd_line'][pos]),
                        'signal_line': float(cache['signal_line'][pos]),
                        'histogram': float(cache['histogram'][pos]),
                    }
        elif isinstance(cache, np.ndarray):
            # EMA 等直接是陣列的情況
            if index < len(cache):
                return float(cache[index])
        
        return None


def get_accelerator(use_gpu: bool = True) -> GPUIndicatorPool:
    """
    取得加速器實例
    
    Args:
        use_gpu: 是否嘗試使用 GPU
        
    Returns:
        GPUIndicatorPool 實例
    """
    return GPUIndicatorPool(use_gpu=use_gpu)


# 便捷函數
def accelerated_rsi(prices: List[float], period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """快速計算 RSI 序列"""
    pool = GPUIndicatorPool()
    return pool.calculate_rsi_batch(prices, period)


def accelerated_macd(
    prices: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """快速計算 MACD 序列"""
    pool = GPUIndicatorPool()
    return pool.calculate_macd_batch(prices, fast, slow, signal)
