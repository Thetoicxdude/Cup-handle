"""
自動調參器 (Auto Tuner)

透過回測尋找最佳指標組合的優化引擎。
支援優化：
1. 指標啟用/停用組合
2. 各指標權重
3. 買入/觀望閾值

支援 GPU 加速（如果 CuPy 可用）。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
from datetime import datetime
from itertools import product
import numpy as np

from .models import IndicatorSnapshot
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
from .signal_optimizer import SignalOptimizer, SignalStrength

# 嘗試導入 GPU 加速模組
try:
    from .gpu_accelerator import GPUIndicatorPool, BatchBacktester, is_gpu_available
    GPU_SUPPORT = True
except ImportError:
    GPU_SUPPORT = False
    def is_gpu_available():
        return False


@dataclass
class BacktestResult:
    """回測結果"""
    config: FactorConfig
    total_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float


@dataclass
class TuningProgress:
    """調參進度"""
    current_combination: int
    total_combinations: int
    best_win_rate_so_far: float
    current_config_description: str
    phase: str = "組合測試"  # 當前階段

    @property
    def progress_percent(self) -> float:
        """計算進度百分比"""
        if self.total_combinations == 0:
            return 0.0
        return (self.current_combination / self.total_combinations) * 100


@dataclass
class TuningOptions:
    """調參選項"""
    # 是否優化權重
    optimize_weights: bool = True
    # 權重候選值
    weight_candidates: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    # 是否優化閾值
    optimize_thresholds: bool = True
    # 買入閾值候選值（降低範圍，因為型態分數通常在 60-80）
    buy_threshold_candidates: List[float] = field(default_factory=lambda: [60.0, 65.0, 70.0, 75.0])
    # 觀望閾值候選值
    watch_threshold_candidates: List[float] = field(default_factory=lambda: [45.0, 50.0, 55.0, 60.0])
    # 是否優化 RSI 參數
    optimize_rsi_params: bool = True
    # RSI 趨勢區間下限候選值（放寬範圍）
    rsi_trend_lower_candidates: List[float] = field(default_factory=lambda: [35.0, 40.0, 45.0])
    # RSI 趨勢區間上限候選值
    rsi_trend_upper_candidates: List[float] = field(default_factory=lambda: [65.0, 70.0, 75.0])
    # 最大組合數限制（避免過長時間）
    max_combinations: int = 500
    # 是否使用 GPU 加速
    use_gpu: bool = True



class AutoTuner:
    """
    自動調參器
    
    透過回測尋找最佳指標組合的優化引擎。
    支援三階段優化：
    1. 第一階段：找出最佳指標啟用組合
    2. 第二階段：優化各指標權重
    3. 第三階段：優化閾值參數
    
    支援 GPU 加速（如果 CuPy 可用）。
    """
    
    def __init__(
        self,
        indicator_pool: IndicatorPool,
        signal_optimizer: SignalOptimizer,
        config_manager: FactorConfigManager,
        progress_callback: Optional[Callable[[TuningProgress], None]] = None,
        tuning_options: Optional[TuningOptions] = None,
    ):
        """
        初始化自動調參器
        
        Args:
            indicator_pool: 指標計算庫
            signal_optimizer: 訊號優化器
            config_manager: 因子配置管理器
            progress_callback: 進度回調函數，用於 UI 更新
            tuning_options: 調參選項
        """
        self.indicator_pool = indicator_pool
        self.signal_optimizer = signal_optimizer
        self.config_manager = config_manager
        self.progress_callback = progress_callback
        self.options = tuning_options or TuningOptions()
        
        # 初始化 GPU 加速器（如果可用）
        self._gpu_pool = None
        self._precomputed_indicators = None
        if GPU_SUPPORT and self.options.use_gpu and is_gpu_available():
            self._gpu_pool = GPUIndicatorPool(use_gpu=True)
            self._use_gpu = True
        else:
            self._use_gpu = False
    
    def _precompute_indicators(
        self,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
    ) -> Dict:
        """
        預先計算所有指標（使用 GPU 加速如果可用）
        
        這樣在回測時就不需要重複計算。
        """
        if self._gpu_pool is not None:
            # 使用 GPU 加速
            cache = {}
            
            # RSI
            rsi_values, rsi_indices = self._gpu_pool.calculate_rsi_batch(prices)
            cache['rsi'] = {'values': rsi_values, 'indices': rsi_indices}
            
            # MACD
            macd, signal, hist, macd_indices = self._gpu_pool.calculate_macd_batch(prices)
            cache['macd'] = {
                'macd_line': macd,
                'signal_line': signal,
                'histogram': hist,
                'indices': macd_indices,
            }
            
            # Bollinger
            upper, middle, lower, bandwidth, bb_indices = self._gpu_pool.calculate_bollinger_batch(prices)
            cache['bollinger'] = {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'bandwidth': bandwidth,
                'indices': bb_indices,
            }
            
            # ATR
            atr_values, atr_indices = self._gpu_pool.calculate_atr_batch(highs, lows, prices)
            cache['atr'] = {'values': atr_values, 'indices': atr_indices}
            
            # Volume Ratio
            vol_ratios, vol_indices = self._gpu_pool.calculate_volume_ratio_batch(volumes)
            cache['volume_ratio'] = {'values': vol_ratios, 'indices': vol_indices}
            
            # EMA
            prices_arr = self._gpu_pool._to_array(prices)
            ema_20 = self._gpu_pool.calculate_ema_vectorized(prices_arr, 20)
            ema_50 = self._gpu_pool.calculate_ema_vectorized(prices_arr, 50)
            cache['ema_20'] = self._gpu_pool._to_numpy(ema_20)
            cache['ema_50'] = self._gpu_pool._to_numpy(ema_50)
            
            return cache
        else:
            # CPU 版本的預計算
            return self._precompute_indicators_cpu(prices, highs, lows, volumes)
    
    def _precompute_indicators_cpu(
        self,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
    ) -> Dict:
        """CPU 版本的指標預計算"""
        cache = {}
        n = len(prices)
        
        # RSI - 計算完整序列
        rsi_values = []
        rsi_indices = []
        for i in range(14, n):
            result = self.indicator_pool.calculate_rsi(prices[:i+1])
            if result:
                rsi_values.append(result.value)
                rsi_indices.append(i)
        cache['rsi'] = {'values': np.array(rsi_values), 'indices': np.array(rsi_indices)}
        
        # MACD
        macd_lines = []
        signal_lines = []
        histograms = []
        macd_indices = []
        for i in range(35, n):  # MACD 需要至少 26+9=35 天數據
            result = self.indicator_pool.calculate_macd(prices[:i+1])
            if result:
                macd_lines.append(result.macd_line)
                signal_lines.append(result.signal_line)
                histograms.append(result.histogram)
                macd_indices.append(i)
        cache['macd'] = {
            'macd_line': np.array(macd_lines),
            'signal_line': np.array(signal_lines),
            'histogram': np.array(histograms),
            'indices': np.array(macd_indices),
        }
        
        # Bollinger
        uppers = []
        middles = []
        lowers = []
        bandwidths = []
        bb_indices = []
        for i in range(20, n):
            result = self.indicator_pool.calculate_bollinger(prices[:i+1])
            if result:
                uppers.append(result.upper)
                middles.append(result.middle)
                lowers.append(result.lower)
                bandwidths.append(result.bandwidth)
                bb_indices.append(i)
        cache['bollinger'] = {
            'upper': np.array(uppers),
            'middle': np.array(middles),
            'lower': np.array(lowers),
            'bandwidth': np.array(bandwidths),
            'indices': np.array(bb_indices),
        }
        
        # ATR
        atr_values = []
        atr_indices = []
        for i in range(15, n):
            result = self.indicator_pool.calculate_atr(highs[:i+1], lows[:i+1], prices[:i+1])
            if result:
                atr_values.append(result)
                atr_indices.append(i)
        cache['atr'] = {'values': np.array(atr_values), 'indices': np.array(atr_indices)}
        
        # Volume Ratio
        vol_ratios = []
        vol_indices = []
        for i in range(20, n):
            result = self.indicator_pool.calculate_volume_ratio(volumes[:i+1])
            if result:
                vol_ratios.append(result)
                vol_indices.append(i)
        cache['volume_ratio'] = {'values': np.array(vol_ratios), 'indices': np.array(vol_indices)}
        
        # EMA
        ema_20_values = []
        for i in range(20, n):
            result = self.indicator_pool.calculate_ema(prices[:i+1], 20)
            if result:
                ema_20_values.append(result)
        cache['ema_20'] = np.array(ema_20_values)
        
        ema_50_values = []
        for i in range(50, n):
            result = self.indicator_pool.calculate_ema(prices[:i+1], 50)
            if result:
                ema_50_values.append(result)
        cache['ema_50'] = np.array(ema_50_values)
        
        return cache
    
    def _generate_combinations(self, symbol: str) -> List[FactorConfig]:
        """
        生成所有指標啟用/停用組合
        
        Args:
            symbol: 股票代碼
            
        Returns:
            所有可能的配置組合列表
        """
        # 5 個指標，每個有啟用/停用兩種狀態 = 2^5 = 32 種組合
        indicator_states = [True, False]
        combinations = []
        
        for rsi_enabled, vol_enabled, macd_enabled, ema_enabled, bb_enabled in product(
            indicator_states, repeat=5
        ):
            # 至少要啟用一個指標
            if not any([rsi_enabled, vol_enabled, macd_enabled, ema_enabled, bb_enabled]):
                continue
            
            config = FactorConfig(
                symbol=symbol,
                rsi=RSIConfig(enabled=rsi_enabled),
                volume=VolumeConfig(enabled=vol_enabled),
                macd=MACDConfig(enabled=macd_enabled),
                ema=EMAConfig(enabled=ema_enabled),
                bollinger=BollingerConfig(enabled=bb_enabled),
            )
            combinations.append(config)
        
        return combinations
    
    def _generate_weight_combinations(
        self,
        base_config: FactorConfig,
    ) -> List[FactorConfig]:
        """
        基於最佳組合生成權重變化組合
        
        Args:
            base_config: 基礎配置（已確定啟用哪些指標）
            
        Returns:
            權重變化的配置列表
        """
        combinations = []
        weights = self.options.weight_candidates
        
        # 只對啟用的指標進行權重優化
        enabled_indicators = []
        if base_config.rsi.enabled:
            enabled_indicators.append("rsi")
        if base_config.volume.enabled:
            enabled_indicators.append("volume")
        if base_config.macd.enabled:
            enabled_indicators.append("macd")
        if base_config.ema.enabled:
            enabled_indicators.append("ema")
        if base_config.bollinger.enabled:
            enabled_indicators.append("bollinger")
        
        if not enabled_indicators:
            return [base_config]
        
        # 生成權重組合（限制數量）
        weight_combos = list(product(weights, repeat=len(enabled_indicators)))
        
        # 如果組合太多，隨機抽樣
        if len(weight_combos) > 100:
            import random
            weight_combos = random.sample(weight_combos, 100)
        
        for weight_combo in weight_combos:
            config = FactorConfig(
                symbol=base_config.symbol,
                rsi=RSIConfig(
                    enabled=base_config.rsi.enabled,
                    weight=weight_combo[enabled_indicators.index("rsi")] if "rsi" in enabled_indicators else 1.0,
                    trend_lower=base_config.rsi.trend_lower,
                    trend_upper=base_config.rsi.trend_upper,
                    overbought=base_config.rsi.overbought,
                    oversold=base_config.rsi.oversold,
                    trend_zone_bonus=base_config.rsi.trend_zone_bonus,
                    overbought_penalty=base_config.rsi.overbought_penalty,
                    support_bounce_bonus=base_config.rsi.support_bounce_bonus,
                    divergence_penalty=base_config.rsi.divergence_penalty,
                    weak_penalty=base_config.rsi.weak_penalty,
                    check_divergence=base_config.rsi.check_divergence,
                ),
                volume=VolumeConfig(
                    enabled=base_config.volume.enabled,
                    weight=weight_combo[enabled_indicators.index("volume")] if "volume" in enabled_indicators else 1.0,
                ),
                macd=MACDConfig(
                    enabled=base_config.macd.enabled,
                    weight=weight_combo[enabled_indicators.index("macd")] if "macd" in enabled_indicators else 1.0,
                ),
                ema=EMAConfig(
                    enabled=base_config.ema.enabled,
                    weight=weight_combo[enabled_indicators.index("ema")] if "ema" in enabled_indicators else 1.0,
                ),
                bollinger=BollingerConfig(
                    enabled=base_config.bollinger.enabled,
                    weight=weight_combo[enabled_indicators.index("bollinger")] if "bollinger" in enabled_indicators else 1.0,
                ),
                buy_threshold=base_config.buy_threshold,
                watch_threshold=base_config.watch_threshold,
                use_atr_stop_loss=base_config.use_atr_stop_loss,
                atr_multiplier=base_config.atr_multiplier,
            )
            combinations.append(config)
        
        return combinations
    
    def _generate_threshold_combinations(
        self,
        base_config: FactorConfig,
    ) -> List[FactorConfig]:
        """
        基於最佳配置生成閾值變化組合
        
        Args:
            base_config: 基礎配置
            
        Returns:
            閾值變化的配置列表
        """
        combinations = []
        
        buy_thresholds = self.options.buy_threshold_candidates
        watch_thresholds = self.options.watch_threshold_candidates
        
        for buy_th, watch_th in product(buy_thresholds, watch_thresholds):
            # 確保 buy > watch
            if buy_th <= watch_th:
                continue
            
            config = FactorConfig(
                symbol=base_config.symbol,
                rsi=base_config.rsi,
                volume=base_config.volume,
                macd=base_config.macd,
                ema=base_config.ema,
                bollinger=base_config.bollinger,
                buy_threshold=buy_th,
                watch_threshold=watch_th,
                use_atr_stop_loss=base_config.use_atr_stop_loss,
                atr_multiplier=base_config.atr_multiplier,
            )
            combinations.append(config)
        
        return combinations
    
    def _generate_rsi_param_combinations(
        self,
        base_config: FactorConfig,
    ) -> List[FactorConfig]:
        """
        基於最佳配置生成 RSI 參數變化組合
        
        Args:
            base_config: 基礎配置
            
        Returns:
            RSI 參數變化的配置列表
        """
        if not base_config.rsi.enabled:
            return [base_config]
        
        combinations = []
        
        trend_lowers = self.options.rsi_trend_lower_candidates
        trend_uppers = self.options.rsi_trend_upper_candidates
        
        for trend_lower, trend_upper in product(trend_lowers, trend_uppers):
            # 確保 upper > lower
            if trend_upper <= trend_lower:
                continue
            
            new_rsi = RSIConfig(
                enabled=True,
                weight=base_config.rsi.weight,
                trend_lower=trend_lower,
                trend_upper=trend_upper,
                overbought=base_config.rsi.overbought,
                oversold=base_config.rsi.oversold,
                trend_zone_bonus=base_config.rsi.trend_zone_bonus,
                overbought_penalty=base_config.rsi.overbought_penalty,
                support_bounce_bonus=base_config.rsi.support_bounce_bonus,
                divergence_penalty=base_config.rsi.divergence_penalty,
                weak_penalty=base_config.rsi.weak_penalty,
                check_divergence=base_config.rsi.check_divergence,
            )
            
            config = FactorConfig(
                symbol=base_config.symbol,
                rsi=new_rsi,
                volume=base_config.volume,
                macd=base_config.macd,
                ema=base_config.ema,
                bollinger=base_config.bollinger,
                buy_threshold=base_config.buy_threshold,
                watch_threshold=base_config.watch_threshold,
                use_atr_stop_loss=base_config.use_atr_stop_loss,
                atr_multiplier=base_config.atr_multiplier,
            )
            combinations.append(config)
        
        return combinations
    
    def _get_config_description(self, config: FactorConfig) -> str:
        """生成配置描述字串"""
        enabled = []
        if config.rsi.enabled:
            enabled.append(f"RSI({config.rsi.weight:.1f})")
        if config.volume.enabled:
            enabled.append(f"Vol({config.volume.weight:.1f})")
        if config.macd.enabled:
            enabled.append(f"MACD({config.macd.weight:.1f})")
        if config.ema.enabled:
            enabled.append(f"EMA({config.ema.weight:.1f})")
        if config.bollinger.enabled:
            enabled.append(f"BB({config.bollinger.weight:.1f})")
        
        desc = ", ".join(enabled) if enabled else "None"
        desc += f" | 買入>{config.buy_threshold:.0f} 觀望>{config.watch_threshold:.0f}"
        return desc


    def _backtest_config(
        self,
        config: FactorConfig,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        pattern_scores: List[float],
        precomputed_indicators: Optional[Dict] = None,
    ) -> BacktestResult:
        """
        對單一配置執行回測
        
        Args:
            config: 要測試的配置
            prices: 歷史收盤價
            highs: 歷史最高價
            lows: 歷史最低價
            volumes: 歷史成交量
            pattern_scores: 歷史型態分數（與價格對齊）
            precomputed_indicators: 預計算的指標（可選，用於加速）
            
        Returns:
            回測結果
        """
        # 暫時儲存原配置
        original_config = self.config_manager.get_config(config.symbol)
        
        # 套用測試配置
        self.config_manager.save_config(config)
        
        # 回測參數
        min_data_length = 60  # 至少需要 60 天數據計算指標
        holding_period = 5    # 持有期間（天）
        
        trades: List[Tuple[float, float]] = []  # (entry_price, exit_price)
        equity_curve: List[float] = [100.0]     # 權益曲線，初始 100
        
        # 決定是否使用預計算指標
        use_precomputed = precomputed_indicators is not None
        
        i = min_data_length
        while i < len(prices) - holding_period:
            pattern_score = pattern_scores[i] if i < len(pattern_scores) else 50.0
            entry_price = prices[i]
            
            # 使用 SignalOptimizer 計算訊號
            if use_precomputed:
                # 使用預計算指標的快速版本
                signal = self.signal_optimizer.optimize_with_precomputed(
                    symbol=config.symbol,
                    pattern_score=pattern_score,
                    entry_price=entry_price,
                    precomputed_indicators=precomputed_indicators,
                    index=i,
                )
            else:
                # 原始版本（較慢）
                current_prices = prices[:i + 1]
                current_highs = highs[:i + 1]
                current_lows = lows[:i + 1]
                current_volumes = volumes[:i + 1]
                
                signal = self.signal_optimizer.optimize(
                    symbol=config.symbol,
                    pattern_score=pattern_score,
                    prices=current_prices,
                    highs=current_highs,
                    lows=current_lows,
                    volumes=current_volumes,
                    entry_price=entry_price,
                )
            
            # 只有強烈買入訊號才進場
            if signal.strength == SignalStrength.STRONG_BUY:
                exit_price = prices[i + holding_period]
                trades.append((entry_price, exit_price))
                
                # 更新權益曲線
                pct_return = (exit_price - entry_price) / entry_price
                new_equity = equity_curve[-1] * (1 + pct_return)
                equity_curve.append(new_equity)
                
                # 跳過持有期間
                i += holding_period
            else:
                i += 1
        
        # 還原原配置
        if original_config:
            self.config_manager.save_config(original_config)
        
        # 計算回測指標
        total_trades = len(trades)
        
        if total_trades == 0:
            return BacktestResult(
                config=config,
                total_trades=0,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
            )
        
        # 勝率
        wins = sum(1 for entry, exit in trades if exit > entry)
        win_rate = wins / total_trades
        
        # 總報酬
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # 夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio(trades)
        
        return BacktestResult(
            config=config,
            total_trades=total_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
        )
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """計算最大回撤"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(
        self,
        trades: List[Tuple[float, float]],
        risk_free_rate: float = 0.02,
        annualization_factor: float = 252,
    ) -> float:
        """
        計算夏普比率
        
        Args:
            trades: 交易列表 (entry_price, exit_price)
            risk_free_rate: 無風險利率（年化）
            annualization_factor: 年化因子
            
        Returns:
            夏普比率
        """
        if len(trades) < 2:
            return 0.0
        
        returns = [(exit - entry) / entry for entry, exit in trades]
        
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # 年化
        daily_rf = risk_free_rate / annualization_factor
        sharpe = (avg_return - daily_rf) / std_return * np.sqrt(annualization_factor / len(trades))
        
        return float(sharpe)


    def tune(
        self,
        symbol: str,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        pattern_scores: List[float],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        執行自動調參（三階段優化）
        
        階段 1: 找出最佳指標啟用組合
        階段 2: 優化各指標權重
        階段 3: 優化閾值參數
        
        使用預計算指標加速回測（如果 GPU 可用則使用 GPU 加速）。
        
        Args:
            symbol: 股票代碼
            prices: 歷史收盤價
            highs: 歷史最高價
            lows: 歷史最低價
            volumes: 歷史成交量
            pattern_scores: 歷史型態分數（與價格對齊）
            start_date: 回測起始日（目前未使用，保留供未來擴展）
            end_date: 回測結束日（目前未使用，保留供未來擴展）
            
        Returns:
            最佳配置的回測結果
        """
        # 預計算所有指標（一次性計算，避免重複）
        precomputed = self._precompute_indicators(prices, highs, lows, volumes)
        
        # ========== 階段 1: 找出最佳指標組合 ==========
        combinations = self._generate_combinations(symbol)
        total_phase1 = len(combinations)
        
        best_result: Optional[BacktestResult] = None
        best_win_rate = 0.0
        
        for idx, config in enumerate(combinations):
            if self.progress_callback:
                progress = TuningProgress(
                    current_combination=idx + 1,
                    total_combinations=total_phase1,
                    best_win_rate_so_far=best_win_rate,
                    current_config_description=self._get_config_description(config),
                    phase="階段1: 指標組合",
                )
                self.progress_callback(progress)
            
            result = self._backtest_config(
                config=config,
                prices=prices,
                highs=highs,
                lows=lows,
                volumes=volumes,
                pattern_scores=pattern_scores,
                precomputed_indicators=precomputed,
            )
            
            if result.win_rate > best_win_rate or best_result is None:
                best_win_rate = result.win_rate
                best_result = result
        
        if best_result is None:
            default_config = self.config_manager.get_default_config(symbol)
            return BacktestResult(
                config=default_config,
                total_trades=0,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
            )
        
        # ========== 階段 2: 優化權重 ==========
        if self.options.optimize_weights:
            weight_combinations = self._generate_weight_combinations(best_result.config)
            total_phase2 = len(weight_combinations)
            
            for idx, config in enumerate(weight_combinations):
                if self.progress_callback:
                    progress = TuningProgress(
                        current_combination=idx + 1,
                        total_combinations=total_phase2,
                        best_win_rate_so_far=best_win_rate,
                        current_config_description=self._get_config_description(config),
                        phase="階段2: 權重優化",
                    )
                    self.progress_callback(progress)
                
                result = self._backtest_config(
                    config=config,
                    prices=prices,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    pattern_scores=pattern_scores,
                    precomputed_indicators=precomputed,
                )
                
                if result.win_rate > best_win_rate:
                    best_win_rate = result.win_rate
                    best_result = result
        
        # ========== 階段 3: 優化閾值 ==========
        if self.options.optimize_thresholds:
            threshold_combinations = self._generate_threshold_combinations(best_result.config)
            total_phase3 = len(threshold_combinations)
            
            for idx, config in enumerate(threshold_combinations):
                if self.progress_callback:
                    progress = TuningProgress(
                        current_combination=idx + 1,
                        total_combinations=total_phase3,
                        best_win_rate_so_far=best_win_rate,
                        current_config_description=self._get_config_description(config),
                        phase="階段3: 閾值優化",
                    )
                    self.progress_callback(progress)
                
                result = self._backtest_config(
                    config=config,
                    prices=prices,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    pattern_scores=pattern_scores,
                    precomputed_indicators=precomputed,
                )
                
                if result.win_rate > best_win_rate:
                    best_win_rate = result.win_rate
                    best_result = result
        
        # ========== 階段 4: 優化 RSI 參數（如果啟用）==========
        if self.options.optimize_rsi_params and best_result.config.rsi.enabled:
            rsi_combinations = self._generate_rsi_param_combinations(best_result.config)
            total_phase4 = len(rsi_combinations)
            
            for idx, config in enumerate(rsi_combinations):
                if self.progress_callback:
                    progress = TuningProgress(
                        current_combination=idx + 1,
                        total_combinations=total_phase4,
                        best_win_rate_so_far=best_win_rate,
                        current_config_description=self._get_config_description(config),
                        phase="階段4: RSI參數優化",
                    )
                    self.progress_callback(progress)
                
                result = self._backtest_config(
                    config=config,
                    prices=prices,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    pattern_scores=pattern_scores,
                    precomputed_indicators=precomputed,
                )
                
                if result.win_rate > best_win_rate:
                    best_win_rate = result.win_rate
                    best_result = result
        
        # 儲存最佳配置
        self.config_manager.save_config(best_result.config)
        
        return best_result

    def calculate_correlation_matrix(
        self,
        backtest_results: List[BacktestResult],
    ) -> Dict[str, float]:
        """
        計算各指標與勝率的相關性
        
        Args:
            backtest_results: 回測結果列表
            
        Returns:
            {indicator_name: correlation_coefficient}
        """
        if len(backtest_results) < 2:
            return {
                "rsi": 0.0,
                "volume": 0.0,
                "macd": 0.0,
                "ema": 0.0,
                "bollinger": 0.0,
            }
        
        # 建立指標啟用狀態矩陣
        indicators = ["rsi", "volume", "macd", "ema", "bollinger"]
        correlations: Dict[str, float] = {}
        
        win_rates = np.array([r.win_rate for r in backtest_results])
        
        for indicator in indicators:
            # 取得該指標的啟用狀態
            enabled_states = np.array([
                1.0 if getattr(getattr(r.config, indicator), "enabled", False) else 0.0
                for r in backtest_results
            ])
            
            # 計算相關係數
            if np.std(enabled_states) == 0 or np.std(win_rates) == 0:
                correlations[indicator] = 0.0
            else:
                correlation = np.corrcoef(enabled_states, win_rates)[0, 1]
                correlations[indicator] = float(correlation) if not np.isnan(correlation) else 0.0
        
        return correlations

    def tune_with_all_results(
        self,
        symbol: str,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        pattern_scores: List[float],
    ) -> Tuple[BacktestResult, List[BacktestResult], Dict[str, float]]:
        """
        執行自動調參並返回所有結果（用於相關性分析）
        
        三階段優化：
        1. 指標組合優化
        2. 權重優化
        3. 閾值優化
        
        使用預計算指標加速回測（如果 GPU 可用則使用 GPU 加速）。
        
        Args:
            symbol: 股票代碼
            prices: 歷史收盤價
            highs: 歷史最高價
            lows: 歷史最低價
            volumes: 歷史成交量
            pattern_scores: 歷史型態分數
            
        Returns:
            (最佳結果, 所有結果列表, 相關性矩陣)
        """
        all_results: List[BacktestResult] = []
        best_result: Optional[BacktestResult] = None
        best_win_rate = 0.0
        
        # 預計算所有指標（一次性計算，避免重複）
        precomputed = self._precompute_indicators(prices, highs, lows, volumes)
        
        # ========== 階段 1: 指標組合 ==========
        combinations = self._generate_combinations(symbol)
        total_phase1 = len(combinations)
        
        for idx, config in enumerate(combinations):
            if self.progress_callback:
                progress = TuningProgress(
                    current_combination=idx + 1,
                    total_combinations=total_phase1,
                    best_win_rate_so_far=best_win_rate,
                    current_config_description=self._get_config_description(config),
                    phase="階段1: 指標組合",
                )
                self.progress_callback(progress)
            
            result = self._backtest_config(
                config=config,
                prices=prices,
                highs=highs,
                lows=lows,
                volumes=volumes,
                pattern_scores=pattern_scores,
                precomputed_indicators=precomputed,
            )
            all_results.append(result)
            
            if result.win_rate > best_win_rate or best_result is None:
                best_win_rate = result.win_rate
                best_result = result
        
        if best_result is None:
            default_config = self.config_manager.get_default_config(symbol)
            best_result = BacktestResult(
                config=default_config,
                total_trades=0,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
            )
            return best_result, all_results, self.calculate_correlation_matrix(all_results)
        
        # ========== 階段 2: 權重優化 ==========
        if self.options.optimize_weights:
            weight_combinations = self._generate_weight_combinations(best_result.config)
            total_phase2 = len(weight_combinations)
            
            for idx, config in enumerate(weight_combinations):
                if self.progress_callback:
                    progress = TuningProgress(
                        current_combination=idx + 1,
                        total_combinations=total_phase2,
                        best_win_rate_so_far=best_win_rate,
                        current_config_description=self._get_config_description(config),
                        phase="階段2: 權重優化",
                    )
                    self.progress_callback(progress)
                
                result = self._backtest_config(
                    config=config,
                    prices=prices,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    pattern_scores=pattern_scores,
                    precomputed_indicators=precomputed,
                )
                all_results.append(result)
                
                if result.win_rate > best_win_rate:
                    best_win_rate = result.win_rate
                    best_result = result
        
        # ========== 階段 3: 閾值優化 ==========
        if self.options.optimize_thresholds:
            threshold_combinations = self._generate_threshold_combinations(best_result.config)
            total_phase3 = len(threshold_combinations)
            
            for idx, config in enumerate(threshold_combinations):
                if self.progress_callback:
                    progress = TuningProgress(
                        current_combination=idx + 1,
                        total_combinations=total_phase3,
                        best_win_rate_so_far=best_win_rate,
                        current_config_description=self._get_config_description(config),
                        phase="階段3: 閾值優化",
                    )
                    self.progress_callback(progress)
                
                result = self._backtest_config(
                    config=config,
                    prices=prices,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    pattern_scores=pattern_scores,
                    precomputed_indicators=precomputed,
                )
                all_results.append(result)
                
                if result.win_rate > best_win_rate:
                    best_win_rate = result.win_rate
                    best_result = result
        
        # ========== 階段 4: RSI 參數優化 ==========
        if self.options.optimize_rsi_params and best_result.config.rsi.enabled:
            rsi_combinations = self._generate_rsi_param_combinations(best_result.config)
            total_phase4 = len(rsi_combinations)
            
            for idx, config in enumerate(rsi_combinations):
                if self.progress_callback:
                    progress = TuningProgress(
                        current_combination=idx + 1,
                        total_combinations=total_phase4,
                        best_win_rate_so_far=best_win_rate,
                        current_config_description=self._get_config_description(config),
                        phase="階段4: RSI參數優化",
                    )
                    self.progress_callback(progress)
                
                result = self._backtest_config(
                    config=config,
                    prices=prices,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    pattern_scores=pattern_scores,
                    precomputed_indicators=precomputed,
                )
                all_results.append(result)
                
                if result.win_rate > best_win_rate:
                    best_win_rate = result.win_rate
                    best_result = result
        
        # 儲存最佳配置
        self.config_manager.save_config(best_result.config)
        
        correlations = self.calculate_correlation_matrix(all_results)
        
        return best_result, all_results, correlations
