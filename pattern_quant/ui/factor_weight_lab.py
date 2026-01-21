"""
因子權重實驗室 UI (Factor Weight Lab)

提供使用者調整各指標的啟用狀態與權重的介面，
包含 RSI 詳細設定面板、自動調參整合與指標相關性熱力圖。

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 12.1, 12.2, 12.3, 12.4, 13.1, 13.2, 13.3
"""

import streamlit as st
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Callable
import numpy as np
import pandas as pd
import importlib

from pattern_quant.optimization.factor_config import (
    FactorConfig,
    FactorConfigManager,
    RSIConfig,
    VolumeConfig,
    MACDConfig,
    EMAConfig,
    BollingerConfig,
    IndicatorType,
)
from pattern_quant.optimization.indicator_pool import IndicatorPool
from pattern_quant.optimization.signal_optimizer import SignalOptimizer

# 強制重新載入 auto_tuner 模組以確保使用最新版本
import pattern_quant.optimization.auto_tuner as auto_tuner_module
importlib.reload(auto_tuner_module)
from pattern_quant.optimization.auto_tuner import AutoTuner, TuningProgress, BacktestResult


class FactorWeightLab:
    """
    因子權重實驗室
    
    提供使用者調整各指標的啟用狀態與權重的介面。
    """
    
    def __init__(
        self,
        config_manager: Optional[FactorConfigManager] = None,
        indicator_pool: Optional[IndicatorPool] = None,
    ):
        """
        初始化因子權重實驗室
        
        Args:
            config_manager: 因子配置管理器
            indicator_pool: 指標計算庫
        """
        # 使用共享的配置管理器，確保回測系統能使用相同的配置
        if config_manager:
            self.config_manager = config_manager
        elif 'shared_config_manager' in st.session_state:
            self.config_manager = st.session_state.shared_config_manager
        else:
            self.config_manager = FactorConfigManager()
            st.session_state.shared_config_manager = self.config_manager
        
        self.indicator_pool = indicator_pool or IndicatorPool()
        self._init_session_state()
    
    def _init_session_state(self) -> None:
        """初始化 session state"""
        if "factor_lab_symbol" not in st.session_state:
            st.session_state.factor_lab_symbol = "AAPL"
        if "factor_lab_config" not in st.session_state:
            st.session_state.factor_lab_config = None
        if "tuning_in_progress" not in st.session_state:
            st.session_state.tuning_in_progress = False
        if "tuning_results" not in st.session_state:
            st.session_state.tuning_results = None
        if "correlation_matrix" not in st.session_state:
            st.session_state.correlation_matrix = None
        if "all_backtest_results" not in st.session_state:
            st.session_state.all_backtest_results = None
    
    def _clear_widget_cache(self) -> None:
        """清除所有 slider 和 toggle 的緩存值，強制 UI 更新"""
        keys_to_clear = [
            "rsi_toggle", "rsi_weight",
            "volume_toggle", "volume_weight",
            "macd_toggle", "macd_weight",
            "ema_toggle", "ema_weight",
            "bollinger_toggle", "bollinger_weight",
            "buy_threshold", "watch_threshold",
            "use_atr_stop", "atr_multiplier",
            "rsi_trend_lower", "rsi_trend_upper",
            "rsi_overbought", "rsi_oversold",
            "rsi_check_divergence",
            "rsi_trend_bonus", "rsi_support_bonus",
            "rsi_overbought_penalty", "rsi_divergence_penalty",
            "rsi_weak_penalty",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    def render_symbol_selector(self) -> str:
        """渲染股票選擇器"""
        st.subheader("📈 選擇股票")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbol = st.text_input(
                "股票代碼",
                value=st.session_state.factor_lab_symbol,
                help="輸入股票代碼，如 AAPL、GOOGL、2330.TW"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("載入配置", use_container_width=True):
                st.session_state.factor_lab_symbol = symbol.upper()
                st.session_state.factor_lab_config = self.config_manager.get_config(symbol.upper())
                st.rerun()
        
        # 顯示已配置的股票列表
        configured_symbols = self.config_manager.list_configured_symbols()
        if configured_symbols:
            st.caption(f"已配置的股票: {', '.join(configured_symbols)}")
        
        return st.session_state.factor_lab_symbol
    
    def render_indicator_toggles(self, config: FactorConfig) -> FactorConfig:
        """
        渲染指標開關與權重滑桿
        
        Requirements: 11.1, 11.2, 11.3
        """
        st.subheader("🎛️ 指標開關與權重")
        
        # 快速預設按鈕
        st.markdown("**快速預設：**")
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        with preset_col1:
            if st.button("僅 RSI", use_container_width=True, key="preset_rsi_only"):
                config.rsi.enabled = True
                config.volume.enabled = False
                config.macd.enabled = False
                config.ema.enabled = False
                config.bollinger.enabled = False
                st.session_state.factor_lab_config = config
                st.rerun()
        
        with preset_col2:
            if st.button("RSI + 成交量", use_container_width=True, key="preset_rsi_vol"):
                config.rsi.enabled = True
                config.volume.enabled = True
                config.macd.enabled = False
                config.ema.enabled = False
                config.bollinger.enabled = False
                st.session_state.factor_lab_config = config
                st.rerun()
        
        with preset_col3:
            if st.button("全部啟用", use_container_width=True, key="preset_all"):
                config.rsi.enabled = True
                config.volume.enabled = True
                config.macd.enabled = True
                config.ema.enabled = True
                config.bollinger.enabled = True
                st.session_state.factor_lab_config = config
                st.rerun()
        
        with preset_col4:
            if st.button("全部停用", use_container_width=True, key="preset_none"):
                config.rsi.enabled = False
                config.volume.enabled = False
                config.macd.enabled = False
                config.ema.enabled = False
                config.bollinger.enabled = False
                st.session_state.factor_lab_config = config
                st.rerun()
        
        st.markdown("---")
        
        # RSI 設定
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            rsi_enabled = st.toggle("RSI", value=config.rsi.enabled, key="rsi_toggle")
        with col2:
            rsi_weight = st.slider(
                "RSI 權重",
                0.0, 2.0, float(config.rsi.weight), 0.1,
                disabled=not rsi_enabled,
                key="rsi_weight"
            )
        with col3:
            if rsi_enabled:
                st.caption(f"權重: {rsi_weight:.1f}")
        
        # Volume 設定
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            volume_enabled = st.toggle("成交量", value=config.volume.enabled, key="volume_toggle")
        with col2:
            volume_weight = st.slider(
                "成交量權重",
                0.0, 2.0, float(config.volume.weight), 0.1,
                disabled=not volume_enabled,
                key="volume_weight"
            )
        with col3:
            if volume_enabled:
                st.caption(f"權重: {volume_weight:.1f}")
        
        # MACD 設定
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            macd_enabled = st.toggle("MACD", value=config.macd.enabled, key="macd_toggle")
        with col2:
            macd_weight = st.slider(
                "MACD 權重",
                0.0, 2.0, float(config.macd.weight), 0.1,
                disabled=not macd_enabled,
                key="macd_weight"
            )
        with col3:
            if macd_enabled:
                st.caption(f"權重: {macd_weight:.1f}")
        
        # EMA 設定
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            ema_enabled = st.toggle("均線 (EMA)", value=config.ema.enabled, key="ema_toggle")
        with col2:
            ema_weight = st.slider(
                "均線權重",
                0.0, 2.0, float(config.ema.weight), 0.1,
                disabled=not ema_enabled,
                key="ema_weight"
            )
        with col3:
            if ema_enabled:
                st.caption(f"權重: {ema_weight:.1f}")
        
        # Bollinger 設定
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            bollinger_enabled = st.toggle("布林通道", value=config.bollinger.enabled, key="bollinger_toggle")
        with col2:
            bollinger_weight = st.slider(
                "布林通道權重",
                0.0, 2.0, float(config.bollinger.weight), 0.1,
                disabled=not bollinger_enabled,
                key="bollinger_weight"
            )
        with col3:
            if bollinger_enabled:
                st.caption(f"權重: {bollinger_weight:.1f}")
        
        # 更新配置
        config.rsi.enabled = rsi_enabled
        config.rsi.weight = rsi_weight
        config.volume.enabled = volume_enabled
        config.volume.weight = volume_weight
        config.macd.enabled = macd_enabled
        config.macd.weight = macd_weight
        config.ema.enabled = ema_enabled
        config.ema.weight = ema_weight
        config.bollinger.enabled = bollinger_enabled
        config.bollinger.weight = bollinger_weight
        
        return config

    def render_threshold_settings(self, config: FactorConfig) -> FactorConfig:
        """渲染閾值設定"""
        st.subheader("🎯 訊號閾值設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            buy_threshold = st.slider(
                "買入閾值",
                50.0, 100.0, float(config.buy_threshold), 5.0,
                help="最終分數超過此閾值時生成強烈買入訊號",
                key="buy_threshold"
            )
        
        with col2:
            watch_threshold = st.slider(
                "觀望閾值",
                30.0, 80.0, float(config.watch_threshold), 5.0,
                help="最終分數介於觀望閾值與買入閾值之間時生成觀望訊號",
                key="watch_threshold"
            )
        
        # ATR 止損設定
        st.markdown("**ATR 動態止損**")
        col1, col2 = st.columns(2)
        
        with col1:
            use_atr_stop = st.toggle(
                "啟用 ATR 動態止損",
                value=config.use_atr_stop_loss,
                key="use_atr_stop"
            )
        
        with col2:
            atr_multiplier = st.slider(
                "ATR 倍數",
                1.0, 4.0, float(config.atr_multiplier), 0.5,
                disabled=not use_atr_stop,
                help="止損價位 = 進場價 - (ATR × 倍數)",
                key="atr_multiplier"
            )
        
        config.buy_threshold = buy_threshold
        config.watch_threshold = watch_threshold
        config.use_atr_stop_loss = use_atr_stop
        config.atr_multiplier = atr_multiplier
        
        return config
    
    def render_rsi_detail_panel(self, config: FactorConfig) -> FactorConfig:
        """
        渲染 RSI 詳細設定面板
        
        Requirements: 12.1, 12.2, 12.3, 12.4
        """
        st.subheader("📊 RSI 詳細設定")
        
        if not config.rsi.enabled:
            st.info("RSI 指標已停用，啟用後可調整詳細設定")
            return config
        
        # 趨勢區間設定
        st.markdown("**趨勢區間設定**")
        col1, col2 = st.columns(2)
        
        with col1:
            trend_lower = st.slider(
                "趨勢區間下限",
                30.0, 60.0, float(config.rsi.trend_lower), 5.0,
                help="RSI 高於此值視為趨勢區間",
                key="rsi_trend_lower"
            )
        
        with col2:
            trend_upper = st.slider(
                "趨勢區間上限",
                60.0, 85.0, float(config.rsi.trend_upper), 5.0,
                help="RSI 低於此值視為趨勢區間",
                key="rsi_trend_upper"
            )
        
        # 超買/超賣閾值設定
        st.markdown("**超買/超賣閾值**")
        col1, col2 = st.columns(2)
        
        with col1:
            overbought = st.slider(
                "超買閾值",
                70.0, 95.0, float(config.rsi.overbought), 5.0,
                help="RSI 超過此值視為超買",
                key="rsi_overbought"
            )
        
        with col2:
            oversold = st.slider(
                "超賣閾值",
                5.0, 40.0, float(config.rsi.oversold), 5.0,
                help="RSI 低於此值視為超賣",
                key="rsi_oversold"
            )
        
        # 背離偵測開關
        st.markdown("**背離偵測**")
        check_divergence = st.toggle(
            "啟用 RSI 背離偵測",
            value=config.rsi.check_divergence,
            help="偵測價格與 RSI 的背離現象",
            key="rsi_check_divergence"
        )
        
        # 分數調整
        st.markdown("**分數調整**")
        col1, col2 = st.columns(2)
        
        with col1:
            trend_zone_bonus = st.slider(
                "趨勢區間加分",
                0.0, 30.0, float(config.rsi.trend_zone_bonus), 5.0,
                key="rsi_trend_bonus"
            )
            support_bounce_bonus = st.slider(
                "支撐反彈加分",
                0.0, 30.0, float(config.rsi.support_bounce_bonus), 5.0,
                key="rsi_support_bonus"
            )
        
        with col2:
            overbought_penalty = st.slider(
                "超買扣分",
                -40.0, 0.0, float(config.rsi.overbought_penalty), 5.0,
                key="rsi_overbought_penalty"
            )
            divergence_penalty = st.slider(
                "背離扣分",
                -30.0, 0.0, float(config.rsi.divergence_penalty), 5.0,
                disabled=not check_divergence,
                key="rsi_divergence_penalty"
            )
        
        # 更新配置
        config.rsi.trend_lower = trend_lower
        config.rsi.trend_upper = trend_upper
        config.rsi.overbought = overbought
        config.rsi.oversold = oversold
        config.rsi.check_divergence = check_divergence
        config.rsi.trend_zone_bonus = trend_zone_bonus
        config.rsi.support_bounce_bonus = support_bounce_bonus
        config.rsi.overbought_penalty = overbought_penalty
        config.rsi.divergence_penalty = divergence_penalty
        
        return config

    def render_auto_tune_section(self, symbol: str, config: FactorConfig) -> None:
        """
        渲染自動調參 UI 整合
        
        Requirements: 11.4, 11.5
        """
        st.subheader("🤖 自動調參")
        
        st.markdown("""
        自動調參會測試多種指標組合，找出勝率最高的配置。
        需要提供歷史數據進行回測。
        """)
        
        # 回測期間設定
        col1, col2 = st.columns(2)
        
        with col1:
            default_start = date.today() - timedelta(days=365 * 3)  # 3 年
            start_date = st.date_input(
                "回測起始日",
                value=default_start,
                key="tune_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "回測結束日",
                value=date.today(),
                key="tune_end_date"
            )
        
        # Auto-Tune 按鈕
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            tune_button = st.button(
                "🚀 Auto-Tune",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.tuning_in_progress,
                key="auto_tune_button"
            )
        
        if tune_button:
            self._run_auto_tune(symbol, start_date, end_date)
        
        # 顯示調參進度
        if st.session_state.tuning_in_progress:
            st.info("⏳ 調參進行中...")
        
        # 顯示調參結果
        if st.session_state.tuning_results is not None:
            self._render_tuning_results(st.session_state.tuning_results)
    
    def _run_auto_tune(self, symbol: str, start_date: date, end_date: date) -> None:
        """執行自動調參"""
        st.session_state.tuning_in_progress = True
        
        # 進度顯示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(progress: TuningProgress) -> None:
            pct = progress.current_combination / progress.total_combinations
            progress_bar.progress(pct)
            # 取得階段名稱（相容舊版本）
            phase_name = getattr(progress, 'phase', '組合測試')
            status_text.text(
                f"[{phase_name}] "
                f"測試 {progress.current_combination}/{progress.total_combinations}: "
                f"{progress.current_config_description} | "
                f"目前最佳勝率: {progress.best_win_rate_so_far * 100:.1f}%"
            )
        
        try:
            # 抓取歷史數據
            status_text.text("正在抓取歷史數據...")
            
            from pattern_quant.data.yfinance_source import YFinanceDataSource
            data_source = YFinanceDataSource()
            
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())
            
            raw_data = data_source.fetch_ohlcv(symbol, start_dt, end_dt)
            
            if not raw_data or len(raw_data) < 100:
                st.error("❌ 數據不足，無法執行自動調參")
                st.session_state.tuning_in_progress = False
                return
            
            prices = [d['close'] for d in raw_data]
            highs = [d['high'] for d in raw_data]
            lows = [d['low'] for d in raw_data]
            volumes = [d['volume'] for d in raw_data]
            
            # 生成模擬型態分數（實際應用中應從 PatternEngine 取得）
            pattern_scores = [50.0 + np.random.uniform(-10, 30) for _ in prices]
            
            # 建立 AutoTuner
            signal_optimizer = SignalOptimizer(self.indicator_pool, self.config_manager)
            auto_tuner = AutoTuner(
                indicator_pool=self.indicator_pool,
                signal_optimizer=signal_optimizer,
                config_manager=self.config_manager,
                progress_callback=progress_callback,
            )
            
            # 執行調參並取得所有結果
            best_result, all_results, correlations = auto_tuner.tune_with_all_results(
                symbol=symbol,
                prices=prices,
                highs=highs,
                lows=lows,
                volumes=volumes,
                pattern_scores=pattern_scores,
            )
            
            # 儲存結果
            st.session_state.tuning_results = best_result
            st.session_state.all_backtest_results = all_results
            st.session_state.correlation_matrix = correlations
            st.session_state.factor_lab_config = best_result.config
            
            progress_bar.progress(1.0)
            status_text.text("✅ 調參完成！")
            
        except ImportError:
            st.error("❌ 請安裝 yfinance: pip install yfinance")
        except Exception as e:
            st.error(f"❌ 調參失敗: {e}")
        finally:
            st.session_state.tuning_in_progress = False
    
    def _render_tuning_results(self, result: BacktestResult) -> None:
        """渲染調參結果"""
        st.markdown("---")
        st.markdown("**📊 調參結果**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("勝率", f"{result.win_rate * 100:.1f}%")
        with col2:
            st.metric("總報酬", f"{result.total_return * 100:+.2f}%")
        with col3:
            st.metric("最大回撤", f"{result.max_drawdown * 100:.2f}%")
        with col4:
            st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
        
        # 顯示最佳配置詳情
        st.markdown("**🎯 最佳指標組合與權重:**")
        
        config = result.config
        indicator_info = []
        
        if config.rsi.enabled:
            indicator_info.append(f"✅ RSI (權重: {config.rsi.weight:.1f})")
        else:
            indicator_info.append("❌ RSI")
            
        if config.volume.enabled:
            indicator_info.append(f"✅ 成交量 (權重: {config.volume.weight:.1f})")
        else:
            indicator_info.append("❌ 成交量")
            
        if config.macd.enabled:
            indicator_info.append(f"✅ MACD (權重: {config.macd.weight:.1f})")
        else:
            indicator_info.append("❌ MACD")
            
        if config.ema.enabled:
            indicator_info.append(f"✅ 均線 (權重: {config.ema.weight:.1f})")
        else:
            indicator_info.append("❌ 均線")
            
        if config.bollinger.enabled:
            indicator_info.append(f"✅ 布林通道 (權重: {config.bollinger.weight:.1f})")
        else:
            indicator_info.append("❌ 布林通道")
        
        # 使用兩欄顯示
        col1, col2 = st.columns(2)
        with col1:
            for info in indicator_info[:3]:
                st.markdown(f"- {info}")
        with col2:
            for info in indicator_info[3:]:
                st.markdown(f"- {info}")
        
        # 顯示閾值設定
        st.markdown("**📏 最佳閾值設定:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("買入閾值", f"{config.buy_threshold:.0f}")
        with col2:
            st.metric("觀望閾值", f"{config.watch_threshold:.0f}")
        with col3:
            st.metric("ATR 倍數", f"{config.atr_multiplier:.1f}")
        
        # 如果 RSI 啟用，顯示 RSI 詳細參數
        if config.rsi.enabled:
            st.markdown("**📊 RSI 最佳參數:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("趨勢區間下限", f"{config.rsi.trend_lower:.0f}")
            with col2:
                st.metric("趨勢區間上限", f"{config.rsi.trend_upper:.0f}")
            with col3:
                st.metric("超買閾值", f"{config.rsi.overbought:.0f}")
            with col4:
                st.metric("超賣閾值", f"{config.rsi.oversold:.0f}")
        
        # 套用最佳配置按鈕
        if st.button("📥 套用最佳配置", key="apply_best_config"):
            # 更新配置
            st.session_state.factor_lab_config = result.config
            
            # 清除緩存強制 UI 更新
            self._clear_widget_cache()
            
            st.success("✅ 已套用最佳配置！")
            st.rerun()

    def render_correlation_heatmap(self) -> None:
        """
        渲染指標相關性熱力圖
        
        Requirements: 13.1, 13.2, 13.3
        """
        st.subheader("🔥 指標相關性熱力圖")
        
        if st.session_state.correlation_matrix is None:
            st.info("💡 執行自動調參後可查看指標與勝率的相關性分析")
            return
        
        correlations = st.session_state.correlation_matrix
        
        # 準備熱力圖數據
        indicator_names = {
            "rsi": "RSI",
            "volume": "成交量",
            "macd": "MACD",
            "ema": "均線",
            "bollinger": "布林通道",
        }
        
        # 建立 DataFrame
        data = {
            "指標": [indicator_names.get(k, k) for k in correlations.keys()],
            "與勝率相關性": list(correlations.values()),
        }
        df = pd.DataFrame(data)
        
        # 顯示數據表格
        st.dataframe(
            df.style.background_gradient(
                subset=["與勝率相關性"],
                cmap="RdYlGn",
                vmin=-1,
                vmax=1,
            ),
            use_container_width=True,
            hide_index=True,
        )
        
        # 使用 Plotly 繪製熱力圖
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # 建立熱力圖矩陣
            indicators = list(indicator_names.values())
            corr_values = list(correlations.values())
            
            # 單行熱力圖
            fig = go.Figure(data=go.Heatmap(
                z=[corr_values],
                x=indicators,
                y=["勝率相關性"],
                colorscale="RdYlGn",
                zmin=-1,
                zmax=1,
                text=[[f"{v:.3f}" for v in corr_values]],
                texttemplate="%{text}",
                textfont={"size": 14},
                hovertemplate="指標: %{x}<br>相關性: %{z:.3f}<extra></extra>",
            ))
            
            fig.update_layout(
                title="指標與勝率相關性",
                xaxis_title="指標",
                yaxis_title="",
                height=200,
            )
            
            st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")
            
        except ImportError:
            st.warning("安裝 plotly 可獲得更好的視覺化效果: pip install plotly")
        
        # 相關性解讀
        st.markdown("**📖 相關性解讀:**")
        st.markdown("""
        - **正相關 (紅色)**: 啟用該指標傾向於提高勝率
        - **負相關 (綠色)**: 啟用該指標傾向於降低勝率
        - **接近零 (黃色)**: 該指標對勝率影響不明顯
        """)
        
        # 找出最重要的指標
        if correlations:
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            top_indicator = sorted_corr[0]
            st.info(
                f"💡 最具影響力的指標: **{indicator_names.get(top_indicator[0], top_indicator[0])}** "
                f"(相關性: {top_indicator[1]:.3f})"
            )
    
    def render_save_controls(self, config: FactorConfig) -> None:
        """渲染儲存控制區"""
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("💾 儲存配置", type="primary", use_container_width=True):
                if self.config_manager.save_config(config):
                    st.success(f"✅ 已儲存 {config.symbol} 的配置")
                else:
                    st.error("❌ 儲存失敗")
        
        with col2:
            if st.button("🔄 重置為預設", use_container_width=True):
                default_config = self.config_manager.get_default_config(config.symbol)
                st.session_state.factor_lab_config = default_config
                
                # 清除緩存強制 UI 更新
                self._clear_widget_cache()
                
                st.rerun()
        
        with col3:
            if st.button("🗑️ 刪除自訂配置", use_container_width=True):
                if self.config_manager.delete_config(config.symbol):
                    st.session_state.factor_lab_config = self.config_manager.get_default_config(config.symbol)
                    
                    # 清除緩存強制 UI 更新
                    self._clear_widget_cache()
                    
                    st.success(f"✅ 已刪除 {config.symbol} 的自訂配置")
                    st.rerun()
                else:
                    st.info("此股票沒有自訂配置")
    
    def render_config_summary(self, config: FactorConfig) -> None:
        """渲染配置摘要"""
        st.subheader("📋 配置摘要")
        
        # 啟用的指標
        enabled = []
        if config.rsi.enabled:
            enabled.append(f"RSI (權重: {config.rsi.weight:.1f})")
        if config.volume.enabled:
            enabled.append(f"成交量 (權重: {config.volume.weight:.1f})")
        if config.macd.enabled:
            enabled.append(f"MACD (權重: {config.macd.weight:.1f})")
        if config.ema.enabled:
            enabled.append(f"均線 (權重: {config.ema.weight:.1f})")
        if config.bollinger.enabled:
            enabled.append(f"布林通道 (權重: {config.bollinger.weight:.1f})")
        
        if enabled:
            st.markdown("**啟用的指標:**")
            for indicator in enabled:
                st.markdown(f"- {indicator}")
        else:
            st.warning("⚠️ 沒有啟用任何指標")
        
        # 閾值設定
        st.markdown(f"""
        **訊號閾值:**
        - 買入閾值: {config.buy_threshold}
        - 觀望閾值: {config.watch_threshold}
        - ATR 止損: {'啟用' if config.use_atr_stop_loss else '停用'} (倍數: {config.atr_multiplier})
        """)
    
    def render(self) -> None:
        """渲染完整的因子權重實驗室頁面"""
        st.header("⚗️ 因子權重實驗室")
        st.markdown("調整各指標的啟用狀態與權重，優化交易訊號。")
        st.divider()
        
        # 股票選擇
        symbol = self.render_symbol_selector()
        
        # 載入或初始化配置
        if st.session_state.factor_lab_config is None:
            st.session_state.factor_lab_config = self.config_manager.get_config(symbol)
        
        config = st.session_state.factor_lab_config
        
        # 確保配置的 symbol 與當前選擇一致
        if config.symbol != symbol:
            config = self.config_manager.get_config(symbol)
            st.session_state.factor_lab_config = config
        
        st.divider()
        
        # 使用 tabs 組織內容
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎛️ 指標設定",
            "📊 RSI 詳細設定",
            "🤖 自動調參",
            "🔥 相關性分析"
        ])
        
        with tab1:
            config = self.render_indicator_toggles(config)
            st.divider()
            config = self.render_threshold_settings(config)
            st.divider()
            self.render_config_summary(config)
        
        with tab2:
            config = self.render_rsi_detail_panel(config)
        
        with tab3:
            self.render_auto_tune_section(symbol, config)
        
        with tab4:
            self.render_correlation_heatmap()
        
        # 儲存控制
        self.render_save_controls(config)
        
        # 更新 session state
        st.session_state.factor_lab_config = config


def run_factor_weight_lab():
    """執行因子權重實驗室"""
    lab = FactorWeightLab()
    lab.render()


if __name__ == "__main__":
    st.set_page_config(
        page_title="AI PatternQuant - 因子權重實驗室",
        page_icon="⚗️",
        layout="wide"
    )
    run_factor_weight_lab()
