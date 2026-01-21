"""
演化優化實驗室 UI (Evolution Lab)

提供生物演化優化引擎的使用者介面，包含：
- 目標函數選擇器
- 演化參數配置介面
- 進度顯示與視覺化
- 結果展示（最佳基因組、演化曲線）

Requirements: 10.5, 12.4
"""

import streamlit as st
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Callable
import numpy as np
import pandas as pd

from pattern_quant.evolution.models import (
    Genome,
    Individual,
    DEFAULT_GENOME_BOUNDS,
)
from pattern_quant.evolution.engine import (
    EvolutionaryEngine,
    EvolutionConfig,
)
from pattern_quant.evolution.fitness import FitnessObjective
from pattern_quant.evolution.generation import GenerationStats, EvolutionHistory
from pattern_quant.evolution.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSummary,
)


# 目標函數中文名稱對照
FITNESS_OBJECTIVE_NAMES: Dict[FitnessObjective, str] = {
    FitnessObjective.SHARPE_RATIO: "夏普比率 (Sharpe Ratio)",
    FitnessObjective.SORTINO_RATIO: "索提諾比率 (Sortino Ratio)",
    FitnessObjective.NET_PROFIT: "淨利潤 (Net Profit)",
    FitnessObjective.MIN_MAX_DRAWDOWN: "最小化回撤 (Min Max Drawdown)",
}

FITNESS_OBJECTIVE_DESCRIPTIONS: Dict[FitnessObjective, str] = {
    FitnessObjective.SHARPE_RATIO: "風險調整後收益，適合追求穩定報酬的策略",
    FitnessObjective.SORTINO_RATIO: "下行風險調整後收益，只懲罰負報酬的波動",
    FitnessObjective.NET_PROFIT: "最大化總收益，適合追求高報酬的策略",
    FitnessObjective.MIN_MAX_DRAWDOWN: "最小化最大回撤，適合防禦型策略",
}


class EvolutionLab:
    """
    演化優化實驗室
    
    提供生物演化優化引擎的使用者介面。
    
    Requirements: 10.5, 12.4
    """
    
    def __init__(self):
        """初始化演化優化實驗室"""
        self._init_session_state()
    
    def _init_session_state(self) -> None:
        """初始化 session state"""
        if "evo_symbol" not in st.session_state:
            st.session_state.evo_symbol = "AAPL"
        if "evo_config" not in st.session_state:
            st.session_state.evo_config = EvolutionConfig()
        if "evo_in_progress" not in st.session_state:
            st.session_state.evo_in_progress = False
        if "evo_history" not in st.session_state:
            st.session_state.evo_history = None
        if "evo_best_genome" not in st.session_state:
            st.session_state.evo_best_genome = None
        if "wf_summary" not in st.session_state:
            st.session_state.wf_summary = None
        if "evo_generation_data" not in st.session_state:
            st.session_state.evo_generation_data = []

    def render_symbol_selector(self) -> str:
        """渲染股票選擇器"""
        st.subheader("📈 選擇股票")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbol = st.text_input(
                "股票代碼",
                value=st.session_state.evo_symbol,
                help="輸入股票代碼，如 AAPL、GOOGL、2330.TW",
                key="evo_symbol_input"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("確認", use_container_width=True, key="evo_confirm_symbol"):
                st.session_state.evo_symbol = symbol.upper()
                st.rerun()
        
        return st.session_state.evo_symbol
    
    def render_objective_selector(self) -> FitnessObjective:
        """
        渲染目標函數選擇器
        
        Requirements: 10.5
        """
        st.subheader("🎯 目標函數選擇")
        
        # 建立選項列表
        options = list(FitnessObjective)
        option_names = [FITNESS_OBJECTIVE_NAMES[obj] for obj in options]
        
        # 找出當前選擇的索引
        current_objective = st.session_state.evo_config.fitness_objective
        current_index = options.index(current_objective) if current_objective in options else 0
        
        selected_name = st.selectbox(
            "選擇優化目標",
            options=option_names,
            index=current_index,
            help="選擇演化優化的目標函數",
            key="evo_objective_select"
        )
        
        # 找出選擇的目標
        selected_index = option_names.index(selected_name)
        selected_objective = options[selected_index]
        
        # 顯示目標描述
        st.info(f"💡 {FITNESS_OBJECTIVE_DESCRIPTIONS[selected_objective]}")
        
        return selected_objective
    
    def render_evolution_params(self) -> EvolutionConfig:
        """
        渲染演化參數配置介面
        
        Requirements: 10.5
        """
        st.subheader("⚙️ 演化參數配置")
        
        config = st.session_state.evo_config
        
        # 基本參數
        st.markdown("**基本參數**")
        col1, col2 = st.columns(2)
        
        with col1:
            population_size = st.slider(
                "種群大小",
                min_value=50,
                max_value=100,
                value=config.population_size,
                step=10,
                help="每一世代的個體數量 (50-100)",
                key="evo_pop_size"
            )
            
            max_generations = st.slider(
                "最大世代數",
                min_value=10,
                max_value=50,
                value=config.max_generations,
                step=5,
                help="演化迭代的最大次數 (10-50)",
                key="evo_max_gen"
            )
        
        with col2:
            tournament_size = st.slider(
                "競賽選擇大小",
                min_value=2,
                max_value=10,
                value=config.tournament_size,
                step=1,
                help="競賽選擇時參與的個體數量",
                key="evo_tournament"
            )
            
            min_trades = st.slider(
                "最低交易次數",
                min_value=5,
                max_value=30,
                value=config.min_trades_threshold,
                step=5,
                help="低於此交易次數的個體適應度為零",
                key="evo_min_trades"
            )
        
        # 演化算子參數
        st.markdown("**演化算子參數**")
        col1, col2 = st.columns(2)
        
        with col1:
            elitism_rate = st.slider(
                "精英保留率",
                min_value=0.05,
                max_value=0.20,
                value=config.elitism_rate,
                step=0.05,
                format="%.2f",
                help="直接保留到下一代的最佳個體比例 (5%-20%)",
                key="evo_elitism"
            )
            
            crossover_rate = st.slider(
                "交叉率",
                min_value=0.6,
                max_value=0.9,
                value=config.crossover_rate,
                step=0.1,
                format="%.1f",
                help="執行交叉操作的機率 (60%-90%)",
                key="evo_crossover"
            )
        
        with col2:
            mutation_rate = st.slider(
                "突變率",
                min_value=0.01,
                max_value=0.05,
                value=config.mutation_rate,
                step=0.01,
                format="%.2f",
                help="每個基因發生突變的機率 (1%-5%)",
                key="evo_mutation"
            )
            
            mutation_strength = st.slider(
                "突變強度",
                min_value=0.05,
                max_value=0.30,
                value=config.mutation_strength,
                step=0.05,
                format="%.2f",
                help="高斯突變的標準差",
                key="evo_mut_strength"
            )
        
        # 收斂參數
        st.markdown("**收斂參數**")
        col1, col2 = st.columns(2)
        
        with col1:
            convergence_threshold = st.slider(
                "收斂閾值",
                min_value=0.0001,
                max_value=0.01,
                value=config.convergence_threshold,
                step=0.0001,
                format="%.4f",
                help="適應度改善低於此值視為收斂",
                key="evo_conv_thresh"
            )
        
        with col2:
            convergence_patience = st.slider(
                "收斂耐心值",
                min_value=3,
                max_value=10,
                value=config.convergence_patience,
                step=1,
                help="連續多少世代無改善後提前終止",
                key="evo_conv_patience"
            )
        
        return EvolutionConfig(
            population_size=population_size,
            max_generations=max_generations,
            tournament_size=tournament_size,
            elitism_rate=elitism_rate,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            fitness_objective=st.session_state.evo_config.fitness_objective,
            min_trades_threshold=min_trades,
            convergence_threshold=convergence_threshold,
            convergence_patience=convergence_patience,
        )

    def render_walk_forward_params(self) -> Optional[WalkForwardConfig]:
        """渲染滾動視窗參數配置"""
        st.subheader("📊 滾動視窗驗證 (Walk-Forward)")
        
        use_walk_forward = st.toggle(
            "啟用滾動視窗驗證",
            value=False,
            help="在多個時間視窗上驗證參數的泛化能力",
            key="evo_use_wf"
        )
        
        if not use_walk_forward:
            st.info("💡 啟用滾動視窗驗證可確保參數在未見數據上的有效性")
            return None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            in_sample_days = st.slider(
                "訓練視窗 (天)",
                min_value=126,
                max_value=504,
                value=252,
                step=21,
                help="用於演化優化的歷史數據天數（約半年到兩年）",
                key="evo_is_days"
            )
        
        with col2:
            out_of_sample_days = st.slider(
                "測試視窗 (天)",
                min_value=21,
                max_value=126,
                value=63,
                step=21,
                help="用於驗證最佳參數的數據天數（約一個月到半年）",
                key="evo_oos_days"
            )
        
        with col3:
            step_size_days = st.slider(
                "步進長度 (天)",
                min_value=5,
                max_value=63,
                value=21,
                step=7,
                help="視窗推進的步長（約一週到一季）",
                key="evo_step_days"
            )
        
        return WalkForwardConfig(
            in_sample_days=in_sample_days,
            out_of_sample_days=out_of_sample_days,
            step_size_days=step_size_days,
        )
    
    def render_data_period_selector(self) -> tuple:
        """渲染數據期間選擇器"""
        st.subheader("📅 數據期間")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_start = date.today() - timedelta(days=365 * 3)  # 3 年
            start_date = st.date_input(
                "起始日期",
                value=default_start,
                help="歷史數據起始日期",
                key="evo_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "結束日期",
                value=date.today(),
                help="歷史數據結束日期",
                key="evo_end_date"
            )
        
        return start_date, end_date
    
    def render_run_controls(
        self,
        symbol: str,
        config: EvolutionConfig,
        wf_config: Optional[WalkForwardConfig],
        start_date: date,
        end_date: date,
    ) -> None:
        """渲染執行控制區"""
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if wf_config:
                button_text = "🧬 執行滾動視窗演化優化"
            else:
                button_text = "🧬 執行演化優化"
            
            run_button = st.button(
                button_text,
                type="primary",
                use_container_width=True,
                disabled=st.session_state.evo_in_progress,
                key="evo_run_button"
            )
        
        if run_button:
            self._run_evolution(symbol, config, wf_config, start_date, end_date)
    
    def _run_evolution(
        self,
        symbol: str,
        config: EvolutionConfig,
        wf_config: Optional[WalkForwardConfig],
        start_date: date,
        end_date: date,
    ) -> None:
        """執行演化優化"""
        st.session_state.evo_in_progress = True
        st.session_state.evo_generation_data = []
        
        # 進度顯示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 抓取歷史數據
            status_text.text("正在抓取歷史數據...")
            
            from pattern_quant.data.yfinance_source import YFinanceDataSource
            data_source = YFinanceDataSource()
            
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())
            
            raw_data = data_source.fetch_ohlcv(symbol, start_dt, end_dt)
            
            if not raw_data or len(raw_data) < 100:
                st.error("❌ 數據不足，無法執行演化優化")
                st.session_state.evo_in_progress = False
                return
            
            prices = [d['close'] for d in raw_data]
            highs = [d['high'] for d in raw_data]
            lows = [d['low'] for d in raw_data]
            volumes = [d['volume'] for d in raw_data]
            
            status_text.text(f"已載入 {len(prices)} 筆數據")
            
            # 建立演化引擎
            engine = EvolutionaryEngine(config=config)
            
            if wf_config:
                # 滾動視窗優化
                def wf_progress_callback(window_idx: int, total: int, result: WalkForwardResult):
                    pct = (window_idx + 1) / total
                    progress_bar.progress(pct)
                    status_text.text(
                        f"滾動視窗 {window_idx + 1}/{total} | "
                        f"IS 適應度: {result.in_sample_fitness:.4f} | "
                        f"OOS 適應度: {result.out_of_sample_fitness:.4f}"
                    )
                
                summary = engine.walk_forward_optimize(
                    symbol=symbol,
                    prices=prices,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    walk_forward_config=wf_config,
                    progress_callback=wf_progress_callback,
                )
                
                st.session_state.wf_summary = summary
                st.session_state.evo_best_genome = summary.windows[-1].best_genome if summary.windows else None
                st.session_state.evo_history = None
                
            else:
                # 單次演化優化
                def evo_progress_callback(gen: int, stats: GenerationStats):
                    pct = (gen + 1) / config.max_generations
                    progress_bar.progress(pct)
                    status_text.text(
                        f"世代 {gen + 1}/{config.max_generations} | "
                        f"最佳: {stats.best_fitness:.4f} | "
                        f"平均: {stats.average_fitness:.4f}"
                    )
                    # 記錄世代數據用於繪圖
                    st.session_state.evo_generation_data.append({
                        "generation": gen + 1,
                        "best_fitness": stats.best_fitness,
                        "average_fitness": stats.average_fitness,
                        "worst_fitness": stats.worst_fitness,
                    })
                
                history = engine.optimize(
                    symbol=symbol,
                    prices=prices,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    progress_callback=evo_progress_callback,
                )
                
                st.session_state.evo_history = history
                st.session_state.evo_best_genome = history.final_best.genome
                st.session_state.wf_summary = None
            
            progress_bar.progress(1.0)
            status_text.text("✅ 演化優化完成！")
            
        except ImportError:
            st.error("❌ 請安裝 yfinance: pip install yfinance")
        except Exception as e:
            st.error(f"❌ 演化優化失敗: {e}")
        finally:
            st.session_state.evo_in_progress = False

    def render_evolution_curve(self) -> None:
        """
        渲染演化曲線
        
        Requirements: 12.4
        """
        st.subheader("📈 演化曲線")
        
        if not st.session_state.evo_generation_data:
            st.info("💡 執行演化優化後可查看演化曲線")
            return
        
        # 建立 DataFrame
        df = pd.DataFrame(st.session_state.evo_generation_data)
        
        # 使用 Streamlit 內建圖表
        st.line_chart(
            df.set_index("generation")[["best_fitness", "average_fitness", "worst_fitness"]],
            use_container_width=True,
        )
        
        # 圖例說明
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("🟢 最佳適應度")
        with col2:
            st.caption("🟡 平均適應度")
        with col3:
            st.caption("🔴 最差適應度")
        
        # 嘗試使用 Plotly 繪製更詳細的圖表
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df["generation"],
                y=df["best_fitness"],
                mode="lines+markers",
                name="最佳適應度",
                line=dict(color="green", width=2),
            ))
            
            fig.add_trace(go.Scatter(
                x=df["generation"],
                y=df["average_fitness"],
                mode="lines+markers",
                name="平均適應度",
                line=dict(color="orange", width=2),
            ))
            
            fig.add_trace(go.Scatter(
                x=df["generation"],
                y=df["worst_fitness"],
                mode="lines+markers",
                name="最差適應度",
                line=dict(color="red", width=2, dash="dash"),
            ))
            
            fig.update_layout(
                title="演化適應度曲線",
                xaxis_title="世代",
                yaxis_title="適應度",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True, key="evo_curve_plotly")
            
        except ImportError:
            pass  # 使用上面的 Streamlit 內建圖表
    
    def render_walk_forward_results(self) -> None:
        """渲染滾動視窗結果"""
        st.subheader("📊 滾動視窗驗證結果")
        
        summary = st.session_state.wf_summary
        if summary is None:
            st.info("💡 執行滾動視窗演化優化後可查看結果")
            return
        
        # 彙總指標
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "彙總報酬率",
                f"{summary.aggregate_return * 100:+.2f}%"
            )
        
        with col2:
            st.metric(
                "彙總夏普比率",
                f"{summary.aggregate_sharpe:.2f}"
            )
        
        with col3:
            st.metric(
                "視窗勝率",
                f"{summary.aggregate_win_rate * 100:.1f}%"
            )
        
        with col4:
            st.metric(
                "穩健性評分",
                f"{summary.robustness_score:.2f}"
            )
        
        # 各視窗詳情
        st.markdown("**各視窗詳情**")
        
        window_data = []
        for w in summary.windows:
            window_data.append({
                "視窗": w.window_index + 1,
                "IS 起始": w.in_sample_start,
                "IS 結束": w.in_sample_end,
                "OOS 起始": w.out_of_sample_start,
                "OOS 結束": w.out_of_sample_end,
                "IS 適應度": f"{w.in_sample_fitness:.4f}",
                "OOS 適應度": f"{w.out_of_sample_fitness:.4f}",
                "OOS 交易數": w.out_of_sample_trades,
                "OOS 報酬": f"{w.out_of_sample_return * 100:+.2f}%",
            })
        
        st.dataframe(
            pd.DataFrame(window_data),
            use_container_width=True,
            hide_index=True,
        )
        
        # 繪製 IS vs OOS 適應度對比圖
        try:
            import plotly.graph_objects as go
            
            windows = [w.window_index + 1 for w in summary.windows]
            is_fitness = [w.in_sample_fitness for w in summary.windows]
            oos_fitness = [w.out_of_sample_fitness for w in summary.windows]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=windows,
                y=is_fitness,
                name="訓練視窗 (IS)",
                marker_color="blue",
            ))
            
            fig.add_trace(go.Bar(
                x=windows,
                y=oos_fitness,
                name="測試視窗 (OOS)",
                marker_color="orange",
            ))
            
            fig.update_layout(
                title="訓練視窗 vs 測試視窗適應度對比",
                xaxis_title="視窗編號",
                yaxis_title="適應度",
                barmode="group",
                height=350,
            )
            
            st.plotly_chart(fig, use_container_width=True, key="wf_comparison")
            
        except ImportError:
            pass

    def render_best_genome(self) -> None:
        """
        渲染最佳基因組結果
        
        Requirements: 12.4
        """
        st.subheader("🧬 最佳基因組")
        
        genome = st.session_state.evo_best_genome
        if genome is None:
            st.info("💡 執行演化優化後可查看最佳基因組")
            return
        
        # 歸一化權重以便顯示
        normalized = genome.normalize_weights()
        
        # 雙引擎控制基因
        st.markdown("**🎛️ 雙引擎控制基因 (Segment A)**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("趨勢閾值", f"{normalized.dual_engine.trend_threshold:.2f}")
            st.metric("震盪閾值", f"{normalized.dual_engine.range_threshold:.2f}")
        
        with col2:
            st.metric("趨勢資金權重", f"{normalized.dual_engine.trend_allocation:.2f}")
            st.metric("震盪資金權重", f"{normalized.dual_engine.range_allocation:.2f}")
        
        with col3:
            st.metric("波動穩定性", f"{normalized.dual_engine.volatility_stability:.3f}")
        
        st.divider()
        
        # 因子權重基因
        st.markdown("**⚖️ 因子權重基因 (Segment B)**")
        
        # 權重視覺化
        weights = {
            "RSI": normalized.factor_weights.rsi_weight,
            "成交量": normalized.factor_weights.volume_weight,
            "MACD": normalized.factor_weights.macd_weight,
            "均線": normalized.factor_weights.ema_weight,
            "布林通道": normalized.factor_weights.bollinger_weight,
        }
        
        # 使用進度條顯示權重
        for name, weight in weights.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{name}**")
            with col2:
                st.progress(weight, text=f"{weight:.2%}")
        
        st.metric("買入閾值", f"{normalized.factor_weights.score_threshold:.1f}")
        
        st.divider()
        
        # 微觀指標基因
        st.markdown("**🔬 微觀指標基因 (Segment C)**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("RSI 週期", f"{normalized.micro_indicators.rsi_period}")
            st.metric("RSI 超買線", f"{normalized.micro_indicators.rsi_overbought:.1f}")
        
        with col2:
            st.metric("RSI 超賣線", f"{normalized.micro_indicators.rsi_oversold:.1f}")
            st.metric("成交量突變倍數", f"{normalized.micro_indicators.volume_spike_multiplier:.2f}")
        
        with col3:
            st.metric("MACD 加成", f"{normalized.micro_indicators.macd_bonus:.1f}")
            st.metric("布林壓縮閾值", f"{normalized.micro_indicators.bollinger_squeeze_threshold:.3f}")
        
        st.divider()
        
        # 匯出功能
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📋 複製 JSON", key="evo_copy_json"):
                json_str = genome.to_json()
                st.code(json_str, language="json")
        
        with col2:
            if st.button("💾 下載基因組", key="evo_download"):
                json_str = genome.to_json()
                st.download_button(
                    label="下載 JSON",
                    data=json_str,
                    file_name=f"genome_{st.session_state.evo_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="evo_download_btn"
                )
        
        with col3:
            if st.button("🔄 套用到策略", key="evo_apply"):
                self._apply_genome_to_strategy(genome)
    
    def _apply_genome_to_strategy(self, genome: Genome) -> None:
        """將基因組套用到策略配置"""
        try:
            from pattern_quant.evolution.engine import EvolutionaryEngine
            
            engine = EvolutionaryEngine()
            dual_config, factor_config = engine.genome_to_configs(
                genome, st.session_state.evo_symbol
            )
            
            # 儲存到 session state 供其他模組使用
            st.session_state.evolved_dual_config = dual_config
            st.session_state.evolved_factor_config = factor_config
            
            st.success("✅ 已將最佳基因組套用到策略配置！")
            st.info("💡 可在「因子權重實驗室」中查看並調整配置")
            
        except Exception as e:
            st.error(f"❌ 套用失敗: {e}")
    
    def render_evolution_history(self) -> None:
        """渲染演化歷史詳情"""
        history = st.session_state.evo_history
        if history is None:
            return
        
        with st.expander("📜 演化歷史詳情"):
            st.markdown(f"**總世代數**: {history.total_generations}")
            st.markdown(f"**是否收斂**: {'是' if history.converged else '否'}")
            st.markdown(f"**最終最佳適應度**: {history.final_best.fitness:.4f}")
            
            # 各世代統計表格
            gen_data = []
            for stats in history.generations:
                gen_data.append({
                    "世代": stats.generation + 1,
                    "最佳適應度": f"{stats.best_fitness:.4f}",
                    "平均適應度": f"{stats.average_fitness:.4f}",
                    "最差適應度": f"{stats.worst_fitness:.4f}",
                })
            
            st.dataframe(
                pd.DataFrame(gen_data),
                use_container_width=True,
                hide_index=True,
            )

    def render(self) -> None:
        """
        渲染完整的演化優化實驗室頁面
        
        Requirements: 10.5, 12.4
        """
        st.header("🧬 演化優化實驗室")
        st.markdown(
            "使用生物演化優化引擎自動尋找最佳策略參數，"
            "透過遺傳演算法在多維度參數空間中搜索全域最優解。"
        )
        st.divider()
        
        # 使用 tabs 組織內容
        tab1, tab2, tab3 = st.tabs([
            "⚙️ 參數配置",
            "📈 演化結果",
            "🧬 最佳基因組"
        ])
        
        with tab1:
            # 股票選擇
            symbol = self.render_symbol_selector()
            st.divider()
            
            # 目標函數選擇
            objective = self.render_objective_selector()
            st.session_state.evo_config.fitness_objective = objective
            st.divider()
            
            # 演化參數配置
            config = self.render_evolution_params()
            config.fitness_objective = objective
            st.session_state.evo_config = config
            st.divider()
            
            # 滾動視窗配置
            wf_config = self.render_walk_forward_params()
            st.divider()
            
            # 數據期間選擇
            start_date, end_date = self.render_data_period_selector()
            
            # 執行控制
            self.render_run_controls(symbol, config, wf_config, start_date, end_date)
        
        with tab2:
            # 演化曲線
            self.render_evolution_curve()
            st.divider()
            
            # 滾動視窗結果
            self.render_walk_forward_results()
            st.divider()
            
            # 演化歷史詳情
            self.render_evolution_history()
        
        with tab3:
            # 最佳基因組
            self.render_best_genome()


def run_evolution_lab():
    """執行演化優化實驗室"""
    lab = EvolutionLab()
    lab.render()


if __name__ == "__main__":
    st.set_page_config(
        page_title="AI PatternQuant - 演化優化實驗室",
        page_icon="🧬",
        layout="wide"
    )
    run_evolution_lab()
