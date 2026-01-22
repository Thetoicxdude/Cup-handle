"""Enhanced Strategy Lab UI for AI PatternQuant

This module provides an enhanced Strategy Lab with:
- Real stock data from Yahoo Finance
- Trade detail charts showing why each trade was made
- Integration with chart view for pattern visualization
- Dual-Engine Strategy Mode integration
- Evolutionary Optimization integration for adaptive parameter tuning

Requirements: 12.1, 12.2, 12.3, 13.1, 13.2, 13.3, 13.4
"""

import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import altair as alt

from pattern_quant.core.models import OHLCV, PatternResult, CupPattern, HandlePattern, MatchScore
from pattern_quant.core.pattern_engine import PatternEngine
from pattern_quant.ui.chart_view import ChartView
from pattern_quant.strategy.models import DualEngineConfig, MarketState
from pattern_quant.strategy.config import DualEngineConfigManager
from pattern_quant.db.state_manager import get_state_manager
from pattern_quant.ui.simulation_runner import get_simulation_runner

# 演化優化模組
try:
    from pattern_quant.evolution import (
        EvolutionaryEngine,
        EvolutionConfig,
        FitnessObjective,
        Genome,
        WalkForwardConfig,
    )
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False


from pattern_quant.core.backtest_engine import (
    RealDataBacktestEngine,
    StrategyParameters,
    PortfolioAllocation,
    MixedPortfolioConfig,
    EnhancedBacktestTrade,
    StrategyPerformance,
    PerformanceDiff,
    StrategyComparisonReport,
    DualEngineBacktestReport,
    EnhancedBacktestResult,
    EvolutionBacktestConfig
)


    

    




class EnhancedStrategyLab:
    """增強版策略實驗室 UI"""
    
    def __init__(
        self,
        backtest_engine: Optional[RealDataBacktestEngine] = None,
        default_params: Optional[StrategyParameters] = None
    ):
        self.backtest_engine = backtest_engine or RealDataBacktestEngine()
        self.default_params = default_params or StrategyParameters()
        self.chart_view = ChartView()
    
    def render_parameter_sliders(self) -> Tuple[StrategyParameters, Optional[Dict[str, Any]]]:
        """渲染參數調整滑桿"""
        st.subheader("🎛️ 參數調整")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**杯身深度設定**")
            min_depth = st.slider(
                "最小杯身深度 (%)", 1.0, 30.0,
                self.default_params.min_depth, 1.0
            )
            max_depth = st.slider(
                "最大杯身深度 (%)", 10.0, 100.0,
                self.default_params.max_depth, 1.0
            )
            
            st.markdown("**成型天數設定**")
            min_cup_days = st.slider(
                "最小成型天數", 1, 100,
                self.default_params.min_cup_days, 5
            )
            max_cup_days = st.slider(
                "最大成型天數", 30, 500,
                self.default_params.max_cup_days, 10
            )
        
        with col2:
            st.markdown("**風控參數設定**")
            stop_loss_ratio = st.slider(
                "止損比例 (%)", 0.0, 100.0,
                self.default_params.stop_loss_ratio, 1.0
            )
            profit_threshold = st.slider(
                "移動止盈啟動閾值 (%)", 0.0, 100.0,
                self.default_params.profit_threshold, 1.0
            )
            trailing_ratio = st.slider(
                "移動止盈回調比例 (%)", 0.1, 20.0,
                self.default_params.trailing_ratio, 0.1
            )
            
            st.markdown("**訊號過濾設定**")
            score_threshold = st.slider(
                "吻合分數閾值", 20.0, 95.0,
                self.default_params.score_threshold, 5.0
            )
            
            st.markdown("**資金管理設定**")
            position_size = st.slider(
                "單筆倉位比例 (%)", 1.0, 100.0,
                self.default_params.position_size, 5.0,
                help="每筆交易投入的資金比例，越高報酬/風險越大"
            )
        


        # 參數掃描設定
        sweep_config = None
        
        st.markdown("---")
        st.subheader("📊 參數掃描設定")
        
        sweep_enabled = st.toggle(
            "啟用參數掃描 (Parameter Sweep)",
            value=st.session_state.get("sweep_enabled", False),
            key="sweep_enabled_toggle",
            help="啟用此功能將對選定的參數進行掃描測試，找出最佳參數值"
        )
        st.session_state.sweep_enabled = sweep_enabled
        
        if sweep_enabled:
            st.info("💡 啟用參數掃描時，將執行多次回測，請耐心等待。")
            
            # 參數分類
            sweep_category = st.selectbox(
                "參數分類",
                options=["基礎型態", "雙引擎模式", "因子權重", "演化優化"],
                index=0,
                key="sweep_category_selector"
            )
            
            # 定義各分類的可掃描參數
            category_params = {
                "基礎型態": {
                    "min_depth": "最小杯身深度 (%)",
                    "max_depth": "最大杯身深度 (%)",
                    "min_cup_days": "最小成型天數",
                    "max_cup_days": "最大成型天數",
                    "stop_loss_ratio": "止損比例 (%)",
                    "profit_threshold": "移動止盈啟動閾值 (%)",
                    "trailing_ratio": "移動止盈回調比例 (%)",
                    "score_threshold": "吻合分數閾值",
                    "position_size": "單筆倉位比例 (%)"
                },
                "雙引擎模式": {
                    "adx_trend_threshold": "趨勢判定閾值 (ADX >)",
                    "adx_range_threshold": "震盪判定閾值 (ADX <)",
                    "trend_allocation": "趨勢狀態權重 (0-1)",
                    "range_allocation": "震盪狀態權重 (0-1)",
                    "trend_score_threshold": "趨勢型態分數閾值",
                    "reversion_rsi_oversold": "RSI 超賣閾值"
                },
                "因子權重": {
                    "rsi_weight": "RSI 權重",
                    "volume_weight": "成交量權重",
                    "macd_weight": "MACD 權重",
                    "ema_weight": "均線權重",
                    "bollinger_weight": "布林通道權重",
                    "buy_threshold": "買入分數閾值",
                    "watch_threshold": "觀望分數閾值"
                },
                "演化優化": {
                    "population_size": "種群大小",
                    "max_generations": "最大世代數",
                    "window_size_days": "演化視窗大小 (天)",
                    "step_size_days": "步進大小 (天)",
                    "elitism_rate": "精英保留率 (0-1)",
                    "crossover_rate": "基因交叉率 (0-1)",
                    "mutation_rate": "變異發生率 (0-1)",
                    "tournament_size": "競賽選擇規模",
                    "mutation_strength": "變異強度 (0-1)"
                }
            }
            
            param_options = category_params.get(sweep_category, {})
            
            col1, col2 = st.columns(2)
            with col1:
                sweep_param = st.selectbox(
                    "掃描參數",
                    options=list(param_options.keys()),
                    format_func=lambda x: param_options[x],
                    key="sweep_param_selector"
                )
            
            # 獲取初始值
            is_int = False
            current_value = 0.0
            
            if sweep_category == "基礎型態":
                current_value = getattr(self.default_params, sweep_param)
                if sweep_param in ["min_cup_days", "max_cup_days"]:
                    is_int = True
            elif sweep_category == "雙引擎模式":
                # 從 session state 或管理員獲取
                config = st.session_state.get("dual_engine_config") or DualEngineConfigManager().get_config()
                current_value = getattr(config, sweep_param)
            elif sweep_category == "因子權重":
                # 從 session state 獲取當前 symbol 的 config
                symbol = st.session_state.get("factor_lab_symbol", "AAPL")
                config = st.session_state.get("factor_lab_config") or FactorConfigManager().get_config(symbol)
                
                # 特殊處理巢狀結構
                if sweep_param.endswith("_weight"):
                    indicator = sweep_param.split("_")[0]
                    current_value = getattr(getattr(config, indicator), "weight")
                else:
                    current_value = getattr(config, sweep_param)
            elif sweep_category == "演化優化":
                # 從 session state 獲取
                config = st.session_state.get("evolution_config")
                if config:
                    current_value = getattr(config, sweep_param)
                    if sweep_param in ["population_size", "max_generations", "window_size_days", "step_size_days", "tournament_size"]:
                        is_int = True
                else:
                    # 預設值（對應 EvolutionBacktestConfig 預設）
                    defaults = {
                        "population_size": 50, "max_generations": 15, "window_size_days": 126, 
                        "step_size_days": 21, "elitism_rate": 0.1, "crossover_rate": 0.8,
                        "mutation_rate": 0.02, "tournament_size": 3, "mutation_strength": 0.1
                    }
                    current_value = defaults.get(sweep_param, 0.0)
                    if sweep_param in ["population_size", "max_generations", "window_size_days", "step_size_days", "tournament_size"]:
                        is_int = True            
            # 設定步進與範圍
            if is_int:
                default_start = int(float(current_value) * 0.5)
                default_end = int(float(current_value) * 1.5)
                default_step = 1
            else:
                default_start = float(current_value) * 0.5
                default_end = float(current_value) * 1.5
                default_step = 0.1
                if sweep_param in ["adx_trend_threshold", "adx_range_threshold", "score_threshold", "buy_threshold"]:
                    default_step = 5.0
                elif sweep_param.endswith("_weight"):
                    default_step = 0.2
            
            with col2:
                st.caption(f"當前設定值: {current_value}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                sweep_start = st.number_input(
                    "起始值", value=default_start, step=default_step, format="%.2f" if not is_int else "%d"
                )
            with c2:
                sweep_end = st.number_input(
                    "結束值", value=default_end, step=default_step, format="%.2f" if not is_int else "%d"
                )
            with c3:
                sweep_step = st.number_input(
                    "步進值", value=default_step, step=default_step, min_value=0.01 if not is_int else 1, format="%.2f" if not is_int else "%d"
                )
            
            # 計算預計回測次數
            if sweep_step > 0:
                steps = int((sweep_end - sweep_start) / sweep_step) + 1
                st.caption(f"預計執行回測次數: {steps} 次")
                
                if steps > 20:
                    st.warning("⚠️ 回測次數過多 (>20)，可能會執行很長時間")
            
            sweep_config = {
                "enabled": True,
                "category": sweep_category,
                "param_name": sweep_param,
                "display_name": param_options[sweep_param],
                "start": sweep_start,
                "end": sweep_end,
                "step": sweep_step,
                "is_int": is_int
            }
        
        return StrategyParameters(
            min_depth=min_depth,
            max_depth=max_depth,
            min_cup_days=min_cup_days,
            max_cup_days=max_cup_days,
            stop_loss_ratio=stop_loss_ratio,
            profit_threshold=profit_threshold,
            trailing_ratio=trailing_ratio,
            score_threshold=score_threshold,
            position_size=position_size
        ), sweep_config

    def _run_parameter_sweep(
        self,
        base_params: StrategyParameters,
        sweep_config: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        portfolio_allocations: Optional[List[PortfolioAllocation]] = None
    ) -> pd.DataFrame:
        """執行參數掃描回測"""
        param_name = sweep_config["param_name"]
        start_val = sweep_config["start"]
        end_val = sweep_config["end"]
        step_val = sweep_config["step"]
        is_int = sweep_config["is_int"]
        
        results = []
        
        # 產生參數序列
        # 使用 numpy arange 可能會有浮點數誤差，手動生成
        current_val = start_val
        values = []
        while current_val <= end_val + (step_val * 0.01): # 加上微小緩衝處理浮點數
            values.append(current_val)
            current_val += step_val
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(values)
        
        for i, val in enumerate(values):
            # 更新參數
            if is_int:
                val = int(round(val))
            else:
                val = float(val)
                
            status_text.text(f"正在執行參數掃描: {sweep_config['display_name']} = {val} ({i+1}/{total_steps})")
            progress_bar.progress((i) / total_steps)
            
        # 備份原始配置
        from dataclasses import replace
        original_dual_config = self.backtest_engine.dual_engine_config
        original_evo_config = self.backtest_engine.evolution_config
        original_use_optimizer = self.backtest_engine.use_signal_optimizer
        
        # 因子權重備份 (針對第一個 symbol，通常 sweep 時只選一個標的)
        original_factor_config = None
        target_symbol = symbols[0] if symbols else "AAPL"
        if st.session_state.get('shared_config_manager'):
            original_factor_config = st.session_state.shared_config_manager.get_config(target_symbol)
            
        try:
            for i, val in enumerate(values):
                # 更新參數
                if is_int:
                    val = int(round(val))
                else:
                    val = float(val)
                    
                status_text.text(f"正在執行參數掃描: {sweep_config['display_name']} = {val} ({i+1}/{total_steps})")
                progress_bar.progress((i) / total_steps)
                
                current_params = base_params
                category = sweep_config.get("category", "基礎型態")
                
                # 根據分類處理參數
                if category == "基礎型態":
                    current_params = replace(base_params, **{param_name: val})
                elif category == "雙引擎模式":
                    # 克隆並修改雙引擎配置
                    config = original_dual_config or DualEngineConfig(enabled=True)
                    new_dual_config = replace(config, enabled=True, **{param_name: val})
                    self.backtest_engine.dual_engine_config = new_dual_config
                elif category == "因子權重":
                    # 修改因子權重配置
                    self.backtest_engine.use_signal_optimizer = True
                    manager = st.session_state.shared_config_manager
                    config = original_factor_config or manager.get_default_config(target_symbol)
                    
                    # 處理巢狀權重
                    if param_name.endswith("_weight"):
                        indicator = param_name.split("_")[0]
                        ind_config = getattr(config, indicator)
                        # 建立新的指標配置並賦回
                        from dataclasses import replace as dc_replace
                        new_ind_config = dc_replace(ind_config, weight=val, enabled=True)
                        setattr(config, indicator, new_ind_config)
                    else:
                        config = replace(config, **{param_name: val})
                    
                    manager.save_config(config) # 保存到 manager，engine 會讀取
                elif category == "演化優化":
                    # 克隆並修改演化配置
                    config = original_evo_config or EvolutionBacktestConfig(enabled=True)
                    new_evo_config = replace(config, enabled=True, **{param_name: val})
                    self.backtest_engine.evolution_config = new_evo_config
                
                # 執行回測
                backtest_result = self.backtest_engine.run_backtest(
                    parameters=current_params,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                    progress_callback=None,
                    portfolio_allocations=portfolio_allocations
                )
                
                # 記錄結果
                results.append({
                    "param_value": val,
                    "Total Return (%)": backtest_result.total_return,
                    "Sharpe Ratio": backtest_result.sharpe_ratio,
                    "Win Rate (%)": backtest_result.win_rate,
                    "Max Drawdown (%)": backtest_result.max_drawdown,
                    "Total Trades": backtest_result.total_trades,
                    "Profit Factor": getattr(backtest_result, "profit_factor", 0.0)
                })
        finally:
            # 還原原始配置
            self.backtest_engine.dual_engine_config = original_dual_config
            self.backtest_engine.evolution_config = original_evo_config
            self.backtest_engine.use_signal_optimizer = original_use_optimizer
            if original_factor_config and st.session_state.get('shared_config_manager'):
                st.session_state.shared_config_manager.save_config(original_factor_config)
            
        progress_bar.progress(1.0)
        status_text.text("參數掃描完成！")
        
        return pd.DataFrame(results)

    def _render_sweep_results(self, df: pd.DataFrame, sweep_config: Dict[str, Any]):
        """渲染參數掃描結果"""
        st.subheader(f"📊 參數掃描報告: {sweep_config['display_name']}")
        
        # 確保數據按參數值排序
        df = df.sort_values("param_value")
        
        # 顯示最佳結果
        best_return_idx = df["Total Return (%)"].idxmax()
        best_sharpe_idx = df["Sharpe Ratio"].idxmax()
        
        col1, col2 = st.columns(2)
        with col1:
            best_ret_val = df.iloc[best_return_idx]
            st.metric(
                label=f"最佳回報參數 ({best_ret_val['param_value']})",
                value=f"{best_ret_val['Total Return (%)']:.2f}%"
            )
        with col2:
            best_sharpe_val = df.iloc[best_sharpe_idx]
            st.metric(
                label=f"最佳夏普參數 ({best_sharpe_val['param_value']})",
                value=f"{best_sharpe_val['Sharpe Ratio']:.2f}"
            )
            
        st.divider()
        
        # 繪圖 - 使用明確的 DataFrame 並指定欄位
        st.markdown("##### 參數 vs 績效指標")
        
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["總報酬率", "夏普比率", "勝率 & 回撤"])
        
        # 準備繪圖數據
        # 強制轉換為數值型別，避免因格式問題導致無法繪圖
        plot_df = df.copy()
        try:
            plot_df["param_value"] = pd.to_numeric(plot_df["param_value"], errors='coerce')
            plot_df["Return"] = pd.to_numeric(plot_df["Total Return (%)"], errors='coerce')
            plot_df["Sharpe"] = pd.to_numeric(plot_df["Sharpe Ratio"], errors='coerce')
            plot_df["WinRate"] = pd.to_numeric(plot_df["Win Rate (%)"], errors='coerce')
            plot_df["Drawdown"] = pd.to_numeric(plot_df["Max Drawdown (%)"], errors='coerce')
        except Exception as e:
            st.error(f"數據轉換錯誤: {e}")
        
        # 設置索引為參數值，這是 st.line_chart 最穩定的繪圖方式
        plot_df = plot_df.set_index("param_value").sort_index()

        # Debug 資訊 (排查測試)
        with st.expander("🛠️ 排查測試數據 (Debug Info)", expanded=True):
            st.info("說明: 下方的 'NaN 檢查' 顯示的是數據缺失的數量，顯示 0 表示數據完整（這是好事）。若要查看實際數值，請看 '繪圖數據預覽'。")
            
            c_dbg1, c_dbg2 = st.columns(2)
            with c_dbg1:
                st.write("**NaN 缺失值檢查 (應為 0):**")
                st.write(plot_df.isna().sum())
            with c_dbg2:
                st.write("**繪圖數據預覽 (前 5 筆):**")
                st.dataframe(plot_df.head(), use_container_width=True)

        # 使用 matplotlib 繪圖 (最穩定的方案)
        import matplotlib.pyplot as plt
        
        chart_df = plot_df.reset_index()

        with chart_tab1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(chart_df['param_value'], chart_df['Return'], marker='o', linewidth=2, color='steelblue')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Total Return (%)')
            ax.set_title('Total Return vs Parameter')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
        with chart_tab2:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(chart_df['param_value'], chart_df['Sharpe'], marker='o', linewidth=2, color='orange')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Sharpe Ratio vs Parameter')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
        with chart_tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("勝率 (%)")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(chart_df['param_value'], chart_df['WinRate'], marker='o', linewidth=2, color='green')
                ax.set_xlabel('Parameter Value')
                ax.set_ylabel('Win Rate (%)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            with col_b:
                st.caption("最大回撤 (%)")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(chart_df['param_value'], chart_df['Drawdown'], marker='o', linewidth=2, color='red')
                ax.set_xlabel('Parameter Value')
                ax.set_ylabel('Max Drawdown (%)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
        
        # 表格數據
        st.markdown("##### 詳細數據")
        st.dataframe(df, use_container_width=True)
    
    def render_stock_selection(self) -> Tuple[List[str], Optional[List[PortfolioAllocation]]]:
        """渲染股票選擇區 - 支援多種資產類別與混合投資
        
        Returns:
            Tuple of (symbols, portfolio_allocations)
        """
        st.subheader("📈 標的選擇")
        
        # ============ 美股 ============
        us_tech = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NFLX', 'PYPL', 'SQ']
        us_finance = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'BLK']
        us_consumer = ['WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'PG', 'COST', 'TGT']
        us_health = ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN']
        us_energy = ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'VLO', 'MPC', 'HAL']
        
        # ============ ETF ============
        etf_index = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'IVV', 'VEA', 'VWO', 'EFA']
        etf_sector = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
        etf_theme = ['ARKK', 'ARKG', 'ARKW', 'ARKF', 'SOXX', 'SMH', 'HACK', 'BOTZ', 'ICLN', 'TAN']
        
        # ============ 債券 ETF ============
        etf_bond = ['TLT', 'IEF', 'SHY', 'BND', 'AGG', 'LQD', 'HYG', 'JNK', 'TIP', 'GOVT', 'EMB', 'MUB']
        
        # ============ 商品 ETF ============
        etf_commodity = ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC', 'PDBC', 'PPLT', 'PALL', 'CPER']
        
        # ============ 期貨相關 ETF ============
        etf_futures = ['VXX', 'UVXY', 'SVXY', 'KOLD', 'BOIL', 'UCO', 'SCO']
        
        # ============ 槓桿 ETF ============
        etf_leveraged = ['TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'UPRO', 'SOXL', 'SOXS', 'FNGU', 'LABU', 'LABD']
        
        # ============ 國際市場 ETF ============
        etf_intl = ['EWJ', 'FXI', 'EWZ', 'EWY', 'EWT', 'EWG', 'EWU', 'EWA', 'EWC', 'INDA', 'MCHI', 'KWEB']
        
        # ============ REITs ============
        etf_reit = ['VNQ', 'IYR', 'SCHH', 'RWR', 'XLRE', 'O', 'AMT', 'PLD', 'CCI', 'EQIX']
        
        # ============ 加密貨幣 ============
        crypto = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD', 'LTC-USD']
        crypto_etf = ['BITO', 'BTF', 'GBTC', 'ETHE', 'IBIT', 'FBTC']
        
        # ============ 台股 ============
        tw_stocks = ['2330.TW', '2317.TW', '2454.TW', '2308.TW', '2881.TW', '2882.TW', '2303.TW', '2412.TW', '2886.TW', '1301.TW', '2891.TW', '3711.TW', '2357.TW', '2382.TW', '2395.TW']
        
        # ============ 台股 ETF ============
        # 市值型 ETF
        tw_etf_market = ['0050.TW', '0051.TW', '0052.TW', '0053.TW', '0055.TW', '0056.TW', '0057.TW', '006201.TW', '006203.TW', '006204.TW', '006208.TW']
        # 高股息 ETF
        tw_etf_dividend = ['0056.TW', '00713.TW', '00878.TW', '00900.TW', '00919.TW', '00929.TW', '00934.TW', '00936.TW', '00940.TW']
        # 科技型 ETF
        tw_etf_tech = ['00881.TW', '00891.TW', '00892.TW', '00893.TW', '00895.TW', '00896.TW']
        # 槓桿/反向 ETF
        tw_etf_leveraged = ['00631L.TW', '00632R.TW', '00633L.TW', '00634R.TW', '00637L.TW', '00638R.TW', '00663L.TW', '00664R.TW', '00675L.TW', '00676R.TW']
        # 產業型 ETF
        tw_etf_sector = ['00850.TW', '00851.TW', '00852.TW', '00861.TW', '00876.TW', '00888.TW']
        
        # ============ 台股債券 ETF ============
        # 政府公債 ETF
        tw_bond_gov = ['00679B.TW', '00687B.TW', '00695B.TW', '00696B.TW', '00697B.TW', '00719B.TW', '00720B.TW', '00721B.TW']
        # 投資等級公司債 ETF
        tw_bond_corp = ['00720B.TW', '00724B.TW', '00725B.TW', '00726B.TW', '00727B.TW', '00740B.TW', '00741B.TW', '00751B.TW']
        # 新興市場債 ETF
        tw_bond_em = ['00749B.TW', '00750B.TW', '00761B.TW', '00762B.TW', '00763B.TW']
        # 高收益債 ETF
        tw_bond_hy = ['00710B.TW', '00711B.TW', '00712B.TW', '00714B.TW', '00718B.TW', '00719B.TW']
        
        # 選擇模式：單一市場 vs 混合投資組合
        selection_mode = st.radio(
            "選擇模式",
            options=["單一市場", "混合投資組合"],
            horizontal=True,
            help="混合投資組合可同時選擇不同類型的資產並設定各自的倉位比例"
        )
        
        if selection_mode == "混合投資組合":
            # 混合投資組合模式
            st.info("💡 混合投資組合可同時選擇不同類型的資產，實現多元化配置")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 選擇資產類別**")
                
                # 美股
                with st.expander("🇺🇸 美股", expanded=False):
                    us_selected = st.multiselect(
                        "選擇美股",
                        options=us_tech + us_finance + us_consumer + us_health + us_energy,
                        default=[],
                        key="mixed_us_stocks"
                    )
                
                # ETF
                with st.expander("📈 ETF", expanded=False):
                    etf_selected = st.multiselect(
                        "選擇 ETF",
                        options=etf_index + etf_sector + etf_theme,
                        default=[],
                        key="mixed_etf"
                    )
                
                # 債券
                with st.expander("💵 美國債券", expanded=False):
                    bond_selected = st.multiselect(
                        "選擇美國債券 ETF",
                        options=etf_bond,
                        default=[],
                        key="mixed_bond"
                    )
            
            with col2:
                # 商品
                with st.expander("🥇 商品", expanded=False):
                    commodity_selected = st.multiselect(
                        "選擇商品 ETF",
                        options=etf_commodity,
                        default=[],
                        key="mixed_commodity"
                    )
                
                # 加密貨幣
                with st.expander("₿ 加密貨幣", expanded=False):
                    crypto_selected = st.multiselect(
                        "選擇加密貨幣",
                        options=crypto + crypto_etf,
                        default=[],
                        key="mixed_crypto"
                    )
                
                # 國際市場
                with st.expander("🌍 國際市場", expanded=False):
                    intl_selected = st.multiselect(
                        "選擇國際市場 ETF",
                        options=etf_intl,
                        default=[],
                        key="mixed_intl"
                    )
                
                # 台股
                with st.expander("🇹🇼 台股個股", expanded=False):
                    tw_selected = st.multiselect(
                        "選擇台股個股",
                        options=tw_stocks,
                        default=[],
                        key="mixed_tw"
                    )
                
                # 台股 ETF
                with st.expander("🇹🇼 台股 ETF", expanded=False):
                    tw_etf_type = st.selectbox(
                        "ETF 類型",
                        options=["市值型", "高股息", "科技型", "槓桿/反向", "產業型", "全部"],
                        key="tw_etf_type_mixed"
                    )
                    tw_etf_map = {
                        "市值型": tw_etf_market,
                        "高股息": tw_etf_dividend,
                        "科技型": tw_etf_tech,
                        "槓桿/反向": tw_etf_leveraged,
                        "產業型": tw_etf_sector,
                        "全部": tw_etf_market + tw_etf_dividend + tw_etf_tech + tw_etf_leveraged + tw_etf_sector
                    }
                    tw_etf_selected = st.multiselect(
                        "選擇台股 ETF",
                        options=list(set(tw_etf_map[tw_etf_type])),
                        default=[],
                        key="mixed_tw_etf"
                    )
                
                # 台股債券 ETF
                with st.expander("🇹🇼 台股債券 ETF", expanded=False):
                    tw_bond_type = st.selectbox(
                        "債券類型",
                        options=["政府公債", "投資等級公司債", "新興市場債", "高收益債", "全部"],
                        key="tw_bond_type_mixed"
                    )
                    tw_bond_map = {
                        "政府公債": tw_bond_gov,
                        "投資等級公司債": tw_bond_corp,
                        "新興市場債": tw_bond_em,
                        "高收益債": tw_bond_hy,
                        "全部": tw_bond_gov + tw_bond_corp + tw_bond_em + tw_bond_hy
                    }
                    tw_bond_selected = st.multiselect(
                        "選擇台股債券 ETF",
                        options=list(set(tw_bond_map[tw_bond_type])),
                        default=[],
                        key="mixed_tw_bond"
                    )
            
            # 合併所有選擇
            symbols = (
                us_selected + etf_selected + bond_selected + 
                commodity_selected + crypto_selected + intl_selected + 
                tw_selected + tw_etf_selected + tw_bond_selected
            )
            
            # 自訂輸入
            st.markdown("**✏️ 自訂標的**")
            custom_input = st.text_input(
                "輸入額外代碼（逗號分隔）",
                value="",
                key="mixed_custom",
                help="可輸入上方列表中沒有的標的代碼"
            )
            if custom_input:
                custom_symbols = [s.strip().upper() for s in custom_input.split(',') if s.strip()]
                symbols.extend(custom_symbols)
            
            # 去重
            symbols = list(dict.fromkeys(symbols))
        
        else:
            # 單一市場模式
            market = st.radio(
                "選擇市場類型",
                options=["美股", "ETF", "債券", "商品/期貨", "加密貨幣", "國際市場", "台股", "自訂"],
                horizontal=True
            )
            
            if market == "美股":
                sector = st.selectbox(
                    "選擇板塊",
                    options=["科技股", "金融股", "消費股", "醫療股", "能源股", "全部"]
                )
                
                sector_map = {
                    "科技股": us_tech,
                    "金融股": us_finance,
                    "消費股": us_consumer,
                    "醫療股": us_health,
                    "能源股": us_energy,
                    "全部": us_tech + us_finance + us_consumer + us_health + us_energy
                }
                available = sector_map[sector]
                
                symbols = st.multiselect(
                    "選擇股票",
                    options=available,
                    default=available[:5]
                )
                
            elif market == "ETF":
                etf_type = st.selectbox(
                    "選擇 ETF 類型",
                    options=["指數型", "產業型", "主題型", "槓桿型", "REITs", "全部"]
                )
                
                etf_map = {
                    "指數型": etf_index,
                    "產業型": etf_sector,
                    "主題型": etf_theme,
                    "槓桿型": etf_leveraged,
                    "REITs": etf_reit,
                    "全部": etf_index + etf_sector + etf_theme + etf_leveraged + etf_reit
                }
                available = etf_map[etf_type]
                
                if etf_type == "槓桿型":
                    st.warning("⚠️ 槓桿 ETF 風險較高，適合短線交易，不建議長期持有")
                
                symbols = st.multiselect(
                    "選擇 ETF",
                    options=available,
                    default=available[:5]
                )
                
            elif market == "債券":
                st.info("💡 債券 ETF 波動較小，適合穩健型投資組合")
                symbols = st.multiselect(
                    "選擇債券 ETF",
                    options=etf_bond,
                    default=etf_bond[:5]
                )
                
            elif market == "商品/期貨":
                commodity_type = st.selectbox(
                    "選擇類型",
                    options=["商品 ETF", "期貨 ETF", "全部"]
                )
                
                if commodity_type == "商品 ETF":
                    available = etf_commodity
                    st.info("💡 商品 ETF 追蹤黃金、白銀、原油等實物商品價格")
                elif commodity_type == "期貨 ETF":
                    available = etf_futures
                    st.warning("⚠️ 期貨 ETF 有轉倉成本，長期持有可能有損耗")
                else:
                    available = etf_commodity + etf_futures
                
                symbols = st.multiselect(
                    "選擇標的",
                    options=available,
                    default=available[:5]
                )
                
            elif market == "加密貨幣":
                crypto_type = st.selectbox(
                    "選擇類型",
                    options=["現貨 (直接追蹤)", "加密貨幣 ETF"]
                )
                
                if crypto_type == "現貨 (直接追蹤)":
                    available = crypto
                    st.info("💡 加密貨幣 24 小時交易，波動較大")
                else:
                    available = crypto_etf
                    st.info("💡 加密貨幣 ETF 在傳統交易所交易，有交易時間限制")
                
                symbols = st.multiselect(
                    "選擇加密貨幣",
                    options=available,
                    default=available[:5]
                )
                
            elif market == "國際市場":
                st.info("💡 國際市場 ETF 可分散地區風險")
                symbols = st.multiselect(
                    "選擇國際市場 ETF",
                    options=etf_intl,
                    default=etf_intl[:5],
                    help="EWJ=日本, FXI=中國, EWZ=巴西, EWY=韓國, EWT=台灣, INDA=印度"
                )
                
            elif market == "台股":
                tw_type = st.selectbox(
                    "選擇類型",
                    options=["個股", "ETF", "債券 ETF"]
                )
                
                if tw_type == "個股":
                    symbols = st.multiselect(
                        "選擇台股個股",
                        options=tw_stocks,
                        default=tw_stocks[:5]
                    )
                elif tw_type == "ETF":
                    tw_etf_subtype = st.selectbox(
                        "ETF 類型",
                        options=["市值型", "高股息", "科技型", "槓桿/反向", "產業型", "全部"]
                    )
                    tw_etf_options = {
                        "市值型": tw_etf_market,
                        "高股息": tw_etf_dividend,
                        "科技型": tw_etf_tech,
                        "槓桿/反向": tw_etf_leveraged,
                        "產業型": tw_etf_sector,
                        "全部": list(set(tw_etf_market + tw_etf_dividend + tw_etf_tech + tw_etf_leveraged + tw_etf_sector))
                    }
                    available = tw_etf_options[tw_etf_subtype]
                    
                    if tw_etf_subtype == "槓桿/反向":
                        st.warning("⚠️ 槓桿/反向 ETF 風險較高，適合短線交易")
                    
                    symbols = st.multiselect(
                        "選擇台股 ETF",
                        options=available,
                        default=available[:5] if len(available) >= 5 else available
                    )
                else:  # 債券 ETF
                    tw_bond_subtype = st.selectbox(
                        "債券類型",
                        options=["政府公債", "投資等級公司債", "新興市場債", "高收益債", "全部"]
                    )
                    tw_bond_options = {
                        "政府公債": tw_bond_gov,
                        "投資等級公司債": tw_bond_corp,
                        "新興市場債": tw_bond_em,
                        "高收益債": tw_bond_hy,
                        "全部": list(set(tw_bond_gov + tw_bond_corp + tw_bond_em + tw_bond_hy))
                    }
                    available = tw_bond_options[tw_bond_subtype]
                    
                    st.info("💡 台股債券 ETF 波動較小，適合穩健型投資組合")
                    
                    symbols = st.multiselect(
                        "選擇台股債券 ETF",
                        options=available,
                        default=available[:5] if len(available) >= 5 else available
                    )
            else:
                st.markdown("""
                **代碼格式說明：**
                - 美股：直接輸入代碼 (如 `AAPL`, `MSFT`)
                - 台股個股：股票代碼加 `.TW` 後綴 (如 `2330.TW`, `2317.TW`)
                - 台股 ETF：ETF 代碼加 `.TW` 後綴 (如 `0050.TW`, `0056.TW`, `00878.TW`)
                - 台股債券 ETF：債券 ETF 代碼加 `.TW` 後綴 (如 `00679B.TW`, `00720B.TW`)
                - 加密貨幣：加 `-USD` 後綴 (如 `BTC-USD`, `ETH-USD`)
                - 美股 ETF：直接輸入代碼 (如 `SPY`, `QQQ`)
                """)
                custom_input = st.text_input(
                    "輸入代碼（逗號分隔）",
                    value="AAPL, 2330.TW, 0050.TW, 00878.TW, SPY"
                )
                symbols = [s.strip().upper() for s in custom_input.split(',') if s.strip()]
        
        # 顯示已選數量和資產類別
        if symbols:
            st.caption(f"已選擇 {len(symbols)} 個標的")
        
        # 倉位比例設定
        portfolio_allocations = None
        if symbols:
            portfolio_allocations = self._render_portfolio_weight_settings(symbols)
        
        return symbols, portfolio_allocations
    
    def _render_portfolio_weight_settings(
        self, 
        symbols: List[str]
    ) -> Optional[List[PortfolioAllocation]]:
        """渲染投資組合倉位比例設定
        
        Args:
            symbols: 已選擇的標的代碼列表
            
        Returns:
            投資組合配置列表，如果未啟用自訂權重則返回 None
        """
        # 資產類別對應表
        asset_class_map = self._get_asset_class_map()
        
        st.markdown("---")
        st.markdown("**🎯 倉位比例配置**")
        
        use_custom_weights = st.checkbox(
            "自訂各標的倉位比例",
            value=st.session_state.get("use_custom_weights", False),
            help="啟用後可為每個標的設定不同的倉位權重，否則使用等權重"
        )
        st.session_state.use_custom_weights = use_custom_weights
        
        if not use_custom_weights:
            # 等權重模式
            equal_weight = 100.0 / len(symbols)
            st.info(f"💡 使用等權重配置：每個標的 {equal_weight:.2f}%")
            return [
                PortfolioAllocation(
                    symbol=s,
                    weight=equal_weight,
                    asset_class=asset_class_map.get(s, "自訂")
                )
                for s in symbols
            ]
        
        # 自訂權重模式
        # 檢查是否有快速配置觸發
        quick_config_key = "quick_config_trigger"
        quick_config_value = st.session_state.get(quick_config_key, None)
        
        # 在 widget 創建前處理快速配置
        if quick_config_value == "等權重":
            equal_weight = 100.0 / len(symbols)
            for symbol in symbols:
                st.session_state[f"weight_{symbol}"] = equal_weight
            st.session_state[quick_config_key] = None
        elif quick_config_value == "市值加權（模擬）":
            weights = self._simulate_market_cap_weights(symbols)
            for symbol, weight in zip(symbols, weights):
                st.session_state[f"weight_{symbol}"] = weight
            st.session_state[quick_config_key] = None
        elif quick_config_value == "風險平價（模擬）":
            weights = self._simulate_risk_parity_weights(symbols)
            for symbol, weight in zip(symbols, weights):
                st.session_state[f"weight_{symbol}"] = weight
            st.session_state[quick_config_key] = None
        
        # 檢查是否需要正規化（在 widget 創建前處理）
        normalize_key = "normalize_weights_trigger"
        if st.session_state.get(normalize_key, False):
            current_total = 0.0
            for symbol in symbols:
                key = f"weight_{symbol}"
                current_total += st.session_state.get(key, 100.0 / len(symbols))
            
            if current_total > 0:
                for symbol in symbols:
                    key = f"weight_{symbol}"
                    old_val = st.session_state.get(key, 100.0 / len(symbols))
                    st.session_state[key] = old_val / current_total * 100
            
            st.session_state[normalize_key] = False
        
        # 快速配置按鈕
        st.markdown("**快速配置：**")
        btn_cols = st.columns(4)
        with btn_cols[0]:
            if st.button("等權重", key="btn_equal"):
                st.session_state[quick_config_key] = "等權重"
                st.rerun()
        with btn_cols[1]:
            if st.button("市值加權", key="btn_mcap"):
                st.session_state[quick_config_key] = "市值加權（模擬）"
                st.rerun()
        with btn_cols[2]:
            if st.button("風險平價", key="btn_rp"):
                st.session_state[quick_config_key] = "風險平價（模擬）"
                st.rerun()
        
        allocations = []
        
        # 手動調整權重
        st.markdown("**調整各標的權重：**")
        
        # 使用 columns 排列滑桿
        num_cols = min(3, len(symbols))
        cols = st.columns(num_cols)
        
        total_weight = 0.0
        
        for i, symbol in enumerate(symbols):
            col_idx = i % num_cols
            with cols[col_idx]:
                key = f"weight_{symbol}"
                default_weight = st.session_state.get(key, 100.0 / len(symbols))
                
                weight = st.slider(
                    f"{symbol}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_weight),
                    step=1.0,
                    key=key,
                    help=f"資產類別: {asset_class_map.get(symbol, '自訂')}"
                )
                
                allocations.append(PortfolioAllocation(
                    symbol=symbol,
                    weight=weight,
                    asset_class=asset_class_map.get(symbol, "自訂")
                ))
                total_weight += weight
        
        # 顯示總權重並提供正規化選項
        if abs(total_weight - 100.0) > 0.01:
            st.warning(f"⚠️ 總權重為 {total_weight:.1f}%，建議調整為 100%")
            
            if st.button("🔄 正規化權重至 100%", key="normalize_btn"):
                st.session_state[normalize_key] = True
                st.rerun()
        else:
            st.success(f"✅ 總權重: {total_weight:.1f}%")
        
        # 顯示資產配置摘要
        self._render_allocation_summary(allocations, asset_class_map)
        
        return allocations
    
    def _get_asset_class_map(self) -> Dict[str, str]:
        """取得資產類別對應表"""
        us_tech = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NFLX', 'PYPL', 'SQ']
        us_finance = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'BLK']
        us_consumer = ['WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'PG', 'COST', 'TGT']
        us_health = ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN']
        us_energy = ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'VLO', 'MPC', 'HAL']
        etf_index = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'IVV', 'VEA', 'VWO', 'EFA']
        etf_sector = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
        etf_theme = ['ARKK', 'ARKG', 'ARKW', 'ARKF', 'SOXX', 'SMH', 'HACK', 'BOTZ', 'ICLN', 'TAN']
        etf_bond = ['TLT', 'IEF', 'SHY', 'BND', 'AGG', 'LQD', 'HYG', 'JNK', 'TIP', 'GOVT', 'EMB', 'MUB']
        etf_commodity = ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC', 'PDBC', 'PPLT', 'PALL', 'CPER']
        etf_futures = ['VXX', 'UVXY', 'SVXY', 'KOLD', 'BOIL', 'UCO', 'SCO']
        etf_leveraged = ['TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'UPRO', 'SOXL', 'SOXS', 'FNGU', 'LABU', 'LABD']
        etf_intl = ['EWJ', 'FXI', 'EWZ', 'EWY', 'EWT', 'EWG', 'EWU', 'EWA', 'EWC', 'INDA', 'MCHI', 'KWEB']
        etf_reit = ['VNQ', 'IYR', 'SCHH', 'RWR', 'XLRE', 'O', 'AMT', 'PLD', 'CCI', 'EQIX']
        crypto = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD', 'LTC-USD']
        crypto_etf = ['BITO', 'BTF', 'GBTC', 'ETHE', 'IBIT', 'FBTC']
        tw_stocks = ['2330.TW', '2317.TW', '2454.TW', '2308.TW', '2881.TW', '2882.TW', '2303.TW', '2412.TW', '2886.TW', '1301.TW', '2891.TW', '3711.TW', '2357.TW', '2382.TW', '2395.TW']
        
        # 台股 ETF
        tw_etf_market = ['0050.TW', '0051.TW', '0052.TW', '0053.TW', '0055.TW', '0056.TW', '0057.TW', '006201.TW', '006203.TW', '006204.TW', '006208.TW']
        tw_etf_dividend = ['0056.TW', '00713.TW', '00878.TW', '00900.TW', '00919.TW', '00929.TW', '00934.TW', '00936.TW', '00940.TW']
        tw_etf_tech = ['00881.TW', '00891.TW', '00892.TW', '00893.TW', '00895.TW', '00896.TW']
        tw_etf_leveraged = ['00631L.TW', '00632R.TW', '00633L.TW', '00634R.TW', '00637L.TW', '00638R.TW', '00663L.TW', '00664R.TW', '00675L.TW', '00676R.TW']
        tw_etf_sector = ['00850.TW', '00851.TW', '00852.TW', '00861.TW', '00876.TW', '00888.TW']
        
        # 台股債券 ETF
        tw_bond_gov = ['00679B.TW', '00687B.TW', '00695B.TW', '00696B.TW', '00697B.TW', '00719B.TW', '00720B.TW', '00721B.TW']
        tw_bond_corp = ['00720B.TW', '00724B.TW', '00725B.TW', '00726B.TW', '00727B.TW', '00740B.TW', '00741B.TW', '00751B.TW']
        tw_bond_em = ['00749B.TW', '00750B.TW', '00761B.TW', '00762B.TW', '00763B.TW']
        tw_bond_hy = ['00710B.TW', '00711B.TW', '00712B.TW', '00714B.TW', '00718B.TW', '00719B.TW']
        
        return {
            **{s: "美股-科技" for s in us_tech},
            **{s: "美股-金融" for s in us_finance},
            **{s: "美股-消費" for s in us_consumer},
            **{s: "美股-醫療" for s in us_health},
            **{s: "美股-能源" for s in us_energy},
            **{s: "ETF-指數" for s in etf_index},
            **{s: "ETF-產業" for s in etf_sector},
            **{s: "ETF-主題" for s in etf_theme},
            **{s: "ETF-槓桿" for s in etf_leveraged},
            **{s: "ETF-REITs" for s in etf_reit},
            **{s: "美國債券" for s in etf_bond},
            **{s: "商品" for s in etf_commodity},
            **{s: "期貨" for s in etf_futures},
            **{s: "加密貨幣" for s in crypto},
            **{s: "加密貨幣ETF" for s in crypto_etf},
            **{s: "國際市場" for s in etf_intl},
            **{s: "台股個股" for s in tw_stocks},
            **{s: "台股ETF-市值型" for s in tw_etf_market},
            **{s: "台股ETF-高股息" for s in tw_etf_dividend},
            **{s: "台股ETF-科技型" for s in tw_etf_tech},
            **{s: "台股ETF-槓桿/反向" for s in tw_etf_leveraged},
            **{s: "台股ETF-產業型" for s in tw_etf_sector},
            **{s: "台股債券-政府公債" for s in tw_bond_gov},
            **{s: "台股債券-公司債" for s in tw_bond_corp},
            **{s: "台股債券-新興市場" for s in tw_bond_em},
            **{s: "台股債券-高收益" for s in tw_bond_hy},
        }
    
    def _simulate_market_cap_weights(self, symbols: List[str]) -> List[float]:
        """模擬市值加權權重"""
        large_caps = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'SPY', 'QQQ', 'BTC-USD', '2330.TW', '0050.TW'}
        mid_caps = {'TSLA', 'AMD', 'NFLX', 'JPM', 'V', 'MA', 'ETH-USD', 'VOO', 'IVV', '2317.TW', '2454.TW', '00878.TW'}
        
        weights = []
        for symbol in symbols:
            if symbol in large_caps:
                weights.append(3.0)
            elif symbol in mid_caps:
                weights.append(2.0)
            else:
                weights.append(1.0)
        
        total = sum(weights)
        return [w / total * 100 for w in weights]
    
    def _simulate_risk_parity_weights(self, symbols: List[str]) -> List[float]:
        """模擬風險平價權重"""
        # 低波動資產（債券類）
        low_vol = {'TLT', 'IEF', 'SHY', 'BND', 'AGG', 'GLD', 'SPY', 'VOO', 
                   '00679B.TW', '00687B.TW', '00720B.TW', '00724B.TW', '0050.TW'}
        # 高波動資產
        high_vol = {'TQQQ', 'SQQQ', 'BTC-USD', 'ETH-USD', 'TSLA', 'NVDA', 'ARKK', 'SOXL',
                    '00631L.TW', '00632R.TW', '00637L.TW', '00638R.TW'}
        
        weights = []
        for symbol in symbols:
            if symbol in low_vol:
                weights.append(3.0)
            elif symbol in high_vol:
                weights.append(0.5)
            else:
                weights.append(1.5)
        
        total = sum(weights)
        return [w / total * 100 for w in weights]
    
    def _render_allocation_summary(
        self, 
        allocations: List[PortfolioAllocation],
        asset_class_map: Dict[str, str]
    ) -> None:
        """渲染資產配置摘要"""
        # 按資產類別分組
        class_weights: Dict[str, float] = {}
        for alloc in allocations:
            asset_class = alloc.asset_class
            if asset_class not in class_weights:
                class_weights[asset_class] = 0.0
            class_weights[asset_class] += alloc.weight
        
        with st.expander("📊 資產配置摘要", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**按資產類別：**")
                for asset_class, weight in sorted(class_weights.items(), key=lambda x: -x[1]):
                    st.write(f"- {asset_class}: {weight:.1f}%")
            
            with col2:
                st.markdown("**各標的權重：**")
                for alloc in sorted(allocations, key=lambda x: -x.weight)[:10]:
                    st.write(f"- {alloc.symbol}: {alloc.weight:.1f}%")
                if len(allocations) > 10:
                    st.write(f"... 及其他 {len(allocations) - 10} 個標的")
    
    def render_backtest_controls(self) -> Optional[tuple]:
        """渲染回測控制區"""
        # 標題與模擬選項在同一排
        header_col, sim_col = st.columns([1, 2])
        
        with header_col:
            st.subheader("📅 回測設定")
            
        with sim_col:
            # 模擬選項
            c1, c2 = st.columns(2)
            with c1:
                progressive_mode = st.checkbox(
                    "🎬 逐日模擬",
                    value=st.session_state.get("progressive_mode", False),
                    help="逐日顯示交易過程",
                    key="progressive_mode_checkbox"
                )
                st.session_state.progressive_mode = progressive_mode
            with c2:
                live_trading_mode = st.checkbox(
                    "🔴 實時模擬",
                    value=st.session_state.get("live_trading_mode", False),
                    help="持續抓取即時數據",
                    key="live_trading_mode_check"
                )
                st.session_state.live_trading_mode = live_trading_mode
        
        # 主要設定區
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            default_start = date.today() - timedelta(days=365*20)  # 20 年 (Requirement 3)
            start_date = st.date_input(
                "起始日期", 
                value=default_start,
                min_value=date(2000, 1, 1),
                max_value=date.today()
            )
            
            # 逐日模擬速度
            if progressive_mode and not live_trading_mode:
                sim_speed = st.select_slider(
                    "模擬速度",
                    options=["慢速", "正常", "快速", "最快"],
                    value="正常",
                    key="sim_speed_slider"
                )
                speed_map = {"慢速": 0.5, "正常": 0.2, "快速": 0.05, "最快": 0.0}
                st.session_state.sim_speed = speed_map.get(sim_speed, 0.2)
        
        with col2:
            if live_trading_mode:
                update_interval = st.selectbox(
                    "即時更新頻率",
                    options=["1分鐘", "5分鐘", "15分鐘", "60分鐘"],
                    index=1,
                    key="live_update_interval"
                )
                interval_map = {"1分鐘": 60, "5分鐘": 300, "15分鐘": 900, "60分鐘": 3600}
                st.session_state.live_interval = interval_map.get(update_interval, 300)
                end_date = date.today() # 實時模式固定結束日期
            else:
                end_date = st.date_input(
                    "結束日期", 
                    value=date.today(),
                    min_value=date(2000, 1, 1),
                    max_value=date.today()
                )
        
        with col3:
            st.write("")
            st.write("")
            if live_trading_mode:
                if st.session_state.get("live_sim_active", False):
                    if st.button("🛑 停止實時模擬", type="secondary", use_container_width=True):
                        st.session_state.live_sim_active = False
                        st.rerun()
                    return "LIVE_SIMULATION"
                else:
                    if st.button("▶️ 啟動實時模擬", type="primary", use_container_width=True):
                        st.session_state.live_sim_active = True
                        st.warning("⚠️ 實時模擬將持續運行，請勿關閉網頁")
                        st.rerun()
                        return "LIVE_SIMULATION"
            else:
                if st.button("🚀 執行回測", type="primary", use_container_width=True):
                    if start_date >= end_date:
                        st.error("❌ 起始日期必須早於結束日期")
                        return None
                    return (
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.min.time())
                    )
        
        return None
    
    def render_backtest_results(self, result: EnhancedBacktestResult) -> None:
        """渲染回測結果 - 使用頁籤分區"""
        st.subheader("📊 回測結果")
        
        # 關鍵指標摘要（始終顯示）
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("勝率", f"{result.win_rate:.1f}%",
                     delta=f"{result.winning_trades}勝 / {result.losing_trades}負")
        with col2:
            st.metric("總報酬率", f"{result.total_return:+.2f}%",
                     delta=f"共 {result.total_trades} 筆交易")
        with col3:
            st.metric("最大回撤", f"{result.max_drawdown:.2f}%")
        with col4:
            st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
        
        st.divider()
        
        # 偵測啟用的策略
        enabled_strategies = self._detect_enabled_strategies(result)
        
        # 根據是否有比較報告決定頁籤結構
        has_comparison = result.dual_engine_report and result.dual_engine_report.comparison_report
        
        if has_comparison:
            # 有比較報告時，使用頁籤分區
            tab_titles = ["📊 差異比較", "📈 資金曲線", "🎯 分策略績效", "📋 交易明細"]
            tabs = st.tabs(tab_titles)
            
            # 頁籤 1: 差異比較
            with tabs[0]:
                self._render_comparison_report(result.dual_engine_report.comparison_report)
            
            # 頁籤 2: 資金曲線
            with tabs[1]:
                self._render_equity_curve(result)
            
            # 頁籤 3: 分策略績效
            with tabs[2]:
                self._render_strategy_performance_tabs(result.dual_engine_report)
            
            # 頁籤 4: 交易明細
            with tabs[3]:
                self._render_trade_list(result.trades)
        else:
            # 無比較報告時，使用簡化的頁籤
            if result.dual_engine_report:
                tab_titles = ["📈 資金曲線", "🎯 分策略績效", "📋 交易明細"]
                tabs = st.tabs(tab_titles)
                
                with tabs[0]:
                    self._render_equity_curve(result)
                
                with tabs[1]:
                    self._render_strategy_performance_tabs(result.dual_engine_report)
                
                with tabs[2]:
                    self._render_trade_list(result.trades)
            else:
                # 無雙引擎報告時，使用最簡化的頁籤
                tab_titles = ["📈 資金曲線", "📋 交易明細"]
                tabs = st.tabs(tab_titles)
                
                with tabs[0]:
                    self._render_equity_curve(result)
                
                with tabs[1]:
                    self._render_trade_list(result.trades)
    
    def _detect_enabled_strategies(self, result: EnhancedBacktestResult) -> Dict[str, bool]:
        """偵測啟用的策略類型
        
        Returns:
            包含各策略啟用狀態的字典
        """
        enabled = {
            'dual_engine': False,
            'factor_weight': False,
            'evolution': False,
            'pattern': True,  # 型態策略始終啟用
        }
        
        # 檢查雙引擎策略
        if result.dual_engine_report:
            trend_trades = result.dual_engine_report.trend_performance.total_trades
            reversion_trades = result.dual_engine_report.reversion_performance.total_trades
            if trend_trades > 0 or reversion_trades > 0:
                enabled['dual_engine'] = True
        
        # 檢查因子權重優化
        for trade in result.trades:
            if trade.optimized_signal_details:
                enabled['factor_weight'] = True
                break
        
        # 檢查演化優化
        if result.evolution_history:
            enabled['evolution'] = True
        
        return enabled
    
    def _render_strategy_explanation(
        self,
        enabled_strategies: Dict[str, bool],
        result: EnhancedBacktestResult
    ) -> None:
        """渲染策略類型說明區塊
        
        Requirements: Req 1, 2, 3, 4 from backtest-strategy-comparison
        """
        st.markdown("### 📖 策略類型說明")
        
        # 顯示當前啟用的策略
        st.markdown("**🔧 當前啟用的策略**")
        active_badges = []
        if enabled_strategies.get('pattern'):
            active_badges.append("🔷 型態識別")
        if enabled_strategies.get('dual_engine'):
            active_badges.append("🎛️ 雙引擎策略")
        if enabled_strategies.get('factor_weight'):
            active_badges.append("⚖️ 因子權重優化")
        if enabled_strategies.get('evolution'):
            active_badges.append("🧬 演化優化")
        
        st.success(" + ".join(active_badges) if active_badges else "純型態策略")
        
        st.divider()
        
        # 策略差異比較表
        st.markdown("**📊 策略方法差異比較表**")
        
        comparison_data = [
            {
                "策略方法": "🔷 型態識別 (Pattern Recognition)",
                "運作原理": "識別杯柄型態等技術圖形，在突破時進場",
                "適用場景": "趨勢明確、型態清晰的市場",
                "關鍵參數": "杯身深度、成型天數、吻合分數門檻",
                "啟用狀態": "✅ 啟用" if enabled_strategies.get('pattern') else "❌ 未啟用"
            },
            {
                "策略方法": "🎛️ 雙引擎策略 (Dual Engine)",
                "運作原理": "根據 ADX 判斷市場狀態，趨勢市場用趨勢追蹤，震盪市場用均值回歸",
                "適用場景": "市場狀態多變、需要自適應的環境",
                "關鍵參數": "ADX 趨勢閾值、震盪閾值、資金配置比例",
                "啟用狀態": "✅ 啟用" if enabled_strategies.get('dual_engine') else "❌ 未啟用"
            },
            {
                "策略方法": "⚖️ 因子權重優化 (Factor Weight)",
                "運作原理": "調整 RSI、MACD、成交量等技術指標的權重，優化訊號品質",
                "適用場景": "需要精細調整訊號過濾的策略",
                "關鍵參數": "各指標權重、買入閾值、觀望閾值",
                "啟用狀態": "✅ 啟用" if enabled_strategies.get('factor_weight') else "❌ 未啟用"
            },
            {
                "策略方法": "🧬 演化優化 (Evolutionary)",
                "運作原理": "使用遺傳演算法在多維度參數空間中尋找全域最佳解",
                "適用場景": "參數眾多、需要自動尋優的複雜策略",
                "關鍵參數": "種群大小、世代數、適應度目標函數",
                "啟用狀態": "✅ 啟用" if enabled_strategies.get('evolution') else "❌ 未啟用"
            },
        ]
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 各策略詳細說明（可展開）
        st.markdown("**📚 各策略詳細說明**")
        
        # 型態識別說明
        with st.expander("🔷 型態識別策略", expanded=enabled_strategies.get('pattern', False)):
            st.markdown("""
            **運作原理**
            
            型態識別策略透過數學方法識別股價圖表中的經典技術型態（如杯柄型態），
            並在型態完成、價格突破關鍵壓力位時產生買入訊號。
            
            **核心邏輯**
            1. 使用極值點偵測找出價格的高低點
            2. 透過曲線擬合驗證杯身的 U 型結構
            3. 識別杯柄的回調與整理
            4. 計算型態吻合分數（擬合度、對稱性、成交量、深度）
            5. 當分數超過門檻且價格突破壓力位時進場
            
            **優點**
            - 基於經典技術分析理論，邏輯清晰
            - 進場點明確（突破壓力位）
            - 有明確的止損位（杯柄低點）
            
            **限制**
            - 需要足夠的歷史數據形成型態
            - 在震盪市場中可能產生假突破
            - 型態識別有一定的主觀性
            """)
        
        # 雙引擎策略說明
        with st.expander("🎛️ 雙引擎策略", expanded=enabled_strategies.get('dual_engine', False)):
            st.markdown("""
            **運作原理**
            
            雙引擎策略根據市場狀態自動切換交易策略：
            - **趨勢市場** (ADX > 趨勢閾值)：使用趨勢追蹤策略，順勢而為
            - **震盪市場** (ADX < 震盪閾值)：使用均值回歸策略，高拋低吸
            - **混沌市場** (介於兩者之間)：減少交易或觀望
            
            **核心邏輯**
            1. 計算 ADX (Average Directional Index) 判斷趨勢強度
            2. 根據 ADX 值分類市場狀態
            3. 趨勢市場：追蹤突破訊號，設定移動止盈
            4. 震盪市場：在支撐位買入，壓力位賣出
            5. 動態調整各策略的資金配置比例
            
            **優點**
            - 自適應市場狀態，減少逆勢交易
            - 在不同市場環境都有對應策略
            - 可分別優化各策略參數
            
            **限制**
            - ADX 有滯後性，狀態切換可能不及時
            - 需要更多參數調整
            - 策略切換時可能產生額外成本
            """)
            
            # 如果啟用，顯示當前配置
            if enabled_strategies.get('dual_engine') and result.dual_engine_report:
                st.markdown("**當前績效歸因**")
                report = result.dual_engine_report
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "趨勢策略交易數",
                        report.trend_performance.total_trades,
                        delta=f"勝率 {report.trend_performance.win_rate:.1f}%"
                    )
                with col2:
                    st.metric(
                        "均值回歸交易數",
                        report.reversion_performance.total_trades,
                        delta=f"勝率 {report.reversion_performance.win_rate:.1f}%"
                    )
        
        # 因子權重優化說明
        with st.expander("⚖️ 因子權重優化", expanded=enabled_strategies.get('factor_weight', False)):
            st.markdown("""
            **運作原理**
            
            因子權重優化透過調整各技術指標的權重，來優化訊號的品質和可靠性。
            每個指標根據其當前狀態對最終分數產生正面或負面的影響。
            
            **支援的技術指標**
            - **RSI (相對強弱指標)**：判斷超買超賣狀態
            - **MACD**：判斷動能和趨勢方向
            - **成交量**：確認價格變動的有效性
            - **EMA (指數移動平均)**：判斷價格與均線的關係
            - **布林通道**：判斷波動率和價格位置
            
            **核心邏輯**
            1. 計算型態基礎分數
            2. 根據各指標狀態計算調整分數
            3. 加權彙總得到最終分數
            4. 根據最終分數決定訊號強度（強力買入/觀望/跳過）
            
            **優點**
            - 多維度確認訊號，減少假訊號
            - 權重可根據市場特性調整
            - 提供詳細的訊號分解說明
            
            **限制**
            - 權重設定需要經驗或優化
            - 過多指標可能導致訊號過少
            - 各指標可能產生矛盾訊號
            """)
            
            # 如果啟用，顯示因子權重分布
            if enabled_strategies.get('factor_weight'):
                st.markdown("**因子權重對績效的影響**")
                st.info("💡 在「交易明細」頁籤中可查看每筆交易的因子權重評分詳情")
        
        # 演化優化說明
        with st.expander("🧬 演化優化", expanded=enabled_strategies.get('evolution', False)):
            st.markdown("""
            **運作原理**
            
            演化優化使用遺傳演算法 (Genetic Algorithm) 在多維度參數空間中
            自動搜索最佳參數組合，模擬生物演化的過程。
            
            **演化流程**
            1. **初始化種群**：隨機生成多組參數（個體）
            2. **適應度評估**：用歷史數據回測每個個體的績效
            3. **選擇**：保留績效較好的個體
            4. **交叉**：將優秀個體的參數混合產生後代
            5. **突變**：隨機微調部分參數，增加多樣性
            6. **迭代**：重複步驟 2-5 直到收斂或達到最大世代數
            
            **基因組結構**
            - **Segment A (雙引擎控制)**：趨勢閾值、震盪閾值、資金配置
            - **Segment B (因子權重)**：RSI、MACD、成交量等指標權重
            - **Segment C (微觀指標)**：RSI 週期、超買超賣線等細節參數
            
            **適應度目標函數**
            - 夏普比率 (Sharpe Ratio)：風險調整後收益
            - 索提諾比率 (Sortino Ratio)：下行風險調整後收益
            - 淨利潤 (Net Profit)：最大化總收益
            - 最小化回撤 (Min Max Drawdown)：防禦型策略
            
            **優點**
            - 自動尋找全域最優解
            - 可同時優化多個參數
            - 避免人工調參的主觀性
            
            **限制**
            - 計算成本較高
            - 可能過度擬合歷史數據
            - 需要足夠的歷史數據進行驗證
            """)
            
            # 如果啟用，顯示演化歷史
            if enabled_strategies.get('evolution') and result.evolution_history:
                st.markdown("**演化優化歷史**")
                history_df = pd.DataFrame(result.evolution_history)
                if 'generation' in history_df.columns and 'best_fitness' in history_df.columns:
                    st.line_chart(history_df.set_index('generation')['best_fitness'])
        
        st.divider()
        
        # 策略組合效果說明
        self._render_strategy_combination_effect(enabled_strategies, result)
    
    def _render_strategy_combination_effect(
        self,
        enabled_strategies: Dict[str, bool],
        result: EnhancedBacktestResult
    ) -> None:
        """渲染策略組合效果說明
        
        Requirements: Req 4 from backtest-strategy-comparison
        """
        st.markdown("**🔗 策略組合效果**")
        
        # 計算啟用的策略數量
        active_count = sum([
            enabled_strategies.get('dual_engine', False),
            enabled_strategies.get('factor_weight', False),
            enabled_strategies.get('evolution', False),
        ])
        
        if active_count == 0:
            st.info("""
            **純型態策略模式**
            
            目前僅使用型態識別策略，適合：
            - 初學者了解型態交易的基本邏輯
            - 作為其他策略的基準比較
            - 市場趨勢明確時的簡單策略
            
            💡 建議：可嘗試啟用「雙引擎策略」來適應不同市場狀態
            """)
        
        elif active_count == 1:
            if enabled_strategies.get('dual_engine'):
                st.info("""
                **型態 + 雙引擎模式**
                
                結合型態識別與市場狀態判斷，策略會：
                - 在趨勢市場中追蹤型態突破
                - 在震盪市場中尋找均值回歸機會
                - 在混沌市場中減少交易
                
                💡 建議：可進一步啟用「因子權重優化」來提升訊號品質
                """)
            elif enabled_strategies.get('factor_weight'):
                st.info("""
                **型態 + 因子權重模式**
                
                使用多維度技術指標確認型態訊號，策略會：
                - 根據 RSI、MACD 等指標調整訊號分數
                - 過濾掉指標不支持的假訊號
                - 提供更詳細的進場理由
                
                💡 建議：可進一步啟用「雙引擎策略」來適應市場狀態變化
                """)
            elif enabled_strategies.get('evolution'):
                st.info("""
                **型態 + 演化優化模式**
                
                使用遺傳演算法自動優化型態參數，策略會：
                - 自動尋找最佳的型態識別參數
                - 根據歷史數據調整進出場條件
                - 持續演化以適應市場變化
                
                💡 建議：可同時啟用「雙引擎」和「因子權重」讓演化優化更全面
                """)
        
        elif active_count == 2:
            if enabled_strategies.get('dual_engine') and enabled_strategies.get('factor_weight'):
                st.success("""
                **型態 + 雙引擎 + 因子權重模式** ⭐ 推薦組合
                
                這是一個平衡的策略組合：
                - 雙引擎根據市場狀態選擇策略方向
                - 因子權重優化訊號品質和可靠性
                - 型態識別提供具體的進場時機
                
                ✅ 優點：多層次過濾，訊號品質較高
                ⚠️ 注意：可能因過濾過嚴導致交易次數減少
                """)
            elif enabled_strategies.get('dual_engine') and enabled_strategies.get('evolution'):
                st.info("""
                **型態 + 雙引擎 + 演化優化模式**
                
                演化優化會自動調整雙引擎的參數：
                - 優化 ADX 閾值以更準確判斷市場狀態
                - 調整各策略的資金配置比例
                - 尋找最佳的趨勢/震盪策略參數
                
                💡 建議：可加入「因子權重優化」讓演化同時優化指標權重
                """)
            elif enabled_strategies.get('factor_weight') and enabled_strategies.get('evolution'):
                st.info("""
                **型態 + 因子權重 + 演化優化模式**
                
                演化優化會自動調整因子權重：
                - 尋找各指標的最佳權重配置
                - 優化買入/觀望閾值
                - 調整各指標的細節參數
                
                💡 建議：可加入「雙引擎策略」讓策略能適應不同市場狀態
                """)
        
        elif active_count == 3:
            st.success("""
            **全策略整合模式** 🚀 最完整配置
            
            整合所有策略層的完整系統：
            
            1. **演化優化** 自動尋找最佳參數組合
            2. **雙引擎策略** 根據市場狀態切換策略
            3. **因子權重優化** 多維度確認訊號品質
            4. **型態識別** 提供具體進場時機
            
            ✅ 優點：
            - 參數自動優化，減少人工調整
            - 多層次過濾，訊號品質最高
            - 自適應市場狀態變化
            
            ⚠️ 注意：
            - 計算成本較高
            - 需要足夠的歷史數據
            - 可能存在過度擬合風險
            
            💡 建議：使用滾動視窗驗證 (Walk-Forward) 確保參數的泛化能力
            """)
        
        # 如果有比較報告，顯示策略組合的實際效果
        if result.dual_engine_report and result.dual_engine_report.comparison_report:
            report = result.dual_engine_report.comparison_report
            st.markdown("**📈 策略組合實際效果**")
            
            # 判斷是否為正向綜效
            improvements = [
                report.total_return_diff.is_improvement,
                report.sharpe_ratio_diff.is_improvement,
                report.max_drawdown_diff.is_improvement,
            ]
            positive_count = sum(improvements)
            
            if positive_count >= 2:
                st.success(f"""
                ✅ **正向綜效**：相比 {report.baseline_name}，{report.current_name} 在多數指標上有改善
                - 報酬率變化: {report.total_return_diff.diff_value:+.2f}%
                - 夏普比率變化: {report.sharpe_ratio_diff.diff_value:+.2f}
                - 最大回撤變化: {report.max_drawdown_diff.diff_value:+.2f}%
                """)
            elif positive_count == 1:
                st.warning(f"""
                ⚠️ **混合效果**：相比 {report.baseline_name}，{report.current_name} 的效果不一致
                - 報酬率變化: {report.total_return_diff.diff_value:+.2f}%
                - 夏普比率變化: {report.sharpe_ratio_diff.diff_value:+.2f}
                - 最大回撤變化: {report.max_drawdown_diff.diff_value:+.2f}%
                
                💡 建議：檢視各策略的參數設定，或嘗試不同的策略組合
                """)
            else:
                st.error(f"""
                ❌ **負向綜效**：相比 {report.baseline_name}，{report.current_name} 的績效下降
                - 報酬率變化: {report.total_return_diff.diff_value:+.2f}%
                - 夏普比率變化: {report.sharpe_ratio_diff.diff_value:+.2f}
                - 最大回撤變化: {report.max_drawdown_diff.diff_value:+.2f}%
                
                ⚠️ 警告：當前策略組合可能不適合此市場環境
                💡 建議：考慮簡化策略配置，或使用演化優化重新尋找最佳參數
                """)

    def _render_equity_curve(self, result: EnhancedBacktestResult) -> None:
        """渲染資金曲線（支援多策略比較）"""
        if not result.equity_curve:
            st.info("無資金曲線數據")
            return
        
        st.markdown("### 📈 資金曲線")
        
        # 準備總體資金曲線數據
        df = pd.DataFrame(result.equity_curve)
        df['日期'] = pd.to_datetime(df['date'])
        df = df.set_index('日期')
        
        # 檢查是否有多策略資金曲線
        has_multi_strategy = (
            result.strategy_equity_curves and 
            len(result.strategy_equity_curves) > 1
        )
        
        if has_multi_strategy:
            # 多策略比較模式
            st.markdown("**📊 多策略資金曲線比較**")
            
            # 建立多策略比較 DataFrame
            multi_df = pd.DataFrame({'日期': pd.to_datetime([e['date'] for e in result.equity_curve])})
            multi_df = multi_df.set_index('日期')
            
            # 加入總體資金曲線
            multi_df['總體'] = df['equity'].values
            
            # 策略名稱映射
            strategy_names = {
                'trend': '🟢 趨勢策略',
                'mean_reversion': '🔵 均值回歸',
                'pattern': '🔷 型態策略',
            }
            
            # 加入各策略資金曲線
            for strategy_key, curve_data in result.strategy_equity_curves.items():
                if curve_data:
                    strategy_name = strategy_names.get(strategy_key, strategy_key)
                    # 確保數據長度一致
                    if len(curve_data) == len(multi_df):
                        multi_df[strategy_name] = [e['equity'] for e in curve_data]
            
            # 顯示多策略資金曲線
            st.line_chart(multi_df, use_container_width=True)
            
            # 顯示策略圖例說明
            with st.expander("📖 圖例說明", expanded=False):
                st.markdown("""
                | 曲線 | 說明 |
                |------|------|
                | **總體** | 所有策略合併的總資金曲線 |
                | **🟢 趨勢策略** | 趨勢追蹤策略的獨立資金曲線 |
                | **🔵 均值回歸** | 均值回歸策略的獨立資金曲線 |
                | **🔷 型態策略** | 型態識別策略的獨立資金曲線 |
                
                💡 **提示**：比較各策略曲線可以了解哪種策略在不同市場環境下表現較好
                """)
            
            # 計算各策略最終報酬率
            st.markdown("**📊 各策略最終報酬率**")
            initial_capital = result.equity_curve[0]['equity'] if result.equity_curve else 1000000
            
            returns_data = []
            for col in multi_df.columns:
                final_value = multi_df[col].iloc[-1]
                total_return = ((final_value - initial_capital) / initial_capital) * 100
                returns_data.append({
                    '策略': col,
                    '最終資金': f"${final_value:,.0f}",
                    '總報酬率': f"{total_return:+.2f}%"
                })
            
            returns_df = pd.DataFrame(returns_data)
            st.dataframe(returns_df, use_container_width=True, hide_index=True)
        else:
            # 單一資金曲線模式
            st.line_chart(df['equity'], use_container_width=True)
        
        # 顯示回撤曲線
        if 'drawdown' in df.columns:
            with st.expander("📉 回撤曲線", expanded=False):
                st.area_chart(df['drawdown'], use_container_width=True, color="#ff6b6b")
    
    def _render_trade_list(self, trades: List[EnhancedBacktestTrade]) -> None:
        """渲染交易明細列表"""
        if not trades:
            st.info("無交易記錄")
            return
        
        st.markdown("### 📋 交易明細與分析")
        st.caption(f"共 {len(trades)} 筆交易")
        
        for i, trade in enumerate(trades):
            # 策略類型標籤
            strategy_badge = self._get_strategy_badge(trade.strategy_type)
            with st.expander(
                f"{'🟢' if trade.pnl > 0 else '🔴'} {strategy_badge} {trade.symbol} | "
                f"{trade.entry_date.strftime('%Y-%m-%d')} → {trade.exit_date.strftime('%Y-%m-%d')} | "
                f"損益: {trade.pnl_pct:+.2f}%",
                expanded=(i == 0)  # 展開第一筆
            ):
                self._render_trade_detail(trade, trade_index=i)
    
    def _render_strategy_performance_tabs(self, report: DualEngineBacktestReport) -> None:
        """渲染分策略績效（使用子頁籤）"""
        st.markdown("### 🎯 分策略績效分析")
        
        # 總體績效摘要
        st.markdown("**📈 總體績效**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta = None
            if report.baseline_total_return is not None:
                delta = f"{report.total_return - report.baseline_total_return:+.2f}%"
            st.metric("總報酬率", f"{report.total_return:+.2f}%", delta=delta)
        
        with col2:
            delta = None
            if report.baseline_sharpe_ratio is not None:
                delta = f"{report.sharpe_ratio - report.baseline_sharpe_ratio:+.2f}"
            st.metric("夏普比率", f"{report.sharpe_ratio:.2f}", delta=delta)
        
        with col3:
            delta = None
            if report.baseline_max_drawdown is not None:
                diff = report.max_drawdown - report.baseline_max_drawdown
                delta = f"{-diff:+.2f}%" if diff != 0 else None
            st.metric("最大回撤", f"{report.max_drawdown:.2f}%", delta=delta, delta_color="inverse")
        
        with col4:
            st.metric("總交易次數", report.total_trades)
        
        st.divider()
        
        # 各策略績效子頁籤
        strategy_tabs = []
        strategy_names = []
        
        if report.trend_performance.total_trades > 0:
            strategy_tabs.append(("🟢 趨勢策略", report.trend_performance, "trend"))
            strategy_names.append("🟢 趨勢策略")
        
        if report.reversion_performance.total_trades > 0:
            strategy_tabs.append(("🔵 均值回歸", report.reversion_performance, "reversion"))
            strategy_names.append("🔵 均值回歸")
        
        if report.pattern_performance and report.pattern_performance.total_trades > 0:
            strategy_tabs.append(("🔷 型態策略", report.pattern_performance, "pattern"))
            strategy_names.append("🔷 型態策略")
        
        if strategy_tabs:
            sub_tabs = st.tabs(strategy_names + ["📊 比較圖表"])
            
            for i, (title, performance, key) in enumerate(strategy_tabs):
                with sub_tabs[i]:
                    self._render_strategy_performance_card(title, performance, key)
            
            # 比較圖表
            with sub_tabs[-1]:
                self._render_strategy_comparison_chart(report)
        else:
            st.info("無分策略績效數據")
    
    def _get_strategy_badge(self, strategy_type: str) -> str:
        """取得策略類型標籤"""
        badges = {
            "trend": "📈趨勢",
            "mean_reversion": "📊回歸",
            "pattern": "🔷型態"
        }
        return badges.get(strategy_type, "")
    
    def _render_dual_engine_report(self, report: DualEngineBacktestReport) -> None:
        """渲染雙引擎分策略回測報告（舊版，保留向後兼容）
        
        注意：此方法已被 render_backtest_results 中的頁籤版本取代
        Requirements: 11.1, 11.2, 11.3, 11.4
        """
        # 此方法現在主要用於非頁籤模式的向後兼容
        # 主要邏輯已移至 _render_strategy_performance_tabs
        self._render_strategy_performance_tabs(report)
    
    def _render_comparison_report(self, report: StrategyComparisonReport) -> None:
        """渲染策略差異比較報告"""
        st.markdown("### 📊 策略差異比較")
        st.markdown(f"**{report.baseline_name}** vs **{report.current_name}**")
        
        # 摘要
        if report.summary:
            st.info(report.summary)
        
        # 主要指標差異
        st.markdown("**📈 主要指標差異**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_diff_metric(
                "總報酬率",
                report.total_return_diff,
                suffix="%"
            )
        
        with col2:
            self._render_diff_metric(
                "夏普比率",
                report.sharpe_ratio_diff,
                suffix=""
            )
        
        with col3:
            self._render_diff_metric(
                "最大回撤",
                report.max_drawdown_diff,
                suffix="%",
                inverse=True
            )
        
        with col4:
            self._render_diff_metric(
                "勝率",
                report.win_rate_diff,
                suffix="%"
            )
        
        # 詳細比較表格
        with st.expander("📋 詳細差異比較", expanded=False):
            comparison_data = [
                {
                    "指標": "總報酬率",
                    f"{report.baseline_name}": f"{report.total_return_diff.baseline_value:+.2f}%",
                    f"{report.current_name}": f"{report.total_return_diff.current_value:+.2f}%",
                    "差異": f"{report.total_return_diff.diff_value:+.2f}%",
                    "變化率": f"{report.total_return_diff.diff_percent:+.1f}%",
                    "評估": "✅ 改善" if report.total_return_diff.is_improvement else "⚠️ 下降"
                },
                {
                    "指標": "夏普比率",
                    f"{report.baseline_name}": f"{report.sharpe_ratio_diff.baseline_value:.2f}",
                    f"{report.current_name}": f"{report.sharpe_ratio_diff.current_value:.2f}",
                    "差異": f"{report.sharpe_ratio_diff.diff_value:+.2f}",
                    "變化率": f"{report.sharpe_ratio_diff.diff_percent:+.1f}%",
                    "評估": "✅ 改善" if report.sharpe_ratio_diff.is_improvement else "⚠️ 下降"
                },
                {
                    "指標": "最大回撤",
                    f"{report.baseline_name}": f"{report.max_drawdown_diff.baseline_value:.2f}%",
                    f"{report.current_name}": f"{report.max_drawdown_diff.current_value:.2f}%",
                    "差異": f"{report.max_drawdown_diff.diff_value:+.2f}%",
                    "變化率": f"{report.max_drawdown_diff.diff_percent:+.1f}%",
                    "評估": "✅ 改善" if report.max_drawdown_diff.is_improvement else "⚠️ 下降"
                },
                {
                    "指標": "勝率",
                    f"{report.baseline_name}": f"{report.win_rate_diff.baseline_value:.1f}%",
                    f"{report.current_name}": f"{report.win_rate_diff.current_value:.1f}%",
                    "差異": f"{report.win_rate_diff.diff_value:+.1f}%",
                    "變化率": f"{report.win_rate_diff.diff_percent:+.1f}%",
                    "評估": "✅ 改善" if report.win_rate_diff.is_improvement else "⚠️ 下降"
                },
                {
                    "指標": "交易次數",
                    f"{report.baseline_name}": f"{report.trade_count_diff.baseline_value:.0f}",
                    f"{report.current_name}": f"{report.trade_count_diff.current_value:.0f}",
                    "差異": f"{report.trade_count_diff.diff_value:+.0f}",
                    "變化率": f"{report.trade_count_diff.diff_percent:+.1f}%",
                    "評估": "-"
                },
                {
                    "指標": "平均獲利",
                    f"{report.baseline_name}": f"{report.avg_profit_diff.baseline_value:+.2f}%",
                    f"{report.current_name}": f"{report.avg_profit_diff.current_value:+.2f}%",
                    "差異": f"{report.avg_profit_diff.diff_value:+.2f}%",
                    "變化率": f"{report.avg_profit_diff.diff_percent:+.1f}%",
                    "評估": "✅ 改善" if report.avg_profit_diff.is_improvement else "⚠️ 下降"
                },
                {
                    "指標": "獲利因子",
                    f"{report.baseline_name}": f"{report.profit_factor_diff.baseline_value:.2f}",
                    f"{report.current_name}": f"{report.profit_factor_diff.current_value:.2f}",
                    "差異": f"{report.profit_factor_diff.diff_value:+.2f}",
                    "變化率": f"{report.profit_factor_diff.diff_percent:+.1f}%",
                    "評估": "✅ 改善" if report.profit_factor_diff.is_improvement else "⚠️ 下降"
                },
            ]
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _render_diff_metric(
        self,
        label: str,
        diff: PerformanceDiff,
        suffix: str = "",
        inverse: bool = False
    ) -> None:
        """渲染差異指標"""
        # 對於回撤，inverse=True 表示降低是好事
        delta_color = "normal"
        if inverse:
            delta_color = "inverse"
        
        delta_str = f"{diff.diff_value:+.2f}{suffix}"
        if abs(diff.diff_percent) > 0.1:
            delta_str += f" ({diff.diff_percent:+.1f}%)"
        
        st.metric(
            label=label,
            value=f"{diff.current_value:.2f}{suffix}",
            delta=delta_str,
            delta_color=delta_color
        )
    
    def _render_strategy_performance_card(
        self,
        title: str,
        performance: StrategyPerformance,
        strategy_key: str
    ) -> None:
        """渲染單一策略績效卡片
        
        Requirements: 11.2, 11.3, 11.4
        """
        st.markdown(f"**{title}**")
        
        if performance.total_trades == 0:
            st.info("此策略無交易記錄")
            return
        
        # 主要指標
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "勝率",
                f"{performance.win_rate:.1f}%",
                delta=f"{performance.winning_trades}勝 / {performance.losing_trades}負"
            )
        with col2:
            st.metric(
                "交易次數",
                performance.total_trades
            )
        
        # 詳細指標表格
        st.markdown(f"""
        | 指標 | 數值 |
        |------|------|
        | 平均獲利 | {performance.avg_profit:+.2f}% |
        | 平均虧損 | {performance.avg_loss:+.2f}% |
        | 最大回撤 | {performance.max_drawdown:.2f}% |
        | 獲利因子 | {performance.profit_factor:.2f} |
        """)
    
    def _render_strategy_comparison_chart(self, report: DualEngineBacktestReport) -> None:
        """渲染策略績效比較圖表"""
        st.markdown("**📊 策略績效比較**")
        
        # 準備數據
        strategies = []
        win_rates = []
        trade_counts = []
        avg_profits = []
        
        if report.trend_performance.total_trades > 0:
            strategies.append("趨勢策略")
            win_rates.append(report.trend_performance.win_rate)
            trade_counts.append(report.trend_performance.total_trades)
            avg_profits.append(report.trend_performance.avg_profit)
        
        if report.reversion_performance.total_trades > 0:
            strategies.append("均值回歸")
            win_rates.append(report.reversion_performance.win_rate)
            trade_counts.append(report.reversion_performance.total_trades)
            avg_profits.append(report.reversion_performance.avg_profit)
        
        if report.pattern_performance and report.pattern_performance.total_trades > 0:
            strategies.append("型態策略")
            win_rates.append(report.pattern_performance.win_rate)
            trade_counts.append(report.pattern_performance.total_trades)
            avg_profits.append(report.pattern_performance.avg_profit)
        
        if not strategies:
            st.info("無足夠數據生成比較圖表")
            return
        
        # 建立比較 DataFrame
        comparison_df = pd.DataFrame({
            "策略": strategies,
            "勝率 (%)": win_rates,
            "交易次數": trade_counts,
            "平均獲利 (%)": avg_profits
        })
        
        # 顯示表格
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "勝率 (%)": st.column_config.ProgressColumn(
                    "勝率 (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            }
        )
        
        # 勝率比較柱狀圖
        if len(strategies) > 1:
            chart_df = pd.DataFrame({
                "勝率": win_rates
            }, index=strategies)
            st.bar_chart(chart_df)
    
    def _render_trade_detail(self, trade: EnhancedBacktestTrade, trade_index: int = 0) -> None:
        """渲染單筆交易詳情"""
        col1, col2 = st.columns([2, 1])
        
        # 生成唯一的 key
        unique_key = f"trade_{trade_index}_{trade.symbol}_{trade.entry_date.strftime('%Y%m%d')}"
        
        with col1:
            # 顯示圖表
            if trade.ohlcv_data:
                st.markdown("**📊 型態圖表**")
                fig = self.chart_view.create_candlestick_chart(
                    trade.ohlcv_data,
                    trade.symbol,
                    trade.pattern_result
                )
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{unique_key}")
            else:
                st.info("無圖表數據")
        
        with col2:
            # 交易資訊
            st.markdown("**📝 交易資訊**")
            st.markdown(f"""
            | 項目 | 數值 |
            |------|------|
            | 股票代碼 | {trade.symbol} |
            | 進場日期 | {trade.entry_date.strftime('%Y-%m-%d')} |
            | 進場價格 | ${trade.entry_price:.2f} |
            | 成交股數 | {getattr(trade, 'shares', 'N/A')} |
            | 出場日期 | {trade.exit_date.strftime('%Y-%m-%d')} |
            | 出場價格 | ${trade.exit_price:.2f} |
            | 出場原因 | {self._translate_exit_reason(trade.exit_reason)} |
            | 持有天數 | {trade.holding_days} 天 |
            | 損益金額 | ${trade.pnl:+,.2f} |
            | 損益比例 | {trade.pnl_pct:+.2f}% |
            """)
            
            # 關鍵價位
            if trade.resistance_price > 0:
                st.markdown("**💰 關鍵價位**")
                st.markdown(f"""
                | 價位 | 數值 |
                |------|------|
                | 壓力位 | ${trade.resistance_price:.2f} |
                | 突破價 | ${trade.breakout_price:.2f} |
                | 止損價 | ${trade.stop_loss_price:.2f} |
                """)
            
            # 分數明細
            if trade.score_breakdown:
                st.markdown("**🎯 吻合分數明細**")
                scores = trade.score_breakdown
                
                # 顯示是否使用因子權重優化
                if scores.get('optimized', False):
                    base_score = scores.get('base_score', 0)
                    final_score = scores.get('total', 0)
                    score_diff = final_score - base_score
                    st.markdown(f"""
                    | 分項 | 分數 |
                    |------|------|
                    | 擬合度 | {scores.get('r_squared', 0):.1f} |
                    | 對稱性 | {scores.get('symmetry', 0):.1f} |
                    | 成交量 | {scores.get('volume', 0):.1f} |
                    | 深度 | {scores.get('depth', 0):.1f} |
                    | 型態基礎分 | {base_score:.1f} |
                    | 因子調整 | {score_diff:+.1f} |
                    | **最終分數** | **{final_score:.1f}** |
                    """)
                else:
                    st.markdown(f"""
                    | 分項 | 分數 |
                    |------|------|
                    | 擬合度 | {scores.get('r_squared', 0):.1f} |
                    | 對稱性 | {scores.get('symmetry', 0):.1f} |
                    | 成交量 | {scores.get('volume', 0):.1f} |
                    | 深度 | {scores.get('depth', 0):.1f} |
                    | **總分** | **{scores.get('total', 0):.1f}** |
                    """)
        
        # 進場原因說明
        if trade.entry_reason:
            st.markdown("**🔍 進場原因分析**")
            st.info(trade.entry_reason)
        
        # 因子權重評分詳情
        if trade.optimized_signal_details:
            self._render_factor_weight_details(trade, unique_key)
        
        # 型態數學註解
        if trade.pattern_result and trade.pattern_result.is_valid:
            with st.expander("📐 型態數學註解"):
                self.chart_view.render_pattern_annotations(trade.pattern_result)
    
    def _render_factor_weight_details(self, trade: EnhancedBacktestTrade, unique_key: str) -> None:
        """渲染因子權重評分詳情"""
        with st.expander("⚗️ 因子權重評分詳情", expanded=True):
            # 訊號強度
            if trade.signal_strength:
                strength_map = {
                    'strong_buy': ('🟢 強力買入', 'success'),
                    'watch': ('🟡 觀望', 'warning'),
                    'skip': ('🔴 跳過', 'error')
                }
                strength_text, strength_type = strength_map.get(
                    trade.signal_strength, ('❓ 未知', 'info')
                )
                st.markdown(f"**訊號強度**: {strength_text}")
            
            st.markdown("**各指標評分明細**")
            
            # 建立評分表格
            details = trade.optimized_signal_details
            
            # 分類顯示
            pattern_details = []
            indicator_details = []
            
            for detail in details:
                if detail.source == 'pattern':
                    pattern_details.append(detail)
                else:
                    indicator_details.append(detail)
            
            # 型態分數
            if pattern_details:
                st.markdown("**📊 型態識別**")
                for d in pattern_details:
                    st.markdown(f"- {d.reason}: **{d.score_change:+.1f}** 分")
            
            # 技術指標分數
            if indicator_details:
                st.markdown("**📈 技術指標調整**")
                
                # 按來源分組
                source_groups = {}
                for d in indicator_details:
                    if d.source not in source_groups:
                        source_groups[d.source] = []
                    source_groups[d.source].append(d)
                
                # 來源名稱對照
                source_names = {
                    'rsi': 'RSI 相對強弱指標',
                    'volume': '成交量',
                    'macd': 'MACD 指標',
                    'ema': '均線 (EMA)',
                    'bollinger': '布林通道'
                }
                
                for source, group_details in source_groups.items():
                    source_name = source_names.get(source, source.upper())
                    total_change = sum(d.score_change for d in group_details)
                    
                    # 顯示來源標題和總分變化
                    color = "green" if total_change > 0 else ("red" if total_change < 0 else "gray")
                    st.markdown(f"**{source_name}** ({total_change:+.1f} 分)")
                    
                    # 顯示每個細項
                    for d in group_details:
                        icon = "✅" if d.score_change > 0 else ("❌" if d.score_change < 0 else "➖")
                        st.markdown(f"  - {icon} {d.reason} ({d.score_change:+.1f})")
            
            # 總結
            if trade.score_breakdown:
                scores = trade.score_breakdown
                base_score = scores.get('base_score', 0)
                final_score = scores.get('total', 0)
                
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("型態基礎分", f"{base_score:.1f}")
                with col2:
                    diff = final_score - base_score
                    st.metric("因子調整", f"{diff:+.1f}", delta=f"{diff:+.1f}")
                with col3:
                    st.metric("最終分數", f"{final_score:.1f}")
    
    def _translate_exit_reason(self, reason: str) -> str:
        """翻譯出場原因"""
        translations = {
            'stop_loss': '🔴 硬止損',
            'technical_stop': '🟠 技術止損',
            'trailing_stop': '🟢 移動止盈',
            'target': '🎯 達到目標',
            'max_holding': '⏰ 最大持有期限',
            'end_of_backtest': '📅 回測結束'
        }
        return translations.get(reason, reason)
    
    def _render_embedded_factor_lab(self) -> None:
        """渲染嵌入式因子權重實驗室
        
        Requirements: 11.1
        """
        try:
            from pattern_quant.ui.factor_weight_lab import FactorWeightLab
            
            # 建立因子權重實驗室實例
            factor_lab = FactorWeightLab()
            factor_lab.render()
            
        except ImportError as e:
            st.error(f"❌ 無法載入因子權重實驗室: {e}")
            st.info("請確保已安裝所有必要的依賴套件")
        except Exception as e:
            st.error(f"❌ 因子權重實驗室發生錯誤: {e}")
    
    def render_dual_engine_panel(self) -> Optional[DualEngineConfig]:
        """渲染雙引擎模式設定面板
        
        Requirements: 13.1, 13.2, 13.3, 13.4
        
        Returns:
            DualEngineConfig 若啟用雙引擎模式，否則 None
        """
        st.subheader("🔄 雙引擎策略模式")
        
        # 初始化配置管理器
        if 'dual_engine_config_manager' not in st.session_state:
            st.session_state.dual_engine_config_manager = DualEngineConfigManager()
        
        config_manager = st.session_state.dual_engine_config_manager
        
        # 取得當前配置
        current_config = config_manager.get_config()
        
        # 雙引擎模式開關 (Requirement 13.1)
        col1, col2 = st.columns([1, 2])
        with col1:
            dual_engine_enabled = st.toggle(
                "啟用雙引擎模式",
                value=st.session_state.get("dual_engine_enabled", current_config.enabled),
                help="啟用後系統會根據市場狀態自動切換趨勢策略與均值回歸策略",
                key="dual_engine_toggle"
            )
            st.session_state.dual_engine_enabled = dual_engine_enabled
        
        with col2:
            if dual_engine_enabled:
                st.success("✅ 已啟用雙引擎模式，系統將根據 ADX 與 BBW 自動判定市場狀態")
            else:
                st.info("💡 未啟用雙引擎模式，僅使用型態突破策略")
        
        if not dual_engine_enabled:
            return None
        
        # 市場狀態分類器參數設定面板 (Requirement 13.2)
        with st.expander("⚙️ 市場狀態分類器設定", expanded=True):
            st.markdown("""
            **市場狀態分類規則：**
            - 🟢 **TREND（趨勢）**: ADX > 趨勢閾值 → 執行型態突破策略
            - 🔵 **RANGE（震盪）**: ADX < 震盪閾值 且 BBW 穩定 → 執行均值回歸策略
            - ⚪ **NOISE（混沌）**: 介於兩者之間 → 暫停開新倉
            """)
            
            st.divider()
            
            # ADX 閾值設定 (Requirement 13.3)
            st.markdown("**📊 ADX 閾值設定**")
            col1, col2 = st.columns(2)
            
            with col1:
                adx_trend_threshold = st.slider(
                    "趨勢判定閾值 (ADX >)",
                    min_value=15.0,
                    max_value=40.0,
                    value=st.session_state.get("adx_trend_threshold", current_config.adx_trend_threshold),
                    step=1.0,
                    help="ADX 高於此值判定為趨勢市場",
                    key="adx_trend_slider"
                )
                st.session_state.adx_trend_threshold = adx_trend_threshold
            
            with col2:
                adx_range_threshold = st.slider(
                    "震盪判定閾值 (ADX <)",
                    min_value=10.0,
                    max_value=30.0,
                    value=st.session_state.get("adx_range_threshold", current_config.adx_range_threshold),
                    step=1.0,
                    help="ADX 低於此值判定為震盪市場",
                    key="adx_range_slider"
                )
                st.session_state.adx_range_threshold = adx_range_threshold
            
            # 驗證閾值邏輯
            if adx_range_threshold >= adx_trend_threshold:
                st.warning("⚠️ 震盪閾值應小於趨勢閾值，請調整設定")
            
            st.divider()
            
            # 資金權重設定 (Requirement 13.4)
            st.markdown("**💰 各策略資金權重**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_allocation = st.slider(
                    "🟢 趨勢狀態權重",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("trend_allocation", current_config.trend_allocation),
                    step=0.1,
                    format="%.0f%%",
                    help="趨勢市場時的資金使用比例",
                    key="trend_allocation_slider"
                )
                st.session_state.trend_allocation = trend_allocation
                st.caption(f"使用 {trend_allocation*100:.0f}% 資金")
            
            with col2:
                range_allocation = st.slider(
                    "🔵 震盪狀態權重",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("range_allocation", current_config.range_allocation),
                    step=0.1,
                    format="%.0f%%",
                    help="震盪市場時的資金使用比例",
                    key="range_allocation_slider"
                )
                st.session_state.range_allocation = range_allocation
                st.caption(f"使用 {range_allocation*100:.0f}% 資金")
            
            with col3:
                noise_allocation = st.slider(
                    "⚪ 混沌狀態權重",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("noise_allocation", current_config.noise_allocation),
                    step=0.1,
                    format="%.0f%%",
                    help="混沌市場時的資金使用比例（建議為 0）",
                    key="noise_allocation_slider"
                )
                st.session_state.noise_allocation = noise_allocation
                st.caption(f"使用 {noise_allocation*100:.0f}% 資金")
            
            st.divider()
            
            # 進階策略參數
            with st.expander("🔧 進階策略參數", expanded=False):
                st.markdown("**趨勢策略參數**")
                col1, col2 = st.columns(2)
                with col1:
                    trend_score_threshold = st.slider(
                        "型態分數閾值",
                        min_value=60.0,
                        max_value=95.0,
                        value=st.session_state.get("trend_score_threshold", current_config.trend_score_threshold),
                        step=5.0,
                        help="型態分數高於此值才觸發趨勢策略",
                        key="trend_score_slider"
                    )
                    st.session_state.trend_score_threshold = trend_score_threshold
                
                with col2:
                    trend_risk_per_trade = st.slider(
                        "單筆風險比例 (%)",
                        min_value=0.5,
                        max_value=3.0,
                        value=st.session_state.get("trend_risk_per_trade", current_config.trend_risk_per_trade * 100),
                        step=0.5,
                        help="趨勢策略每筆交易的風險比例",
                        key="trend_risk_slider"
                    )
                    st.session_state.trend_risk_per_trade = trend_risk_per_trade
                
                st.markdown("**均值回歸策略參數**")
                col1, col2 = st.columns(2)
                with col1:
                    reversion_rsi_oversold = st.slider(
                        "RSI 超賣閾值",
                        min_value=20.0,
                        max_value=40.0,
                        value=st.session_state.get("reversion_rsi_oversold", current_config.reversion_rsi_oversold),
                        step=5.0,
                        help="RSI 低於此值確認超賣",
                        key="rsi_oversold_slider"
                    )
                    st.session_state.reversion_rsi_oversold = reversion_rsi_oversold
                
                with col2:
                    reversion_position_ratio = st.slider(
                        "倉位比例 (%)",
                        min_value=2.0,
                        max_value=10.0,
                        value=st.session_state.get("reversion_position_ratio", current_config.reversion_position_ratio * 100),
                        step=1.0,
                        help="均值回歸策略每筆交易的倉位比例",
                        key="reversion_position_slider"
                    )
                    st.session_state.reversion_position_ratio = reversion_position_ratio
        
        # 建立並返回配置
        config = DualEngineConfig(
            enabled=dual_engine_enabled,
            adx_trend_threshold=adx_trend_threshold,
            adx_range_threshold=adx_range_threshold,
            trend_allocation=trend_allocation,
            range_allocation=range_allocation,
            noise_allocation=noise_allocation,
            trend_score_threshold=st.session_state.get("trend_score_threshold", current_config.trend_score_threshold),
            trend_risk_per_trade=st.session_state.get("trend_risk_per_trade", current_config.trend_risk_per_trade * 100) / 100,
            reversion_rsi_oversold=st.session_state.get("reversion_rsi_oversold", current_config.reversion_rsi_oversold),
            reversion_position_ratio=st.session_state.get("reversion_position_ratio", current_config.reversion_position_ratio * 100) / 100,
        )
        
        # 儲存配置
        config_manager.save_config(config)
        st.session_state.dual_engine_config = config
        
        return config
    
    def render_evolution_panel(self) -> Optional[EvolutionBacktestConfig]:
        """渲染演化優化設定面板
        
        Returns:
            EvolutionBacktestConfig 若啟用演化優化，否則 None
        """
        st.subheader("🧬 演化優化")
        
        if not EVOLUTION_AVAILABLE:
            st.warning("⚠️ 演化優化模組未安裝，請確保 pattern_quant.evolution 模組可用")
            return None
        
        col1, col2 = st.columns([1, 2])
        with col1:
            evo_enabled = st.toggle(
                "啟用演化優化",
                value=st.session_state.get("evo_backtest_enabled", False),
                help="啟用後系統會在回測過程中自動演化優化參數",
                key="evo_backtest_toggle"
            )
            st.session_state.evo_backtest_enabled = evo_enabled
        
        with col2:
            if evo_enabled:
                st.success("✅ 已啟用演化優化，系統將在回測斷點自動調整最佳參數")
            else:
                st.info("💡 未啟用演化優化，使用固定參數進行回測")
        
        if not evo_enabled:
            return None
        
        # 演化優化設定
        with st.expander("⚙️ 演化優化設定", expanded=True):
            st.markdown("""
            **演化優化說明：**
            - 🧬 系統會在每個回測視窗使用遺傳演算法尋找最佳參數
            - 📊 每個視窗結束後，最佳參數會應用到下一個視窗
            - 🔄 這模擬了實際交易中的參數自適應調整
            """)
            
            st.divider()
            
            # 優化目標選擇
            st.markdown("**🎯 優化目標**")
            objective_options = {
                "sharpe_ratio": "夏普比率 (Sharpe Ratio) - 風險調整後收益",
                "sortino_ratio": "索提諾比率 (Sortino Ratio) - 下行風險調整",
                "net_profit": "淨利潤 (Net Profit) - 最大化收益",
                "min_max_drawdown": "最小化回撤 (Min Drawdown) - 防禦型",
            }
            
            fitness_objective = st.selectbox(
                "選擇優化目標",
                options=list(objective_options.keys()),
                format_func=lambda x: objective_options[x],
                index=0,
                key="evo_fitness_objective"
            )
            
            st.divider()
            
            # 優化範圍選擇
            st.markdown("**📋 優化範圍**")
            col1, col2 = st.columns(2)
            
            with col1:
                optimize_dual_engine = st.checkbox(
                    "優化雙引擎參數",
                    value=st.session_state.get("evo_optimize_dual_engine", True),
                    help="包含 ADX 閾值、資金權重等參數",
                    key="evo_optimize_dual_engine_cb"
                )
                st.session_state.evo_optimize_dual_engine = optimize_dual_engine
            
            with col2:
                optimize_factor_weights = st.checkbox(
                    "優化因子權重",
                    value=st.session_state.get("evo_optimize_factor_weights", True),
                    help="包含 RSI、MACD、成交量等因子權重",
                    key="evo_optimize_factor_weights_cb"
                )
                st.session_state.evo_optimize_factor_weights = optimize_factor_weights
            
            st.divider()
            
            # 演化參數
            st.markdown("**⚙️ 演化參數**")
            col1, col2 = st.columns(2)
            
            with col1:
                population_size = st.slider(
                    "種群大小",
                    min_value=50,
                    max_value=100,
                    value=st.session_state.get("evo_population_size", 50),
                    step=10,
                    help="每一世代的個體數量",
                    key="evo_pop_size_slider"
                )
                st.session_state.evo_population_size = population_size
                
                max_generations = st.slider(
                    "最大世代數",
                    min_value=10,
                    max_value=30,
                    value=st.session_state.get("evo_max_generations", 15),
                    step=5,
                    help="每個視窗的演化迭代次數",
                    key="evo_max_gen_slider"
                )
                st.session_state.evo_max_generations = max_generations
            
            with col2:
                window_size_days = st.slider(
                    "演化視窗大小 (天)",
                    min_value=63,
                    max_value=252,
                    value=st.session_state.get("evo_window_size", 126),
                    step=21,
                    help="用於演化優化的歷史數據天數",
                    key="evo_window_size_slider"
                )
                st.session_state.evo_window_size = window_size_days
                
                step_size_days = st.slider(
                    "步進大小 (天)",
                    min_value=5,
                    max_value=63,
                    value=st.session_state.get("evo_step_size", 21),
                    step=7,
                    help="每次演化後推進的天數",
                    key="evo_step_size_slider"
                )
                st.session_state.evo_step_size = step_size_days
            
            # 進階參數
            with st.expander("🔧 進階演化參數", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    elitism_rate = st.slider(
                        "精英保留率",
                        min_value=0.05,
                        max_value=0.20,
                        value=st.session_state.get("evo_elitism_rate", 0.1),
                        step=0.05,
                        format="%.2f",
                        key="evo_elitism_slider"
                    )
                    st.session_state.evo_elitism_rate = elitism_rate
                
                with col2:
                    crossover_rate = st.slider(
                        "交叉率",
                        min_value=0.6,
                        max_value=0.9,
                        value=st.session_state.get("evo_crossover_rate", 0.8),
                        step=0.1,
                        format="%.1f",
                        key="evo_crossover_slider"
                    )
                    st.session_state.evo_crossover_rate = crossover_rate
                
                with col3:
                    mutation_rate = st.slider(
                        "突變率",
                        min_value=0.01,
                        max_value=0.05,
                        value=st.session_state.get("evo_mutation_rate", 0.02),
                        step=0.01,
                        format="%.2f",
                        key="evo_mutation_slider"
                    )
                    st.session_state.evo_mutation_rate = mutation_rate
        
        return EvolutionBacktestConfig(
            enabled=evo_enabled,
            optimize_dual_engine=optimize_dual_engine,
            optimize_factor_weights=optimize_factor_weights,
            fitness_objective=fitness_objective,
            population_size=population_size,
            max_generations=max_generations,
            window_size_days=window_size_days,
            step_size_days=step_size_days,
            elitism_rate=elitism_rate,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
        )
    
    def _render_evolution_results(
        self, 
        evolution_history: List[Dict[str, Any]],
        result: 'EnhancedBacktestResult',
        baseline_result: Optional['EnhancedBacktestResult'] = None
    ) -> None:
        """渲染演化優化結果 - 詳細版本
        
        Args:
            evolution_history: 演化歷史記錄
            result: 當前回測結果（使用演化優化）
            baseline_result: 基準回測結果（未使用演化優化）
        """
        if not evolution_history:
            return
        
        st.markdown("### 🧬 演化優化分析")
        
        # 使用頁籤組織演化結果
        evo_tabs = st.tabs(["📊 績效對比", "📈 演化曲線", "🧬 參數演化", "📋 詳細數據"])
        
        # 頁籤 1: 績效對比
        with evo_tabs[0]:
            self._render_evolution_comparison(result, baseline_result)
        
        # 頁籤 2: 演化曲線
        with evo_tabs[1]:
            self._render_evolution_fitness_chart(evolution_history)
        
        # 頁籤 3: 參數演化
        with evo_tabs[2]:
            self._render_evolution_params_chart(evolution_history)
        
        # 頁籤 4: 詳細數據
        with evo_tabs[3]:
            self._render_evolution_details(evolution_history)
    
    def _render_evolution_comparison(
        self,
        result: 'EnhancedBacktestResult',
        baseline_result: Optional['EnhancedBacktestResult']
    ) -> None:
        """渲染演化優化前後績效對比
        
        比較「演化優化後的策略」與「使用者手動設定的雙引擎/因子權重策略」
        """
        st.markdown("**🧬 演化優化 vs 📊 手動設定策略**")
        
        # 說明比較的內容
        st.caption("""
        比較說明：
        - **演化優化**：使用演化算法自動調整的雙引擎參數和因子權重
        - **手動設定**：使用者在介面上手動設定的雙引擎參數和因子權重
        """)
        
        if baseline_result is None:
            st.info("💡 未執行基準比較，無法顯示對比數據。請同時啟用雙引擎或因子權重來進行比較。")
            # 只顯示當前結果
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("總報酬率", f"{result.total_return:+.2f}%")
            with col2:
                st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
            with col3:
                st.metric("最大回撤", f"{result.max_drawdown:.2f}%")
            with col4:
                st.metric("勝率", f"{result.win_rate:.1f}%")
            return
        
        # 計算差異
        return_diff = result.total_return - baseline_result.total_return
        sharpe_diff = result.sharpe_ratio - baseline_result.sharpe_ratio
        drawdown_diff = result.max_drawdown - baseline_result.max_drawdown
        winrate_diff = result.win_rate - baseline_result.win_rate
        
        # 顯示對比指標 - 使用兩欄對比格式
        st.markdown("##### 績效指標對比")
        
        col_evo, col_manual = st.columns(2)
        
        with col_evo:
            st.markdown("**🧬 演化優化策略**")
            st.metric("總報酬率", f"{result.total_return:+.2f}%", 
                     delta=f"{return_diff:+.2f}% vs 手動", delta_color="normal")
            st.metric("夏普比率", f"{result.sharpe_ratio:.2f}",
                     delta=f"{sharpe_diff:+.2f} vs 手動", delta_color="normal")
            st.metric("最大回撤", f"{result.max_drawdown:.2f}%",
                     delta=f"{-drawdown_diff:+.2f}% vs 手動", delta_color="inverse")
            st.metric("勝率", f"{result.win_rate:.1f}%",
                     delta=f"{winrate_diff:+.1f}% vs 手動", delta_color="normal")
        
        with col_manual:
            st.markdown("**📊 手動設定策略**")
            st.metric("總報酬率", f"{baseline_result.total_return:+.2f}%")
            st.metric("夏普比率", f"{baseline_result.sharpe_ratio:.2f}")
            st.metric("最大回撤", f"{baseline_result.max_drawdown:.2f}%")
            st.metric("勝率", f"{baseline_result.win_rate:.1f}%")
        
        # 交易統計對比
        st.markdown("##### 交易統計對比")
        
        evo_avg_holding = (sum(t.holding_days for t in result.trades) / len(result.trades) 
                          if result.trades else 0)
        manual_avg_holding = (sum(t.holding_days for t in baseline_result.trades) / len(baseline_result.trades) 
                             if baseline_result.trades else 0)
        
        comparison_data = {
            "指標": ["總交易次數", "獲利交易", "虧損交易", "平均持有天數"],
            "🧬 演化優化": [
                result.total_trades,
                result.winning_trades,
                result.losing_trades,
                f"{evo_avg_holding:.1f}"
            ],
            "📊 手動設定": [
                baseline_result.total_trades,
                baseline_result.winning_trades,
                baseline_result.losing_trades,
                f"{manual_avg_holding:.1f}"
            ],
            "差異": [
                f"{result.total_trades - baseline_result.total_trades:+d}",
                f"{result.winning_trades - baseline_result.winning_trades:+d}",
                f"{result.losing_trades - baseline_result.losing_trades:+d}",
                f"{evo_avg_holding - manual_avg_holding:+.1f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # 績效改善摘要
        st.markdown("##### 演化優化效果評估")
        
        improvements = []
        regressions = []
        
        if return_diff > 0.5:
            improvements.append(f"📈 報酬率提升 {return_diff:.2f}%")
        elif return_diff < -0.5:
            regressions.append(f"📉 報酬率下降 {abs(return_diff):.2f}%")
        
        if sharpe_diff > 0.1:
            improvements.append(f"📈 夏普比率提升 {sharpe_diff:.2f}")
        elif sharpe_diff < -0.1:
            regressions.append(f"📉 夏普比率下降 {abs(sharpe_diff):.2f}")
        
        if drawdown_diff < -0.5:
            improvements.append(f"📈 回撤降低 {abs(drawdown_diff):.2f}%")
        elif drawdown_diff > 0.5:
            regressions.append(f"📉 回撤增加 {drawdown_diff:.2f}%")
        
        if winrate_diff > 1.0:
            improvements.append(f"📈 勝率提升 {winrate_diff:.1f}%")
        elif winrate_diff < -1.0:
            regressions.append(f"📉 勝率下降 {abs(winrate_diff):.1f}%")
        
        # 總結評估
        if len(improvements) > len(regressions):
            st.success(f"✅ **演化優化表現較佳**\n\n改善項目: {', '.join(improvements)}")
            if regressions:
                st.warning(f"⚠️ 需注意: {', '.join(regressions)}")
        elif len(regressions) > len(improvements):
            st.warning(f"⚠️ **手動設定表現較佳**\n\n退步項目: {', '.join(regressions)}")
            if improvements:
                st.info(f"💡 改善項目: {', '.join(improvements)}")
        else:
            if improvements:
                st.info(f"📊 **表現相近**\n\n改善: {', '.join(improvements)}\n需注意: {', '.join(regressions)}")
            else:
                st.info("📊 **績效與手動設定相近，無顯著差異**")
        
        # 建議
        st.markdown("##### 💡 建議")
        if return_diff > 2.0 and sharpe_diff > 0.2:
            st.success("演化優化顯著提升績效，建議採用演化優化的參數設定。")
        elif return_diff < -2.0 or sharpe_diff < -0.2:
            st.warning("手動設定的參數表現較佳，可能需要調整演化優化的目標函數或增加演化代數。")
        else:
            st.info("兩種方法表現相近，可根據實際需求選擇。演化優化可自動適應市場變化。")
    
    def _render_evolution_fitness_chart(self, evolution_history: List[Dict[str, Any]]) -> None:
        """渲染演化適應度曲線"""
        st.markdown("**適應度演化曲線**")
        
        evo_df = pd.DataFrame(evolution_history)
        
        if "fitness" not in evo_df.columns:
            st.info("無適應度數據")
            return
        
        # 適應度曲線
        st.line_chart(
            evo_df.set_index("window")["fitness"],
            use_container_width=True
        )
        
        # 統計摘要
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("初始適應度", f"{evo_df['fitness'].iloc[0]:.4f}")
        with col2:
            st.metric("最終適應度", f"{evo_df['fitness'].iloc[-1]:.4f}")
        with col3:
            improvement = evo_df['fitness'].iloc[-1] - evo_df['fitness'].iloc[0]
            st.metric("適應度變化", f"{improvement:+.4f}")
        with col4:
            st.metric("演化視窗數", len(evo_df))
    
    def _render_evolution_params_chart(self, evolution_history: List[Dict[str, Any]]) -> None:
        """渲染參數演化圖表"""
        st.markdown("**參數演化趨勢**")
        
        evo_df = pd.DataFrame(evolution_history)
        
        # 雙引擎參數演化
        if "trend_threshold" in evo_df.columns:
            st.markdown("**雙引擎參數**")
            
            params_df = evo_df[["window", "trend_threshold", "range_threshold", "trend_allocation"]].copy()
            params_df = params_df.set_index("window")
            params_df.columns = ["趨勢閾值", "震盪閾值", "趨勢權重"]
            
            st.line_chart(params_df, use_container_width=True)
            
            # 參數變化摘要
            col1, col2, col3 = st.columns(3)
            
            with col1:
                initial = evo_df['trend_threshold'].iloc[0]
                final = evo_df['trend_threshold'].iloc[-1]
                st.metric(
                    "趨勢閾值",
                    f"{final:.1f}",
                    delta=f"{final - initial:+.1f}"
                )
            
            with col2:
                initial = evo_df['range_threshold'].iloc[0]
                final = evo_df['range_threshold'].iloc[-1]
                st.metric(
                    "震盪閾值",
                    f"{final:.1f}",
                    delta=f"{final - initial:+.1f}"
                )
            
            with col3:
                initial = evo_df['trend_allocation'].iloc[0]
                final = evo_df['trend_allocation'].iloc[-1]
                st.metric(
                    "趨勢權重",
                    f"{final:.2f}",
                    delta=f"{final - initial:+.2f}"
                )
    
    def _render_evolution_details(self, evolution_history: List[Dict[str, Any]]) -> None:
        """渲染演化詳細數據"""
        st.markdown("**演化視窗詳細數據**")
        
        evo_df = pd.DataFrame(evolution_history)
        
        # 格式化顯示
        display_df = evo_df.copy()
        
        # 重命名欄位
        column_names = {
            "window": "視窗",
            "date": "日期",
            "fitness": "適應度",
            "symbol": "股票",
            "trend_threshold": "趨勢閾值",
            "range_threshold": "震盪閾值",
            "trend_allocation": "趨勢權重",
        }
        
        display_df = display_df.rename(columns=column_names)
        
        # 格式化數值
        if "適應度" in display_df.columns:
            display_df["適應度"] = display_df["適應度"].apply(lambda x: f"{x:.4f}")
        if "趨勢閾值" in display_df.columns:
            display_df["趨勢閾值"] = display_df["趨勢閾值"].apply(lambda x: f"{x:.1f}")
        if "震盪閾值" in display_df.columns:
            display_df["震盪閾值"] = display_df["震盪閾值"].apply(lambda x: f"{x:.1f}")
        if "趨勢權重" in display_df.columns:
            display_df["趨勢權重"] = display_df["趨勢權重"].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # 下載按鈕
        csv = evo_df.to_csv(index=False)
        st.download_button(
            label="📥 下載演化數據 (CSV)",
            data=csv,
            file_name="evolution_history.csv",
            mime="text/csv",
        )
    
    def _render_optimization_tabs(self) -> Tuple[Optional[DualEngineConfig], Optional[EvolutionBacktestConfig], bool]:
        """渲染優化設定頁籤區塊
        
        Returns:
            Tuple of (dual_engine_config, evolution_config, use_optimizer)
        """
        st.subheader("⚙️ 策略優化設定")
        
        # 顯示當前啟用的功能摘要
        enabled_features = []
        if st.session_state.get("dual_engine_enabled", False):
            enabled_features.append("🔄 雙引擎")
        if st.session_state.get("evo_backtest_enabled", False):
            enabled_features.append("🧬 演化優化")
        if st.session_state.get("use_signal_optimizer", False):
            enabled_features.append("🎯 因子權重")
        
        if enabled_features:
            st.success(f"已啟用: {' | '.join(enabled_features)}")
        else:
            st.info("💡 尚未啟用任何優化功能，使用純型態策略")
        
        # 使用頁籤組織三個優化功能
        tab1, tab2, tab3 = st.tabs(["🔄 雙引擎模式", "🧬 演化優化", "🎯 因子權重"])
        
        # 頁籤 1: 雙引擎模式
        with tab1:
            dual_engine_config = self._render_dual_engine_tab()
        
        # 頁籤 2: 演化優化
        with tab2:
            evolution_config = self._render_evolution_tab()
        
        # 頁籤 3: 因子權重
        with tab3:
            use_optimizer = self._render_factor_weight_tab()
        
        return dual_engine_config, evolution_config, use_optimizer
    
    def _render_dual_engine_tab(self) -> Optional[DualEngineConfig]:
        """渲染雙引擎模式頁籤內容"""
        col1, col2 = st.columns([1, 3])
        
        with col1:
            dual_engine_enabled = st.toggle(
                "啟用雙引擎",
                value=st.session_state.get("dual_engine_enabled", False),
                help="根據市場狀態自動切換趨勢/震盪策略",
                key="dual_engine_tab_toggle"
            )
            st.session_state.dual_engine_enabled = dual_engine_enabled
        
        with col2:
            if dual_engine_enabled:
                st.success("✅ 雙引擎模式已啟用")
            else:
                st.caption("未啟用雙引擎模式")
        
        if not dual_engine_enabled:
            st.markdown("""
            **雙引擎模式說明：**
            - 🔄 根據 ADX 指標自動判斷市場狀態（趨勢/震盪/混沌）
            - 📈 趨勢市場：使用突破追蹤策略
            - 📊 震盪市場：使用均值回歸策略
            - ⚠️ 混沌市場：降低倉位或暫停交易
            """)
            return None
        
        # 雙引擎詳細設定
        config_manager = DualEngineConfigManager()
        current_config = config_manager.get_config()
        
        st.markdown("**市場狀態判定參數**")
        col1, col2 = st.columns(2)
        
        with col1:
            adx_trend_threshold = st.slider(
                "ADX 趨勢閾值",
                min_value=20.0,
                max_value=40.0,
                value=st.session_state.get("adx_trend_threshold", current_config.adx_trend_threshold),
                step=1.0,
                help="ADX 高於此值判定為趨勢市場",
                key="adx_trend_tab_slider"
            )
            st.session_state.adx_trend_threshold = adx_trend_threshold
        
        with col2:
            adx_range_threshold = st.slider(
                "ADX 震盪閾值",
                min_value=10.0,
                max_value=25.0,
                value=st.session_state.get("adx_range_threshold", current_config.adx_range_threshold),
                step=1.0,
                help="ADX 低於此值判定為震盪市場",
                key="adx_range_tab_slider"
            )
            st.session_state.adx_range_threshold = adx_range_threshold
        
        st.markdown("**資金分配權重**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_allocation = st.slider(
                "趨勢模式權重",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.get("trend_allocation", current_config.trend_allocation),
                step=0.1,
                format="%.1f",
                key="trend_alloc_tab_slider"
            )
            st.session_state.trend_allocation = trend_allocation
        
        with col2:
            range_allocation = st.slider(
                "震盪模式權重",
                min_value=0.3,
                max_value=0.8,
                value=st.session_state.get("range_allocation", current_config.range_allocation),
                step=0.1,
                format="%.1f",
                key="range_alloc_tab_slider"
            )
            st.session_state.range_allocation = range_allocation
        
        with col3:
            noise_allocation = st.slider(
                "混沌模式權重",
                min_value=0.0,
                max_value=0.3,
                value=st.session_state.get("noise_allocation", current_config.noise_allocation),
                step=0.1,
                format="%.1f",
                key="noise_alloc_tab_slider"
            )
            st.session_state.noise_allocation = noise_allocation
        
        # 建立並返回配置
        config = DualEngineConfig(
            enabled=dual_engine_enabled,
            adx_trend_threshold=adx_trend_threshold,
            adx_range_threshold=adx_range_threshold,
            trend_allocation=trend_allocation,
            range_allocation=range_allocation,
            noise_allocation=noise_allocation,
        )
        
        config_manager.save_config(config)
        st.session_state.dual_engine_config = config
        
        return config
    
    def _render_evolution_tab(self) -> Optional[EvolutionBacktestConfig]:
        """渲染演化優化頁籤內容"""
        if not EVOLUTION_AVAILABLE:
            st.warning("⚠️ 演化優化模組未安裝")
            return None
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            evo_enabled = st.toggle(
                "啟用演化優化",
                value=st.session_state.get("evo_backtest_enabled", False),
                help="在回測過程中自動演化優化參數",
                key="evo_tab_toggle"
            )
            st.session_state.evo_backtest_enabled = evo_enabled
        
        with col2:
            if evo_enabled:
                st.success("✅ 演化優化已啟用")
            else:
                st.caption("未啟用演化優化")
        
        # 預設值
        optimize_dual_engine = True
        optimize_factor_weights = True
        fitness_objective = "sharpe_ratio"
        population_size = 50
        max_generations = 15
        window_size_days = 126
        step_size_days = 21
        elitism_rate = 0.1
        crossover_rate = 0.8
        mutation_rate = 0.02
        
        if not evo_enabled:
            st.markdown("""
            **演化優化說明：**
            - 🧬 使用遺傳演算法自動尋找最佳參數
            - 📊 在每個回測視窗結束後更新參數
            - 🔄 模擬實際交易中的參數自適應調整
            """)
            return None
        
        # 優化目標
        st.markdown("**優化目標**")
        objective_options = {
            "sharpe_ratio": "夏普比率 - 風險調整後收益",
            "sortino_ratio": "索提諾比率 - 下行風險調整",
            "net_profit": "淨利潤 - 最大化收益",
            "min_max_drawdown": "最小化回撤 - 防禦型",
        }
        
        fitness_objective = st.selectbox(
            "選擇優化目標",
            options=list(objective_options.keys()),
            format_func=lambda x: objective_options[x],
            index=0,
            key="evo_fitness_tab"
        )
        
        # 優化範圍
        st.markdown("**優化範圍**")
        col1, col2 = st.columns(2)
        
        with col1:
            optimize_dual_engine = st.checkbox(
                "優化雙引擎參數",
                value=st.session_state.get("evo_optimize_dual_engine", True),
                key="evo_dual_tab_cb"
            )
            st.session_state.evo_optimize_dual_engine = optimize_dual_engine
        
        with col2:
            optimize_factor_weights = st.checkbox(
                "優化因子權重",
                value=st.session_state.get("evo_optimize_factor_weights", True),
                key="evo_factor_tab_cb"
            )
            st.session_state.evo_optimize_factor_weights = optimize_factor_weights
        
        # 演化參數
        st.markdown("**演化參數**")
        col1, col2 = st.columns(2)
        
        with col1:
            population_size = st.slider(
                "種群大小",
                min_value=50,
                max_value=100,
                value=st.session_state.get("evo_population_size", 50),
                step=10,
                key="evo_pop_tab_slider"
            )
            st.session_state.evo_population_size = population_size
            
            window_size_days = st.slider(
                "演化視窗 (天)",
                min_value=63,
                max_value=252,
                value=st.session_state.get("evo_window_size", 126),
                step=21,
                key="evo_window_tab_slider"
            )
            st.session_state.evo_window_size = window_size_days
        
        with col2:
            max_generations = st.slider(
                "最大世代數",
                min_value=10,
                max_value=30,
                value=st.session_state.get("evo_max_generations", 15),
                step=5,
                key="evo_gen_tab_slider"
            )
            st.session_state.evo_max_generations = max_generations
            
            step_size_days = st.slider(
                "步進大小 (天)",
                min_value=5,
                max_value=63,
                value=st.session_state.get("evo_step_size", 21),
                step=7,
                key="evo_step_tab_slider"
            )
            st.session_state.evo_step_size = step_size_days
        
        # 進階參數
        with st.expander("🔧 進階參數", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                elitism_rate = st.slider(
                    "精英保留率",
                    min_value=0.05,
                    max_value=0.20,
                    value=st.session_state.get("evo_elitism_rate", 0.1),
                    step=0.05,
                    format="%.2f",
                    key="evo_elite_tab_slider"
                )
                st.session_state.evo_elitism_rate = elitism_rate
            
            with col2:
                crossover_rate = st.slider(
                    "交叉率",
                    min_value=0.6,
                    max_value=0.9,
                    value=st.session_state.get("evo_crossover_rate", 0.8),
                    step=0.1,
                    format="%.1f",
                    key="evo_cross_tab_slider"
                )
                st.session_state.evo_crossover_rate = crossover_rate
            
            with col3:
                mutation_rate = st.slider(
                    "突變率",
                    min_value=0.01,
                    max_value=0.05,
                    value=st.session_state.get("evo_mutation_rate", 0.02),
                    step=0.01,
                    format="%.2f",
                    key="evo_mut_tab_slider"
                )
                st.session_state.evo_mutation_rate = mutation_rate
        
        return EvolutionBacktestConfig(
            enabled=evo_enabled,
            optimize_dual_engine=optimize_dual_engine,
            optimize_factor_weights=optimize_factor_weights,
            fitness_objective=fitness_objective,
            population_size=population_size,
            max_generations=max_generations,
            window_size_days=window_size_days,
            step_size_days=step_size_days,
            elitism_rate=elitism_rate,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
        )
    
    def _render_factor_weight_tab(self) -> bool:
        """渲染因子權重頁籤內容"""
        col1, col2 = st.columns([1, 3])
        
        with col1:
            use_optimizer = st.toggle(
                "啟用因子權重",
                value=st.session_state.get("use_signal_optimizer", False),
                help="使用技術指標對訊號進行加權評分",
                key="factor_tab_toggle"
            )
            st.session_state.use_signal_optimizer = use_optimizer
        
        with col2:
            if use_optimizer:
                st.success("✅ 因子權重優化已啟用")
            else:
                st.caption("未啟用因子權重優化")
        
        if not use_optimizer:
            st.markdown("""
            **因子權重說明：**
            - 🎯 使用 RSI、MACD、成交量等技術指標
            - ⚖️ 對型態識別訊號進行加權評分
            - 📊 過濾弱訊號，強化強訊號
            """)
            return False
        
        # 因子權重實驗室入口
        st.markdown("**因子權重實驗室**")
        st.markdown("""
        在因子權重實驗室中，您可以：
        - 調整各技術指標的啟用狀態與權重
        - 細緻調整 RSI 等指標的評分參數
        - 使用自動調參功能尋找最佳指標組合
        """)
        
        if st.button("🚀 開啟因子權重實驗室", type="secondary", use_container_width=True, key="factor_lab_btn"):
            st.session_state.show_factor_lab = True
            st.rerun()
        
        return use_optimizer
    
    def _fetch_realtime_data(self, symbols: List[str], interval: str = "5m") -> Dict[str, Any]:
        """抓取即時數據"""
        import yfinance as yf
        
        data = {}
        # 轉換 interval 格式 (1分鐘 -> 1m)
        yf_interval = "5m"
        if interval == 60: yf_interval = "1m"
        elif interval == 300: yf_interval = "5m"
        elif interval == 900: yf_interval = "15m"
        elif interval == 3600: yf_interval = "60m"
        
        # 批量抓取，獲取最近 5 天數據以確保有足夠歷史計算指標
        try:
            tickers = yf.download(symbols, period="5d", interval=yf_interval, group_by='ticker', progress=False)
            
            for symbol in symbols:
                if len(symbols) == 1:
                    df = tickers
                else:
                    df = tickers[symbol]
                
                if not df.empty:
                    # 轉換為 OHLCV 格式
                    ohlcv_list = []
                    for idx, row in df.iterrows():
                        ohlcv = OHLCV(
                            time=idx.to_pydatetime(),  # Use 'time' instead of 'date'
                            symbol=symbol,  # Add missing 'symbol' argument
                            open=float(row['Open']) if not pd.isna(row['Open']) else 0.0,
                            high=float(row['High']) if not pd.isna(row['High']) else 0.0,
                            low=float(row['Low']) if not pd.isna(row['Low']) else 0.0,
                            close=float(row['Close']) if not pd.isna(row['Close']) else 0.0,
                            volume=int(row['Volume']) if not pd.isna(row['Volume']) else 0
                        )
                        ohlcv_list.append(ohlcv)
                    data[symbol] = ohlcv_list
        except Exception as e:
            st.error(f"數據抓取失敗: {str(e)}")
            
        return data

    def _run_live_simulation(self, parameters: StrategyParameters, symbols: List[str], update_interval: int):
        """執行實時模擬交易"""
        import time
        from pattern_quant.ui.strategy_lab_enhanced import OHLCV  # 確保導入
        
        st.subheader("🔴 實時模擬交易中 (Live Paper Trading)")
        
        # 顯示當前優化參數 (若啟用)
        evo_config = st.session_state.get("live_evolution_config")
        if evo_config and evo_config.enabled:
            st.markdown("---")
            st.markdown("### 🧬 當前演化優化參數")
            
            # 使用列表或表格顯示當前雙引擎配置與因子權重
            dual_config = self.backtest_engine.dual_engine_config
            if dual_config:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("趨勢閾值", f"{dual_config.adx_trend_threshold:.1f}")
                with c2:
                    st.metric("震盪閾值", f"{dual_config.adx_range_threshold:.1f}")
                with c3:
                    st.metric("趨勢分配", f"{dual_config.trend_allocation:.1f}")
            
            # 如果有演化歷史，顯示最近一次更新
            if self.backtest_engine._evolution_history:
                last_evo = self.backtest_engine._evolution_history[-1]
                st.info(f"💡 最近演化更新: {last_evo['date']} | 適應度: {last_evo['fitness']:.4f}")
            st.markdown("---")
            
        # 建立 UI 容器
        containers = {
            'status': st.empty(),
            'metrics': st.empty(),
            'positions': st.empty(),
            'signals': st.empty(),
            'log': st.empty()
        }
        
        # ==================== 持久化狀態管理 ====================
        state_manager = get_state_manager()
        
        # 檢查是否有活躍的模擬可以恢復
        if 'live_sim_id' not in st.session_state:
            # 檢查是否有正在運行的模擬
            active_sims = state_manager.get_active_simulations()
            matching_sim = None
            
            for sim in active_sims:
                # 找到匹配當前標的的模擬
                if set(sim.symbols) == set(symbols) and sim.is_alive:
                    matching_sim = sim
                    break
            
            if matching_sim:
                # 恢復現有模擬
                st.session_state.live_sim_id = matching_sim.id
                saved_state = state_manager.load_state(matching_sim.id)
                
                if saved_state:
                    st.session_state.live_sim_capital = saved_state.capital
                    st.session_state.live_sim_positions = saved_state.positions
                    st.session_state.live_sim_trades = saved_state.trades
                    st.session_state.live_sim_logs = saved_state.logs
                    st.session_state.live_sim_last_update = datetime.strptime(
                        saved_state.updated_at, '%Y-%m-%d %H:%M:%S'
                    )
                    st.toast(f"✅ 已恢復模擬: {matching_sim.name}", icon="🔄")
                else:
                    # 狀態載入失敗，使用預設值
                    st.session_state.live_sim_capital = self.backtest_engine.initial_capital
                    st.session_state.live_sim_positions = {}
                    st.session_state.live_sim_trades = []
                    st.session_state.live_sim_logs = []
                    st.session_state.live_sim_last_update = datetime.now()
            else:
                # 建立新模擬
                sim_name = f"模擬_{datetime.now().strftime('%m%d_%H%M')}"
                params_dict = {
                    'min_depth': parameters.min_depth,
                    'max_depth': parameters.max_depth,
                    'stop_loss_ratio': parameters.stop_loss_ratio,
                    'profit_threshold': parameters.profit_threshold,
                    'score_threshold': parameters.score_threshold,
                }
                
                sim_id = state_manager.create_simulation(
                    name=sim_name,
                    symbols=symbols,
                    parameters=params_dict,
                    update_interval=update_interval,
                    initial_capital=self.backtest_engine.initial_capital
                )
                
                if sim_id:
                    st.session_state.live_sim_id = sim_id
                    st.toast(f"✅ 已建立新模擬: {sim_name}", icon="🆕")
                
                # 初始化狀態
                st.session_state.live_sim_capital = self.backtest_engine.initial_capital
                st.session_state.live_sim_positions = {}
                st.session_state.live_sim_trades = []
                st.session_state.live_sim_logs = []
                st.session_state.live_sim_last_update = datetime.now()
        
        capital = st.session_state.live_sim_capital
        positions = st.session_state.live_sim_positions
        trades = st.session_state.live_sim_trades
        logs = st.session_state.live_sim_logs
        
        # 模擬一次循環
        current_time = datetime.now()
        last_update = st.session_state.live_sim_last_update
        
        # 檢查是否需要執行更新 (如果距離上次更新不到 interval 秒，則等待或直接渲染)
        seconds_since_update = (current_time - last_update).total_seconds()
        
        if seconds_since_update >= update_interval or not logs:
            # 更新狀態顯示
            with containers['status']:
                st.info(f"⚡ 正在抓取數據 | 更新頻率: {update_interval}秒")
            
            # 抓取數據
            realtime_data = self._fetch_realtime_data(symbols, update_interval)
            
            # --- 演化優化檢查點 ---
            if evo_config and evo_config.enabled:
                if 'live_sim_evo_count' not in st.session_state:
                    st.session_state.live_sim_evo_count = 0
                
                # 在實時模擬中，我們可以縮短演化週期，或者根據數據量觸發
                # 這裡假設每收集 5 個更新週期嘗試一次小型演化 (僅為示範，實際應根據配置)
                st.session_state.live_sim_evo_count += 1
                if st.session_state.live_sim_evo_count >= 5:
                    st.session_state.live_sim_evo_count = 0
                    with containers['status']:
                        st.info("🧬 正在執行實時演化優化...")
                    
                    # 使用收集到的數據執行演化
                    for sym in symbols:
                        if sym in realtime_data:
                            data = realtime_data[sym]
                            if len(data) >= 60: # 至少需要一些基礎數據
                                prices = [d.close for d in data]
                                highs = [d.high for d in data]
                                lows = [d.low for d in data]
                                volumes = [float(d.volume) for d in data]
                                
                                evo_res = self.backtest_engine._run_evolution_window(
                                    symbol=sym,
                                    prices=prices,
                                    highs=highs,
                                    lows=lows,
                                    volumes=volumes,
                                    window_idx=len(self.backtest_engine._evolution_history),
                                    progress_callback=None
                                )
                                
                                if evo_res:
                                    best_genome, fitness = evo_res
                                    # 應用最新最優基因組
                                    new_dual, new_factor = self.backtest_engine._apply_genome_to_configs(best_genome)
                                    if new_dual and evo_config.optimize_dual_engine:
                                        self.backtest_engine.dual_engine_config = new_dual
                                        self.backtest_engine._dual_engine_strategy = None
                                        logs.append(f"[{current_time.strftime('%H:%M:%S')}] 🧬 雙引擎參數已優化: Trend={new_dual.adx_trend_threshold:.1f}")
            
            # 分析每個標的
            active_signals = []
            
            for symbol in symbols:
                if symbol not in realtime_data:
                    continue
                    
                ohlcv_data = realtime_data[symbol]
                if len(ohlcv_data) < 20: continue
                
                current_price = ohlcv_data[-1].close
                analysis_date = ohlcv_data[-1].time
                
                # 1. 檢查持倉出場
                if symbol in positions:
                    pos = positions[symbol]
                    # 更新現價
                    pos['current_price'] = current_price
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                    
                    # 出場邏輯 (止損比例放在 parameters 中)
                    exit_signal = False
                    reason = ""
                    
                    # 止損
                    if pnl_pct <= -parameters.stop_loss_ratio:
                        exit_signal = True
                        reason = f"止損 ({pnl_pct:.1f}%)"
                    elif pnl_pct >= parameters.profit_threshold:
                        # 簡易止盈
                        exit_signal = True
                        reason = f"止盈 ({pnl_pct:.1f}%)"
                        
                    if exit_signal:
                        # 平倉
                        pnl_val = (current_price - pos['entry_price']) * pos['shares']
                        capital += current_price * pos['shares']
                        del positions[symbol]
                        
                        trade = {
                            'symbol': symbol,
                            'type': 'SELL',
                            'price': current_price,
                            'time': current_time,
                            'reason': reason,
                            'pnl': pnl_val,
                            'pnl_pct': pnl_pct
                        }
                        trades.append(trade)
                        logs.append(f"[{current_time.strftime('%H:%M:%S')}] 🔴 平倉 {symbol} @ {current_price} ({reason})")
                
                # 2. 檢查進場 (若無持倉)
                else:
                    # 使用引擎的型態搜尋
                    pattern = self.backtest_engine._find_simple_patterns(ohlcv_data, analysis_date, params=parameters)
                    
                    if pattern and pattern.is_valid and pattern.score:
                        score_val = pattern.score.total_score
                        if score_val >= parameters.score_threshold:
                            # 記錄訊號
                            active_signals.append({
                                'symbol': symbol,
                                'pattern': pattern.pattern_type if hasattr(pattern, 'pattern_type') else "Cup-Handle",
                                'score': score_val,
                                'price': current_price
                            })
                            
                            # 進場
                            pos_ratio = parameters.position_size / 100
                            available = capital * pos_ratio
                            shares = int(available / current_price)
                            if shares > 0:
                                positions[symbol] = {
                                    'symbol': symbol,
                                    'entry_price': current_price,
                                    'entry_date': analysis_date,
                                    'shares': shares,
                                    'current_price': current_price,
                                    'pattern': "Cup-Handle"
                                }
                                capital -= current_price * shares
                                trades.append({
                                    'symbol': symbol,
                                    'type': 'BUY',
                                    'price': current_price,
                                    'time': current_time, 
                                    'reason': f"型態進場 (分:{score_val:.1f})"
                                })
                                logs.append(f"[{current_time.strftime('%H:%M:%S')}] 🟢 進場 {symbol} @ {current_price} (Score: {score_val:.1f})")
            
            # 更新 session state
            st.session_state.live_sim_capital = capital
            st.session_state.live_sim_positions = positions
            st.session_state.live_sim_trades = trades
            st.session_state.live_sim_logs = logs
            st.session_state.live_sim_last_update = current_time
            st.session_state.live_sim_active_signals = active_signals
            
            # ==================== 持久化儲存 ====================
            sim_id = st.session_state.get('live_sim_id')
            if sim_id:
                # 將 datetime 轉換為字串以便 JSON 序列化
                serializable_trades = []
                for t in trades:
                    trade_copy = dict(t)
                    if 'time' in trade_copy and isinstance(trade_copy['time'], datetime):
                        trade_copy['time'] = trade_copy['time'].strftime('%Y-%m-%d %H:%M:%S')
                    serializable_trades.append(trade_copy)
                
                serializable_positions = {}
                for sym, pos in positions.items():
                    pos_copy = dict(pos)
                    if 'entry_date' in pos_copy and isinstance(pos_copy['entry_date'], datetime):
                        pos_copy['entry_date'] = pos_copy['entry_date'].strftime('%Y-%m-%d %H:%M:%S')
                    serializable_positions[sym] = pos_copy
                
                state_manager.save_state(
                    simulation_id=sim_id,
                    capital=capital,
                    positions=serializable_positions,
                    trades=serializable_trades,
                    logs=logs[-100:],  # 只保留最近 100 條日誌
                    active_signals=active_signals,
                    evolution_history=self.backtest_engine._evolution_history if hasattr(self.backtest_engine, '_evolution_history') else []
                )
        else:
            active_signals = st.session_state.get('live_sim_active_signals', [])
        
        # ==================== 每次刷新都更新心跳 ====================
        sim_id = st.session_state.get('live_sim_id')
        if sim_id:
            state_manager.update_heartbeat(sim_id)


        # --- 渲染 UI ---
        # 更新狀態顯示
        with containers['status']:
            next_update = update_interval - seconds_since_update
            st.info(f"⚡ 系統運作中 | 當前時間: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | 更新頻率: {update_interval}秒")
            
        # 更新 UI (Metrics)
        return_pct = ((capital - self.backtest_engine.initial_capital) / self.backtest_engine.initial_capital) * 100
        with containers['metrics']:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("即時資金", f"${capital:,.0f}", f"{return_pct:+.2f}%")
            m2.metric("持倉分佈", f"{len(positions)} 檔")
            m3.metric("總交易數", len(trades))
            m4.metric("監控標的", f"{len(symbols)} 檔")
        
        # 更新持倉
        with containers['positions']:
            if positions:
                st.markdown("### 📋 當前持倉")
                pos_list = []
                for s, p in positions.items():
                    cp = p.get('current_price', p['entry_price'])
                    pp = (cp - p['entry_price']) / p['entry_price'] * 100
                    pos_list.append({
                        "標的": s,
                        "進場價": f"${p['entry_price']:.2f}",
                        "現價": f"${cp:.2f}",
                        "損益%": f"{pp:+.2f}%",
                        "持有股數": p['shares'],
                        "進場日": p['entry_date'].strftime('%Y-%m-%d')
                    })
                st.dataframe(pos_list, use_container_width=True, hide_index=True)
            else:
                st.info("目前無持倉")
        
        # 更新訊號與日誌
        col_sig, col_log = st.columns([1, 1])
        with col_sig:
            st.markdown("### 📡 偵測到訊號")
            if active_signals:
                st.dataframe(active_signals, use_container_width=True, hide_index=True)
            else:
                st.caption("暫無即時訊號")
        
        with col_log:
            st.markdown("### 📝 交易日誌")
            if logs:
                for log in reversed(logs[-10:]):
                    st.text(log)
            else:
                st.caption("暫無日誌記錄")
        
        # 控制循環
        if st.session_state.get("live_sim_active", False):
            time.sleep(1)  # 每 1 秒檢查一次 UI 刷新，但數據更新由 update_interval 控制
            st.rerun()

    def _render_simulation_management(self):
        """渲染模擬管理面板 - 查看與刪除已儲存的模擬狀態"""
        state_manager = get_state_manager()
        all_sims = state_manager.get_all_simulations()
        
        if not all_sims:
            st.caption("目前沒有已儲存的模擬記錄")
            return
        
        st.markdown("### 📊 已儲存的模擬")
        
        # 顯示資料庫大小
        db_size = state_manager.get_db_size()
        db_size_kb = db_size / 1024
        st.caption(f"💾 狀態資料庫大小: {db_size_kb:.1f} KB")
        
        # 模擬列表
        for sim in all_sims:
            status_emoji = "🟢" if sim.is_alive else ("⏸️" if sim.status == "paused" else "⏹️")
            status_text = "運行中" if sim.is_alive else ("已暫停" if sim.status == "paused" else "已停止")
            
            with st.expander(f"{status_emoji} {sim.name} ({status_text})", expanded=sim.is_alive):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**標的**: {', '.join(sim.symbols)}")
                    st.markdown(f"**建立時間**: {sim.created_at}")
                
                with col2:
                    st.markdown(f"**更新間隔**: {sim.update_interval} 秒")
                    st.markdown(f"**最後心跳**: {sim.last_heartbeat}")
                
                with col3:
                    # 載入狀態查看資金
                    state = state_manager.load_state(sim.id)
                    if state:
                        return_pct = ((state.capital - sim.initial_capital) / sim.initial_capital) * 100
                        st.metric("當前資金", f"${state.capital:,.0f}", f"{return_pct:+.2f}%")
                
                # 操作按鈕
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                
                with btn_col1:
                    if sim.is_alive:
                        if st.button("⏹️ 停止", key=f"stop_{sim.id}", use_container_width=True):
                            state_manager.stop_simulation(sim.id)
                            st.rerun()
                    else:
                        if st.button("🔄 恢復", key=f"resume_{sim.id}", use_container_width=True):
                            st.session_state.live_sim_id = sim.id
                            st.session_state.live_sim_active = True
                            st.rerun()
                
                with btn_col2:
                    if not sim.is_alive:
                        if st.button("🗑️ 刪除", key=f"delete_{sim.id}", type="secondary", use_container_width=True):
                            state_manager.delete_simulation(sim.id)
                            st.toast(f"已刪除模擬: {sim.name}", icon="🗑️")
                            st.rerun()
        
        st.divider()
        
        # 批量清理按鈕
        col1, col2 = st.columns(2)
        with col1:
            days = st.number_input("清理幾天前的已停止模擬", min_value=1, max_value=30, value=7, key="cleanup_days")
        with col2:
            if st.button("🧹 清理舊模擬", use_container_width=True):
                count = state_manager.delete_old_simulations(days=days)
                state_manager.vacuum()  # 壓縮資料庫
                st.toast(f"已清理 {count} 個舊模擬", icon="🧹")
                st.rerun()

    def _create_progressive_containers(self):
        """建立逐日模擬的 UI 容器"""
        containers = {
            'header': st.empty(),
            'metrics': st.empty(),
            'positions': st.empty(),
            'trades': st.empty(),
            'chart': st.empty()
        }
        return containers
    
    def _update_progressive_display(
        self, 
        containers: dict,
        current_date: datetime,
        capital: float,
        positions: dict,
        today_trades: list,
        equity_point: dict,
        initial_capital: float
    ):
        """更新逐日模擬顯示"""
        import time
        
        # 計算報酬率
        return_pct = ((capital - initial_capital) / initial_capital) * 100
        
        # 更新日期標題
        with containers['header']:
            st.markdown(f"### 📅 模擬日期：{current_date.strftime('%Y-%m-%d')}")
        
        # 更新指標
        with containers['metrics']:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 當前資金", f"${capital:,.0f}", delta=f"{return_pct:+.2f}%")
            with col2:
                st.metric("📊 持倉數量", len(positions))
            with col3:
                today_new = len([t for t in today_trades if hasattr(t, 'entry_date') and t.entry_date.date() == current_date.date()])
                st.metric("🔄 今日進場", today_new)
            with col4:
                today_closed = len([t for t in today_trades if hasattr(t, 'exit_date') and t.exit_date.date() == current_date.date()])
                st.metric("✅ 今日平倉", today_closed)
        
        # 更新持倉表
        with containers['positions']:
            if positions:
                st.markdown("**📋 當前持倉**")
                pos_data = []
                for sym, pos in positions.items():
                    if pos:
                        pnl_pct = ((pos.get('current_price', 0) - pos['entry_price']) / pos['entry_price'] * 100) if pos['entry_price'] > 0 else 0
                        pos_data.append({
                            "標的": sym,
                            "進場日": pos['entry_date'].strftime('%Y-%m-%d'),
                            "進場價": f"${pos['entry_price']:.2f}",
                            "現價": f"${pos.get('current_price', 0):.2f}",
                            "損益": f"{pnl_pct:+.2f}%",
                            "持有天數": (current_date - pos['entry_date']).days
                        })
                if pos_data:
                    st.dataframe(pos_data, use_container_width=True, hide_index=True)
            else:
                st.markdown("**📋 當前持倉**：無")
        
        # 更新今日交易
        with containers['trades']:
            if today_trades:
                st.markdown("**🔄 今日交易**")
                for trade in today_trades[-3:]:  # 只顯示最近 3 筆
                    if hasattr(trade, 'exit_date') and trade.exit_date.date() == current_date.date():
                        emoji = "🟢" if trade.pnl > 0 else "🔴"
                        st.markdown(f"{emoji} 平倉 **{trade.symbol}** @ ${trade.exit_price:.2f} (損益: {trade.pnl_pct:+.2f}%)")
                    elif hasattr(trade, 'entry_date') and trade.entry_date.date() == current_date.date():
                        st.markdown(f"🔵 進場 **{trade.symbol}** @ ${trade.entry_price:.2f}")
        
        # 根據速度設定延遲
        sim_speed = st.session_state.get('sim_speed', 0.2)
        if sim_speed > 0:
            time.sleep(sim_speed)
    
    def render(self) -> None:
        """渲染完整的策略實驗室頁面"""
        st.header("🧪 策略實驗室（增強版）")
        st.markdown("使用真實股票數據進行回測，每筆交易都有詳細的型態分析與圖表說明。")
        
        # 如果選擇開啟因子權重實驗室，渲染它
        if st.session_state.get("show_factor_lab", False):
            self._render_embedded_factor_lab()
            if st.button("← 返回策略實驗室"):
                st.session_state.show_factor_lab = False
                st.rerun()
            return
        
        # ==================== 主視圖切換 ====================
        view_mode = st.radio(
            "選擇視圖",
            ["📊 策略設定", "📂 模擬管理"],
            horizontal=True,
            key="main_view_mode",
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # 模擬管理視圖
        if view_mode == "📂 模擬管理":
            self._render_simulation_management()
            return
        
        # ==================== 策略設定視圖 ====================
        # 使用頁籤組織優化設定
        dual_engine_config, evolution_config, use_optimizer = self._render_optimization_tabs()
        
        st.divider()
        
        # 參數設定
        params, sweep_config = self.render_parameter_sliders()
        
        st.divider()
        
        # 股票選擇
        symbols, portfolio_allocations = self.render_stock_selection()
        
        # 將倉位配置存入 session state 供回測使用
        st.session_state.portfolio_allocations = portfolio_allocations
        
        st.divider()
        
        # 回測控制
        backtest_dates = self.render_backtest_controls()
        
        # 實時模擬模式
        if backtest_dates == "LIVE_SIMULATION" or st.session_state.get("live_sim_active", False):
            if not symbols:
                st.error("❌ 請至少選擇一支股票")
                st.session_state.live_sim_active = False
                return
            
            # 確保標的在 session state 中
            st.session_state.live_symbols = symbols
            st.session_state.live_params = params
            st.session_state.live_evolution_config = evolution_config
            st.session_state.live_dual_config = dual_engine_config
            st.session_state.use_optimizer = use_optimizer
            
            update_interval = st.session_state.get("live_interval", 300)
            self._run_live_simulation(params, symbols, update_interval)
            return
        
        # 基準比較選項
        enable_comparison = False
        if dual_engine_config or use_optimizer or evolution_config:
            st.markdown("---")
            st.markdown("**📊 差異比較設定**")
            enable_comparison = st.checkbox(
                "啟用基準比較",
                value=st.session_state.get("enable_comparison", True),
                help="執行兩次回測：一次使用純型態策略作為基準，一次使用當前設定，然後比較差異",
                key="enable_comparison_checkbox"
            )
            st.session_state.enable_comparison = enable_comparison
            
            if enable_comparison:
                st.info("💡 系統將先執行純型態策略回測作為基準，再執行當前策略回測，最後顯示差異比較報告")
        
        # 執行回測
        if backtest_dates:
            start_date, end_date = backtest_dates
            
            if not symbols:
                st.error("❌ 請至少選擇一支股票")
                return
            
            # 調試信息
            st.info(f"🔍 開始回測 - 演化優化: {evolution_config is not None and evolution_config.enabled if evolution_config else False}, 雙引擎: {dual_engine_config is not None and dual_engine_config.enabled if dual_engine_config else False}")
            
            # 檢查是否執行參數掃描
            if sweep_config and sweep_config.get("enabled"):
                try:
                    results_df = self._run_parameter_sweep(
                        base_params=params,
                        sweep_config=sweep_config,
                        start_date=start_date,
                        end_date=end_date,
                        symbols=symbols,
                        portfolio_allocations=portfolio_allocations
                    )
                    
                    self._render_sweep_results(results_df, sweep_config)
                    return
                except Exception as e:
                    st.error(f"參數掃描執行失敗: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
            
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(message: str, progress: float):
                    status_text.text(message)
                    progress_bar.progress(min(progress, 1.0))
                
                baseline_result = None
                
                # 如果啟用比較，先執行基準回測
                if enable_comparison and (dual_engine_config or use_optimizer or evolution_config):
                    status_text.text("正在執行基準回測（純型態策略）...")
                    progress_bar.progress(0.1)
                    
                    # 暫時關閉優化器和雙引擎執行基準回測
                    original_use_optimizer = self.backtest_engine.use_signal_optimizer
                    original_dual_engine_config = self.backtest_engine.dual_engine_config
                    original_dual_engine_strategy = self.backtest_engine._dual_engine_strategy
                    original_evolution_config = self.backtest_engine.evolution_config
                    
                    self.backtest_engine.use_signal_optimizer = False
                    self.backtest_engine.dual_engine_config = None
                    self.backtest_engine._dual_engine_strategy = None
                    self.backtest_engine.evolution_config = None
                    
                    with st.spinner("正在執行基準回測..."):
                        baseline_result = self.backtest_engine.run_backtest(
                            parameters=params,
                            start_date=start_date,
                            end_date=end_date,
                            symbols=symbols,
                            progress_callback=lambda msg, prog: update_progress(f"[基準] {msg}", prog * 0.3),
                            portfolio_allocations=portfolio_allocations
                        )
                    
                    # 恢復優化器和雙引擎設定
                    self.backtest_engine.use_signal_optimizer = original_use_optimizer
                    self.backtest_engine.dual_engine_config = original_dual_engine_config
                    self.backtest_engine._dual_engine_strategy = original_dual_engine_strategy
                    self.backtest_engine.evolution_config = original_evolution_config
                    
                    # 儲存基準結果
                    st.session_state.baseline_backtest = baseline_result
                
                # 更新回測引擎的優化器設定
                self.backtest_engine.use_signal_optimizer = use_optimizer
                
                # 更新回測引擎的雙引擎配置
                self.backtest_engine.dual_engine_config = dual_engine_config
                self.backtest_engine._dual_engine_strategy = None  # 重置策略實例以使用新配置
                
                # 更新回測引擎的演化優化配置
                self.backtest_engine.evolution_config = evolution_config
                self.backtest_engine._evolution_engine = None  # 重置演化引擎以使用新配置
                
                status_text.text("正在執行當前策略回測...")
                progress_bar.progress(0.4)
                
                # 如果啟用演化優化，顯示額外提示
                if evolution_config and evolution_config.enabled:
                    st.info("🧬 演化優化已啟用，系統將在回測過程中自動調整參數...")
                
                # 準備逐日模擬回調（如果啟用）
                progressive_mode = st.session_state.get("progressive_mode", False)
                progressive_callback = None
                progressive_containers = None
                
                if progressive_mode:
                    st.divider()
                    st.subheader("🎬 逐日模擬進行中...")
                    progressive_containers = self._create_progressive_containers()
                    initial_capital = self.backtest_engine.initial_capital
                    
                    def progressive_callback(current_date, capital, positions, today_trades, equity_point):
                        self._update_progressive_display(
                            progressive_containers,
                            current_date,
                            capital,
                            positions,
                            today_trades,
                            equity_point,
                            initial_capital
                        )
                
                with st.spinner("正在執行回測..." if not progressive_mode else ""):
                    # 執行當前參數的回測
                    result = self.backtest_engine.run_backtest(
                        parameters=params,
                        start_date=start_date,
                        end_date=end_date,
                        symbols=symbols,
                        progress_callback=lambda msg, prog: update_progress(f"[當前] {msg}", 0.4 + prog * 0.5) if baseline_result else update_progress,
                        portfolio_allocations=portfolio_allocations,
                        progressive_callback=progressive_callback
                    )
                
                # 清空逐日模擬顯示
                if progressive_containers:
                    for container in progressive_containers.values():
                        container.empty()
                    st.success("✅ 逐日模擬完成！")
                
                progress_bar.empty()
                status_text.empty()
                
                st.divider()
                self.render_backtest_results(result)
                
                # 如果有演化歷史，顯示演化結果
                if result.evolution_history:
                    baseline_result = st.session_state.get('baseline_backtest')
                    self._render_evolution_results(result.evolution_history, result, baseline_result)
                
                # 儲存結果
                st.session_state.last_enhanced_backtest = result
            
            except ValueError as e:
                # 處理找不到股票等錯誤
                st.error(f"❌ 回測失敗：{str(e)}")
                import traceback
                with st.expander("查看詳細錯誤"):
                    st.code(traceback.format_exc())
            except Exception as e:
                # 處理其他未預期的錯誤
                st.error(f"❌ 回測發生錯誤：{str(e)}")
                import traceback
                with st.expander("查看詳細錯誤"):
                    st.code(traceback.format_exc())
        
        # 顯示上次結果
        elif 'last_enhanced_backtest' in st.session_state:
            st.divider()
            st.info("📌 顯示上次回測結果")
            self.render_backtest_results(st.session_state.last_enhanced_backtest)
            
            # 如果有演化歷史，顯示演化結果
            if st.session_state.last_enhanced_backtest.evolution_history:
                self._render_evolution_results(
                    st.session_state.last_enhanced_backtest.evolution_history,
                    st.session_state.last_enhanced_backtest,
                    st.session_state.get('baseline_backtest')
                )



    def _run_live_simulation(self, params: StrategyParameters, symbols: List[str], update_interval: int = 300):
        """啟動即時模擬 (Background Mode)"""
        runner = get_simulation_runner()
        manager = get_state_manager()
        
        # 1. 確保 Runner 啟動
        if not runner.is_running():
            runner.start()
            
        # 2. 創建模擬記錄
        import json
        name = f"Live Sim {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # 序列化參數
        # 這裡簡化處理，實際可能需要更完整的序列化
        params_dict = params.__dict__.copy() if hasattr(params, '__dict__') else {}
        # 移除不可序列化的對象
        if 'portfolio_allocations' in params_dict and params_dict['portfolio_allocations']:
             params_dict['portfolio_allocations'] = [
                 {'symbol': a.symbol, 'weight': a.weight} for a in params_dict['portfolio_allocations']
             ]
        if 'mixed_portfolio' in params_dict and params_dict['mixed_portfolio']:
             # create simple dict for mixed portfolio
             mp = params_dict['mixed_portfolio']
             allocs = [{'symbol': a.symbol, 'weight': a.weight} for a in mp.allocations]
             params_dict['mixed_portfolio'] = {'allocations': allocs}
        
        sim_id = manager.create_simulation(
            name=name,
            symbols=symbols,
            parameters=params_dict,
            update_interval=update_interval
        )
        
        if sim_id:
            st.success(f"✅ 模擬 '{name}' 已啟動！(ID: {sim_id})")
            st.info("模擬將在後台持續運行，您可以關閉瀏覽器。請至「📂 模擬管理」查看狀態。")
            st.session_state.live_sim_active = False # Reset UI flag
        else:
            st.error("❌ 無法創建模擬，請查看日誌。")

    def _render_simulation_management(self):
        """渲染模擬管理介面"""
        st.subheader("📂 模擬管理")
        
        runner = get_simulation_runner()
        manager = get_state_manager()
        
        # Runner 狀態 with more info
        status_col1, status_col2, status_col3 = st.columns([2, 1, 1])
        with status_col1:
            st.markdown(f"**後台服務狀態:** {'🟢 運行中' if runner.is_running() else '🔴 已停止'}")
        with status_col2:
            if runner.is_running():
                if st.button("停止服務"):
                    runner.stop()
                    st.rerun()
            else:
                if st.button("啟動服務"):
                    runner.start()
                    st.rerun()
        with status_col3:
            if st.button("🔄 重新整理"):
                st.rerun()
        
        # 顯示當前時間作為參考
        st.caption(f"頁面時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.divider()
        
        # 模擬列表
        sims = manager.get_all_simulations()
        if not sims:
            st.info("目前沒有模擬記錄。請至「策略設定」啟動新的模擬。")
            return
        
        # 顯示所有模擬的摘要表格
        st.markdown("### 模擬列表")
        sim_table = []
        for s in sims:
            sim_table.append({
                "ID": s.id,
                "名稱": s.name,
                "狀態": s.status,
                "標的": ", ".join(s.symbols[:3]) + ("..." if len(s.symbols) > 3 else ""),
                "最後心跳": s.last_heartbeat,
                "存活": "✅" if s.is_alive else "❌",
                "創建時間": s.created_at
            })
        st.dataframe(sim_table, use_container_width=True)
            
        # 選擇模擬 - 顯示 ID, Name, Status, Last Heartbeat
        sim_options = {f"{s.id}: {s.name} ({s.status})": s.id for s in sims}
        selected_sim_label = st.selectbox("選擇模擬查看詳情", options=list(sim_options.keys()))
        
        if selected_sim_label:
            sim_id = sim_options[selected_sim_label]
            self._render_live_simulation_view(sim_id)

    def _render_live_simulation_view(self, sim_id: int):
        """渲染單一模擬的詳細視圖 (Read-Only)"""
        manager = get_state_manager()
        sim_info = manager.get_simulation(sim_id)
        state = manager.load_state(sim_id)
        
        if not sim_info:
            st.error("找不到模擬記錄")
            return

        # 控制按鈕
        col1, col2, col3 = st.columns(3)
        with col1:
            if sim_info.status == "running":
                if st.button("⏸️ 暫停模擬"):
                    manager.update_simulation_status(sim_id, "paused")
                    st.rerun()
            elif sim_info.status == "paused":
                if st.button("▶️ 繼續模擬"):
                    manager.update_simulation_status(sim_id, "running")
                    st.rerun()
        with col2:
             if sim_info.status in ["running", "paused"]:
                if st.button("⏹️ 結束模擬"):
                    manager.update_simulation_status(sim_id, "stopped")
                    st.rerun()
        with col3:
            if st.button("🗑️ 刪除記錄"):
                manager.delete_simulation(sim_id)
                st.rerun()

        if not state:
            st.warning("尚無狀態數據 (等待首次更新...)")
            return

        # 顯示狀態摘要
        st.markdown(f"**最後更新:** {state.updated_at}")
        
        # 計算總權益: 現金 + 持倉市值
        cash_balance = state.capital
        positions_value = 0.0
        unrealized_pnl = 0.0
        
        if state.positions:
            for symbol, pos in state.positions.items():
                if isinstance(pos, dict):
                    mv = pos.get('current_price', 0) * pos.get('shares', 0)
                    positions_value += mv
                    entry_val = pos.get('entry_price', 0) * pos.get('shares', 0)
                    unrealized_pnl += mv - entry_val
        
        total_equity = cash_balance + positions_value
        
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            st.metric("總權益", f"${total_equity:,.0f}")
        with sum_col2:
            st.metric("現金", f"${cash_balance:,.0f}")
        with sum_col3:
            pnl_pct = (unrealized_pnl / total_equity * 100) if total_equity else 0
            st.metric("未實現損益", f"${unrealized_pnl:,.0f}", 
                      delta=f"{pnl_pct:.2f}%")

        # 最新股票報價 (from active_signals)
        st.subheader("📈 最新股票報價")
        if state.active_signals:
            price_data = []
            for sig in state.active_signals:
                if isinstance(sig, dict):
                    change_pct = sig.get('change_pct', 0)
                    price_data.append({
                        "股票": sig.get('symbol', 'N/A'),
                        "現價": f"${sig.get('price', 0):.2f}",
                        "漲跌": f"${sig.get('change', 0):+.2f}",
                        "漲跌幅": f"{change_pct:+.2f}%",
                        "最高": f"${sig.get('high', 0):.2f}",
                        "最低": f"${sig.get('low', 0):.2f}",
                        "成交量": f"{sig.get('volume', 0):,}",
                        "持倉": "✅" if sig.get('has_position') else "❌",
                        "更新時間": sig.get('timestamp', 'N/A')
                    })
            if price_data:
                st.dataframe(price_data, use_container_width=True)
            else:
                st.info("尚無報價數據")
        else:
            st.info("尚無報價數據 (等待下次更新...)")
        
        # 持倉列表 (包含此處要求的 Share Count Display)
        st.subheader("📋 持倉明細")
        if state.positions:
            pos_data = []
            for symbol, p in state.positions.items():
                if isinstance(p, dict):
                    shares = p.get('shares', 0)
                    entry_price = p.get('entry_price', 0)
                    current_price = p.get('current_price', 0)
                    market_value = current_price * shares
                    u_pnl = market_value - (entry_price * shares)
                    u_pnl_pct = (u_pnl / (entry_price * shares) * 100) if (entry_price * shares) else 0
                    entry_date = p.get('entry_date', 'N/A')
                    
                    pos_data.append({
                        "標的": symbol,
                        "股數 (Shares)": f"{shares:,}",
                        "成本價": f"${entry_price:.2f}",
                        "現價": f"${current_price:.2f}",
                        "市值": f"${market_value:,.0f}",
                        "損益": f"${u_pnl:,.0f} ({u_pnl_pct:.2f}%)",
                        "進場日": str(entry_date)
                    })
            if pos_data:
                st.dataframe(pos_data, use_container_width=True)
            else:
                st.info("目前無持倉")
        else:
            st.info("目前無持倉")

        # 交易記錄
        st.subheader("📜 交易記錄")
        if state.trades:
            trade_data = []
            for t in state.trades[-10:]:  # Show last 10
                if isinstance(t, dict):
                    trade_data.append({
                        "標的": t.get('symbol', 'N/A'),
                        "進場": str(t.get('entry_date', 'N/A')),
                        "出場": str(t.get('exit_date', 'N/A')),
                        "股數": t.get('shares', 0),
                        "損益": f"${t.get('pnl', 0):,.0f} ({t.get('pnl_pct', 0):.2f}%)",
                        "原因": t.get('exit_reason', 'N/A')
                    })
            if trade_data:
                st.dataframe(trade_data, use_container_width=True)

        # 執行日誌
        st.subheader("📝 執行日誌")
        if state.logs:
            # Show last 20 logs in reverse order (newest first)
            with st.expander("查看最近日誌", expanded=True):
                for log in reversed(state.logs[-20:]):
                    st.text(log)
        else:
            st.info("尚無日誌記錄")

def run_enhanced_strategy_lab():
    """執行增強版策略實驗室"""
    lab = EnhancedStrategyLab()
    lab.render()


if __name__ == "__main__":
    st.set_page_config(
        page_title="AI PatternQuant - 策略實驗室（增強版）",
        page_icon="🧪",
        layout="wide"
    )
    run_enhanced_strategy_lab()
