"""Streamlit Application Entry Point for AI PatternQuant

Run with: streamlit run pattern_quant/ui/app.py

This module provides the main entry point for the Streamlit dashboard.
It can be configured to use either mock data or a real database connection.
"""

import os
import streamlit as st
from typing import Optional

from pattern_quant.ui.dashboard import Dashboard, MockDataProvider


def get_data_provider():
    """Get the appropriate data provider based on environment.
    
    If DATABASE_URL is set, uses RepositoryDataProvider with real database.
    Otherwise, uses MockDataProvider for demo purposes.
    
    Returns:
        DataProvider instance
    """
    database_url = os.environ.get('DATABASE_URL')
    
    if database_url:
        try:
            import psycopg2
            from pattern_quant.db.repository import DatabaseRepository
            from pattern_quant.ui.data_provider import RepositoryDataProvider
            
            connection = psycopg2.connect(database_url)
            repository = DatabaseRepository(connection)
            
            total_capital = float(os.environ.get('TOTAL_CAPITAL', '1000000'))
            return RepositoryDataProvider(repository, total_capital)
        except Exception as e:
            st.warning(f"無法連接資料庫，使用模擬數據: {e}")
            return MockDataProvider()
    else:
        return MockDataProvider()


def main():
    """Main entry point for the Streamlit application."""
    st.set_page_config(
        page_title="AI PatternQuant",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    with st.sidebar:
        st.title("📈 AI PatternQuant")
        st.divider()
        
        page = st.radio(
            "導航",
            options=["儀表板", "策略實驗室", "因子權重實驗室", "演化優化實驗室", "圖表詳情"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.divider()
    
    data_provider = get_data_provider()
    refresh_interval = int(os.environ.get('REFRESH_INTERVAL', '30'))
    
    if page == "儀表板":
        dashboard = Dashboard(
            data_provider=data_provider,
            page_title="AI PatternQuant Dashboard",
            refresh_interval=refresh_interval
        )
        # Skip configure_page since we already set it
        dashboard.render_header()
        dashboard.render_sidebar()
        
        # Fetch data
        metrics = data_provider.get_portfolio_metrics()
        signals = data_provider.get_active_signals()
        positions = data_provider.get_open_positions()
        
        # Render main content
        dashboard.render_core_metrics(metrics)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            dashboard.render_signal_flow(signals)
        with col2:
            dashboard.render_positions_summary(positions)
    
    elif page == "策略實驗室":
        # 選擇版本
        lab_version = st.sidebar.radio(
            "實驗室版本",
            options=["基礎版（模擬數據）", "增強版（真實數據）"],
            index=0
        )
        
        if lab_version == "增強版（真實數據）":
            try:
                from pattern_quant.ui.strategy_lab_enhanced import EnhancedStrategyLab
                lab = EnhancedStrategyLab()
                lab.render()
            except ImportError as e:
                st.warning(f"無法載入增強版策略實驗室: {e}")
                st.info("自動切換到基礎版（模擬數據）")
                # 回退到基礎版
                from pattern_quant.ui.strategy_lab import StrategyLab, BacktestEngine
                lab = StrategyLab(backtest_engine=BacktestEngine())
                lab.render()
            except Exception as e:
                st.error(f"策略實驗室發生錯誤: {e}")
                st.info("請嘗試重新整理頁面")
        else:
            from pattern_quant.ui.strategy_lab import StrategyLab, BacktestEngine
            lab = StrategyLab(backtest_engine=BacktestEngine())
            lab.render()
    
    elif page == "因子權重實驗室":
        # 因子權重實驗室頁面 (Requirements 11.1)
        try:
            from pattern_quant.ui.factor_weight_lab import FactorWeightLab
            lab = FactorWeightLab()
            lab.render()
        except ImportError as e:
            st.error(f"無法載入因子權重實驗室: {e}")
            st.info("請確保已安裝所有必要的依賴套件")
        except Exception as e:
            st.error(f"因子權重實驗室發生錯誤: {e}")
    
    elif page == "演化優化實驗室":
        # 演化優化實驗室頁面 (Requirements 10.5, 12.4)
        try:
            from pattern_quant.ui.evolution_lab import EvolutionLab
            lab = EvolutionLab()
            lab.render()
        except ImportError as e:
            st.error(f"無法載入演化優化實驗室: {e}")
            st.info("請確保已安裝所有必要的依賴套件")
        except Exception as e:
            st.error(f"演化優化實驗室發生錯誤: {e}")
    
    elif page == "圖表詳情":
        from pattern_quant.ui.chart_view import render_chart_page, ChartDataProvider, MockChartDataProvider
        
        # Use repository-backed provider if database is available
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            try:
                import psycopg2
                from pattern_quant.db.repository import DatabaseRepository
                from pattern_quant.core.pattern_engine import PatternEngine
                
                connection = psycopg2.connect(database_url)
                repository = DatabaseRepository(connection)
                pattern_engine = PatternEngine()
                chart_data_provider = ChartDataProvider(repository, pattern_engine)
            except Exception as e:
                st.warning(f"無法連接資料庫，使用模擬數據: {e}")
                chart_data_provider = MockChartDataProvider()
        else:
            chart_data_provider = MockChartDataProvider()
        
        render_chart_page(data_provider=chart_data_provider)


if __name__ == "__main__":
    main()
