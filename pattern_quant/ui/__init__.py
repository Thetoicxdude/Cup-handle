"""UI Module for AI PatternQuant

This module provides the Streamlit-based user interface components.

Usage:
    Run the dashboard with: streamlit run pattern_quant/ui/app.py
    
    Or programmatically:
        from pattern_quant.ui import Dashboard, run_dashboard
        run_dashboard()
        
        from pattern_quant.ui import StrategyLab, run_strategy_lab
        run_strategy_lab()
        
        from pattern_quant.ui import ChartView, render_chart_page
        render_chart_page()
        
        from pattern_quant.ui import EvolutionLab, run_evolution_lab
        run_evolution_lab()
"""

from pattern_quant.ui.dashboard import (
    Dashboard,
    PortfolioMetrics,
    DataProvider,
    MockDataProvider,
    run_dashboard,
)
from pattern_quant.ui.data_provider import RepositoryDataProvider
from pattern_quant.ui.strategy_lab import (
    StrategyLab,
    StrategyParameters,
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
    run_strategy_lab,
)
from pattern_quant.ui.chart_view import (
    ChartView,
    ChartDataProvider,
    MockChartDataProvider,
    render_chart_page,
)
from pattern_quant.ui.evolution_lab import (
    EvolutionLab,
    run_evolution_lab,
)

__all__ = [
    # Dashboard
    'Dashboard',
    'PortfolioMetrics',
    'DataProvider',
    'MockDataProvider',
    'RepositoryDataProvider',
    'run_dashboard',
    # Strategy Lab
    'StrategyLab',
    'StrategyParameters',
    'BacktestEngine',
    'BacktestResult',
    'BacktestTrade',
    'run_strategy_lab',
    # Chart View
    'ChartView',
    'ChartDataProvider',
    'MockChartDataProvider',
    'render_chart_page',
    # Evolution Lab
    'EvolutionLab',
    'run_evolution_lab',
]
