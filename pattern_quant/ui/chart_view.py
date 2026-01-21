"""Chart Detail View for AI PatternQuant

This module provides the chart detail page with K-line charts,
volume charts, and pattern markers using Plotly.

Requirements: 13.1, 13.2, 13.3, 13.4
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any
from datetime import datetime

from pattern_quant.core.models import (
    OHLCV,
    CupPattern,
    HandlePattern,
    PatternResult,
    MatchScore,
)


class ChartView:
    """Chart detail view component.
    
    Provides K-line (candlestick) charts with volume and pattern markers.
    
    Attributes:
        chart_height: Height of the main chart in pixels
        volume_height_ratio: Ratio of volume chart height to main chart
    """
    
    def __init__(
        self,
        chart_height: int = 600,
        volume_height_ratio: float = 0.3
    ):
        """Initialize the chart view.
        
        Args:
            chart_height: Height of the main chart in pixels
            volume_height_ratio: Ratio of volume chart height to main chart
        """
        self.chart_height = chart_height
        self.volume_height_ratio = volume_height_ratio
    
    def create_candlestick_chart(
        self,
        ohlcv_data: List[OHLCV],
        symbol: str,
        pattern_result: Optional[PatternResult] = None
    ) -> go.Figure:
        """Create a candlestick chart with volume.
        
        Creates a combined chart with:
        - Candlestick (K-line) chart on top
        - Volume bar chart on bottom
        - Pattern markers if pattern_result is provided
        
        Args:
            ohlcv_data: List of OHLCV data points
            symbol: Stock symbol for title
            pattern_result: Optional pattern result for markers
            
        Returns:
            Plotly Figure object
            
        Requirements: 13.1
        """
        if not ohlcv_data:
            fig = go.Figure()
            fig.add_annotation(
                text="無數據可顯示",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Extract data
        dates = [candle.time for candle in ohlcv_data]
        opens = [candle.open for candle in ohlcv_data]
        highs = [candle.high for candle in ohlcv_data]
        lows = [candle.low for candle in ohlcv_data]
        closes = [candle.close for candle in ohlcv_data]
        volumes = [candle.volume for candle in ohlcv_data]
        
        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} K線圖', '成交量')
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name='K線',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                increasing_fillcolor='#26a69a',
                decreasing_fillcolor='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add volume bars with color based on price change
        colors = [
            '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            for i in range(len(closes))
        ]
        
        fig.add_trace(
            go.Bar(
                x=dates,
                y=volumes,
                name='成交量',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add pattern markers if available
        if pattern_result and pattern_result.is_valid and pattern_result.cup:
            self._add_pattern_markers(fig, ohlcv_data, pattern_result)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{symbol} 技術分析圖表',
                font=dict(size=18)
            ),
            xaxis_rangeslider_visible=False,
            height=self.chart_height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="價格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        
        # Update x-axis
        fig.update_xaxes(
            title_text="日期",
            row=2, col=1,
            tickformat='%Y-%m-%d'
        )
        
        return fig

    def _add_pattern_markers(
        self,
        fig: go.Figure,
        ohlcv_data: List[OHLCV],
        pattern_result: PatternResult
    ) -> None:
        """Add pattern markers to the chart.
        
        Adds:
        - Green dots for cup rim positions (left and right peaks)
        - Red horizontal line for resistance level
        - Blue curve for cup bottom fit
        - Orange line for handle region
        - Breakout price line
        - Annotations for pattern details
        
        Args:
            fig: Plotly Figure to add markers to
            ohlcv_data: OHLCV data for date reference
            pattern_result: Pattern result with cup and handle data
            
        Requirements: 13.2, 13.3
        """
        cup = pattern_result.cup
        if not cup:
            return
        
        dates = [candle.time for candle in ohlcv_data]
        
        # Add green dots for cup rim positions (left and right peaks)
        # Left peak marker
        if 0 <= cup.left_peak_index < len(dates):
            fig.add_trace(
                go.Scatter(
                    x=[dates[cup.left_peak_index]],
                    y=[cup.left_peak_price],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='#00c853',
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    name='左杯緣',
                    hovertemplate=(
                        f'左杯緣<br>'
                        f'日期: %{{x|%Y-%m-%d}}<br>'
                        f'價格: ${cup.left_peak_price:.2f}<extra></extra>'
                    )
                ),
                row=1, col=1
            )
        
        # Right peak marker
        if 0 <= cup.right_peak_index < len(dates):
            fig.add_trace(
                go.Scatter(
                    x=[dates[cup.right_peak_index]],
                    y=[cup.right_peak_price],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='#00c853',
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    name='右杯緣',
                    hovertemplate=(
                        f'右杯緣<br>'
                        f'日期: %{{x|%Y-%m-%d}}<br>'
                        f'價格: ${cup.right_peak_price:.2f}<extra></extra>'
                    )
                ),
                row=1, col=1
            )
        
        # Cup bottom marker
        if 0 <= cup.bottom_index < len(dates):
            fig.add_trace(
                go.Scatter(
                    x=[dates[cup.bottom_index]],
                    y=[cup.bottom_price],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='#2196f3',
                        symbol='triangle-up',
                        line=dict(width=2, color='white')
                    ),
                    name='杯底',
                    hovertemplate=(
                        f'杯底<br>'
                        f'日期: %{{x|%Y-%m-%d}}<br>'
                        f'價格: ${cup.bottom_price:.2f}<extra></extra>'
                    )
                ),
                row=1, col=1
            )
        
        # Add cup bottom curve (parabola fit visualization)
        self._add_cup_curve(fig, ohlcv_data, cup, dates)
        
        # Add resistance line (red horizontal line at the higher peak)
        resistance_price = max(cup.left_peak_price, cup.right_peak_price)
        
        # Determine line range
        start_idx = max(0, cup.left_peak_index - 5)
        end_idx = min(len(dates) - 1, cup.right_peak_index + 20)
        
        if start_idx < len(dates) and end_idx < len(dates):
            fig.add_trace(
                go.Scatter(
                    x=[dates[start_idx], dates[end_idx]],
                    y=[resistance_price, resistance_price],
                    mode='lines',
                    line=dict(
                        color='#f44336',
                        width=2,
                        dash='dash'
                    ),
                    name=f'壓力位 ${resistance_price:.2f}',
                    hovertemplate=(
                        f'壓力位<br>'
                        f'價格: ${resistance_price:.2f}<extra></extra>'
                    )
                ),
                row=1, col=1
            )
        
        # Add breakout price line (resistance + 0.5% buffer)
        breakout_price = resistance_price * 1.005
        if start_idx < len(dates) and end_idx < len(dates):
            fig.add_trace(
                go.Scatter(
                    x=[dates[start_idx], dates[end_idx]],
                    y=[breakout_price, breakout_price],
                    mode='lines',
                    line=dict(
                        color='#4caf50',
                        width=1.5,
                        dash='dot'
                    ),
                    name=f'突破價 ${breakout_price:.2f}',
                    hovertemplate=(
                        f'突破價位 (壓力位+0.5%)<br>'
                        f'價格: ${breakout_price:.2f}<extra></extra>'
                    )
                ),
                row=1, col=1
            )
        
        # Add handle markers if available
        handle = pattern_result.handle
        if handle and 0 <= handle.start_index < len(dates) and 0 <= handle.end_index < len(dates):
            # Handle region shading
            handle_dates = dates[handle.start_index:handle.end_index + 1]
            handle_prices = [ohlcv_data[i].close for i in range(handle.start_index, handle.end_index + 1)]
            
            if handle_dates and handle_prices:
                fig.add_trace(
                    go.Scatter(
                        x=handle_dates,
                        y=handle_prices,
                        mode='lines',
                        line=dict(color='#ff9800', width=2),
                        name='柄部',
                        hovertemplate=(
                            f'柄部<br>'
                            f'日期: %{{x|%Y-%m-%d}}<br>'
                            f'價格: $%{{y:.2f}}<extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
            
            # Handle lowest point marker
            lowest_idx = handle.start_index
            for i in range(handle.start_index, min(handle.end_index + 1, len(ohlcv_data))):
                if ohlcv_data[i].close <= handle.lowest_price:
                    lowest_idx = i
                    break
            
            if 0 <= lowest_idx < len(dates):
                fig.add_trace(
                    go.Scatter(
                        x=[dates[lowest_idx]],
                        y=[handle.lowest_price],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='#ff9800',
                            symbol='diamond',
                            line=dict(width=2, color='white')
                        ),
                        name='柄部最低點',
                        hovertemplate=(
                            f'柄部最低點<br>'
                            f'日期: %{{x|%Y-%m-%d}}<br>'
                            f'價格: ${handle.lowest_price:.2f}<extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
            
            # Add technical stop loss line (handle lowest price)
            if start_idx < len(dates) and end_idx < len(dates):
                fig.add_trace(
                    go.Scatter(
                        x=[dates[handle.start_index], dates[end_idx]],
                        y=[handle.lowest_price, handle.lowest_price],
                        mode='lines',
                        line=dict(
                            color='#9c27b0',
                            width=1.5,
                            dash='dashdot'
                        ),
                        name=f'技術止損 ${handle.lowest_price:.2f}',
                        hovertemplate=(
                            f'技術止損位 (柄部最低點)<br>'
                            f'價格: ${handle.lowest_price:.2f}<extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
    
    def _add_cup_curve(
        self,
        fig: go.Figure,
        ohlcv_data: List[OHLCV],
        cup: CupPattern,
        dates: List
    ) -> None:
        """Add cup bottom curve visualization.
        
        Draws a smooth curve representing the parabola fit of the cup bottom.
        
        Args:
            fig: Plotly Figure to add curve to
            ohlcv_data: OHLCV data
            cup: Cup pattern data
            dates: List of dates
        """
        import numpy as np
        
        # Get cup region indices
        start_idx = cup.left_peak_index
        end_idx = cup.right_peak_index
        
        if start_idx >= end_idx or start_idx < 0 or end_idx >= len(dates):
            return
        
        # Extract cup region prices
        cup_prices = [ohlcv_data[i].close for i in range(start_idx, end_idx + 1)]
        cup_dates = dates[start_idx:end_idx + 1]
        
        if len(cup_prices) < 3:
            return
        
        try:
            # Fit parabola to cup region
            x = np.arange(len(cup_prices))
            coeffs = np.polyfit(x, cup_prices, 2)
            
            # Generate smooth curve points
            x_smooth = np.linspace(0, len(cup_prices) - 1, 50)
            y_smooth = np.polyval(coeffs, x_smooth)
            
            # Interpolate dates for smooth curve
            date_indices = np.linspace(0, len(cup_dates) - 1, 50).astype(int)
            smooth_dates = [cup_dates[min(i, len(cup_dates) - 1)] for i in date_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=smooth_dates,
                    y=y_smooth,
                    mode='lines',
                    line=dict(
                        color='#03a9f4',
                        width=2,
                        dash='dot'
                    ),
                    name=f'杯底擬合 (R²={cup.r_squared:.3f})',
                    hovertemplate=(
                        f'杯底擬合曲線<br>'
                        f'R² = {cup.r_squared:.4f}<extra></extra>'
                    )
                ),
                row=1, col=1
            )
        except Exception:
            # If fitting fails, skip the curve
            pass

    def render_pattern_annotations(
        self,
        pattern_result: PatternResult
    ) -> None:
        """Render pattern mathematical annotations in Streamlit.
        
        Displays pattern details including:
        - Left peak price
        - Right peak price
        - R² fit value
        - Depth ratio
        - Symmetry score
        - Handle volume slope
        - Mathematical formulas
        
        Args:
            pattern_result: Pattern result with cup and handle data
            
        Requirements: 13.4
        """
        if not pattern_result.is_valid or not pattern_result.cup:
            st.info("無有效型態可顯示")
            return
        
        cup = pattern_result.cup
        handle = pattern_result.handle
        score = pattern_result.score
        
        st.subheader("📐 數學註解")
        
        # Mathematical formulas explanation
        with st.expander("📊 型態識別公式說明", expanded=False):
            st.markdown("""
            **茶杯型態識別使用以下數學方法：**
            
            1. **杯底擬合**: 使用二次函數 $y = ax^2 + bx + c$ 擬合杯底曲線
               - 要求 $a > 0$ (開口向上)
               - 要求 $R^2 > 0.8$ (擬合度)
            
            2. **左右峰對稱性**: $|P_{left} - P_{right}| / P_{left} < \\alpha$ (預設 5%)
            
            3. **杯身深度**: $(P_{left} - Min_{cup}) / P_{left}$ 介於 12% 至 33%
            
            4. **柄部成交量**: 線性回歸斜率 $Slope_{vol} < 0$ (萎縮趨勢)
            """)
        
        # Cup pattern details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**茶杯型態參數**")
            
            # Calculate additional metrics
            peak_diff_ratio = abs(cup.left_peak_price - cup.right_peak_price) / cup.left_peak_price * 100
            
            st.markdown(f"""
            | 參數 | 數值 | 說明 |
            |------|------|------|
            | 左峰價格 $P_{{left}}$ | ${cup.left_peak_price:.2f} | 茶杯左側高點 |
            | 右峰價格 $P_{{right}}$ | ${cup.right_peak_price:.2f} | 茶杯右側高點 |
            | 杯底價格 $Min_{{cup}}$ | ${cup.bottom_price:.2f} | 茶杯最低點 |
            | 擬合度 $R^2$ | {cup.r_squared:.4f} | 二次函數擬合品質 |
            | 杯身深度 | {cup.depth_ratio * 100:.1f}% | $(P_{{left}} - Min_{{cup}}) / P_{{left}}$ |
            | 峰值差異 | {peak_diff_ratio:.2f}% | 左右峰對稱性 |
            | 對稱性分數 | {cup.symmetry_score:.2f} | 0-1 範圍 |
            """)
            
            # Resistance and breakout prices
            resistance_price = max(cup.left_peak_price, cup.right_peak_price)
            breakout_price = resistance_price * 1.005
            
            st.markdown("**關鍵價位**")
            st.markdown(f"""
            | 價位 | 數值 | 計算方式 |
            |------|------|----------|
            | 壓力位 | ${resistance_price:.2f} | $max(P_{{left}}, P_{{right}})$ |
            | 突破價 | ${breakout_price:.2f} | 壓力位 × 1.005 |
            """)
        
        with col2:
            if handle:
                st.markdown("**柄部型態參數**")
                
                # Calculate handle depth relative to cup
                cup_upper_half = (cup.left_peak_price + cup.bottom_price) / 2
                handle_depth_pct = (cup.right_peak_price - handle.lowest_price) / cup.right_peak_price * 100
                
                st.markdown(f"""
                | 參數 | 數值 | 說明 |
                |------|------|------|
                | 柄部最低價 | ${handle.lowest_price:.2f} | 柄部區間最低點 |
                | 成交量斜率 | {handle.volume_slope:.4f} | 負值表示萎縮 |
                | 柄部天數 | {handle.end_index - handle.start_index + 1} 天 | 柄部持續時間 |
                | 柄部深度 | {handle_depth_pct:.1f}% | 相對右峰回調 |
                | 杯身上半部 | ${cup_upper_half:.2f} | 柄部不應跌破 |
                """)
                
                # Technical stop loss
                st.markdown("**止損價位**")
                hard_stop = cup.right_peak_price * 0.95  # 5% hard stop
                st.markdown(f"""
                | 止損類型 | 價位 | 說明 |
                |----------|------|------|
                | 技術止損 | ${handle.lowest_price:.2f} | 柄部最低點 |
                | 硬止損 (5%) | ${hard_stop:.2f} | 進場價 × 0.95 |
                """)
            
            if score:
                st.markdown("**吻合分數明細**")
                st.markdown(f"""
                | 分項 | 分數 | 權重 |
                |------|------|------|
                | 擬合度分數 | {score.r_squared_score:.1f} | 30% |
                | 對稱性分數 | {score.symmetry_score:.1f} | 25% |
                | 成交量分數 | {score.volume_score:.1f} | 25% |
                | 深度分數 | {score.depth_score:.1f} | 20% |
                | **總分** | **{score.total_score:.1f}** | 100% |
                """)
        
        # Visual indicators
        st.divider()
        
        # Progress bars for scores
        if score:
            st.markdown("**分數視覺化**")
            
            score_col1, score_col2, score_col3 = st.columns(3)
            
            with score_col1:
                st.metric("總吻合分數", f"{score.total_score:.1f}%")
                st.progress(score.total_score / 100)
            
            with score_col2:
                # Determine pattern quality
                if score.total_score >= 90:
                    quality = "🌟 優秀"
                    quality_desc = "高品質型態，突破機率高"
                elif score.total_score >= 80:
                    quality = "✅ 良好"
                    quality_desc = "型態清晰，值得關注"
                elif score.total_score >= 70:
                    quality = "⚠️ 一般"
                    quality_desc = "型態尚可，需謹慎"
                else:
                    quality = "❌ 較弱"
                    quality_desc = "型態不明顯，建議觀望"
                
                st.metric("型態品質", quality)
                st.caption(quality_desc)
            
            with score_col3:
                # Trading recommendation
                if score.total_score >= 80 and handle and handle.volume_slope < 0:
                    recommendation = "🟢 可考慮進場"
                    rec_desc = "等待突破確認"
                elif score.total_score >= 70:
                    recommendation = "🟡 觀察中"
                    rec_desc = "等待型態完善"
                else:
                    recommendation = "🔴 暫不建議"
                    rec_desc = "型態品質不足"
                
                st.metric("交易建議", recommendation)
                st.caption(rec_desc)
        
        # Chart legend
        st.divider()
        st.markdown("**圖表標記說明**")
        legend_col1, legend_col2, legend_col3 = st.columns(3)
        
        with legend_col1:
            st.markdown("""
            - 🟢 **綠色圓點**: 杯緣位置 (左/右峰)
            - 🔵 **藍色三角**: 杯底位置
            - 🔵 **藍色虛線**: 杯底擬合曲線
            """)
        
        with legend_col2:
            st.markdown("""
            - 🔴 **紅色虛線**: 壓力位
            - 🟢 **綠色點線**: 突破價位
            - 🟣 **紫色點劃線**: 技術止損位
            """)
        
        with legend_col3:
            st.markdown("""
            - 🟠 **橙色線**: 柄部區間
            - 🟠 **橙色菱形**: 柄部最低點
            """)

    def render(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV],
        pattern_result: Optional[PatternResult] = None
    ) -> None:
        """Render the complete chart view.
        
        Main entry point for displaying the chart detail page.
        
        Args:
            symbol: Stock symbol
            ohlcv_data: OHLCV data for the chart
            pattern_result: Optional pattern result for markers
            
        Requirements: 13.1, 13.2, 13.3, 13.4
        """
        st.header(f"📊 {symbol} 圖表詳情")
        
        if not ohlcv_data:
            st.warning(f"無法載入 {symbol} 的數據")
            return
        
        # Create and display the chart
        fig = self.create_candlestick_chart(ohlcv_data, symbol, pattern_result)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display pattern annotations if available
        if pattern_result:
            self.render_pattern_annotations(pattern_result)
        else:
            st.info("尚未進行型態識別分析")


class ChartDataProvider:
    """Data provider for chart view.
    
    Provides methods to fetch OHLCV data and pattern results.
    """
    
    def __init__(self, repository=None, pattern_engine=None):
        """Initialize the chart data provider.
        
        Args:
            repository: Database repository instance
            pattern_engine: Pattern engine instance
        """
        self.repository = repository
        self.pattern_engine = pattern_engine
    
    def get_ohlcv_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Get OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of records
            
        Returns:
            List of OHLCV records
        """
        if self.repository:
            return self.repository.get_ohlcv_by_symbol(
                symbol, start_time, end_time, limit
            )
        return []
    
    def analyze_pattern(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV]
    ) -> Optional[PatternResult]:
        """Analyze pattern for the given data.
        
        Args:
            symbol: Stock symbol
            ohlcv_data: OHLCV data to analyze
            
        Returns:
            PatternResult or None
        """
        if self.pattern_engine and ohlcv_data:
            return self.pattern_engine.analyze_ohlcv(symbol, ohlcv_data)
        return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols.
        
        Returns:
            List of symbol strings
        """
        if self.repository:
            return self.repository.get_symbols_with_data()
        return []


class MockChartDataProvider:
    """Mock data provider for demo/testing purposes."""
    
    def get_ohlcv_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Generate mock OHLCV data."""
        import random
        from datetime import timedelta
        
        # Generate 200 days of mock data
        num_days = limit or 200
        base_price = 100.0
        base_date = datetime.now() - timedelta(days=num_days)
        
        data = []
        current_price = base_price
        
        for i in range(num_days):
            date = base_date + timedelta(days=i)
            
            # Random price movement
            change = random.uniform(-0.03, 0.03)
            open_price = current_price
            close_price = current_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))
            volume = random.randint(100000, 1000000)
            
            data.append(OHLCV(
                time=date,
                symbol=symbol,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            ))
            
            current_price = close_price
        
        return data
    
    def analyze_pattern(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV]
    ) -> Optional[PatternResult]:
        """Return mock pattern result."""
        if not ohlcv_data or len(ohlcv_data) < 50:
            return None
        
        # Create a mock cup pattern
        mid_idx = len(ohlcv_data) // 2
        left_idx = mid_idx - 30
        right_idx = mid_idx + 20
        bottom_idx = mid_idx - 5
        
        if left_idx < 0 or right_idx >= len(ohlcv_data):
            return None
        
        cup = CupPattern(
            left_peak_index=left_idx,
            left_peak_price=ohlcv_data[left_idx].high,
            right_peak_index=right_idx,
            right_peak_price=ohlcv_data[right_idx].high,
            bottom_index=bottom_idx,
            bottom_price=ohlcv_data[bottom_idx].low,
            r_squared=0.85,
            depth_ratio=0.18,
            symmetry_score=0.92
        )
        
        handle = HandlePattern(
            start_index=right_idx,
            end_index=min(right_idx + 15, len(ohlcv_data) - 1),
            lowest_price=ohlcv_data[right_idx + 5].low if right_idx + 5 < len(ohlcv_data) else ohlcv_data[right_idx].low,
            volume_slope=-0.05
        )
        
        score = MatchScore(
            total_score=82.5,
            r_squared_score=85.0,
            symmetry_score=92.0,
            volume_score=75.0,
            depth_score=78.0
        )
        
        return PatternResult(
            symbol=symbol,
            pattern_type="cup_and_handle",
            cup=cup,
            handle=handle,
            score=score,
            is_valid=True,
            rejection_reason=None
        )
    
    def get_available_symbols(self) -> List[str]:
        """Return mock symbols."""
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


def render_chart_page(
    data_provider=None,
    selected_symbol: Optional[str] = None
) -> None:
    """Render the chart detail page.
    
    Standalone function to render the chart page in Streamlit.
    
    Args:
        data_provider: Data provider instance
        selected_symbol: Pre-selected symbol
    """
    if data_provider is None:
        data_provider = MockChartDataProvider()
    
    chart_view = ChartView()
    
    # Symbol selection
    available_symbols = data_provider.get_available_symbols()
    
    if not available_symbols:
        st.warning("無可用股票數據")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if selected_symbol and selected_symbol in available_symbols:
            default_idx = available_symbols.index(selected_symbol)
        else:
            default_idx = 0
        
        symbol = st.selectbox(
            "選擇股票",
            options=available_symbols,
            index=default_idx
        )
    
    with col2:
        analyze_pattern = st.checkbox("顯示型態分析", value=True)
    
    # Fetch data
    ohlcv_data = data_provider.get_ohlcv_data(symbol)
    
    # Analyze pattern if requested
    pattern_result = None
    if analyze_pattern:
        pattern_result = data_provider.analyze_pattern(symbol, ohlcv_data)
    
    # Render chart
    chart_view.render(symbol, ohlcv_data, pattern_result)
