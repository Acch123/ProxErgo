"""Plotly chart components for the dashboard."""

import plotly.graph_objects as go
from typing import List, Optional, Dict, Any

# Light theme colors (app uses #f8fafc background)
TEXT_COLOR = "#334155"
GRID_COLOR = "rgba(0,0,0,0.08)"
TICK_COLOR = "#64748b"
BORDER_COLOR = "#94a3b8"


def create_posture_gauge(cva_angle: float, classification: str, max_angle: float = 30.0) -> go.Figure:
    """Create CVA gauge chart."""
    color = "#059669" if cva_angle < 5 else "#d97706" if cva_angle < 12 else "#dc2626"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cva_angle,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "CVA Angle (°)", 'font': {'size': 14, 'color': TEXT_COLOR}},
        number={'font': {'size': 32, 'color': color}, 'suffix': '°'},
        gauge={
            'axis': {'range': [0, max_angle], 'tickcolor': TICK_COLOR},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [0, 5], 'color': 'rgba(5,150,105,0.15)'},
                {'range': [5, 12], 'color': 'rgba(217,119,6,0.15)'},
                {'range': [12, 20], 'color': 'rgba(234,88,12,0.15)'},
                {'range': [20, max_angle], 'color': 'rgba(220,38,38,0.15)'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': TEXT_COLOR}, height=220, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_fatigue_chart(perclos: float, blink_rate: float, level: str) -> go.Figure:
    """Create fatigue metrics bar chart."""
    color = "#059669" if level == 'alert' else "#d97706" if level == 'mild' else "#dc2626"
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[perclos], y=['PERCLOS'], orientation='h',
        marker=dict(color=color), text=[f'{perclos:.1f}%'], textposition='inside'
    ))
    fig.add_trace(go.Bar(
        x=[min(blink_rate / 30 * 100, 100)], y=['Blink Rate'], orientation='h',
        marker=dict(color="#0891b2"), text=[f'{blink_rate:.1f}/min'], textposition='inside'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': TEXT_COLOR}, height=120, showlegend=False,
        margin=dict(l=70, r=20, t=30, b=20),
        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False),
        title=dict(text=f"Fatigue: {level.upper()}", font=dict(size=14, color=color), x=0.5)
    )
    return fig


def create_trend_chart(timestamps: List[float], cva_values: List[float], 
                       fatigue_values: Optional[List[float]] = None) -> go.Figure:
    """Create time-series trend chart."""
    fig = go.Figure()
    
    if timestamps:
        base = timestamps[0]
        times = [(t - base) / 60 for t in timestamps]
        
        fig.add_trace(go.Scatter(
            x=times, y=cva_values, mode='lines', name='CVA',
            line=dict(color='#0891b2', width=2), fill='tozeroy', fillcolor='rgba(8,145,178,0.1)'
        ))
        
        if fatigue_values:
            # Align lengths (fatigue may have fewer samples when eyes not detected)
            n = min(len(times), len(fatigue_values))
            if n >= 2:
                times_f = times[-n:]
                fatigue_f = fatigue_values[-n:]
                fig.add_trace(go.Scatter(
                    x=times_f, y=fatigue_f, mode='lines', name='Fatigue (%)',
                    line=dict(color='#dc2626', width=2), yaxis='y2'
                ))
    
    fig.add_hline(y=12, line_dash="dash", line_color="#d97706", annotation_text="Warning")
    fig.add_hline(y=20, line_dash="dash", line_color="#dc2626", annotation_text="Danger")
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': TEXT_COLOR}, height=250, margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(title="Time (min)", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title="CVA (°)", range=[0, 35], showgrid=True, gridcolor=GRID_COLOR),
        yaxis2=dict(title="Fatigue %", overlaying='y', side='right', range=[0, 100]) if fatigue_values else None,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )
    return fig
