import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a hex color string to an rgba string with specified alpha."""
    h = hex_color.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'


def apply_moving_average(data_series: list, window: int) -> list:
    """Applies a simple moving average to the data series."""
    if window <= 1:
        return data_series
    s = pd.Series(data_series)
    result = s.rolling(window=window, center=True, min_periods=1).mean()
    result = result.bfill().ffill()
    return result.tolist()


def load_percentile_file(kinship_level, threshold, data_dir='new'):
    """Load individual percentile TSV file."""
    kinship_str = str(kinship_level)
    file_path = f'data/raw/{data_dir}/cm_over_{threshold}/percentile_{kinship_str}.tsv'
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path, sep='\t')


def filter_un_data(df):
    """Filter UN kinship data to remove duplicate pairs."""
    duplicate_patterns = [f"[{i}-1_vs_{i}-2]" for i in range(1, 7)]
    mask = ~df['PAIR_ID'].isin(duplicate_patterns)
    return df[mask].copy()


def get_distribution_data(df):
    """
    Process dataframe into histogram bins for plotting.
    Returns lists of x (lengths), y_pct (percentage), y_count (raw counts).
    """
    percentile_cols = [col for col in df.columns if col.endswith('%')]
    percentile_cols = sorted(percentile_cols, key=lambda x: float(x.rstrip('%')))
    
    all_values = []
    for col in percentile_cols:
        all_values.extend(df[col].dropna().values)
    
    if not all_values:
        return None, None, None, 0

    bin_size = 0.05
    max_val = max(all_values)
    bin_edges = np.arange(0, max_val + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2
    
    counts = np.histogram(all_values, bins=bin_edges)[0]
    total = len(all_values)
    percentages = (counts / total * 100) if total > 0 else counts
    
    return bin_centers, percentages, counts, total


def main(data_dir='new'):
    thresholds = [1, 3, 6, 10]
    kinship_levels = [1, 2, 3, 4, 5, 6, 'UN']
    colors = ['#4C78A8', '#E45756', '#54A24B', '#F58518', '#B279A2', '#FF9DA6', '#9D755D']
    
    FILL_OPACITY = 0.6
    LINE_WIDTH = 2.0
    
    smoothing_levels = {
        'None (0cM)': 1,
        'Light (0.25cM)': 5,
        'Medium (0.5cM)': 10,
        'Heavy (1cM)': 20
    }
    
    trace_data = []
    
    print("\n" + "="*70)
    print("Generating Interactive Plotly Studio (Double Plot Mode)...")
    print("="*70)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=(
            "1. Data Manipulation: Moving Average (Standard Lines)", 
            "2. Visual Smoothing: Spline Interpolation (Curved Lines)"
        )
    )
    
    total_logical_groups = len(thresholds) * len(kinship_levels)
    print(f"Processing {total_logical_groups} logical groups ({total_logical_groups * 2} traces)...")

    for t_idx, threshold in enumerate(thresholds):
        print(f"\n  Threshold > {threshold} cM:")
        is_visible = (t_idx == 0)

        for k_idx, kinship in enumerate(kinship_levels):
            df = load_percentile_file(kinship, threshold, data_dir)
            
            if df is None:
                 x_vals, y_pct_raw, y_count, total = [], [], [], 0
            else:
                if kinship == 'UN':
                    df = filter_un_data(df)
                x_vals, y_pct_raw, y_count, total = get_distribution_data(df)
                if x_vals is None:
                    x_vals, y_pct_raw, y_count, total = [], [], [], 0

            smoothed_y = {}
            if len(y_pct_raw) > 0:
                for label, window in smoothing_levels.items():
                    smoothed_y[label] = apply_moving_average(y_pct_raw, window)
                current_y = smoothed_y['None (0cM)']
            else:
                for label in smoothing_levels:
                    smoothed_y[label] = []
                current_y = []

            hex_color = colors[k_idx]
            fill_color_rgba = hex_to_rgba(hex_color, FILL_OPACITY)
            
            hover_txt = (
                "<b>Kinship %s (cM > %d)</b><br>" % (kinship, threshold) +
                "Length: %{x:.2f} cM<br>" +
                "Percentage: %{y:.2f}%<br>" +
                "Count: %{customdata[0]:,.0f} pairs (of %{customdata[1]:.0f} total)<br>" +
                "<extra></extra>"
            )
            
            legend_group_id = f"group_{threshold}_{kinship}"

            # --- Trace 1: Top Plot ---
            fig.add_trace(go.Scatter(
                x=x_vals, 
                y=current_y,
                mode='lines',
                fill='tozeroy',
                fillcolor=fill_color_rgba,
                name=f'Kinship {kinship}',
                legendgroup=legend_group_id,
                showlegend=True,
                line=dict(color=hex_color, width=LINE_WIDTH, shape='linear'),
                visible=is_visible,
                customdata=np.column_stack((y_count, [total]*len(y_count))) if len(y_count) > 0 else None,
                hovertemplate=hover_txt
            ), row=1, col=1)
            
            trace_data.append({'smoothed_y': smoothed_y})

            # --- Trace 2: Bottom Plot ---
            fig.add_trace(go.Scatter(
                x=x_vals, 
                y=current_y,
                mode='lines',
                fill='tozeroy',
                fillcolor=fill_color_rgba,
                name=f'Kinship {kinship}',
                legendgroup=legend_group_id,
                showlegend=False,
                line=dict(color=hex_color, width=LINE_WIDTH, shape='spline', smoothing=1.3),
                visible=is_visible,
                customdata=np.column_stack((y_count, [total]*len(y_count))) if len(y_count) > 0 else None,
                hovertemplate=hover_txt
            ), row=2, col=1)
            
            trace_data.append({'smoothed_y': smoothed_y})

            if len(y_count) > 0:
                print(f"    ✓ Kinship {kinship}: {total} pairs")

    # --- Interactive Controls ---
    print("\nConfiguring interactive controls...")
    
    # 1. Threshold Dropdown
    traces_per_threshold = len(kinship_levels) * 2 
    dropdown_buttons = []
    for i, threshold in enumerate(thresholds):
        total_traces = len(trace_data)
        visibility = [False] * total_traces
        start_idx = i * traces_per_threshold
        end_idx = start_idx + traces_per_threshold
        for j in range(start_idx, end_idx):
            visibility[j] = True
        
        button = dict(
            label=f"Threshold > {threshold} cM",
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"Kinship Distribution Analysis (Threshold > {threshold} cM)"}
            ]
        )
        dropdown_buttons.append(button)

    # 2. Scale Toggle
    scale_buttons = [
        dict(label="Linear Scale", method="relayout", args=[{"yaxis.type": "linear", "yaxis2.type": "linear"}]),
        dict(label="Log Scale", method="relayout", args=[{"yaxis.type": "log", "yaxis2.type": "log"}])
    ]
    
    # 3. Smoothing Control
    smoothing_buttons = []
    for s_label, window in smoothing_levels.items():
        new_y_values = []
        for data_point in trace_data:
            if not data_point['smoothed_y']:
                new_y_values.append(None)
            else:
                new_y_values.append(data_point['smoothed_y'][s_label])
        
        button = dict(
            label=s_label,
            method="restyle",
            args=[{"y": new_y_values}]
        )
        smoothing_buttons.append(button)

    # 4. Zoom Sync Control (UPDATED)
    # Now syncs BOTH X (width) and Y (height)
    zoom_buttons = [
        dict(
            label="Sync On",
            method="relayout",
            args=[{
                "xaxis2.matches": "x",
                "yaxis2.matches": "y",
                "xaxis2.showticklabels": True, # Ensure labels don't disappear
                "yaxis2.showticklabels": True
            }] 
        ),
        dict(
            label="Sync Off",
            method="relayout",
            args=[{
                "xaxis2.matches": None,
                "yaxis2.matches": None,
                "xaxis2.autorange": True, # Reset bounds on unsync
                "yaxis2.autorange": True
            }] 
        )
    ]

    fig.update_layout(
        title={
            "text": "Kinship Distribution Analysis (Threshold > 1 cM)",
            "x": 0.5, "xanchor": "center", "font": {"size": 20}
        },
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.01,
            bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#cccccc", borderwidth=1,
            itemclick="toggleothers", itemdoubleclick="toggle"
        ),
        margin=dict(l=80, r=200, t=180, b=80),
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        paper_bgcolor="white",
        width=1400,
        height=1000,
        
        # --- DEFAULT SYNC CONFIGURATION ---
        # Syncing both X and Y axes by default
        xaxis2=dict(matches='x'),
        yaxis2=dict(matches='y', showticklabels=True),
        
        updatemenus=[
             # Threshold Selector
             dict(
                 buttons=dropdown_buttons,
                 direction="down", pad={"r": 10, "t": 10}, showactive=True,
                 x=0.0, xanchor="left", y=1.15, yanchor="top",
                 bgcolor="#ffffff", bordercolor="#cccccc", borderwidth=1
             ),
             # Scale Toggle
             dict(
                 buttons=scale_buttons,
                 direction="right", pad={"r": 10, "t": 10}, showactive=True,
                 x=1.0, xanchor="right", y=1.15, yanchor="top",
                 bgcolor="#ffffff", bordercolor="#cccccc", borderwidth=1
             ),
             # Smoothing Control
             dict(
                 buttons=smoothing_buttons,
                 direction="right", pad={"r": 10, "t": 10}, showactive=True,
                 x=0.45, xanchor="center", y=1.15, yanchor="top",
                 bgcolor="#ffffff", bordercolor="#cccccc", borderwidth=1
             ),
             # Zoom Sync Toggle
             dict(
                 buttons=zoom_buttons,
                 direction="right", pad={"r": 10, "t": 10}, showactive=True,
                 x=0.65, xanchor="center", y=1.15, yanchor="top",
                 bgcolor="#ffffff", bordercolor="#cccccc", borderwidth=1
             )
         ]
    )

    # Axis Titles
    fig.update_xaxes(title_text="Length (centiMorgans)", showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="Percentage (%)", showgrid=True, row=1, col=1)
    
    fig.update_xaxes(title_text="Length (centiMorgans)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Percentage (%)", showgrid=True, row=2, col=1)

    # Annotations
    fig.add_annotation(text="<b>Select Threshold:</b>", x=0.0, y=1.20, xref="paper", yref="paper", xanchor="left", showarrow=False, font=dict(size=12))
    fig.add_annotation(text="<b>Scale:</b>", x=1.0, y=1.20, xref="paper", yref="paper", xanchor="right", showarrow=False, font=dict(size=12))
    fig.add_annotation(text="<b>Data Smoothing:</b>", x=0.45, y=1.20, xref="paper", yref="paper", xanchor="center", showarrow=False, font=dict(size=12))
    fig.add_annotation(text="<b>Zoom Sync:</b>", x=0.65, y=1.20, xref="paper", yref="paper", xanchor="center", showarrow=False, font=dict(size=12))

    output_file = 'kinship_interactive_studio.html'
    print(f"\nSaving interactive studio to {output_file}...")
    fig.write_html(
        output_file, 
        include_plotlyjs='cdn',
        full_html=True,
        config=dict(responsive=True, displayModeBar=True, displaylogo=False)
    )
    
    print("\n" + "="*70)
    print("✓ SUCCESS: Interactive Studio Created!")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate interactive kinship studio visualization')
    parser.add_argument('--data-dir', type=str, default='new', help="Data directory name")
    args = parser.parse_args()
    main(args.data_dir)