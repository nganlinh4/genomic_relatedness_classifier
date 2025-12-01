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


def load_raw_kinship_file(kinship_level):
    """Load raw kinship TSV file with actual cM lengths."""
    kinship_str = str(kinship_level)
    file_path = f'data/raw/kinship_{kinship_str}.tsv'
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path, sep='\t')


def filter_un_data(df):
    """Filter UN kinship data to remove duplicate pairs."""
    duplicate_patterns = [f"[{i}-1_vs_{i}-2]" for i in range(1, 7)]
    # Ensure PAIR_ID exists
    if 'PAIR_ID' not in df.columns:
        return df
    mask = ~df['PAIR_ID'].isin(duplicate_patterns)
    return df[mask].copy()


def get_data_values(df):
    """
    Extracts the raw values from the dataframe for box plots and histograms.
    Returns:
        all_values: list of all data points
    """
    percentile_cols = [col for col in df.columns if col.endswith('%')]
    percentile_cols = sorted(percentile_cols, key=lambda x: float(x.rstrip('%')))
    
    all_values = []
    for col in percentile_cols:
        all_values.extend(df[col].dropna().values)
        
    return np.array(all_values)


def get_cm_lengths(kinship_level, threshold):
    """
    Extract actual cM lengths from raw kinship file, filtered by threshold.
    Includes FIX for UN duplicate data.
    """
    df_raw = load_raw_kinship_file(kinship_level)
    if df_raw is None or len(df_raw) == 0:
        return np.array([])

    # --- FIX: Apply Duplicate Filter to Raw Data for UN Group ---
    if str(kinship_level) == 'UN':
        df_raw = filter_un_data(df_raw)
    # ------------------------------------------------------------
    
    # Identify the column with cM lengths
    cM_col = None
    col_lower_map = {col.lower(): col for col in df_raw.columns}
    
    for key in ['length_cm', 'cm', 'centimorgan', 'centimorgans', 'length']:
        if key in col_lower_map:
            cM_col = col_lower_map[key]
            break
    
    if cM_col is None:
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            cM_col = numeric_cols[0]
    
    if cM_col is None:
        return np.array([])
    
    lengths = df_raw[df_raw[cM_col] > threshold][cM_col].values
    return np.array(lengths)


def get_distribution_data(all_values):
    """Process raw values into histogram bins for plotting."""
    if len(all_values) == 0:
        return None, None, None, 0

    bin_size = 0.05
    max_val = np.max(all_values)
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
    print("Generating Optimized Interactive Studio (Low Lag Mode)...")
    print("="*70)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.35, 0.35, 0.30],
        subplot_titles=(
            "1. Data Manipulation: Moving Average (Standard Lines)", 
            "2. Visual Smoothing: Spline Interpolation (Curved Lines)",
            "3. Statistical Summary: Box Plot Distribution"
        )
    )
    
    traces_per_group = 3
    traces_per_threshold = len(kinship_levels) * traces_per_group

    for t_idx, threshold in enumerate(thresholds):
        print(f"\n  Threshold > {threshold} cM:")
        is_visible = (t_idx == 0)

        for k_idx, kinship in enumerate(kinship_levels):
            # Load and process Histogram Data (for Lines)
            df = load_percentile_file(kinship, threshold, data_dir)
            
            x_vals, y_pct_raw, y_count, total = [], [], [], 0
            if df is not None:
                if kinship == 'UN':
                    df = filter_un_data(df)
                
                histogram_data_points = get_data_values(df)
                if len(histogram_data_points) > 0:
                    x_vals, y_pct_raw, y_count, total = get_distribution_data(histogram_data_points)

            # Load Raw Data (for Box Plot)
            box_plot_cm_lengths = get_cm_lengths(kinship, threshold)

            # Smoothing logic
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
                "Count: %{customdata[0]:,.0f} pairs<br>" +
                "<extra></extra>"
            )
            
            legend_group_id = f"group_{threshold}_{kinship}"

            # Trace 1: Linear Line
            fig.add_trace(go.Scatter(
                x=x_vals, y=current_y, mode='lines', fill='tozeroy', fillcolor=fill_color_rgba,
                name=f'Kinship {kinship}', legendgroup=legend_group_id, showlegend=True,
                line=dict(color=hex_color, width=LINE_WIDTH, shape='linear'),
                visible=is_visible,
                customdata=np.column_stack((y_count, [total]*len(y_count))) if len(y_count) > 0 else None,
                hovertemplate=hover_txt
            ), row=1, col=1)
            trace_data.append({'smoothed_y': smoothed_y, 'type': 'line'})

            # Trace 2: Spline Line
            fig.add_trace(go.Scatter(
                x=x_vals, y=current_y, mode='lines', fill='tozeroy', fillcolor=fill_color_rgba,
                name=f'Kinship {kinship}', legendgroup=legend_group_id, showlegend=False,
                line=dict(color=hex_color, width=LINE_WIDTH, shape='spline', smoothing=1.3),
                visible=is_visible,
                customdata=np.column_stack((y_count, [total]*len(y_count))) if len(y_count) > 0 else None,
                hovertemplate=hover_txt
            ), row=2, col=1)
            trace_data.append({'smoothed_y': smoothed_y, 'type': 'line'})

            # Trace 3: Box Plot (PERFORMANCE OPTIMIZED)
            fig.add_trace(go.Box(
                y=box_plot_cm_lengths if len(box_plot_cm_lengths) > 0 else [None],
                name=f'Kinship {kinship}',
                legendgroup=legend_group_id,
                showlegend=False,
                marker_color=hex_color,
                # --- OPTIMIZATION HERE ---
                boxpoints=False, # Default to False (No points drawn)
                # -------------------------
                jitter=0.3,
                pointpos=-1.8,
                visible=is_visible,
                hovertemplate="<b>Kinship %s (cM > %d)</b><br>Length: %%{y:.2f} cM<extra></extra>" % (kinship, threshold)
            ), row=3, col=1)
            trace_data.append({'smoothed_y': None, 'type': 'box'})

            if len(y_count) > 0:
                print(f"    ✓ Kinship {kinship}: {total} pairs")

    # --- Interactive Controls ---
    print("\nConfiguring interactive controls...")
    
    # 1. Threshold Selector
    dropdown_buttons = []
    for i, threshold in enumerate(thresholds):
        visibility = [False] * len(trace_data)
        start_idx = i * traces_per_threshold
        end_idx = start_idx + traces_per_threshold
        for j in range(start_idx, end_idx):
            visibility[j] = True
        
        dropdown_buttons.append(dict(
            label=f"Threshold > {threshold} cM",
            method="update",
            args=[{"visible": visibility}]
        ))

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
            if data_point['type'] == 'box':
                new_y_values.append(None) 
            elif not data_point['smoothed_y']:
                new_y_values.append([])
            else:
                new_y_values.append(data_point['smoothed_y'][s_label])
        
        smoothing_buttons.append(dict(label=s_label, method="restyle", args=[{"y": new_y_values}]))

    # 4. Zoom Sync
    zoom_buttons = [
        dict(label="Sync On", method="relayout", args=[{"xaxis2.matches": "x", "yaxis2.matches": "y", "xaxis2.showticklabels": True, "yaxis2.showticklabels": True}]),
        dict(label="Sync Off", method="relayout", args=[{"xaxis2.matches": None, "yaxis2.matches": None, "xaxis2.autorange": True, "yaxis2.autorange": True}])
    ]

    # 5. NEW: Outlier Toggle (Performance Control)
    # This button toggles the 'boxpoints' attribute of the Box traces
    outlier_buttons = [
        dict(
            label="Outliers: OFF (Fast)", 
            method="restyle", 
            args=[{"boxpoints": False}] # Hides points
        ),
        dict(
            label="Outliers: ON (Slow)", 
            method="restyle", 
            args=[{"boxpoints": "outliers"}] # Shows outlier points
        )
    ]

    fig.update_layout(
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="v", y=0.99, x=1.01, bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1, itemclick="toggle", itemdoubleclick="toggleothers"),
        margin=dict(l=80, r=200, t=180, b=80),
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        paper_bgcolor="white",
        width=1400,
        height=1400,
        
        xaxis2=dict(matches='x'),
        yaxis2=dict(matches='y', showticklabels=True),
        yaxis3=dict(title_text="Length (cM)"),
        
        updatemenus=[
             dict(buttons=dropdown_buttons, direction="down", showactive=True, x=0.0, xanchor="left", y=1.10, bgcolor="#ffffff"),
             dict(buttons=scale_buttons, direction="right", showactive=True, x=1.0, xanchor="right", y=1.10, bgcolor="#ffffff"),
             dict(buttons=smoothing_buttons, direction="right", showactive=True, x=0.35, xanchor="center", y=1.10, bgcolor="#ffffff"),
             dict(buttons=zoom_buttons, direction="right", showactive=True, x=0.55, xanchor="center", y=1.10, bgcolor="#ffffff"),
             # New Button: Outlier Toggle
             dict(buttons=outlier_buttons, direction="right", showactive=True, x=0.75, xanchor="center", y=1.10, bgcolor="#ffffff")
         ]
    )

    # Titles and Annotations
    for r in [1, 2]:
        fig.update_xaxes(title_text="Length (centiMorgans)", showgrid=True, row=r, col=1)
        fig.update_yaxes(title_text="Percentage (%)", showgrid=True, row=r, col=1)
    fig.update_xaxes(title_text="Kinship Group", showgrid=True, row=3, col=1)

    fig.add_annotation(text="<b>Select Threshold:</b>", x=0.0, y=1.14, xref="paper", yref="paper", xanchor="left", showarrow=False, font=dict(size=12))
    fig.add_annotation(text="<b>Scale:</b>", x=1.0, y=1.14, xref="paper", yref="paper", xanchor="right", showarrow=False, font=dict(size=12))
    fig.add_annotation(text="<b>Smoothing:</b>", x=0.35, y=1.14, xref="paper", yref="paper", xanchor="center", showarrow=False, font=dict(size=12))
    fig.add_annotation(text="<b>Zoom Sync:</b>", x=0.55, y=1.14, xref="paper", yref="paper", xanchor="center", showarrow=False, font=dict(size=12))
    fig.add_annotation(text="<b>Outlier Data:</b>", x=0.75, y=1.14, xref="paper", yref="paper", xanchor="center", showarrow=False, font=dict(size=12))

    output_file = 'kinship_interactive_studio.html'
    print(f"\nSaving optimized studio to {output_file}...")
    fig.write_html(output_file, include_plotlyjs='cdn', full_html=True, config=dict(responsive=True, displayModeBar=True, displaylogo=False))
    
    print("\n" + "="*70)
    print("✓ SUCCESS: Lag Reduced & Outlier Toggle Added!")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate interactive kinship studio visualization')
    parser.add_argument('--data-dir', type=str, default='new', help="Data directory name")
    args = parser.parse_args()
    main(args.data_dir)