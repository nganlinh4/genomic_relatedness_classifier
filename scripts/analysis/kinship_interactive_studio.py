"""
Generate fully interactive Plotly dashboard with all visualization controls.
Single HTML file with:
- Threshold selector dropdown
- Interactive legend (click to hide/show kinship levels)
- Zoom, pan, hover tooltips
- Linear/Log scale toggle
- No server needed
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def load_percentile_file(kinship_level, threshold):
    """Load individual percentile TSV file."""
    kinship_str = str(kinship_level)
    file_path = f'data/raw/new/cm_over_{threshold}/percentile_{kinship_str}.tsv'
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

    # Create bins (0.05 cM granularity)
    bin_size = 0.05
    max_val = max(all_values)
    bin_edges = np.arange(0, max_val + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2
    
    counts = np.histogram(all_values, bins=bin_edges)[0]
    total = len(all_values)
    percentages = (counts / total * 100) if total > 0 else counts
    
    return bin_centers, percentages, counts, total


def main():
    thresholds = [1, 3, 6, 10]
    kinship_levels = [1, 2, 3, 4, 5, 6, 'UN']
    colors = ['#4C78A8', '#E45756', '#54A24B', '#F58518', '#B279A2', '#FF9DA6', '#9D755D']
    
    print("\n" + "="*70)
    print("Generating Interactive Plotly Studio...")
    print("="*70)

    # Initialize Figure
    fig = go.Figure()
    
    total_traces = len(thresholds) * len(kinship_levels)
    print(f"Processing {total_traces} traces ({len(thresholds)} thresholds × {len(kinship_levels)} kinship levels)...")

    trace_idx = 0
    for t_idx, threshold in enumerate(thresholds):
        print(f"\n  Threshold > {threshold} cM:")
        
        # Visibility: Only first threshold visible by default
        is_visible = (t_idx == 0)

        for k_idx, kinship in enumerate(kinship_levels):
            # Load Data
            df = load_percentile_file(kinship, threshold)
            
            if df is None:
                # Add placeholder trace to maintain index alignment
                fig.add_trace(go.Scatter(
                    x=[], y=[], 
                    visible=False, 
                    showlegend=False,
                    hoverinfo='skip'
                ))
                trace_idx += 1
                continue

            # Filter UN duplicates
            if kinship == 'UN':
                df = filter_un_data(df)

            # Calculate Statistics
            x_vals, y_pct, y_count, total = get_distribution_data(df)

            if x_vals is None:
                # Add placeholder
                fig.add_trace(go.Scatter(
                    x=[], y=[], 
                    visible=False, 
                    showlegend=False,
                    hoverinfo='skip'
                ))
                trace_idx += 1
                continue

            # Add Trace with full interactivity
            fig.add_trace(go.Scatter(
                x=x_vals, 
                y=y_pct,
                mode='lines',
                name=f'Kinship {kinship}',
                line=dict(color=colors[k_idx], width=1.5),
                visible=is_visible,
                customdata=np.column_stack((y_count, [total]*len(y_count))),
                hovertemplate=(
                    "<b>Kinship %s (cM > %d)</b><br>" % (kinship, threshold) +
                    "Length: %{x:.2f} cM<br>" +
                    "Percentage: %{y:.2f}%<br>" +
                    "Count: %{customdata[0]:,.0f} pairs (of %{customdata[1]:.0f} total)<br>" +
                    "<extra></extra>"
                )
            ))
            
            print(f"    ✓ Kinship {kinship}: {total} pairs, max {max(y_pct):.2f}%")
            trace_idx += 1

    # --- Create Interactive Controls ---

    # 1. Threshold Dropdown Menu
    print("\nConfiguring interactive controls...")
    dropdown_buttons = []
    for i, threshold in enumerate(thresholds):
        # Create visibility array
        visibility = [False] * total_traces
        
        # Set visibility for this threshold's traces
        start_idx = i * len(kinship_levels)
        end_idx = start_idx + len(kinship_levels)
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

    # 2. Scale Toggle Buttons (Linear vs Log)
    scale_buttons = [
        dict(
            label="Linear Scale",
            method="relayout",
            args=[{"yaxis.type": "linear", "yaxis.title": "Percentage (%)"}]
        ),
        dict(
            label="Log Scale",
            method="relayout",
            args=[{"yaxis.type": "log", "yaxis.title": "Percentage (%) [log scale]"}]
        )
    ]

    # Layout with controls
    fig.update_layout(
        title={
            "text": "Kinship Distribution Analysis (Threshold > 1 cM)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20}
        },
        xaxis=dict(
            title="Length (centiMorgans)",
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='#cccccc'
        ),
        yaxis=dict(
            title="Percentage (%)",
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='#cccccc'
        ),
        updatemenus=[
            # Threshold Selector
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="#ffffff",
                bordercolor="#cccccc",
                borderwidth=1
            ),
            # Scale Toggle
            dict(
                buttons=scale_buttons,
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top",
                bgcolor="#ffffff",
                bordercolor="#cccccc",
                borderwidth=1
            )
        ],
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#cccccc",
            borderwidth=1
        ),
        margin=dict(
            l=80,
            r=200,  # Extra space for legend
            t=120,  # Space for controls
            b=80
        ),
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        paper_bgcolor="white",
        width=1400,
        height=700
    )

    # Annotations for button labels
    fig.add_annotation(
        text="<b>Select Threshold:</b>",
        x=0.0,
        y=1.22,
        xref="paper",
        yref="paper",
        xanchor="left",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.add_annotation(
        text="<b>Scale:</b>",
        x=1.0,
        y=1.22,
        xref="paper",
        yref="paper",
        xanchor="right",
        showarrow=False,
        font=dict(size=12)
    )

    # Save to HTML
    output_file = 'kinship_interactive_studio.html'
    print(f"\nSaving interactive studio to {output_file}...")
    fig.write_html(
        output_file, 
        include_plotlyjs='cdn',
        full_html=True,
        config=dict(
            responsive=True,
            displayModeBar=True,
            displaylogo=False,
            modeBarButtonsToRemove=['lasso2d', 'select2d']
        )
    )
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    
    print("\n" + "="*70)
    print("✓ SUCCESS: Interactive Studio Created!")
    print("="*70)
    print(f"\nFile: {output_file}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Location: {os.path.abspath(output_file)}")
    print("\n" + "="*70)
    print("How to use:")
    print("  1. Open kinship_interactive_studio.html in any web browser")
    print("  2. Use dropdown to switch between thresholds (cM > 1, 3, 6, 10)")
    print("  3. Click legend entries to toggle kinship levels on/off")
    print("  4. Click 'Linear Scale' or 'Log Scale' to change Y-axis")
    print("  5. Hover over lines to see exact values")
    print("  6. Click and drag to zoom, double-click to reset")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
