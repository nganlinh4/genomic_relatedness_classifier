"""
Generate interactive HTML dashboard with embedded kinship analysis plots.
All images are Base64-encoded and embedded in a single HTML file.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import base64
import io


def load_percentile_file(kinship_level, threshold):
    """Load individual percentile TSV file."""
    kinship_str = str(kinship_level)
    file_path = f'data/raw/new/cm_over_{threshold}/percentile_{kinship_str}.tsv'
    
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path, sep='\t')


def filter_un_data(df):
    """Filter UN kinship data to remove duplicate pairs."""
    duplicate_patterns = []
    for i in range(1, 7):
        pattern = f"[{i}-1_vs_{i}-2]"
        duplicate_patterns.append(pattern)
    
    mask = ~df['PAIR_ID'].isin(duplicate_patterns)
    filtered_df = df[mask].copy()
    return filtered_df


def create_length_distribution(df):
    """Create distribution data from percentile columns."""
    percentile_cols = [col for col in df.columns if col.endswith('%')]
    percentile_cols = sorted(percentile_cols, key=lambda x: float(x.rstrip('%')))
    
    all_values = []
    for col in percentile_cols:
        all_values.extend(df[col].dropna().values)
    
    if len(all_values) == 0:
        return {'lengths': [], 'counts': [], 'percentages': [], 'total': 0, 'max_length': 0}
    
    bin_size = 0.05
    max_val = max(all_values)
    bin_edges = np.arange(0, max_val + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2
    
    counts = np.histogram(all_values, bins=bin_edges)[0]
    total = len(all_values)
    percentages = (counts / total * 100) if total > 0 else counts
    
    return {
        'lengths': bin_centers.tolist(),
        'counts': counts.tolist(),
        'percentages': percentages.tolist(),
        'total': total,
        'max_length': max_val
    }


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str


def generate_individual_plot(kinship_num, df, threshold, global_max_length=None):
    """Generate individual plot and return as base64 string."""
    dist = create_length_distribution(df)
    
    if not dist['lengths']:
        return None, dist

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#4C78A8'
    ax1.set_xlabel('LENGTH_CM (centiMorgans)', fontsize=12)
    ax1.set_ylabel('Count', color=color1, fontsize=12)
    line1 = ax1.plot(dist['lengths'], dist['counts'], color=color1, linewidth=1, markersize=3, label='Count')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    if global_max_length:
        ax1.set_xlim(-global_max_length * 0.02, global_max_length * 1.05)
    
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    
    ax2 = ax1.twinx()
    color2 = '#E45756'
    ax2.set_ylabel('Percentage (%)', color=color2, fontsize=12)
    line2 = ax2.plot(dist['lengths'], dist['percentages'], color=color2, linewidth=1, markersize=3, label='Percentage', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    kinship_label = f'Kinship {kinship_num}' if kinship_num != 'UN' else 'Kinship UN'
    plt.title(f'{kinship_label} (cM > {threshold}, n={dist["total"]})', fontsize=14, pad=15)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    b64_img = fig_to_base64(fig)
    plt.close(fig)
    
    return b64_img, dist


def generate_combined_plot(all_distributions, threshold):
    """Generate combined plot and return as base64 string."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#4C78A8', '#E45756', '#54A24B', '#F58518', '#B279A2', '#FF9DA6', '#9D755D']
    kinship_nums = [1, 2, 3, 4, 5, 6, 'UN']
    labels = [f'Kinship {k}' for k in kinship_nums]
    
    max_length = 0
    has_data = False
    
    for kinship_num, color, label in zip(kinship_nums, colors, labels):
        key = f'kinship_{kinship_num}'
        if key in all_distributions and all_distributions[key]['lengths']:
            dist = all_distributions[key]
            ax.plot(dist['lengths'], dist['percentages'], color=color, linewidth=1.2, markersize=4, label=label, alpha=0.8)
            max_length = max(max_length, dist['max_length'])
            has_data = True
            
    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel('LENGTH_CM (centiMorgans)', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Combined Distribution Comparison (cM > {threshold})', fontsize=14, pad=15)
    ax.set_xlim(-max_length * 0.02, max_length * 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, alpha=0.3, which='major')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    b64_img = fig_to_base64(fig)
    plt.close(fig)
    return b64_img


def generate_html(data_store):
    """
    Generates a single HTML file with embedded Base64 images and JS for navigation.
    """
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kinship Length Analysis Dashboard</title>
    <style>
        :root { --primary: #2c3e50; --accent: #3498db; --bg: #f8f9fa; }
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: var(--bg); color: #333; }
        header { background-color: var(--primary); color: white; padding: 1.5rem 2rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { margin: 0; font-size: 1.8rem; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        /* Tabs */
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #ddd; padding-bottom: 10px; flex-wrap: wrap; }
        .tab-btn { padding: 10px 20px; background: #e9ecef; border: none; border-radius: 5px; cursor: pointer; font-size: 1rem; transition: 0.2s; }
        .tab-btn:hover { background: #dee2e6; }
        .tab-btn.active { background: var(--accent); color: white; font-weight: bold; }
        
        /* Content Layout */
        .threshold-section { display: none; animation: fadeIn 0.3s; }
        .threshold-section.active { display: block; }
        
        .summary-box { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; }
        .summary-table { width: 100%; border-collapse: collapse; }
        .summary-table th { background-color: var(--accent); color: white; padding: 12px; text-align: left; }
        .summary-table td { padding: 10px; border-bottom: 1px solid #eee; }
        .summary-table tr:hover { background-color: #f9f9f9; }
        
        .plot-container { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; text-align: center; }
        .plot-container img { max-width: 100%; height: auto; border-radius: 4px; }
        .plot-container h3 { margin: 0 0 15px 0; color: var(--primary); }
        
        .grid-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); gap: 20px; }
        
        h2 { border-left: 5px solid var(--accent); padding-left: 10px; margin-top: 30px; color: var(--primary); }
        
        footer { text-align: center; padding: 20px; color: #777; font-size: 0.9rem; border-top: 1px solid #ddd; margin-top: 40px; }
        
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
    <script>
        function openThreshold(evt, thresholdId) {
            var i, x, tablinks;
            x = document.getElementsByClassName("threshold-section");
            for (i = 0; i < x.length; i++) {
                x[i].className = x[i].className.replace(" active", "");
            }
            tablinks = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(thresholdId).className += " active";
            evt.currentTarget.className += " active";
        }
    </script>
</head>
<body>
    <header>
        <h1>Kinship Length Distribution Analysis Dashboard</h1>
    </header>
    
    <div class="container">
        <div class="tabs">
"""
    
    # Create Tab buttons
    thresholds = sorted(data_store.keys())
    for i, thresh in enumerate(thresholds):
        active_class = " active" if i == 0 else ""
        html_content += f'            <button class="tab-btn{active_class}" onclick="openThreshold(event, \'thresh_{thresh}\')">Threshold > {thresh} cM</button>\n'
    
    html_content += "        </div>\n"
    
    # Create Content for each threshold
    for i, thresh in enumerate(thresholds):
        active_class = " active" if i == 0 else ""
        data = data_store[thresh]
        
        html_content += f'        <div id="thresh_{thresh}" class="threshold-section{active_class}">\n'
        
        # Combined Plot Section
        if data['combined_plot']:
            html_content += f"""            <div class="plot-container" style="max-width: 900px; margin: 0 auto 30px auto;">
                <h3>Combined Kinship Overview (cM > {thresh})</h3>
                <img src="data:image/png;base64,{data['combined_plot']}" alt="Combined Plot">
            </div>\n"""
        
        # Summary Table
        html_content += """            <div class="summary-box">
                <h3>Dataset Summary</h3>
                <table class="summary-table">
                <tr>
                    <th>Kinship</th>
                    <th>Total Pairs</th>
                    <th>Max Length (cM)</th>
                    <th>Max Percentage Bin</th>
                </tr>\n"""
        
        for k_key in ['kinship_1', 'kinship_2', 'kinship_3', 'kinship_4', 'kinship_5', 'kinship_6', 'kinship_UN']:
            if k_key in data['stats']:
                stats = data['stats'][k_key]
                if stats['total'] > 0:
                    max_pct = max(stats['percentages']) if stats['percentages'] else 0
                    max_len = stats['max_length']
                    label = k_key.replace('kinship_', 'Kinship ')
                    html_content += f"                <tr><td>{label}</td><td>{stats['total']}</td><td>{max_len:.2f}</td><td>{max_pct:.2f}%</td></tr>\n"
        
        html_content += """                </table>
            </div>\n"""
        
        # Individual Plots Grid
        html_content += '            <h2>Individual Distributions</h2>\n            <div class="grid-gallery">\n'
        for k in [1, 2, 3, 4, 5, 6, 'UN']:
            key = f'kinship_{k}'
            if key in data['plots']:
                html_content += f"""                <div class="plot-container">
                    <img src="data:image/png;base64,{data['plots'][key]}" alt="{key}">
                </div>\n"""
        html_content += '            </div>\n        </div>\n'

    html_content += """    </div>
    <footer>
        Generated by Kinship Analysis Dashboard Script | All plots embedded in single HTML file
    </footer>
</body>
</html>"""
    
    output_path = 'kinship_analysis_dashboard.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n{'='*70}")
    print(f"SUCCESS: Dashboard created at: {os.path.abspath(output_path)}")
    print(f"{'='*70}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"Open in any web browser to view interactive dashboard")


def main():
    thresholds = [1, 3, 6, 10]
    data_store = {}
    
    print("\n" + "="*70)
    print("Kinship Analysis Dashboard Generator")
    print("="*70)
    
    for threshold in thresholds:
        print(f"\nProcessing Threshold > {threshold} cM...")
        data_store[threshold] = {
            'plots': {},
            'stats': {},
            'combined_plot': None
        }
        
        all_distributions = {}
        kinship_data = {}
        global_max = 0
        
        # Load Data
        for kinship_num in range(1, 7):
            df = load_percentile_file(kinship_num, threshold)
            if df is not None:
                kinship_data[kinship_num] = df
        
        df_un = load_percentile_file('UN', threshold)
        if df_un is not None:
            kinship_data['UN'] = df_un
            
        # Determine Global Max for Axis Scaling
        for k, df in kinship_data.items():
            percentile_cols = [col for col in df.columns if col.endswith('%')]
            if percentile_cols:
                global_max = max(global_max, df[percentile_cols].max().max())

        # Generate Individual Plots
        # Kinships 1-6
        for k in range(1, 7):
            df = kinship_data.get(k)
            if df is not None:
                b64, dist = generate_individual_plot(k, df, threshold, global_max)
                if b64:
                    data_store[threshold]['plots'][f'kinship_{k}'] = b64
                    data_store[threshold]['stats'][f'kinship_{k}'] = dist
                    all_distributions[f'kinship_{k}'] = dist
                    print(f"  ✓ Kinship {k} plot generated")
        
        # Kinship UN
        if 'UN' in kinship_data:
            df_un = filter_un_data(kinship_data['UN'])
            b64, dist = generate_individual_plot('UN', df_un, threshold, global_max)
            if b64:
                data_store[threshold]['plots']['kinship_UN'] = b64
                data_store[threshold]['stats']['kinship_UN'] = dist
                all_distributions['kinship_UN'] = dist
                print(f"  ✓ Kinship UN plot generated")
                
        # Generate Combined Plot
        combined_b64 = generate_combined_plot(all_distributions, threshold)
        data_store[threshold]['combined_plot'] = combined_b64
        print(f"  ✓ Combined plot generated")

    generate_html(data_store)


if __name__ == '__main__':
    main()
