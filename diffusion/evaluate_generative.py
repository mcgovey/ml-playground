import pandas as pd
import numpy as np
import pickle
import os
import json
import plotly.graph_objects as go
from plotly.io import to_html
from scipy.stats import ks_2samp, chi2_contingency
import fire


def main(real_path, generated_path, encoders_path='processed/label_encoders.pkl', scaler_path='processed/scaler.pkl', num_cols_path='processed/numerical_cols.pkl', binary_cols_path='processed/binary_cols.pkl', output_dir='processed'):
    """
    Evaluate generative model output by comparing distributions of real and generated data.
    Saves per-column statistics, statistical test results, and a comprehensive HTML report with all visualizations and actionable insights.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(num_cols_path, 'rb') as f:
        num_cols = pickle.load(f)
    with open(binary_cols_path, 'rb') as f:
        binary_cols = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)

    real = pd.read_parquet(real_path)
    gen = pd.read_parquet(generated_path)

    # Identify categorical columns (encoded as int, not binary)
    cat_cols = [col for col in label_encoders.keys() if col not in binary_cols]

    results = {}
    plot_html_blocks = []
    summary_rows = []
    toc_links = []
    significant_cols = []

    for col in real.columns:
        results[col] = {}
        anchor = f"col_{col}"
        # Continuous/numerical columns
        if col in num_cols:
            ks_stat, ks_p = ks_2samp(real[col], gen[col])
            results[col]['real_mean'] = float(real[col].mean())
            results[col]['gen_mean'] = float(gen[col].mean())
            results[col]['real_std'] = float(real[col].std())
            results[col]['gen_std'] = float(gen[col].std())
            results[col]['ks_stat'] = float(ks_stat)
            results[col]['ks_p'] = float(ks_p)
            sig = ks_p < 0.05
            sig_color = 'red' if sig else 'green'
            sig_text = "Yes" if sig else "No"
            summary_rows.append(
                f'<tr><td><a href="#{anchor}">{col}</a></td><td>Numerical</td><td>{ks_p:.3g}</td>'
                f'<td style="color:{sig_color}">{sig_text}</td></tr>'
            )
            if sig:
                significant_cols.append(col)
            toc_links.append(f'<li><a href="#{anchor}">{col}</a></li>')
            # Plot histogram with Plotly
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=real[col], nbinsx=30, name='Real', opacity=0.5, histnorm='probability density'))
            fig.add_trace(go.Histogram(x=gen[col], nbinsx=30, name='Generated', opacity=0.5, histnorm='probability density'))
            fig.update_layout(
                title=f"{col} (KS p={ks_p:.3g})",
                barmode='overlay',
                xaxis_title=col,
                yaxis_title='Density',
                legend=dict(x=0.7, y=0.95)
            )
            block = f'<a id="{anchor}"></a>'
            block += f'<h2>{col} (Numerical)</h2>'
            block += f'<pre><b>Mean (real):</b> {results[col]["real_mean"]:.4f} | <b>Mean (gen):</b> {results[col]["gen_mean"]:.4f}\n' \
                     f'<b>Std (real):</b> {results[col]["real_std"]:.4f} | <b>Std (gen):</b> {results[col]["gen_std"]:.4f}\n' \
                     f'<b>KS stat:</b> {ks_stat:.4f} | <b>KS p-value:</b> <span title="p < 0.05 means significant difference" style="color:{sig_color}">{ks_p:.4g}</span></pre>'
            block += to_html(fig, include_plotlyjs=False, full_html=False)
            plot_html_blocks.append(block)
        # Binary/categorical columns
        elif col in binary_cols or col in cat_cols:
            real_counts = real[col].value_counts().sort_index()
            gen_counts = gen[col].value_counts().sort_index()
            all_idx = sorted(set(real_counts.index) | set(gen_counts.index))
            real_counts = real_counts.reindex(all_idx, fill_value=0)
            gen_counts = gen_counts.reindex(all_idx, fill_value=0)
            if len(all_idx) > 1:
                chi2, p, _, _ = chi2_contingency([real_counts, gen_counts])
            else:
                chi2, p = np.nan, np.nan
            results[col]['real_dist'] = {int(k): int(v) for k, v in real_counts.items()}
            results[col]['gen_dist'] = {int(k): int(v) for k, v in gen_counts.items()}
            results[col]['chi2'] = float(chi2)
            results[col]['chi2_p'] = float(p)
            sig = (not np.isnan(p)) and (p < 0.05)
            sig_color = 'red' if sig else 'green'
            sig_text = "Yes" if sig else "No"
            summary_rows.append(
                f'<tr><td><a href="#{anchor}">{col}</a></td><td>Categorical/Binary</td><td>{p:.3g}</td>'
                f'<td style="color:{sig_color}">{sig_text}</td></tr>'
            )
            if sig:
                significant_cols.append(col)
            toc_links.append(f'<li><a href="#{anchor}">{col}</a></li>')
            # Plot bar chart with Plotly
            fig = go.Figure()
            fig.add_trace(go.Bar(x=all_idx, y=real_counts, name='Real'))
            fig.add_trace(go.Bar(x=all_idx, y=gen_counts, name='Generated'))
            fig.update_layout(
                title=f"{col} (Chi2 p={p:.3g})",
                xaxis_title=col,
                yaxis_title='Count',
                barmode='group',
                legend=dict(x=0.7, y=0.95)
            )
            block = f'<a id="{anchor}"></a>'
            block += f'<h2>{col} (Categorical/Binary)</h2>'
            block += f'<pre><b>Value counts (real):</b> {results[col]["real_dist"]}\n<b>Value counts (gen):</b> {results[col]["gen_dist"]}\n' \
                     f'<b>Chi2 stat:</b> {chi2:.4f} | <b>Chi2 p-value:</b> <span title="p < 0.05 means significant difference" style="color:{sig_color}">{p:.4g}</span></pre>'
            block += to_html(fig, include_plotlyjs=False, full_html=False)
            plot_html_blocks.append(block)
        else:
            # Unknown type, just plot histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=real[col], nbinsx=30, name='Real', opacity=0.5, histnorm='probability density'))
            fig.add_trace(go.Histogram(x=gen[col], nbinsx=30, name='Generated', opacity=0.5, histnorm='probability density'))
            fig.update_layout(
                title=f"{col}",
                barmode='overlay',
                xaxis_title=col,
                yaxis_title='Density',
                legend=dict(x=0.7, y=0.95)
            )
            block = f'<a id="{anchor}"></a>'
            block += f'<h2>{col} (Unknown Type)</h2>'
            block += to_html(fig, include_plotlyjs=False, full_html=False)
            plot_html_blocks.append(block)

    # Save results as JSON
    with open(os.path.join(output_dir, 'generative_eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Compose a summary section
    summary_table = (
        '<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; margin:auto;">'
        '<thead style="background:#f0f0f0;">'
        '<tr><th>Column</th><th>Type</th><th>p-value</th><th>Significant Difference?</th></tr>'
        '</thead>'
        '<tbody>'
        + ''.join(summary_rows) +
        '</tbody>'
        '</table>'
    )
    summary_text = (
        '<div style="margin-bottom:20px;">'
        '<b>Key Takeaways:</b><br>'
        '<ul>'
        f'<li><b>{len(significant_cols)}</b> out of <b>{len(real.columns)}</b> columns show a statistically significant distribution difference (p &lt; 0.05).</li>'
        '<li>See the table below for details. Click column names to jump to their section.</li>'
        '</ul>'
        '</div>'
    )
    toc_html = (
        '<div id="toc" style="position:sticky;top:0;background:#fff;padding:10px 0 10px 0;z-index:1000;">'
        '<b>Jump to column:</b><ul style="margin:0;">' + ''.join(toc_links) + '</ul></div>'
    )

    # Compose a single HTML report
    html_report = f"""
    <html>
    <head>
        <title>Generative Model Evaluation Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0 10%; }}
            h1 {{ text-align: center; }}
            h2 {{ margin-top: 40px; }}
            pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; }}
            #toc {{ background: #f9f9f9; border-bottom: 1px solid #ddd; }}
            table {{ width: 100%; margin-bottom: 30px; }}
            th, td {{ text-align: left; }}
            th {{ background: #f0f0f0; }}
        </style>
    </head>
    <body>
        <h1>Generative Model Evaluation Report</h1>
        <p>Real data: <code>{os.path.basename(real_path)}</code><br>
        Generated data: <code>{os.path.basename(generated_path)}</code></p>
        {summary_text}
        {summary_table}
        {toc_html}
        {''.join(plot_html_blocks)}
    </body>
    </html>
    """
    with open(os.path.join(output_dir, 'generative_eval_report.html'), 'w') as f:
        f.write(html_report)
    print(f"Evaluation complete. Results and comprehensive report saved to {output_dir}")

if __name__ == '__main__':
    fire.Fire(main)
