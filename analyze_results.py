#!/usr/bin/env python3
"""Analyze measurements exported by the GPU timing harness.
Expected CSV columns: prefix_len,lora_rank,prefill_saved_ms,swap_ms
Each row represents paired or comparable measured costs (NOT end-to-end wall time).
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='results/timings.csv')
    p.add_argument('--output-dir', default='results/analysis')
    a = p.parse_args()
    df = pd.read_csv(a.input)
    required = {'prefix_len', 'lora_rank', 'prefill_saved_ms', 'swap_ms'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}. Do not use run_generation.py wall times in place of these costs.')
    for key in required:
        df[key] = pd.to_numeric(df[key], errors='raise')
    if (df[['prefill_saved_ms', 'swap_ms']] < 0).any().any():
        raise ValueError('Costs must be non-negative')
    group = df.groupby(['lora_rank', 'prefix_len'], as_index=False).agg(
        prefill_saved_ms=('prefill_saved_ms', 'median'),
        swap_ms=('swap_ms', 'median'), n=('swap_ms', 'size'))
    group['delta_ms'] = group['prefill_saved_ms'] - group['swap_ms']
    group['winner'] = group['delta_ms'].map(lambda x: 'prefix-first' if x > 0 else 'adapter-first' if x < 0 else 'tie')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    group.to_csv(out / 'summary.csv', index=False)
    pivot = group.pivot(index='lora_rank', columns='prefix_len', values='delta_ms').sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    img = ax.imshow(pivot.to_numpy(), aspect='auto', cmap='coolwarm', interpolation='nearest')
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel('Prefix length (tokens)')
    ax.set_ylabel('LoRA rank')
    ax.set_title('Prefill saved − adapter swap (ms): positive favors prefix-first')
    fig.colorbar(img, ax=ax, label='Difference (ms)')
    fig.tight_layout()
    fig.savefig(out / 'latency_heatmap.png', dpi=160)
    plt.close(fig)
    # First interpolated sign change per rank, if measurements bracket an intersection.
    boundaries = []
    for rank, subset in group.groupby('lora_rank'):
        subset = subset.sort_values('prefix_len')
        pairs = list(subset[['prefix_len', 'delta_ms']].itertuples(index=False, name=None))
        crossings = []
        for (x1, y1), (x2, y2) in zip(pairs, pairs[1:]):
            if y1 == 0:
                crossings.append(float(x1))
            elif y1 * y2 < 0:
                crossings.append(x1 + (0 - y1) * (x2 - x1) / (y2 - y1))
        if pairs and pairs[-1][1] == 0:
            crossings.append(float(pairs[-1][0]))
        boundaries.append({'lora_rank': rank, 'break_even_prefix_tokens': crossings[0] if crossings else float('nan'),
                           'crossing_count': len(crossings)})
    pd.DataFrame(boundaries).to_csv(out / 'break_even.csv', index=False)
    print(f'Saved {out / "summary.csv"}, {out / "break_even.csv"}, and heatmap.')
    print('NaN break-even means no observed crossing; multiple crossings may indicate noise/non-monotonicity.')

if __name__ == '__main__':
    main()
