#!/usr/bin/env python3
"""Generate fixed-length, deterministic tokenized prompts for prefix experiments."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', default='models/base', help='Local base directory or HF model ID')
    p.add_argument('--lengths', type=int, nargs='+', default=[128, 256, 512, 1024, 2048])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', default='data/workloads.jsonl')
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True)
    # Stable non-random prefix. Use tokenizer-native IDs to ensure exact prefix lengths.
    corpus = ('Research notes: systems evaluate cache locality and scheduling tradeoffs. '
              'The following report compares prefill computation, adapter swaps, and latency. ') * 3000
    ids = tok.encode(corpus, add_special_tokens=False)
    max_len = max(a.lengths)
    if len(ids) < max_len:
        raise RuntimeError('Corpus too short for requested prefix length')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for length in a.lengths:
            if length <= 0:
                raise ValueError('Lengths must be positive')
            prefix_ids = ids[:length]
            for rep in range(a.repeats):
                row = {'request_id': f'L{length}_R{rep}', 'prefix_len': length,
                       'repeat': rep, 'prompt_token_ids': prefix_ids,
                       'prompt': tok.decode(prefix_ids, skip_special_tokens=False)}
                f.write(json.dumps(row) + '\n')
    print(f'Created {len(a.lengths)*a.repeats} records at {out}')
    print('Use prompt_token_ids for exact lengths; decoded prompt may retokenize differently.')

if __name__ == '__main__':
    main()
