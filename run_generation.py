#!/usr/bin/env python3
"""vLLM offline inference smoke test; timings here are NOT true TTFT or swap timings."""
import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/base')
    p.add_argument('--workloads', default='data/workloads.jsonl')
    p.add_argument('--output', default='results/generations.jsonl')
    p.add_argument('--adapter', help='Optional local LoRA adapter directory')
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--max-tokens', type=int, default=16)
    p.add_argument('--max-lora-rank', type=int, default=64)
    p.add_argument('--limit', type=int, default=5)
    a = p.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    records = [json.loads(line) for line in Path(a.workloads).open()]
    records = records[:a.limit] if a.limit > 0 else records
    lora = LoRARequest('experiment_adapter', 1, a.adapter) if a.adapter else None
    llm = LLM(model=a.model, enable_lora=bool(lora), max_lora_rank=a.max_lora_rank,
              max_model_len=a.max_model_len, enforce_eager=True)
    params = SamplingParams(temperature=0, max_tokens=a.max_tokens, ignore_eos=True)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for row in records:
            request = {'prompt_token_ids': row['prompt_token_ids']}
            start = time.perf_counter()
            result = llm.generate([request], params, lora_request=lora, use_tqdm=False)[0]
            elapsed_ms = (time.perf_counter() - start) * 1000
            record = {'request_id': row['request_id'], 'prefix_len': row['prefix_len'],
                      'adapter': a.adapter or 'base', 'generated_tokens': len(result.outputs[0].token_ids),
                      'text': result.outputs[0].text, 'end_to_end_wall_ms': elapsed_ms}
            f.write(json.dumps(record) + '\n')
            print(f"{record['request_id']}: {elapsed_ms:.1f} ms, {record['generated_tokens']} tokens", flush=True)
    print('Note: offline wall-clock duration includes engine overhead; use Olivia\'s event harness for kernel timings and true TTFT.')

if __name__ == '__main__':
    main()
