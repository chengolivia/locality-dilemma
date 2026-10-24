#!/usr/bin/env python3
"""Download base and PEFT LoRA adapters with the Hugging Face CLI."""
import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_BASE = 'mistralai/Mistral-7B-v0.1'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default=DEFAULT_BASE)
    p.add_argument('--adapters', nargs='*', default=[], help='Hugging Face LoRA repo IDs')
    p.add_argument('--output', default='models')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    root = Path(a.output)
    entries = [('base', a.base, root / 'base')]
    entries += [('adapter', repo, root / 'adapters' / repo.replace('/', '__')) for repo in a.adapters]
    for kind, repo, path in entries:
        cmd = ['hf', 'download', repo, '--local-dir', str(path)]
        print('+', ' '.join(cmd), flush=True)
        if a.dry_run:
            continue
        subprocess.run(cmd, check=True)
        if a.dry_run or kind != 'adapter':
            continue
        config_path = path / 'adapter_config.json'
        if not config_path.exists():
            raise RuntimeError(f'{repo}: missing adapter_config.json, not a standard PEFT adapter')
        config = json.loads(config_path.read_text())
        if config.get('peft_type', '').upper() != 'LORA':
            raise RuntimeError(f'{repo}: PEFT type is not LoRA')
        configured_base = config.get('base_model_name_or_path', '')
        if configured_base and configured_base.rstrip('/') != a.base.rstrip('/'):
            raise RuntimeError(f'{repo}: base {configured_base!r} does not match {a.base!r}; investigate before inference')
        print(f'Validated {repo}: rank={config.get("r")}, targets={config.get("target_modules")}')
    if not a.dry_run:
        print(f'Finished. Base model: {root / "base"}. Adapters: {root / "adapters"}')

if __name__ == '__main__':
    main()
