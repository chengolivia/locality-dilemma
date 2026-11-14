#!/usr/bin/env python3
"""Verify the selected trace files and inspect the five real LoRA weight headers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='reports/theagentcompany_20tasks.json')
    p.add_argument('--source-repo', default='data/theagentcompany-experiments')
    p.add_argument('--adapters-dir', default='models/adapters')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    tasks = manifest['tasks']
    counts = Counter(t['role'] for t in tasks)
    if len({t['task_id'] for t in tasks}) != 20 or len(counts) != 5 or set(counts.values()) != {4}:
        raise ValueError('Expected 20 distinct tasks and five roles of four tasks each')
    repo = Path(a.source_repo).resolve()
    for task in tasks:
        path = (repo / task['trajectory_path_in_source_repo']).resolve()
        if not path.is_relative_to(repo) or not path.is_file():
            raise FileNotFoundError(f"Missing or invalid trace for {task['task_id']}: {path}")
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if manifest.get('source_commit') and manifest['source_commit'] != revision:
        raise ValueError('Source commit does not match manifest')

    records = []
    for adapter in sorted({t['adapter_id'] for t in tasks}):
        folder = Path(a.adapters_dir) / adapter.replace('/', '__')
        config_path = folder / 'adapter_config.json'
        if not config_path.is_file():
            raise FileNotFoundError(f'Missing {config_path}; download {adapter} first')
        config = json.loads(config_path.read_text())
        if config.get('base_model_name_or_path') != manifest['base_model']:
            raise ValueError(f'Base-model mismatch: {adapter}')
        if config.get('peft_type', '').upper() != 'LORA':
            raise ValueError(f'Not a LoRA: {adapter}')
        files = sorted(folder.glob('*.safetensors'))
        if not files:
            raise FileNotFoundError(f'No Safetensors weights: {adapter}')
        tensors = []
        for file in files:
            with file.open('rb') as f:
                size_bytes = f.read(8)
                if len(size_bytes) != 8:
                    raise ValueError(f'Incomplete Safetensors file: {file}')
                header_size = struct.unpack('<Q', size_bytes)[0]
                if header_size > file.stat().st_size - 8:
                    raise ValueError(f'Incomplete Safetensors header: {file}')
                header = json.loads(f.read(header_size))
            payload_size = file.stat().st_size - 8 - header_size
            for name, tensor in header.items():
                if name == '__metadata__':
                    continue
                start, end = tensor['data_offsets']
                if not 0 <= start <= end <= payload_size:
                    raise ValueError(f'Invalid tensor offsets in {file}: {name}')
                tensors.append({'name': name, 'shape': tensor['shape'], 'dtype': tensor['dtype'], 'bytes': end-start})
        record = {
            'adapter': adapter, 'base_model': config['base_model_name_or_path'],
            'rank': config['r'], 'target_modules': config['target_modules'],
            'stored_dtypes': sorted({t['dtype'] for t in tensors}),
            'tensor_count': len(tensors), 'tensor_bytes': sum(t['bytes'] for t in tensors),
            'weight_file_bytes': sum(f.stat().st_size for f in files),
            'config_sha256': digest(config_path),
            'weight_sha256': {f.name: digest(f) for f in files}, 'tensors': tensors,
            'validation_scope': 'Local headers, content hashes and declared base checked; engine loading and inference not verified',
        }
        records.append(record)
        print(f"{adapter}: rank={record['rank']} dtypes={record['stored_dtypes']} tensor_MB={record['tensor_bytes']/1e6:.2f}")

    outputs = [Path('reports/theagentcompany_20tasks_verified.json'), Path('reports/adapter_manifest_20tasks.json')]
    if not a.overwrite and any(f.exists() for f in outputs):
        raise FileExistsError('Verified output exists; use --overwrite only if replacement is intended')
    manifest['source_commit'] = revision
    manifest['local_preflight'] = 'Trace files exist; five LoRA configurations and headers inspected; GPU execution untested'
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(manifest, indent=2) + '\n')
    outputs[1].write_text(json.dumps(records, indent=2) + '\n')
    print('PASS: 20 trace files, 5 roles, 5 adapters. Source commit:', revision)
    for path in outputs:
        print('Saved:', path)


if __name__ == '__main__':
    main()
