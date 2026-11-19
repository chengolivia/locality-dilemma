"""Small offline checks: python -m unittest tests/test_workplace_tools.py -v"""
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

# Ensure repo root and workplace scripts are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_WORKPLACE_DIR = _REPO_ROOT / "scripts" / "workplace"
if str(_WORKPLACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKPLACE_DIR))

import build_workplace_replay as replay
import verify_workplace_setup as verify


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class WorkplaceTests(unittest.TestCase):
    def test_replay_order_overlap_and_no_future_leakage(self):
        events = [
            {'id': 0, 'source': 'user', 'action': 'message', 'args': {'content': 'TASK'}},
            {'id': 1, 'source': 'agent', 'action': 'run', 'args': {'command': 'FIRST_ACTION'}},
            {'id': 2, 'source': 'agent', 'observation': 'run', 'content': 'TOOL_RESULT'},
            {'id': 3, 'source': 'agent', 'action': 'run', 'args': {'command': 'FUTURE_ACTION'}},
        ]
        a = list(replay.requests({'task_id': 'a', 'adapter_id': 'a1', 'role': 'coding'}, events, Tokenizer()))
        b = list(replay.requests({'task_id': 'b', 'adapter_id': 'a2', 'role': 'qa'}, events, Tokenizer()))
        self.assertNotIn('FIRST_ACTION', a[0]['prompt'])
        self.assertIn('TOOL_RESULT', a[1]['prompt'])
        self.assertNotIn('FUTURE_ACTION', a[1]['prompt'])
        self.assertIn('workplace task', a[0]['prompt'])
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'replay.jsonl'
            result = replay.write_replay(path, a+b, 10000, 32)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['task_id'] for r in rows], ['a', 'b', 'a', 'b'])
            self.assertEqual(result['requests_per_role'], {'coding': 2, 'qa': 2})
            self.assertEqual(rows[1]['shared_prefix_tokens'], 0)
            self.assertGreater(rows[2]['shared_prefix_tokens'], 0)
            short = replay.write_replay(path, a+b, len(a[0]['prompt_token_ids'])+32, 32)
            self.assertEqual(short['requests'], 2)

    def test_preflight_checks_five_adapters_and_source_paths(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / 'repo'
            repo.mkdir()
            adapters = root / 'adapters'
            tasks = []
            for role in range(5):
                for n in range(4):
                    name = f'task-{role}-{n}'
                    with gzip.open(repo / f'{name}.json.gz', 'wt') as f:
                        json.dump([], f)
                    tasks.append({'task_id': name, 'role': str(role), 'adapter_id': f'p/a{role}',
                                  'trajectory_path_in_source_repo': f'{name}.json.gz'})
                folder = adapters / f'p__a{role}'
                folder.mkdir(parents=True)
                (folder / 'adapter_config.json').write_text(json.dumps({
                    'base_model_name_or_path': 'base', 'peft_type': 'LORA', 'r': 8,
                    'target_modules': ['q_proj'],
                }))
                header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
                (folder / 'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(header))+header+b'\0'*4)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'base_model': 'base', 'tasks': tasks}))
            original = Path.cwd()
            try:
                os.chdir(root)
                argv = ['verify', '--manifest', str(manifest), '--source-repo', str(repo), '--adapters-dir', str(adapters)]
                with patch('sys.argv', argv), patch.object(verify.subprocess, 'check_output', return_value='a'*40), contextlib.redirect_stdout(io.StringIO()):
                    verify.main()
                result = json.loads(Path('reports/adapter_manifest_20tasks.json').read_text())
                self.assertEqual(len(result), 5)
                self.assertTrue(all(r['tensor_bytes'] == 4 for r in result))
                bad = adapters / 'p__a0/adapter_config.json'
                config = json.loads(bad.read_text())
                config['base_model_name_or_path'] = 'wrong-base'
                bad.write_text(json.dumps(config))
                with patch('sys.argv', argv), patch.object(verify.subprocess, 'check_output', return_value='a'*40):
                    with self.assertRaisesRegex(ValueError, 'Base-model mismatch'):
                        verify.main()
            finally:
                os.chdir(original)


if __name__ == '__main__':
    unittest.main()
