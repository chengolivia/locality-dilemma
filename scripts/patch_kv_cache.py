#!/usr/bin/env python3
"""Helper script to inspect, apply, and restore vLLM KV-cache hashing patch.

The patch controls whether LoRA adapter identity is included in the KV-cache
block hash:
- Native: LoRA adapter name and path are included in the hash key.
- Forced (patched): _gen_lora_extra_hash_keys returns [], removing LoRA
  identity and allowing cross-adapter KV-cache hits.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

DEFAULT_FILE = Path.home() / "venvs/vllm/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py"

NATIVE_FUNC = '''def _gen_lora_extra_hash_keys(request: Request) -> list[tuple[str, str, str]]:
    """Generate extra keys related to LoRA for block hash computation.

    The adapter path is included so that re-pointing a LoRA name at a different
    adapter does not reuse KV computed with the previous one.

    Args:
        request: The request object.

    Returns:
        Return the LoRA name and path of the request if it is a LoRA request.
        Return empty list otherwise.

    """
    lora_request = request.lora_request
    if not lora_request:
        return []
    return [("lora", lora_request.lora_name, lora_request.lora_path)]'''

PATCHED_FUNC = '''def _gen_lora_extra_hash_keys(request: Request) -> list[tuple[str, str, str]]:
    """Generate extra keys related to LoRA for block hash computation.

    The adapter path is included so that re-pointing a LoRA name at a different
    adapter does not reuse KV computed with the previous one.

    Args:
        request: The request object.

    Returns:
        Return the LoRA name and path of the request if it is a LoRA request.
        Return empty list otherwise.

    """
    return []'''

PATTERN = re.compile(
    r"def _gen_lora_extra_hash_keys\(request:\s*Request\)\s*->\s*list\[tuple\[str,\s*str,\s*str\]\]:.*?(?=\n\n(?:def|\S|\Z))",
    re.DOTALL
)


def get_status(content: str) -> str:
    match = PATTERN.search(content)
    if not match:
        return "not_found"
    block = match.group(0)
    try:
        mod = ast.parse(block)
        fn = mod.body[0]
        stmts = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
        if len(stmts) == 1 and isinstance(stmts[0], ast.Return) and isinstance(stmts[0].value, ast.List) and len(stmts[0].value.elts) == 0:
            return "forced"
        if any(isinstance(s, ast.Assign) for s in stmts):
            return "native"
    except Exception:
        pass
    return "unknown"


def extract_func(content: str) -> str:
    match = PATTERN.search(content)
    return match.group(0) if match else "Function not found"


def main():
    parser = argparse.ArgumentParser(description="Manage vLLM KV-cache hashing patch")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="Path to kv_cache_utils.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="Check current patch status")
    group.add_argument("--native", "--restore", action="store_true", help="Restore native vLLM implementation")
    group.add_argument("--patch", "--forced", action="store_true", help="Apply forced cross-adapter sharing patch")
    args = parser.parse_args()

    target = args.file
    if not target.is_file():
        print(f"Error: Target file {target} does not exist.", file=sys.stderr)
        sys.exit(1)

    content = target.read_text()
    status = get_status(content)

    if args.status:
        print(f"File: {target}")
        print(f"Status: {status.upper()}")
        print("\nCurrent function implementation:")
        print(extract_func(content))
        return

    bak_file = target.with_suffix(".py.bak")

    if args.native:
        if status == "native":
            print(f"Already in NATIVE mode at {target}")
            return
        new_content, count = PATTERN.subn(NATIVE_FUNC, content)
        if count != 1:
            print(f"Error: could not find single function pattern to replace (matches={count})", file=sys.stderr)
            sys.exit(1)
        ast.parse(new_content)  # Syntax check
        target.write_text(new_content)
        # Also ensure backup exists with native content
        bak_file.write_text(new_content)
        print(f"Successfully restored NATIVE cache hashing in {target}")
        print(f"Native backup updated at {bak_file}")

    elif args.patch:
        if status == "forced":
            print(f"Already in FORCED (patched) mode at {target}")
            return
        # Ensure we have a backup of native if not present
        if not bak_file.is_file():
            bak_file.write_text(content)
            print(f"Created backup at {bak_file}")
        new_content, count = PATTERN.subn(PATCHED_FUNC, content)
        if count != 1:
            print(f"Error: could not find single function pattern to replace (matches={count})", file=sys.stderr)
            sys.exit(1)
        ast.parse(new_content)  # Syntax check
        target.write_text(new_content)
        print(f"Successfully applied FORCED (patched) cache hashing in {target}")


if __name__ == "__main__":
    main()
