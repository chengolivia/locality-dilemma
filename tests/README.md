# Unit Tests & Verification Suite

This directory contains test suites for validating the benchmark harnesses, offline workplace trace processors, and analysis modules.

## Test Files

- `test_workplace_tools.py`: Tests the event reconstruction logic in `build_workplace_replay.py` (verifying token overlap calculation and asserting zero future-action leakage) and preflight integrity checks in `verify_workplace_setup.py`.
- `test_benchmarks.py`: Tests the callable benchmark interfaces (`measure_t_swap`, `measure_t_prefill`), parameter calculations, and summary statistics in lightweight CPU dry-run mode.

## Running Tests

### Standard Python / Unittest
Using the active virtual environment:
```bash
python -m unittest discover tests -v
```
Or with a specific Python interpreter:
```bash
~/venvs/vllm/bin/python -m unittest discover tests -v
```

### Pytest (if installed)
```bash
pytest tests/ -v
```
