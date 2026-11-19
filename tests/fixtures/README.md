# Test Fixtures

This directory contains synthetic fixtures for unit tests and local verification.

## Files

- [`mock_workload_measurements_test_fixture.jsonl`](mock_workload_measurements_test_fixture.jsonl): A 10-request mock dataset containing 5 requests under `adapter_interleaved` and 5 requests under `adapter_grouped`. Used exclusively to test analysis and measurement serialization without requiring a live GPU server. Includes realistic edge cases:
  - Missing `mean_itl_ms` (`null`) on single-token completions.
  - Missing Prometheus cache counters (`null`) to test non-zero-imputing error handling.
  - Prefix cache hits with reduced TTFT under same-adapter grouped requests.
  - Every record explicitly contains `"synthetic": true`.

> **Note:** These records are strictly synthetic test fixtures and must never be merged with real empirical measurement results.
