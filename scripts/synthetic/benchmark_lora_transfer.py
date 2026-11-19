#!/usr/bin/env python3
"""
PCIe Host-to-Device (H2D) Transfer Latency Microbenchmark for LoRA Adapters.

This script measures the adapter swap time (T_swap) when transferring LoRA adapter
weights from host (CPU) memory to GPU (CUDA) memory across different LoRA ranks
(8, 16, 32, 64, 128).

Measurement Methodology:
-------------------------
1. State Dictionary Simulation:
   - For each rank r in [8, 16, 32, 64, 128], generates dummy state dicts representing
     a transformer model (default: 32 layers, hidden dimension 4096, 4 attention projections
     `q_proj`, `k_proj`, `v_proj`, `o_proj` in float16/bfloat16 precision).
   - In each module, lora_A has shape (r, d_in) and lora_B has shape (d_out, r).
   - Weights reside in CPU host memory (optionally pinned for hardware DMA).

2. GPU Transfer Timing:
   - Uses `torch.cuda.Event(enable_timing=True)` to record timestamps on the CUDA stream:
       start_event.record()
       gpu_state_dict = {k: v.to(device='cuda', non_blocking=...) for k, v in cpu_state_dict.items()}
       end_event.record()
       torch.cuda.synchronize()
       latency_ms = start_event.elapsed_time(end_event)
   - Performs a warmup phase (default: 10 iterations) to prime CUDA context and driver queues.
   - Executes 100 timed iterations per rank.
   - Clears the PyTorch CUDA memory cache (`torch.cuda.empty_cache()`) between runs.

3. Reporting:
   - Computes median transfer latency T_swap in milliseconds (ms), along with mean,
     standard deviation, min, max, and 95th percentile (P95).
   - Computes effective PCIe transfer bandwidth (GB/s = Total Bytes / T_swap).
   - Prints a formatted summary table and optionally exports results to JSON or CSV.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

# Prevent OpenMP / threading deadlocks on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

try:
    import torch
except ImportError:
    torch = None


def get_pcie_link_info(device_idx: int = 0) -> Optional[str]:
    """
    Attempt to query NVML (via nvidia-ml-py or pynvml) for PCIe generation and link width.
    Returns None if NVML is unavailable (e.g., on macOS or non-GPU environments).
    """
    try:
        import pynvml
    except ImportError:
        try:
            from pynvml import nvml as pynvml
        except ImportError:
            return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
        width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        pynvml.nvmlShutdown()
        return f"Gen{gen} x{width}"
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PCIe Host-to-Device transfer latency for LoRA adapters."
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="List of LoRA ranks to benchmark (default: 8 16 32 64 128).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of transformer layers to simulate (e.g., 32 for LLaMA-7B/8B; default: 32).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Model hidden dimension d_model (default: 4096).",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target linear projections per layer (default: q_proj k_proj v_proj o_proj).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Tensor data type for adapter weights (default: float16).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed iterations per rank (default: 100).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before timing (default: 10).",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin CPU tensors in page-locked host memory for hardware DMA (default: True).",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned memory (use standard pageable host memory).",
    )
    parser.add_argument(
        "--transfer-mode",
        type=str,
        choices=["to_device", "copy_preallocated"],
        default="to_device",
        help="Transfer mechanism: 'to_device' (allocates GPU memory dynamically) or "
        "'copy_preallocated' (in-place copy into pre-allocated GPU buffers). Default: to_device.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Target CUDA device string (default: cuda:0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution on CPU (useful for syntax/flow testing on machines without CUDA).",
    )
    parser.add_argument(
        "--adapters-dir",
        type=str,
        default=None,
        help="Path to directory containing downloaded LoRA adapters (e.g. models/adapters from download_models.py).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to a single downloaded LoRA adapter directory (e.g. models/adapters/repo__name).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to export benchmark results as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to export benchmark results as CSV.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> "torch.dtype":
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]


def create_dummy_lora_state_dict(
    rank: int,
    num_layers: int,
    hidden_dim: int,
    target_modules: List[str],
    dtype: "torch.dtype",
    pin_memory: bool = True,
    device: str = "cpu",
) -> Dict[str, "torch.Tensor"]:
    """
    Generate a dummy state dictionary simulating LoRA adapter weights.

    For each layer and target linear module:
      - lora_A: shape (rank, hidden_dim)
      - lora_B: shape (hidden_dim, rank)

    Tensors are initialized in host (CPU) memory and optionally pinned.
    """
    state_dict = {}
    for layer_idx in range(num_layers):
        for module in target_modules:
            key_a = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
            key_b = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"

            tensor_a = torch.randn(rank, hidden_dim, dtype=dtype, device=device)
            tensor_b = torch.randn(hidden_dim, rank, dtype=dtype, device=device)

            if pin_memory and torch.cuda.is_available() and device == "cpu":
                tensor_a = tensor_a.pin_memory()
                tensor_b = tensor_b.pin_memory()

            state_dict[key_a] = tensor_a
            state_dict[key_b] = tensor_b

    return state_dict


def load_real_peft_adapter(
    adapter_path_str: str,
    dtype: "torch.dtype",
    pin_memory: bool = True,
) -> Tuple[int, str, Dict[str, "torch.Tensor"]]:
    """
    Loads real PEFT LoRA adapter weights downloaded via download_models.py.
    Returns (rank, adapter_name, state_dict).
    """
    from pathlib import Path
    adapter_path = Path(adapter_path_str)
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")

    with open(config_file, "r") as f:
        config = json.load(f)

    rank = config.get("r", 8)
    adapter_name = adapter_path.name

    safetensors_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"

    state_dict = {}
    if safetensors_path.exists():
        from safetensors.torch import load_file
        raw_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        raw_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {adapter_path}")

    for k, v in raw_dict.items():
        t = v.to(dtype=dtype)
        if pin_memory and torch.cuda.is_available():
            t = t.pin_memory()
        state_dict[k] = t

    return rank, adapter_name, state_dict


def calculate_state_dict_size(state_dict: Dict[str, "torch.Tensor"]) -> Tuple[int, float]:
    """
    Calculates total number of parameters and total size in Megabytes (MB).
    """
    total_params = sum(t.numel() for t in state_dict.values())
    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    size_mb = total_bytes / (1024.0 * 1024.0)
    return total_params, size_mb


def compute_statistics(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Computes summary statistics for a list of latency measurements (in ms).
    """
    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    # Median calculation
    if n % 2 == 1:
        median = sorted_latencies[n // 2]
    else:
        median = (sorted_latencies[n // 2 - 1] + sorted_latencies[n // 2]) / 2.0

    mean = sum(sorted_latencies) / n
    variance = sum((x - mean) ** 2 for x in sorted_latencies) / (n - 1) if n > 1 else 0.0
    std_dev = math.sqrt(variance)

    def percentile(p: float) -> float:
        idx = int(round(p * (n - 1)))
        return sorted_latencies[idx]

    return {
        "median": median,
        "mean": mean,
        "std": std_dev,
        "min": sorted_latencies[0],
        "max": sorted_latencies[-1],
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
    }


def benchmark_rank_cuda(
    cpu_state_dict: Dict[str, "torch.Tensor"],
    device: "torch.device",
    iterations: int = 100,
    warmup: int = 10,
    pin_memory: bool = True,
    transfer_mode: str = "to_device",
) -> Tuple[List[float], Dict[str, float]]:
    """
    Benchmark Host-to-Device transfer latency on an active CUDA GPU.

    Uses `torch.cuda.Event(enable_timing=True)` and `torch.cuda.synchronize()`
    to capture the exact GPU DMA transfer duration.
    """
    non_blocking = pin_memory

    # Pre-allocate GPU buffers if transfer_mode is copy_preallocated
    preallocated_gpu_dict = None
    if transfer_mode == "copy_preallocated":
        preallocated_gpu_dict = {
            k: torch.empty_like(v, device=device) for k, v in cpu_state_dict.items()
        }

    # -------------------------------------------------------------------------
    # 1. Warmup Phase
    # -------------------------------------------------------------------------
    for _ in range(warmup):
        if transfer_mode == "to_device":
            gpu_state_dict = {
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in cpu_state_dict.items()
            }
            torch.cuda.synchronize(device)
            del gpu_state_dict
            torch.cuda.empty_cache()
        else:
            for k, v in cpu_state_dict.items():
                preallocated_gpu_dict[k].copy_(v, non_blocking=non_blocking)
            torch.cuda.synchronize(device)

    # -------------------------------------------------------------------------
    # 2. Timed Iterations
    # -------------------------------------------------------------------------
    latencies_ms: List[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        # Synchronize before recording start event to flush any prior GPU kernels
        torch.cuda.synchronize(device)

        start_event.record()

        if transfer_mode == "to_device":
            # Move all tensors from CPU memory to device='cuda'
            gpu_state_dict = {
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in cpu_state_dict.items()
            }
            end_event.record()
            torch.cuda.synchronize(device)

            # Measure elapsed time via CUDA events in milliseconds
            elapsed_ms = start_event.elapsed_time(end_event)
            latencies_ms.append(elapsed_ms)

            # Free GPU memory and clear cache between runs
            del gpu_state_dict
            torch.cuda.empty_cache()
        else:
            for k, v in cpu_state_dict.items():
                preallocated_gpu_dict[k].copy_(v, non_blocking=non_blocking)
            end_event.record()
            torch.cuda.synchronize(device)

            elapsed_ms = start_event.elapsed_time(end_event)
            latencies_ms.append(elapsed_ms)

    if preallocated_gpu_dict is not None:
        del preallocated_gpu_dict
        torch.cuda.empty_cache()

    stats = compute_statistics(latencies_ms)
    return latencies_ms, stats


def benchmark_rank_dryrun(
    cpu_state_dict: Dict[str, "torch.Tensor"],
    iterations: int = 100,
    warmup: int = 10,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Fallback CPU dry-run timing for validation on environments without CUDA.
    """
    for _ in range(warmup):
        _ = {k: v.clone() for k, v in cpu_state_dict.items()}

    latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        copied = {k: v.clone() for k, v in cpu_state_dict.items()}
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)
        del copied

    return latencies_ms, compute_statistics(latencies_ms)


def measure_t_swap(
    rank: int,
    hidden_dim: int = 4096,
    num_layers: int = 32,
    target_modules: Optional[List[str]] = None,
    dtype: Optional["torch.dtype"] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    iterations: int = 100,
    warmup: int = 10,
    pin_memory: bool = True,
    transfer_mode: str = "to_device",
    return_stats: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Callable benchmark function: measure_t_swap(rank, hidden_dim).

    Measures the PCIe Host-to-Device transfer latency (T_swap) in milliseconds
    for loading a LoRA adapter with the specified rank and hidden_dim onto GPU.

    Args:
        rank: LoRA rank (e.g., 8, 16, 32, 64, 128).
        hidden_dim: Model hidden dimension d_model (default: 4096).
        num_layers: Number of transformer layers to simulate (default: 32).
        target_modules: Projection names (default: ['q_proj', 'k_proj', 'v_proj', 'o_proj']).
        dtype: PyTorch tensor dtype (default: float16 on CUDA, float32 on CPU).
        device: Target CUDA device or 'cpu' (default: cuda:0 if available, else cpu).
        iterations: Number of timed iterations (default: 100).
        warmup: Number of untimed warmup iterations (default: 10).
        pin_memory: Whether to pin host memory for DMA (default: True).
        transfer_mode: 'to_device' or 'copy_preallocated' (default: 'to_device').
        return_stats: If True, returns (median_ms, stats_dict). Otherwise returns median_ms as float.

    Returns:
        float: Median transfer latency T_swap in milliseconds.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    if dtype is None:
        dtype = torch.float16 if target_device.type == "cuda" else torch.float32

    # Generate state dict on host (CPU)
    state_dict = create_dummy_lora_state_dict(
        rank=rank,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        target_modules=target_modules,
        dtype=dtype,
        pin_memory=(pin_memory and target_device.type == "cuda"),
        device="cpu",
    )

    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        _, stats = benchmark_rank_cuda(
            cpu_state_dict=state_dict,
            device=target_device,
            iterations=iterations,
            warmup=warmup,
            pin_memory=pin_memory,
            transfer_mode=transfer_mode,
        )
    else:
        _, stats = benchmark_rank_dryrun(
            cpu_state_dict=state_dict,
            iterations=iterations,
            warmup=warmup,
        )

    median_ms = stats["median"]
    if return_stats:
        return median_ms, stats
    return median_ms


def main():
    args = parse_args()

    if torch is None:
        print("Error: PyTorch is required to run this script. Please install `torch`.", file=sys.stderr, flush=True)
        sys.exit(1)

    cuda_available = torch.cuda.is_available()

    if not cuda_available and not args.dry_run:
        print("=" * 80, file=sys.stderr, flush=True)
        print("ERROR: CUDA is not available on this machine.", file=sys.stderr, flush=True)
        print("This benchmark measures Host-to-Device transfer latency using", file=sys.stderr, flush=True)
        print("`torch.cuda.Event` and requires an NVIDIA GPU with CUDA enabled.", file=sys.stderr, flush=True)
        print("To test the benchmarking script logic on CPU, rerun with `--dry-run`.", file=sys.stderr, flush=True)
        print("=" * 80, file=sys.stderr, flush=True)
        sys.exit(1)

    dtype = get_torch_dtype(args.dtype)
    is_dry_run = not cuda_available or args.dry_run

    print("=" * 88, flush=True)
    print("      PyTorch PCIe Host-to-Device LoRA Transfer Latency Microbenchmark      ", flush=True)
    print("=" * 88, flush=True)

    if not is_dry_run:
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        device_name = torch.cuda.get_device_name(device)
        pcie_link = get_pcie_link_info(device.index or 0)
        print(f"Device:                 {device} ({device_name})", flush=True)
        print(f"CUDA Version:           {torch.version.cuda}", flush=True)
        if pcie_link:
            print(f"PCIe Link Status:       {pcie_link}", flush=True)
    else:
        device = torch.device("cpu")
        device_name = "cpu_dry_run"
        print(f"Device:                 CPU (DRY RUN SIMULATION - CUDA NOT ACTIVE)", flush=True)

    print(f"PyTorch Version:        {torch.__version__}", flush=True)
    print(f"LoRA Ranks:             {args.ranks}", flush=True)
    print(f"Base Architecture:      {args.num_layers} layers, {args.hidden_dim} hidden dimension", flush=True)
    print(f"Target Modules:         {', '.join(args.target_modules)} ({len(args.target_modules)} projections/layer)", flush=True)
    print(f"Precision:              {args.dtype}", flush=True)
    print(f"Transfer Mode:          {args.transfer_mode}", flush=True)
    print(f"Pinned Memory (DMA):    {args.pin_memory and not is_dry_run}", flush=True)
    print(f"Warmup Iterations:      {args.warmup}", flush=True)
    print(f"Timed Iterations:       {args.iterations} per rank", flush=True)
    print("=" * 88, flush=True)
    print(flush=True)

    benchmark_results = []

    # Check if testing downloaded real adapters from download_models.py
    adapter_items = []
    if args.adapter:
        from pathlib import Path
        p = Path(args.adapter)
        if p.exists():
            adapter_items.append((p.name, p))
    elif args.adapters_dir:
        from pathlib import Path
        dir_path = Path(args.adapters_dir)
        if dir_path.exists():
            for p in sorted(dir_path.iterdir()):
                if p.is_dir() and (p / "adapter_config.json").exists():
                    adapter_items.append((p.name, p))

    if adapter_items:
        print(f"[*] Found {len(adapter_items)} real downloaded LoRA adapter(s) to benchmark.\n", flush=True)
        for adapter_name, adapter_path in adapter_items:
            print(f"[*] Benchmark Real Adapter: {adapter_name}", flush=True)
            rank, _, state_dict = load_real_peft_adapter(
                adapter_path_str=str(adapter_path),
                dtype=dtype,
                pin_memory=args.pin_memory if not is_dry_run else False,
            )
            total_params, size_mb = calculate_state_dict_size(state_dict)
            size_bytes = size_mb * (1024.0 * 1024.0)
            print(f"    - Inferred Rank:      {rank}", flush=True)
            print(f"    - Parameter Count:    {total_params:,}", flush=True)
            print(f"    - Adapter Size:       {size_mb:.2f} MB", flush=True)
            print(f"    - Profiling {args.warmup} warmup + {args.iterations} timed transfer iterations...", flush=True)

            if not is_dry_run:
                _, stats = benchmark_rank_cuda(
                    cpu_state_dict=state_dict,
                    device=device,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    pin_memory=args.pin_memory,
                    transfer_mode=args.transfer_mode,
                )
            else:
                _, stats = benchmark_rank_dryrun(
                    cpu_state_dict=state_dict,
                    iterations=args.iterations,
                    warmup=args.warmup,
                )

            median_ms = stats["median"]
            median_sec = median_ms / 1000.0
            effective_bandwidth = (size_bytes / median_sec) / 1e9 if median_sec > 0 else 0.0

            stats["rank"] = rank
            stats["adapter_name"] = adapter_name
            stats["total_params"] = total_params
            stats["size_mb"] = size_mb
            stats["effective_bandwidth_gbps"] = effective_bandwidth
            benchmark_results.append(stats)
            print(f"    >> [{adapter_name}] Rank {rank:3d}: Median T_swap = {median_ms:.3f} ms | Bandwidth = {effective_bandwidth:.2f} GB/s\n", flush=True)
    else:
        for rank in args.ranks:
            print(f"[*] Benchmark LoRA Rank = {rank}", flush=True)
            state_dict = create_dummy_lora_state_dict(
                rank=rank,
                num_layers=args.num_layers,
                hidden_dim=args.hidden_dim,
                target_modules=args.target_modules,
                dtype=dtype,
                pin_memory=args.pin_memory if not is_dry_run else False,
                device="cpu",
            )

            total_params, size_mb = calculate_state_dict_size(state_dict)
            size_bytes = size_mb * (1024.0 * 1024.0)

            print(f"    - Parameter Count:    {total_params:,}", flush=True)
            print(f"    - Adapter Size:       {size_mb:.2f} MB", flush=True)
            print(f"    - Profiling {args.warmup} warmup + {args.iterations} timed transfer iterations...", flush=True)

            if not is_dry_run:
                _, stats = benchmark_rank_cuda(
                    cpu_state_dict=state_dict,
                    device=device,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    pin_memory=args.pin_memory,
                    transfer_mode=args.transfer_mode,
                )
            else:
                _, stats = benchmark_rank_dryrun(
                    cpu_state_dict=state_dict,
                    iterations=args.iterations,
                    warmup=args.warmup,
                )

            # Calculate effective PCIe bandwidth in GB/s: Size (Bytes) / (T_swap in seconds) / 1e9
            median_ms = stats["median"]
            median_sec = median_ms / 1000.0
            effective_bandwidth = (size_bytes / median_sec) / 1e9 if median_sec > 0 else 0.0

            stats["rank"] = rank
            stats["total_params"] = total_params
            stats["size_mb"] = size_mb
            stats["effective_bandwidth_gbps"] = effective_bandwidth
            benchmark_results.append(stats)

            print(f"    >> Rank {rank:3d}: Median T_swap = {median_ms:.3f} ms | Bandwidth = {effective_bandwidth:.2f} GB/s\n", flush=True)

    # -------------------------------------------------------------------------
    # Summary Table Output
    # -------------------------------------------------------------------------
    header = (
        f"{'Rank':<6} | {'Params':<11} | {'Size (MB)':<10} | "
        f"{'Median T_swap (ms)':<18} | {'Mean ± Std (ms)':<20} | "
        f"{'Min / Max (ms)':<17} | {'P95 (ms)':<10} | {'Bandwidth (GB/s)':<16}"
    )
    separator = "-" * len(header)

    print("=" * len(header), flush=True)
    print(header, flush=True)
    print(separator, flush=True)
    for res in benchmark_results:
        mean_std = f"{res['mean']:.3f} ± {res['std']:.3f}"
        min_max = f"{res['min']:.2f} / {res['max']:.2f}"
        print(
            f"{res['rank']:<6} | "
            f"{res['total_params']:<11,d} | "
            f"{res['size_mb']:<10.2f} | "
            f"{res['median']:<18.3f} | "
            f"{mean_std:<20} | "
            f"{min_max:<17} | "
            f"{res['p95']:<10.3f} | "
            f"{res['effective_bandwidth_gbps']:<16.2f}",
            flush=True,
        )
    print("=" * len(header), flush=True)
    print(flush=True)

    # Optional JSON / CSV Export
    if args.output_json:
        metadata = {
            "config": vars(args),
            "cuda_available": cuda_available,
            "device_name": device_name if not is_dry_run else "cpu_dry_run",
            "results": benchmark_results,
        }
        with open(args.output_json, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[+] Benchmark results exported to JSON: {args.output_json}")

    if args.output_csv:
        csv_fields = [
            "rank",
            "total_params",
            "size_mb",
            "median",
            "mean",
            "std",
            "min",
            "max",
            "p25",
            "p75",
            "p95",
            "effective_bandwidth_gbps",
        ]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for res in benchmark_results:
                writer.writerow({k: res[k] for k in csv_fields})
        print(f"[+] Benchmark results exported to CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

