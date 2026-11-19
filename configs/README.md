# Experiment Configurations

This directory contains declarative configuration files for experiments and benchmarks across the Locality Dilemma repository.

## Files

- [`real_workload.yaml`](real_workload.yaml): Centralized configuration for real-world workload replay (TheAgentCompany workplace tasks). Specifies base model, 5 verified LoRA adapters, role-to-adapter mapping, vLLM serving arguments, replay ordering strategies (`adapter_interleaved` vs. `adapter_grouped`), metrics specifications, and output path templates.
