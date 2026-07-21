# Changelog

All notable changes to TensorPlan are documented here.

## [1.0.0.0] - 2026-07-21

### Added

- Estimate model weights, KV cache, runtime workspace, GPU count, and decode throughput in one browser-based tool.
- Compare common Qwen, GLM, DeepSeek, Gemma, Llama, Mistral, and gpt-oss model profiles across consumer and datacenter GPUs.
- Load public model configurations from ModelScope with Hugging Face fallback.
- Account for dense, MoE, GQA, MLA, and hybrid linear/full-attention architectures.
- Save deployment scenarios locally and export calculation plans as JSON.
- Deploy automatically to GitHub Pages with calculation and static-site checks on every change.

### Fixed

- Normalize nested model configurations and repository-declared parameter counts.
- Report exact GPU requirements beyond the preferred maximum.
- Prevent invalid reset state from generating unbounded GPU counts.
