# TensorPlan

[![Tests](https://github.com/terrywangcode/tensorplan/actions/workflows/test.yml/badge.svg)](https://github.com/terrywangcode/tensorplan/actions/workflows/test.yml)
[![Deploy](https://github.com/terrywangcode/tensorplan/actions/workflows/pages.yml/badge.svg)](https://github.com/terrywangcode/tensorplan/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TensorPlan is a dependency-free browser tool for estimating GPU capacity and throughput for local or on-premise open-weight LLM inference.

**Public tool:** https://terrywangcode.github.io/tensorplan/

## What it calculates

- Model weights including quantization metadata
- Architecture-aware KV cache and recurrent state
- Runtime workspace and usable-VRAM reserve
- Exact GPU requirements, including requirements above a preferred maximum
- Compute and memory-bandwidth throughput ceilings
- Per-request generation speed at configured concurrency
- Tensor-parallel interconnect efficiency

## Model and hardware support

The built-in catalog includes Qwen, GLM, DeepSeek, Gemma, Llama, Mistral, and gpt-oss profiles plus NVIDIA consumer/datacenter GPUs and AMD MI300X. Public `config.json` files can be loaded from ModelScope, with Hugging Face as fallback.

TensorPlan understands conventional dense transformers, MoE, GQA, MLA, and hybrid full/linear-attention configurations. Every result displays whether its model profile is verified or estimated.

## Run locally

```bash
git clone git@github.com:terrywangcode/tensorplan.git
cd tensorplan
python3 -m http.server 4173
```

Open `http://localhost:4173`.

No installation, build step, account, analytics, or backend is required.

## Tests

```bash
node --check app.js
node --check model-config.js
node model-config.test.js
node static-site.test.js
```

## Calculation policy

Memory sizing reserves the full configured context for every concurrent request. Runtime overhead includes a per-GPU workspace floor, weight/cache-related workspace, and multi-GPU communication buffers.

Decode throughput uses a roofline-style estimate and takes the lower of active-parameter compute capacity and memory bandwidth after weight batching and cache reads. Runtime, traffic profile, batching, and interconnect factors are empirical planning assumptions, not hardware guarantees.

Always benchmark the exact checkpoint, runtime, prompt distribution, and sampling settings before procurement.

## Registry lookup limitations

Static browser deployments depend on the registry permitting cross-origin configuration requests. Hugging Face generally supports this for public repositories. ModelScope availability may depend on its current browser-access policy or the user's network.

TensorPlan prefers machine-readable or model-card parameter counts when repositories publish them. Otherwise it adaptively estimates the architecture from common and emerging configuration fields, including dense-prefix MoE layouts, latent and shared experts, MLA, KDA, and hybrid attention layer maps, and clearly labels the confidence level.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Incorrect model configurations and reproducible GPU benchmark data are particularly useful.

## License

[MIT](LICENSE)
