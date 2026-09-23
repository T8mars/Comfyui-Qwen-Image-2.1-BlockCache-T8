# Sources and attribution

Copyright 2026 T8mars. The code in this repository is licensed under Apache-2.0.

- Execution-scoped first-block caching and prefetch cleanup adapt the author's [MiniMax H3 Block Cache](https://github.com/T8mars/comfyui-minimax-h3-blockcache-T8), Apache-2.0. The new adapter targets Qwen-Image-2.1's single stream and native output head.
- Spectral prediction is an independent, reduced Qwen-specific implementation of Chebyshev/ridge feature forecasting, informed by [Spectrum](https://github.com/hanjq17/Spectrum), Jiaqi Han et al. (MIT), and the history-weight formulation reviewed in [Comfyui-Spectrum-Qwen2.1](https://github.com/awdqwdasdg/Comfyui-Spectrum-Qwen2.1) (MIT). It does not reproduce the complete original schedule/controller, and no upstream speed/quality claim is inherited.
- SageAttention and Sol kernels are called through installed ComfyUI/[Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) APIs. No CUDA/Triton kernels or KJNodes source are bundled. [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) was reviewed for interoperability; its GPL code is not relicensed here.
- [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) and its weights retain their own [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE). This plugin's license does not change model permissions. Models and original inference source are not distributed here.

Native Core reference: `Comfy-Org/ComfyUI@e638023d54497dbe0579565e5de4bb7076899592`.
Spectrum algorithm reference: `hanjq17/Spectrum@4e41f91c93ff66b045332530dd7bcb4c60eafbab`, `src/utils/basis_utils.py`.
