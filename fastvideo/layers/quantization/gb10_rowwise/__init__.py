# SPDX-License-Identifier: Apache-2.0
"""GB10 (SM121) FP8 GEMM with per-token activation and per-output-channel weight scales.

gb10_cute_fp8_kernel.py is sglang's GB10 CuTe kernel, copied verbatim from georgeliu95/sglang
opt/h3-dgx-spark (5b1fa7a9e7); cute_fp8.py registers it as the custom op sol_h3::fp8_rowwise_mm.
"""
