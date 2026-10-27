"""Row/column-scaled FP8 GEMM on GB10 (SM121) for the MiniMax-H3 DiT projections, as a torch custom op.

The kernel is sglang's (opt/h3-dgx-spark 5b1fa7a9e7, python/sglang/kernels/ops/gemm/gb10_cute_fp8_kernel.py, copied
verbatim as gb10_cute_fp8_kernel.py): a persistent 256x192x64-tile CuTe DSL kernel, FP32 accumulation over K, then
acc * scale_b * scale_a and one BF16 rounding. This wrapper is sglang's gb10_cute_fp8.py without its environment
switch, with one more variant for the separate q/k/v projections (N 7168), registered as `sol_h3::fp8_rowwise_mm` so
a compiled H3 block calls it without a graph break. A kernel is compiled per (M, N, K, device, variant) on first use
(tens of seconds; the deployment's warmup request pays it). Shapes without a variant, or M below 1024, fall back to
torch._scaled_mm with the same row/column scales."""

from __future__ import annotations

from typing import Final, NamedTuple

import torch


class _Variant(NamedTuple):
    """Output staging buffers, SMEM-staged scales, and the 8-tile band direction."""

    output_buffers: int
    stage_scales: bool
    along_m: bool


# (N, K) of the MiniMax-H3 DiT projections. sglang picked these on GB10 from 16 measured shapes; (7168, 5376) is
# the q, k or v projection alone, given the variant of the fused (21504, 5376) qkv projection.
_VARIANTS: Final = {
    (21504, 5376): _Variant(3, True, False),  # qkv
    (7168, 5376): _Variant(3, True, False),  # q, k or v
    (5376, 7168): _Variant(3, True, False),  # attention out
    (28672, 5376): _Variant(3, True, True),  # fc_in (gate+up)
    (5376, 14336): _Variant(3, True, False),  # fc_out
}
_MIN_M: Final = 1024
_COMPILED: Final[dict[tuple, object]] = {}


def _compile(a, b, c, raw, sa, sb, variant: _Variant, sm_count: int, stream) -> object:
    import cutlass
    import cutlass.cute as cute

    from .gb10_cute_fp8_kernel import Sm120RowwiseGemmKernel

    kernel = Sm120RowwiseGemmKernel(
        cutlass.Float32, (256, 192, 64), variant.output_buffers,
        False, False, False, True, 8, 8, True, True, 1, 4, True, True,
        mma_register_requirement=240, load_register_requirement=24,
        dedup_scale_loads=True, epilogue_tile_n=32, defer_epilogue_drain=False,
        output_warp=True, physical_a_copy=False, b_cache_evict_last=False,
        reuse_column_scales=False, specialize_output_state=True,
        swizzle_along_m=variant.along_m, prefetch_scales=False,
        stage_scales=variant.stage_scales,
    )  # fmt: skip
    return cute.compile(kernel, a, b, c, raw, sa, sb, sm_count, stream)


def eligible(a: torch.Tensor, b: torch.Tensor) -> bool:
    m, k = a.shape
    return (b.shape[1], k) in _VARIANTS and m >= _MIN_M and a.is_contiguous() and b.stride() == (1, k)


def cute_mm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """BF16(scale_a * (FP32(a @ b) * scale_b)); a [M, K] row-major FP8, b [K, N] column-major FP8, FP32 scales."""
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    m, k = a.shape
    n = b.shape[1]
    variant = _VARIANTS[(n, k)]
    c = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    cute_a = from_dlpack(a.unsqueeze(-1), assumed_align=16)
    cute_b = from_dlpack(b.T.unsqueeze(-1), assumed_align=16)
    cute_c = from_dlpack(c.unsqueeze(-1), assumed_align=16)
    cute_sa = from_dlpack(sa.reshape(m), assumed_align=4)
    cute_sb = from_dlpack(sb.reshape(n), assumed_align=4)
    # The kernel ABI carries an unused raw-accumulator operand (write_raw=False).
    raw = from_dlpack(torch.empty((1, 1), device=a.device), assumed_align=4)
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    key = (m, n, k, a.device.index or 0, variant)
    compiled = _COMPILED.get(key)
    if compiled is None:
        sm_count = torch.cuda.get_device_properties(a.device).multi_processor_count
        compiled = _COMPILED[key] = _compile(cute_a, cute_b, cute_c, raw, cute_sa, cute_sb, variant, sm_count, stream)
    compiled(cute_a, cute_b, cute_c, raw, cute_sa, cute_sb, stream)
    return c


@torch.library.custom_op("sol_h3::fp8_rowwise_mm", mutates_args=())
def fp8_rowwise_mm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """a [M, K] FP8 with sa [M, 1]; b [K, N] FP8 (column-major) with sb [1, N]; BF16 [M, N]."""
    if eligible(a, b):
        return cute_mm(a, b, sa.contiguous(), sb.contiguous())
    return torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


@fp8_rowwise_mm.register_fake
def _(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16)
