# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3's value-first packed SwiGLU fusion."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only in environments without Triton
    triton = None
    tl = None
    HAVE_TRITON = False


def _validate_input(x: torch.Tensor) -> int:
    if x.ndim == 0:
        raise ValueError("MiniMax H3 SwiGLU expects at least one dimension")

    packed_width = x.shape[-1]
    if packed_width == 0 or packed_width % 2 != 0:
        raise ValueError(
            "MiniMax H3 SwiGLU expects a positive even last dimension containing packed (value, gate) halves, "
            f"got {packed_width}"
        )
    if not x.is_floating_point():
        raise TypeError(f"MiniMax H3 SwiGLU expects a floating-point tensor, got {x.dtype}")
    return packed_width // 2


if HAVE_TRITON:

    @triton.jit
    def _minimax_h3_swiglu_kernel(
        out_ptr,
        x_ptr,
        ffn_dim,
        stride_in_row,
        stride_out_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < ffn_dim

        value = tl.load(x_ptr + row * stride_in_row + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * stride_in_row + ffn_dim + cols, mask=mask, other=0.0).to(tl.float32)
        # Match Sol-Engine: keep the complete SwiGLU expression in FP32 and
        # convert only the final output store.
        out = value * (gate * tl.sigmoid(gate))
        tl.store(out_ptr + row * stride_out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

else:
    _minimax_h3_swiglu_kernel = None


def _num_warps(block_size: int) -> int:
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def _minimax_h3_swiglu_impl(x: torch.Tensor) -> torch.Tensor:
    ffn_dim = _validate_input(x)
    if not x.is_cuda:
        raise ValueError("MiniMax H3 fused SwiGLU requires a CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"MiniMax H3 fused SwiGLU supports float16, bfloat16, and float32, got {x.dtype}")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("MiniMax H3 fused SwiGLU is forward-only and does not implement autograd")
    if _minimax_h3_swiglu_kernel is None:
        raise RuntimeError("MiniMax H3 fused SwiGLU requires Triton")

    packed_width = x.shape[-1]
    flat = x.reshape(-1, packed_width).contiguous()
    output_shape = (*x.shape[:-1], ffn_dim)
    if flat.shape[0] == 0:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)

    out = torch.empty((flat.shape[0], ffn_dim), dtype=x.dtype, device=x.device)
    block_size = triton.next_power_of_2(ffn_dim)
    _minimax_h3_swiglu_kernel[(flat.shape[0],)](
        out,
        flat,
        ffn_dim,
        flat.stride(0),
        out.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return out.view(output_shape)


@torch.library.custom_op(
    "fastvideo::_minimax_h3_swiglu",
    mutates_args=(),
    device_types="cuda",
)
def _minimax_h3_swiglu_op(x: torch.Tensor) -> torch.Tensor:
    return _minimax_h3_swiglu_impl(x)


@torch.library.register_fake("fastvideo::_minimax_h3_swiglu")
def _minimax_h3_swiglu_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def minimax_h3_swiglu(x: torch.Tensor) -> torch.Tensor:
    """Run the forward-only Triton fusion over an H3 ``(..., 2 * ffn_dim)`` input.

    This is intentionally a strict kernel wrapper: callers own fallback policy and
    must only invoke it for a supported CUDA inference path.
    """
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("MiniMax H3 fused SwiGLU is forward-only and does not implement autograd")
    if torch.compiler.is_compiling():
        return torch.ops.fastvideo._minimax_h3_swiglu(x)
    return _minimax_h3_swiglu_impl(x)


# Same kernel plus ``output.abs().amax()`` over the stored values, for the dynamic
# tensorwise FP8 quantization of fc_out (saves that separate read of the output).
if HAVE_TRITON:
    @triton.jit
    def _minimax_h3_swiglu_amax_kernel(
        out_ptr,
        amax_ptr,
        x_ptr,
        ffn_dim,
        stride_in_row,
        stride_out_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < ffn_dim
        value = tl.load(x_ptr + row * stride_in_row + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * stride_in_row + ffn_dim + cols, mask=mask, other=0.0).to(tl.float32)
        out = (value * (gate * tl.sigmoid(gate))).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_out_row + cols, out, mask=mask)
        tl.atomic_max(amax_ptr, tl.max(tl.where(mask, tl.abs(out.to(tl.float32)), 0.0), axis=0))
else:
    _minimax_h3_swiglu_amax_kernel = None


_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_FP8_MIN_SCALE = 1.0 / (_FP8_MAX * 512.0)


def _tensorwise_scale(amax: torch.Tensor) -> torch.Tensor:
    """Tensorwise FP8 scale of the stored output, computed right after the producer (see modulation.py)."""
    return (amax / _FP8_MAX).clamp(min=_FP8_MIN_SCALE)


def _minimax_h3_swiglu_amax_impl(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ffn_dim = _validate_input(x)
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("MiniMax H3 fused SwiGLU+amax requires a CUDA float16/bfloat16/float32 tensor")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("MiniMax H3 fused SwiGLU is forward-only and does not implement autograd")
    if _minimax_h3_swiglu_amax_kernel is None:
        raise RuntimeError("MiniMax H3 fused SwiGLU requires Triton")
    packed_width = x.shape[-1]
    flat = x.reshape(-1, packed_width).contiguous()
    output_shape = (*x.shape[:-1], ffn_dim)
    amax = torch.zeros((), device=x.device, dtype=torch.float32)
    if flat.shape[0] == 0:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device), _tensorwise_scale(amax)
    out = torch.empty((flat.shape[0], ffn_dim), dtype=x.dtype, device=x.device)
    block_size = triton.next_power_of_2(ffn_dim)
    _minimax_h3_swiglu_amax_kernel[(flat.shape[0],)](
        out,
        amax,
        flat,
        ffn_dim,
        flat.stride(0),
        out.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return out.view(output_shape), _tensorwise_scale(amax)


@torch.library.custom_op(
    "fastvideo::_minimax_h3_swiglu_amax",
    mutates_args=(),
    device_types="cuda",
)
def _minimax_h3_swiglu_amax_op(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _minimax_h3_swiglu_amax_impl(x)


@torch.library.register_fake("fastvideo::_minimax_h3_swiglu_amax")
def _minimax_h3_swiglu_amax_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2)), x.new_empty((), dtype=torch.float32)


def minimax_h3_swiglu_amax(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`minimax_h3_swiglu` plus the output's ``abs().amax()`` as an FP32 scalar."""
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("MiniMax H3 fused SwiGLU is forward-only and does not implement autograd")
    if torch.compiler.is_compiling():
        return torch.ops.fastvideo._minimax_h3_swiglu_amax(x)
    return _minimax_h3_swiglu_amax_impl(x)


__all__ = ["HAVE_TRITON", "minimax_h3_swiglu", "minimax_h3_swiglu_amax"]
