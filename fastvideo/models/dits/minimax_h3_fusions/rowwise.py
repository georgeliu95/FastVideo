"""H3 producers that emit rowwise FP8 directly (h3-rowwise).

With per-token activation scales each row's scale depends only on that row, so the kernels that produce the inputs
of the FP8 projections -- RMSNorm + modulate (q/k/v), residual + gate + RMSNorm + modulate (fc_in), SwiGLU (fc_out)
-- can quantize the row they just computed: no separate quantization pass, and no BF16 activation written for a
consumer that only reads FP8. The BF16 arithmetic is that of the h3-scale producers (minimax_h3_fusions
modulation.py / swiglu.py: same loads, same BLOCK and num_warps, so the same RMSNorm reduction); the quantization is
fastvideo fp8_config._quantize_rowwise applied to the BF16-rounded row, operation for operation:

    scale = max(amax(|x_bf16|) * fp32(1 / 448), 1 / (448 * 512))     (FP32, returned for the GEMM)
    q     = fp8(clamp(bf16(x_bf16 / bf16(scale)), -448, 448))

as PyTorch computes it (tensor / scalar as a multiplication by the reciprocal; tensor / tensor with IEEE division,
div_rn). Each op returns the FP8 rows and the [M, 1] FP32 scales; the q/k/v producer also returns its BF16 output,
which the BF16 gate-compression projection reads. The quantization in the same kernel lets Triton lay the row out
differently, so the RMSNorm sum can round differently in the last bit for a few elements (tests/
test_h3_rowwise_fusions.py reports how many); the definition above is otherwise followed bit for bit."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
FP8_MIN_SCALE = 1.0 / (448.0 * 512.0)


@triton.jit
def _quantize_row(v, mask, fp8_ptr, offsets, scale_ptr, row):
    """fp8_config._quantize_rowwise for one row; ``v`` holds the FP32 values of the BF16-rounded row."""
    amax = tl.max(tl.where(mask, tl.abs(v), 0.0), axis=0)
    # PyTorch divides a CUDA tensor by a Python scalar as a multiplication by the FP32 reciprocal.
    scale = tl.maximum(amax * (1.0 / 448.0), 1.0 / (448.0 * 512.0))
    scale_bf16 = scale.to(tl.bfloat16).to(tl.float32)
    q = tl.math.div_rn(v, scale_bf16).to(tl.bfloat16).to(tl.float32)
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    tl.store(fp8_ptr + offsets, q.to(tl.float8e4nv), mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit
def _rmsnorm_modulate_rowwise_kernel(out_ptr, fp8_ptr, row_scale_ptr, x_ptr, weight_ptr, scale_ptr, shift_ptr,
                                     index_ptr, n_cols, n_index, eps, stride_x_row, stride_scale_row,
                                     stride_shift_row, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x_offsets = row * stride_x_row + cols
    table_row = tl.load(index_ptr + row % n_index).to(tl.int64)
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normed = x * tl.math.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + table_row * stride_scale_row + cols, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + table_row * stride_shift_row + cols, mask=mask, other=0.0).to(tl.float32)
    output = (normed * weight * (1.0 + scale) + shift).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + x_offsets, output, mask=mask)
    _quantize_row(output.to(tl.float32), mask, fp8_ptr, row * n_cols + cols, row_scale_ptr, row)


@triton.jit
def _residual_gate_rmsnorm_modulate_rowwise_kernel(hidden_out_ptr, fp8_ptr, row_scale_ptr, residual_ptr, branch_ptr,
                                                   gate_ptr, weight_ptr, scale_ptr, shift_ptr, index_ptr, n_cols,
                                                   n_index, eps, stride_input_row, stride_gate_row, stride_scale_row,
                                                   stride_shift_row, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    input_offsets = row * stride_input_row + cols
    table_row = tl.load(index_ptr + row % n_index).to(tl.int64)
    residual = tl.load(residual_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    branch = tl.load(branch_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + table_row * stride_gate_row + cols, mask=mask, other=0.0).to(tl.float32)
    hidden = residual + gate * branch
    tl.store(hidden_out_ptr + input_offsets, hidden.to(hidden_out_ptr.dtype.element_ty), mask=mask)
    variance = tl.sum(hidden * hidden, axis=0) / n_cols
    normed = hidden * tl.math.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + table_row * stride_scale_row + cols, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + table_row * stride_shift_row + cols, mask=mask, other=0.0).to(tl.float32)
    output = (normed * weight * (1.0 + scale) + shift).to(tl.bfloat16)
    _quantize_row(output.to(tl.float32), mask, fp8_ptr, row * n_cols + cols, row_scale_ptr, row)


@triton.jit
def _swiglu_rowwise_kernel(fp8_ptr, row_scale_ptr, x_ptr, ffn_dim, stride_in_row, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < ffn_dim
    value = tl.load(x_ptr + row * stride_in_row + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(x_ptr + row * stride_in_row + ffn_dim + cols, mask=mask, other=0.0).to(tl.float32)
    out = (value * (gate * tl.sigmoid(gate))).to(tl.bfloat16)
    _quantize_row(out.to(tl.float32), mask, fp8_ptr, row * ffn_dim + cols, row_scale_ptr, row)


def _helpers():
    from fastvideo.models.dits.minimax_h3_fusions import modulation
    return modulation


def _rmsnorm_modulate_rowwise_impl(x, weight, scale, shift, index, eps):
    m = _helpers()
    m._validate_contract(x, weight, (scale, shift), index, eps)
    hidden_size = x.shape[-1]
    flat_x = x.reshape(-1, hidden_size).contiguous()
    weight, scale, shift = weight.contiguous(), m._row_addressable(scale), m._row_addressable(shift)
    index = index.contiguous()
    rows = flat_x.shape[0]
    out = torch.empty_like(flat_x)
    fp8 = torch.empty((rows, hidden_size), device=x.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    block = m._next_power_of_two(hidden_size)
    _rmsnorm_modulate_rowwise_kernel[(rows,)](out, fp8, row_scale, flat_x, weight, scale, shift, index, hidden_size,
                                              index.numel(), eps, flat_x.stride(0), scale.stride(0), shift.stride(0),
                                              BLOCK=block, num_warps=m._num_warps(block))
    return out.view_as(x), fp8, row_scale


def _residual_gate_rmsnorm_modulate_rowwise_impl(residual, branch, gate, weight, scale, shift, index, eps):
    m = _helpers()
    m._validate_contract(residual, weight, (gate, scale, shift), index, eps)
    hidden_size = residual.shape[-1]
    flat_residual = residual.reshape(-1, hidden_size).contiguous()
    flat_branch = branch.reshape(-1, hidden_size).contiguous()
    weight, gate = weight.contiguous(), m._row_addressable(gate)
    scale, shift, index = m._row_addressable(scale), m._row_addressable(shift), index.contiguous()
    rows = flat_residual.shape[0]
    hidden = torch.empty_like(flat_residual)
    fp8 = torch.empty((rows, hidden_size), device=residual.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((rows, 1), device=residual.device, dtype=torch.float32)
    block = m._next_power_of_two(hidden_size)
    _residual_gate_rmsnorm_modulate_rowwise_kernel[(rows,)](
        hidden, fp8, row_scale, flat_residual, flat_branch, gate, weight, scale, shift, index, hidden_size,
        index.numel(), eps, flat_residual.stride(0), gate.stride(0), scale.stride(0), shift.stride(0),
        BLOCK=block, num_warps=m._num_warps(block))
    return hidden.view_as(residual), fp8, row_scale


def _swiglu_rowwise_impl(x):
    packed = x.shape[-1]
    ffn_dim = packed // 2
    flat = x.reshape(-1, packed).contiguous()
    rows = flat.shape[0]
    fp8 = torch.empty((rows, ffn_dim), device=x.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(ffn_dim)
    _swiglu_rowwise_kernel[(rows,)](fp8, row_scale, flat, ffn_dim, flat.stride(0), BLOCK_SIZE=block,
                                    num_warps=_helpers()._num_warps(block))
    return fp8, row_scale


@torch.library.custom_op("sol_h3::rmsnorm_modulate_rowwise", mutates_args=(), device_types="cuda")
def rmsnorm_modulate_rowwise(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                             index: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _rmsnorm_modulate_rowwise_impl(x, weight, scale, shift, index, eps)


@rmsnorm_modulate_rowwise.register_fake
def _(x, weight, scale, shift, index, eps):
    rows = x.numel() // x.shape[-1]
    return (x.new_empty(x.shape), x.new_empty((rows, x.shape[-1]), dtype=torch.float8_e4m3fn),
            x.new_empty((rows, 1), dtype=torch.float32))


@torch.library.custom_op("sol_h3::residual_gate_rmsnorm_modulate_rowwise", mutates_args=(), device_types="cuda")
def residual_gate_rmsnorm_modulate_rowwise(residual: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor,
                                           weight: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                                           index: torch.Tensor, eps: float
                                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _residual_gate_rmsnorm_modulate_rowwise_impl(residual, branch, gate, weight, scale, shift, index, eps)


@residual_gate_rmsnorm_modulate_rowwise.register_fake
def _(residual, branch, gate, weight, scale, shift, index, eps):
    rows = residual.numel() // residual.shape[-1]
    return (residual.new_empty(residual.shape),
            residual.new_empty((rows, residual.shape[-1]), dtype=torch.float8_e4m3fn),
            residual.new_empty((rows, 1), dtype=torch.float32))


@torch.library.custom_op("sol_h3::swiglu_rowwise", mutates_args=(), device_types="cuda")
def swiglu_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _swiglu_rowwise_impl(x)


@swiglu_rowwise.register_fake
def _(x):
    rows = x.numel() // x.shape[-1]
    return (x.new_empty((rows, x.shape[-1] // 2), dtype=torch.float8_e4m3fn), x.new_empty((rows, 1), dtype=torch.float32))
