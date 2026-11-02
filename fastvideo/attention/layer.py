# SPDX-License-Identifier: Apache-2.0

import os
import weakref
from functools import wraps
from typing import Optional

import torch
import torch.nn as nn

from fastvideo.attention.selector import backend_name_to_enum, get_attn_backend
from fastvideo.distributed.communication_op import (sequence_model_parallel_all_gather,
                                                    sequence_model_parallel_all_to_all_4D)
from fastvideo.distributed.parallel_state import (get_sp_group, get_sp_parallel_rank, get_sp_world_size)
from fastvideo.forward_context import ForwardContext, get_forward_context
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import get_compute_dtype
from fastvideo.layers.rotary_embedding import _apply_rotary_emb


def _attention_compile_disabled() -> bool:
    """Whether to keep attention ``forward`` out of the torch.compile graph.

    Defaults to ``True`` (the historical behavior: attention runs eager via
    ``torch.compiler.disable``). Set ``FASTVIDEO_DISABLE_ATTENTION_COMPILE=0``
    to let attention instances constructed under that environment be traced.
    """
    val = os.environ.get("FASTVIDEO_DISABLE_ATTENTION_COMPILE")
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _attention_compile_explicitly_disabled() -> bool:
    """Whether the environment explicitly requests the eager boundary.

    Regional compile can override the historical default for one loaded
    transformer, but it must still honor an explicit debugging escape hatch.
    """
    return "FASTVIDEO_DISABLE_ATTENTION_COMPILE" in os.environ and _attention_compile_disabled()


def _maybe_compiler_disable(fn):
    """Defer the eager/traceable choice to each attention instance.

    A class-definition-time choice makes a process-wide default the only
    option. The deferred wrapper keeps ordinary instances on the historical
    eager boundary while allowing the regional loader to opt in only the
    attention modules owned by the transformer it is compiling.
    """
    disabled_fn = torch.compiler.disable(fn)

    @wraps(fn)
    def _dispatch(self, *args, **kwargs):
        if self._compile_forward_enabled:
            return fn(self, *args, **kwargs)
        return disabled_fn(self, *args, **kwargs)

    return _dispatch


class DistributedAttention(nn.Module):
    """Distributed attention layer.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 num_kv_heads: int | None = None,
                 softmax_scale: float | None = None,
                 causal: bool = False,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 prefix: str = "",
                 **extra_impl_args) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale

        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(head_size, dtype, supported_attention_backends=supported_attention_backends)
        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(num_heads=num_heads,
                                  head_size=head_size,
                                  causal=causal,
                                  softmax_scale=self.softmax_scale,
                                  num_kv_heads=num_kv_heads,
                                  prefix=f"{prefix}.impl",
                                  **extra_impl_args)
        # Register attn_impl as submodule if it has learnable parameters (e.g., SLA's proj_l)
        # This ensures its parameters are included in state_dict() for saving/loading
        if isinstance(self.attn_impl, nn.Module):
            self.add_module('attn_impl', self.attn_impl)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype
        # Preserve the historical compiler-disabled default. The regional
        # inference loader may enable this one instance after validating the
        # transformer's resolved backend; no process-global default changes.
        self._compile_forward_enabled = not _attention_compile_disabled()

    def _set_compile_forward_enabled(self, enabled: bool) -> None:
        self._compile_forward_enabled = enabled

    @_maybe_compiler_disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original_seq_len: int | None = None,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.
        
        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            original_seq_len (int): Original (unpadded) full sequence length
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"
        batch_size, _, num_heads, _ = q.shape
        local_rank = get_sp_parallel_rank()
        world_size = get_sp_world_size()

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        # Stack QKV
        qkv = torch.cat([q, k, v], dim=0)  # [3*batch, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)

        # After all-to-all, each rank has the full sequence but only a subset of heads.
        # Trim away SP padding for attention compute, then pad back before returning.
        original_seq_len = original_seq_len or qkv.shape[1]
        pad_seq_len = qkv.shape[1] - original_seq_len
        qkv = qkv[:, :original_seq_len, :, :]

        if freqs_cis is not None:
            cos, sin = freqs_cis
            qkv[:batch_size * 2] = _apply_rotary_emb(qkv[:batch_size * 2], cos, sin, is_neox_style=False)
        # Apply backend-specific preprocess_qkv
        qkv = self.attn_impl.preprocess_qkv(qkv, ctx_attn_metadata)

        # Concatenate with replicated QKV if provided
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            replicated_qkv = torch.cat([replicated_q, replicated_k, replicated_v],
                                       dim=0)  # [3, seq_len, num_heads, head_dim]
            heads_per_rank = num_heads // world_size
            replicated_qkv = replicated_qkv[:, :, local_rank * heads_per_rank:(local_rank + 1) * heads_per_rank]
            qkv = torch.cat([qkv, replicated_qkv], dim=1)

        q, k, v = qkv.chunk(3, dim=0)

        output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        # Redistribute back if using sequence parallelism
        replicated_output = None
        if replicated_q is not None:
            split_idx = original_seq_len
            replicated_output = output[:, split_idx:]
            output = output[:, :split_idx]
            # TODO: make this asynchronous
            replicated_output = sequence_model_parallel_all_gather(replicated_output.contiguous(), dim=2)
        # Apply backend-specific postprocess_output
        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        output = torch.nn.functional.pad(output, (0, 0, 0, 0, 0, pad_seq_len))

        output = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)

        return output, replicated_output


# Sol-H3-Spark v4.4: VSA attention with its Ulysses exchanges pipelined over head groups by the ulysses_head_pipeline
# package (gitlab-master georgel/ulysses-head-pipeline; imported only when enabled), SPARK_CP_H3_PIPELINE_GROUPS = G > 1
# groups per rank. SPARK_CP_H3_PIPELINE_CHECK = N compares the first N calls of every layer with the unpipelined
# exchange, bit for bit: only meaningful when the attention's bits per head do not depend on the group size (VSA-H3
# on GB10 needs Sol-H3-Spark's exact mode, SPARK_CP_H3_PIPELINE_EXACT=1).
_H3_PIPELINE_GROUPS = int(os.environ.get("SPARK_CP_H3_PIPELINE_GROUPS", "1"))
_H3_PIPELINE_LAYERS: list[weakref.ref] = []


@torch.library.custom_op("fastvideo_sp::vsa_head_group_pipeline", mutates_args=(), device_types="cuda")
def _vsa_head_group_pipeline(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: Optional[torch.Tensor],  # noqa: UP045 -- the custom-op schema spelling
        layer: torch.Tensor,
        original_seq_len: int) -> torch.Tensor:
    """Opaque to the regional fullgraph compile: the pipeline orders streams and calls the attention per group.
    ``layer`` is a CPU int64 scalar so the 50 block instances share one graph."""
    module = _H3_PIPELINE_LAYERS[int(layer)]()
    return module._pipelined_forward(q, k, v, gate, original_seq_len)


@_vsa_head_group_pipeline.register_fake
def _vsa_head_group_pipeline_fake(q, k, v, gate, layer, original_seq_len):
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


class DistributedAttention_VSA(DistributedAttention):
    """Distributed attention layer with VSA support.
    """

    # Optional transport for the head-group pipeline's exchanges instead of NCCL (2 ranks), set by the caller before
    # the first forward: transport_factory(group, nbytes, slots) -> transport (ulysses_head_pipeline.HeadGroupPipeline),
    # e.g. ulysses_head_pipeline.spark_rdma.factory(...) (RDMA from host-pinned buffers on DGX Spark).
    pipeline_transport_factory = None
    # Process-wide: pipelined forwards, inner attention calls they made (G each, plus one per check) and checks.
    pipeline_stats = {"calls": 0, "attention_calls": 0, "checks": 0, "groups": _H3_PIPELINE_GROUPS, "transport": None}
    _pipeline = None  # ulysses_head_pipeline.HeadGroupPipeline, made at the first pipelined call

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pipeline_index = torch.tensor(len(_H3_PIPELINE_LAYERS), dtype=torch.int64, device="cpu")
        _H3_PIPELINE_LAYERS.append(weakref.ref(self))
        self._pipeline_checks = int(os.environ.get("SPARK_CP_H3_PIPELINE_CHECK", "0"))

    def _pipelined_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate_compress: torch.Tensor | None,
                           original_seq_len: int) -> torch.Tensor:
        cls = DistributedAttention_VSA
        if cls._pipeline is None:
            from ulysses_head_pipeline import HeadGroupPipeline
            cls._pipeline = HeadGroupPipeline(get_sp_group().device_group,
                                              _H3_PIPELINE_GROUPS,
                                              cls.pipeline_transport_factory,
                                              name="h3 pipeline")
        ctx_attn_metadata = get_forward_context().attn_metadata
        impl = self.attn_impl
        xs = [q, k, v] if gate_compress is None else [q, k, v, gate_compress]

        def attention(x: torch.Tensor, group: int) -> torch.Tensor:
            parts = impl.preprocess_qkv(x, ctx_attn_metadata).chunk(len(xs), dim=0)
            gate = parts[3] if len(parts) == 4 else None
            output = impl.forward(parts[0], parts[1], parts[2], gate, ctx_attn_metadata)  # type: ignore[call-arg]
            return impl.postprocess_output(output, ctx_attn_metadata)

        result = cls._pipeline.run(xs, attention, original_seq_len)  # fresh [1, t, heads, d]
        stats = cls.pipeline_stats
        stats["calls"] += 1
        stats["attention_calls"] += _H3_PIPELINE_GROUPS
        stats["transport"] = cls._pipeline.describe()
        if self._pipeline_checks > 0:
            self._pipeline_checks -= 1
            reference = self._exchange_attention(q, k, v, gate_compress, original_seq_len, None, ctx_attn_metadata)
            stats["checks"] += 1
            stats["attention_calls"] += 1
            same = bool(torch.equal(result, reference))
            print(
                f"[h3 pipeline] rank {get_sp_parallel_rank()} groups {_H3_PIPELINE_GROUPS} heads {q.shape[2]} "
                f"tokens {q.shape[1]} ({stats['transport']}): equal to the unpipelined exchange {same}",
                flush=True)
            if not same:
                raise RuntimeError("head-group pipelined VSA attention differs from the unpipelined exchange")
        return result

    @_maybe_compiler_disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original_seq_len: int,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        gate_compress: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.
        
        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            original_seq_len (int): Original (unpadded) full sequence length
            gate_compress (torch.Tensor): Gate compress tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check text tokens are not supported for VSA now
        assert replicated_q is None and replicated_k is None and replicated_v is None, "Replicated QKV is not supported for VSA now"
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        batch_size, seq_len, num_heads, head_dim = q.shape
        if (_H3_PIPELINE_GROUPS > 1 and get_sp_world_size() > 1 and batch_size == 1 and freqs_cis is None
                and num_heads % (get_sp_world_size() * _H3_PIPELINE_GROUPS) == 0 and k.shape == q.shape
                and v.shape == q.shape and (gate_compress is None or gate_compress.shape == q.shape)):
            output = _vsa_head_group_pipeline(q, k, v, gate_compress, self._pipeline_index, original_seq_len)
            return output, None
        return self._exchange_attention(q, k, v, gate_compress, original_seq_len, freqs_cis, ctx_attn_metadata), None

    def _exchange_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate_compress: torch.Tensor | None,
                            original_seq_len: int, freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
                            ctx_attn_metadata) -> torch.Tensor:
        """The unpipelined exchange: one all-to-all of the stacked inputs, attention over all local heads, one
        all-to-all of the output."""
        batch_size = q.shape[0]
        # Stack QKV (a caller with a structurally-zero gate passes None and
        # skips the gate's share of the all-to-all/tile traffic)
        stack = [q, k, v] if gate_compress is None else [q, k, v, gate_compress]
        qkvg = torch.cat(stack, dim=0)  # [3or4*batch, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        # Before: [3or4*batch, shard_seq_len, num_heads, head_dim]
        # After:  [4*batch, full_seq_len, shard_num_heads, head_dim]
        qkvg = sequence_model_parallel_all_to_all_4D(qkvg, scatter_dim=2, gather_dim=1)

        # After all-to-all, each rank has the full sequence but only a subset of heads
        pad_seq_len = qkvg.shape[1] - original_seq_len
        qkvg = qkvg[:, :original_seq_len, :, :]

        if freqs_cis is not None:
            cos, sin = freqs_cis
            qkvg[:batch_size * 2] = _apply_rotary_emb(qkvg[:batch_size * 2], cos, sin, is_neox_style=False)

        qkvg = self.attn_impl.preprocess_qkv(qkvg, ctx_attn_metadata)

        if gate_compress is None:
            q, k, v = qkvg.chunk(3, dim=0)
        else:
            q, k, v, gate_compress = qkvg.chunk(4, dim=0)
        output = self.attn_impl.forward(q, k, v, gate_compress, ctx_attn_metadata)  # type: ignore[call-arg]

        # Apply backend-specific postprocess_output
        if os.environ.get("SPARK_CP_H3_PACK", "0") == "1" and get_sp_world_size() > 1:
            from runtime.stage1_ops.cp_transport import reverse_tiled_exchange
            output = reverse_tiled_exchange(output, ctx_attn_metadata.untile_combined_index, get_sp_world_size())
        else:
            output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)
            output = torch.nn.functional.pad(output, (0, 0, 0, 0, 0, pad_seq_len))
            output = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)
        return output


class LocalAttention(nn.Module):
    """Attention layer.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 num_kv_heads: int | None = None,
                 softmax_scale: float | None = None,
                 causal: bool = False,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 default_backend: AttentionBackendEnum | None = None,
                 **extra_impl_args) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale
        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(head_size,
                                        dtype,
                                        supported_attention_backends=supported_attention_backends,
                                        default_backend=default_backend)
        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(num_heads=num_heads,
                                  head_size=head_size,
                                  softmax_scale=self.softmax_scale,
                                  num_kv_heads=num_kv_heads,
                                  causal=causal,
                                  **extra_impl_args)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim] 
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            
        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        if freqs_cis is not None:
            cos, sin = freqs_cis
            q = _apply_rotary_emb(q, cos, sin, is_neox_style=False)
            k = _apply_rotary_emb(k, cos, sin, is_neox_style=False)

        output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
        return output
