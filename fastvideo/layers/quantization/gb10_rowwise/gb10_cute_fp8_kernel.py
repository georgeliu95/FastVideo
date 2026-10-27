# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# noqa: SIZE_OK - Vendored indivisible CuTe persistent kernel; changes documented in README.
"""Experimental SM120 ordinary FP8 GEMM with register-local row/column scaling."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir import ir
from cutlass.cutlass_dsl import dsl_user_op


class PackedAbConsumerState:
    """Three-stage consumer state: count plus one cursor containing index and phase."""

    def __init__(self, stages: int, count: cutlass.Int32, cursor: cutlass.Int32):
        assert stages == 3, "Packed AB consumer cursor supports exactly three stages"
        self._stages = stages
        self._count = count
        self._cursor = cursor

    @property
    @cute.jit
    def index(self) -> cutlass.Int32:
        return self._cursor & 3

    @property
    @cute.jit
    def phase(self) -> cutlass.Int32:
        return self._cursor >> 2

    @property
    @cute.jit
    def count(self) -> cutlass.Int32:
        return self._count

    @property
    def stages(self) -> int:
        return self._stages

    @dsl_user_op
    @cute.jit
    def reset_count(
        self,
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        # Persistent index/phase remain untouched at each new output tile.
        self._count = cutlass.Int32(0, loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def advance(
        self,
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._count += 1
        # Legal cursor cycle is 0,1,2,4,5,6; skip the two invalid low-bit values.
        self._cursor = (
            self._cursor + 1 + cutlass.Int32((self._cursor & 3) == 2)
        ) & 7

    def __get_mlir_types__(self) -> list[ir.Type]:
        return [self._count.type, self._cursor.type]

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self._count.ir_value(), self._cursor.ir_value()]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "PackedAbConsumerState":
        assert len(values) == 2
        return PackedAbConsumerState(
            self.stages, cutlass.Int32(values[0]), cutlass.Int32(values[1])
        )


class Sm120RowwiseGemmKernel:
    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
        epilogue_stages: int = 2,
        promote_each_tile: bool = False,
        write_raw: bool = False,
        native_mma: bool = False,
        sync_epilogue_reuse: bool = True,
        swizzle_size: int = 1,
        mma_warps: int = 4,
        mma_before_wait: bool = False,
        emit_setmaxregister: bool = True,
        min_blocks_per_mp: int = 0,
        producer_warps: int = 1,
        compact_tail: bool = False,
        mma_before_prefetch: bool = False,
        mma_register_requirement: int = 232,
        load_register_requirement: int = 40,
        dedup_scale_loads: bool = False,
        epilogue_tile_n: int = 32,
        defer_epilogue_drain: bool = False,
        output_warp: bool = False,
        physical_a_copy: bool = False,
        b_cache_evict_last: bool = False,
        reuse_column_scales: bool = False,
        specialize_output_state: bool = False,
        swizzle_along_m: bool = False,
        prefetch_scales: bool = False,
        stage_scales: bool = False,
        multicast: str | None = None,
        a_cache_evict_last: bool = False,
        b_cache_evict_first: bool = False,
        c_store_evict_first: bool = False,
    ):
        # Tile order only: bands of swizzle_size M tiles (sweep N inside a band) instead of bands of N tiles.
        self.swizzle_along_m = swizzle_along_m
        # Issue epilogue job j+1's scale loads before job j's math and barriers (same values, same arithmetic).
        self.prefetch_scales = prefetch_scales
        if prefetch_scales:
            assert dedup_scale_loads and not reuse_column_scales and epilogue_tile_n == 32, "Scale prefetch requires deduplicated epilogue64x32 scales"
        # An idle producer warp stages the next tile's row and column scales in SMEM; the epilogue reads them
        # with LDS instead of a dependent LDG per job (same values, same arithmetic).
        self.stage_scales = stage_scales
        # Two M-adjacent CTAs per cluster; the tile order decides which operand the pair shares: N bands give
        # both CTAs the same M tile (multicast A, half of 16 KB per stage), M bands the same N tile (B, 12 KB).
        self.multicast = multicast
        if multicast is not None:
            assert multicast in ("a", "b") and compact_tail and swizzle_size % 2 == 0, "Multicast pairs need compact bands"
            assert (multicast == "b") == swizzle_along_m, "A multicast needs N bands, B multicast needs M bands"
            assert not physical_a_copy, "Physical A view assumes a single-CTA A tile"
        # L2 EVICT_LAST on the A loads: with M bands the A band is the resident operand.
        self.a_cache_evict_last = a_cache_evict_last
        # L2 EVICT_FIRST on B loads (streamed under M bands) and on output stores (never reread).
        self.b_cache_evict_first = b_cache_evict_first
        self.c_store_evict_first = c_store_evict_first
        assert not (b_cache_evict_first and b_cache_evict_last), "Pick one B cache policy"
        if stage_scales:
            assert output_warp and producer_warps == 4 and dedup_scale_loads and not prefetch_scales, "Scale staging uses the third producer warp and deduplicated scales"
            assert tuple(tile_shape_mnk) == (256, 192, 64) and epilogue_tile_n == 32 and not reuse_column_scales, "Scale staging requires tile256x192x64 with epilogue64x32"
        self.specialize_output_state = specialize_output_state
        if specialize_output_state:
            assert output_warp and epilogue_stages in (2, 3), "Static output state requires the dedicated output warp with two or three buffers"
            assert tuple(tile_shape_mnk) == (256, 192, 64) and mma_warps == 8 and epilogue_tile_n == 32, "Static output state requires exactly 24 epilogue jobs per tile"
            assert not reuse_column_scales, "Column-scale reuse is outside this bounded state experiment"
        self.reuse_column_scales = reuse_column_scales
        if reuse_column_scales:
            assert tuple(tile_shape_mnk) == (256, 192, 64) and mma_warps == 8, "Column-scale reuse requires tile256x192x64 and atom4x2"
            assert epilogue_tile_n == 32 and dedup_scale_loads, "Column-scale reuse requires epilogue64x32 with scale dedup"
        self.physical_a_copy = physical_a_copy
        self.b_cache_evict_last = b_cache_evict_last
        if physical_a_copy:
            assert tuple(tile_shape_mnk) == (256, 192, 64) and mma_warps == 8, "Physical A view is restricted to tile256x192x64 and atom4x2"
        self.output_warp = output_warp
        if output_warp:
            assert mma_warps == 8 and producer_warps == 4, "Output warp requires eight MMA and four producer warps"
            assert epilogue_stages in (2, 3) and sync_epilogue_reuse, "Output warp requires two or three synchronized output buffers"
            assert not defer_epilogue_drain, "Dedicated output warp owns the final drain"
        self.defer_epilogue_drain = defer_epilogue_drain
        self.epilogue_tile_n = epilogue_tile_n
        self.dedup_scale_loads = dedup_scale_loads
        self.mma_before_prefetch = mma_before_prefetch
        self.compact_tail = compact_tail
        self.producer_warps = producer_warps
        self.min_blocks_per_mp = min_blocks_per_mp
        self.emit_setmaxregister = emit_setmaxregister
        self.mma_before_wait = mma_before_wait
        self.swizzle_size = swizzle_size
        self.native_mma = native_mma
        self.sync_epilogue_reuse = sync_epilogue_reuse
        self.promote_each_tile = promote_each_tile
        self.write_raw = write_raw
        self.epilogue_stages = epilogue_stages
        self.acc_dtype = acc_dtype
        self.cluster_shape_mnk = (2, 1, 1) if multicast is not None else (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.tiled_mma = None
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1
        # TODO: remove this hard code for user input ?
        self.atom_layout = (mma_warps // 2, 2, 1)
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + self.producer_warps
        ) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = load_register_requirement
        self.mma_register_requirement = mma_register_requirement

    @cute.jit
    def _band_m_tile_coord(self, tile_coord: cute.Coord, problem_shape: cute.Shape) -> cute.Coord:
        """Transpose of the N-band mapping: an A band of swizzle_size M tiles stays resident while N streams."""
        m_tiles = problem_shape[0]
        n_tiles = problem_shape[1]
        linear = tile_coord[0] + m_tiles * tile_coord[1]
        full_m = (m_tiles // self.swizzle_size) * self.swizzle_size
        group_size = n_tiles * self.swizzle_size
        group = linear // group_size
        local = linear % group_size
        mapped_n = local // self.swizzle_size
        mapped_m = group * self.swizzle_size + local % self.swizzle_size
        tail_m = m_tiles % self.swizzle_size
        if cutlass.const_expr(tail_m != 0):
            if linear >= n_tiles * full_m:
                tail_linear = linear - n_tiles * full_m
                mapped_n = tail_linear // tail_m
                mapped_m = full_m + tail_linear % tail_m
        return (cute.arch.make_warp_uniform(cutlass.Int32(mapped_m)), cute.arch.make_warp_uniform(cutlass.Int32(mapped_n)), tile_coord[2])

    @cute.jit
    def _compact_tile_coord(self, tile_coord: cute.Coord, problem_shape: cute.Shape) -> cute.Coord:
        if cutlass.const_expr(self.swizzle_along_m):
            return self._band_m_tile_coord(tile_coord, problem_shape)
        m_tiles = problem_shape[0]
        n_tiles = problem_shape[1]
        linear = tile_coord[0] + m_tiles * tile_coord[1]
        full_n = (n_tiles // self.swizzle_size) * self.swizzle_size
        group_size = m_tiles * self.swizzle_size
        group = linear // group_size
        local = linear % group_size
        mapped_m = local // self.swizzle_size
        mapped_n = group * self.swizzle_size + local % self.swizzle_size
        tail_n = n_tiles % self.swizzle_size
        if cutlass.const_expr(tail_n != 0):
            if linear >= m_tiles * full_n:
                tail_linear = linear - m_tiles * full_n
                mapped_m = tail_linear // tail_n
                mapped_n = full_n + tail_linear % tail_n
        return (cute.arch.make_warp_uniform(cutlass.Int32(mapped_m)), cute.arch.make_warp_uniform(cutlass.Int32(mapped_n)), tile_coord[2])

    @cute.jit
    def _physical_a_source(
        self,
        physical_base: cutlass.Int32,
        stage: cutlass.Int32,
        k_half: cutlass.Constexpr,
        half_layout: cute.Layout,
    ) -> cute.Tensor:
        # The K32 halves differ by physical address bit5; stage/M strides leave it unchanged.
        address = (physical_base + stage * 16384) ^ (k_half * 32)
        pointer = cute.make_ptr(
            self.a_dtype, address, cute.AddressSpace.smem, assumed_align=16
        )
        return cute.make_tensor(pointer, half_layout)

    def _setup_attributes(self):
        # TODO: remove this hard code for user input ?
        self.mma_inst_mnk = (16, 8, 32)
        if self.native_mma:
            op = cute.nvgpu.warp.MmaMXF8Op(
                self.a_dtype,
                self.acc_dtype,
                cutlass.Float8E8M0FNU,
            )
        else:
            op = cute.nvgpu.warp.MmaFP8Op(
                self.a_dtype,
                self.acc_dtype,
                self.mma_inst_mnk,
            )
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            # TODO: to leverage ldmatrix.x4, when self.atom_layout[1] is 1, mma tile is ((8x16)x2)
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.num_mcast_ctas_a = 2 if self.multicast == "a" else 1
        self.num_mcast_ctas_b = 2 if self.multicast == "b" else 1
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=False
        )
        assert self.epilogue_tile_n in (32, 64), "Only epilogue N32/N64 is supported"
        self.epi_tile = (self.epi_tile[0], self.epilogue_tile_n)
        if self.epilogue_tile_n == 64:
            assert self.tile_shape_mnk == (192, 192, 64), "N64 epilogue probe is square-only"
            assert self.atom_layout == (4, 2, 1), "N64 epilogue requires atom 4x2"

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        import sys

        if self.ab_stage == 0:
            print("ab_stage == 0, no enough shared memory. This case will be skipped.")
            sys.exit(0)

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        raw_output: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")
        if cutlass.const_expr(self.b_dtype.width != 16 and self.b_dtype.width != 8):
            raise TypeError("b_dtype should be float16 or float8")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.num_mcast_ctas_a,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.num_mcast_ctas_b,
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        if cutlass.const_expr(self.stage_scales):
            @cute.struct
            class SharedStorage:
                mainloop_pipeline_array_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.ab_stage * 2 + self.epi_stage * 2 + 2
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sScale: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.tile_shape_mnk[0] + self.tile_shape_mnk[1]], 16
                ]

        else:
            @cute.struct
            class SharedStorage:
                mainloop_pipeline_array_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.ab_stage * 2 + (self.epi_stage * 2 if self.output_warp else 0)
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            raw_output,
            scale_a,
            scale_b,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=list(self.cluster_shape_mnk),
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        raw_output: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        # bidx, bidy, bidz = cute.arch.block_idx()
        # bdimx, bdimy, bdimz = cute.arch.grid_dim()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        if cutlass.const_expr(self.multicast is not None):
            pair_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)
            a_mcast_mask = pair_mask if self.multicast == "a" else 0
            b_mcast_mask = pair_mask if self.multicast == "b" else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cutlass.const_expr(self.output_warp):
            # Separate ready/free barriers use the existing padding before aligned sA.
            output_pipeline = pipeline.PipelineAsync.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                barrier_storage=mainloop_pipeline_array_ptr + self.ab_stage * 2,
            )

        if cutlass.const_expr(self.stage_scales):
            # One staged tile: 32 loader lanes arrive full, MMA thread 0 releases after the last epilogue job.
            scale_pipeline = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads_per_warp),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                barrier_storage=mainloop_pipeline_array_ptr + self.ab_stage * 2 + self.epi_stage * 2,
            )
            sScale = storage.sScale.get_tensor(cute.make_layout(self.tile_shape_mnk[0] + self.tile_shape_mnk[1]))

        #  Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        if cutlass.const_expr(self.multicast == "a"):
            a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
            a_cta_crd = cluster_coord_mnk[0]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        if cutlass.const_expr(self.multicast == "a"):
            b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
            b_cta_crd = cluster_coord_mnk[1]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        #  Make frangments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        if cutlass.const_expr(self.promote_each_tile):
            accum_sum = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # cluster wait for barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Create the tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Create the pipeline states for producer and consumer
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        if cutlass.const_expr(self.physical_a_copy):
            mainloop_consumer_state = PackedAbConsumerState(
                self.ab_stage, cutlass.Int32(0), cutlass.Int32(0)
            )
        else:
            mainloop_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            if cutlass.const_expr(self.emit_setmaxregister):
                cute.arch.setmaxregister_increase(self.mma_register_requirement)

            # Native SM120 block-scaled MMA with exact multiplicative identities.
            # All four UE8M0 scale bytes are 127 == 1.0; FP8 operand bytes stay unchanged.
            if cutlass.const_expr(self.native_mma):
                unit_scales = cute.make_rmem_tensor((4,), cutlass.Float8E8M0FNU)
                unit_scale_bytes = cute.recast_tensor(unit_scales, cutlass.Uint8)
                unit_scale_bytes.fill(127)
                tiled_mma.set(cute.nvgpu.warp.Field.SFA, unit_scales.iterator)
                tiled_mma.set(cute.nvgpu.warp.Field.SFB, unit_scales.iterator)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # ///////////////////////////////////////////////////////////////////////////////
            # Copy Atom A/B retiling for TMA load A/B
            # ///////////////////////////////////////////////////////////////////////////////
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)

            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)

            if cutlass.const_expr(self.physical_a_copy):
                assert self.ab_stage == 3 and self.a_dtype.width == 8
                # Materialize one physical lane base from raw, unswizzled allocation storage.
                a_copy_half_layout = tCsA_copy_view[None, None, 0, 0].layout
                a_lane_offset = ((tidx << 6) & 0x3C0) | (tidx & 0x10) | ((tidx << 5) & 0xC00)
                a_raw_address = storage.sA.data_ptr().toint() + a_lane_offset
                a_physical_base = a_raw_address ^ ((a_raw_address >> 3) & 0x30)

            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            if cutlass.const_expr(self.output_warp):
                # This state must not reset at persistent-tile boundaries (nine square subtiles).
                output_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.epi_stage
                )

            if cutlass.const_expr(self.stage_scales):
                scale_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                if cutlass.const_expr(self.compact_tail):
                    tile_coord_mnl = self._compact_tile_coord(tile_coord_mnl, tile_sched_params.problem_shape_ntile_mnl)
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                # Clear the accumulator
                accumulators.fill(0.0)
                if cutlass.const_expr(self.promote_each_tile):
                    accum_sum.fill(0.0)

                # /////////////////////////////////////////////////////////////////////////////
                #  Pipelined MAINLOOP
                # /////////////////////////////////////////////////////////////////////////////

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                #  Wait for TMA copies to complete
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
                if cutlass.const_expr(not self.physical_a_copy):
                    tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                if cutlass.const_expr(self.physical_a_copy):
                    cute.copy(
                        smem_tiled_copy_A,
                        self._physical_a_source(
                            a_physical_base, mainloop_consumer_state.index,
                            0, a_copy_half_layout,
                        ),
                        tCrA_copy_view[None, None, 0],
                    )
                else:
                    cute.copy(
                        smem_tiled_copy_A,
                        tCsA_p[None, None, 0],
                        tCrA_copy_view[None, None, 0],
                    )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                for k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    # unroll the loop
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )
                            if cutlass.const_expr(self.mma_before_wait):
                                # The last block's operands are already in registers.
                                # Issue it before blocking on unrelated next-stage data.
                                cute.gemm(
                                    tiled_mma,
                                    accumulators,
                                    tCrA[None, None, k_block_idx],
                                    tCrB[None, None, k_block_idx],
                                    accumulators,
                                )

                            # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
                            if cutlass.const_expr(not self.physical_a_copy):
                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        if cutlass.const_expr(self.mma_before_prefetch):
                            if cutlass.const_expr(
                                not (self.mma_before_wait and k_block_idx == num_k_blocks - 1)
                            ):
                                cute.gemm(
                                    tiled_mma,
                                    accumulators,
                                    tCrA[None, None, k_block_idx],
                                    tCrB[None, None, k_block_idx],
                                    accumulators,
                                )
                        # Copy data from smem to tCrA/tCrB for the next k_block
                        if cutlass.const_expr(self.physical_a_copy):
                            cute.copy(
                                smem_tiled_copy_A,
                                self._physical_a_source(
                                    a_physical_base, mainloop_consumer_state.index,
                                    k_block_next, a_copy_half_layout,
                                ),
                                tCrA_copy_view[None, None, k_block_next],
                            )
                        else:
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        if cutlass.const_expr(not self.mma_before_prefetch):
                            if cutlass.const_expr(
                                not (self.mma_before_wait and k_block_idx == num_k_blocks - 1)
                            ):
                                cute.gemm(
                                    tiled_mma,
                                    accumulators,
                                    tCrA[None, None, k_block_idx],
                                    tCrB[None, None, k_block_idx],
                                    accumulators,
                                )
                    if cutlass.const_expr(self.promote_each_tile):
                        accum_sum.store(accum_sum.load() + accumulators.load())
                        accumulators.fill(0.0)
                # end of for loop
                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if cutlass.const_expr(self.mma_before_prefetch):
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )
                    if k_block_next > 0:
                        if cutlass.const_expr(self.physical_a_copy):
                            cute.copy(
                                smem_tiled_copy_A,
                                self._physical_a_source(
                                    a_physical_base, mainloop_consumer_state.index,
                                    k_block_next, a_copy_half_layout,
                                ),
                                tCrA_copy_view[None, None, k_block_next],
                            )
                        else:
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                    if cutlass.const_expr(not self.mma_before_prefetch):
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )

                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOG
                # /////////////////////////////////////////////////////////////////////////////

                if cutlass.const_expr(self.promote_each_tile):
                    accumulators.store(accum_sum.load() + accumulators.load())

                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )

                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        4,
                    ),
                    self.c_dtype,
                )

                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                # (R2S, R2S_M, R2S_N, PIPE_D)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                # (R2S, R2S_M, R2S_N)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                # Match every accumulator register to its tile-local (m,n).
                local_coords = cute.make_identity_tensor(self.tile_shape_mnk[:2])
                mma_coords = thr_mma.partition_C(local_coords)
                r2s_coords = tiled_copy_r2s.retile(mma_coords)

                # Allocate D registers.
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                size_tRS_rD = cute.size(tRS_rD)
                if cutlass.const_expr(self.dedup_scale_loads):
                    assert size_tRS_rD == self.epi_tile[1] // 4, "Scale dedup value count does not match epilogue width"
                    assert self.atom_layout == (4, 2, 1), "Scale dedup requires atom 4x2"
                    assert self.epi_tile in ((64, 32), (64, 64)), "Scale dedup requires epilogue 64x32 or 64x64"

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                if cutlass.const_expr(not self.output_warp):
                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        sepi_for_tma_partition,
                        tcgc_for_tma_partition,
                    )

                epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
                epi_tile_shape = tcgc_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(1, epi_tile_shape[0])
                )

                if cutlass.const_expr(not self.output_warp):
                    # Initialize tma store pipeline
                    tma_store_producer_group = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp,
                    )
                    tma_store_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.epi_stage,
                        producer_group=tma_store_producer_group,
                    )
                    if cutlass.const_expr(self.defer_epilogue_drain):
                        # Previous output stores may read sC while this tile computes on sA/sB.
                        # Drain their shared-source reads before any consumer overwrites sC.
                        tma_store_pipeline.producer_tail()
                        if cutlass.const_expr(self.sync_epilogue_reuse):
                            self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.reuse_column_scales):
                    assert self.epi_tile == (64, 32) and self.atom_layout == (4, 2, 1), "Column-scale reuse requires epilogue64x32 and atom4x2"
                    assert size_tRS_rD == 8 and epi_tile_num == 24, "Column-scale reuse requires four M subtiles per N group"
                    column_scale_cache = cute.make_rmem_tensor((4,), cutlass.Float32)
                # Trace-time map: epilogue job -> (row scales, column scales) registers.
                epi_scales = {}
                if cutlass.const_expr(self.stage_scales):
                    scale_pipeline.consumer_wait(scale_consumer_state)

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    if cutlass.const_expr(self.epi_tile[1] == 64):
                        # R2S retile keeps 64x32 basic tiles in M-first order.
                        # Gather adjacent N tiles instead of adjacent flattened M tiles.
                        epi_m_tiles = self.tile_shape_mnk[0] // 64
                        epi_acc_base = 8 * (
                            epi_idx % epi_m_tiles
                            + 2 * epi_m_tiles * (epi_idx // epi_m_tiles)
                        )
                    if cutlass.const_expr(self.dedup_scale_loads):
                        # Without prefetch each job loads its own scales; with it, job 0 loads jobs 0 and 1
                        # and job j loads job j+1, so the loads cross one job's math and barriers.
                        scale_job_begin = epi_idx
                        scale_job_end = epi_idx + 1
                        if cutlass.const_expr(self.prefetch_scales):
                            scale_job_begin = 0 if epi_idx == 0 else epi_idx + 1
                            scale_job_end = min(epi_idx + 2, epi_tile_num)
                        for scale_job in cutlass.range_constexpr(scale_job_begin, scale_job_end):
                            # Row scales are local; column scales may span four M subtiles.
                            job_scale_a = cute.make_rmem_tensor((2,), cutlass.Float32)
                            if cutlass.const_expr(self.reuse_column_scales):
                                job_scale_b = column_scale_cache
                            else:
                                job_scale_b = cute.make_rmem_tensor((self.epi_tile[1] // 8,), cutlass.Float32)
                            for scale_m in cutlass.range_constexpr(2):
                                row_acc_idx = scale_job * size_tRS_rD + 2 * scale_m
                                if cutlass.const_expr(self.epi_tile[1] == 64):
                                    row_acc_idx = epi_acc_base + 2 * scale_m
                                row_local_m = r2s_coords[row_acc_idx][0]
                                if cutlass.const_expr(self.stage_scales):
                                    job_scale_a[scale_m] = sScale[row_local_m]
                                else:
                                    row_global_m = tile_coord_mnl[0] * self.tile_shape_mnk[0] + row_local_m
                                    job_scale_a[scale_m] = cutlass.Float32(0.0)
                                    if row_global_m < cute.size(scale_a):
                                        job_scale_a[scale_m] = scale_a[row_global_m]
                            if cutlass.const_expr(not self.reuse_column_scales or scale_job % 4 == 0):
                                for scale_n in cutlass.range_constexpr(self.epi_tile[1] // 8):
                                    col_acc_idx = scale_job * size_tRS_rD + 4 * (scale_n // 2) + scale_n % 2
                                    if cutlass.const_expr(self.epi_tile[1] == 64):
                                        col_epi_v = 4 * (scale_n // 2) + scale_n % 2
                                        col_acc_idx = epi_acc_base + col_epi_v % 8 + 8 * epi_m_tiles * (col_epi_v // 8)
                                    col_local_n = r2s_coords[col_acc_idx][1]
                                    if cutlass.const_expr(self.stage_scales):
                                        job_scale_b[scale_n] = sScale[self.tile_shape_mnk[0] + col_local_n]
                                    else:
                                        col_global_n = tile_coord_mnl[1] * self.tile_shape_mnk[1] + col_local_n
                                        job_scale_b[scale_n] = cutlass.Float32(0.0)
                                        if col_global_n < cute.size(scale_b):
                                            job_scale_b[scale_n] = scale_b[col_global_n]
                            epi_scales[scale_job] = (job_scale_a, job_scale_b)
                        epi_scale_a, epi_scale_b = epi_scales[epi_idx]
                    # Copy from accumulators to D registers
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        acc_idx = epi_idx * size_tRS_rD + epi_v
                        if cutlass.const_expr(self.epi_tile[1] == 64):
                            acc_idx = epi_acc_base + epi_v % 8 + 8 * epi_m_tiles * (epi_v // 8)
                        local_m, local_n = r2s_coords[acc_idx]
                        global_m = tile_coord_mnl[0] * self.tile_shape_mnk[0] + local_m
                        global_n = tile_coord_mnl[1] * self.tile_shape_mnk[1] + local_n
                        if cutlass.const_expr(self.write_raw):
                            if global_m < cute.size(scale_a) and global_n < cute.size(scale_b):
                                raw_output[global_m, global_n] = tRS_rAcc[acc_idx]
                        if cutlass.const_expr(self.dedup_scale_loads):
                            sa = epi_scale_a[(epi_v // 2) % 2]
                            sb = epi_scale_b[(epi_v % 2) + 2 * (epi_v // 4)]
                        else:
                            # Guard the M tail before loading scale_a; TMA masks C stores.
                            sa = cutlass.Float32(0.0)
                            sb = cutlass.Float32(0.0)
                            if global_m < cute.size(scale_a):
                                sa = scale_a[global_m]
                            if global_n < cute.size(scale_b):
                                sb = scale_b[global_n]
                        scaled_by_b = tRS_rAcc[acc_idx] * sb
                        tRS_rD[epi_v] = scaled_by_b * sa

                    # Type conversion
                    tRS_rD_out = cute.make_rmem_tensor(
                        tRS_rD_layout.shape, self.c_dtype
                    )
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    # Register to shared memory
                    if cutlass.const_expr(self.output_warp):
                        # Every writer must observe source-read completion before R2S.
                        output_pipeline.producer_acquire(output_producer_state)
                        epi_buffer = output_producer_state.index
                    else:
                        epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    # barrier for sync
                    self.epilog_sync_barrier.arrive_and_wait()

                    if cutlass.const_expr(self.output_warp):
                        if tidx == 0:
                            output_pipeline.producer_commit(output_producer_state)
                        output_producer_state.advance()
                    else:
                        # Get the global memory coordinate for the current epi tile.
                        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                        # Copy from shared memory to global memory
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, gmem_coord)],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()
                        if cutlass.const_expr(self.sync_epilogue_reuse):
                            # TMA wait_group is local to the issuing warp. All writers
                            # must observe its completion before reusing the next buffer.
                            self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.stage_scales):
                    # Every MMA thread passed the last job's named barrier, so all of this tile's LDS are done.
                    if tidx == 0:
                        scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()

                # Advance to the next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                if cutlass.const_expr(not self.output_warp):
                    if cutlass.const_expr(self.defer_epilogue_drain):
                        if not work_tile.is_valid_tile:
                            # No following K tile will reach the pre-epilogue drain.
                            tma_store_pipeline.producer_tail()
                            if cutlass.const_expr(self.sync_epilogue_reuse):
                                self.epilog_sync_barrier.arrive_and_wait()
                    else:
                        tma_store_pipeline.producer_tail()
                        if cutlass.const_expr(self.sync_epilogue_reuse):
                            # Do not let another warp reuse sC for the next persistent tile
                            # until the issuing warp has drained its remaining stores.
                            self.epilog_sync_barrier.arrive_and_wait()
                # End of for k_tile loop
            # End of while loop
            if cutlass.const_expr(self.output_warp):
                # All writers wait for both free phases before leaving the CTA.
                output_pipeline.producer_tail(output_producer_state)
        # End of MMA warp group
        # Start of DMA warp group
        elif warp_idx < self.num_mma_warps + self.producer_warps:
            if cutlass.const_expr(self.emit_setmaxregister):
                cute.arch.setmaxregister_decrease(self.load_register_requirement)

            # All producer warps release registers; only the first issues TMA.
            if warp_idx == self.num_mma_warps:
                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    if cutlass.const_expr(self.compact_tail):
                        tile_coord_mnl = self._compact_tile_coord(tile_coord_mnl, tile_sched_params.problem_shape_ntile_mnl)
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

                    mainloop_producer_state.reset_count()

                    for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                        # /////////////////////////////////////////////////////////////////////////////
                        #  Wait for A/B buffers to be empty before loading into them
                        #  Also sets the transaction barrier for the A/B buffers
                        # /////////////////////////////////////////////////////////////////////////////
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)

                        # /////////////////////////////////////////////////////////////////////////////
                        #  Slice to global/shared memref to current k_tile
                        # /////////////////////////////////////////////////////////////////////////////
                        tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                        tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                        tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                        tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                        # /////////////////////////////////////////////////////////////////////////////
                        #  TMA load A/B
                        # /////////////////////////////////////////////////////////////////////////////
                        if cutlass.const_expr(self.a_cache_evict_last):
                            cute.copy(
                                tma_atom_a,
                                tAgA_k,
                                tAsA_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=a_mcast_mask,
                                cache_policy=cutlass.Int64(0x14F0000000000000),
                            )
                        else:
                            cute.copy(
                                tma_atom_a,
                                tAgA_k,
                                tAsA_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=a_mcast_mask,
                            )
                        if cutlass.const_expr(self.b_cache_evict_first):
                            # CacheHintSm90::EVICT_FIRST.
                            cute.copy(
                                tma_atom_b,
                                tBgB_k,
                                tBsB_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=b_mcast_mask,
                                cache_policy=cutlass.Int64(0x12F0000000000000),
                            )
                        elif cutlass.const_expr(self.b_cache_evict_last):
                            # CacheHintSm90/Sm100::EVICT_LAST; installed TMA API requires Int64.
                            cute.copy(
                                tma_atom_b,
                                tBgB_k,
                                tBsB_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=b_mcast_mask,
                                cache_policy=cutlass.Int64(0x14F0000000000000),
                            )
                        else:
                            cute.copy(
                                tma_atom_b,
                                tBgB_k,
                                tBsB_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=b_mcast_mask,
                            )
                        # Mainloop pipeline's producer commit is a NOP
                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                # end of while loop

                # Wait A/B buffer empty
                mainloop_pipeline.producer_tail(mainloop_producer_state)

            if cutlass.const_expr(self.stage_scales):
                if warp_idx == self.num_mma_warps + 2:
                    scale_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                    scale_lane = cute.arch.lane_idx()
                    while work_tile.is_valid_tile:
                        scale_tile_coord = work_tile.tile_idx
                        if cutlass.const_expr(self.compact_tail):
                            scale_tile_coord = self._compact_tile_coord(
                                scale_tile_coord, tile_sched_params.problem_shape_ntile_mnl
                            )
                        scale_pipeline.producer_acquire(scale_producer_state)
                        for scale_j in cutlass.range_constexpr(self.tile_shape_mnk[0] // self.num_threads_per_warp):
                            scale_row = scale_lane + self.num_threads_per_warp * scale_j
                            scale_row_global = scale_tile_coord[0] * self.tile_shape_mnk[0] + scale_row
                            sScale[scale_row] = cutlass.Float32(0.0)
                            if scale_row_global < cute.size(scale_a):
                                sScale[scale_row] = scale_a[scale_row_global]
                        for scale_j in cutlass.range_constexpr(self.tile_shape_mnk[1] // self.num_threads_per_warp):
                            scale_col = scale_lane + self.num_threads_per_warp * scale_j
                            scale_col_global = scale_tile_coord[1] * self.tile_shape_mnk[1] + scale_col
                            sScale[self.tile_shape_mnk[0] + scale_col] = cutlass.Float32(0.0)
                            if scale_col_global < cute.size(scale_b):
                                sScale[self.tile_shape_mnk[0] + scale_col] = scale_b[scale_col_global]
                        scale_pipeline.producer_commit(scale_producer_state)
                        scale_producer_state.advance()
                        tile_sched.advance_to_next_work()
                        work_tile = tile_sched.get_current_work()

            if cutlass.const_expr(self.output_warp):
                if warp_idx == self.num_mma_warps + 1:
                    # Keep scheduler coordinates warp-uniform; only this fixed lane issues stores.
                    output_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.epi_stage
                    )
                    output_release_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.epi_stage
                    )
                    if cutlass.const_expr(self.specialize_output_state):
                        output_has_pending = cutlass.Boolean(False)
                    # Two store stages = wait_group.read 1: release still lags one job, so a third
                    # output buffer is pure slack for the MMA writers, not a longer release lag.
                    output_store_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=2,
                        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                    )
                    while work_tile.is_valid_tile:
                        output_tile_coord = work_tile.tile_idx
                        if cutlass.const_expr(self.compact_tail):
                            output_tile_coord = self._compact_tile_coord(
                                output_tile_coord, tile_sched_params.problem_shape_ntile_mnl
                            )
                        output_gC_slice = gC_mnl[(None, None, *output_tile_coord)]
                        output_gC_divided = cute.zipped_divide(output_gC_slice, self.epi_tile)
                        output_sD, output_gD = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_c,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sC, 0, 2),
                            output_gC_divided,
                        )
                        output_epi_shape = output_gC_divided.shape[1]
                        output_epi_layout = cute.make_layout(
                            output_epi_shape, stride=(1, output_epi_shape[0])
                        )
                        for output_epi_idx in cutlass.range_constexpr(
                            cute.size(output_gC_divided, mode=[1])
                        ):
                            if cutlass.const_expr(self.specialize_output_state):
                                # 24 jobs close the S-stage index/phase cycle (period 2S jobs) at each tile.
                                assert cute.size(output_gC_divided, mode=[1]) % (2 * self.epi_stage) == 0
                                output_consumer_state = pipeline.PipelineState(
                                    self.epi_stage, cutlass.Int32(0),
                                    cutlass.Int32(output_epi_idx % self.epi_stage),
                                    cutlass.Int32((output_epi_idx // self.epi_stage) % 2),
                                )
                                output_release_state = pipeline.PipelineState(
                                    self.epi_stage, cutlass.Int32(0),
                                    cutlass.Int32((output_epi_idx - 1) % self.epi_stage),
                                    cutlass.Int32(((output_epi_idx - 1) % (2 * self.epi_stage)) // self.epi_stage),
                                )
                            output_pipeline.consumer_wait(output_consumer_state)
                            output_gmem_coord = output_epi_layout.get_hier_coord(output_epi_idx)
                            if tidx == (self.num_mma_warps + 1) * self.num_threads_per_warp:
                                if cutlass.const_expr(self.c_store_evict_first):
                                    cute.copy(
                                        tma_atom_c,
                                        output_sD[(None, output_consumer_state.index)],
                                        output_gD[(None, output_gmem_coord)],
                                        cache_policy=cutlass.Int64(0x12F0000000000000),
                                    )
                                else:
                                    cute.copy(
                                        tma_atom_c,
                                        output_sD[(None, output_consumer_state.index)],
                                        output_gD[(None, output_gmem_coord)],
                                    )
                                output_store_pipeline.producer_commit()
                                if cutlass.const_expr(self.specialize_output_state):
                                    if cutlass.const_expr(output_epi_idx == 0):
                                        if output_has_pending:
                                            output_store_pipeline.producer_acquire()
                                            output_pipeline.consumer_release(output_release_state)
                                    else:
                                        output_store_pipeline.producer_acquire()
                                        output_pipeline.consumer_release(output_release_state)
                                else:
                                    if output_consumer_state.count > 0:
                                        # Release the previous source BEFORE waiting for the next ready.
                                        output_store_pipeline.producer_acquire()
                                        output_pipeline.consumer_release(output_release_state)
                            if cutlass.const_expr(not self.specialize_output_state):
                                if output_consumer_state.count > 0:
                                    output_release_state.advance()
                                output_consumer_state.advance()
                        if cutlass.const_expr(self.specialize_output_state):
                            output_has_pending = cutlass.Boolean(True)
                        tile_sched.advance_to_next_work()
                        work_tile = tile_sched.get_current_work()
                    if cutlass.const_expr(self.specialize_output_state):
                        if output_has_pending:
                            if tidx == (self.num_mma_warps + 1) * self.num_threads_per_warp:
                                output_store_pipeline.producer_tail()
                                # Last job (23) of a tile: index 23 % S, phase (23 % 2S) // S.
                                output_final_release_state = pipeline.PipelineState(
                                    self.epi_stage, cutlass.Int32(0),
                                    cutlass.Int32(23 % self.epi_stage),
                                    cutlass.Int32((23 % (2 * self.epi_stage)) // self.epi_stage),
                                )
                                output_pipeline.consumer_release(output_final_release_state)
                    else:
                        if output_consumer_state.count > 0:
                            if tidx == (self.num_mma_warps + 1) * self.num_threads_per_warp:
                                # read0 drains shared-source reads; kernel completion retains its contract.
                                output_store_pipeline.producer_tail()
                                output_pipeline.consumer_release(output_release_state)
        return

    def _compute_stages(
        self,
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """
        epi_stage = self.epilogue_stages
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_dtype: type[cutlass.Numeric],
        b_layout: cute.Layout,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: cute.Layout,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout for matrix A
        :type a_layout: Layout
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout for matrix B
        :type b_layout: Layout
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: leading dimension of the output matrix C
        :type c_layout: Layout
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    def _compute_grid(
        self,
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = self.cluster_shape_mnk
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size=1 if self.compact_tail else self.swizzle_size,
            raster_along_m=True,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

