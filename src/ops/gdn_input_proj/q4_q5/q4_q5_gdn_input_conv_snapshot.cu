#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Geometry is selected per launch: 27B (hidden 5120, 48 value heads) and
// 9B (hidden 4096, 32 value heads) share every kernel below.
template <int CHidden>
struct ConvGeom {
    static constexpr int kHidden      = CHidden;
    static constexpr int kQueryRows   = 2048;
    static constexpr int kKeyRows     = 2048;
    static constexpr int kValueRows   = CHidden == 4096 ? 4096 : 6144;
    static constexpr int kZRows       = kValueRows;
    static constexpr int kValueZRows  = kValueRows + kZRows;
    static constexpr int kQkRows      = kQueryRows + kKeyRows;
    static constexpr int kChannels    = kQkRows + kValueRows;
    static constexpr int kValueOffset = kQkRows;
};

using Q4ScheduleC4 = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4ScheduleC8 = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

enum class PdlOrder {
    Q4ThenQ5,
    Q5ThenQ4,
};

template <int CH, class Publish>
GdnConvEpilogue<Publish> make_epilogue(const Tensor& conv_weight, const Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       Tensor& query, Tensor& key, Tensor& value,
                                       int global_row_offset, Publish publish) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        ConvGeom<CH>::kChannels,
        ConvGeom<CH>::kQueryRows,
        ConvGeom<CH>::kKeyRows,
        ConvGeom<CH>::kValueRows,
        global_row_offset,
        static_cast<std::int32_t>(query.ne[1]),
        0,
        publish,
    };
}

template <class Publish>
struct Q4GdnDecodeEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        const float projected[1]{value};
        conv.store(row, projected);
    }
};

template <int Tokens, class Publish>
struct Q4GdnSmallTEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <bool, int, int TileCols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t row,
               std::int32_t, std::int32_t active_cols, const float (&values)[TileCols]) const {
        float projected[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) { projected[token] = values[token]; }
        if (active_cols == Tokens) { conv.store(row, projected); }
    }
};

template <int CH, class Publish>
struct Q5GdnDecodeEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        if (row < ConvGeom<CH>::kValueRows) {
            const float projected[1]{value};
            conv.store(row, projected);
        } else {
            z[row - ConvGeom<CH>::kValueRows] = __float2bfloat16_rn(value);
        }
    }
};

template <int CH, int Tokens, class Publish>
struct Q5GdnSmallTEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <bool, int, int ProducedTokens>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                                               std::int32_t, std::int32_t row,
                                               const float (&values)[ProducedTokens]) const {
        static_assert(ProducedTokens == Tokens);
        if (row < ConvGeom<CH>::kValueRows) {
            conv.store(row, values);
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                z[static_cast<std::int64_t>(token) * ConvGeom<CH>::kZRows + row -
                  ConvGeom<CH>::kValueRows] = __float2bfloat16_rn(values[token]);
            }
        }
    }
};

template <int Tokens, class Publish>
struct Q4GdnMmaConvEpilogue {
    static constexpr bool kIsTileEpilogue = true;
    GdnConvEpilogue<Publish> conv;

    template <int ActiveCols, int TileCols, int OutputRows>
    __device__ __forceinline__ void store_tile(float* smem_raw, int row0, int gid, int lane,
                                               float (&acc)[TileCols / 8][4]) const {
        static_assert(Tokens == ActiveCols);
        constexpr int kNt = TileCols / 8;
        auto* smem = reinterpret_cast<float(*)[TileCols]>(smem_raw);

        const int col_base = (lane & 3) * 2;
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const int col0 = nt * 8 + col_base;
            const int col1 = col0 + 1;
            smem[gid][col0]     = acc[nt][0];
            smem[gid][col1]     = acc[nt][1];
            smem[gid + 8][col0] = acc[nt][2];
            smem[gid + 8][col1] = acc[nt][3];
        }

        __syncwarp();

        if (lane < 16) {
            const int local_row = row0 + lane;
            if (local_row < OutputRows) {
                float projected[Tokens];
#pragma unroll
                for (int t = 0; t < Tokens; ++t) {
                    projected[t] = smem[lane][t];
                }
                conv.store(local_row, projected);
            }
        }
    }
};

template <int CH, int Tokens, class Publish>
struct Q5GdnMmaConvEpilogue {
    static constexpr bool kIsTileEpilogue = true;
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <int ActiveCols, int TileCols, int OutputRows>
    __device__ __forceinline__ void store_tile(float* smem_raw, int row0, int gid, int lane,
                                               float (&acc)[TileCols / 8][4]) const {
        static_assert(Tokens == ActiveCols);
        constexpr int kNt = TileCols / 8;
        auto* smem = reinterpret_cast<float(*)[TileCols]>(smem_raw);

        const int col_base = (lane & 3) * 2;
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const int col0 = nt * 8 + col_base;
            const int col1 = col0 + 1;
            smem[gid][col0]     = acc[nt][0];
            smem[gid][col1]     = acc[nt][1];
            smem[gid + 8][col0] = acc[nt][2];
            smem[gid + 8][col1] = acc[nt][3];
        }

        __syncwarp();

        if (lane < 16) {
            const int local_row = row0 + lane;
            if (local_row < OutputRows) {
                float projected[Tokens];
#pragma unroll
                for (int t = 0; t < Tokens; ++t) {
                    projected[t] = smem[lane][t];
                }

                if (local_row < ConvGeom<CH>::kValueRows) {
                    conv.store(local_row, projected);
                } else {
#pragma unroll
                    for (int t = 0; t < Tokens; ++t) {
                        z[static_cast<std::int64_t>(t) * ConvGeom<CH>::kZRows +
                          (local_row - ConvGeom<CH>::kValueRows)] =
                            __float2bfloat16_rn(projected[t]);
                    }
                }
            }
        }
    }
};

template <int CH, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q4_t1(const Tensor& x, const Weight& qk_weight,
                  const GdnConvEpilogue<Publish>& qk_epilogue, Tensor& query, cudaStream_t stream) {
    constexpr int q4_threads = Q4GemvR1W8DirectSchedule::kThreads;
    constexpr int q4_blocks  = ConvGeom<CH>::kQkRows / Q4GemvR1W8DirectSchedule::kRowsPerCta;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(q4_blocks), dim3(q4_threads), 0, stream},
            q4_rowsplit_gemv_kernel<Q4GemvR1W8DirectSchedule, false, 0,
                                    Q4GdnDecodeEpilogue<Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, ConvGeom<CH>::kQkRows,
            ConvGeom<CH>::kHidden, Q4GdnDecodeEpilogue<Publish>{qk_epilogue}));
    } else {
        q4_rowsplit_gemv_kernel<Q4GemvR1W8DirectSchedule, false, 0, Q4GdnDecodeEpilogue<Publish>,
                                TriggerPdl, JoinPdl><<<q4_blocks, q4_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, ConvGeom<CH>::kQkRows,
            ConvGeom<CH>::kHidden, Q4GdnDecodeEpilogue<Publish>{qk_epilogue});
    }
}

template <int CH, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_t1(const Tensor& x, const Weight& value_z_weight,
                  const GdnConvEpilogue<Publish>& value_epilogue, Tensor& value, Tensor& z,
                  cudaStream_t stream) {
    constexpr int q5_rows_per_block = 16;
    constexpr int q5_threads        = q5_rows_per_block * 32;
    constexpr int q5_blocks         = ConvGeom<CH>::kValueZRows / q5_rows_per_block;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(q5_blocks), dim3(q5_threads), 0, stream},
            q5_rowsplit_gemv_kernel<ConvGeom<CH>::kValueZRows, ConvGeom<CH>::kHidden,
                                    q5_rows_per_block, 2, true, false, true,
                                    ConvGeom<CH>::kValueRows, Q5GdnDecodeEpilogue<CH, Publish>,
                                    TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Q5GdnDecodeEpilogue<CH, Publish>{value_epilogue,
                                             static_cast<__nv_bfloat16*>(z.data)}));
    } else {
        q5_rowsplit_gemv_kernel<ConvGeom<CH>::kValueZRows, ConvGeom<CH>::kHidden,
                                q5_rows_per_block, 2, true, false, true,
                                ConvGeom<CH>::kValueRows, Q5GdnDecodeEpilogue<CH, Publish>,
                                TriggerPdl, JoinPdl>
            <<<q5_blocks, q5_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                Q5GdnDecodeEpilogue<CH, Publish>{value_epilogue,
                                                 static_cast<__nv_bfloat16*>(z.data)});
    }
}

template <int CH, int Tokens, class Q4Schedule, class Publish, bool TriggerPdl, bool JoinPdl,
          bool Dependent>
void launch_q4_small_t(const Tensor& x, const Weight& qk_weight,
                       const GdnConvEpilogue<Publish>& qk_epilogue, Tensor& query,
                       cudaStream_t stream) {
    const dim3 q4_grid(ConvGeom<CH>::kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q4_grid, dim3(Q4Schedule::kThreads), 0, stream},
            q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0,
                                         Q4GdnSmallTEpilogue<Tokens, Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, ConvGeom<CH>::kQueryRows, 0,
            ConvGeom<CH>::kQkRows, ConvGeom<CH>::kHidden, Tokens, ConvGeom<CH>::kHidden,
            Q4GdnSmallTEpilogue<Tokens, Publish>{qk_epilogue}));
    } else {
        q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0,
                                     Q4GdnSmallTEpilogue<Tokens, Publish>, TriggerPdl, JoinPdl>
            <<<q4_grid, Q4Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(qk_weight.qdata),
                static_cast<const std::uint8_t*>(qk_weight.scales),
                static_cast<__nv_bfloat16*>(query.data), nullptr, ConvGeom<CH>::kQueryRows, 0,
                ConvGeom<CH>::kQkRows, ConvGeom<CH>::kHidden, Tokens, ConvGeom<CH>::kHidden,
                Q4GdnSmallTEpilogue<Tokens, Publish>{qk_epilogue});
    }
}

template <int CH, int Tokens, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_small_t(const Tensor& x, const Weight& value_z_weight,
                       const GdnConvEpilogue<Publish>& value_epilogue, Tensor& value, Tensor& z,
                       cudaStream_t stream) {
    constexpr int q5_threads = 4 * 32;
    const dim3 q5_grid(ConvGeom<CH>::kValueZRows, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q5_grid, dim3(q5_threads), 0, stream},
            q5_rowsplit_gemm_simt_split4_kernel<
                Q5RowSplitSimtSchedule, Tokens, 5, ConvGeom<CH>::kHidden, true,
                ConvGeom<CH>::kValueRows, Q5GdnSmallTEpilogue<CH, Tokens, Publish>, TriggerPdl,
                JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            ConvGeom<CH>::kValueZRows, ConvGeom<CH>::kValueRows, ConvGeom<CH>::kHidden, Tokens,
            ConvGeom<CH>::kHidden, 5,
            Q5GdnSmallTEpilogue<CH, Tokens, Publish>{
                value_epilogue,
                static_cast<__nv_bfloat16*>(z.data),
            }));
    } else {
        q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, 5,
                                            ConvGeom<CH>::kHidden, true,
                                            ConvGeom<CH>::kValueRows,
                                            Q5GdnSmallTEpilogue<CH, Tokens, Publish>, TriggerPdl,
                                            JoinPdl>
            <<<q5_grid, q5_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                ConvGeom<CH>::kValueZRows, ConvGeom<CH>::kValueRows, ConvGeom<CH>::kHidden, Tokens,
                ConvGeom<CH>::kHidden, 5,
                Q5GdnSmallTEpilogue<CH, Tokens, Publish>{
                    value_epilogue,
                    static_cast<__nv_bfloat16*>(z.data),
                });
    }
}

template <int CH, PdlOrder Order, class Publish>
void launch_t1(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
               const GdnConvEpilogue<Publish>& qk_epilogue,
               const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
               Tensor& z, cudaStream_t stream) {
    // The Q4 and Q5 sides read the same activation but write disjoint output/state rows. The
    // dependent side therefore computes before waiting, then joins the producer at kernel exit.
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_t1<CH, Publish, true, false, false>(x, value_z_weight, value_epilogue, value, z,
                                                      stream);
        launch_q4_t1<CH, Publish, false, true, true>(x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_q4_t1<CH, Publish, true, false, false>(x, qk_weight, qk_epilogue, query, stream);
        launch_q5_t1<CH, Publish, false, true, true>(x, value_z_weight, value_epilogue, value, z,
                                                     stream);
    }
}

template <int CH, int Tokens, class Q4Schedule, PdlOrder Order, class Publish>
void launch_small_t_schedule(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const GdnConvEpilogue<Publish>& qk_epilogue,
                             const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query,
                             Tensor& value, Tensor& z, cudaStream_t stream) {
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_small_t<CH, Tokens, Publish, true, false, false>(x, value_z_weight,
                                                                   value_epilogue, value, z,
                                                                   stream);
        launch_q4_small_t<CH, Tokens, Q4Schedule, Publish, false, true, true>(
            x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_q4_small_t<CH, Tokens, Q4Schedule, Publish, true, false, false>(
            x, qk_weight, qk_epilogue, query, stream);
        launch_q5_small_t<CH, Tokens, Publish, false, true, true>(x, value_z_weight,
                                                                  value_epilogue, value, z,
                                                                  stream);
    }
}

template <int CH, int Tokens, PdlOrder Order, class Publish>
void launch_small_t_mma(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                        const GdnConvEpilogue<Publish>& qk_epilogue,
                        const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
                        Tensor& z, cudaStream_t stream) {
    constexpr int TileCols = Tokens <= 8 ? 8 : 16;
    using Q4Geometry = Q4SmallTGeometry<ConvGeom<CH>::kQkRows, ConvGeom<CH>::kHidden>;
    using Q5Geometry = Q5SmallTGeometry<ConvGeom<CH>::kValueZRows, ConvGeom<CH>::kHidden>;
    using Q4Epilogue = Q4GdnMmaConvEpilogue<Tokens, Publish>;
    using Q5Epilogue = Q5GdnMmaConvEpilogue<CH, Tokens, Publish>;

    constexpr int kQ4Blocks = ConvGeom<CH>::kQkRows / Q4DraftSmallTSchedule::kRowsPerCta;
    constexpr int kQ5Blocks = ConvGeom<CH>::kValueZRows / Q5SmallTSchedule::kRowsPerCta;

    const auto in_ld    = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto value_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));

    const Q4Epilogue q4_epilogue{qk_epilogue};
    const Q5Epilogue q5_epilogue{value_epilogue, static_cast<__nv_bfloat16*>(z.data)};

    const auto launch_q4 = [&] {
        q4_small_t_mma_kernel<Q4Geometry, TileCols, Tokens, Q4Epilogue>
            <<<kQ4Blocks, Q4DraftSmallTSchedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(qk_weight.qdata),
                static_cast<const std::uint8_t*>(qk_weight.scales),
                static_cast<__nv_bfloat16*>(query.data),
                q4_epilogue);
    };

    const auto launch_q5 = [&] {
        q5_small_t_mma_kernel<Q5Geometry, TileCols, Tokens, Q5Epilogue>
            <<<kQ5Blocks, Q5SmallTSchedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data),
                in_ld, value_ld, q5_epilogue);
    };

    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5();
        launch_q4();
    } else {
        launch_q4();
        launch_q5();
    }
}

template <int CH, int Tokens, PdlOrder Order, class Publish>
void launch_small_t(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    const GdnConvEpilogue<Publish>& qk_epilogue,
                    const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
                    Tensor& z, cudaStream_t stream) {
    if constexpr (Tokens <= 4) {
        launch_small_t_schedule<CH, Tokens, Q4ScheduleC4, Order, Publish>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    } else {
        launch_small_t_schedule<CH, Tokens, Q4ScheduleC8, Order, Publish>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    }
}

template <int CH, PdlOrder Order, class Publish>
void launch_conv(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                 const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
                 const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                 Publish publish, cudaStream_t stream) {
    const GdnConvEpilogue<Publish> qk_epilogue =
        make_epilogue<CH>(conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
                          0, publish);
    const GdnConvEpilogue<Publish> value_epilogue =
        make_epilogue<CH>(conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
                          ConvGeom<CH>::kValueOffset, publish);

    switch (x.ne[1]) {
    case 1:
        launch_t1<CH, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                      query, value, z, stream);
        break;
    case 2:
        launch_small_t_mma<CH, 2, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue,
                                                  value_epilogue, query, value, z, stream);
        break;
    case 3:
        launch_small_t_mma<CH, 3, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue,
                                                  value_epilogue, query, value, z, stream);
        break;
    case 5:
        launch_small_t_mma<CH, 5, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue,
                                                  value_epilogue, query, value, z, stream);
        break;
    case 6:
        launch_small_t_mma<CH, 6, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue,
                                                  value_epilogue, query, value, z, stream);
        break;
    default:
        throw std::invalid_argument("Q4/Q5 projection-epilogue GDN conv requires T=1..3 or 5..6");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q5_gdn_input_conv_snapshot_launch(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          Tensor& conv_states, const Tensor& valid_columns,
                                          const Tensor& initial_slot,
                                          const Tensor& snapshot_base_slot, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream) {
    if (x.ne[0] == 4096) {
        if (x.ne[1] == 2) {
            launch_conv<4096, PdlOrder::Q4ThenQ5>(
                x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
                query, key, value, z,
                SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                       static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                       ConvGeom<4096>::kChannels},
                stream);
        } else {
            launch_conv<4096, PdlOrder::Q5ThenQ4>(
                x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
                query, key, value, z,
                SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                       static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                       ConvGeom<4096>::kChannels},
                stream);
        }
        return;
    }
    if (x.ne[1] == 2) {
        launch_conv<5120, PdlOrder::Q4ThenQ5>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
            query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   ConvGeom<5120>::kChannels},
            stream);
    } else {
        launch_conv<5120, PdlOrder::Q5ThenQ4>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
            query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   ConvGeom<5120>::kChannels},
            stream);
    }
}

void q4_q5_gdn_input_conv_record_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, const Tensor& conv_weight,
                                        const Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_slot, Tensor& conv_record,
                                        Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                        cudaStream_t stream) {
    const auto publish_for = [&](auto channels) {
        return RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data), channels,
                                   x.ne[1]};
    };
    if (x.ne[0] == 4096) {
        const auto publish = publish_for(ConvGeom<4096>::kChannels);
        if (x.ne[1] == 2) {
            launch_conv<4096, PdlOrder::Q4ThenQ5>(x, qk_weight, value_z_weight, conv_weight,
                                                  conv_states, valid_columns, initial_slot, query,
                                                  key, value, z, publish, stream);
        } else {
            launch_conv<4096, PdlOrder::Q5ThenQ4>(x, qk_weight, value_z_weight, conv_weight,
                                                  conv_states, valid_columns, initial_slot, query,
                                                  key, value, z, publish, stream);
        }
        return;
    }
    const auto publish = publish_for(ConvGeom<5120>::kChannels);
    if (x.ne[1] == 2) {
        launch_conv<5120, PdlOrder::Q4ThenQ5>(x, qk_weight, value_z_weight, conv_weight,
                                              conv_states, valid_columns, initial_slot, query, key,
                                              value, z, publish, stream);
    } else {
        launch_conv<5120, PdlOrder::Q5ThenQ4>(x, qk_weight, value_z_weight, conv_weight,
                                              conv_states, valid_columns, initial_slot, query, key,
                                              value, z, publish, stream);
    }
}

} // namespace ninfer::ops::detail
