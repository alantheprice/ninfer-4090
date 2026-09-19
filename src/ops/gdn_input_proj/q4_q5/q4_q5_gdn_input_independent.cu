#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <int QkRows, int ValueRows, int ZRows, int Hidden>
struct GdnInputGeom {
    static constexpr std::int32_t kQkRows     = QkRows;
    static constexpr std::int32_t kValueRows  = ValueRows;
    static constexpr std::int32_t kZRows      = ZRows;
    static constexpr std::int32_t kValueZRows = ValueRows + ZRows;
    static constexpr std::int32_t kHidden     = Hidden;
};

// 27B: 48 value heads; 9B: 32 value heads.
using Geom27 = GdnInputGeom<4096, 6144, 6144, 5120>;
using Geom9  = GdnInputGeom<4096, 4096, 4096, 4096>;

using Q4GdnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

template <class Geom>
void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(Geom::kQkRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, Geom::kQkRows, Geom::kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geom, class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(Geom::kQkRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, out_ld, 0, Geom::kQkRows, Geom::kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geom, class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool full = (Geom::kQkRows % Schedule::kRowsPerCta) == 0 &&
                      ((Geom::kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Geom, Schedule, true>(x, weight, out, stream);
    } else {
        launch_q4_simt<Geom, Schedule, false>(x, weight, out, stream);
    }
}

template <class Geom, int ActiveCols>
void launch_q4_gdn_small_t_mma_active(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    using Geometry         = Q4SmallTGeometry<Geom::kQkRows, Geom::kHidden>;
    using Epilogue         = Q4SmallTStrideEpilogue;
    constexpr int kBlocks  = Geom::kQkRows / Q4DraftSmallTSchedule::kRowsPerCta;
    const auto out_ld      = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const Epilogue epilogue{static_cast<__nv_bfloat16*>(out.data), out_ld};

    q4_small_t_mma_kernel<Geometry, TileCols, ActiveCols, Epilogue>
        <<<kBlocks, Q4DraftSmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), epilogue);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geom>
using Q4GdnSmallTLauncherFor = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geom, std::size_t... Offsets>
constexpr auto make_q4_gdn_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4GdnSmallTLauncherFor<Geom>, sizeof...(Offsets)>{
        &launch_q4_gdn_small_t_mma_active<Geom, 2 + static_cast<int>(Offsets)>...};
}

template <class Geom>
void launch_q4(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q4_gemv<Geom>(x, weight, out, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        static constexpr auto kLaunchers =
            make_q4_gdn_small_t_launchers<Geom>(std::make_index_sequence<15>{});
        kLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,16]");
}

template <class Geom>
void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kThreads      = kRowsPerBlock * 32;
    q5_rowsplit_gemv_kernel<Geom::kValueZRows, Geom::kHidden, kRowsPerBlock, 2, true, false, true,
                            Geom::kValueRows>
        <<<Geom::kValueZRows / kRowsPerBlock, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geom, int Cols>
void launch_q5_split4_rows(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                           cudaStream_t stream) {
    constexpr int kThreads    = 4 * 32;
    constexpr int kRows       = 2;
    const std::int32_t out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(Geom::kValueZRows, kRows)), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_rows_kernel<Q5RowSplitSimtSchedule, kRows, Cols, Geom::kHidden / 1024,
                                             Geom::kHidden, true, Geom::kValueRows>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(value.data),
            static_cast<__nv_bfloat16*>(z.data), Geom::kValueZRows, out_ld, Geom::kHidden, Cols,
            weight.padded_shape[1], Geom::kHidden / 1024);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geom, int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream) {
    // Two rows per CTA share one widened activation. Below eight columns the extra accumulators
    // cost more than the shared conversion saves, so the single-row kernel stays.
    if constexpr (Cols >= 8) {
        launch_q5_split4_rows<Geom, Cols>(x, weight, value, z, stream);
        return;
    }
    constexpr int kThreads    = 4 * 32;
    const std::int32_t out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(Geom::kValueZRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, Geom::kHidden, true,
                                        Geom::kValueRows>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(value.data),
                                        static_cast<__nv_bfloat16*>(z.data), Geom::kValueZRows,
                                        out_ld, Geom::kHidden, Cols, weight.padded_shape[1],
                                        Geom::kHidden / 1024);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geom>
void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2: launch_q5_split4<Geom, 2>(x, weight, value, z, stream); return;
    case 3: launch_q5_split4<Geom, 3>(x, weight, value, z, stream); return;
    case 4: launch_q5_split4<Geom, 4>(x, weight, value, z, stream); return;
    case 5: launch_q5_split4<Geom, 5>(x, weight, value, z, stream); return;
    case 6: launch_q5_split4<Geom, 6>(x, weight, value, z, stream); return;
    case 7: launch_q5_split4<Geom, 7>(x, weight, value, z, stream); return;
    case 8: launch_q5_split4<Geom, 8>(x, weight, value, z, stream); return;
    case 9: launch_q5_split4<Geom, 9>(x, weight, value, z, stream); return;
    case 10: launch_q5_split4<Geom, 10>(x, weight, value, z, stream); return;
    case 11: launch_q5_split4<Geom, 11>(x, weight, value, z, stream); return;
    case 12: launch_q5_split4<Geom, 12>(x, weight, value, z, stream); return;
    case 13: launch_q5_split4<Geom, 13>(x, weight, value, z, stream); return;
    case 14: launch_q5_split4<Geom, 14>(x, weight, value, z, stream); return;
    case 15: launch_q5_split4<Geom, 15>(x, weight, value, z, stream); return;
    case 16: launch_q5_split4<Geom, 16>(x, weight, value, z, stream); return;
    default:
        throw std::invalid_argument("GDN Q5 split4 requires T in [2,16]");
    }
}

template <class Geom>
void launch_q5_simt_r8_c8(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                          cudaStream_t stream) {
    constexpr int kColsPerTile  = 8;
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(Geom::kValueZRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, kColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, kColsPerTile, kRowsPerBlock, kStages, true,
                                 Geom::kValueRows><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data), Geom::kValueZRows, out_ld, Geom::kHidden, cols,
        weight.padded_shape[1], Geom::kHidden / 1024);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geom, int ActiveCols>
void launch_q5_gdn_small_t_mma_active(const Tensor& x, const Weight& weight, Tensor& value,
                                      Tensor& z, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    using Geometry         = Q5SmallTGeometry<Geom::kValueZRows, Geom::kHidden>;
    using Epilogue         = Q5SmallTSplitEpilogue<Geom::kValueRows>;
    constexpr int kBlocks  = Geom::kValueZRows / Q5SmallTSchedule::kRowsPerCta;
    const auto in_ld       = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto value_ld    = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const auto z_ld        = static_cast<std::int32_t>(z.nb[1] / sizeof(__nv_bfloat16));
    const Epilogue epilogue{static_cast<__nv_bfloat16*>(z.data), z_ld};

    q5_small_t_mma_kernel<Geometry, TileCols, ActiveCols, Epilogue>
        <<<kBlocks, Q5SmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), in_ld, value_ld, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geom>
using Q5GdnSmallTLauncherFor =
    void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

template <class Geom, std::size_t... Offsets>
constexpr auto make_q5_gdn_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q5GdnSmallTLauncherFor<Geom>, sizeof...(Offsets)>{
        &launch_q5_gdn_small_t_mma_active<Geom, 2 + static_cast<int>(Offsets)>...};
}

template <class Geom>
void launch_q5(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv<Geom>(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        static constexpr auto kLaunchers =
            make_q5_gdn_small_t_launchers<Geom>(std::make_index_sequence<15>{});
        kLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, weight, value, z, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,16]");
}

} // namespace

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    if (x.ne[0] == 4096) {
        launch_q4<Geom9>(x, qk_weight, qk, stream);
        launch_q5<Geom9>(x, value_z_weight, value, z, stream);
        return;
    }
    launch_q4<Geom27>(x, qk_weight, qk, stream);
    launch_q5<Geom27>(x, value_z_weight, value, z, stream);
}

} // namespace ninfer::ops::detail
