using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TerraFX.Interop.Windows;

namespace SoftwareRasterizer;

using f128 = Vector128<float>;
using i128 = Vector128<int>;
using f256 = Vector256<float>;
using i256 = Vector256<int>;

public static unsafe class Intrinsics
{
    #region SSE

    public static f128 _mm_setzero_ps() => f128.Zero;

    public static f128 _mm_set1_ps(float a) => Vector128.Create(a);

    public static f128 _mm_setr_ps(float e3, float e2, float e1, float e0) => Vector128.Create(e3, e2, e1, e0);

    public static f128 _mm_loadu_ps(float* mem_addr) => Sse.LoadVector128(mem_addr);

    public static void _mm_storeu_ps(float* mem_addr, f128 a) => Sse.Store(mem_addr, a);

    public static void _mm_store_ss(float* mem_addr, f128 a) => Sse.StoreScalar(mem_addr, a);

    public static int _mm_movemask_ps(f128 a) => Sse.MoveMask(a);

    public static f128 _mm_unpacklo_ps(f128 a, f128 b) => Sse.UnpackLow(a, b);

    public static f128 _mm_unpackhi_ps(f128 a, f128 b) => Sse.UnpackHigh(a, b);

    public static f128 _mm_shuffle_ps(f128 a, f128 b, byte control) => Sse.Shuffle(a, b, control);

    public static f128 _mm_cvtsi32_ss(f128 a, int b) => Sse.ConvertScalarToVector128Single(a, b);

    public static f128 _mm_and_ps(f128 a, f128 b) => Sse.And(a, b);

    public static f128 _mm_andnot_ps(f128 a, f128 b) => Sse.AndNot(a, b);

    public static f128 _mm_or_ps(f128 a, f128 b) => Sse.Or(a, b);

    public static f128 _mm_xor_ps(f128 a, f128 b) => Sse.Xor(a, b);

    public static f128 _mm_rcp_ps(f128 a) => Sse.Reciprocal(a);

    public static f128 _mm_rsqrt_ps(f128 a) => Sse.ReciprocalSqrt(a);

    public static f128 _mm_add_ss(f128 a, f128 b) => Sse.AddScalar(a, b);
    public static f128 _mm_add_ps(f128 a, f128 b) => Sse.Add(a, b);

    public static f128 _mm_sub_ps(f128 a, f128 b) => Sse.Subtract(a, b);

    public static f128 _mm_mul_ss(f128 a, f128 b) => Sse.MultiplyScalar(a, b);
    public static f128 _mm_mul_ps(f128 a, f128 b) => Sse.Multiply(a, b);

    public static f128 _mm_div_ps(f128 a, f128 b) => Sse.Divide(a, b);

    public static f128 _mm_min_ps(f128 a, f128 b) => Sse.Min(a, b);

    public static f128 _mm_max_ps(f128 a, f128 b) => Sse.Max(a, b);

    public static f128 _mm_cmplt_ps(f128 a, f128 b) => Sse.CompareLessThan(a, b);

    public static f128 _mm_cmpgt_ps(f128 a, f128 b) => Sse.CompareGreaterThan(a, b);

    public static f128 _mm_cmpge_ps(f128 a, f128 b) => Sse.CompareGreaterThanOrEqual(a, b);

    public static bool _mm_comilt_ss(f128 a, f128 b) => Sse.CompareScalarOrderedLessThan(a, b);

    public static bool _mm_comigt_ss(f128 a, f128 b) => Sse.CompareScalarOrderedGreaterThan(a, b);

    public static bool _mm_comige_ss(f128 a, f128 b) => Sse.CompareScalarOrderedGreaterThanOrEqual(a, b);

    #endregion

    #region SSE2

    public static i128 _mm_castps_si128(f128 a) => a.AsInt32();

    public static i128 _mm_setzero_si128() => i128.Zero;

    public static i128 _mm_set1_epi16(short a) => Vector128.Create(a).AsInt32();

    public static i128 _mm_set1_epi32(int a) => Vector128.Create(a);

    public static i128 _mm_cvtsi64x_si128(long a) => Vector128.CreateScalar(a).AsInt32();

    public static i128 _mm_load_si128(i128* mem_addr) => Sse2.LoadAlignedVector128((int*)mem_addr);

    public static void _mm_storeu_si128(i128* mem_addr, i128 a) => Sse2.Store((int*)mem_addr, a);

    public static int _mm_movemask_epi8(i128 a) => Sse2.MoveMask(a.AsByte());

    public static int _mm_extract_epi16(i128 a, byte imm8) => Sse2.Extract(a.AsUInt16(), imm8);

    public static int _mm_cvtsi128_si32(i128 a) => Sse2.ConvertToInt32(a);

    public static i128 _mm_cvttps_epi32(f128 a) => Sse2.ConvertToVector128Int32WithTruncation(a);

    public static i128 _mm_sub_epi32(i128 a, i128 b) => Sse2.Subtract(a, b);

    public static i128 _mm_or_si128(i128 a, i128 b) => Sse2.Or(a, b);

    public static i128 _mm_xor_si128(i128 a, i128 b) => Sse2.Xor(a, b);

    public static i128 _mm_slli_epi32(i128 a, byte immediate) => Sse2.ShiftLeftLogical(a, immediate);

    public static i128 _mm_slli_epi64(i128 a, byte imm8) => Sse2.ShiftLeftLogical(a.AsInt64(), imm8).AsInt32();

    public static i128 _mm_srai_epi32(i128 a, byte immediate) => Sse2.ShiftRightArithmetic(a, immediate);

    public static i128 _mm_cmpeq_epi16(i128 a, i128 b) => Sse2.CompareEqual(a.AsInt16(), b.AsInt16()).AsInt32();

    #endregion

    #region SSE4.1

    public static int _mm_extract_epi32(i128 a, byte imm8) => Sse41.Extract(a, imm8);

    public static f128 _mm_blend_ps(f128 a, f128 b, byte imm8) => Sse41.Blend(a, b, imm8);

    public static i128 _mm_packus_epi32(i128 a, i128 b) => Sse41.PackUnsignedSaturate(a, b).AsInt32();

    public static f128 _mm_dp_ps(f128 a, f128 b, byte imm8) => Sse41.DotProduct(a, b, imm8);

    public static f128 _mm_round_to_neg_inf_ps(f128 a) => Sse41.RoundToNegativeInfinity(a);

    public static i128 _mm_min_epu16(i128 a, i128 b) => Sse41.Min(a.AsUInt16(), b.AsUInt16()).AsInt32();

    public static i128 _mm_min_epi32(i128 a, i128 b) => Sse41.Min(a, b);

    public static i128 _mm_max_epu16(i128 a, i128 b) => Sse41.Max(a.AsUInt16(), b.AsUInt16()).AsInt32();

    public static i128 _mm_max_epi32(i128 a, i128 b) => Sse41.Max(a, b);

    public static i128 _mm_minpos_epu16(i128 a) => Sse41.MinHorizontal(a.AsUInt16()).AsInt32();

    #endregion

    #region AVX

    public const FloatComparisonMode _CMP_LT_OQ = FloatComparisonMode.OrderedLessThanNonSignaling;
    public const FloatComparisonMode _CMP_LE_OQ = FloatComparisonMode.OrderedLessThanOrEqualNonSignaling;
    public const FloatComparisonMode _CMP_GT_OQ = FloatComparisonMode.OrderedGreaterThanNonSignaling;

    public static f256 _mm256_castsi256_ps(i256 a) => a.AsSingle();

    public static i256 _mm256_castsi128_si256(i128 a) => a.ToVector256Unsafe();

    public static i256 _mm256_castps_si256(f256 a) => a.AsInt32();

    public static i128 _mm256_castsi256_si128(i256 a) => a.GetLower();

    public static i256 _mm256_setzero_si256() => i256.Zero;

    public static f256 _mm256_setzero_ps() => f256.Zero;

    public static f256 _mm256_set1_ps(float a) => Vector256.Create(a);

    public static i256 _mm256_set1_epi32(int a) => Vector256.Create(a);

    public static f256 _mm256_setr_ps(float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0) =>
        Vector256.Create(e7, e6, e5, e4, e3, e2, e1, e0);

    public static f256 _mm256_broadcast_ss(float* mem_addr) => Avx.BroadcastScalarToVector256(mem_addr);

    public static i256 _mm256_load_si256(i256* mem_addr) => Avx.LoadAlignedVector256((int*)mem_addr);

    public static i256 _mm256_loadu_si256(i256* mem_addr) => Avx.LoadVector256((int*)mem_addr);

    public static void _mm256_store_ps(float* mem_addr, f256 a) => Avx.StoreAligned(mem_addr, a);

    public static void _mm256_storeu_ps(float* mem_addr, f256 a) => Avx.Store(mem_addr, a);

    public static void _mm256_store_si256(i256* mem_addr, i256 a) => Avx.StoreAligned((int*)mem_addr, a);

    public static void _mm256_storeu_si256(i256* mem_addr, i256 a) => Avx.Store((int*)mem_addr, a);

    public static int _mm256_movemask_ps(f256 a) => Avx.MoveMask(a);

    public static f256 _mm256_shuffle_ps(f256 a, f256 b, byte imm8) => Avx.Shuffle(a, b, imm8);

    public static f256 _mm256_cvtepi32_ps(i256 a) => Avx.ConvertToVector256Single(a);

    public static i256 _mm256_cvttps_epi32(f256 a) => Avx.ConvertToVector256Int32WithTruncation(a);

    public static f128 _mm_permute_ps(f128 a, byte imm8) => Avx.Permute(a, imm8);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static f256 _mm256_blendv_ps(f256 a, f256 b, f256 mask) => Avx.BlendVariable(a, b, mask);

    public static f256 _mm256_rcp_ps(f256 a) => Avx.Reciprocal(a);

    public static f256 _mm256_add_ps(f256 a, f256 b) => Avx.Add(a, b);

    public static f256 _mm256_sub_ps(f256 a, f256 b) => Avx.Subtract(a, b);

    public static f256 _mm256_mul_ps(f256 a, f256 b) => Avx.Multiply(a, b);

    public static f256 _mm256_div_ps(f256 a, f256 b) => Avx.Divide(a, b);

    public static f256 _mm256_and_ps(f256 a, f256 b) => Avx.And(a, b);

    public static f256 _mm256_andnot_ps(f256 a, f256 b) => Avx.AndNot(a, b);

    public static f256 _mm256_or_ps(f256 a, f256 b) => Avx.Or(a, b);

    public static f256 _mm256_xor_ps(f256 a, f256 b) => Avx.Xor(a, b);

    public static f256 _mm256_round_to_nearest_int_ps(f256 a) => Avx.RoundToNearestInteger(a);

    public static f256 _mm256_min_ps(f256 a, f256 b) => Avx.Min(a, b);

    public static f256 _mm256_max_ps(f256 a, f256 b) => Avx.Max(a, b);

    public static f256 _mm256_cmp_ps(f256 a, f256 b, FloatComparisonMode imm8) => Avx.Compare(a, b, imm8);

    public static bool _mm_testz_ps(f128 a, f128 b) => Avx.TestZ(a, b);

    public static bool _mm256_testz_si256(i256 a, i256 b) => Avx.TestZ(a, b);

    #endregion

    #region AVX2

    public static i128 _mm256_extracti128_si256(i256 a, byte imm8) => Avx2.ExtractVector128(a, imm8);

    public static f128 _mm_broadcastss_ps(f128 a) => Avx2.BroadcastScalarToVector128(a);

    public static f256 _mm256_broadcastss_ps(f128 a) => Avx2.BroadcastScalarToVector256(a);

    public static i256 _mm256_cvtepu16_epi32(i128 a) => Avx2.ConvertToVector256Int32(a.AsUInt16());

    public static i256 _mm256_stream_load_si256(i256* mem_addr) => Avx2.LoadAlignedVector256NonTemporal((int*)mem_addr);

    public static i256 _mm256_inserti128_si256(i256 a, i128 b, byte imm8) => Avx2.InsertVector128(a, b, imm8);

    public static i256 _mm256_i32gather_epi32(int* base_addr, i256 vindex, byte scale) => Avx2.GatherVector256(base_addr, vindex, scale);

    public static i256 _mm256_unpacklo_epi8(i256 a, i256 b) => Avx2.UnpackLow(a.AsByte(), b.AsByte()).AsInt32();

    public static i256 _mm256_unpacklo_epi32(i256 a, i256 b) => Avx2.UnpackLow(a, b);

    public static i256 _mm256_unpackhi_epi32(i256 a, i256 b) => Avx2.UnpackHigh(a, b);

    public static i256 _mm256_unpacklo_epi64(i256 a, i256 b) => Avx2.UnpackLow(a.AsInt64(), b.AsInt64()).AsInt32();

    public static i256 _mm256_unpackhi_epi64(i256 a, i256 b) => Avx2.UnpackHigh(a.AsInt64(), b.AsInt64()).AsInt32();

    public static i256 _mm256_packus_epi32(i256 a, i256 b) => Avx2.PackUnsignedSaturate(a, b).AsInt32();

    public static i256 _mm256_blendv_epi8(i256 a, i256 b, i256 mask) => Avx2.BlendVariable(a.AsByte(), b.AsByte(), mask.AsByte()).AsInt32();

    public static i256 _mm256_add_epi16(i256 a, i256 b) => Avx2.Add(a.AsInt16(), b.AsInt16()).AsInt32();

    public static i256 _mm256_add_epi32(i256 a, i256 b) => Avx2.Add(a, b);

    public static i256 _mm256_sub_epi32(i256 a, i256 b) => Avx2.Subtract(a, b);

    public static i256 _mm256_mullo_epi16(i256 a, i256 b) => Avx2.MultiplyLow(a.AsInt16(), b.AsInt16()).AsInt32();

    public static i256 _mm256_and_si256(i256 a, i256 b) => Avx2.And(a, b);

    public static i256 _mm256_or_si256(i256 a, i256 b) => Avx2.Or(a, b);

    public static i256 _mm256_slli_epi16(i256 a, byte imm8) => Avx2.ShiftLeftLogical(a.AsInt16(), imm8).AsInt32();

    public static i256 _mm256_slli_epi32(i256 a, byte imm8) => Avx2.ShiftLeftLogical(a, imm8);

    public static i256 _mm256_srli_epi32(i256 a, byte imm8) => Avx2.ShiftRightLogical(a, imm8);

    public static i256 _mm256_srai_epi32(i256 a, byte immediate) => Avx2.ShiftRightArithmetic(a, immediate);

    public static i256 _mm256_avg_epu16(i256 a, i256 b) => Avx2.Average(a.AsUInt16(), b.AsUInt16()).AsInt32();

    public static i256 _mm256_min_epu16(i256 a, i256 b) => Avx2.Min(a.AsUInt16(), b.AsUInt16()).AsInt32();

    public static i256 _mm256_min_epi32(i256 a, i256 b) => Avx2.Min(a, b);

    public static i256 _mm256_max_epu16(i256 a, i256 b) => Avx2.Max(a.AsUInt16(), b.AsUInt16()).AsInt32();

    public static i256 _mm256_max_epi32(i256 a, i256 b) => Avx2.Max(a, b);

    public static i256 _mm256_cmpeq_epi32(i256 a, i256 b) => Avx2.CompareEqual(a, b);

    public static i256 _mm256_cmpgt_epi32(i256 a, i256 b) => Avx2.CompareGreaterThan(a, b);

    #endregion

    #region FMA

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static f128 _mm_fmadd_ps(f128 a, f128 b, f128 c) => Fma.MultiplyAdd(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static f256 _mm256_fmadd_ps(f256 a, f256 b, f256 c) => Fma.MultiplyAdd(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static f128 _mm_fnmadd_ps(f128 a, f128 b, f128 c) => Fma.MultiplyAddNegated(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static f256 _mm256_fnmadd_ps(f256 a, f256 b, f256 c) => Fma.MultiplyAddNegated(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static f256 _mm256_fmsub_ps(f256 a, f256 b, f256 c) => Fma.MultiplySubtract(a, b, c);

    #endregion
}
