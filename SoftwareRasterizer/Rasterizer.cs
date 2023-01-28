/*
#include "Rasterizer.h"

#include "Occluder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
*/

using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

using static VectorMath;
using static Intrinsics;

public unsafe partial class Rasterizer
{
    public interface IPossiblyNearClipped
    {
        static abstract bool PossiblyNearClipped { get; }
    }

    public readonly struct NearClipped : IPossiblyNearClipped
    {
        public static bool PossiblyNearClipped => true;
    }

    public readonly struct NotNearClipped : IPossiblyNearClipped
    {
        public static bool PossiblyNearClipped => false;
    }

    private const float floatCompressionBias = 2.5237386e-29f; // 0xFFFF << 12 reinterpreted as float
    private const float minEdgeOffset = -0.45f;
    private const float maxInvW = 18446742974197923840; // MathF.Sqrt(float.MaxValue)

    private const int OFFSET_QUANTIZATION_BITS = 6;
    private const int OFFSET_QUANTIZATION_FACTOR = 1 << OFFSET_QUANTIZATION_BITS;

    private const int SLOPE_QUANTIZATION_BITS = 6;
    private const int SLOPE_QUANTIZATION_FACTOR = 1 << SLOPE_QUANTIZATION_BITS;

    private float* m_modelViewProjection;
    private float* m_modelViewProjectionRaw;

    private ulong* m_precomputedRasterTables;
    private Vector128<int>* m_depthBuffer;

    private uint m_hiZ_Size;
    private ushort* m_hiZ;

    private uint m_width;
    private uint m_height;
    private uint m_blocksX;
    private uint m_blocksY;

    public Rasterizer(uint width, uint height)
    {
        m_width = (width);
        m_height = (height);
        m_blocksX = (width / 8);
        m_blocksY = (height / 8);

        Debug.Assert(width % 8 == 0 && height % 8 == 0);

        m_modelViewProjection = (float*)NativeMemory.AlignedAlloc(16 * sizeof(float), (uint)sizeof(Vector128<float>));
        m_modelViewProjectionRaw = (float*)NativeMemory.AlignedAlloc(16 * sizeof(float), (uint)sizeof(Vector128<float>));

        m_depthBuffer = (Vector128<int>*)NativeMemory.AlignedAlloc(width * height / 8 * (uint)sizeof(Vector128<float>), (uint)sizeof(Vector256<int>));

        m_hiZ_Size = m_blocksX * m_blocksY + 8; // Add some extra padding to support out-of-bounds reads
        uint hiZ_Bytes = m_hiZ_Size * sizeof(ushort);
        m_hiZ = (ushort*)NativeMemory.AlignedAlloc(hiZ_Bytes, (uint)sizeof(Vector128<int>));
        Unsafe.InitBlockUnaligned(m_hiZ, 0, hiZ_Bytes);

        m_precomputedRasterTables = precomputeRasterizationTable();
    }

    public void setModelViewProjection(float* matrix)
    {
        Vector128<float> mat0 = _mm_loadu_ps(matrix + 0);
        Vector128<float> mat1 = _mm_loadu_ps(matrix + 4);
        Vector128<float> mat2 = _mm_loadu_ps(matrix + 8);
        Vector128<float> mat3 = _mm_loadu_ps(matrix + 12);

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Store rows
        _mm_storeu_ps(m_modelViewProjectionRaw + 0, mat0);
        _mm_storeu_ps(m_modelViewProjectionRaw + 4, mat1);
        _mm_storeu_ps(m_modelViewProjectionRaw + 8, mat2);
        _mm_storeu_ps(m_modelViewProjectionRaw + 12, mat3);

        // Bake viewport transform into matrix and 6shift by half a block
        mat0 = _mm_mul_ps(_mm_add_ps(mat0, mat3), _mm_set1_ps(m_width * 0.5f - 4.0f));
        mat1 = _mm_mul_ps(_mm_add_ps(mat1, mat3), _mm_set1_ps(m_height * 0.5f - 4.0f));

        // Map depth from [-1, 1] to [bias, 0]
        mat2 = _mm_mul_ps(_mm_sub_ps(mat3, mat2), _mm_set1_ps(0.5f * floatCompressionBias));

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Store prebaked cols
        _mm_storeu_ps(m_modelViewProjection + 0, mat0);
        _mm_storeu_ps(m_modelViewProjection + 4, mat1);
        _mm_storeu_ps(m_modelViewProjection + 8, mat2);
        _mm_storeu_ps(m_modelViewProjection + 12, mat3);
    }

    public void clear()
    {
        // Mark blocks as cleared by setting Hi Z to 1 (one unit separated from far plane). 
        // This value is extremely unlikely to occur during normal rendering, so we don't
        // need to guard against a HiZ of 1 occuring naturally. This is different from a value of 0, 
        // which will occur every time a block is partially covered for the first time.
        Vector128<int> clearValue = _mm_set1_epi16(1);
        uint count = m_hiZ_Size / 8;
        Vector128<int>* pHiZ = (Vector128<int>*)m_hiZ;
        for (uint offset = 0; offset < count; ++offset)
        {
            _mm_storeu_si128(pHiZ, clearValue);
            pHiZ++;
        }
    }

    public bool queryVisibility(Vector128<float> boundsMin, Vector128<float> boundsMax, out bool needsClipping)
    {
        // Frustum cull
        Vector128<float> extents = _mm_sub_ps(boundsMax, boundsMin);
        Vector128<float> center = _mm_add_ps(boundsMax, boundsMin); // Bounding box center times 2 - but since W = 2, the plane equations work out correctly
        Vector128<float> minusZero = _mm_set1_ps(-0.0f);

        Vector128<float> row0 = _mm_loadu_ps(m_modelViewProjectionRaw + 0);
        Vector128<float> row1 = _mm_loadu_ps(m_modelViewProjectionRaw + 4);
        Vector128<float> row2 = _mm_loadu_ps(m_modelViewProjectionRaw + 8);
        Vector128<float> row3 = _mm_loadu_ps(m_modelViewProjectionRaw + 12);

        // Compute distance from each frustum plane
        Vector128<float> plane0 = _mm_add_ps(row3, row0);
        Vector128<float> offset0 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane0, minusZero)));
        Vector128<float> dist0 = _mm_dp_ps(plane0, offset0, 0xff);

        Vector128<float> plane1 = _mm_sub_ps(row3, row0);
        Vector128<float> offset1 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane1, minusZero)));
        Vector128<float> dist1 = _mm_dp_ps(plane1, offset1, 0xff);

        Vector128<float> plane2 = _mm_add_ps(row3, row1);
        Vector128<float> offset2 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane2, minusZero)));
        Vector128<float> dist2 = _mm_dp_ps(plane2, offset2, 0xff);

        Vector128<float> plane3 = _mm_sub_ps(row3, row1);
        Vector128<float> offset3 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane3, minusZero)));
        Vector128<float> dist3 = _mm_dp_ps(plane3, offset3, 0xff);

        Vector128<float> plane4 = _mm_add_ps(row3, row2);
        Vector128<float> offset4 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane4, minusZero)));
        Vector128<float> dist4 = _mm_dp_ps(plane4, offset4, 0xff);

        Vector128<float> plane5 = _mm_sub_ps(row3, row2);
        Vector128<float> offset5 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane5, minusZero)));
        Vector128<float> dist5 = _mm_dp_ps(plane5, offset5, 0xff);

        // Combine plane distance signs
        Vector128<float> combined = _mm_or_ps(_mm_or_ps(_mm_or_ps(dist0, dist1), _mm_or_ps(dist2, dist3)), _mm_or_ps(dist4, dist5));

        // Can't use _mm_testz_ps or _mm_comile_ss here because the OR's above created garbage in the non-sign bits
        if (_mm_movemask_ps(combined) != 0)
        {
            needsClipping = false;
            return false;
        }

        // Load prebaked projection matrix
        Vector128<float> col0 = _mm_loadu_ps(m_modelViewProjection + 0);
        Vector128<float> col1 = _mm_loadu_ps(m_modelViewProjection + 4);
        Vector128<float> col2 = _mm_loadu_ps(m_modelViewProjection + 8);
        Vector128<float> col3 = _mm_loadu_ps(m_modelViewProjection + 12);

        // Transform edges
        Vector128<float> egde0 = _mm_mul_ps(col0, _mm_broadcastss_ps(extents));
        Vector128<float> egde1 = _mm_mul_ps(col1, _mm_permute_ps(extents, 0b01_01_01_01));
        Vector128<float> egde2 = _mm_mul_ps(col2, _mm_permute_ps(extents, 0b10_10_10_10));

        Vector128<float> corners0;
        Vector128<float> corners1;
        Vector128<float> corners2;
        Vector128<float> corners3;
        Vector128<float> corners4;
        Vector128<float> corners5;
        Vector128<float> corners6;
        Vector128<float> corners7;

        // Transform first corner
        corners0 =
          _mm_fmadd_ps(col0, _mm_broadcastss_ps(boundsMin),
            _mm_fmadd_ps(col1, _mm_permute_ps(boundsMin, 0b01_01_01_01),
              _mm_fmadd_ps(col2, _mm_permute_ps(boundsMin, 0b10_10_10_10),
                col3)));

        // Transform remaining corners by adding edge vectors
        corners1 = _mm_add_ps(corners0, egde0);
        corners2 = _mm_add_ps(corners0, egde1);
        corners4 = _mm_add_ps(corners0, egde2);

        corners3 = _mm_add_ps(corners1, egde1);
        corners5 = _mm_add_ps(corners4, egde0);
        corners6 = _mm_add_ps(corners2, egde2);

        corners7 = _mm_add_ps(corners6, egde0);

        // Transpose into SoA
        _MM_TRANSPOSE4_PS(ref corners0, ref corners1, ref corners2, ref corners3);
        _MM_TRANSPOSE4_PS(ref corners4, ref corners5, ref corners6, ref corners7);

        // Even if all bounding box corners have W > 0 here, we may end up with some vertices with W < 0 to due floating point differences; so test with some epsilon if any W < 0.
        Vector128<float> maxExtent = _mm_max_ps(extents, _mm_permute_ps(extents, 0b01_00_11_10));
        maxExtent = _mm_max_ps(maxExtent, _mm_permute_ps(maxExtent, 0b10_11_00_01));
        Vector128<float> nearPlaneEpsilon = _mm_mul_ps(maxExtent, _mm_set1_ps(0.001f));
        Vector128<float> closeToNearPlane = _mm_or_ps(_mm_cmplt_ps(corners3, nearPlaneEpsilon), _mm_cmplt_ps(corners7, nearPlaneEpsilon));
        if (!_mm_testz_ps(closeToNearPlane, closeToNearPlane))
        {
            needsClipping = true;
            return true;
        }

        needsClipping = false;

        // Perspective division
        corners3 = _mm_rcp_ps(corners3);
        corners0 = _mm_mul_ps(corners0, corners3);
        corners1 = _mm_mul_ps(corners1, corners3);
        corners2 = _mm_mul_ps(corners2, corners3);

        corners7 = _mm_rcp_ps(corners7);
        corners4 = _mm_mul_ps(corners4, corners7);
        corners5 = _mm_mul_ps(corners5, corners7);
        corners6 = _mm_mul_ps(corners6, corners7);

        // Vertical mins and maxes
        Vector128<float> minsX = _mm_min_ps(corners0, corners4);
        Vector128<float> maxsX = _mm_max_ps(corners0, corners4);

        Vector128<float> minsY = _mm_min_ps(corners1, corners5);
        Vector128<float> maxsY = _mm_max_ps(corners1, corners5);

        // Horizontal reduction, step 1
        Vector128<float> minsXY = _mm_min_ps(_mm_unpacklo_ps(minsX, minsY), _mm_unpackhi_ps(minsX, minsY));
        Vector128<float> maxsXY = _mm_max_ps(_mm_unpacklo_ps(maxsX, maxsY), _mm_unpackhi_ps(maxsX, maxsY));

        // Clamp bounds
        minsXY = _mm_max_ps(minsXY, _mm_setzero_ps());
        maxsXY = _mm_min_ps(maxsXY, _mm_setr_ps(m_width - 1f, m_height - 1f, m_width - 1f, m_height - 1f));

        // Negate maxes so we can round in the same direction
        maxsXY = _mm_xor_ps(maxsXY, minusZero);

        // Horizontal reduction, step 2
        Vector128<float> boundsF = _mm_min_ps(_mm_unpacklo_ps(minsXY, maxsXY), _mm_unpackhi_ps(minsXY, maxsXY));

        // Round towards -infinity and convert to int
        Vector128<int> boundsI = _mm_cvttps_epi32(_mm_round_to_neg_inf_ps(boundsF));

        // Store as scalars
        int* bounds = stackalloc int[4];
        _mm_storeu_si128((Vector128<int>*)bounds, boundsI);

        uint minX = (uint)bounds[0];
        uint maxX = (uint)bounds[1];
        uint minY = (uint)bounds[2];
        uint maxY = (uint)bounds[3];

        // Revert the sign change we did for the maxes
        maxX = (uint)(-(int)maxX);
        maxY = (uint)(-(int)maxY);

        // No intersection between quad and screen area
        if (minX >= maxX || minY >= maxY)
        {
            return false;
        }

        Vector128<int> depth = packDepthPremultiplied(corners2, corners6);

        ushort maxZ = (ushort)(0xFFFF ^ _mm_extract_epi16(_mm_minpos_epu16(_mm_xor_si128(depth, _mm_set1_epi16(-1))), 0));

        if (!query2D(minX, maxX, minY, maxY, maxZ))
        {
            return false;
        }

        return true;
    }

    public bool query2D(uint minX, uint maxX, uint minY, uint maxY, uint maxZ)
    {
        ushort* pHiZBuffer = m_hiZ;
        Vector128<int>* pDepthBuffer = m_depthBuffer;

        uint blockMinX = minX / 8;
        uint blockMaxX = maxX / 8;

        uint blockMinY = minY / 8;
        uint blockMaxY = maxY / 8;

        Vector128<int> maxZV = _mm_set1_epi16((short)(maxZ));

        // Pretest against Hi-Z
        for (uint blockY = blockMinY; blockY <= blockMaxY; ++blockY)
        {
            uint startY = (uint)Math.Max((int)(minY - 8 * blockY), 0);
            uint endY = (uint)Math.Min((int)(maxY - 8 * blockY), 7);

            ushort* pHiZ = pHiZBuffer + (blockY * m_blocksX + blockMinX);
            Vector128<int>* pBlockDepth = pDepthBuffer + 8 * (blockY * m_blocksX + blockMinX) + startY;

            bool interiorLine = (startY == 0) && (endY == 7);

            for (uint blockX = blockMinX; blockX <= blockMaxX; ++blockX, ++pHiZ, pBlockDepth += 8)
            {
                // Skip this block if it fully occludes the query box
                if (maxZ <= *pHiZ)
                {
                    continue;
                }

                uint startX = (uint)Math.Max((int)(minX - blockX * 8), 0);
                uint endX = (uint)Math.Min((int)(maxX - blockX * 8), 7);

                bool interiorBlock = interiorLine && (startX == 0) && (endX == 7);

                // No pixels are masked, so there exists one where maxZ > pixelZ, and the query region is visible
                if (interiorBlock)
                {
                    return true;
                }

                ushort rowSelector = (ushort)((0xFFFFu << (int)(2 * startX)) & (0xFFFFu >> (int)(2 * (7 - endX))));

                Vector128<int>* pRowDepth = pBlockDepth;

                for (uint y = startY; y <= endY; ++y)
                {
                    Vector128<int> rowDepth = *pRowDepth++;

                    Vector128<int> notVisible = _mm_cmpeq_epi16(_mm_min_epu16(rowDepth, maxZV), maxZV);

                    uint visiblePixelMask = (uint)(~_mm_movemask_epi8(notVisible));

                    if ((rowSelector & visiblePixelMask) != 0)
                    {
                        return true;
                    }
                }
            }
        }

        // Not visible
        return false;
    }

    public void readBackDepth(byte* target)
    {
        const float bias = 3.9623753e+28f; // 1.0f / floatCompressionBias

        float* linDepthA = stackalloc float[16];

        for (uint blockY = 0; blockY < m_blocksY; ++blockY)
        {
            for (uint blockX = 0; blockX < m_blocksX; ++blockX)
            {
                if (m_hiZ[blockY * m_blocksX + blockX] == 1)
                {
                    for (uint y = 0; y < 8; ++y)
                    {
                        byte* dest = target + 4 * (8 * blockX + m_width * (8 * blockY + y));
                        Unsafe.InitBlockUnaligned(dest, 0, 32);
                    }
                    continue;
                }

                Vector128<int>* source = &m_depthBuffer[8 * (blockY * m_blocksX + blockX)];
                for (uint y = 0; y < 8; ++y)
                {
                    byte* dest = target + 4 * (8 * blockX + m_width * (8 * blockY + y));

                    Vector128<int> depthI = _mm_load_si128(source++);

                    Vector256<int> depthI256 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(depthI), 12);
                    Vector256<float> depth = _mm256_mul_ps(_mm256_castsi256_ps(depthI256), _mm256_set1_ps(bias));

                    Vector256<float> linDepth = _mm256_div_ps(_mm256_set1_ps(2 * 0.25f), _mm256_sub_ps(_mm256_set1_ps(0.25f + 1000.0f), _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), depth), _mm256_set1_ps(1000.0f - 0.25f))));

                    _mm256_storeu_ps(linDepthA, linDepth);

                    for (uint x = 0; x < 8; ++x)
                    {
                        float l = linDepthA[x];
                        uint d = (uint)(100 * 256 * l);
                        byte v0 = (byte)(d / 100);
                        byte v1 = (byte)(d % 256);

                        dest[4 * x + 0] = v0;
                        dest[4 * x + 1] = v1;
                        dest[4 * x + 2] = 0;
                        dest[4 * x + 3] = 255;
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float decompressFloat(ushort depth)
    {
        const float bias = 3.9623753e+28f; // 1.0f / floatCompressionBias

        uint u = (uint)depth << 12;
        return BitConverter.UInt32BitsToSingle(u) * bias;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void transpose256(Vector256<float> A, Vector256<float> B, Vector256<float> C, Vector256<float> D, Vector128<float>* @out)
    {
        Vector256<float> _Tmp0 = _mm256_shuffle_ps(A, B, 0x44);
        Vector256<float> _Tmp2 = _mm256_shuffle_ps(A, B, 0xEE);
        Vector256<float> _Tmp1 = _mm256_shuffle_ps(C, D, 0x44);
        Vector256<float> _Tmp3 = _mm256_shuffle_ps(C, D, 0xEE);

        Vector256<float> tA = _mm256_shuffle_ps(_Tmp0, _Tmp1, 0x88);
        Vector256<float> tB = _mm256_shuffle_ps(_Tmp0, _Tmp1, 0xDD);
        Vector256<float> tC = _mm256_shuffle_ps(_Tmp2, _Tmp3, 0x88);
        Vector256<float> tD = _mm256_shuffle_ps(_Tmp2, _Tmp3, 0xDD);

        _mm256_storeu_ps((float*)(@out + 0), tA);
        _mm256_storeu_ps((float*)(@out + 2), tB);
        _mm256_storeu_ps((float*)(@out + 4), tC);
        _mm256_storeu_ps((float*)(@out + 6), tD);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void transpose256i(Vector256<int> A, Vector256<int> B, Vector256<int> C, Vector256<int> D, Vector128<int>* @out)
    {
        Vector256<int> _Tmp0 = _mm256_unpacklo_epi32(A, B);
        Vector256<int> _Tmp1 = _mm256_unpacklo_epi32(C, D);
        Vector256<int> _Tmp2 = _mm256_unpackhi_epi32(A, B);
        Vector256<int> _Tmp3 = _mm256_unpackhi_epi32(C, D);

        Vector256<int> tA = _mm256_unpacklo_epi64(_Tmp0, _Tmp1);
        Vector256<int> tB = _mm256_unpackhi_epi64(_Tmp0, _Tmp1);
        Vector256<int> tC = _mm256_unpacklo_epi64(_Tmp2, _Tmp3);
        Vector256<int> tD = _mm256_unpackhi_epi64(_Tmp2, _Tmp3);

        _mm256_storeu_si256((Vector256<int>*)(@out + 0), tA);
        _mm256_storeu_si256((Vector256<int>*)(@out + 2), tB);
        _mm256_storeu_si256((Vector256<int>*)(@out + 4), tC);
        _mm256_storeu_si256((Vector256<int>*)(@out + 6), tD);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void normalizeEdge<T>(ref Vector256<float> nx, ref Vector256<float> ny, Vector256<float> edgeFlipMask)
        where T : IPossiblyNearClipped
    {
        Vector256<float> minusZero = _mm256_set1_ps(-0.0f);
        Vector256<float> invLen = _mm256_rcp_ps(_mm256_add_ps(_mm256_andnot_ps(minusZero, nx), _mm256_andnot_ps(minusZero, ny)));

        const float maxOffset = -minEdgeOffset;
        Vector256<float> mul = _mm256_set1_ps((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        if (T.PossiblyNearClipped)
        {
            mul = _mm256_xor_ps(mul, edgeFlipMask);
        }

        invLen = _mm256_mul_ps(mul, invLen);
        nx = _mm256_mul_ps(nx, invLen);
        ny = _mm256_mul_ps(ny, invLen);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> quantizeSlopeLookup(Vector128<float> nx, Vector128<float> ny)
    {
        Vector128<int> yNeg = _mm_castps_si128(_mm_cmplt_ps(ny, _mm_setzero_ps()));

        // Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
        const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f;
        const float add = mul + 0.5f;

        Vector128<int> quantizedSlope = _mm_cvttps_epi32(_mm_fmadd_ps(nx, _mm_set1_ps(mul), _mm_set1_ps(add)));
        return _mm_slli_epi32(_mm_sub_epi32(_mm_slli_epi32(quantizedSlope, 1), yNeg), OFFSET_QUANTIZATION_BITS);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> quantizeSlopeLookup(Vector256<float> nx, Vector256<float> ny)
    {
        Vector256<int> yNeg = _mm256_castps_si256(_mm256_cmp_ps(ny, _mm256_setzero_ps(), _CMP_LE_OQ));

        // Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
        const float maxOffset = -minEdgeOffset;
        const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f / ((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        const float add = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f + 0.5f;

        Vector256<int> quantizedSlope = _mm256_cvttps_epi32(_mm256_fmadd_ps(nx, _mm256_set1_ps(mul), _mm256_set1_ps(add)));
        return _mm256_slli_epi32(_mm256_sub_epi32(_mm256_slli_epi32(quantizedSlope, 1), yNeg), OFFSET_QUANTIZATION_BITS);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint quantizeOffsetLookup(float offset)
    {
        const float maxOffset = -minEdgeOffset;

        // Remap [minOffset, maxOffset] to [0, OFFSET_QUANTIZATION]
        const float mul = (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset);
        const float add = 0.5f - minEdgeOffset * mul;

        float lookup = offset * mul + add;
        return (uint)Math.Min(Math.Max((int)lookup, 0), OFFSET_QUANTIZATION_FACTOR - 1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> packDepthPremultiplied(Vector128<float> depthA, Vector128<float> depthB)
    {
        return _mm_packus_epi32(_mm_srai_epi32(_mm_castps_si128(depthA), 12), _mm_srai_epi32(_mm_castps_si128(depthB), 12));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> packDepthPremultiplied(Vector256<float> depth)
    {
        Vector256<int> x = _mm256_srai_epi32(_mm256_castps_si256(depth), 12);
        return _mm_packus_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> packDepthPremultiplied(Vector256<float> depthA, Vector256<float> depthB)
    {
        Vector256<int> x1 = _mm256_srai_epi32(_mm256_castps_si256(depthA), 12);
        Vector256<int> x2 = _mm256_srai_epi32(_mm256_castps_si256(depthB), 12);

        return _mm256_packus_epi32(x1, x2);
    }

    private static ulong transposeMask(ulong mask)
    {
#if false
        ulong maskA = _pdep_u64(_pext_u64(mask, 0x5555555555555555ull), 0xF0F0F0F0F0F0F0F0ull);
        ulong maskB = _pdep_u64(_pext_u64(mask, 0xAAAAAAAAAAAAAAAAull), 0x0F0F0F0F0F0F0F0Full);
#else
        ulong maskA = 0;
        ulong maskB = 0;
        for (uint group = 0; group < 8; ++group)
        {
            for (uint bit = 0; bit < 4; ++bit)
            {
                maskA |= ((mask >> (int)(8 * group + 2 * bit + 0)) & 1) << (int)(4 + group * 8 + bit);
                maskB |= ((mask >> (int)(8 * group + 2 * bit + 1)) & 1) << (int)(0 + group * 8 + bit);
            }
        }
#endif
        return maskA | maskB;
    }

    private static ulong* precomputeRasterizationTable()
    {
        const uint angularResolution = 2000;
        const uint offsetResolution = 2000;

        uint precomputedRasterTablesByteCount = OFFSET_QUANTIZATION_FACTOR * SLOPE_QUANTIZATION_FACTOR * sizeof(ulong);
        ulong* precomputedRasterTables = (ulong*)NativeMemory.AlignedAlloc(
            byteCount: precomputedRasterTablesByteCount,
            alignment: sizeof(ulong));

        Unsafe.InitBlockUnaligned(precomputedRasterTables, 0, precomputedRasterTablesByteCount);

        for (uint i = 0; i < angularResolution; ++i)
        {
            float angle = -0.1f + 6.4f * i / (angularResolution - 1);

            (float ny, float nx) = MathF.SinCos(angle);
            float l = 1.0f / (MathF.Abs(nx) + MathF.Abs(ny));

            nx *= l;
            ny *= l;

            uint slopeLookup = (uint)_mm_extract_epi32(quantizeSlopeLookup(_mm_set1_ps(nx), _mm_set1_ps(ny)), 0);

            for (uint j = 0; j < offsetResolution; ++j)
            {
                float offset = -0.6f + 1.2f * j / (angularResolution - 1);

                uint offsetLookup = quantizeOffsetLookup(offset);

                uint lookup = slopeLookup | offsetLookup;

                ulong block = 0;

                for (int x = 0; x < 8; ++x)
                {
                    for (int y = 0; y < 8; ++y)
                    {
                        float edgeDistance = offset + (x - 3.5f) / 8.0f * nx + (y - 3.5f) / 8.0f * ny;
                        if (edgeDistance <= 0.0f)
                        {
                            int bitIndex = 8 * x + y;
                            block |= 1ul << bitIndex;
                        }
                    }
                }

                precomputedRasterTables[lookup] |= transposeMask(block);
            }

            // For each slope, the first block should be all ones, the last all zeroes
            Debug.Assert(precomputedRasterTables[slopeLookup] == 0xffff_ffff_ffff_ffff);
            Debug.Assert(precomputedRasterTables[slopeLookup + OFFSET_QUANTIZATION_FACTOR - 1] == 0);
        }

        return precomputedRasterTables;
    }

    public void rasterize<T>(Occluder occluder)
        where T : IPossiblyNearClipped
    {
        Vector256<int>* vertexData = occluder.m_vertexData;
        uint packetCount = occluder.m_packetCount;

        Vector256<int> maskY = _mm256_set1_epi32(2047 << 10);
        Vector256<int> maskZ = _mm256_set1_epi32(1023);

        // Note that unaligned loads do not have a latency penalty on CPUs with SSE4 support
        Vector128<float> mat0 = _mm_loadu_ps(m_modelViewProjection + 0);
        Vector128<float> mat1 = _mm_loadu_ps(m_modelViewProjection + 4);
        Vector128<float> mat2 = _mm_loadu_ps(m_modelViewProjection + 8);
        Vector128<float> mat3 = _mm_loadu_ps(m_modelViewProjection + 12);

        Vector128<float> boundsMin = occluder.m_refMin;
        Vector128<float> boundsExtents = _mm_sub_ps(occluder.m_refMax, boundsMin);

        // Bake integer => bounding box transform into matrix
        mat3 =
          _mm_fmadd_ps(mat0, _mm_broadcastss_ps(boundsMin),
            _mm_fmadd_ps(mat1, _mm_permute_ps(boundsMin, 0b01_01_01_01),
              _mm_fmadd_ps(mat2, _mm_permute_ps(boundsMin, 0b10_10_10_10),
                mat3)));

        mat0 = _mm_mul_ps(mat0, _mm_mul_ps(_mm_broadcastss_ps(boundsExtents), _mm_set1_ps(1.0f / (2047ul << 21))));
        mat1 = _mm_mul_ps(mat1, _mm_mul_ps(_mm_permute_ps(boundsExtents, 0b01_01_01_01), _mm_set1_ps(1.0f / (2047 << 10))));
        mat2 = _mm_mul_ps(mat2, _mm_mul_ps(_mm_permute_ps(boundsExtents, 0b10_10_10_10), _mm_set1_ps(1.0f / 1023)));

        // Bias X coordinate back into positive range
        mat3 = _mm_fmadd_ps(mat0, _mm_set1_ps(1024ul << 21), mat3);

        // Skew projection to correct bleeding of Y and Z into X due to lack of masking
        mat1 = _mm_sub_ps(mat1, mat0);
        mat2 = _mm_sub_ps(mat2, mat0);

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Due to linear relationship between Z and W, it's cheaper to compute Z from W later in the pipeline than using the full projection matrix up front
        float c0, c1;
        {
            Vector128<float> Za = _mm_permute_ps(mat2, 0b11_11_11_11);
            Vector128<float> Zb = _mm_dp_ps(mat2, _mm_setr_ps(1 << 21, 1 << 10, 1, 1), 0xFF);

            Vector128<float> Wa = _mm_permute_ps(mat3, 0b11_11_11_11);
            Vector128<float> Wb = _mm_dp_ps(mat3, _mm_setr_ps(1 << 21, 1 << 10, 1, 1), 0xFF);

            _mm_store_ss(&c0, _mm_div_ps(_mm_sub_ps(Za, Zb), _mm_sub_ps(Wa, Wb)));
            _mm_store_ss(&c1, _mm_fnmadd_ps(_mm_div_ps(_mm_sub_ps(Za, Zb), _mm_sub_ps(Wa, Wb)), Wa, Za));
        }

        uint* primModes = stackalloc uint[8];

        uint* firstBlocks = stackalloc uint[8];
        uint* rangesX = stackalloc uint[8];
        uint* rangesY = stackalloc uint[8];
        ushort* depthBounds = stackalloc ushort[8];

        Vector128<float>* depthPlane = stackalloc Vector128<float>[8];
        Vector128<float>* edgeNormalsX = stackalloc Vector128<float>[8];
        Vector128<float>* edgeNormalsY = stackalloc Vector128<float>[8];
        Vector128<float>* edgeOffsets = stackalloc Vector128<float>[8];
        Vector128<int>* slopeLookups = stackalloc Vector128<int>[8];

        for (uint packetIdx = 0; packetIdx < packetCount; packetIdx += 4)
        {
            // Load data - only needed once per frame, so use streaming load
            Vector256<int> I0 = _mm256_stream_load_si256(vertexData + packetIdx + 0);
            Vector256<int> I1 = _mm256_stream_load_si256(vertexData + packetIdx + 1);
            Vector256<int> I2 = _mm256_stream_load_si256(vertexData + packetIdx + 2);
            Vector256<int> I3 = _mm256_stream_load_si256(vertexData + packetIdx + 3);

            // Vertex transformation - first W, then X & Y after camera plane culling, then Z after backface culling
            Vector256<float> Xf0 = _mm256_cvtepi32_ps(I0);
            Vector256<float> Xf1 = _mm256_cvtepi32_ps(I1);
            Vector256<float> Xf2 = _mm256_cvtepi32_ps(I2);
            Vector256<float> Xf3 = _mm256_cvtepi32_ps(I3);

            Vector256<float> Yf0 = _mm256_cvtepi32_ps(_mm256_and_si256(I0, maskY));
            Vector256<float> Yf1 = _mm256_cvtepi32_ps(_mm256_and_si256(I1, maskY));
            Vector256<float> Yf2 = _mm256_cvtepi32_ps(_mm256_and_si256(I2, maskY));
            Vector256<float> Yf3 = _mm256_cvtepi32_ps(_mm256_and_si256(I3, maskY));

            Vector256<float> Zf0 = _mm256_cvtepi32_ps(_mm256_and_si256(I0, maskZ));
            Vector256<float> Zf1 = _mm256_cvtepi32_ps(_mm256_and_si256(I1, maskZ));
            Vector256<float> Zf2 = _mm256_cvtepi32_ps(_mm256_and_si256(I2, maskZ));
            Vector256<float> Zf3 = _mm256_cvtepi32_ps(_mm256_and_si256(I3, maskZ));

            Vector256<float> mat00 = _mm256_broadcast_ss((float*)(&mat0) + 0);
            Vector256<float> mat01 = _mm256_broadcast_ss((float*)(&mat0) + 1);
            Vector256<float> mat02 = _mm256_broadcast_ss((float*)(&mat0) + 2);
            Vector256<float> mat03 = _mm256_broadcast_ss((float*)(&mat0) + 3);

            Vector256<float> X0 = _mm256_fmadd_ps(Xf0, mat00, _mm256_fmadd_ps(Yf0, mat01, _mm256_fmadd_ps(Zf0, mat02, mat03)));
            Vector256<float> X1 = _mm256_fmadd_ps(Xf1, mat00, _mm256_fmadd_ps(Yf1, mat01, _mm256_fmadd_ps(Zf1, mat02, mat03)));
            Vector256<float> X2 = _mm256_fmadd_ps(Xf2, mat00, _mm256_fmadd_ps(Yf2, mat01, _mm256_fmadd_ps(Zf2, mat02, mat03)));
            Vector256<float> X3 = _mm256_fmadd_ps(Xf3, mat00, _mm256_fmadd_ps(Yf3, mat01, _mm256_fmadd_ps(Zf3, mat02, mat03)));

            Vector256<float> mat10 = _mm256_broadcast_ss((float*)(&mat1) + 0);
            Vector256<float> mat11 = _mm256_broadcast_ss((float*)(&mat1) + 1);
            Vector256<float> mat12 = _mm256_broadcast_ss((float*)(&mat1) + 2);
            Vector256<float> mat13 = _mm256_broadcast_ss((float*)(&mat1) + 3);

            Vector256<float> Y0 = _mm256_fmadd_ps(Xf0, mat10, _mm256_fmadd_ps(Yf0, mat11, _mm256_fmadd_ps(Zf0, mat12, mat13)));
            Vector256<float> Y1 = _mm256_fmadd_ps(Xf1, mat10, _mm256_fmadd_ps(Yf1, mat11, _mm256_fmadd_ps(Zf1, mat12, mat13)));
            Vector256<float> Y2 = _mm256_fmadd_ps(Xf2, mat10, _mm256_fmadd_ps(Yf2, mat11, _mm256_fmadd_ps(Zf2, mat12, mat13)));
            Vector256<float> Y3 = _mm256_fmadd_ps(Xf3, mat10, _mm256_fmadd_ps(Yf3, mat11, _mm256_fmadd_ps(Zf3, mat12, mat13)));

            Vector256<float> mat30 = _mm256_broadcast_ss((float*)(&mat3) + 0);
            Vector256<float> mat31 = _mm256_broadcast_ss((float*)(&mat3) + 1);
            Vector256<float> mat32 = _mm256_broadcast_ss((float*)(&mat3) + 2);
            Vector256<float> mat33 = _mm256_broadcast_ss((float*)(&mat3) + 3);

            Vector256<float> W0 = _mm256_fmadd_ps(Xf0, mat30, _mm256_fmadd_ps(Yf0, mat31, _mm256_fmadd_ps(Zf0, mat32, mat33)));
            Vector256<float> W1 = _mm256_fmadd_ps(Xf1, mat30, _mm256_fmadd_ps(Yf1, mat31, _mm256_fmadd_ps(Zf1, mat32, mat33)));
            Vector256<float> W2 = _mm256_fmadd_ps(Xf2, mat30, _mm256_fmadd_ps(Yf2, mat31, _mm256_fmadd_ps(Zf2, mat32, mat33)));
            Vector256<float> W3 = _mm256_fmadd_ps(Xf3, mat30, _mm256_fmadd_ps(Yf3, mat31, _mm256_fmadd_ps(Zf3, mat32, mat33)));

            Vector256<float> invW0, invW1, invW2, invW3;
            // Clamp W and invert
            if (T.PossiblyNearClipped)
            {
                Vector256<float> lowerBound = _mm256_set1_ps(-maxInvW);
                Vector256<float> upperBound = _mm256_set1_ps(+maxInvW);
                invW0 = _mm256_min_ps(upperBound, _mm256_max_ps(lowerBound, _mm256_rcp_ps(W0)));
                invW1 = _mm256_min_ps(upperBound, _mm256_max_ps(lowerBound, _mm256_rcp_ps(W1)));
                invW2 = _mm256_min_ps(upperBound, _mm256_max_ps(lowerBound, _mm256_rcp_ps(W2)));
                invW3 = _mm256_min_ps(upperBound, _mm256_max_ps(lowerBound, _mm256_rcp_ps(W3)));
            }
            else
            {
                invW0 = _mm256_rcp_ps(W0);
                invW1 = _mm256_rcp_ps(W1);
                invW2 = _mm256_rcp_ps(W2);
                invW3 = _mm256_rcp_ps(W3);
            }

            // Round to integer coordinates to improve culling of zero-area triangles
            Vector256<float> x0 = _mm256_mul_ps(_mm256_round_to_nearest_int_ps(_mm256_mul_ps(X0, invW0)), _mm256_set1_ps(0.125f));
            Vector256<float> x1 = _mm256_mul_ps(_mm256_round_to_nearest_int_ps(_mm256_mul_ps(X1, invW1)), _mm256_set1_ps(0.125f));
            Vector256<float> x2 = _mm256_mul_ps(_mm256_round_to_nearest_int_ps(_mm256_mul_ps(X2, invW2)), _mm256_set1_ps(0.125f));
            Vector256<float> x3 = _mm256_mul_ps(_mm256_round_to_nearest_int_ps(_mm256_mul_ps(X3, invW3)), _mm256_set1_ps(0.125f));

            Vector256<float> y0 = _mm256_mul_ps(_mm256_round_to_nearest_int_ps(_mm256_mul_ps(Y0, invW0)), _mm256_set1_ps(0.125f));
            Vector256<float> y1 = _mm256_mul_ps(_mm256_round_to_nearest_int_ps(_mm256_mul_ps(Y1, invW1)), _mm256_set1_ps(0.125f));
            Vector256<float> y2 = _mm256_mul_ps(_mm256_round_to_nearest_int_ps(_mm256_mul_ps(Y2, invW2)), _mm256_set1_ps(0.125f));
            Vector256<float> y3 = _mm256_mul_ps(_mm256_round_to_nearest_int_ps(_mm256_mul_ps(Y3, invW3)), _mm256_set1_ps(0.125f));

            // Compute unnormalized edge directions
            Vector256<float> edgeNormalsX0 = _mm256_sub_ps(y1, y0);
            Vector256<float> edgeNormalsX1 = _mm256_sub_ps(y2, y1);
            Vector256<float> edgeNormalsX2 = _mm256_sub_ps(y3, y2);
            Vector256<float> edgeNormalsX3 = _mm256_sub_ps(y0, y3);

            Vector256<float> edgeNormalsY0 = _mm256_sub_ps(x0, x1);
            Vector256<float> edgeNormalsY1 = _mm256_sub_ps(x1, x2);
            Vector256<float> edgeNormalsY2 = _mm256_sub_ps(x2, x3);
            Vector256<float> edgeNormalsY3 = _mm256_sub_ps(x3, x0);

            Vector256<float> area0 = _mm256_fmsub_ps(edgeNormalsX0, edgeNormalsY1, _mm256_mul_ps(edgeNormalsX1, edgeNormalsY0));
            Vector256<float> area1 = _mm256_fmsub_ps(edgeNormalsX1, edgeNormalsY2, _mm256_mul_ps(edgeNormalsX2, edgeNormalsY1));
            Vector256<float> area2 = _mm256_fmsub_ps(edgeNormalsX2, edgeNormalsY3, _mm256_mul_ps(edgeNormalsX3, edgeNormalsY2));
            Vector256<float> area3 = _mm256_sub_ps(_mm256_add_ps(area0, area2), area1);

            Vector256<float> minusZero256 = _mm256_set1_ps(-0.0f);

            Vector256<float> wSign0, wSign1, wSign2, wSign3;
            if (T.PossiblyNearClipped)
            {
                wSign0 = _mm256_and_ps(invW0, minusZero256);
                wSign1 = _mm256_and_ps(invW1, minusZero256);
                wSign2 = _mm256_and_ps(invW2, minusZero256);
                wSign3 = _mm256_and_ps(invW3, minusZero256);
            }
            else
            {
                wSign0 = _mm256_setzero_ps();
                wSign1 = _mm256_setzero_ps();
                wSign2 = _mm256_setzero_ps();
                wSign3 = _mm256_setzero_ps();
            }

            // Compute signs of areas. We treat 0 as negative as this allows treating primitives with zero area as backfacing.
            Vector256<float> areaSign0, areaSign1, areaSign2, areaSign3;
            if (T.PossiblyNearClipped)
            {
                // Flip areas for each vertex with W < 0. This needs to be done before comparison against 0 rather than afterwards to make sure zero-are triangles are handled correctly.
                areaSign0 = _mm256_cmp_ps(_mm256_xor_ps(_mm256_xor_ps(area0, wSign0), _mm256_xor_ps(wSign1, wSign2)), _mm256_setzero_ps(), _CMP_LE_OQ);
                areaSign1 = _mm256_and_ps(minusZero256, _mm256_cmp_ps(_mm256_xor_ps(_mm256_xor_ps(area1, wSign1), _mm256_xor_ps(wSign2, wSign3)), _mm256_setzero_ps(), _CMP_LE_OQ));
                areaSign2 = _mm256_and_ps(minusZero256, _mm256_cmp_ps(_mm256_xor_ps(_mm256_xor_ps(area2, wSign0), _mm256_xor_ps(wSign2, wSign3)), _mm256_setzero_ps(), _CMP_LE_OQ));
                areaSign3 = _mm256_and_ps(minusZero256, _mm256_cmp_ps(_mm256_xor_ps(_mm256_xor_ps(area3, wSign1), _mm256_xor_ps(wSign0, wSign3)), _mm256_setzero_ps(), _CMP_LE_OQ));
            }
            else
            {
                areaSign0 = _mm256_cmp_ps(area0, _mm256_setzero_ps(), _CMP_LE_OQ);
                areaSign1 = _mm256_and_ps(minusZero256, _mm256_cmp_ps(area1, _mm256_setzero_ps(), _CMP_LE_OQ));
                areaSign2 = _mm256_and_ps(minusZero256, _mm256_cmp_ps(area2, _mm256_setzero_ps(), _CMP_LE_OQ));
                areaSign3 = _mm256_and_ps(minusZero256, _mm256_cmp_ps(area3, _mm256_setzero_ps(), _CMP_LE_OQ));
            }

            Vector256<int> config = _mm256_or_si256(
              _mm256_or_si256(_mm256_srli_epi32(_mm256_castps_si256(areaSign3), 28), _mm256_srli_epi32(_mm256_castps_si256(areaSign2), 29)),
              _mm256_or_si256(_mm256_srli_epi32(_mm256_castps_si256(areaSign1), 30), _mm256_srli_epi32(_mm256_castps_si256(areaSign0), 31)));

            if (T.PossiblyNearClipped)
            {
                config = _mm256_or_si256(config,
                  _mm256_or_si256(
                    _mm256_or_si256(_mm256_srli_epi32(_mm256_castps_si256(wSign3), 24), _mm256_srli_epi32(_mm256_castps_si256(wSign2), 25)),
                    _mm256_or_si256(_mm256_srli_epi32(_mm256_castps_si256(wSign1), 26), _mm256_srli_epi32(_mm256_castps_si256(wSign0), 27))));
            }

            Vector256<int> modes;
            fixed (PrimitiveMode* modeTablePtr = modeTable)
            {
                modes = _mm256_and_si256(_mm256_i32gather_epi32((int*)modeTablePtr, config, 1), _mm256_set1_epi32(0xff));
            }

            if (_mm256_testz_si256(modes, modes))
            {
                continue;
            }

            Vector256<int> primitiveValid = _mm256_cmpgt_epi32(modes, _mm256_setzero_si256());

            _mm256_storeu_si256((Vector256<int>*)(primModes), modes);

            Vector256<float> minFx, minFy, maxFx, maxFy;

            if (T.PossiblyNearClipped)
            {
                // Clipless bounding box computation
                Vector256<float> infP = _mm256_set1_ps(+10000.0f);
                Vector256<float> infN = _mm256_set1_ps(-10000.0f);

                // Find interval of points with W > 0
                Vector256<float> minPx0 = _mm256_blendv_ps(x0, infP, wSign0);
                Vector256<float> minPx1 = _mm256_blendv_ps(x1, infP, wSign1);
                Vector256<float> minPx2 = _mm256_blendv_ps(x2, infP, wSign2);
                Vector256<float> minPx3 = _mm256_blendv_ps(x3, infP, wSign3);

                Vector256<float> minPx = _mm256_min_ps(
                  _mm256_min_ps(minPx0, minPx1),
                  _mm256_min_ps(minPx2, minPx3));

                Vector256<float> minPy0 = _mm256_blendv_ps(y0, infP, wSign0);
                Vector256<float> minPy1 = _mm256_blendv_ps(y1, infP, wSign1);
                Vector256<float> minPy2 = _mm256_blendv_ps(y2, infP, wSign2);
                Vector256<float> minPy3 = _mm256_blendv_ps(y3, infP, wSign3);

                Vector256<float> minPy = _mm256_min_ps(
                  _mm256_min_ps(minPy0, minPy1),
                  _mm256_min_ps(minPy2, minPy3));

                Vector256<float> maxPx0 = _mm256_xor_ps(minPx0, wSign0);
                Vector256<float> maxPx1 = _mm256_xor_ps(minPx1, wSign1);
                Vector256<float> maxPx2 = _mm256_xor_ps(minPx2, wSign2);
                Vector256<float> maxPx3 = _mm256_xor_ps(minPx3, wSign3);

                Vector256<float> maxPx = _mm256_max_ps(
                  _mm256_max_ps(maxPx0, maxPx1),
                  _mm256_max_ps(maxPx2, maxPx3));

                Vector256<float> maxPy0 = _mm256_xor_ps(minPy0, wSign0);
                Vector256<float> maxPy1 = _mm256_xor_ps(minPy1, wSign1);
                Vector256<float> maxPy2 = _mm256_xor_ps(minPy2, wSign2);
                Vector256<float> maxPy3 = _mm256_xor_ps(minPy3, wSign3);

                Vector256<float> maxPy = _mm256_max_ps(
                  _mm256_max_ps(maxPy0, maxPy1),
                  _mm256_max_ps(maxPy2, maxPy3));

                // Find interval of points with W < 0
                Vector256<float> minNx0 = _mm256_blendv_ps(infP, x0, wSign0);
                Vector256<float> minNx1 = _mm256_blendv_ps(infP, x1, wSign1);
                Vector256<float> minNx2 = _mm256_blendv_ps(infP, x2, wSign2);
                Vector256<float> minNx3 = _mm256_blendv_ps(infP, x3, wSign3);

                Vector256<float> minNx = _mm256_min_ps(
                  _mm256_min_ps(minNx0, minNx1),
                  _mm256_min_ps(minNx2, minNx3));

                Vector256<float> minNy0 = _mm256_blendv_ps(infP, y0, wSign0);
                Vector256<float> minNy1 = _mm256_blendv_ps(infP, y1, wSign1);
                Vector256<float> minNy2 = _mm256_blendv_ps(infP, y2, wSign2);
                Vector256<float> minNy3 = _mm256_blendv_ps(infP, y3, wSign3);

                Vector256<float> minNy = _mm256_min_ps(
                  _mm256_min_ps(minNy0, minNy1),
                  _mm256_min_ps(minNy2, minNy3));

                Vector256<float> maxNx0 = _mm256_blendv_ps(infN, x0, wSign0);
                Vector256<float> maxNx1 = _mm256_blendv_ps(infN, x1, wSign1);
                Vector256<float> maxNx2 = _mm256_blendv_ps(infN, x2, wSign2);
                Vector256<float> maxNx3 = _mm256_blendv_ps(infN, x3, wSign3);

                Vector256<float> maxNx = _mm256_max_ps(
                  _mm256_max_ps(maxNx0, maxNx1),
                  _mm256_max_ps(maxNx2, maxNx3));

                Vector256<float> maxNy0 = _mm256_blendv_ps(infN, y0, wSign0);
                Vector256<float> maxNy1 = _mm256_blendv_ps(infN, y1, wSign1);
                Vector256<float> maxNy2 = _mm256_blendv_ps(infN, y2, wSign2);
                Vector256<float> maxNy3 = _mm256_blendv_ps(infN, y3, wSign3);

                Vector256<float> maxNy = _mm256_max_ps(
                  _mm256_max_ps(maxNy0, maxNy1),
                  _mm256_max_ps(maxNy2, maxNy3));

                // Include interval bounds resp. infinity depending on ordering of intervals
                Vector256<float> incAx = _mm256_blendv_ps(minPx, infN, _mm256_cmp_ps(maxNx, minPx, _CMP_GT_OQ));
                Vector256<float> incAy = _mm256_blendv_ps(minPy, infN, _mm256_cmp_ps(maxNy, minPy, _CMP_GT_OQ));

                Vector256<float> incBx = _mm256_blendv_ps(maxPx, infP, _mm256_cmp_ps(maxPx, minNx, _CMP_GT_OQ));
                Vector256<float> incBy = _mm256_blendv_ps(maxPy, infP, _mm256_cmp_ps(maxPy, minNy, _CMP_GT_OQ));

                minFx = _mm256_min_ps(incAx, incBx);
                minFy = _mm256_min_ps(incAy, incBy);

                maxFx = _mm256_max_ps(incAx, incBx);
                maxFy = _mm256_max_ps(incAy, incBy);
            }
            else
            {
                // Standard bounding box inclusion
                minFx = _mm256_min_ps(_mm256_min_ps(x0, x1), _mm256_min_ps(x2, x3));
                minFy = _mm256_min_ps(_mm256_min_ps(y0, y1), _mm256_min_ps(y2, y3));

                maxFx = _mm256_max_ps(_mm256_max_ps(x0, x1), _mm256_max_ps(x2, x3));
                maxFy = _mm256_max_ps(_mm256_max_ps(y0, y1), _mm256_max_ps(y2, y3));
            }

            // Clamp and round
            Vector256<int> minX, minY, maxX, maxY;
            minX = _mm256_max_epi32(_mm256_cvttps_epi32(_mm256_add_ps(minFx, _mm256_set1_ps(4.9999f / 8.0f))), _mm256_setzero_si256());
            minY = _mm256_max_epi32(_mm256_cvttps_epi32(_mm256_add_ps(minFy, _mm256_set1_ps(4.9999f / 8.0f))), _mm256_setzero_si256());
            maxX = _mm256_min_epi32(_mm256_cvttps_epi32(_mm256_add_ps(maxFx, _mm256_set1_ps(11.0f / 8.0f))), _mm256_set1_epi32((int)m_blocksX));
            maxY = _mm256_min_epi32(_mm256_cvttps_epi32(_mm256_add_ps(maxFy, _mm256_set1_ps(11.0f / 8.0f))), _mm256_set1_epi32((int)m_blocksY));

            // Check overlap between bounding box and frustum
            Vector256<int> inFrustum = _mm256_and_si256(_mm256_cmpgt_epi32(maxX, minX), _mm256_cmpgt_epi32(maxY, minY));
            primitiveValid = _mm256_and_si256(inFrustum, primitiveValid);

            if (_mm256_testz_si256(primitiveValid, primitiveValid))
            {
                continue;
            }

            // Convert bounds from [min, max] to [min, range]
            Vector256<int> rangeX = _mm256_sub_epi32(maxX, minX);
            Vector256<int> rangeY = _mm256_sub_epi32(maxY, minY);

            // Compute Z from linear relation with 1/W
            Vector256<float> z0, z1, z2, z3;
            Vector256<float> C0 = _mm256_broadcast_ss(&c0);
            Vector256<float> C1 = _mm256_broadcast_ss(&c1);
            z0 = _mm256_fmadd_ps(invW0, C1, C0);
            z1 = _mm256_fmadd_ps(invW1, C1, C0);
            z2 = _mm256_fmadd_ps(invW2, C1, C0);
            z3 = _mm256_fmadd_ps(invW3, C1, C0);

            Vector256<float> maxZ = _mm256_max_ps(_mm256_max_ps(z0, z1), _mm256_max_ps(z2, z3));

            // If any W < 0, assume maxZ = 1 (effectively disabling Hi-Z)
            if (T.PossiblyNearClipped)
            {
                maxZ = _mm256_blendv_ps(maxZ, _mm256_set1_ps(1.0f), _mm256_or_ps(_mm256_or_ps(wSign0, wSign1), _mm256_or_ps(wSign2, wSign3)));
            }

            Vector128<int> packedDepthBounds = packDepthPremultiplied(maxZ);

            _mm_storeu_si128((Vector128<int>*)(depthBounds), packedDepthBounds);

            // Compute screen space depth plane
            Vector256<float> greaterArea = _mm256_cmp_ps(_mm256_andnot_ps(minusZero256, area0), _mm256_andnot_ps(minusZero256, area2), _CMP_LT_OQ);

            // Force triangle area to be picked in the relevant mode.
            Vector256<float> modeTriangle0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(modes, _mm256_set1_epi32((int)PrimitiveMode.Triangle0)));
            Vector256<float> modeTriangle1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(modes, _mm256_set1_epi32((int)PrimitiveMode.Triangle1)));
            greaterArea = _mm256_andnot_ps(modeTriangle0, _mm256_or_ps(modeTriangle1, greaterArea));


            Vector256<float> invArea;
            if (T.PossiblyNearClipped)
            {
                // Do a precise divison to reduce error in depth plane. Note that the area computed here
                // differs from the rasterized region if W < 0, so it can be very small for large covered screen regions.
                invArea = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_blendv_ps(area0, area2, greaterArea));
            }
            else
            {
                invArea = _mm256_rcp_ps(_mm256_blendv_ps(area0, area2, greaterArea));
            }

            Vector256<float> z12 = _mm256_sub_ps(z1, z2);
            Vector256<float> z20 = _mm256_sub_ps(z2, z0);
            Vector256<float> z30 = _mm256_sub_ps(z3, z0);


            Vector256<float> edgeNormalsX4 = _mm256_sub_ps(y0, y2);
            Vector256<float> edgeNormalsY4 = _mm256_sub_ps(x2, x0);

            Vector256<float> depthPlane0, depthPlane1, depthPlane2;
            depthPlane1 = _mm256_mul_ps(invArea, _mm256_blendv_ps(_mm256_fmsub_ps(z20, edgeNormalsX1, _mm256_mul_ps(z12, edgeNormalsX4)), _mm256_fnmadd_ps(z20, edgeNormalsX3, _mm256_mul_ps(z30, edgeNormalsX4)), greaterArea));
            depthPlane2 = _mm256_mul_ps(invArea, _mm256_blendv_ps(_mm256_fmsub_ps(z20, edgeNormalsY1, _mm256_mul_ps(z12, edgeNormalsY4)), _mm256_fnmadd_ps(z20, edgeNormalsY3, _mm256_mul_ps(z30, edgeNormalsY4)), greaterArea));

            x0 = _mm256_sub_ps(x0, _mm256_cvtepi32_ps(minX));
            y0 = _mm256_sub_ps(y0, _mm256_cvtepi32_ps(minY));

            depthPlane0 = _mm256_fnmadd_ps(x0, depthPlane1, _mm256_fnmadd_ps(y0, depthPlane2, z0));

            // If mode == Triangle0, replace edge 2 with edge 4; if mode == Triangle1, replace edge 0 with edge 4
            edgeNormalsX2 = _mm256_blendv_ps(edgeNormalsX2, edgeNormalsX4, modeTriangle0);
            edgeNormalsY2 = _mm256_blendv_ps(edgeNormalsY2, edgeNormalsY4, modeTriangle0);
            edgeNormalsX0 = _mm256_blendv_ps(edgeNormalsX0, _mm256_xor_ps(minusZero256, edgeNormalsX4), modeTriangle1);
            edgeNormalsY0 = _mm256_blendv_ps(edgeNormalsY0, _mm256_xor_ps(minusZero256, edgeNormalsY4), modeTriangle1);

            // Flip edges if W < 0
            Vector256<float> edgeFlipMask0, edgeFlipMask1, edgeFlipMask2, edgeFlipMask3;
            if (T.PossiblyNearClipped)
            {
                edgeFlipMask0 = _mm256_xor_ps(wSign0, _mm256_blendv_ps(wSign1, wSign2, modeTriangle1));
                edgeFlipMask1 = _mm256_xor_ps(wSign1, wSign2);
                edgeFlipMask2 = _mm256_xor_ps(wSign2, _mm256_blendv_ps(wSign3, wSign0, modeTriangle0));
                edgeFlipMask3 = _mm256_xor_ps(wSign0, wSign3);
            }
            else
            {
                edgeFlipMask0 = _mm256_setzero_ps();
                edgeFlipMask1 = _mm256_setzero_ps();
                edgeFlipMask2 = _mm256_setzero_ps();
                edgeFlipMask3 = _mm256_setzero_ps();
            }

            // Normalize edge equations for lookup
            normalizeEdge<T>(ref edgeNormalsX0, ref edgeNormalsY0, edgeFlipMask0);
            normalizeEdge<T>(ref edgeNormalsX1, ref edgeNormalsY1, edgeFlipMask1);
            normalizeEdge<T>(ref edgeNormalsX2, ref edgeNormalsY2, edgeFlipMask2);
            normalizeEdge<T>(ref edgeNormalsX3, ref edgeNormalsY3, edgeFlipMask3);

            const float maxOffset = -minEdgeOffset;
            Vector256<float> add256 = _mm256_set1_ps(0.5f - minEdgeOffset * (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
            Vector256<float> edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3;

            edgeOffsets0 = _mm256_fnmadd_ps(x0, edgeNormalsX0, _mm256_fnmadd_ps(y0, edgeNormalsY0, add256));
            edgeOffsets1 = _mm256_fnmadd_ps(x1, edgeNormalsX1, _mm256_fnmadd_ps(y1, edgeNormalsY1, add256));
            edgeOffsets2 = _mm256_fnmadd_ps(x2, edgeNormalsX2, _mm256_fnmadd_ps(y2, edgeNormalsY2, add256));
            edgeOffsets3 = _mm256_fnmadd_ps(x3, edgeNormalsX3, _mm256_fnmadd_ps(y3, edgeNormalsY3, add256));

            edgeOffsets1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minX), edgeNormalsX1, edgeOffsets1);
            edgeOffsets2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minX), edgeNormalsX2, edgeOffsets2);
            edgeOffsets3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minX), edgeNormalsX3, edgeOffsets3);

            edgeOffsets1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minY), edgeNormalsY1, edgeOffsets1);
            edgeOffsets2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minY), edgeNormalsY2, edgeOffsets2);
            edgeOffsets3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minY), edgeNormalsY3, edgeOffsets3);

            // Quantize slopes
            Vector256<int> slopeLookups0, slopeLookups1, slopeLookups2, slopeLookups3;
            slopeLookups0 = quantizeSlopeLookup(edgeNormalsX0, edgeNormalsY0);
            slopeLookups1 = quantizeSlopeLookup(edgeNormalsX1, edgeNormalsY1);
            slopeLookups2 = quantizeSlopeLookup(edgeNormalsX2, edgeNormalsY2);
            slopeLookups3 = quantizeSlopeLookup(edgeNormalsX3, edgeNormalsY3);

            Vector256<int> firstBlockIdx = _mm256_add_epi32(_mm256_mullo_epi16(minY, _mm256_set1_epi32((int)m_blocksX)), minX);

            _mm256_storeu_si256((Vector256<int>*)(firstBlocks), firstBlockIdx);

            _mm256_storeu_si256((Vector256<int>*)(rangesX), rangeX);

            _mm256_storeu_si256((Vector256<int>*)(rangesY), rangeY);

            // Transpose into AoS
            transpose256(depthPlane0, depthPlane1, depthPlane2, _mm256_setzero_ps(), depthPlane);

            transpose256(edgeNormalsX0, edgeNormalsX1, edgeNormalsX2, edgeNormalsX3, edgeNormalsX);

            transpose256(edgeNormalsY0, edgeNormalsY1, edgeNormalsY2, edgeNormalsY3, edgeNormalsY);

            transpose256(edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3, edgeOffsets);

            transpose256i(slopeLookups0, slopeLookups1, slopeLookups2, slopeLookups3, slopeLookups);

            uint validMask = (uint)_mm256_movemask_ps(_mm256_castsi256_ps(primitiveValid));

            // Fetch data pointers since we'll manually strength-reduce memory arithmetic
            ulong* pTable = m_precomputedRasterTables;
            ushort* pHiZBuffer = m_hiZ;
            Vector128<int>* pDepthBuffer = m_depthBuffer;

            // Loop over set bits
            while (validMask != 0)
            {
                uint primitiveIdx = (uint)BitOperations.TrailingZeroCount(validMask);

                // Clear lowest set bit in mask
                validMask &= validMask - 1;

                uint primitiveIdxTransposed = ((primitiveIdx << 1) & 7) | (primitiveIdx >> 2);

                // Extract and prepare per-primitive data
                ushort primitiveMaxZ = depthBounds[primitiveIdx];

                Vector256<float> depthDx = _mm256_broadcastss_ps(_mm_permute_ps(depthPlane[primitiveIdxTransposed], 0b01_01_01_01));
                Vector256<float> depthDy = _mm256_broadcastss_ps(_mm_permute_ps(depthPlane[primitiveIdxTransposed], 0b10_10_10_10));

                const float depthSamplePos = -0.5f + 1.0f / 16.0f;
                Vector256<float> lineDepth =
                  _mm256_fmadd_ps(depthDx, _mm256_setr_ps(depthSamplePos + 0.0f, depthSamplePos + 0.125f, depthSamplePos + 0.25f, depthSamplePos + 0.375f, depthSamplePos + 0.0f, depthSamplePos + 0.125f, depthSamplePos + 0.25f, depthSamplePos + 0.375f),
                    _mm256_fmadd_ps(depthDy, _mm256_setr_ps(depthSamplePos + 0.0f, depthSamplePos + 0.0f, depthSamplePos + 0.0f, depthSamplePos + 0.0f, depthSamplePos + 0.125f, depthSamplePos + 0.125f, depthSamplePos + 0.125f, depthSamplePos + 0.125f),
                      _mm256_broadcastss_ps(depthPlane[primitiveIdxTransposed])));

                Vector128<int> slopeLookup = slopeLookups[primitiveIdxTransposed];
                Vector128<float> edgeNormalX = edgeNormalsX[primitiveIdxTransposed];
                Vector128<float> edgeNormalY = edgeNormalsY[primitiveIdxTransposed];
                Vector128<float> lineOffset = edgeOffsets[primitiveIdxTransposed];

                uint blocksX = m_blocksX;

                uint firstBlock = firstBlocks[primitiveIdx];
                uint blockRangeX = rangesX[primitiveIdx];
                uint blockRangeY = rangesY[primitiveIdx];

                ushort* pPrimitiveHiZ = pHiZBuffer + firstBlock;
                Vector256<int>* pPrimitiveOut = (Vector256<int>*)(pDepthBuffer) + 4 * firstBlock;

                uint primitiveMode = primModes[primitiveIdx];

                for (uint blockY = 0;
                  blockY < blockRangeY;
                  ++blockY,
                  pPrimitiveHiZ += blocksX,
                  pPrimitiveOut += 4 * blocksX,
                  lineDepth = _mm256_add_ps(lineDepth, depthDy),
                  lineOffset = _mm_add_ps(lineOffset, edgeNormalY))
                {
                    ushort* pBlockRowHiZ = pPrimitiveHiZ;
                    Vector256<int>* @out = pPrimitiveOut;

                    Vector128<float> offset = lineOffset;
                    Vector256<float> depth = lineDepth;

                    bool anyBlockHit = false;
                    for (uint blockX = 0;
                      blockX < blockRangeX;
                      ++blockX,
                      pBlockRowHiZ += 1,
                      @out += 4,
                      depth = _mm256_add_ps(depthDx, depth),
                      offset = _mm_add_ps(edgeNormalX, offset))
                    {
                        ushort hiZ = *pBlockRowHiZ;
                        if (hiZ >= primitiveMaxZ)
                        {
                            continue;
                        }

                        ulong blockMask;
                        if (primitiveMode == (uint)PrimitiveMode.Convex)    // 83-97%
                        {
                            // Simplified conservative test: combined block mask will be zero if any offset is outside of range
                            Vector128<float> anyOffsetOutsideMask = _mm_cmpge_ps(offset, _mm_set1_ps(OFFSET_QUANTIZATION_FACTOR - 1));
                            if (!_mm_testz_ps(anyOffsetOutsideMask, anyOffsetOutsideMask))
                            {
                                if (anyBlockHit)
                                {
                                    // Convexity implies we won't hit another block in this row and can skip to the next line.
                                    break;
                                }
                                continue;
                            }

                            anyBlockHit = true;

                            Vector128<int> offsetClamped = _mm_max_epi32(_mm_cvttps_epi32(offset), _mm_setzero_si128());

                            Vector128<int> lookup = _mm_or_si128(slopeLookup, offsetClamped);

                            // Generate block mask
                            ulong A = pTable[(uint)(_mm_cvtsi128_si32(lookup))];
                            ulong B = pTable[(uint)(_mm_extract_epi32(lookup, 1))];
                            ulong C = pTable[(uint)(_mm_extract_epi32(lookup, 2))];
                            ulong D = pTable[(uint)(_mm_extract_epi32(lookup, 3))];

                            blockMask = (A & B) & (C & D);

                            // It is possible but very unlikely that blockMask == 0 if all A,B,C,D != 0 according to the conservative test above, so we skip the additional branch here.
                        }
                        else
                        {
                            Vector128<int> offsetClamped = _mm_min_epi32(_mm_max_epi32(_mm_cvttps_epi32(offset), _mm_setzero_si128()), _mm_set1_epi32(OFFSET_QUANTIZATION_FACTOR - 1));
                            Vector128<int> lookup = _mm_or_si128(slopeLookup, offsetClamped);

                            // Generate block mask
                            ulong A = pTable[(uint)(_mm_cvtsi128_si32(lookup))];
                            ulong B = pTable[(uint)(_mm_extract_epi32(lookup, 1))];
                            ulong C = pTable[(uint)(_mm_extract_epi32(lookup, 2))];
                            ulong D = pTable[(uint)(_mm_extract_epi32(lookup, 3))];

                            // Switch over primitive mode. MSVC compiles this as a "sub eax, 1; jz label;" ladder, so the mode enum is ordered by descending frequency of occurence
                            // to optimize branch efficiency. By ensuring we have a default case that falls through to the last possible value (ConcaveLeft if not near clipped,
                            // ConcaveCenter otherwise) we avoid the last branch in the ladder.
                            switch (primitiveMode)
                            {
                                case (uint)PrimitiveMode.Triangle0:             // 2.3-11%
                                    blockMask = A & B & C;
                                    break;

                                case (uint)PrimitiveMode.Triangle1:             // 0.1-4%
                                    blockMask = A & C & D;
                                    break;

                                case (uint)PrimitiveMode.ConcaveRight:          // 0.01-0.9%
                                    blockMask = (A | D) & (B & C);
                                    break;

                                default:
                                    // Case ConcaveCenter can only occur if any W < 0
                                    if (T.PossiblyNearClipped)
                                    {
                                        // case ConcaveCenter:			// < 1e-6%
                                        blockMask = (A & B) | (C & D);
                                        break;
                                    }
                                    // Fall-through
                                    goto case (uint)PrimitiveMode.ConcaveLeft;

                                case (uint)PrimitiveMode.ConcaveLeft:           // 0.01-0.6%
                                    blockMask = (A & D) & (B | C);
                                    break;
                            }

                            // No pixels covered => skip block
                            if (blockMask == 0)
                            {
                                continue;
                            }
                        }

                        // Generate depth values around block
                        Vector256<float> depth0 = depth;
                        Vector256<float> depth1 = _mm256_fmadd_ps(depthDx, _mm256_set1_ps(0.5f), depth0);
                        Vector256<float> depth8 = _mm256_add_ps(depthDy, depth0);
                        Vector256<float> depth9 = _mm256_add_ps(depthDy, depth1);

                        // Pack depth
                        Vector256<int> d0 = packDepthPremultiplied(depth0, depth1);
                        Vector256<int> d4 = packDepthPremultiplied(depth8, depth9);

                        // Interpolate remaining values in packed space
                        Vector256<int> d2 = _mm256_avg_epu16(d0, d4);
                        Vector256<int> d1 = _mm256_avg_epu16(d0, d2);
                        Vector256<int> d3 = _mm256_avg_epu16(d2, d4);

                        // Not all pixels covered - mask depth 
                        if (blockMask != 0xffff_ffff_ffff_ffff)
                        {
                            Vector128<int> A = _mm_cvtsi64x_si128((long)blockMask);
                            Vector128<int> B = _mm_slli_epi64(A, 4);
                            Vector256<int> C = _mm256_inserti128_si256(_mm256_castsi128_si256(A), B, 1);
                            Vector256<int> rowMask = _mm256_unpacklo_epi8(C, C);

                            d0 = _mm256_blendv_epi8(_mm256_setzero_si256(), d0, _mm256_slli_epi16(rowMask, 3));
                            d1 = _mm256_blendv_epi8(_mm256_setzero_si256(), d1, _mm256_slli_epi16(rowMask, 2));
                            d2 = _mm256_blendv_epi8(_mm256_setzero_si256(), d2, _mm256_add_epi16(rowMask, rowMask));
                            d3 = _mm256_blendv_epi8(_mm256_setzero_si256(), d3, rowMask);
                        }

                        // Test fast clear flag
                        if (hiZ != 1)
                        {
                            // Merge depth values
                            d0 = _mm256_max_epu16(_mm256_load_si256(@out + 0), d0);
                            d1 = _mm256_max_epu16(_mm256_load_si256(@out + 1), d1);
                            d2 = _mm256_max_epu16(_mm256_load_si256(@out + 2), d2);
                            d3 = _mm256_max_epu16(_mm256_load_si256(@out + 3), d3);
                        }

                        // Store back new depth
                        _mm256_store_si256(@out + 0, d0);
                        _mm256_store_si256(@out + 1, d1);
                        _mm256_store_si256(@out + 2, d2);
                        _mm256_store_si256(@out + 3, d3);

                        // Update HiZ
                        Vector256<int> newMinZ = _mm256_min_epu16(_mm256_min_epu16(d0, d1), _mm256_min_epu16(d2, d3));
                        Vector128<int> newMinZ16 = _mm_minpos_epu16(_mm_min_epu16(_mm256_castsi256_si128(newMinZ), _mm256_extracti128_si256(newMinZ, 1)));

                        *pBlockRowHiZ = (ushort)((uint)(_mm_cvtsi128_si32(newMinZ16)));
                    }
                }
            }
        }
    }
}