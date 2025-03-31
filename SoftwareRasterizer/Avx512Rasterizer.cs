using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

using static VectorMath;

public unsafe class Avx512Rasterizer<Fma> : Rasterizer
    where Fma : IMultiplyAdd, IMultiplyAdd128, IMultiplyAdd256
{
    private const FloatComparisonMode _CMP_LT_OQ = FloatComparisonMode.OrderedLessThanNonSignaling;
    private const FloatComparisonMode _CMP_LE_OQ = FloatComparisonMode.OrderedLessThanOrEqualNonSignaling;
    private const FloatComparisonMode _CMP_GT_OQ = FloatComparisonMode.OrderedGreaterThanNonSignaling;

    private const int Alignment = 512 / 8; // sizeof(Vector512<>)

    public Avx512Rasterizer(RasterizationTable rasterizationTable, uint width, uint height) :
        base(rasterizationTable, width, height, Alignment)
    {
    }

    public static Avx512Rasterizer<Fma> Create(RasterizationTable rasterizationTable, uint width, uint height)
    {
        bool success = false;
        rasterizationTable.DangerousAddRef(ref success);
        if (success)
        {
            return new Avx512Rasterizer<Fma>(rasterizationTable, width, height);
        }
        throw new ObjectDisposedException(rasterizationTable.GetType().Name);
    }

    public override unsafe void setModelViewProjection(float* matrix)
    {
        Vector128<float> mat0 = Sse.LoadVector128(matrix + 0);
        Vector128<float> mat1 = Sse.LoadVector128(matrix + 4);
        Vector128<float> mat2 = Sse.LoadVector128(matrix + 8);
        Vector128<float> mat3 = Sse.LoadVector128(matrix + 12);

        (mat0, mat1, mat2, mat3) = _MM_TRANSPOSE4_PS(mat0, mat1, mat2, mat3);

        // Store rows
        Sse.StoreAligned(m_modelViewProjectionRaw + 0, mat0);
        Sse.StoreAligned(m_modelViewProjectionRaw + 4, mat1);
        Sse.StoreAligned(m_modelViewProjectionRaw + 8, mat2);
        Sse.StoreAligned(m_modelViewProjectionRaw + 12, mat3);

        // Bake viewport transform into matrix and 6shift by half a block
        mat0 = (mat0 + mat3) * Vector128.Create(m_width * 0.5f - 4.0f);
        mat1 = (mat1 + mat3) * Vector128.Create(m_height * 0.5f - 4.0f);

        // Map depth from [-1, 1] to [bias, 0]
        mat2 = ((mat3 - mat2) * Vector128.Create(0.5f * floatCompressionBias));

        (mat0, mat1, mat2, mat3) = _MM_TRANSPOSE4_PS(mat0, mat1, mat2, mat3);

        // Store prebaked cols
        Sse.StoreAligned(m_modelViewProjection + 0, mat0);
        Sse.StoreAligned(m_modelViewProjection + 4, mat1);
        Sse.StoreAligned(m_modelViewProjection + 8, mat2);
        Sse.StoreAligned(m_modelViewProjection + 12, mat3);
    }

    public override void clear()
    {
        // Mark blocks as cleared by setting Hi Z to 1 (one unit separated from far plane). 
        // This value is extremely unlikely to occur during normal rendering, so we don't
        // need to guard against a HiZ of 1 occuring naturally. This is different from a value of 0, 
        // which will occur every time a block is partially covered for the first time.
        Vector128<int> clearValue = Vector128.Create((short)1).AsInt32();
        uint count = m_hiZ_Size / 8;
        Vector128<int>* pHiZ = (Vector128<int>*)m_hiZ;
        for (uint offset = 0; offset < count; ++offset)
        {
            Sse2.StoreAligned((int*)pHiZ, clearValue);
            pHiZ++;
        }
    }

    public override bool queryVisibility(Vector4 vBoundsMin, Vector4 vBoundsMax, out bool needsClipping)
    {
        Vector128<float> boundsMin = vBoundsMin.AsVector128();
        Vector128<float> boundsMax = vBoundsMax.AsVector128();

        // Frustum cull
        Vector128<float> extents = (boundsMax - boundsMin);
        Vector128<float> center = (boundsMax + boundsMin); // Bounding box center times 2 - but since W = 2, the plane equations work out correctly
        Vector128<float> minusZero = Vector128.Create(-0.0f);

        Vector128<float> row0 = Sse.LoadVector128(m_modelViewProjectionRaw + 0);
        Vector128<float> row1 = Sse.LoadVector128(m_modelViewProjectionRaw + 4);
        Vector128<float> row2 = Sse.LoadVector128(m_modelViewProjectionRaw + 8);
        Vector128<float> row3 = Sse.LoadVector128(m_modelViewProjectionRaw + 12);

        // Compute distance from each frustum plane
        Vector128<float> plane0 = (row3 + row0);
        Vector128<float> offset0 = (center + (extents ^ (plane0 & minusZero)));
        int combined = BitConverter.SingleToInt32Bits(Vector128.Dot(plane0, offset0));

        Vector128<float> plane1 = (row3 - row0);
        Vector128<float> offset1 = (center + (extents ^ (plane1 & minusZero)));
        combined |= BitConverter.SingleToInt32Bits(Vector128.Dot(plane1, offset1));

        Vector128<float> plane2 = (row3 + row1);
        Vector128<float> offset2 = (center + (extents ^ (plane2 & minusZero)));
        combined |= BitConverter.SingleToInt32Bits(Vector128.Dot(plane2, offset2));

        Vector128<float> plane3 = (row3 - row1);
        Vector128<float> offset3 = (center + (extents ^ (plane3 & minusZero)));
        combined |= BitConverter.SingleToInt32Bits(Vector128.Dot(plane3, offset3));

        Vector128<float> plane4 = (row3 + row2);
        Vector128<float> offset4 = (center + (extents ^ (plane4 & minusZero)));
        combined |= BitConverter.SingleToInt32Bits(Vector128.Dot(plane4, offset4));

        Vector128<float> plane5 = (row3 - row2);
        Vector128<float> offset5 = (center + (extents ^ (plane5 & minusZero)));
        combined |= BitConverter.SingleToInt32Bits(Vector128.Dot(plane5, offset5));

        // Can't use Avx.TestZ or _mm_comile_ss here because the OR's above created garbage in the non-sign bits
        if (combined < 0)
        {
            needsClipping = false;
            return false;
        }

        // Load prebaked projection matrix
        Vector128<float> col0 = Sse.LoadVector128(m_modelViewProjection + 0);
        Vector128<float> col1 = Sse.LoadVector128(m_modelViewProjection + 4);
        Vector128<float> col2 = Sse.LoadVector128(m_modelViewProjection + 8);
        Vector128<float> col3 = Sse.LoadVector128(m_modelViewProjection + 12);

        // Transform edges
        Vector128<float> egde0 = col0 * Avx2.BroadcastScalarToVector128(extents);
        Vector128<float> egde1 = col1 * Avx.Permute(extents, 0b01_01_01_01);
        Vector128<float> egde2 = col2 * Avx.Permute(extents, 0b10_10_10_10);

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
          Fma.MulAdd(col0, Avx2.BroadcastScalarToVector128(boundsMin),
            Fma.MulAdd(col1, Avx.Permute(boundsMin, 0b01_01_01_01),
              Fma.MulAdd(col2, Avx.Permute(boundsMin, 0b10_10_10_10),
                col3)));

        // Transform remaining corners by adding edge vectors
        corners1 = corners0 + egde0;
        corners2 = corners0 + egde1;
        corners4 = corners0 + egde2;
                   
        corners3 = corners1 + egde1;
        corners5 = corners4 + egde0;
        corners6 = corners2 + egde2;

        corners7 = corners6 + egde0;

        // Transpose into SoA
        (corners0, corners1, corners2, corners3) = _MM_TRANSPOSE4_PS(corners0, corners1, corners2, corners3);
        (corners4, corners5, corners6, corners7) = _MM_TRANSPOSE4_PS(corners4, corners5, corners6, corners7);

        // Even if all bounding box corners have W > 0 here, we may end up with some vertices with W < 0 to due floating point differences; so test with some epsilon if any W < 0.
        Vector128<float> maxExtent = Sse.Max(extents, Avx.Permute(extents, 0b01_00_11_10));
        maxExtent = Sse.Max(maxExtent, Avx.Permute(maxExtent, 0b10_11_00_01));
        Vector128<float> nearPlaneEpsilon = (maxExtent * Vector128.Create(0.001f));
        Vector128<float> closeToNearPlane = (Vector128.LessThan(corners3, nearPlaneEpsilon) | Vector128.LessThan(corners7, nearPlaneEpsilon));
        if (!Avx.TestZ(closeToNearPlane, closeToNearPlane))
        {
            needsClipping = true;
            return true;
        }

        needsClipping = false;

        // Perspective division
        corners3 = Sse.Reciprocal(corners3);
        corners0 *= corners3;
        corners1 *= corners3;
        corners2 *= corners3;

        corners7 = Sse.Reciprocal(corners7);
        corners4 *= corners7;
        corners5 *= corners7;
        corners6 *= corners7;

        // Vertical mins and maxes
        Vector128<float> minsX = Sse.Min(corners0, corners4);
        Vector128<float> maxsX = Sse.Max(corners0, corners4);

        Vector128<float> minsY = Sse.Min(corners1, corners5);
        Vector128<float> maxsY = Sse.Max(corners1, corners5);

        // Horizontal reduction, step 1
        Vector128<float> minsXY = Sse.Min(Sse.UnpackLow(minsX, minsY), Sse.UnpackHigh(minsX, minsY));
        Vector128<float> maxsXY = Sse.Max(Sse.UnpackLow(maxsX, maxsY), Sse.UnpackHigh(maxsX, maxsY));

        // Clamp bounds
        minsXY = Sse.Max(minsXY, Vector128<float>.Zero);
        maxsXY = Sse.Min(maxsXY, Vector128.Create(m_width - 1f, m_height - 1f, m_width - 1f, m_height - 1f));

        // Negate maxes so we can round in the same direction
        maxsXY ^= minusZero;

        // Horizontal reduction, step 2
        Vector128<float> boundsF = Sse.Min(Sse.UnpackLow(minsXY, maxsXY), Sse.UnpackHigh(minsXY, maxsXY));

        // Round towards -infinity and convert to int
        Vector128<int> boundsI = Sse2.ConvertToVector128Int32WithTruncation(Sse41.RoundToNegativeInfinity(boundsF));

        // Store as scalars
        Span<uint> bounds = stackalloc uint[4];
        boundsI.AsUInt32().CopyTo(bounds);

        uint minX = bounds[0];
        uint maxX = bounds[1];
        uint minY = bounds[2];
        uint maxY = bounds[3];

        // Revert the sign change we did for the maxes
        maxX = (uint)-(int)maxX;
        maxY = (uint)-(int)maxY;

        // No intersection between quad and screen area
        if (minX >= maxX || minY >= maxY)
        {
            return false;
        }

        Vector128<ushort> depth = packDepthPremultiplied(corners2, corners6);

        ushort maxZ = (ushort)(0xFFFFu ^ V128Helper.MinHorizontal(depth ^ Vector128<ushort>.AllBitsSet));
        
        if (!query2D(minX, maxX, minY, maxY, maxZ))
        {
            return false;
        }

        return true;
    }

    public override bool query2D(uint minX, uint maxX, uint minY, uint maxY, uint maxZ)
    {
        ushort* pHiZBuffer = m_hiZ;
        Vector128<int>* pDepthBuffer = m_depthBuffer;

        uint blockMinX = minX / 8;
        uint blockMaxX = maxX / 8;

        uint blockMinY = minY / 8;
        uint blockMaxY = maxY / 8;

        Vector128<ushort> maxZV = Vector128.Create((ushort)maxZ);

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

                    Vector128<int> notVisible = Vector128.Equals(Vector128.Min(rowDepth.AsUInt16(), maxZV), maxZV).AsInt32();

                    uint visiblePixelMask = ~notVisible.AsByte().ExtractMostSignificantBits();

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

    [SkipLocalsInit]
    public override void readBackDepth(byte* target)
    {
        const float bias = 1.0f / floatCompressionBias;

        const int stackBufferSize =
            Alignment - 1 +
            sizeof(float) * 16 * 4 * 1; // Vector512<float>[8] x 1

        byte* stackBuffer = stackalloc byte[stackBufferSize];
        byte* alignedBuffer = (byte*)((nint)(stackBuffer + (Alignment - 1)) & -Alignment);

        Vector512<float>* linDepthA = (Vector512<float>*)alignedBuffer;
        uint width = m_width;

        for (uint blockY = 0; blockY < m_blocksY; ++blockY)
        {
            for (uint blockX = 0; blockX < m_blocksX; ++blockX)
            {
                uint* dst = (uint*) target + (8 * blockX + width * (8 * blockY));

                if (m_hiZ[blockY * m_blocksX + blockX] == 1)
                {
                    Vector256<uint> zero = Vector256<uint>.Zero;
                    for (uint y = 0; y < 8; ++y)
                    {
                        zero.StoreAligned(dst);
                        dst += width;
                    }
                    continue;
                }

                Vector512<float> vBias = Vector512.Create(bias);
                Vector512<float> vOne = Vector512.Create(1.0f);
                Vector512<float> vDiv = Vector512.Create(100 * 256 * 2 * 0.25f);
                Vector512<float> vSt = Vector512.Create(0.25f + 1000.0f);
                Vector512<float> vSf = Vector512.Create(1000.0f - 0.25f);

                Vector128<int>* source = &m_depthBuffer[8 * (blockY * m_blocksX + blockX)];
                for (uint y = 0; y < 4; y++)
                {
                    Vector256<int> depthI = Vector256.Load((int*)(source + y * 2));

                    Vector512<int> depthI512 = (Avx512F.ConvertToVector512Int32(depthI.AsUInt16()) << 12);
                    Vector512<float> depth = (depthI512.AsSingle() * vBias);

                    Vector512<float> linDepth = (vDiv / (vSt - ((vOne - depth) * vSf)));
                    linDepth.StoreAligned((float*)(linDepthA + y));
                }
                
                Vector512<float> vRcp100 = Vector512.Create(1.0f / 100.0f);
                Vector512<ushort> vZeroMax = Avx512BW.UnpackLow(Vector512<byte>.Zero, Vector512<byte>.AllBitsSet).AsUInt16();
                Vector512<ushort> vMask = Vector512.Create((ushort)0xff);

                {
                    Vector512<float> depth0 = Vector512.LoadAligned((float*) (linDepthA + 0));
                    Vector512<float> depth1 = Vector512.LoadAligned((float*) (linDepthA + 1));
                    Vector512<float> depth2 = Vector512.LoadAligned((float*) (linDepthA + 2));
                    Vector512<float> depth3 = Vector512.LoadAligned((float*) (linDepthA + 3));

                    Vector512<int> vR32_0 = Avx512F.ConvertToVector512Int32WithTruncation(depth0 * vRcp100);
                    Vector512<int> vR32_1 = Avx512F.ConvertToVector512Int32WithTruncation(depth1 * vRcp100);
                    Vector512<int> vR32_2 = Avx512F.ConvertToVector512Int32WithTruncation(depth2 * vRcp100);
                    Vector512<int> vR32_3 = Avx512F.ConvertToVector512Int32WithTruncation(depth3 * vRcp100);

                    Vector512<ushort> vR16_0 = Avx512BW.PackUnsignedSaturate(vR32_0, vR32_1) & vMask;
                    Vector512<ushort> vR16_1 = Avx512BW.PackUnsignedSaturate(vR32_2, vR32_3) & vMask;
                    Vector512<byte> vR8 = Avx512BW.PackUnsignedSaturate(vR16_0.AsInt16(), vR16_1.AsInt16());

                    Vector512<int> vG32_0 = Avx512F.ConvertToVector512Int32WithTruncation(depth0);
                    Vector512<int> vG32_1 = Avx512F.ConvertToVector512Int32WithTruncation(depth1);
                    Vector512<int> vG32_2 = Avx512F.ConvertToVector512Int32WithTruncation(depth2);
                    Vector512<int> vG32_3 = Avx512F.ConvertToVector512Int32WithTruncation(depth3);

                    Vector512<ushort> vG16_0 = Avx512BW.PackUnsignedSaturate(vG32_0, vG32_1) & vMask;
                    Vector512<ushort> vG16_1 = Avx512BW.PackUnsignedSaturate(vG32_2, vG32_3) & vMask;
                    Vector512<byte> vG8 = Avx512BW.PackUnsignedSaturate(vG16_0.AsInt16(), vG16_1.AsInt16());

                    Vector512<ushort> vRG_Lo = Avx512BW.UnpackLow(vR8, vG8).AsUInt16();
                    Vector512<ushort> vRG_Hi = Avx512BW.UnpackHigh(vR8, vG8).AsUInt16();

                    Vector512<uint> result1 = Avx512BW.UnpackLow(vRG_Lo, vZeroMax).AsUInt32();
                    Vector512<uint> result2 = Avx512BW.UnpackHigh(vRG_Lo, vZeroMax).AsUInt32();
                    Vector512<uint> result3 = Avx512BW.UnpackLow(vRG_Hi, vZeroMax).AsUInt32();
                    Vector512<uint> result4 = Avx512BW.UnpackHigh(vRG_Hi, vZeroMax).AsUInt32();

                    result1.GetLower().StoreAligned(dst);
                    result1.GetUpper().StoreAligned(dst += width);
                    result2.GetLower().StoreAligned(dst += width);
                    result2.GetUpper().StoreAligned(dst += width);
                    
                    result3.GetLower().StoreAligned(dst += width);
                    result3.GetUpper().StoreAligned(dst += width);
                    result4.GetLower().StoreAligned(dst += width);
                    result4.GetUpper().StoreAligned(dst += width);
                    dst += width;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void transpose256(Vector256<float> A, Vector256<float> B, Vector256<float> C, Vector256<float> D, Vector128<float>* @out)
    {
        Vector256<float> _Tmp0 = Avx.Shuffle(A, B, 0x44);
        Vector256<float> _Tmp2 = Avx.Shuffle(A, B, 0xEE);
        Vector256<float> _Tmp1 = Avx.Shuffle(C, D, 0x44);
        Vector256<float> _Tmp3 = Avx.Shuffle(C, D, 0xEE);

        Vector256<float> tA = Avx.Shuffle(_Tmp0, _Tmp1, 0x88);
        Vector256<float> tB = Avx.Shuffle(_Tmp0, _Tmp1, 0xDD);
        Vector256<float> tC = Avx.Shuffle(_Tmp2, _Tmp3, 0x88);
        Vector256<float> tD = Avx.Shuffle(_Tmp2, _Tmp3, 0xDD);

        Avx.StoreAligned((float*)(@out + 0), tA);
        Avx.StoreAligned((float*)(@out + 2), tB);
        Avx.StoreAligned((float*)(@out + 4), tC);
        Avx.StoreAligned((float*)(@out + 6), tD);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void transpose256i(Vector256<int> A, Vector256<int> B, Vector256<int> C, Vector256<int> D, Vector128<int>* @out)
    {
        Vector256<long> _Tmp0 = Avx2.UnpackLow(A, B).AsInt64();
        Vector256<long> _Tmp1 = Avx2.UnpackLow(C, D).AsInt64();
        Vector256<long> _Tmp2 = Avx2.UnpackHigh(A, B).AsInt64();
        Vector256<long> _Tmp3 = Avx2.UnpackHigh(C, D).AsInt64();

        Vector256<int> tA = Avx2.UnpackLow(_Tmp0, _Tmp1).AsInt32();
        Vector256<int> tB = Avx2.UnpackHigh(_Tmp0, _Tmp1).AsInt32();
        Vector256<int> tC = Avx2.UnpackLow(_Tmp2, _Tmp3).AsInt32();
        Vector256<int> tD = Avx2.UnpackHigh(_Tmp2, _Tmp3).AsInt32();

        Avx.StoreAligned((int*)(@out + 0), tA);
        Avx.StoreAligned((int*)(@out + 2), tB);
        Avx.StoreAligned((int*)(@out + 4), tC);
        Avx.StoreAligned((int*)(@out + 6), tD);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (Vector256<float>, Vector256<float>) normalizeEdge<T>(
        Vector256<float> nx, Vector256<float> ny, Vector256<float> edgeFlipMask)
        where T : IPossiblyNearClipped
    {
        Vector256<float> minusZero = Vector256.Create(-0.0f);
        Vector256<float> invLen = Avx.Reciprocal(Vector256.AndNot(nx, minusZero) + Vector256.AndNot(ny, minusZero));

        const float maxOffset = -minEdgeOffset;
        Vector256<float> mul = Vector256.Create((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        if (T.PossiblyNearClipped)
        {
            mul ^= edgeFlipMask;
        }

        invLen *= mul;
        nx *= invLen;
        ny *= invLen;

        return (nx, ny);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> quantizeSlopeLookup(Vector256<float> nx, Vector256<float> ny)
    {
        Vector256<int> yNeg = Avx.Compare(ny, Vector256<float>.Zero, _CMP_LE_OQ).AsInt32();

        // Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
        const float maxOffset = -minEdgeOffset;
        const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f / ((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        const float add = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f + 0.5f;

        Vector256<int> quantizedSlope = Avx.ConvertToVector256Int32WithTruncation(Fma.MulAdd(nx, Vector256.Create(mul), Vector256.Create(add)));
        return ((quantizedSlope << 1) - yNeg) << OFFSET_QUANTIZATION_BITS;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> packDepthPremultiplied(Vector128<float> depthA, Vector128<float> depthB)
    {
        Vector128<int> x1 = depthA.AsInt32() >> 12;
        Vector128<int> x2 = depthB.AsInt32() >> 12;
        return Sse41.PackUnsignedSaturate(x1, x2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> packDepthPremultiplied(Vector256<float> depth)
    {
        Vector256<int> x = depth.AsInt32() >> 12;
        return Sse41.PackUnsignedSaturate(x.GetLower(), x.GetUpper());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<ushort> packDepthPremultiplied(Vector256<float> depthA, Vector256<float> depthB)
    {
        Vector256<int> x1 = depthA.AsInt32() >> 12;
        Vector256<int> x2 = depthB.AsInt32() >> 12;
        return Avx2.PackUnsignedSaturate(x1, x2);
    }

    public override void rasterize<T>(in Occluder occluder)
    {
        Vector256<int>* vertexData = occluder.m_vertexData;
        uint packetCount = occluder.m_packetCount;

        Vector256<int> maskY = Vector256.Create(2047 << 10);
        Vector256<int> maskZ = Vector256.Create(1023);

        // Note that unaligned loads do not have a latency penalty on CPUs with SSE4 support
        Vector128<float> mat0 = Sse.LoadVector128(m_modelViewProjection + 0);
        Vector128<float> mat1 = Sse.LoadVector128(m_modelViewProjection + 4);
        Vector128<float> mat2 = Sse.LoadVector128(m_modelViewProjection + 8);
        Vector128<float> mat3 = Sse.LoadVector128(m_modelViewProjection + 12);

        Vector128<float> boundsMin = occluder.m_refMin.AsVector128();
        Vector128<float> boundsExtents = occluder.m_refMax.AsVector128() - boundsMin;

        // Bake integer => bounding box transform into matrix
        mat3 =
          Fma.MulAdd(mat0, Avx2.BroadcastScalarToVector128(boundsMin),
            Fma.MulAdd(mat1, Avx.Permute(boundsMin, 0b01_01_01_01),
              Fma.MulAdd(mat2, Avx.Permute(boundsMin, 0b10_10_10_10),
                mat3)));

        mat0 *= (Avx2.BroadcastScalarToVector128(boundsExtents) * Vector128.Create(1.0f / (2047ul << 21)));
        mat1 *= (Avx.Permute(boundsExtents, 0b01_01_01_01) * Vector128.Create(1.0f / (2047 << 10)));
        mat2 *= (Avx.Permute(boundsExtents, 0b10_10_10_10) * Vector128.Create(1.0f / 1023));

        // Bias X coordinate back into positive range
        mat3 = Fma.MulAdd(mat0, Vector128.Create((float)(1024ul << 21)), mat3);

        // Skew projection to correct bleeding of Y and Z into X due to lack of masking
        mat1 -= mat0;
        mat2 -= mat0;

        (mat0, mat1, mat2, mat3) = _MM_TRANSPOSE4_PS(mat0, mat1, mat2, mat3);

        // Due to linear relationship between Z and W, it's cheaper to compute Z from W later in the pipeline than using the full projection matrix up front
        float c0, c1;
        {
            float Za = mat2.GetElement(3);
            float Zb = Vector128.Dot(mat2, Vector128.Create((float)(1 << 21), 1 << 10, 1, 1));

            float Wa = mat3.GetElement(3);
            float Wb = Vector128.Dot(mat3, Vector128.Create((float)(1 << 21), 1 << 10, 1, 1));

            c0 = ((Za - Zb) / (Wa - Wb));
            c1 = Fma.MulAddNeg((Za - Zb) / (Wa - Wb), Wa, Za);
        }

        const int alignment = 256 / 8;
        const int stackBufferSize =
            alignment - 1 +
            sizeof(uint) * 8 * 4 + // uint[8] x 4
            sizeof(float) * 4 * 8 * 4 + // Vector128<float>[8] x 4
            sizeof(int) * 4 * 8 * 1 + // Vector128<int>[8] x 1
            sizeof(ushort) * 8 * 1; // ushort[8] x 1

        byte* stackBuffer = stackalloc byte[stackBufferSize];
        byte* alignedBuffer = (byte*)((nint)(stackBuffer + (alignment - 1)) & -alignment);

        uint* primModes = (uint*)alignedBuffer;
        alignedBuffer += sizeof(uint) * 8;

        uint* firstBlocks = (uint*)alignedBuffer;
        alignedBuffer += sizeof(uint) * 8;

        uint* rangesX = (uint*)alignedBuffer;
        alignedBuffer += sizeof(uint) * 8;

        uint* rangesY = (uint*)alignedBuffer;
        alignedBuffer += sizeof(uint) * 8;

        Vector128<float>* depthPlane = (Vector128<float>*)alignedBuffer;
        alignedBuffer += sizeof(Vector128<float>) * 8;

        Vector128<float>* edgeNormalsX = (Vector128<float>*)alignedBuffer;
        alignedBuffer += sizeof(Vector128<float>) * 8;

        Vector128<float>* edgeNormalsY = (Vector128<float>*)alignedBuffer;
        alignedBuffer += sizeof(Vector128<float>) * 8;

        Vector128<float>* edgeOffsets = (Vector128<float>*)alignedBuffer;
        alignedBuffer += sizeof(Vector128<float>) * 8;

        Vector128<int>* slopeLookups = (Vector128<int>*)alignedBuffer;
        alignedBuffer += sizeof(Vector128<int>) * 8;

        ushort* depthBounds = (ushort*)alignedBuffer;

        for (uint packetIdx = 0; packetIdx < packetCount; packetIdx += 4)
        {
            // Load data - only needed once per frame, so use streaming load
            Vector256<int> I0 = Avx2.LoadAlignedVector256NonTemporal((int*)(vertexData + packetIdx + 0));
            Vector256<int> I1 = Avx2.LoadAlignedVector256NonTemporal((int*)(vertexData + packetIdx + 1));
            Vector256<int> I2 = Avx2.LoadAlignedVector256NonTemporal((int*)(vertexData + packetIdx + 2));
            Vector256<int> I3 = Avx2.LoadAlignedVector256NonTemporal((int*)(vertexData + packetIdx + 3));

            // Vertex transformation - first W, then X & Y after camera plane culling, then Z after backface culling
            Vector256<float> Xf0 = Avx.ConvertToVector256Single(I0);
            Vector256<float> Xf1 = Avx.ConvertToVector256Single(I1);
            Vector256<float> Xf2 = Avx.ConvertToVector256Single(I2);
            Vector256<float> Xf3 = Avx.ConvertToVector256Single(I3);

            Vector256<float> Yf0 = Avx.ConvertToVector256Single(I0 & maskY);
            Vector256<float> Yf1 = Avx.ConvertToVector256Single(I1 & maskY);
            Vector256<float> Yf2 = Avx.ConvertToVector256Single(I2 & maskY);
            Vector256<float> Yf3 = Avx.ConvertToVector256Single(I3 & maskY);
                                                                   
            Vector256<float> Zf0 = Avx.ConvertToVector256Single(I0 & maskZ);
            Vector256<float> Zf1 = Avx.ConvertToVector256Single(I1 & maskZ);
            Vector256<float> Zf2 = Avx.ConvertToVector256Single(I2 & maskZ);
            Vector256<float> Zf3 = Avx.ConvertToVector256Single(I3 & maskZ);

            Vector256<float> mat00 = Vector256.Create(mat0.GetElement(0));
            Vector256<float> mat01 = Vector256.Create(mat0.GetElement(1));
            Vector256<float> mat02 = Vector256.Create(mat0.GetElement(2));
            Vector256<float> mat03 = Vector256.Create(mat0.GetElement(3));

            Vector256<float> X0 = Fma.MulAdd(Xf0, mat00, Fma.MulAdd(Yf0, mat01, Fma.MulAdd(Zf0, mat02, mat03)));
            Vector256<float> X1 = Fma.MulAdd(Xf1, mat00, Fma.MulAdd(Yf1, mat01, Fma.MulAdd(Zf1, mat02, mat03)));
            Vector256<float> X2 = Fma.MulAdd(Xf2, mat00, Fma.MulAdd(Yf2, mat01, Fma.MulAdd(Zf2, mat02, mat03)));
            Vector256<float> X3 = Fma.MulAdd(Xf3, mat00, Fma.MulAdd(Yf3, mat01, Fma.MulAdd(Zf3, mat02, mat03)));

            Vector256<float> mat10 = Vector256.Create(mat1.GetElement(0));
            Vector256<float> mat11 = Vector256.Create(mat1.GetElement(1));
            Vector256<float> mat12 = Vector256.Create(mat1.GetElement(2));
            Vector256<float> mat13 = Vector256.Create(mat1.GetElement(3));

            Vector256<float> Y0 = Fma.MulAdd(Xf0, mat10, Fma.MulAdd(Yf0, mat11, Fma.MulAdd(Zf0, mat12, mat13)));
            Vector256<float> Y1 = Fma.MulAdd(Xf1, mat10, Fma.MulAdd(Yf1, mat11, Fma.MulAdd(Zf1, mat12, mat13)));
            Vector256<float> Y2 = Fma.MulAdd(Xf2, mat10, Fma.MulAdd(Yf2, mat11, Fma.MulAdd(Zf2, mat12, mat13)));
            Vector256<float> Y3 = Fma.MulAdd(Xf3, mat10, Fma.MulAdd(Yf3, mat11, Fma.MulAdd(Zf3, mat12, mat13)));

            Vector256<float> mat30 = Vector256.Create(mat3.GetElement(0));
            Vector256<float> mat31 = Vector256.Create(mat3.GetElement(1));
            Vector256<float> mat32 = Vector256.Create(mat3.GetElement(2));
            Vector256<float> mat33 = Vector256.Create(mat3.GetElement(3));

            Vector256<float> W0 = Fma.MulAdd(Xf0, mat30, Fma.MulAdd(Yf0, mat31, Fma.MulAdd(Zf0, mat32, mat33)));
            Vector256<float> W1 = Fma.MulAdd(Xf1, mat30, Fma.MulAdd(Yf1, mat31, Fma.MulAdd(Zf1, mat32, mat33)));
            Vector256<float> W2 = Fma.MulAdd(Xf2, mat30, Fma.MulAdd(Yf2, mat31, Fma.MulAdd(Zf2, mat32, mat33)));
            Vector256<float> W3 = Fma.MulAdd(Xf3, mat30, Fma.MulAdd(Yf3, mat31, Fma.MulAdd(Zf3, mat32, mat33)));

            Vector256<float> invW0, invW1, invW2, invW3;
            // Clamp W and invert
            if (T.PossiblyNearClipped)
            {
                Vector256<float> lowerBound = Vector256.Create((float)-maxInvW);
                Vector256<float> upperBound = Vector256.Create((float)+maxInvW);
                invW0 = Avx.Min(upperBound, Avx.Max(lowerBound, Avx.Reciprocal(W0)));
                invW1 = Avx.Min(upperBound, Avx.Max(lowerBound, Avx.Reciprocal(W1)));
                invW2 = Avx.Min(upperBound, Avx.Max(lowerBound, Avx.Reciprocal(W2)));
                invW3 = Avx.Min(upperBound, Avx.Max(lowerBound, Avx.Reciprocal(W3)));
            }
            else
            {
                invW0 = Avx.Reciprocal(W0);
                invW1 = Avx.Reciprocal(W1);
                invW2 = Avx.Reciprocal(W2);
                invW3 = Avx.Reciprocal(W3);
            }

            // Round to integer coordinates to improve culling of zero-area triangles
            Vector256<float> roundFactor = Vector256.Create(0.125f);
            Vector256<float> x0 = Avx.RoundToNearestInteger(X0 * invW0) * roundFactor;
            Vector256<float> x1 = Avx.RoundToNearestInteger(X1 * invW1) * roundFactor;
            Vector256<float> x2 = Avx.RoundToNearestInteger(X2 * invW2) * roundFactor;
            Vector256<float> x3 = Avx.RoundToNearestInteger(X3 * invW3) * roundFactor;

            Vector256<float> y0 = Avx.RoundToNearestInteger(Y0 * invW0) * roundFactor;
            Vector256<float> y1 = Avx.RoundToNearestInteger(Y1 * invW1) * roundFactor;
            Vector256<float> y2 = Avx.RoundToNearestInteger(Y2 * invW2) * roundFactor;
            Vector256<float> y3 = Avx.RoundToNearestInteger(Y3 * invW3) * roundFactor;

            // Compute unnormalized edge directions
            Vector256<float> edgeNormalsX0 = y1 - y0;
            Vector256<float> edgeNormalsX1 = y2 - y1;
            Vector256<float> edgeNormalsX2 = y3 - y2;
            Vector256<float> edgeNormalsX3 = y0 - y3;

            Vector256<float> edgeNormalsY0 = x0 - x1;
            Vector256<float> edgeNormalsY1 = x1 - x2;
            Vector256<float> edgeNormalsY2 = x2 - x3;
            Vector256<float> edgeNormalsY3 = x3 - x0;

            Vector256<float> area0 = Fma.MulSub(edgeNormalsX0, edgeNormalsY1, (edgeNormalsX1 * edgeNormalsY0));
            Vector256<float> area1 = Fma.MulSub(edgeNormalsX1, edgeNormalsY2, (edgeNormalsX2 * edgeNormalsY1));
            Vector256<float> area2 = Fma.MulSub(edgeNormalsX2, edgeNormalsY3, (edgeNormalsX3 * edgeNormalsY2));
            Vector256<float> area3 = (area0 + area2) - area1;

            Vector256<float> minusZero256 = Vector256.Create(-0.0f);

            Vector256<float> wSign0, wSign1, wSign2, wSign3;
            if (T.PossiblyNearClipped)
            {
                wSign0 = invW0 & minusZero256;
                wSign1 = invW1 & minusZero256;
                wSign2 = invW2 & minusZero256;
                wSign3 = invW3 & minusZero256;
            }
            else
            {
                wSign0 = Vector256<float>.Zero;
                wSign1 = Vector256<float>.Zero;
                wSign2 = Vector256<float>.Zero;
                wSign3 = Vector256<float>.Zero;
            }

            // Compute signs of areas. We treat 0 as negative as this allows treating primitives with zero area as backfacing.
            Vector256<float> areaSign0, areaSign1, areaSign2, areaSign3;
            if (T.PossiblyNearClipped)
            {
                // Flip areas for each vertex with W < 0. This needs to be done before comparison against 0 rather than afterwards to make sure zero-are triangles are handled correctly.
                areaSign0 = Avx.Compare(((area0 ^ wSign0) ^ (wSign1 ^ wSign2)), Vector256<float>.Zero, _CMP_LE_OQ);
                areaSign1 = minusZero256 & Avx.Compare((area1 ^ wSign1) ^ (wSign2 ^ wSign3), Vector256<float>.Zero, _CMP_LE_OQ);
                areaSign2 = minusZero256 & Avx.Compare((area2 ^ wSign0) ^ (wSign2 ^ wSign3), Vector256<float>.Zero, _CMP_LE_OQ);
                areaSign3 = minusZero256 & Avx.Compare((area3 ^ wSign1) ^ (wSign0 ^ wSign3), Vector256<float>.Zero, _CMP_LE_OQ);
            }
            else
            {
                areaSign0 = Avx.Compare(area0, Vector256<float>.Zero, _CMP_LE_OQ);
                areaSign1 = minusZero256 & Avx.Compare(area1, Vector256<float>.Zero, _CMP_LE_OQ);
                areaSign2 = minusZero256 & Avx.Compare(area2, Vector256<float>.Zero, _CMP_LE_OQ);
                areaSign3 = minusZero256 & Avx.Compare(area3, Vector256<float>.Zero, _CMP_LE_OQ);
            }

            Vector256<int> config = Avx2.Or(
              ((areaSign3.AsInt32() >>> 28) | (areaSign2.AsInt32() >>> 29)),
              ((areaSign1.AsInt32() >>> 30) | (areaSign0.AsInt32() >>> 31)));

            if (T.PossiblyNearClipped)
            {
                config = Avx2.Or(config,
                  Avx2.Or(
                    ((wSign3.AsInt32() >>> 24) | (wSign2.AsInt32() >>> 25)),
                    ((wSign1.AsInt32() >>> 26) | (wSign0.AsInt32() >>> 27))));
            }

            Vector256<int> modes;
            fixed (PrimitiveMode* modeTablePtr = modeTable)
            {
                modes = Avx2.GatherVector256((int*)modeTablePtr, config, 1) & Vector256.Create(0xff);
            }

            if (Avx.TestZ(modes, modes))
            {
                continue;
            }

            Vector256<int> primitiveValid = Avx2.CompareGreaterThan(modes, Vector256<int>.Zero);

            Avx.StoreAligned((int*)primModes, modes);

            Vector256<float> minFx, minFy, maxFx, maxFy;

            if (T.PossiblyNearClipped)
            {
                // Clipless bounding box computation
                Vector256<float> infP = Vector256.Create(+10000.0f);
                Vector256<float> infN = Vector256.Create(-10000.0f);

                // Find interval of points with W > 0
                Vector256<float> minPx0 = Avx.BlendVariable(x0, infP, wSign0);
                Vector256<float> minPx1 = Avx.BlendVariable(x1, infP, wSign1);
                Vector256<float> minPx2 = Avx.BlendVariable(x2, infP, wSign2);
                Vector256<float> minPx3 = Avx.BlendVariable(x3, infP, wSign3);

                Vector256<float> minPx = Avx.Min(
                  Avx.Min(minPx0, minPx1),
                  Avx.Min(minPx2, minPx3));

                Vector256<float> minPy0 = Avx.BlendVariable(y0, infP, wSign0);
                Vector256<float> minPy1 = Avx.BlendVariable(y1, infP, wSign1);
                Vector256<float> minPy2 = Avx.BlendVariable(y2, infP, wSign2);
                Vector256<float> minPy3 = Avx.BlendVariable(y3, infP, wSign3);

                Vector256<float> minPy = Avx.Min(
                  Avx.Min(minPy0, minPy1),
                  Avx.Min(minPy2, minPy3));

                Vector256<float> maxPx0 = minPx0 ^ wSign0;
                Vector256<float> maxPx1 = minPx1 ^ wSign1;
                Vector256<float> maxPx2 = minPx2 ^ wSign2;
                Vector256<float> maxPx3 = minPx3 ^ wSign3;

                Vector256<float> maxPx = Avx.Max(
                  Avx.Max(maxPx0, maxPx1),
                  Avx.Max(maxPx2, maxPx3));

                Vector256<float> maxPy0 = minPy0 ^ wSign0;
                Vector256<float> maxPy1 = minPy1 ^ wSign1;
                Vector256<float> maxPy2 = minPy2 ^ wSign2;
                Vector256<float> maxPy3 = minPy3 ^ wSign3;

                Vector256<float> maxPy = Avx.Max(
                  Avx.Max(maxPy0, maxPy1),
                  Avx.Max(maxPy2, maxPy3));

                // Find interval of points with W < 0
                Vector256<float> minNx0 = Avx.BlendVariable(infP, x0, wSign0);
                Vector256<float> minNx1 = Avx.BlendVariable(infP, x1, wSign1);
                Vector256<float> minNx2 = Avx.BlendVariable(infP, x2, wSign2);
                Vector256<float> minNx3 = Avx.BlendVariable(infP, x3, wSign3);

                Vector256<float> minNx = Avx.Min(
                  Avx.Min(minNx0, minNx1),
                  Avx.Min(minNx2, minNx3));

                Vector256<float> minNy0 = Avx.BlendVariable(infP, y0, wSign0);
                Vector256<float> minNy1 = Avx.BlendVariable(infP, y1, wSign1);
                Vector256<float> minNy2 = Avx.BlendVariable(infP, y2, wSign2);
                Vector256<float> minNy3 = Avx.BlendVariable(infP, y3, wSign3);

                Vector256<float> minNy = Avx.Min(
                  Avx.Min(minNy0, minNy1),
                  Avx.Min(minNy2, minNy3));

                Vector256<float> maxNx0 = Avx.BlendVariable(infN, x0, wSign0);
                Vector256<float> maxNx1 = Avx.BlendVariable(infN, x1, wSign1);
                Vector256<float> maxNx2 = Avx.BlendVariable(infN, x2, wSign2);
                Vector256<float> maxNx3 = Avx.BlendVariable(infN, x3, wSign3);

                Vector256<float> maxNx = Avx.Max(
                  Avx.Max(maxNx0, maxNx1),
                  Avx.Max(maxNx2, maxNx3));

                Vector256<float> maxNy0 = Avx.BlendVariable(infN, y0, wSign0);
                Vector256<float> maxNy1 = Avx.BlendVariable(infN, y1, wSign1);
                Vector256<float> maxNy2 = Avx.BlendVariable(infN, y2, wSign2);
                Vector256<float> maxNy3 = Avx.BlendVariable(infN, y3, wSign3);

                Vector256<float> maxNy = Avx.Max(
                  Avx.Max(maxNy0, maxNy1),
                  Avx.Max(maxNy2, maxNy3));

                // Include interval bounds resp. infinity depending on ordering of intervals
                Vector256<float> incAx = Avx.BlendVariable(minPx, infN, Avx.Compare(maxNx, minPx, _CMP_GT_OQ));
                Vector256<float> incAy = Avx.BlendVariable(minPy, infN, Avx.Compare(maxNy, minPy, _CMP_GT_OQ));

                Vector256<float> incBx = Avx.BlendVariable(maxPx, infP, Avx.Compare(maxPx, minNx, _CMP_GT_OQ));
                Vector256<float> incBy = Avx.BlendVariable(maxPy, infP, Avx.Compare(maxPy, minNy, _CMP_GT_OQ));

                minFx = Avx.Min(incAx, incBx);
                minFy = Avx.Min(incAy, incBy);

                maxFx = Avx.Max(incAx, incBx);
                maxFy = Avx.Max(incAy, incBy);
            }
            else
            {
                // Standard bounding box inclusion
                minFx = Avx.Min(Avx.Min(x0, x1), Avx.Min(x2, x3));
                minFy = Avx.Min(Avx.Min(y0, y1), Avx.Min(y2, y3));

                maxFx = Avx.Max(Avx.Max(x0, x1), Avx.Max(x2, x3));
                maxFy = Avx.Max(Avx.Max(y0, y1), Avx.Max(y2, y3));
            }

            // Clamp and round
            Vector256<int> minX, minY, maxX, maxY;
            minX = Avx2.Max(Avx.ConvertToVector256Int32WithTruncation(minFx + Vector256.Create(4.9999f / 8.0f)), Vector256<int>.Zero);
            minY = Avx2.Max(Avx.ConvertToVector256Int32WithTruncation(minFy + Vector256.Create(4.9999f / 8.0f)), Vector256<int>.Zero);
            maxX = Avx2.Min(Avx.ConvertToVector256Int32WithTruncation(maxFx + Vector256.Create(11.0f / 8.0f)), Vector256.Create((int)m_blocksX));
            maxY = Avx2.Min(Avx.ConvertToVector256Int32WithTruncation(maxFy + Vector256.Create(11.0f / 8.0f)), Vector256.Create((int)m_blocksY));

            // Check overlap between bounding box and frustum
            Vector256<int> inFrustum = Avx2.CompareGreaterThan(maxX, minX) & Avx2.CompareGreaterThan(maxY, minY);
            Vector256<int> overlappedPrimitiveValid = inFrustum & primitiveValid;

            if (Avx.TestZ(overlappedPrimitiveValid, overlappedPrimitiveValid))
            {
                continue;
            }

            uint validMask = (uint)Avx.MoveMask(overlappedPrimitiveValid.AsSingle());

            // Convert bounds from [min, max] to [min, range]
            Vector256<int> rangeX = maxX - minX;
            Vector256<int> rangeY = maxY - minY;

            // Compute Z from linear relation with 1/W
            Vector256<float> C0 = Vector256.Create(c0);
            Vector256<float> C1 = Vector256.Create(c1);
            Vector256<float> z0, z1, z2, z3;
            z0 = Fma.MulAdd(invW0, C1, C0);
            z1 = Fma.MulAdd(invW1, C1, C0);
            z2 = Fma.MulAdd(invW2, C1, C0);
            z3 = Fma.MulAdd(invW3, C1, C0);

            Vector256<float> maxZ = Avx.Max(Avx.Max(z0, z1), Avx.Max(z2, z3));

            // If any W < 0, assume maxZ = 1 (effectively disabling Hi-Z)
            if (T.PossiblyNearClipped)
            {
                maxZ = Avx.BlendVariable(maxZ, Vector256.Create(1.0f), (wSign0 | wSign1) | (wSign2 | wSign3));
            }

            Vector128<ushort> packedDepthBounds = packDepthPremultiplied(maxZ);

            Sse2.StoreAligned(depthBounds, packedDepthBounds);

            // Compute screen space depth plane
            Vector256<float> greaterArea = Avx.Compare(
                Vector256.AndNot(area0, minusZero256), 
                Vector256.AndNot(area2, minusZero256), 
                _CMP_LT_OQ);

            // Force triangle area to be picked in the relevant mode.
            Vector256<float> modeTriangle0 = Avx2.CompareEqual(modes, Vector256.Create((int)PrimitiveMode.Triangle0)).AsSingle();
            Vector256<float> modeTriangle1 = Avx2.CompareEqual(modes, Vector256.Create((int)PrimitiveMode.Triangle1)).AsSingle();
            greaterArea = Vector256.AndNot(modeTriangle1 | greaterArea, modeTriangle0);


            Vector256<float> invArea;
            if (T.PossiblyNearClipped)
            {
                // Do a precise divison to reduce error in depth plane. Note that the area computed here
                // differs from the rasterized region if W < 0, so it can be very small for large covered screen regions.
                invArea = Vector256.Create(1.0f) / Avx.BlendVariable(area0, area2, greaterArea);
            }
            else
            {
                invArea = Avx.Reciprocal(Avx.BlendVariable(area0, area2, greaterArea));
            }

            Vector256<float> z12 = z1 - z2;
            Vector256<float> z20 = z2 - z0;
            Vector256<float> z30 = z3 - z0;


            Vector256<float> edgeNormalsX4 = y0 - y2;
            Vector256<float> edgeNormalsY4 = x2 - x0;

            Vector256<float> depthPlane0, depthPlane1, depthPlane2;
            depthPlane1 = (invArea * Avx.BlendVariable(Fma.MulSub(z20, edgeNormalsX1, (z12 * edgeNormalsX4)), Fma.MulAddNeg(z20, edgeNormalsX3, (z30 * edgeNormalsX4)), greaterArea));
            depthPlane2 = (invArea * Avx.BlendVariable(Fma.MulSub(z20, edgeNormalsY1, (z12 * edgeNormalsY4)), Fma.MulAddNeg(z20, edgeNormalsY3, (z30 * edgeNormalsY4)), greaterArea));

            x0 -= Avx.ConvertToVector256Single(minX);
            y0 -= Avx.ConvertToVector256Single(minY);

            depthPlane0 = Fma.MulAddNeg(x0, depthPlane1, Fma.MulAddNeg(y0, depthPlane2, z0));

            // If mode == Triangle0, replace edge 2 with edge 4; if mode == Triangle1, replace edge 0 with edge 4
            edgeNormalsX2 = Avx.BlendVariable(edgeNormalsX2, edgeNormalsX4, modeTriangle0);
            edgeNormalsY2 = Avx.BlendVariable(edgeNormalsY2, edgeNormalsY4, modeTriangle0);
            edgeNormalsX0 = Avx.BlendVariable(edgeNormalsX0, minusZero256 ^ edgeNormalsX4, modeTriangle1);
            edgeNormalsY0 = Avx.BlendVariable(edgeNormalsY0, minusZero256 ^ edgeNormalsY4, modeTriangle1);

            // Flip edges if W < 0
            Vector256<float> edgeFlipMask0, edgeFlipMask1, edgeFlipMask2, edgeFlipMask3;
            if (T.PossiblyNearClipped)
            {
                edgeFlipMask0 = wSign0 ^ Avx.BlendVariable(wSign1, wSign2, modeTriangle1);
                edgeFlipMask1 = wSign1 ^ wSign2;
                edgeFlipMask2 = wSign2 ^ Avx.BlendVariable(wSign3, wSign0, modeTriangle0);
                edgeFlipMask3 = wSign0 ^ wSign3;
            }
            else
            {
                edgeFlipMask0 = Vector256<float>.Zero;
                edgeFlipMask1 = Vector256<float>.Zero;
                edgeFlipMask2 = Vector256<float>.Zero;
                edgeFlipMask3 = Vector256<float>.Zero;
            }

            // Normalize edge equations for lookup
            (edgeNormalsX0, edgeNormalsY0) = normalizeEdge<T>(edgeNormalsX0, edgeNormalsY0, edgeFlipMask0);
            (edgeNormalsX1, edgeNormalsY1) = normalizeEdge<T>(edgeNormalsX1, edgeNormalsY1, edgeFlipMask1);
            (edgeNormalsX2, edgeNormalsY2) = normalizeEdge<T>(edgeNormalsX2, edgeNormalsY2, edgeFlipMask2);
            (edgeNormalsX3, edgeNormalsY3) = normalizeEdge<T>(edgeNormalsX3, edgeNormalsY3, edgeFlipMask3);

            const float maxOffset = -minEdgeOffset;
            Vector256<float> add256 = Vector256.Create(0.5f - minEdgeOffset * (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
            Vector256<float> edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3;

            edgeOffsets0 = Fma.MulAddNeg(x0, edgeNormalsX0, Fma.MulAddNeg(y0, edgeNormalsY0, add256));
            edgeOffsets1 = Fma.MulAddNeg(x1, edgeNormalsX1, Fma.MulAddNeg(y1, edgeNormalsY1, add256));
            edgeOffsets2 = Fma.MulAddNeg(x2, edgeNormalsX2, Fma.MulAddNeg(y2, edgeNormalsY2, add256));
            edgeOffsets3 = Fma.MulAddNeg(x3, edgeNormalsX3, Fma.MulAddNeg(y3, edgeNormalsY3, add256));

            edgeOffsets1 = Fma.MulAdd(Avx.ConvertToVector256Single(minX), edgeNormalsX1, edgeOffsets1);
            edgeOffsets2 = Fma.MulAdd(Avx.ConvertToVector256Single(minX), edgeNormalsX2, edgeOffsets2);
            edgeOffsets3 = Fma.MulAdd(Avx.ConvertToVector256Single(minX), edgeNormalsX3, edgeOffsets3);

            edgeOffsets1 = Fma.MulAdd(Avx.ConvertToVector256Single(minY), edgeNormalsY1, edgeOffsets1);
            edgeOffsets2 = Fma.MulAdd(Avx.ConvertToVector256Single(minY), edgeNormalsY2, edgeOffsets2);
            edgeOffsets3 = Fma.MulAdd(Avx.ConvertToVector256Single(minY), edgeNormalsY3, edgeOffsets3);

            // Quantize slopes
            Vector256<int> slopeLookups0 = quantizeSlopeLookup(edgeNormalsX0, edgeNormalsY0);
            Vector256<int> slopeLookups1 = quantizeSlopeLookup(edgeNormalsX1, edgeNormalsY1);
            Vector256<int> slopeLookups2 = quantizeSlopeLookup(edgeNormalsX2, edgeNormalsY2);
            Vector256<int> slopeLookups3 = quantizeSlopeLookup(edgeNormalsX3, edgeNormalsY3);

            Vector256<int> firstBlockIdx = Avx2.Add(Avx2.MultiplyLow(minY.AsInt16(), Vector256.Create((int)m_blocksX).AsInt16()).AsInt32(), minX);

            Avx.StoreAligned((int*)firstBlocks, firstBlockIdx);

            Avx.StoreAligned((int*)rangesX, rangeX);

            Avx.StoreAligned((int*)rangesY, rangeY);

            // Transpose into AoS
            transpose256(depthPlane0, depthPlane1, depthPlane2, Vector256<float>.Zero, depthPlane);

            transpose256(edgeNormalsX0, edgeNormalsX1, edgeNormalsX2, edgeNormalsX3, edgeNormalsX);

            transpose256(edgeNormalsY0, edgeNormalsY1, edgeNormalsY2, edgeNormalsY3, edgeNormalsY);

            transpose256(edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3, edgeOffsets);

            transpose256i(slopeLookups0, slopeLookups1, slopeLookups2, slopeLookups3, slopeLookups);

            rasterizeLoop(
                validMask,
                depthBounds,
                depthPlane,
                slopeLookups,
                edgeNormalsX,
                edgeNormalsY,
                edgeOffsets,
                firstBlocks,
                rangesX,
                rangesY,
                primModes,
                T.PossiblyNearClipped);
        }
    }

    private void rasterizeLoop(
        uint validMask,
        ushort* depthBounds,
        Vector128<float>* depthPlane,
        Vector128<int>* slopeLookups,
        Vector128<float>* edgeNormalsX,
        Vector128<float>* edgeNormalsY,
        Vector128<float>* edgeOffsets,
        uint* firstBlocks,
        uint* rangesX,
        uint* rangesY,
        uint* primModes,
        bool possiblyNearClipped)
    {
        // Fetch data pointers since we'll manually strength-reduce memory arithmetic
        ulong* pTable = (ulong*)m_precomputedRasterTables.DangerousGetHandle();
        ushort* pHiZBuffer = m_hiZ;
        Vector128<int>* pDepthBuffer = m_depthBuffer;

        const float depthSamplePos = -0.5f + 1.0f / 16.0f;

        Vector256<float> depthSamplePosFactor1 = Vector256.Create(
            depthSamplePos + 0.0f, depthSamplePos + 0.125f, depthSamplePos + 0.25f, depthSamplePos + 0.375f,
            depthSamplePos + 0.0f, depthSamplePos + 0.125f, depthSamplePos + 0.25f, depthSamplePos + 0.375f);

        Vector256<float> depthSamplePosFactor2 = Vector256.Create(
            depthSamplePos + 0.0f, depthSamplePos + 0.0f, depthSamplePos + 0.0f, depthSamplePos + 0.0f,
            depthSamplePos + 0.125f, depthSamplePos + 0.125f, depthSamplePos + 0.125f, depthSamplePos + 0.125f);

        // Loop over set bits
        while (validMask != 0)
        {
            uint primitiveIdx = (uint)BitOperations.TrailingZeroCount(validMask);

            // Clear lowest set bit in mask
            validMask &= validMask - 1;

            uint primitiveIdxTransposed = ((primitiveIdx << 1) & 7) | (primitiveIdx >> 2);

            // Extract and prepare per-primitive data
            ushort primitiveMaxZ = depthBounds[primitiveIdx];

            Vector128<float> depthDxy = Sse.LoadAlignedVector128((float*) (depthPlane + primitiveIdxTransposed));
            Vector256<float> depthDx = Vector256.Create(depthDxy.GetElement(1));
            Vector256<float> depthDy = Vector256.Create(depthDxy.GetElement(2));

            Vector256<float> lineDepth =
              Fma.MulAdd(depthDx, depthSamplePosFactor1,
                Fma.MulAdd(depthDy, depthSamplePosFactor2,
                  Avx2.BroadcastScalarToVector256(depthPlane[primitiveIdxTransposed])));

            Vector128<int> slopeLookup = Sse2.LoadAlignedVector128((int*)(slopeLookups + primitiveIdxTransposed));
            Vector128<float> edgeNormalX = Sse.LoadAlignedVector128((float*)(edgeNormalsX + primitiveIdxTransposed));
            Vector128<float> edgeNormalY = Sse.LoadAlignedVector128((float*)(edgeNormalsY + primitiveIdxTransposed));
            Vector128<float> lineOffset = Sse.LoadAlignedVector128((float*)(edgeOffsets + primitiveIdxTransposed));

            uint blocksX = m_blocksX;

            uint firstBlock = firstBlocks[primitiveIdx];
            uint blockRangeX = rangesX[primitiveIdx];
            uint blockRangeY = rangesY[primitiveIdx];

            ushort* pPrimitiveHiZ = pHiZBuffer + firstBlock;
            Vector256<int>* pPrimitiveOut = (Vector256<int>*)pDepthBuffer + 4 * firstBlock;

            uint primitiveMode = primModes[primitiveIdx];

            for (uint blockY = 0;
              blockY < blockRangeY;
              ++blockY,
              pPrimitiveHiZ += blocksX,
              pPrimitiveOut += 4 * blocksX,
              lineDepth += depthDy,
              lineOffset += edgeNormalY)
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
                  depth = (depthDx + depth),
                  offset = (edgeNormalX + offset))
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
                        Vector128<float> anyOffsetOutsideMask = Sse.CompareGreaterThanOrEqual(offset, Vector128.Create((float)(OFFSET_QUANTIZATION_FACTOR - 1)));
                        if (!Avx.TestZ(anyOffsetOutsideMask, anyOffsetOutsideMask))
                        {
                            if (anyBlockHit)
                            {
                                // Convexity implies we won't hit another block in this row and can skip to the next line.
                                break;
                            }
                            continue;
                        }

                        anyBlockHit = true;

                        Vector128<int> offsetClamped = Sse41.Max(Sse2.ConvertToVector128Int32WithTruncation(offset), Vector128<int>.Zero);

                        Vector128<int> lookup = (slopeLookup | offsetClamped);

                        // Generate block mask
                        ulong A = pTable[(uint)lookup.GetElement(0)];
                        ulong B = pTable[(uint)lookup.GetElement(1)];
                        ulong C = pTable[(uint)lookup.GetElement(2)];
                        ulong D = pTable[(uint)lookup.GetElement(3)];

                        blockMask = A & B & C & D;

                        // It is possible but very unlikely that blockMask == 0 if all A,B,C,D != 0 according to the conservative test above, so we skip the additional branch here.
                    }
                    else
                    {
                        Vector128<int> offsetClamped = Sse41.Min(Sse41.Max(Sse2.ConvertToVector128Int32WithTruncation(offset), Vector128<int>.Zero), Vector128.Create(OFFSET_QUANTIZATION_FACTOR - 1));
                        Vector128<int> lookup = (slopeLookup | offsetClamped);

                        // Generate block mask
                        ulong A = pTable[(uint)lookup.GetElement(0)];
                        ulong B = pTable[(uint)lookup.GetElement(1)];
                        ulong C = pTable[(uint)lookup.GetElement(2)];
                        ulong D = pTable[(uint)lookup.GetElement(3)];

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
                                blockMask = (A | D) & B & C;
                                break;

                            default:
                                // Case ConcaveCenter can only occur if any W < 0
                                if (possiblyNearClipped)
                                {
                                    // case ConcaveCenter:			// < 1e-6%
                                    blockMask = (A & B) | (C & D);
                                    break;
                                }
                                // Fall-through
                                goto case (uint)PrimitiveMode.ConcaveLeft;

                            case (uint)PrimitiveMode.ConcaveLeft:           // 0.01-0.6%
                                blockMask = A & D & (B | C);
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
                    Vector256<float> depth1 = Fma.MulAdd(depthDx, Vector256.Create(0.5f), depth0);
                    Vector256<float> depth8 = depthDy + depth0;
                    Vector256<float> depth9 = depthDy + depth1;

                    // Pack depth
                    Vector256<ushort> d0 = packDepthPremultiplied(depth0, depth1);
                    Vector256<ushort> d4 = packDepthPremultiplied(depth8, depth9);

                    // Interpolate remaining values in packed space
                    Vector256<ushort> d2 = Avx2.Average(d0, d4);
                    Vector256<ushort> d1 = Avx2.Average(d0, d2);
                    Vector256<ushort> d3 = Avx2.Average(d2, d4);

                    // Not all pixels covered - mask depth 
                    if (blockMask != 0xffff_ffff_ffff_ffff)
                    {
                        Vector128<int> A = Vector128.CreateScalar((long)blockMask).AsInt32();
                        Vector128<int> B = (A.AsInt16() << 4).AsInt32();
                        Vector256<int> C = Avx2.InsertVector128(A.ToVector256Unsafe(), B, 1);
                        Vector256<short> rowMask = Avx2.UnpackLow(C.AsByte(), C.AsByte()).AsInt16();

                        d0 = Avx2.BlendVariable(Vector256<byte>.Zero, d0.AsByte(), (rowMask << 3).AsByte()).AsUInt16();
                        d1 = Avx2.BlendVariable(Vector256<byte>.Zero, d1.AsByte(), (rowMask << 2).AsByte()).AsUInt16();
                        d2 = Avx2.BlendVariable(Vector256<byte>.Zero, d2.AsByte(), (rowMask + rowMask).AsByte()).AsUInt16();
                        d3 = Avx2.BlendVariable(Vector256<byte>.Zero, d3.AsByte(), rowMask.AsByte()).AsUInt16();
                    }

                    // Test fast clear flag
                    if (hiZ != 1)
                    {
                        // Merge depth values
                        d0 = Avx2.Max(Avx.LoadAlignedVector256((ushort*)(@out + 0)), d0);
                        d1 = Avx2.Max(Avx.LoadAlignedVector256((ushort*)(@out + 1)), d1);
                        d2 = Avx2.Max(Avx.LoadAlignedVector256((ushort*)(@out + 2)), d2);
                        d3 = Avx2.Max(Avx.LoadAlignedVector256((ushort*)(@out + 3)), d3);
                    }

                    // Store back new depth
                    Avx.StoreAligned((ushort*)(@out + 0), d0);
                    Avx.StoreAligned((ushort*)(@out + 1), d1);
                    Avx.StoreAligned((ushort*)(@out + 2), d2);
                    Avx.StoreAligned((ushort*)(@out + 3), d3);

                    // Update HiZ
                    Vector256<ushort> newMinZ = Avx2.Min(Avx2.Min(d0, d1), Avx2.Min(d2, d3));
                    Vector128<int> newMinZ16 = Sse41.MinHorizontal(Sse41.Min(newMinZ.GetLower(), newMinZ.GetUpper())).AsInt32();

                    *pBlockRowHiZ = (ushort)(uint)Sse2.ConvertToInt32(newMinZ16);
                }
            }
        }
    }
}