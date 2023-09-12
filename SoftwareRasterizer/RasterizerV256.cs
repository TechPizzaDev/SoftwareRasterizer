using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

using static VectorMath;

public unsafe class RasterizerV256<Fma> : Rasterizer
    where Fma : IFusedMultiplyAdd128, IFusedMultiplyAdd256
{
    private const FloatComparisonMode _CMP_LT_OQ = FloatComparisonMode.OrderedLessThanNonSignaling;
    private const FloatComparisonMode _CMP_LE_OQ = FloatComparisonMode.OrderedLessThanOrEqualNonSignaling;
    private const FloatComparisonMode _CMP_GT_OQ = FloatComparisonMode.OrderedGreaterThanNonSignaling;

    private const int Alignment = 256 / 8; // sizeof(Vector256<>)

    public RasterizerV256(RasterizationTable rasterizationTable, uint width, uint height) :
        base(rasterizationTable, width, height, Alignment)
    {
    }

    public static RasterizerV256<Fma> Create(RasterizationTable rasterizationTable, uint width, uint height)
    {
        bool success = false;
        rasterizationTable.DangerousAddRef(ref success);
        if (success)
        {
            return new RasterizerV256<Fma>(rasterizationTable, width, height);
        }
        throw new ObjectDisposedException(rasterizationTable.GetType().Name);
    }

    public override unsafe void setModelViewProjection(float* matrix)
    {
        Vector128<float> mat0 = Vector128.Load(matrix + 0);
        Vector128<float> mat1 = Vector128.Load(matrix + 4);
        Vector128<float> mat2 = Vector128.Load(matrix + 8);
        Vector128<float> mat3 = Vector128.Load(matrix + 12);

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Store rows
        mat0.StoreAligned(m_modelViewProjectionRaw + 0);
        mat1.StoreAligned(m_modelViewProjectionRaw + 4);
        mat2.StoreAligned(m_modelViewProjectionRaw + 8);
        mat3.StoreAligned(m_modelViewProjectionRaw + 12);

        // Bake viewport transform into matrix and 6shift by half a block
        mat0 = Vector128.Multiply(Vector128.Add(mat0, mat3), Vector128.Create(m_width * 0.5f - 4.0f));
        mat1 = Vector128.Multiply(Vector128.Add(mat1, mat3), Vector128.Create(m_height * 0.5f - 4.0f));

        // Map depth from [-1, 1] to [bias, 0]
        mat2 = Vector128.Multiply(Vector128.Subtract(mat3, mat2), Vector128.Create(0.5f * floatCompressionBias));

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Store prebaked cols
        mat0.StoreAligned(m_modelViewProjection + 0);
        mat1.StoreAligned(m_modelViewProjection + 4);
        mat2.StoreAligned(m_modelViewProjection + 8);
        mat3.StoreAligned(m_modelViewProjection + 12);
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
            clearValue.StoreAligned((int*)pHiZ);
            pHiZ++;
        }
    }

    public override bool queryVisibility(Vector4 vBoundsMin, Vector4 vBoundsMax, out bool needsClipping)
    {
        Vector128<float> boundsMin = vBoundsMin.AsVector128();
        Vector128<float> boundsMax = vBoundsMax.AsVector128();

        // Frustum cull
        Vector128<float> extents = Vector128.Subtract(boundsMax, boundsMin);
        Vector128<float> center = Vector128.Add(boundsMax, boundsMin); // Bounding box center times 2 - but since W = 2, the plane equations work out correctly
        Vector128<float> minusZero = Vector128.Create(-0.0f);

        Vector128<float> row0 = Vector128.LoadAligned(m_modelViewProjectionRaw + 0);
        Vector128<float> row1 = Vector128.LoadAligned(m_modelViewProjectionRaw + 4);
        Vector128<float> row2 = Vector128.LoadAligned(m_modelViewProjectionRaw + 8);
        Vector128<float> row3 = Vector128.LoadAligned(m_modelViewProjectionRaw + 12);

        // Compute distance from each frustum plane
        Vector128<float> plane0 = Vector128.Add(row3, row0);
        Vector128<float> offset0 = Vector128.Add(center, Vector128.Xor(extents, Vector128.BitwiseAnd(plane0, minusZero)));
        Vector128<float> dist0 = Vector128.Create(Vector128.Dot(plane0, offset0));

        Vector128<float> plane1 = Vector128.Subtract(row3, row0);
        Vector128<float> offset1 = Vector128.Add(center, Vector128.Xor(extents, Vector128.BitwiseAnd(plane1, minusZero)));
        Vector128<float> dist1 = Vector128.Create(Vector128.Dot(plane1, offset1));

        Vector128<float> plane2 = Vector128.Add(row3, row1);
        Vector128<float> offset2 = Vector128.Add(center, Vector128.Xor(extents, Vector128.BitwiseAnd(plane2, minusZero)));
        Vector128<float> dist2 = Vector128.Create(Vector128.Dot(plane2, offset2));

        Vector128<float> plane3 = Vector128.Subtract(row3, row1);
        Vector128<float> offset3 = Vector128.Add(center, Vector128.Xor(extents, Vector128.BitwiseAnd(plane3, minusZero)));
        Vector128<float> dist3 = Vector128.Create(Vector128.Dot(plane3, offset3));

        Vector128<float> plane4 = Vector128.Add(row3, row2);
        Vector128<float> offset4 = Vector128.Add(center, Vector128.Xor(extents, Vector128.BitwiseAnd(plane4, minusZero)));
        Vector128<float> dist4 = Vector128.Create(Vector128.Dot(plane4, offset4));

        Vector128<float> plane5 = Vector128.Subtract(row3, row2);
        Vector128<float> offset5 = Vector128.Add(center, Vector128.Xor(extents, Vector128.BitwiseAnd(plane5, minusZero)));
        Vector128<float> dist5 = Vector128.Create(Vector128.Dot(plane5, offset5));

        // Combine plane distance signs
        Vector128<float> combined = Vector128.BitwiseOr(
            Vector128.BitwiseOr(
                Vector128.BitwiseOr(dist0, dist1),
                Vector128.BitwiseOr(dist2, dist3)),
            Vector128.BitwiseOr(dist4, dist5));

        // Can't use Avx.TestZ or _mm_comile_ss here because the OR's above created garbage in the non-sign bits
        if (Vector128.ExtractMostSignificantBits(combined) != 0)
        {
            needsClipping = false;
            return false;
        }

        // Load prebaked projection matrix
        Vector128<float> col0 = Vector128.LoadAligned(m_modelViewProjection + 0);
        Vector128<float> col1 = Vector128.LoadAligned(m_modelViewProjection + 4);
        Vector128<float> col2 = Vector128.LoadAligned(m_modelViewProjection + 8);
        Vector128<float> col3 = Vector128.LoadAligned(m_modelViewProjection + 12);

        // Transform edges
        Vector128<float> egde0 = Vector128.Multiply(col0, Vector128.Create(extents.ToScalar()));
        Vector128<float> egde1 = Vector128.Multiply(col1, V128Helper.PermuteFrom1(extents));
        Vector128<float> egde2 = Vector128.Multiply(col2, V128Helper.PermuteFrom2(extents));

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
          Fma.MultiplyAdd(col0, Vector128.Create(boundsMin.ToScalar()),
            Fma.MultiplyAdd(col1, V128Helper.PermuteFrom1(boundsMin),
              Fma.MultiplyAdd(col2, V128Helper.PermuteFrom2(boundsMin),
                col3)));

        // Transform remaining corners by adding edge vectors
        corners1 = Vector128.Add(corners0, egde0);
        corners2 = Vector128.Add(corners0, egde1);
        corners4 = Vector128.Add(corners0, egde2);

        corners3 = Vector128.Add(corners1, egde1);
        corners5 = Vector128.Add(corners4, egde0);
        corners6 = Vector128.Add(corners2, egde2);

        corners7 = Vector128.Add(corners6, egde0);

        // Transpose into SoA
        _MM_TRANSPOSE4_PS(ref corners0, ref corners1, ref corners2, ref corners3);
        _MM_TRANSPOSE4_PS(ref corners4, ref corners5, ref corners6, ref corners7);

        // Even if all bounding box corners have W > 0 here, we may end up with some vertices with W < 0 to due floating point differences; so test with some epsilon if any W < 0.
        Vector128<float> maxExtent = Vector128.Max(extents, Avx.Permute(extents, 0b01_00_11_10));
        maxExtent = Vector128.Max(maxExtent, Avx.Permute(maxExtent, 0b10_11_00_01));
        Vector128<float> nearPlaneEpsilon = Vector128.Multiply(maxExtent, Vector128.Create(0.001f));
        Vector128<float> closeToNearPlane = Vector128.BitwiseOr(Vector128.LessThan(corners3, nearPlaneEpsilon), Vector128.LessThan(corners7, nearPlaneEpsilon));
        if (!V128Helper.TestZ(closeToNearPlane, closeToNearPlane))
        {
            needsClipping = true;
            return true;
        }

        needsClipping = false;

        // Perspective division
        corners3 = V128Helper.Reciprocal(corners3);
        corners0 = Vector128.Multiply(corners0, corners3);
        corners1 = Vector128.Multiply(corners1, corners3);
        corners2 = Vector128.Multiply(corners2, corners3);

        corners7 = V128Helper.Reciprocal(corners7);
        corners4 = Vector128.Multiply(corners4, corners7);
        corners5 = Vector128.Multiply(corners5, corners7);
        corners6 = Vector128.Multiply(corners6, corners7);

        // Vertical mins and maxes
        Vector128<float> minsX = Vector128.Min(corners0, corners4);
        Vector128<float> maxsX = Vector128.Max(corners0, corners4);

        Vector128<float> minsY = Vector128.Min(corners1, corners5);
        Vector128<float> maxsY = Vector128.Max(corners1, corners5);

        // Horizontal reduction, step 1
        Vector128<float> minsXY = Vector128.Min(V128Helper.UnpackLow(minsX, minsY), V128Helper.UnpackHigh(minsX, minsY));
        Vector128<float> maxsXY = Vector128.Max(V128Helper.UnpackLow(maxsX, maxsY), V128Helper.UnpackHigh(maxsX, maxsY));

        // Clamp bounds
        minsXY = Vector128.Max(minsXY, Vector128<float>.Zero);
        maxsXY = Vector128.Min(maxsXY, Vector128.Create(m_width - 1f, m_height - 1f, m_width - 1f, m_height - 1f));

        // Negate maxes so we can round in the same direction
        maxsXY = Vector128.Xor(maxsXY, minusZero);

        // Horizontal reduction, step 2
        Vector128<float> boundsF = Vector128.Min(V128Helper.UnpackLow(minsXY, maxsXY), V128Helper.UnpackHigh(minsXY, maxsXY));

        // Round towards -infinity and convert to int
        Vector128<int> boundsI = Vector128.ConvertToInt32(Sse41.RoundToNegativeInfinity(boundsF));

        // Store as scalars
        uint minX = (uint)boundsI.GetElement(0);
        uint maxX = (uint)boundsI.GetElement(1);
        uint minY = (uint)boundsI.GetElement(2);
        uint maxY = (uint)boundsI.GetElement(3);

        // Revert the sign change we did for the maxes
        maxX = (uint)-(int)maxX;
        maxY = (uint)-(int)maxY;

        // No intersection between quad and screen area
        if (minX >= maxX || minY >= maxY)
        {
            return false;
        }

        Vector128<ushort> depth = packDepthPremultiplied(corners2, corners6);

        ushort maxZ = (ushort)(0xFFFFu ^ V128Helper.MinHorizontal(Vector128.Xor(depth, Vector128.Create((short)-1).AsUInt16())));

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

                    uint visiblePixelMask = ~Vector128.ExtractMostSignificantBits(notVisible.AsByte());

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

    public override void readBackDepth(byte* target)
    {
        const float bias = 1.0f / floatCompressionBias;

        const int stackBufferSize =
            Alignment - 1 +
            sizeof(float) * 8 * 8 * 1; // Vector256<float>[8] x 1

        byte* stackBuffer = stackalloc byte[stackBufferSize];
        byte* alignedBuffer = (byte*)((nint)(stackBuffer + (Alignment - 1)) & -Alignment);

        Vector256<float>* linDepthA = (Vector256<float>*)alignedBuffer;

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

                Vector256<float> vBias = Vector256.Create(bias);
                Vector256<float> vOne = Vector256.Create(1.0f);
                Vector256<float> vDiv = Vector256.Create(100 * 256 * 2 * 0.25f);
                Vector256<float> vSt = Vector256.Create(0.25f + 1000.0f);
                Vector256<float> vSf = Vector256.Create(1000.0f - 0.25f);

                Vector128<int>* source = &m_depthBuffer[8 * (blockY * m_blocksX + blockX)];
                for (uint y = 0; y < 8; ++y)
                {
                    Vector128<int> depthI = Vector128.LoadAligned((int*)source++);

                    Vector256<int> depthI256 = Vector256.ShiftLeft(V256Helper.ConvertToInt32(depthI.AsUInt16()), 12);
                    Vector256<float> depth = Vector256.Multiply(depthI256.AsSingle(), vBias);

                    Vector256<float> linDepth = Vector256.Divide(vDiv, Vector256.Subtract(vSt, Vector256.Multiply(Vector256.Subtract(vOne, depth), vSf)));
                    linDepth.StoreAligned((float*)(linDepthA + y));
                }

                Vector256<float> vRcp100 = Vector256.Create(1.0f / 100.0f);
                Vector256<ushort> vZeroMax = V256Helper.UnpackLow(Vector256<byte>.Zero, Vector256<byte>.AllBitsSet).AsUInt16();
                Vector256<ushort> vMask = Vector256.Create((ushort)0xff);

                for (uint y = 0; y < 8; y += 4)
                {
                    Vector256<float> depth0 = Vector256.LoadAligned((float*)(linDepthA + y + 0));
                    Vector256<float> depth1 = Vector256.LoadAligned((float*)(linDepthA + y + 1));
                    Vector256<float> depth2 = Vector256.LoadAligned((float*)(linDepthA + y + 2));
                    Vector256<float> depth3 = Vector256.LoadAligned((float*)(linDepthA + y + 3));

                    Vector256<int> vR32_0 = Vector256.ConvertToInt32(Vector256.Multiply(depth0, vRcp100));
                    Vector256<int> vR32_1 = Vector256.ConvertToInt32(Vector256.Multiply(depth1, vRcp100));
                    Vector256<int> vR32_2 = Vector256.ConvertToInt32(Vector256.Multiply(depth2, vRcp100));
                    Vector256<int> vR32_3 = Vector256.ConvertToInt32(Vector256.Multiply(depth3, vRcp100));

                    Vector256<ushort> vR16_0 = Vector256.BitwiseAnd(V256Helper.PackUnsignedSaturate(vR32_0, vR32_1), vMask);
                    Vector256<ushort> vR16_1 = Vector256.BitwiseAnd(V256Helper.PackUnsignedSaturate(vR32_2, vR32_3), vMask);
                    Vector256<byte> vR8 = V256Helper.PackUnsignedSaturate(vR16_0.AsInt16(), vR16_1.AsInt16());

                    Vector256<int> vG32_0 = Vector256.ConvertToInt32(depth0);
                    Vector256<int> vG32_1 = Vector256.ConvertToInt32(depth1);
                    Vector256<int> vG32_2 = Vector256.ConvertToInt32(depth2);
                    Vector256<int> vG32_3 = Vector256.ConvertToInt32(depth3);

                    Vector256<ushort> vG16_0 = Vector256.BitwiseAnd(V256Helper.PackUnsignedSaturate(vG32_0, vG32_1), vMask);
                    Vector256<ushort> vG16_1 = Vector256.BitwiseAnd(V256Helper.PackUnsignedSaturate(vG32_2, vG32_3), vMask);
                    Vector256<byte> vG8 = V256Helper.PackUnsignedSaturate(vG16_0.AsInt16(), vG16_1.AsInt16());

                    Vector256<ushort> vRG_Lo = V256Helper.UnpackLow(vR8, vG8).AsUInt16();
                    Vector256<ushort> vRG_Hi = V256Helper.UnpackHigh(vR8, vG8).AsUInt16();

                    Vector256<uint> result1 = V256Helper.UnpackLow(vRG_Lo, vZeroMax).AsUInt32();
                    Vector256<uint> result2 = V256Helper.UnpackHigh(vRG_Lo, vZeroMax).AsUInt32();
                    Vector256<uint> result3 = V256Helper.UnpackLow(vRG_Hi, vZeroMax).AsUInt32();
                    Vector256<uint> result4 = V256Helper.UnpackHigh(vRG_Hi, vZeroMax).AsUInt32();

                    result1.StoreAligned((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 0))));
                    result2.StoreAligned((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 1))));
                    result3.StoreAligned((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 2))));
                    result4.StoreAligned((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 3))));
                }
            }
        }
    }

    private static void transpose256(Vector256<float> A, Vector256<float> B, Vector256<float> C, Vector256<float> D, Vector128<float>* @out)
    {
        Vector256<float> _Tmp0 = Avx.Shuffle(A, B, 0b01_00_01_00);
        Vector256<float> _Tmp2 = Avx.Shuffle(A, B, 0b11_10_11_10);
        Vector256<float> _Tmp1 = Avx.Shuffle(C, D, 0b01_00_01_00);
        Vector256<float> _Tmp3 = Avx.Shuffle(C, D, 0b11_10_11_10);

        Vector256<float> tA = Avx.Shuffle(_Tmp0, _Tmp1, 0b10_00_10_00);
        Vector256<float> tB = Avx.Shuffle(_Tmp0, _Tmp1, 0b11_01_11_01);
        Vector256<float> tC = Avx.Shuffle(_Tmp2, _Tmp3, 0b10_00_10_00);
        Vector256<float> tD = Avx.Shuffle(_Tmp2, _Tmp3, 0b11_01_11_01);

        tA.StoreAligned((float*)(@out + 0));
        tB.StoreAligned((float*)(@out + 2));
        tC.StoreAligned((float*)(@out + 4));
        tD.StoreAligned((float*)(@out + 6));
    }

    private static void transpose256i(Vector256<int> A, Vector256<int> B, Vector256<int> C, Vector256<int> D, Vector128<int>* @out)
    {
        Vector256<long> _Tmp0 = V256Helper.UnpackLow(A, B).AsInt64();
        Vector256<long> _Tmp1 = V256Helper.UnpackLow(C, D).AsInt64();
        Vector256<long> _Tmp2 = V256Helper.UnpackHigh(A, B).AsInt64();
        Vector256<long> _Tmp3 = V256Helper.UnpackHigh(C, D).AsInt64();

        Vector256<int> tA = V256Helper.UnpackLow(_Tmp0, _Tmp1).AsInt32();
        Vector256<int> tB = V256Helper.UnpackHigh(_Tmp0, _Tmp1).AsInt32();
        Vector256<int> tC = V256Helper.UnpackLow(_Tmp2, _Tmp3).AsInt32();
        Vector256<int> tD = V256Helper.UnpackHigh(_Tmp2, _Tmp3).AsInt32();

        tA.StoreAligned((int*)(@out + 0));
        tB.StoreAligned((int*)(@out + 2));
        tC.StoreAligned((int*)(@out + 4));
        tD.StoreAligned((int*)(@out + 6));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void normalizeEdge(ref Vector256<float> nx, ref Vector256<float> ny, Vector256<float> edgeFlipMask)
    {
        Vector256<float> minusZero = Vector256.Create(-0.0f);
        Vector256<float> invLen = V256Helper.Reciprocal(Vector256.Add(Vector256.AndNot(nx, minusZero), Vector256.AndNot(ny, minusZero)));

        invLen = Vector256.Multiply(edgeFlipMask, invLen);
        nx = Vector256.Multiply(nx, invLen);
        ny = Vector256.Multiply(ny, invLen);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> quantizeSlopeLookup(Vector256<float> nx, Vector256<float> ny)
    {
        Vector256<int> yNeg = Avx.Compare(ny, Vector256<float>.Zero, _CMP_LE_OQ).AsInt32();

        // Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
        const float maxOffset = -minEdgeOffset;
        const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f / ((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        const float add = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f + 0.5f;

        Vector256<int> quantizedSlope = Vector256.ConvertToInt32(Fma.MultiplyAdd(nx, Vector256.Create(mul), Vector256.Create(add)));
        return Vector256.ShiftLeft(Vector256.Subtract(Vector256.ShiftLeft(quantizedSlope, 1), yNeg), OFFSET_QUANTIZATION_BITS);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> packDepthPremultiplied(Vector128<float> depthA, Vector128<float> depthB)
    {
        Vector128<int> x1 = Vector128.ShiftRightArithmetic(depthA.AsInt32(), 12);
        Vector128<int> x2 = Vector128.ShiftRightArithmetic(depthB.AsInt32(), 12);
        return V128Helper.PackUnsignedSaturate(x1, x2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> packDepthPremultiplied(Vector256<float> depth)
    {
        Vector256<int> x = Vector256.ShiftRightArithmetic(depth.AsInt32(), 12);
        return V128Helper.PackUnsignedSaturate(x.GetLower(), x.GetUpper());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<ushort> packDepthPremultiplied(Vector256<float> depthA, Vector256<float> depthB)
    {
        Vector256<int> x1 = Vector256.ShiftRightArithmetic(depthA.AsInt32(), 12);
        Vector256<int> x2 = Vector256.ShiftRightArithmetic(depthB.AsInt32(), 12);
        return V256Helper.PackUnsignedSaturate(x1, x2);
    }

    private static (Vector256<float> minFx, Vector256<float> minFy, Vector256<float> maxFx, Vector256<float> maxFy) CliplessBoundingBox(
        Vector256<float> x0, Vector256<float> x1, Vector256<float> x2, Vector256<float> x3,
        Vector256<float> y0, Vector256<float> y1, Vector256<float> y2, Vector256<float> y3,
        Vector256<float> wSign0, Vector256<float> wSign1, Vector256<float> wSign2, Vector256<float> wSign3)
    {
        Vector256<float> infP = Vector256.Create(+10000.0f);
        Vector256<float> infN = Vector256.Create(-10000.0f);

        // Find interval of points with W > 0
        Vector256<float> minPx0 = V256Helper.BlendVariable(x0, infP, wSign0);
        Vector256<float> minPx1 = V256Helper.BlendVariable(x1, infP, wSign1);
        Vector256<float> minPx2 = V256Helper.BlendVariable(x2, infP, wSign2);
        Vector256<float> minPx3 = V256Helper.BlendVariable(x3, infP, wSign3);

        Vector256<float> minPx = Vector256.Min(
          Vector256.Min(minPx0, minPx1),
          Vector256.Min(minPx2, minPx3));

        Vector256<float> minPy0 = V256Helper.BlendVariable(y0, infP, wSign0);
        Vector256<float> minPy1 = V256Helper.BlendVariable(y1, infP, wSign1);
        Vector256<float> minPy2 = V256Helper.BlendVariable(y2, infP, wSign2);
        Vector256<float> minPy3 = V256Helper.BlendVariable(y3, infP, wSign3);

        Vector256<float> minPy = Vector256.Min(
          Vector256.Min(minPy0, minPy1),
          Vector256.Min(minPy2, minPy3));

        Vector256<float> maxPx0 = Vector256.Xor(minPx0, wSign0);
        Vector256<float> maxPx1 = Vector256.Xor(minPx1, wSign1);
        Vector256<float> maxPx2 = Vector256.Xor(minPx2, wSign2);
        Vector256<float> maxPx3 = Vector256.Xor(minPx3, wSign3);

        Vector256<float> maxPx = Vector256.Max(
          Vector256.Max(maxPx0, maxPx1),
          Vector256.Max(maxPx2, maxPx3));

        Vector256<float> maxPy0 = Vector256.Xor(minPy0, wSign0);
        Vector256<float> maxPy1 = Vector256.Xor(minPy1, wSign1);
        Vector256<float> maxPy2 = Vector256.Xor(minPy2, wSign2);
        Vector256<float> maxPy3 = Vector256.Xor(minPy3, wSign3);

        Vector256<float> maxPy = Vector256.Max(
          Vector256.Max(maxPy0, maxPy1),
          Vector256.Max(maxPy2, maxPy3));

        // Find interval of points with W < 0
        Vector256<float> minNx0 = V256Helper.BlendVariable(infP, x0, wSign0);
        Vector256<float> minNx1 = V256Helper.BlendVariable(infP, x1, wSign1);
        Vector256<float> minNx2 = V256Helper.BlendVariable(infP, x2, wSign2);
        Vector256<float> minNx3 = V256Helper.BlendVariable(infP, x3, wSign3);

        Vector256<float> minNx = Vector256.Min(
          Vector256.Min(minNx0, minNx1),
          Vector256.Min(minNx2, minNx3));

        Vector256<float> minNy0 = V256Helper.BlendVariable(infP, y0, wSign0);
        Vector256<float> minNy1 = V256Helper.BlendVariable(infP, y1, wSign1);
        Vector256<float> minNy2 = V256Helper.BlendVariable(infP, y2, wSign2);
        Vector256<float> minNy3 = V256Helper.BlendVariable(infP, y3, wSign3);

        Vector256<float> minNy = Vector256.Min(
          Vector256.Min(minNy0, minNy1),
          Vector256.Min(minNy2, minNy3));

        Vector256<float> maxNx0 = V256Helper.BlendVariable(infN, x0, wSign0);
        Vector256<float> maxNx1 = V256Helper.BlendVariable(infN, x1, wSign1);
        Vector256<float> maxNx2 = V256Helper.BlendVariable(infN, x2, wSign2);
        Vector256<float> maxNx3 = V256Helper.BlendVariable(infN, x3, wSign3);

        Vector256<float> maxNx = Vector256.Max(
          Vector256.Max(maxNx0, maxNx1),
          Vector256.Max(maxNx2, maxNx3));

        Vector256<float> maxNy0 = V256Helper.BlendVariable(infN, y0, wSign0);
        Vector256<float> maxNy1 = V256Helper.BlendVariable(infN, y1, wSign1);
        Vector256<float> maxNy2 = V256Helper.BlendVariable(infN, y2, wSign2);
        Vector256<float> maxNy3 = V256Helper.BlendVariable(infN, y3, wSign3);

        Vector256<float> maxNy = Vector256.Max(
          Vector256.Max(maxNy0, maxNy1),
          Vector256.Max(maxNy2, maxNy3));

        // Include interval bounds resp. infinity depending on ordering of intervals
        Vector256<float> incAx = V256Helper.BlendVariable(minPx, infN, Avx.Compare(maxNx, minPx, _CMP_GT_OQ));
        Vector256<float> incAy = V256Helper.BlendVariable(minPy, infN, Avx.Compare(maxNy, minPy, _CMP_GT_OQ));

        Vector256<float> incBx = V256Helper.BlendVariable(maxPx, infP, Avx.Compare(maxPx, minNx, _CMP_GT_OQ));
        Vector256<float> incBy = V256Helper.BlendVariable(maxPy, infP, Avx.Compare(maxPy, minNy, _CMP_GT_OQ));

        Vector256<float> minFx = Vector256.Min(incAx, incBx);
        Vector256<float> minFy = Vector256.Min(incAy, incBy);

        Vector256<float> maxFx = Vector256.Max(incAx, incBx);
        Vector256<float> maxFy = Vector256.Max(incAy, incBy);

        return (minFx, minFy, maxFx, maxFy);
    }

    public override void rasterize<T>(in Occluder occluder)
    {
        Vector256<int>* vertexData = occluder.m_vertexData;
        uint packetCount = occluder.m_packetCount;

        // Note that unaligned loads do not have a latency penalty on CPUs with SSE4 support
        Vector128<float> mat0 = Vector128.LoadAligned(m_modelViewProjection + 0);
        Vector128<float> mat1 = Vector128.LoadAligned(m_modelViewProjection + 4);
        Vector128<float> mat2 = Vector128.LoadAligned(m_modelViewProjection + 8);
        Vector128<float> mat3 = Vector128.LoadAligned(m_modelViewProjection + 12);

        Vector128<float> boundsMin = occluder.m_refMin.AsVector128();
        Vector128<float> boundsExtents = Vector128.Subtract(occluder.m_refMax.AsVector128(), boundsMin);

        // Bake integer => bounding box transform into matrix
        mat3 =
          Fma.MultiplyAdd(mat0, Vector128.Create(boundsMin.ToScalar()),
            Fma.MultiplyAdd(mat1, V128Helper.PermuteFrom1(boundsMin),
              Fma.MultiplyAdd(mat2, V128Helper.PermuteFrom2(boundsMin),
                mat3)));

        mat0 = Vector128.Multiply(mat0, Vector128.Multiply(Vector128.Create(boundsExtents.ToScalar()), Vector128.Create(1.0f / (2047ul << 21))));
        mat1 = Vector128.Multiply(mat1, Vector128.Multiply(V128Helper.PermuteFrom1(boundsExtents), Vector128.Create(1.0f / (2047 << 10))));
        mat2 = Vector128.Multiply(mat2, Vector128.Multiply(V128Helper.PermuteFrom2(boundsExtents), Vector128.Create(1.0f / 1023)));

        // Bias X coordinate back into positive range
        mat3 = Fma.MultiplyAdd(mat0, Vector128.Create((float)(1024ul << 21)), mat3);

        // Skew projection to correct bleeding of Y and Z into X due to lack of masking
        mat1 = Vector128.Subtract(mat1, mat0);
        mat2 = Vector128.Subtract(mat2, mat0);

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Due to linear relationship between Z and W, it's cheaper to compute Z from W later in the pipeline than using the full projection matrix up front
        float c0, c1;
        {
            Vector128<float> Za = V128Helper.PermuteFrom3(mat2);
            Vector128<float> Zb = Vector128.Create(Vector128.Dot(mat2, Vector128.Create((float)(1 << 21), 1 << 10, 1, 1)));

            Vector128<float> Wa = V128Helper.PermuteFrom3(mat3);
            Vector128<float> Wb = Vector128.Create(Vector128.Dot(mat3, Vector128.Create((float)(1 << 21), 1 << 10, 1, 1)));

            c0 = Vector128.Divide(Vector128.Subtract(Za, Zb), Vector128.Subtract(Wa, Wb)).ToScalar();
            c1 = Fma.MultiplyAddNegated(Vector128.Divide(Vector128.Subtract(Za, Zb), Vector128.Subtract(Wa, Wb)), Wa, Za).ToScalar();
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
            Vector256<int> I0 = Vector256.LoadAlignedNonTemporal((int*)(vertexData + packetIdx + 0));
            Vector256<int> I1 = Vector256.LoadAlignedNonTemporal((int*)(vertexData + packetIdx + 1));
            Vector256<int> I2 = Vector256.LoadAlignedNonTemporal((int*)(vertexData + packetIdx + 2));
            Vector256<int> I3 = Vector256.LoadAlignedNonTemporal((int*)(vertexData + packetIdx + 3));

            // Vertex transformation - first W, then X & Y after camera plane culling, then Z after backface culling
            Vector256<float> Xf0 = Vector256.ConvertToSingle(I0);
            Vector256<float> Xf1 = Vector256.ConvertToSingle(I1);
            Vector256<float> Xf2 = Vector256.ConvertToSingle(I2);
            Vector256<float> Xf3 = Vector256.ConvertToSingle(I3);

            Vector256<int> maskY = Vector256.Create(2047 << 10);
            Vector256<float> Yf0 = Vector256.ConvertToSingle(Vector256.BitwiseAnd(I0, maskY));
            Vector256<float> Yf1 = Vector256.ConvertToSingle(Vector256.BitwiseAnd(I1, maskY));
            Vector256<float> Yf2 = Vector256.ConvertToSingle(Vector256.BitwiseAnd(I2, maskY));
            Vector256<float> Yf3 = Vector256.ConvertToSingle(Vector256.BitwiseAnd(I3, maskY));

            Vector256<int> maskZ = Vector256.Create(1023);
            Vector256<float> Zf0 = Vector256.ConvertToSingle(Vector256.BitwiseAnd(I0, maskZ));
            Vector256<float> Zf1 = Vector256.ConvertToSingle(Vector256.BitwiseAnd(I1, maskZ));
            Vector256<float> Zf2 = Vector256.ConvertToSingle(Vector256.BitwiseAnd(I2, maskZ));
            Vector256<float> Zf3 = Vector256.ConvertToSingle(Vector256.BitwiseAnd(I3, maskZ));

            Vector256<float> mat00 = Vector256.Create(*((float*)&mat0 + 0));
            Vector256<float> mat01 = Vector256.Create(*((float*)&mat0 + 1));
            Vector256<float> mat02 = Vector256.Create(*((float*)&mat0 + 2));
            Vector256<float> mat03 = Vector256.Create(*((float*)&mat0 + 3));

            Vector256<float> X0 = Fma.MultiplyAdd(Xf0, mat00, Fma.MultiplyAdd(Yf0, mat01, Fma.MultiplyAdd(Zf0, mat02, mat03)));
            Vector256<float> X1 = Fma.MultiplyAdd(Xf1, mat00, Fma.MultiplyAdd(Yf1, mat01, Fma.MultiplyAdd(Zf1, mat02, mat03)));
            Vector256<float> X2 = Fma.MultiplyAdd(Xf2, mat00, Fma.MultiplyAdd(Yf2, mat01, Fma.MultiplyAdd(Zf2, mat02, mat03)));
            Vector256<float> X3 = Fma.MultiplyAdd(Xf3, mat00, Fma.MultiplyAdd(Yf3, mat01, Fma.MultiplyAdd(Zf3, mat02, mat03)));

            Vector256<float> mat10 = Vector256.Create(*((float*)&mat1 + 0));
            Vector256<float> mat11 = Vector256.Create(*((float*)&mat1 + 1));
            Vector256<float> mat12 = Vector256.Create(*((float*)&mat1 + 2));
            Vector256<float> mat13 = Vector256.Create(*((float*)&mat1 + 3));

            Vector256<float> Y0 = Fma.MultiplyAdd(Xf0, mat10, Fma.MultiplyAdd(Yf0, mat11, Fma.MultiplyAdd(Zf0, mat12, mat13)));
            Vector256<float> Y1 = Fma.MultiplyAdd(Xf1, mat10, Fma.MultiplyAdd(Yf1, mat11, Fma.MultiplyAdd(Zf1, mat12, mat13)));
            Vector256<float> Y2 = Fma.MultiplyAdd(Xf2, mat10, Fma.MultiplyAdd(Yf2, mat11, Fma.MultiplyAdd(Zf2, mat12, mat13)));
            Vector256<float> Y3 = Fma.MultiplyAdd(Xf3, mat10, Fma.MultiplyAdd(Yf3, mat11, Fma.MultiplyAdd(Zf3, mat12, mat13)));

            Vector256<float> mat30 = Vector256.Create(*((float*)&mat3 + 0));
            Vector256<float> mat31 = Vector256.Create(*((float*)&mat3 + 1));
            Vector256<float> mat32 = Vector256.Create(*((float*)&mat3 + 2));
            Vector256<float> mat33 = Vector256.Create(*((float*)&mat3 + 3));

            Vector256<float> W0 = Fma.MultiplyAdd(Xf0, mat30, Fma.MultiplyAdd(Yf0, mat31, Fma.MultiplyAdd(Zf0, mat32, mat33)));
            Vector256<float> W1 = Fma.MultiplyAdd(Xf1, mat30, Fma.MultiplyAdd(Yf1, mat31, Fma.MultiplyAdd(Zf1, mat32, mat33)));
            Vector256<float> W2 = Fma.MultiplyAdd(Xf2, mat30, Fma.MultiplyAdd(Yf2, mat31, Fma.MultiplyAdd(Zf2, mat32, mat33)));
            Vector256<float> W3 = Fma.MultiplyAdd(Xf3, mat30, Fma.MultiplyAdd(Yf3, mat31, Fma.MultiplyAdd(Zf3, mat32, mat33)));

            Vector256<float> invW0 = V256Helper.Reciprocal(W0);
            Vector256<float> invW1 = V256Helper.Reciprocal(W1);
            Vector256<float> invW2 = V256Helper.Reciprocal(W2);
            Vector256<float> invW3 = V256Helper.Reciprocal(W3);

            // Clamp W and invert
            if (T.PossiblyNearClipped)
            {
                Vector256<float> lowerBound = Vector256.Create((float)-maxInvW);
                Vector256<float> upperBound = Vector256.Create((float)+maxInvW);
                invW0 = Vector256.Min(upperBound, Vector256.Max(lowerBound, invW0));
                invW1 = Vector256.Min(upperBound, Vector256.Max(lowerBound, invW1));
                invW2 = Vector256.Min(upperBound, Vector256.Max(lowerBound, invW2));
                invW3 = Vector256.Min(upperBound, Vector256.Max(lowerBound, invW3));
            }

            // Round to integer coordinates to improve culling of zero-area triangles
            Vector256<float> roundFactor = Vector256.Create(0.125f);
            Vector256<float> x0 = Vector256.Multiply(Avx.RoundToNearestInteger(Vector256.Multiply(X0, invW0)), roundFactor);
            Vector256<float> x1 = Vector256.Multiply(Avx.RoundToNearestInteger(Vector256.Multiply(X1, invW1)), roundFactor);
            Vector256<float> x2 = Vector256.Multiply(Avx.RoundToNearestInteger(Vector256.Multiply(X2, invW2)), roundFactor);
            Vector256<float> x3 = Vector256.Multiply(Avx.RoundToNearestInteger(Vector256.Multiply(X3, invW3)), roundFactor);

            Vector256<float> y0 = Vector256.Multiply(Avx.RoundToNearestInteger(Vector256.Multiply(Y0, invW0)), roundFactor);
            Vector256<float> y1 = Vector256.Multiply(Avx.RoundToNearestInteger(Vector256.Multiply(Y1, invW1)), roundFactor);
            Vector256<float> y2 = Vector256.Multiply(Avx.RoundToNearestInteger(Vector256.Multiply(Y2, invW2)), roundFactor);
            Vector256<float> y3 = Vector256.Multiply(Avx.RoundToNearestInteger(Vector256.Multiply(Y3, invW3)), roundFactor);

            // Compute unnormalized edge directions
            Vector256<float> edgeNormalsX0 = Vector256.Subtract(y1, y0);
            Vector256<float> edgeNormalsX1 = Vector256.Subtract(y2, y1);
            Vector256<float> edgeNormalsX2 = Vector256.Subtract(y3, y2);
            Vector256<float> edgeNormalsX3 = Vector256.Subtract(y0, y3);

            Vector256<float> edgeNormalsY0 = Vector256.Subtract(x0, x1);
            Vector256<float> edgeNormalsY1 = Vector256.Subtract(x1, x2);
            Vector256<float> edgeNormalsY2 = Vector256.Subtract(x2, x3);
            Vector256<float> edgeNormalsY3 = Vector256.Subtract(x3, x0);

            Vector256<float> area0 = Fma.MultiplySubtract(edgeNormalsX0, edgeNormalsY1, Vector256.Multiply(edgeNormalsX1, edgeNormalsY0));
            Vector256<float> area1 = Fma.MultiplySubtract(edgeNormalsX1, edgeNormalsY2, Vector256.Multiply(edgeNormalsX2, edgeNormalsY1));
            Vector256<float> area2 = Fma.MultiplySubtract(edgeNormalsX2, edgeNormalsY3, Vector256.Multiply(edgeNormalsX3, edgeNormalsY2));
            Vector256<float> area3 = Vector256.Subtract(Vector256.Add(area0, area2), area1);

            Vector256<float> minusZero256 = Vector256.Create(-0.0f);

            Vector256<float> wSign0, wSign1, wSign2, wSign3;
            if (T.PossiblyNearClipped)
            {
                wSign0 = Vector256.BitwiseAnd(invW0, minusZero256);
                wSign1 = Vector256.BitwiseAnd(invW1, minusZero256);
                wSign2 = Vector256.BitwiseAnd(invW2, minusZero256);
                wSign3 = Vector256.BitwiseAnd(invW3, minusZero256);
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
                areaSign0 = Avx.Compare(Vector256.Xor(Vector256.Xor(area0, wSign0), Vector256.Xor(wSign1, wSign2)), Vector256<float>.Zero, _CMP_LE_OQ);
                areaSign1 = Vector256.BitwiseAnd(minusZero256, Avx.Compare(Vector256.Xor(Vector256.Xor(area1, wSign1), Vector256.Xor(wSign2, wSign3)), Vector256<float>.Zero, _CMP_LE_OQ));
                areaSign2 = Vector256.BitwiseAnd(minusZero256, Avx.Compare(Vector256.Xor(Vector256.Xor(area2, wSign0), Vector256.Xor(wSign2, wSign3)), Vector256<float>.Zero, _CMP_LE_OQ));
                areaSign3 = Vector256.BitwiseAnd(minusZero256, Avx.Compare(Vector256.Xor(Vector256.Xor(area3, wSign1), Vector256.Xor(wSign0, wSign3)), Vector256<float>.Zero, _CMP_LE_OQ));
            }
            else
            {
                areaSign0 = Avx.Compare(area0, Vector256<float>.Zero, _CMP_LE_OQ);
                areaSign1 = Vector256.BitwiseAnd(minusZero256, Avx.Compare(area1, Vector256<float>.Zero, _CMP_LE_OQ));
                areaSign2 = Vector256.BitwiseAnd(minusZero256, Avx.Compare(area2, Vector256<float>.Zero, _CMP_LE_OQ));
                areaSign3 = Vector256.BitwiseAnd(minusZero256, Avx.Compare(area3, Vector256<float>.Zero, _CMP_LE_OQ));
            }

            Vector256<int> config = Vector256.BitwiseOr(
              Vector256.BitwiseOr(Vector256.ShiftRightLogical(areaSign3.AsInt32(), 28), Vector256.ShiftRightLogical(areaSign2.AsInt32(), 29)),
              Vector256.BitwiseOr(Vector256.ShiftRightLogical(areaSign1.AsInt32(), 30), Vector256.ShiftRightLogical(areaSign0.AsInt32(), 31)));

            if (T.PossiblyNearClipped)
            {
                config = Vector256.BitwiseOr(config,
                  Vector256.BitwiseOr(
                    Vector256.BitwiseOr(Vector256.ShiftRightLogical(wSign3.AsInt32(), 24), Vector256.ShiftRightLogical(wSign2.AsInt32(), 25)),
                    Vector256.BitwiseOr(Vector256.ShiftRightLogical(wSign1.AsInt32(), 26), Vector256.ShiftRightLogical(wSign0.AsInt32(), 27))));
            }

            ref int modeTablePtr = ref Unsafe.As<PrimitiveMode, int>(ref MemoryMarshal.GetReference(modeTable));
            Vector256<int> modes = V256Helper.GatherBy4(ref modeTablePtr, config);
            if (V256Helper.TestZ(modes, modes))
            {
                continue;
            }

            modes.StoreAligned((int*)primModes);

            Vector256<int> primitiveValid = Vector256.GreaterThan(modes, Vector256<int>.Zero);

            Vector256<float> minFx, minFy, maxFx, maxFy;
            if (T.PossiblyNearClipped)
            {
                // Clipless bounding box computation
                (minFx, minFy, maxFx, maxFy) = CliplessBoundingBox(x0, x1, x2, x3, y0, y1, y2, y3, wSign0, wSign1, wSign2, wSign3);
            }
            else
            {
                // Standard bounding box inclusion
                minFx = Vector256.Min(Vector256.Min(x0, x1), Vector256.Min(x2, x3));
                minFy = Vector256.Min(Vector256.Min(y0, y1), Vector256.Min(y2, y3));

                maxFx = Vector256.Max(Vector256.Max(x0, x1), Vector256.Max(x2, x3));
                maxFy = Vector256.Max(Vector256.Max(y0, y1), Vector256.Max(y2, y3));
            }

            // Clamp and round
            Vector256<int> minX, minY, maxX, maxY;
            minX = Vector256.Max(Vector256.ConvertToInt32(Vector256.Add(minFx, Vector256.Create(4.9999f / 8.0f))), Vector256<int>.Zero);
            minY = Vector256.Max(Vector256.ConvertToInt32(Vector256.Add(minFy, Vector256.Create(4.9999f / 8.0f))), Vector256<int>.Zero);
            maxX = Vector256.Min(Vector256.ConvertToInt32(Vector256.Add(maxFx, Vector256.Create(11.0f / 8.0f))), Vector256.Create((int)m_blocksX));
            maxY = Vector256.Min(Vector256.ConvertToInt32(Vector256.Add(maxFy, Vector256.Create(11.0f / 8.0f))), Vector256.Create((int)m_blocksY));

            // Check overlap between bounding box and frustum
            Vector256<int> inFrustum = Vector256.BitwiseAnd(Vector256.GreaterThan(maxX, minX), Vector256.GreaterThan(maxY, minY));
            Vector256<int> overlappedPrimitiveValid = Vector256.BitwiseAnd(inFrustum, primitiveValid);

            if (V256Helper.TestZ(overlappedPrimitiveValid, overlappedPrimitiveValid))
            {
                continue;
            }

            uint validMask = Vector256.ExtractMostSignificantBits(overlappedPrimitiveValid.AsSingle());

            // Convert bounds from [min, max] to [min, range]
            Vector256<int> rangeX = Vector256.Subtract(maxX, minX);
            Vector256<int> rangeY = Vector256.Subtract(maxY, minY);

            // Compute Z from linear relation with 1/W
            Vector256<float> C0 = Vector256.Create(c0);
            Vector256<float> C1 = Vector256.Create(c1);
            Vector256<float> z0, z1, z2, z3;
            z0 = Fma.MultiplyAdd(invW0, C1, C0);
            z1 = Fma.MultiplyAdd(invW1, C1, C0);
            z2 = Fma.MultiplyAdd(invW2, C1, C0);
            z3 = Fma.MultiplyAdd(invW3, C1, C0);

            Vector256<float> maxZ = Vector256.Max(Vector256.Max(z0, z1), Vector256.Max(z2, z3));

            // If any W < 0, assume maxZ = 1 (effectively disabling Hi-Z)
            if (T.PossiblyNearClipped)
            {
                Vector256<float> wMask = Vector256.BitwiseOr(Vector256.BitwiseOr(wSign0, wSign1), Vector256.BitwiseOr(wSign2, wSign3));
                maxZ = V256Helper.BlendVariable(maxZ, Vector256.Create(1.0f), wMask);
            }

            Vector128<ushort> packedDepthBounds = packDepthPremultiplied(maxZ);

            packedDepthBounds.StoreAligned(depthBounds);

            // Compute screen space depth plane
            Vector256<float> greaterArea = Avx.Compare(Vector256.AndNot(area0, minusZero256), Vector256.AndNot(area2, minusZero256), _CMP_LT_OQ);

            // Force triangle area to be picked in the relevant mode.
            Vector256<float> modeTriangle0 = Vector256.Equals(modes, Vector256.Create((int)PrimitiveMode.Triangle0)).AsSingle();
            Vector256<float> modeTriangle1 = Vector256.Equals(modes, Vector256.Create((int)PrimitiveMode.Triangle1)).AsSingle();
            greaterArea = Vector256.AndNot(Vector256.BitwiseOr(modeTriangle1, greaterArea), modeTriangle0);

            Vector256<float> invArea;
            if (T.PossiblyNearClipped)
            {
                // Do a precise divison to reduce error in depth plane. Note that the area computed here
                // differs from the rasterized region if W < 0, so it can be very small for large covered screen regions.
                invArea = Vector256.Divide(Vector256.Create(1.0f), V256Helper.BlendVariable(area0, area2, greaterArea));
            }
            else
            {
                invArea = V256Helper.Reciprocal(V256Helper.BlendVariable(area0, area2, greaterArea));
            }

            Vector256<float> z12 = Vector256.Subtract(z1, z2);
            Vector256<float> z20 = Vector256.Subtract(z2, z0);
            Vector256<float> z30 = Vector256.Subtract(z3, z0);

            Vector256<float> edgeNormalsX4 = Vector256.Subtract(y0, y2);
            Vector256<float> edgeNormalsY4 = Vector256.Subtract(x2, x0);

            Vector256<float> depthPlane0, depthPlane1, depthPlane2;
            depthPlane1 = Vector256.Multiply(invArea, V256Helper.BlendVariable(Fma.MultiplySubtract(z20, edgeNormalsX1, Vector256.Multiply(z12, edgeNormalsX4)), Fma.MultiplyAddNegated(z20, edgeNormalsX3, Vector256.Multiply(z30, edgeNormalsX4)), greaterArea));
            depthPlane2 = Vector256.Multiply(invArea, V256Helper.BlendVariable(Fma.MultiplySubtract(z20, edgeNormalsY1, Vector256.Multiply(z12, edgeNormalsY4)), Fma.MultiplyAddNegated(z20, edgeNormalsY3, Vector256.Multiply(z30, edgeNormalsY4)), greaterArea));

            x0 = Vector256.Subtract(x0, Vector256.ConvertToSingle(minX));
            y0 = Vector256.Subtract(y0, Vector256.ConvertToSingle(minY));

            depthPlane0 = Fma.MultiplyAddNegated(x0, depthPlane1, Fma.MultiplyAddNegated(y0, depthPlane2, z0));

            // If mode == Triangle0, replace edge 2 with edge 4; if mode == Triangle1, replace edge 0 with edge 4
            edgeNormalsX2 = V256Helper.BlendVariable(edgeNormalsX2, edgeNormalsX4, modeTriangle0);
            edgeNormalsY2 = V256Helper.BlendVariable(edgeNormalsY2, edgeNormalsY4, modeTriangle0);
            edgeNormalsX0 = V256Helper.BlendVariable(edgeNormalsX0, Vector256.Xor(minusZero256, edgeNormalsX4), modeTriangle1);
            edgeNormalsY0 = V256Helper.BlendVariable(edgeNormalsY0, Vector256.Xor(minusZero256, edgeNormalsY4), modeTriangle1);

            const float maxOffset = -minEdgeOffset;
            Vector256<float> edgeFactor = Vector256.Create((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));

            // Flip edges if W < 0
            Vector256<float> edgeFlipMask0, edgeFlipMask1, edgeFlipMask2, edgeFlipMask3;
            if (T.PossiblyNearClipped)
            {
                edgeFlipMask0 = Vector256.Xor(edgeFactor, Vector256.Xor(wSign0, V256Helper.BlendVariable(wSign1, wSign2, modeTriangle1)));
                edgeFlipMask1 = Vector256.Xor(edgeFactor, Vector256.Xor(wSign1, wSign2));
                edgeFlipMask2 = Vector256.Xor(edgeFactor, Vector256.Xor(wSign2, V256Helper.BlendVariable(wSign3, wSign0, modeTriangle0)));
                edgeFlipMask3 = Vector256.Xor(edgeFactor, Vector256.Xor(wSign0, wSign3));
            }
            else
            {
                edgeFlipMask0 = edgeFactor;
                edgeFlipMask1 = edgeFactor;
                edgeFlipMask2 = edgeFactor;
                edgeFlipMask3 = edgeFactor;
            }

            // Normalize edge equations for lookup
            normalizeEdge(ref edgeNormalsX0, ref edgeNormalsY0, edgeFlipMask0);
            normalizeEdge(ref edgeNormalsX1, ref edgeNormalsY1, edgeFlipMask1);
            normalizeEdge(ref edgeNormalsX2, ref edgeNormalsY2, edgeFlipMask2);
            normalizeEdge(ref edgeNormalsX3, ref edgeNormalsY3, edgeFlipMask3);

            Vector256<float> add256 = Vector256.Create(0.5f - minEdgeOffset * (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
            Vector256<float> edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3;

            edgeOffsets0 = Fma.MultiplyAddNegated(x0, edgeNormalsX0, Fma.MultiplyAddNegated(y0, edgeNormalsY0, add256));
            edgeOffsets1 = Fma.MultiplyAddNegated(x1, edgeNormalsX1, Fma.MultiplyAddNegated(y1, edgeNormalsY1, add256));
            edgeOffsets2 = Fma.MultiplyAddNegated(x2, edgeNormalsX2, Fma.MultiplyAddNegated(y2, edgeNormalsY2, add256));
            edgeOffsets3 = Fma.MultiplyAddNegated(x3, edgeNormalsX3, Fma.MultiplyAddNegated(y3, edgeNormalsY3, add256));

            edgeOffsets1 = Fma.MultiplyAdd(Vector256.ConvertToSingle(minX), edgeNormalsX1, edgeOffsets1);
            edgeOffsets2 = Fma.MultiplyAdd(Vector256.ConvertToSingle(minX), edgeNormalsX2, edgeOffsets2);
            edgeOffsets3 = Fma.MultiplyAdd(Vector256.ConvertToSingle(minX), edgeNormalsX3, edgeOffsets3);

            edgeOffsets1 = Fma.MultiplyAdd(Vector256.ConvertToSingle(minY), edgeNormalsY1, edgeOffsets1);
            edgeOffsets2 = Fma.MultiplyAdd(Vector256.ConvertToSingle(minY), edgeNormalsY2, edgeOffsets2);
            edgeOffsets3 = Fma.MultiplyAdd(Vector256.ConvertToSingle(minY), edgeNormalsY3, edgeOffsets3);

            // Quantize slopes
            Vector256<int> slopeLookups0 = quantizeSlopeLookup(edgeNormalsX0, edgeNormalsY0);
            Vector256<int> slopeLookups1 = quantizeSlopeLookup(edgeNormalsX1, edgeNormalsY1);
            Vector256<int> slopeLookups2 = quantizeSlopeLookup(edgeNormalsX2, edgeNormalsY2);
            Vector256<int> slopeLookups3 = quantizeSlopeLookup(edgeNormalsX3, edgeNormalsY3);

            Vector256<int> firstBlockIdx = Vector256.Add(Vector256.Multiply(minY.AsInt16(), Vector256.Create((int)m_blocksX).AsInt16()).AsInt32(), minX);

            firstBlockIdx.StoreAligned((int*)firstBlocks);

            rangeX.StoreAligned((int*)rangesX);

            rangeY.StoreAligned((int*)rangesY);

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

            Vector256<float> depthDx = Vector256.Create(V128Helper.PermuteFrom1(Vector128.LoadAligned((float*)(depthPlane + primitiveIdxTransposed))).ToScalar());
            Vector256<float> depthDy = Vector256.Create(V128Helper.PermuteFrom2(Vector128.LoadAligned((float*)(depthPlane + primitiveIdxTransposed))).ToScalar());

            Vector256<float> lineDepth =
              Fma.MultiplyAdd(depthDx, depthSamplePosFactor1,
                Fma.MultiplyAdd(depthDy, depthSamplePosFactor2,
                  Vector256.Create(depthPlane[primitiveIdxTransposed].ToScalar())));

            Vector128<int> slopeLookup = Vector128.LoadAligned((int*)(slopeLookups + primitiveIdxTransposed));
            Vector128<float> edgeNormalX = Vector128.LoadAligned((float*)(edgeNormalsX + primitiveIdxTransposed));
            Vector128<float> edgeNormalY = Vector128.LoadAligned((float*)(edgeNormalsY + primitiveIdxTransposed));
            Vector128<float> lineOffset = Vector128.LoadAligned((float*)(edgeOffsets + primitiveIdxTransposed));

            uint blocksX = m_blocksX;

            uint firstBlock = firstBlocks[primitiveIdx];
            uint blockRangeX = rangesX[primitiveIdx];
            uint blockRangeY = rangesY[primitiveIdx];

            ushort* pPrimitiveHiZ = pHiZBuffer + firstBlock;
            Vector256<int>* pPrimitiveOut = (Vector256<int>*)pDepthBuffer + 4 * firstBlock;

            PrimitiveMode primitiveMode = (PrimitiveMode)primModes[primitiveIdx];

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
                  depth += depthDx,
                  offset += edgeNormalX)
                {
                    ushort hiZ = *pBlockRowHiZ;
                    if (hiZ >= primitiveMaxZ)
                    {
                        continue;
                    }

                    ulong blockMask;
                    if (primitiveMode == PrimitiveMode.Convex)    // 83-97%
                    {
                        // Simplified conservative test: combined block mask will be zero if any offset is outside of range
                        Vector128<float> anyOffsetOutsideMask = Vector128.GreaterThanOrEqual(offset, Vector128.Create((float)(OFFSET_QUANTIZATION_FACTOR - 1)));
                        if (!V128Helper.TestZ(anyOffsetOutsideMask, anyOffsetOutsideMask))
                        {
                            if (anyBlockHit)
                            {
                                // Convexity implies we won't hit another block in this row and can skip to the next line.
                                break;
                            }
                            continue;
                        }

                        anyBlockHit = true;

                        Vector128<int> offsetClamped = Vector128.Max(Vector128.ConvertToInt32(offset), Vector128<int>.Zero);

                        Vector128<int> lookup = Vector128.BitwiseOr(slopeLookup, offsetClamped);

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
                        Vector128<int> offsetClamped = Vector128.Min(Vector128.Max(Vector128.ConvertToInt32(offset), Vector128<int>.Zero), Vector128.Create(OFFSET_QUANTIZATION_FACTOR - 1));
                        Vector128<int> lookup = Vector128.BitwiseOr(slopeLookup, offsetClamped);

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
                            case PrimitiveMode.Triangle0:             // 2.3-11%
                                blockMask = A & B & C;
                                break;

                            case PrimitiveMode.Triangle1:             // 0.1-4%
                                blockMask = A & C & D;
                                break;

                            case PrimitiveMode.ConcaveRight:          // 0.01-0.9%
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
                                goto case PrimitiveMode.ConcaveLeft;

                            case PrimitiveMode.ConcaveLeft:           // 0.01-0.6%
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
                    Vector256<float> depth1 = Fma.MultiplyAdd(depthDx, Vector256.Create(0.5f), depth0);
                    Vector256<float> depth8 = Vector256.Add(depthDy, depth0);
                    Vector256<float> depth9 = Vector256.Add(depthDy, depth1);

                    // Pack depth
                    Vector256<ushort> d0 = packDepthPremultiplied(depth0, depth1);
                    Vector256<ushort> d4 = packDepthPremultiplied(depth8, depth9);

                    // Interpolate remaining values in packed space
                    Vector256<ushort> d2 = V256Helper.Average(d0, d4);
                    Vector256<ushort> d1 = V256Helper.Average(d0, d2);
                    Vector256<ushort> d3 = V256Helper.Average(d2, d4);

                    // Not all pixels covered - mask depth
                    if (blockMask != 0xffff_ffff_ffff_ffff)
                    {
                        Vector128<int> A = Vector128.CreateScalar((long)blockMask).AsInt32();
                        Vector128<int> B = Vector128.ShiftLeft(A.AsInt16(), 4).AsInt32();
                        Vector256<int> C = Vector256.Create(A, B);
                        Vector256<short> rowMask = V256Helper.UnpackLow(C.AsByte(), C.AsByte()).AsInt16();

                        d0 = V256Helper.BlendVariable(Vector256<byte>.Zero, d0.AsByte(), Vector256.ShiftLeft(rowMask, 3).AsByte()).AsUInt16();
                        d1 = V256Helper.BlendVariable(Vector256<byte>.Zero, d1.AsByte(), Vector256.ShiftLeft(rowMask, 2).AsByte()).AsUInt16();
                        d2 = V256Helper.BlendVariable(Vector256<byte>.Zero, d2.AsByte(), Vector256.Add(rowMask, rowMask).AsByte()).AsUInt16();
                        d3 = V256Helper.BlendVariable(Vector256<byte>.Zero, d3.AsByte(), rowMask.AsByte()).AsUInt16();
                    }

                    // Test fast clear flag
                    if (hiZ != 1)
                    {
                        // Merge depth values
                        d0 = Vector256.Max(Vector256.LoadAligned((ushort*)(@out + 0)), d0);
                        d1 = Vector256.Max(Vector256.LoadAligned((ushort*)(@out + 1)), d1);
                        d2 = Vector256.Max(Vector256.LoadAligned((ushort*)(@out + 2)), d2);
                        d3 = Vector256.Max(Vector256.LoadAligned((ushort*)(@out + 3)), d3);
                    }

                    // Store back new depth
                    d0.StoreAligned((ushort*)(@out + 0));
                    d1.StoreAligned((ushort*)(@out + 1));
                    d2.StoreAligned((ushort*)(@out + 2));
                    d3.StoreAligned((ushort*)(@out + 3));

                    // Update HiZ
                    Vector256<ushort> newMinZ = Vector256.Min(Vector256.Min(d0, d1), Vector256.Min(d2, d3));
                    ushort newMinZ16 = V128Helper.MinHorizontal(Vector128.Min(newMinZ.GetLower(), newMinZ.GetUpper()));

                    *pBlockRowHiZ = newMinZ16;
                }
            }
        }
    }
}