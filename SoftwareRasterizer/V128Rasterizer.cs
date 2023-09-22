using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

using V128 = Vector128;

using static VectorMath;

public unsafe class V128Rasterizer<Fma> : Rasterizer
    where Fma : IFusedMultiplyAdd128
{
    private const int Alignment = 128 / 8; // sizeof(Vector128<>)

    public V128Rasterizer(RasterizationTable rasterizationTable, uint width, uint height) :
        base(rasterizationTable, width, height, Alignment)
    {
    }

    public static V128Rasterizer<Fma> Create(RasterizationTable rasterizationTable, uint width, uint height)
    {
        bool success = false;
        rasterizationTable.DangerousAddRef(ref success);
        if (success)
        {
            return new V128Rasterizer<Fma>(rasterizationTable, width, height);
        }
        throw new ObjectDisposedException(rasterizationTable.GetType().Name);
    }

    public override unsafe void setModelViewProjection(float* matrix)
    {
        Vector128<float> mat0 = V128.Load(matrix + 0);
        Vector128<float> mat1 = V128.Load(matrix + 4);
        Vector128<float> mat2 = V128.Load(matrix + 8);
        Vector128<float> mat3 = V128.Load(matrix + 12);

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Store rows
        V128.StoreAligned(mat0, m_modelViewProjectionRaw + 0);
        V128.StoreAligned(mat1, m_modelViewProjectionRaw + 4);
        V128.StoreAligned(mat2, m_modelViewProjectionRaw + 8);
        V128.StoreAligned(mat3, m_modelViewProjectionRaw + 12);

        // Bake viewport transform into matrix and 6shift by half a block
        mat0 = V128.Multiply(V128.Add(mat0, mat3), V128.Create(m_width * 0.5f - 4.0f));
        mat1 = V128.Multiply(V128.Add(mat1, mat3), V128.Create(m_height * 0.5f - 4.0f));

        // Map depth from [-1, 1] to [bias, 0]
        mat2 = V128.Multiply(V128.Subtract(mat3, mat2), V128.Create(0.5f * floatCompressionBias));

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Store prebaked cols
        V128.StoreAligned(mat0, m_modelViewProjection + 0);
        V128.StoreAligned(mat1, m_modelViewProjection + 4);
        V128.StoreAligned(mat2, m_modelViewProjection + 8);
        V128.StoreAligned(mat3, m_modelViewProjection + 12);
    }

    public override void clear()
    {
        // Mark blocks as cleared by setting Hi Z to 1 (one unit separated from far plane). 
        // This value is extremely unlikely to occur during normal rendering, so we don't
        // need to guard against a HiZ of 1 occuring naturally. This is different from a value of 0, 
        // which will occur every time a block is partially covered for the first time.
        Vector128<int> clearValue = V128.Create((short)1).AsInt32();
        uint count = m_hiZ_Size / 8;
        Vector128<int>* pHiZ = (Vector128<int>*)m_hiZ;
        for (uint offset = 0; offset < count; ++offset)
        {
            V128.StoreAligned(clearValue, (int*)pHiZ);
            pHiZ++;
        }
    }

    public override bool queryVisibility(Vector4 vBoundsMin, Vector4 vBoundsMax, out bool needsClipping)
    {
        Vector128<float> boundsMin = vBoundsMin.AsVector128();
        Vector128<float> boundsMax = vBoundsMax.AsVector128();

        // Frustum cull
        Vector128<float> extents = V128.Subtract(boundsMax, boundsMin);
        Vector128<float> center = V128.Add(boundsMax, boundsMin); // Bounding box center times 2 - but since W = 2, the plane equations work out correctly
        Vector128<float> minusZero = V128.Create(-0.0f);

        Vector128<float> row0 = V128.LoadAligned(m_modelViewProjectionRaw + 0);
        Vector128<float> row1 = V128.LoadAligned(m_modelViewProjectionRaw + 4);
        Vector128<float> row2 = V128.LoadAligned(m_modelViewProjectionRaw + 8);
        Vector128<float> row3 = V128.LoadAligned(m_modelViewProjectionRaw + 12);

        // Compute distance from each frustum plane
        Vector128<float> plane0 = V128.Add(row3, row0);
        Vector128<float> offset0 = V128.Add(center, V128.Xor(extents, V128.BitwiseAnd(plane0, minusZero)));
        Vector128<float> dist0 = V128.Create(V128.Dot(plane0, offset0));

        Vector128<float> plane1 = V128.Subtract(row3, row0);
        Vector128<float> offset1 = V128.Add(center, V128.Xor(extents, V128.BitwiseAnd(plane1, minusZero)));
        Vector128<float> dist1 = V128.Create(V128.Dot(plane1, offset1));

        Vector128<float> plane2 = V128.Add(row3, row1);
        Vector128<float> offset2 = V128.Add(center, V128.Xor(extents, V128.BitwiseAnd(plane2, minusZero)));
        Vector128<float> dist2 = V128.Create(V128.Dot(plane2, offset2));

        Vector128<float> plane3 = V128.Subtract(row3, row1);
        Vector128<float> offset3 = V128.Add(center, V128.Xor(extents, V128.BitwiseAnd(plane3, minusZero)));
        Vector128<float> dist3 = V128.Create(V128.Dot(plane3, offset3));

        Vector128<float> plane4 = V128.Add(row3, row2);
        Vector128<float> offset4 = V128.Add(center, V128.Xor(extents, V128.BitwiseAnd(plane4, minusZero)));
        Vector128<float> dist4 = V128.Create(V128.Dot(plane4, offset4));

        Vector128<float> plane5 = V128.Subtract(row3, row2);
        Vector128<float> offset5 = V128.Add(center, V128.Xor(extents, V128.BitwiseAnd(plane5, minusZero)));
        Vector128<float> dist5 = V128.Create(V128.Dot(plane5, offset5));

        // Combine plane distance signs
        Vector128<float> combined = V128.BitwiseOr(V128.BitwiseOr(V128.BitwiseOr(dist0, dist1), V128.BitwiseOr(dist2, dist3)), V128.BitwiseOr(dist4, dist5));

        // Can't use Sse41.TestZ or _mm_comile_ss here because the OR's above created garbage in the non-sign bits
        if (V128.ExtractMostSignificantBits(combined) != 0)
        {
            needsClipping = false;
            return false;
        }

        // Load prebaked projection matrix
        Vector128<float> col0 = V128.LoadAligned(m_modelViewProjection + 0);
        Vector128<float> col1 = V128.LoadAligned(m_modelViewProjection + 4);
        Vector128<float> col2 = V128.LoadAligned(m_modelViewProjection + 8);
        Vector128<float> col3 = V128.LoadAligned(m_modelViewProjection + 12);

        // Transform edges
        Vector128<float> egde0 = V128.Multiply(col0, V128.Create(extents.ToScalar()));
        Vector128<float> egde1 = V128.Multiply(col1, V128Helper.PermuteFrom1(extents));
        Vector128<float> egde2 = V128.Multiply(col2, V128Helper.PermuteFrom2(extents));

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
          Fma.MultiplyAdd(col0, V128.Create(boundsMin.ToScalar()),
            Fma.MultiplyAdd(col1, V128Helper.PermuteFrom1(boundsMin),
              Fma.MultiplyAdd(col2, V128Helper.PermuteFrom2(boundsMin),
                col3)));

        // Transform remaining corners by adding edge vectors
        corners1 = V128.Add(corners0, egde0);
        corners2 = V128.Add(corners0, egde1);
        corners4 = V128.Add(corners0, egde2);

        corners3 = V128.Add(corners1, egde1);
        corners5 = V128.Add(corners4, egde0);
        corners6 = V128.Add(corners2, egde2);

        corners7 = V128.Add(corners6, egde0);

        // Transpose into SoA
        _MM_TRANSPOSE4_PS(ref corners0, ref corners1, ref corners2, ref corners3);
        _MM_TRANSPOSE4_PS(ref corners4, ref corners5, ref corners6, ref corners7);

        // Even if all bounding box corners have W > 0 here, we may end up with some vertices with W < 0 to due floating point differences; so test with some epsilon if any W < 0.
        Vector128<float> maxExtent = V128.Max(extents, V128.Shuffle(extents, V128.Create(2, 3, 0, 1)));
        maxExtent = V128.Max(maxExtent, V128.Shuffle(maxExtent, V128.Create(1, 0, 3, 2)));

        Vector128<float> nearPlaneEpsilon = V128.Multiply(maxExtent, V128.Create(0.001f));
        Vector128<float> closeToNearPlane = V128.BitwiseOr(V128.LessThan(corners3, nearPlaneEpsilon), V128.LessThan(corners7, nearPlaneEpsilon));
        if (!V128Helper.TestZ(closeToNearPlane, closeToNearPlane))
        {
            needsClipping = true;
            return true;
        }

        needsClipping = false;

        // Perspective division
        corners3 = V128Helper.Reciprocal(corners3);
        corners0 = V128.Multiply(corners0, corners3);
        corners1 = V128.Multiply(corners1, corners3);
        corners2 = V128.Multiply(corners2, corners3);

        corners7 = V128Helper.Reciprocal(corners7);
        corners4 = V128.Multiply(corners4, corners7);
        corners5 = V128.Multiply(corners5, corners7);
        corners6 = V128.Multiply(corners6, corners7);

        // Vertical mins and maxes
        Vector128<float> minsX = V128.Min(corners0, corners4);
        Vector128<float> maxsX = V128.Max(corners0, corners4);

        Vector128<float> minsY = V128.Min(corners1, corners5);
        Vector128<float> maxsY = V128.Max(corners1, corners5);

        // Horizontal reduction, step 1
        Vector128<float> minsXY = V128.Min(V128Helper.UnpackLow(minsX, minsY), V128Helper.UnpackHigh(minsX, minsY));
        Vector128<float> maxsXY = V128.Max(V128Helper.UnpackLow(maxsX, maxsY), V128Helper.UnpackHigh(maxsX, maxsY));

        // Clamp bounds
        minsXY = V128.Max(minsXY, Vector128<float>.Zero);
        maxsXY = V128.Min(maxsXY, V128.Create(m_width - 1f, m_height - 1f, m_width - 1f, m_height - 1f));

        // Negate maxes so we can round in the same direction
        maxsXY = V128.Xor(maxsXY, minusZero);

        // Horizontal reduction, step 2
        Vector128<float> boundsF = V128.Min(V128Helper.UnpackLow(minsXY, maxsXY), V128Helper.UnpackHigh(minsXY, maxsXY));

        // Round towards -infinity and convert to int
        Vector128<int> boundsI = V128.ConvertToInt32(V128.Floor(boundsF));

        // Store as scalars
        int* bounds = stackalloc int[4];
        V128.Store(boundsI, bounds);

        uint minX = (uint)bounds[0];
        uint maxX = (uint)bounds[1];
        uint minY = (uint)bounds[2];
        uint maxY = (uint)bounds[3];

        // Revert the sign change we did for the maxes
        maxX = (uint)-(int)maxX;
        maxY = (uint)-(int)maxY;

        // No intersection between quad and screen area
        if (minX >= maxX || minY >= maxY)
        {
            return false;
        }

        Vector128<ushort> depth = packDepthPremultiplied(corners2, corners6);

        ushort maxZ = (ushort)(0xFFFFu ^ V128Helper.MinHorizontal(V128.Xor(depth, V128.Create((short)-1).AsUInt16())));

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

        Vector128<ushort> maxZV = V128.Create((ushort)maxZ);

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

                    Vector128<int> notVisible = V128.Equals(V128.Min(rowDepth.AsUInt16(), maxZV), maxZV).AsInt32();

                    uint visiblePixelMask = ~V128.ExtractMostSignificantBits(notVisible.AsByte());

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
            sizeof(float) * 4 * 16 * 1; // Vector128<float>[16] x 1

        byte* stackBuffer = stackalloc byte[stackBufferSize];
        byte* alignedBuffer = (byte*)((nint)(stackBuffer + (Alignment - 1)) & -Alignment);

        Vector128<float>* linDepthA = (Vector128<float>*)alignedBuffer;

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

                Vector128<float> vBias = V128.Create(bias);
                Vector128<float> vOne = V128.Create(1.0f);
                Vector128<float> vDiv = V128.Create(100 * 256 * 2 * 0.25f);
                Vector128<float> vSt = V128.Create(0.25f + 1000.0f);
                Vector128<float> vSf = V128.Create(1000.0f - 0.25f);

                Vector128<int>* source = &m_depthBuffer[8 * (blockY * m_blocksX + blockX)];
                for (uint y = 0; y < 8; ++y)
                {
                    Vector128<int> depthI = V128.LoadAligned((int*)source++);

                    Vector128<int> depthILo = V128.ShiftLeft(V128Helper.ConvertToInt32(depthI.AsUInt16()), 12);
                    Vector128<int> depthIHi = V128.ShiftLeft(V128Helper.ConvertToInt32(V128.Shuffle(depthI, V128.Create(2, 3, 0, 0)).AsUInt16()), 12);

                    Vector128<float> depthLo = V128.Multiply(depthILo.AsSingle(), vBias);
                    Vector128<float> depthHi = V128.Multiply(depthIHi.AsSingle(), vBias);

                    Vector128<float> linDepthLo = V128.Divide(vDiv, V128.Subtract(vSt, V128.Multiply(V128.Subtract(vOne, depthLo), vSf)));
                    Vector128<float> linDepthHi = V128.Divide(vDiv, V128.Subtract(vSt, V128.Multiply(V128.Subtract(vOne, depthHi), vSf)));

                    V128.StoreAligned(linDepthLo, (float*)(linDepthA + y * 2 + 0));
                    V128.StoreAligned(linDepthHi, (float*)(linDepthA + y * 2 + 1));
                }

                Vector128<float> vRcp100 = V128.Create(1.0f / 100.0f);
                Vector128<ushort> vZeroMax = V128Helper.UnpackLow(Vector128<byte>.Zero, Vector128<byte>.AllBitsSet).AsUInt16();
                Vector128<ushort> vMask = V128.Create((ushort)0xff);

                for (uint y = 0; y < 8; y += 2)
                {
                    Vector128<float> depth0 = V128.LoadAligned((float*)(linDepthA + y * 2 + 0));
                    Vector128<float> depth1 = V128.LoadAligned((float*)(linDepthA + y * 2 + 1));
                    Vector128<float> depth2 = V128.LoadAligned((float*)(linDepthA + y * 2 + 2));
                    Vector128<float> depth3 = V128.LoadAligned((float*)(linDepthA + y * 2 + 3));

                    Vector128<int> vR32_0 = V128.ConvertToInt32(V128.Multiply(depth0, vRcp100));
                    Vector128<int> vR32_1 = V128.ConvertToInt32(V128.Multiply(depth1, vRcp100));
                    Vector128<int> vR32_2 = V128.ConvertToInt32(V128.Multiply(depth2, vRcp100));
                    Vector128<int> vR32_3 = V128.ConvertToInt32(V128.Multiply(depth3, vRcp100));

                    Vector128<ushort> vR16_0 = V128.BitwiseAnd(V128Helper.PackUnsignedSaturate(vR32_0, vR32_1), vMask);
                    Vector128<ushort> vR16_1 = V128.BitwiseAnd(V128Helper.PackUnsignedSaturate(vR32_2, vR32_3), vMask);
                    Vector128<byte> vR8 = V128Helper.PackUnsignedSaturate(vR16_0.AsInt16(), vR16_1.AsInt16());

                    Vector128<int> vG32_0 = V128.ConvertToInt32(depth0);
                    Vector128<int> vG32_1 = V128.ConvertToInt32(depth1);
                    Vector128<int> vG32_2 = V128.ConvertToInt32(depth2);
                    Vector128<int> vG32_3 = V128.ConvertToInt32(depth3);

                    Vector128<ushort> vG16_0 = V128.BitwiseAnd(V128Helper.PackUnsignedSaturate(vG32_0, vG32_1), vMask);
                    Vector128<ushort> vG16_1 = V128.BitwiseAnd(V128Helper.PackUnsignedSaturate(vG32_2, vG32_3), vMask);
                    Vector128<byte> vG8 = V128Helper.PackUnsignedSaturate(vG16_0.AsInt16(), vG16_1.AsInt16());

                    Vector128<ushort> vRG_Lo = V128Helper.UnpackLow(vR8, vG8).AsUInt16();
                    Vector128<ushort> vRG_Hi = V128Helper.UnpackHigh(vR8, vG8).AsUInt16();

                    Vector128<uint> result1 = V128Helper.UnpackLow(vRG_Lo, vZeroMax).AsUInt32();
                    Vector128<uint> result2 = V128Helper.UnpackHigh(vRG_Lo, vZeroMax).AsUInt32();
                    Vector128<uint> result3 = V128Helper.UnpackLow(vRG_Hi, vZeroMax).AsUInt32();
                    Vector128<uint> result4 = V128Helper.UnpackHigh(vRG_Hi, vZeroMax).AsUInt32();

                    V128.StoreAligned(result1, (uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 0))) + 0);
                    V128.StoreAligned(result2, (uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 0))) + 4);
                    V128.StoreAligned(result3, (uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 1))) + 0);
                    V128.StoreAligned(result4, (uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 1))) + 4);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void transpose256(
        Vector128<float> A,
        Vector128<float> B,
        Vector128<float> C,
        Vector128<float> D,
        Vector128<float>* @out)
    {
        Vector128<float> _Tmp0 = V128Helper.CombineLower(A, B);
        Vector128<float> _Tmp2 = V128Helper.CombineUpper(A, B);
        Vector128<float> _Tmp1 = V128Helper.CombineLower(C, D);
        Vector128<float> _Tmp3 = V128Helper.CombineUpper(C, D);

        Vector128<float> tA = V128Helper.UnzipEven(_Tmp0, _Tmp1);
        Vector128<float> tB = V128Helper.UnzipOdd(_Tmp0, _Tmp1);
        Vector128<float> tC = V128Helper.UnzipEven(_Tmp2, _Tmp3);
        Vector128<float> tD = V128Helper.UnzipOdd(_Tmp2, _Tmp3);

        V128.StoreAligned(tA, (float*)(@out + 0));
        V128.StoreAligned(tB, (float*)(@out + 2));
        V128.StoreAligned(tC, (float*)(@out + 4));
        V128.StoreAligned(tD, (float*)(@out + 6));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void transpose256i(
        Vector128<int> A,
        Vector128<int> B,
        Vector128<int> C,
        Vector128<int> D,
        Vector128<int>* @out)
    {
        Vector128<long> _Tmp0 = V128Helper.UnpackLow(A, B).AsInt64();
        Vector128<long> _Tmp1 = V128Helper.UnpackLow(C, D).AsInt64();
        Vector128<long> _Tmp2 = V128Helper.UnpackHigh(A, B).AsInt64();
        Vector128<long> _Tmp3 = V128Helper.UnpackHigh(C, D).AsInt64();

        Vector128<int> tA = V128Helper.UnpackLow(_Tmp0, _Tmp1).AsInt32();
        Vector128<int> tB = V128Helper.UnpackHigh(_Tmp0, _Tmp1).AsInt32();
        Vector128<int> tC = V128Helper.UnpackLow(_Tmp2, _Tmp3).AsInt32();
        Vector128<int> tD = V128Helper.UnpackHigh(_Tmp2, _Tmp3).AsInt32();

        V128.StoreAligned(tA, (int*)(@out + 0));
        V128.StoreAligned(tB, (int*)(@out + 2));
        V128.StoreAligned(tC, (int*)(@out + 4));
        V128.StoreAligned(tD, (int*)(@out + 6));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void normalizeEdge(ref Vector128<float> nx, ref Vector128<float> ny, Vector128<float> mul)
    {
        Vector128<float> minusZero = V128.Create(-0.0f);
        Vector128<float> invLen = V128Helper.Reciprocal(V128.Add(V128.AndNot(nx, minusZero), V128.AndNot(ny, minusZero)));

        invLen = V128.Multiply(mul, invLen);
        nx = V128.Multiply(nx, invLen);
        ny = V128.Multiply(ny, invLen);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> quantizeSlopeLookup(Vector128<float> nx, Vector128<float> ny)
    {
        Vector128<int> yNeg = V128.LessThanOrEqual(ny, Vector128<float>.Zero).AsInt32();

        // Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
        const float maxOffset = -minEdgeOffset;
        const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f / ((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        const float add = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f + 0.5f;

        Vector128<int> quantizedSlope = V128.ConvertToInt32(Fma.MultiplyAdd(nx, V128.Create(mul), V128.Create(add)));
        return V128.ShiftLeft(V128.Subtract(V128.ShiftLeft(quantizedSlope, 1), yNeg), OFFSET_QUANTIZATION_BITS);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> packDepthPremultiplied(Vector128<float> depthA, Vector128<float> depthB)
    {
        Vector128<int> x1 = V128.ShiftRightArithmetic(depthA.AsInt32(), 12);
        Vector128<int> x2 = V128.ShiftRightArithmetic(depthB.AsInt32(), 12);
        return V128Helper.PackUnsignedSaturate(x1, x2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector64<ushort> packDepthPremultiplied(Vector128<float> depth)
    {
        Vector128<int> x = V128.ShiftRightArithmetic(depth.AsInt32(), 12);
        return V128Helper.PackUnsignedSaturate(x);
    }

    public override void rasterize<T>(in Occluder occluder)
    {
        Vector128<int>* vertexData = (Vector128<int>*)occluder.m_vertexData;
        uint packetCount = occluder.m_packetCount;

        // Note that unaligned loads do not have a latency penalty on CPUs with SSE4 support
        Vector128<float> mat0 = V128.LoadAligned(m_modelViewProjection + 0);
        Vector128<float> mat1 = V128.LoadAligned(m_modelViewProjection + 4);
        Vector128<float> mat2 = V128.LoadAligned(m_modelViewProjection + 8);
        Vector128<float> mat3 = V128.LoadAligned(m_modelViewProjection + 12);

        Vector128<float> boundsMin = occluder.m_refMin.AsVector128();
        Vector128<float> boundsExtents = V128.Subtract(occluder.m_refMax.AsVector128(), boundsMin);

        // Bake integer => bounding box transform into matrix
        mat3 =
          Fma.MultiplyAdd(mat0, V128Helper.PermuteFrom0(boundsMin),
            Fma.MultiplyAdd(mat1, V128Helper.PermuteFrom1(boundsMin),
              Fma.MultiplyAdd(mat2, V128Helper.PermuteFrom2(boundsMin),
                mat3)));

        mat0 = V128.Multiply(mat0, V128.Multiply(V128Helper.PermuteFrom0(boundsExtents), V128.Create(1.0f / (2047ul << 21))));
        mat1 = V128.Multiply(mat1, V128.Multiply(V128Helper.PermuteFrom1(boundsExtents), V128.Create(1.0f / (2047 << 10))));
        mat2 = V128.Multiply(mat2, V128.Multiply(V128Helper.PermuteFrom2(boundsExtents), V128.Create(1.0f / 1023)));

        // Bias X coordinate back into positive range
        mat3 = Fma.MultiplyAdd(mat0, V128.Create((float)(1024ul << 21)), mat3);

        // Skew projection to correct bleeding of Y and Z into X due to lack of masking
        mat1 = V128.Subtract(mat1, mat0);
        mat2 = V128.Subtract(mat2, mat0);

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Due to linear relationship between Z and W, it's cheaper to compute Z from W later in the pipeline than using the full projection matrix up front
        float c0, c1;
        {
            Vector128<float> Za = V128Helper.PermuteFrom3(mat2);
            Vector128<float> Zb = V128.Create(V128.Dot(mat2, V128.Create((float)(1 << 21), 1 << 10, 1, 1)));

            Vector128<float> Wa = V128Helper.PermuteFrom3(mat3);
            Vector128<float> Wb = V128.Create(V128.Dot(mat3, V128.Create((float)(1 << 21), 1 << 10, 1, 1)));

            c0 = V128.Divide(V128.Subtract(Za, Zb), V128.Subtract(Wa, Wb)).ToScalar();
            c1 = Fma.MultiplyAddNegated(V128.Divide(V128.Subtract(Za, Zb), V128.Subtract(Wa, Wb)), Wa, Za).ToScalar();
        }

        const int alignment = 256 / 8;
        const int stackBufferSize =
            alignment - 1 +
            sizeof(uint) * 8 * 4 + // uint[8] x 4
            sizeof(float) * 4 * 8 * 4 + // Vector128<float>[8] x 4
            sizeof(int) * 4 * 8 * 1 + // Vector128<int>[8] x 1
            sizeof(ushort) * 8 * 1 + // ushort[8] x 1
            sizeof(int) * 4 * 1; // int[4] x 1

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
        alignedBuffer += sizeof(ushort) * 8;

        int* modeTableBuffer = (int*)alignedBuffer;

        for (uint packetIdx = 0; packetIdx < packetCount; packetIdx += 4)
        {
            Vector128<int>* vertexPartData = vertexData + packetIdx * 2;

            int p0 = RenderPacketPart<T>(
                mat0,
                mat1,
                mat3,
                c0,
                c1,
                vertexPartData,
                primModes,
                firstBlocks,
                rangesX,
                rangesY,
                depthPlane,
                edgeNormalsX,
                edgeNormalsY,
                edgeOffsets,
                slopeLookups,
                depthBounds,
                modeTableBuffer,
                out uint validMask0);

            int p1 = RenderPacketPart<T>(
                mat0,
                mat1,
                mat3,
                c0,
                c1,
                vertexPartData + 1,
                primModes + 4,
                firstBlocks + 4,
                rangesX + 4,
                rangesY + 4,
                depthPlane + 1,
                edgeNormalsX + 1,
                edgeNormalsY + 1,
                edgeOffsets + 1,
                slopeLookups + 1,
                depthBounds + 4,
                modeTableBuffer,
                out uint validMask1);

            if (p0 == p1 && p0 != 0)
            {
                continue;
            }

            uint validMask = validMask0 | (validMask1 << 4);

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

    private int RenderPacketPart<T>(
        Vector128<float> mat0,
        Vector128<float> mat1,
        Vector128<float> mat3,
        float c0,
        float c1,
        Vector128<int>* vertexData,
        uint* primModes,
        uint* firstBlocks,
        uint* rangesX,
        uint* rangesY,
        Vector128<float>* depthPlane,
        Vector128<float>* edgeNormalsX,
        Vector128<float>* edgeNormalsY,
        Vector128<float>* edgeOffsets,
        Vector128<int>* slopeLookups,
        ushort* depthBounds,
        int* modeTableBuffer,
        out uint overlappedPrimitiveValidMask)
        where T : IPossiblyNearClipped
    {
        // Load data - only needed once per frame, so use streaming load
        Vector128<int> I0 = V128.LoadAlignedNonTemporal((int*)(vertexData + 0));
        Vector128<int> I1 = V128.LoadAlignedNonTemporal((int*)(vertexData + 2));
        Vector128<int> I2 = V128.LoadAlignedNonTemporal((int*)(vertexData + 4));
        Vector128<int> I3 = V128.LoadAlignedNonTemporal((int*)(vertexData + 6));

        // Vertex transformation - first W, then X & Y after camera plane culling, then Z after backface culling
        Vector128<float> Xf0 = V128.ConvertToSingle(I0);
        Vector128<float> Xf1 = V128.ConvertToSingle(I1);
        Vector128<float> Xf2 = V128.ConvertToSingle(I2);
        Vector128<float> Xf3 = V128.ConvertToSingle(I3);

        Vector128<int> maskY = V128.Create(2047 << 10);
        Vector128<float> Yf0 = V128.ConvertToSingle(V128.BitwiseAnd(I0, maskY));
        Vector128<float> Yf1 = V128.ConvertToSingle(V128.BitwiseAnd(I1, maskY));
        Vector128<float> Yf2 = V128.ConvertToSingle(V128.BitwiseAnd(I2, maskY));
        Vector128<float> Yf3 = V128.ConvertToSingle(V128.BitwiseAnd(I3, maskY));

        Vector128<int> maskZ = V128.Create(1023);
        Vector128<float> Zf0 = V128.ConvertToSingle(V128.BitwiseAnd(I0, maskZ));
        Vector128<float> Zf1 = V128.ConvertToSingle(V128.BitwiseAnd(I1, maskZ));
        Vector128<float> Zf2 = V128.ConvertToSingle(V128.BitwiseAnd(I2, maskZ));
        Vector128<float> Zf3 = V128.ConvertToSingle(V128.BitwiseAnd(I3, maskZ));

        Vector128<float> mat00 = V128Helper.PermuteFrom0(mat0);
        Vector128<float> mat01 = V128Helper.PermuteFrom1(mat0);
        Vector128<float> mat02 = V128Helper.PermuteFrom2(mat0);
        Vector128<float> mat03 = V128Helper.PermuteFrom3(mat0);

        Vector128<float> X0 = Fma.MultiplyAdd(Xf0, mat00, Fma.MultiplyAdd(Yf0, mat01, Fma.MultiplyAdd(Zf0, mat02, mat03)));
        Vector128<float> X1 = Fma.MultiplyAdd(Xf1, mat00, Fma.MultiplyAdd(Yf1, mat01, Fma.MultiplyAdd(Zf1, mat02, mat03)));
        Vector128<float> X2 = Fma.MultiplyAdd(Xf2, mat00, Fma.MultiplyAdd(Yf2, mat01, Fma.MultiplyAdd(Zf2, mat02, mat03)));
        Vector128<float> X3 = Fma.MultiplyAdd(Xf3, mat00, Fma.MultiplyAdd(Yf3, mat01, Fma.MultiplyAdd(Zf3, mat02, mat03)));

        Vector128<float> mat10 = V128Helper.PermuteFrom0(mat1);
        Vector128<float> mat11 = V128Helper.PermuteFrom1(mat1);
        Vector128<float> mat12 = V128Helper.PermuteFrom2(mat1);
        Vector128<float> mat13 = V128Helper.PermuteFrom3(mat1);

        Vector128<float> Y0 = Fma.MultiplyAdd(Xf0, mat10, Fma.MultiplyAdd(Yf0, mat11, Fma.MultiplyAdd(Zf0, mat12, mat13)));
        Vector128<float> Y1 = Fma.MultiplyAdd(Xf1, mat10, Fma.MultiplyAdd(Yf1, mat11, Fma.MultiplyAdd(Zf1, mat12, mat13)));
        Vector128<float> Y2 = Fma.MultiplyAdd(Xf2, mat10, Fma.MultiplyAdd(Yf2, mat11, Fma.MultiplyAdd(Zf2, mat12, mat13)));
        Vector128<float> Y3 = Fma.MultiplyAdd(Xf3, mat10, Fma.MultiplyAdd(Yf3, mat11, Fma.MultiplyAdd(Zf3, mat12, mat13)));

        Vector128<float> mat30 = V128Helper.PermuteFrom0(mat3);
        Vector128<float> mat31 = V128Helper.PermuteFrom1(mat3);
        Vector128<float> mat32 = V128Helper.PermuteFrom2(mat3);
        Vector128<float> mat33 = V128Helper.PermuteFrom3(mat3);

        Vector128<float> W0 = Fma.MultiplyAdd(Xf0, mat30, Fma.MultiplyAdd(Yf0, mat31, Fma.MultiplyAdd(Zf0, mat32, mat33)));
        Vector128<float> W1 = Fma.MultiplyAdd(Xf1, mat30, Fma.MultiplyAdd(Yf1, mat31, Fma.MultiplyAdd(Zf1, mat32, mat33)));
        Vector128<float> W2 = Fma.MultiplyAdd(Xf2, mat30, Fma.MultiplyAdd(Yf2, mat31, Fma.MultiplyAdd(Zf2, mat32, mat33)));
        Vector128<float> W3 = Fma.MultiplyAdd(Xf3, mat30, Fma.MultiplyAdd(Yf3, mat31, Fma.MultiplyAdd(Zf3, mat32, mat33)));

        Vector128<float> invW0, invW1, invW2, invW3;
        // Clamp W and invert
        if (T.PossiblyNearClipped)
        {
            Vector128<float> lowerBound = V128.Create((float)-maxInvW);
            Vector128<float> upperBound = V128.Create((float)+maxInvW);
            invW0 = V128.Min(upperBound, V128.Max(lowerBound, V128Helper.Reciprocal(W0)));
            invW1 = V128.Min(upperBound, V128.Max(lowerBound, V128Helper.Reciprocal(W1)));
            invW2 = V128.Min(upperBound, V128.Max(lowerBound, V128Helper.Reciprocal(W2)));
            invW3 = V128.Min(upperBound, V128.Max(lowerBound, V128Helper.Reciprocal(W3)));
        }
        else
        {
            invW0 = V128Helper.Reciprocal(W0);
            invW1 = V128Helper.Reciprocal(W1);
            invW2 = V128Helper.Reciprocal(W2);
            invW3 = V128Helper.Reciprocal(W3);
        }

        // Round to integer coordinates to improve culling of zero-area triangles
        Vector128<float> roundFactor = V128.Create(0.125f);
        Vector128<float> x0 = V128.Multiply(V128Helper.RoundToNearestInteger(V128.Multiply(X0, invW0)), roundFactor);
        Vector128<float> x1 = V128.Multiply(V128Helper.RoundToNearestInteger(V128.Multiply(X1, invW1)), roundFactor);
        Vector128<float> x2 = V128.Multiply(V128Helper.RoundToNearestInteger(V128.Multiply(X2, invW2)), roundFactor);
        Vector128<float> x3 = V128.Multiply(V128Helper.RoundToNearestInteger(V128.Multiply(X3, invW3)), roundFactor);

        Vector128<float> y0 = V128.Multiply(V128Helper.RoundToNearestInteger(V128.Multiply(Y0, invW0)), roundFactor);
        Vector128<float> y1 = V128.Multiply(V128Helper.RoundToNearestInteger(V128.Multiply(Y1, invW1)), roundFactor);
        Vector128<float> y2 = V128.Multiply(V128Helper.RoundToNearestInteger(V128.Multiply(Y2, invW2)), roundFactor);
        Vector128<float> y3 = V128.Multiply(V128Helper.RoundToNearestInteger(V128.Multiply(Y3, invW3)), roundFactor);

        // Compute unnormalized edge directions
        Vector128<float> edgeNormalsX0 = V128.Subtract(y1, y0);
        Vector128<float> edgeNormalsX1 = V128.Subtract(y2, y1);
        Vector128<float> edgeNormalsX2 = V128.Subtract(y3, y2);
        Vector128<float> edgeNormalsX3 = V128.Subtract(y0, y3);

        Vector128<float> edgeNormalsY0 = V128.Subtract(x0, x1);
        Vector128<float> edgeNormalsY1 = V128.Subtract(x1, x2);
        Vector128<float> edgeNormalsY2 = V128.Subtract(x2, x3);
        Vector128<float> edgeNormalsY3 = V128.Subtract(x3, x0);

        Vector128<float> area0 = Fma.MultiplySubtract(edgeNormalsX0, edgeNormalsY1, V128.Multiply(edgeNormalsX1, edgeNormalsY0));
        Vector128<float> area1 = Fma.MultiplySubtract(edgeNormalsX1, edgeNormalsY2, V128.Multiply(edgeNormalsX2, edgeNormalsY1));
        Vector128<float> area2 = Fma.MultiplySubtract(edgeNormalsX2, edgeNormalsY3, V128.Multiply(edgeNormalsX3, edgeNormalsY2));
        Vector128<float> area3 = V128.Subtract(V128.Add(area0, area2), area1);

        Vector128<float> minusZero128 = V128.Create(-0.0f);

        Vector128<float> wSign0, wSign1, wSign2, wSign3;
        if (T.PossiblyNearClipped)
        {
            wSign0 = V128.BitwiseAnd(invW0, minusZero128);
            wSign1 = V128.BitwiseAnd(invW1, minusZero128);
            wSign2 = V128.BitwiseAnd(invW2, minusZero128);
            wSign3 = V128.BitwiseAnd(invW3, minusZero128);
        }
        else
        {
            wSign0 = Vector128<float>.Zero;
            wSign1 = Vector128<float>.Zero;
            wSign2 = Vector128<float>.Zero;
            wSign3 = Vector128<float>.Zero;
        }

        // Compute signs of areas. We treat 0 as negative as this allows treating primitives with zero area as backfacing.
        Vector128<float> areaSign0, areaSign1, areaSign2, areaSign3;
        if (T.PossiblyNearClipped)
        {
            // Flip areas for each vertex with W < 0. This needs to be done before comparison against 0 rather than afterwards to make sure zero-are triangles are handled correctly.
            areaSign0 = V128.LessThanOrEqual(V128.Xor(V128.Xor(area0, wSign0), V128.Xor(wSign1, wSign2)), Vector128<float>.Zero);
            areaSign1 = V128.BitwiseAnd(minusZero128, V128.LessThanOrEqual(V128.Xor(V128.Xor(area1, wSign1), V128.Xor(wSign2, wSign3)), Vector128<float>.Zero));
            areaSign2 = V128.BitwiseAnd(minusZero128, V128.LessThanOrEqual(V128.Xor(V128.Xor(area2, wSign0), V128.Xor(wSign2, wSign3)), Vector128<float>.Zero));
            areaSign3 = V128.BitwiseAnd(minusZero128, V128.LessThanOrEqual(V128.Xor(V128.Xor(area3, wSign1), V128.Xor(wSign0, wSign3)), Vector128<float>.Zero));
        }
        else
        {
            areaSign0 = V128.LessThanOrEqual(area0, Vector128<float>.Zero);
            areaSign1 = V128.BitwiseAnd(minusZero128, V128.LessThanOrEqual(area1, Vector128<float>.Zero));
            areaSign2 = V128.BitwiseAnd(minusZero128, V128.LessThanOrEqual(area2, Vector128<float>.Zero));
            areaSign3 = V128.BitwiseAnd(minusZero128, V128.LessThanOrEqual(area3, Vector128<float>.Zero));
        }

        Vector128<int> config = V128.BitwiseOr(
          V128.BitwiseOr(V128.ShiftRightLogical(areaSign3.AsInt32(), 28), V128.ShiftRightLogical(areaSign2.AsInt32(), 29)),
          V128.BitwiseOr(V128.ShiftRightLogical(areaSign1.AsInt32(), 30), V128.ShiftRightLogical(areaSign0.AsInt32(), 31)));

        if (T.PossiblyNearClipped)
        {
            config = V128.BitwiseOr(config,
              V128.BitwiseOr(
                V128.BitwiseOr(V128.ShiftRightLogical(wSign3.AsInt32(), 24), V128.ShiftRightLogical(wSign2.AsInt32(), 25)),
                V128.BitwiseOr(V128.ShiftRightLogical(wSign1.AsInt32(), 26), V128.ShiftRightLogical(wSign0.AsInt32(), 27))));
        }

        V128.StoreAligned(config, modeTableBuffer);

        ref PrimitiveMode modeTablePtr = ref MemoryMarshal.GetReference(modeTable);
        modeTableBuffer[0] = (int)Unsafe.Add(ref modeTablePtr, modeTableBuffer[0]);
        modeTableBuffer[1] = (int)Unsafe.Add(ref modeTablePtr, modeTableBuffer[1]);
        modeTableBuffer[2] = (int)Unsafe.Add(ref modeTablePtr, modeTableBuffer[2]);
        modeTableBuffer[3] = (int)Unsafe.Add(ref modeTablePtr, modeTableBuffer[3]);

        Vector128<int> modes = V128.LoadAligned(modeTableBuffer);
        if (V128Helper.TestZ(modes, modes))
        {
            overlappedPrimitiveValidMask = 0;
            return 1;
        }

        Vector128<int> primitiveValid = V128.GreaterThan(modes, Vector128<int>.Zero);

        V128.StoreAligned(modes, (int*)primModes);

        Vector128<float> minFx, minFy, maxFx, maxFy;

        if (T.PossiblyNearClipped)
        {
            // Clipless bounding box computation
            Vector128<float> infP = V128.Create(+10000.0f);
            Vector128<float> infN = V128.Create(-10000.0f);

            // Find interval of points with W > 0
            Vector128<float> minPx0 = V128Helper.BlendVariable(x0, infP, wSign0);
            Vector128<float> minPx1 = V128Helper.BlendVariable(x1, infP, wSign1);
            Vector128<float> minPx2 = V128Helper.BlendVariable(x2, infP, wSign2);
            Vector128<float> minPx3 = V128Helper.BlendVariable(x3, infP, wSign3);

            Vector128<float> minPx = V128.Min(
              V128.Min(minPx0, minPx1),
              V128.Min(minPx2, minPx3));

            Vector128<float> minPy0 = V128Helper.BlendVariable(y0, infP, wSign0);
            Vector128<float> minPy1 = V128Helper.BlendVariable(y1, infP, wSign1);
            Vector128<float> minPy2 = V128Helper.BlendVariable(y2, infP, wSign2);
            Vector128<float> minPy3 = V128Helper.BlendVariable(y3, infP, wSign3);

            Vector128<float> minPy = V128.Min(
              V128.Min(minPy0, minPy1),
              V128.Min(minPy2, minPy3));

            Vector128<float> maxPx0 = V128.Xor(minPx0, wSign0);
            Vector128<float> maxPx1 = V128.Xor(minPx1, wSign1);
            Vector128<float> maxPx2 = V128.Xor(minPx2, wSign2);
            Vector128<float> maxPx3 = V128.Xor(minPx3, wSign3);

            Vector128<float> maxPx = V128.Max(
              V128.Max(maxPx0, maxPx1),
              V128.Max(maxPx2, maxPx3));

            Vector128<float> maxPy0 = V128.Xor(minPy0, wSign0);
            Vector128<float> maxPy1 = V128.Xor(minPy1, wSign1);
            Vector128<float> maxPy2 = V128.Xor(minPy2, wSign2);
            Vector128<float> maxPy3 = V128.Xor(minPy3, wSign3);

            Vector128<float> maxPy = V128.Max(
              V128.Max(maxPy0, maxPy1),
              V128.Max(maxPy2, maxPy3));

            // Find interval of points with W < 0
            Vector128<float> minNx0 = V128Helper.BlendVariable(infP, x0, wSign0);
            Vector128<float> minNx1 = V128Helper.BlendVariable(infP, x1, wSign1);
            Vector128<float> minNx2 = V128Helper.BlendVariable(infP, x2, wSign2);
            Vector128<float> minNx3 = V128Helper.BlendVariable(infP, x3, wSign3);

            Vector128<float> minNx = V128.Min(
              V128.Min(minNx0, minNx1),
              V128.Min(minNx2, minNx3));

            Vector128<float> minNy0 = V128Helper.BlendVariable(infP, y0, wSign0);
            Vector128<float> minNy1 = V128Helper.BlendVariable(infP, y1, wSign1);
            Vector128<float> minNy2 = V128Helper.BlendVariable(infP, y2, wSign2);
            Vector128<float> minNy3 = V128Helper.BlendVariable(infP, y3, wSign3);

            Vector128<float> minNy = V128.Min(
              V128.Min(minNy0, minNy1),
              V128.Min(minNy2, minNy3));

            Vector128<float> maxNx0 = V128Helper.BlendVariable(infN, x0, wSign0);
            Vector128<float> maxNx1 = V128Helper.BlendVariable(infN, x1, wSign1);
            Vector128<float> maxNx2 = V128Helper.BlendVariable(infN, x2, wSign2);
            Vector128<float> maxNx3 = V128Helper.BlendVariable(infN, x3, wSign3);

            Vector128<float> maxNx = V128.Max(
              V128.Max(maxNx0, maxNx1),
              V128.Max(maxNx2, maxNx3));

            Vector128<float> maxNy0 = V128Helper.BlendVariable(infN, y0, wSign0);
            Vector128<float> maxNy1 = V128Helper.BlendVariable(infN, y1, wSign1);
            Vector128<float> maxNy2 = V128Helper.BlendVariable(infN, y2, wSign2);
            Vector128<float> maxNy3 = V128Helper.BlendVariable(infN, y3, wSign3);

            Vector128<float> maxNy = V128.Max(
              V128.Max(maxNy0, maxNy1),
              V128.Max(maxNy2, maxNy3));

            // Include interval bounds resp. infinity depending on ordering of intervals
            Vector128<float> incAx = V128Helper.BlendVariable(minPx, infN, V128.GreaterThan(maxNx, minPx));
            Vector128<float> incAy = V128Helper.BlendVariable(minPy, infN, V128.GreaterThan(maxNy, minPy));

            Vector128<float> incBx = V128Helper.BlendVariable(maxPx, infP, V128.GreaterThan(maxPx, minNx));
            Vector128<float> incBy = V128Helper.BlendVariable(maxPy, infP, V128.GreaterThan(maxPy, minNy));

            minFx = V128.Min(incAx, incBx);
            minFy = V128.Min(incAy, incBy);

            maxFx = V128.Max(incAx, incBx);
            maxFy = V128.Max(incAy, incBy);
        }
        else
        {
            // Standard bounding box inclusion
            minFx = V128.Min(V128.Min(x0, x1), V128.Min(x2, x3));
            minFy = V128.Min(V128.Min(y0, y1), V128.Min(y2, y3));

            maxFx = V128.Max(V128.Max(x0, x1), V128.Max(x2, x3));
            maxFy = V128.Max(V128.Max(y0, y1), V128.Max(y2, y3));
        }

        // Clamp and round
        Vector128<int> minX, minY, maxX, maxY;
        minX = V128.Max(V128.ConvertToInt32(V128.Add(minFx, V128.Create(4.9999f / 8.0f))), Vector128<int>.Zero);
        minY = V128.Max(V128.ConvertToInt32(V128.Add(minFy, V128.Create(4.9999f / 8.0f))), Vector128<int>.Zero);
        maxX = V128.Min(V128.ConvertToInt32(V128.Add(maxFx, V128.Create(11.0f / 8.0f))), V128.Create((int)m_blocksX));
        maxY = V128.Min(V128.ConvertToInt32(V128.Add(maxFy, V128.Create(11.0f / 8.0f))), V128.Create((int)m_blocksY));

        // Check overlap between bounding box and frustum
        Vector128<int> inFrustum = V128.BitwiseAnd(V128.GreaterThan(maxX, minX), V128.GreaterThan(maxY, minY));
        Vector128<int> overlappedPrimitiveValid = V128.BitwiseAnd(inFrustum, primitiveValid);

        if (V128Helper.TestZ(overlappedPrimitiveValid, overlappedPrimitiveValid))
        {
            overlappedPrimitiveValidMask = 0;
            return 2;
        }

        overlappedPrimitiveValidMask = V128.ExtractMostSignificantBits(overlappedPrimitiveValid.AsSingle());

        // Convert bounds from [min, max] to [min, range]
        Vector128<int> rangeX = V128.Subtract(maxX, minX);
        Vector128<int> rangeY = V128.Subtract(maxY, minY);

        // Compute Z from linear relation with 1/W
        Vector128<float> C0 = V128.Create(c0);
        Vector128<float> C1 = V128.Create(c1);
        Vector128<float> z0, z1, z2, z3;
        z0 = Fma.MultiplyAdd(invW0, C1, C0);
        z1 = Fma.MultiplyAdd(invW1, C1, C0);
        z2 = Fma.MultiplyAdd(invW2, C1, C0);
        z3 = Fma.MultiplyAdd(invW3, C1, C0);

        Vector128<float> maxZ = V128.Max(V128.Max(z0, z1), V128.Max(z2, z3));

        // If any W < 0, assume maxZ = 1 (effectively disabling Hi-Z)
        if (T.PossiblyNearClipped)
        {
            maxZ = V128Helper.BlendVariable(maxZ, V128.Create(1.0f), V128.BitwiseOr(V128.BitwiseOr(wSign0, wSign1), V128.BitwiseOr(wSign2, wSign3)));
        }

        Vector64<ushort> packedDepthBounds = packDepthPremultiplied(maxZ);

        Vector64.Store(packedDepthBounds, depthBounds);

        // Compute screen space depth plane
        Vector128<float> greaterArea = V128.LessThan(V128.AndNot(area0, minusZero128), V128.AndNot(area2, minusZero128));

        // Force triangle area to be picked in the relevant mode.
        Vector128<float> modeTriangle0 = V128.Equals(modes, V128.Create((int)PrimitiveMode.Triangle0)).AsSingle();
        Vector128<float> modeTriangle1 = V128.Equals(modes, V128.Create((int)PrimitiveMode.Triangle1)).AsSingle();
        greaterArea = V128.AndNot(V128.BitwiseOr(modeTriangle1, greaterArea), modeTriangle0);

        Vector128<float> invArea;
        if (T.PossiblyNearClipped)
        {
            // Do a precise divison to reduce error in depth plane. Note that the area computed here
            // differs from the rasterized region if W < 0, so it can be very small for large covered screen regions.
            invArea = V128.Divide(V128.Create(1.0f), V128Helper.BlendVariable(area0, area2, greaterArea));
        }
        else
        {
            invArea = V128Helper.Reciprocal(V128Helper.BlendVariable(area0, area2, greaterArea));
        }

        Vector128<float> z12 = V128.Subtract(z1, z2);
        Vector128<float> z20 = V128.Subtract(z2, z0);
        Vector128<float> z30 = V128.Subtract(z3, z0);

        Vector128<float> edgeNormalsX4 = V128.Subtract(y0, y2);
        Vector128<float> edgeNormalsY4 = V128.Subtract(x2, x0);

        Vector128<float> depthPlane0, depthPlane1, depthPlane2;
        depthPlane1 = V128.Multiply(invArea, V128Helper.BlendVariable(Fma.MultiplySubtract(z20, edgeNormalsX1, V128.Multiply(z12, edgeNormalsX4)), Fma.MultiplyAddNegated(z20, edgeNormalsX3, V128.Multiply(z30, edgeNormalsX4)), greaterArea));
        depthPlane2 = V128.Multiply(invArea, V128Helper.BlendVariable(Fma.MultiplySubtract(z20, edgeNormalsY1, V128.Multiply(z12, edgeNormalsY4)), Fma.MultiplyAddNegated(z20, edgeNormalsY3, V128.Multiply(z30, edgeNormalsY4)), greaterArea));

        x0 = V128.Subtract(x0, V128.ConvertToSingle(minX));
        y0 = V128.Subtract(y0, V128.ConvertToSingle(minY));

        depthPlane0 = Fma.MultiplyAddNegated(x0, depthPlane1, Fma.MultiplyAddNegated(y0, depthPlane2, z0));

        // If mode == Triangle0, replace edge 2 with edge 4; if mode == Triangle1, replace edge 0 with edge 4
        edgeNormalsX2 = V128Helper.BlendVariable(edgeNormalsX2, edgeNormalsX4, modeTriangle0);
        edgeNormalsY2 = V128Helper.BlendVariable(edgeNormalsY2, edgeNormalsY4, modeTriangle0);
        edgeNormalsX0 = V128Helper.BlendVariable(edgeNormalsX0, V128.Xor(minusZero128, edgeNormalsX4), modeTriangle1);
        edgeNormalsY0 = V128Helper.BlendVariable(edgeNormalsY0, V128.Xor(minusZero128, edgeNormalsY4), modeTriangle1);

        const float maxOffset = -minEdgeOffset;
        Vector128<float> edgeFactor = V128.Create((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));

        // Flip edges if W < 0
        Vector128<float> edgeFlipMask0, edgeFlipMask1, edgeFlipMask2, edgeFlipMask3;
        if (T.PossiblyNearClipped)
        {
            edgeFlipMask0 = V128.Xor(edgeFactor, V128.Xor(wSign0, V128Helper.BlendVariable(wSign1, wSign2, modeTriangle1)));
            edgeFlipMask1 = V128.Xor(edgeFactor, V128.Xor(wSign1, wSign2));
            edgeFlipMask2 = V128.Xor(edgeFactor, V128.Xor(wSign2, V128Helper.BlendVariable(wSign3, wSign0, modeTriangle0)));
            edgeFlipMask3 = V128.Xor(edgeFactor, V128.Xor(wSign0, wSign3));
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

        Vector128<float> add128 = V128.Create(0.5f - minEdgeOffset * (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        Vector128<float> edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3;

        edgeOffsets0 = Fma.MultiplyAddNegated(x0, edgeNormalsX0, Fma.MultiplyAddNegated(y0, edgeNormalsY0, add128));
        edgeOffsets1 = Fma.MultiplyAddNegated(x1, edgeNormalsX1, Fma.MultiplyAddNegated(y1, edgeNormalsY1, add128));
        edgeOffsets2 = Fma.MultiplyAddNegated(x2, edgeNormalsX2, Fma.MultiplyAddNegated(y2, edgeNormalsY2, add128));
        edgeOffsets3 = Fma.MultiplyAddNegated(x3, edgeNormalsX3, Fma.MultiplyAddNegated(y3, edgeNormalsY3, add128));

        edgeOffsets1 = Fma.MultiplyAdd(V128.ConvertToSingle(minX), edgeNormalsX1, edgeOffsets1);
        edgeOffsets2 = Fma.MultiplyAdd(V128.ConvertToSingle(minX), edgeNormalsX2, edgeOffsets2);
        edgeOffsets3 = Fma.MultiplyAdd(V128.ConvertToSingle(minX), edgeNormalsX3, edgeOffsets3);

        edgeOffsets1 = Fma.MultiplyAdd(V128.ConvertToSingle(minY), edgeNormalsY1, edgeOffsets1);
        edgeOffsets2 = Fma.MultiplyAdd(V128.ConvertToSingle(minY), edgeNormalsY2, edgeOffsets2);
        edgeOffsets3 = Fma.MultiplyAdd(V128.ConvertToSingle(minY), edgeNormalsY3, edgeOffsets3);

        // Quantize slopes
        Vector128<int> slopeLookups0 = quantizeSlopeLookup(edgeNormalsX0, edgeNormalsY0);
        Vector128<int> slopeLookups1 = quantizeSlopeLookup(edgeNormalsX1, edgeNormalsY1);
        Vector128<int> slopeLookups2 = quantizeSlopeLookup(edgeNormalsX2, edgeNormalsY2);
        Vector128<int> slopeLookups3 = quantizeSlopeLookup(edgeNormalsX3, edgeNormalsY3);

        Vector128<int> firstBlockIdx = V128.Add(V128.Multiply(minY.AsInt16(), V128.Create((int)m_blocksX).AsInt16()).AsInt32(), minX);

        V128.StoreAligned(firstBlockIdx, (int*)firstBlocks);

        V128.StoreAligned(rangeX, (int*)rangesX);

        V128.StoreAligned(rangeY, (int*)rangesY);

        // Transpose into AoS
        transpose256(depthPlane0, depthPlane1, depthPlane2, Vector128<float>.Zero, depthPlane);

        transpose256(edgeNormalsX0, edgeNormalsX1, edgeNormalsX2, edgeNormalsX3, edgeNormalsX);

        transpose256(edgeNormalsY0, edgeNormalsY1, edgeNormalsY2, edgeNormalsY3, edgeNormalsY);

        transpose256(edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3, edgeOffsets);

        transpose256i(slopeLookups0, slopeLookups1, slopeLookups2, slopeLookups3, slopeLookups);

        return 0;
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

        Vector128<float> depthSamplePosFactor1 = V128.Create(
            depthSamplePos + 0.0f, depthSamplePos + 0.125f, depthSamplePos + 0.25f, depthSamplePos + 0.375f);

        Vector128<float> depthSamplePosFactor2A = V128.Create(depthSamplePos);
        Vector128<float> depthSamplePosFactor2B = V128.Create(depthSamplePos + 0.125f);

        // Loop over set bits
        while (validMask != 0)
        {
            uint primitiveIdx = (uint)BitOperations.TrailingZeroCount(validMask);

            // Clear lowest set bit in mask
            validMask &= validMask - 1;

            uint primitiveIdxTransposed = ((primitiveIdx << 1) & 7) | (primitiveIdx >> 2);

            // Extract and prepare per-primitive data
            ushort primitiveMaxZ = depthBounds[primitiveIdx];

            Vector128<float> depthDV = V128.LoadAligned((float*)(depthPlane + primitiveIdxTransposed));
            Vector128<float> depthDx = V128Helper.PermuteFrom1(depthDV);
            Vector128<float> depthDy = V128Helper.PermuteFrom2(depthDV);

            Vector128<float> lineDepthTerm = V128Helper.PermuteFrom0(depthDV);

            Vector128<float> lineDepthA =
              Fma.MultiplyAdd(depthDx, depthSamplePosFactor1,
                Fma.MultiplyAdd(depthDy, depthSamplePosFactor2A,
                  lineDepthTerm));

            Vector128<float> lineDepthB =
              Fma.MultiplyAdd(depthDx, depthSamplePosFactor1,
                Fma.MultiplyAdd(depthDy, depthSamplePosFactor2B,
                  lineDepthTerm));

            Vector128<int> slopeLookup = V128.LoadAligned((int*)(slopeLookups + primitiveIdxTransposed));
            Vector128<float> edgeNormalX = V128.LoadAligned((float*)(edgeNormalsX + primitiveIdxTransposed));
            Vector128<float> edgeNormalY = V128.LoadAligned((float*)(edgeNormalsY + primitiveIdxTransposed));
            Vector128<float> lineOffset = V128.LoadAligned((float*)(edgeOffsets + primitiveIdxTransposed));

            uint blocksX = m_blocksX;

            uint firstBlock = firstBlocks[primitiveIdx];
            uint blockRangeX = rangesX[primitiveIdx];
            uint blockRangeY = rangesY[primitiveIdx];

            ushort* pPrimitiveHiZ = pHiZBuffer + firstBlock;
            Vector128<int>* pPrimitiveOut = pDepthBuffer + 8 * firstBlock;

            uint primitiveMode = primModes[primitiveIdx];

            for (uint blockY = 0;
              blockY < blockRangeY;
              ++blockY,
              pPrimitiveHiZ += blocksX,
              pPrimitiveOut += 8 * blocksX,
              lineDepthA = V128.Add(lineDepthA, depthDy),
              lineDepthB = V128.Add(lineDepthB, depthDy),
              lineOffset = V128.Add(lineOffset, edgeNormalY))
            {
                ushort* pBlockRowHiZ = pPrimitiveHiZ;
                Vector128<int>* @out = pPrimitiveOut;

                Vector128<float> offset = lineOffset;
                Vector128<float> depthA = lineDepthA;
                Vector128<float> depthB = lineDepthB;

                bool anyBlockHit = false;
                for (uint blockX = 0;
                  blockX < blockRangeX;
                  ++blockX,
                  pBlockRowHiZ += 1,
                  @out += 8,
                  depthA = V128.Add(depthDx, depthA),
                  depthB = V128.Add(depthDx, depthB),
                  offset = V128.Add(edgeNormalX, offset))
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
                        Vector128<float> anyOffsetOutsideMask = V128.GreaterThanOrEqual(offset, V128.Create((float)(OFFSET_QUANTIZATION_FACTOR - 1)));
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

                        Vector128<int> offsetClamped = V128.Max(V128.ConvertToInt32(offset), Vector128<int>.Zero);

                        Vector128<int> lookup = V128.BitwiseOr(slopeLookup, offsetClamped);

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
                        Vector128<int> offsetClamped = V128.Min(V128.Max(V128.ConvertToInt32(offset), Vector128<int>.Zero), V128.Create(OFFSET_QUANTIZATION_FACTOR - 1));
                        Vector128<int> lookup = V128.BitwiseOr(slopeLookup, offsetClamped);

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
                    Vector128<float> depth0_A = depthA;
                    Vector128<float> depth1_A = Fma.MultiplyAdd(depthDx, V128.Create(0.5f), depth0_A);
                    Vector128<float> depth8_A = V128.Add(depthDy, depth0_A);
                    Vector128<float> depth9_A = V128.Add(depthDy, depth1_A);

                    Vector128<float> depth0_B = depthB;
                    Vector128<float> depth1_B = Fma.MultiplyAdd(depthDx, V128.Create(0.5f), depth0_B);
                    Vector128<float> depth8_B = V128.Add(depthDy, depth0_B);
                    Vector128<float> depth9_B = V128.Add(depthDy, depth1_B);

                    // Pack depth
                    Vector128<ushort> d0_A = packDepthPremultiplied(depth0_A, depth1_A);
                    Vector128<ushort> d4_A = packDepthPremultiplied(depth8_A, depth9_A);

                    Vector128<ushort> d0_B = packDepthPremultiplied(depth0_B, depth1_B);
                    Vector128<ushort> d4_B = packDepthPremultiplied(depth8_B, depth9_B);

                    // Interpolate remaining values in packed space
                    Vector128<ushort> d2_A = V128Helper.Average(d0_A, d4_A);
                    Vector128<ushort> d1_A = V128Helper.Average(d0_A, d2_A);
                    Vector128<ushort> d3_A = V128Helper.Average(d2_A, d4_A);

                    Vector128<ushort> d2_B = V128Helper.Average(d0_B, d4_B);
                    Vector128<ushort> d1_B = V128Helper.Average(d0_B, d2_B);
                    Vector128<ushort> d3_B = V128Helper.Average(d2_B, d4_B);

                    // Not all pixels covered - mask depth 
                    if (blockMask != 0xffff_ffff_ffff_ffff)
                    {
                        Vector128<ushort> A = V128.CreateScalar((long)blockMask).AsUInt16();
                        Vector128<ushort> B = V128.ShiftLeft(A.AsInt16(), 4).AsUInt16();

                        Vector128<byte> C_A = V128Helper.Blend(A, B, 0b11_11_00_00).AsByte();
                        Vector128<byte> C_B = V128Helper.Blend(A, B, 0b00_00_11_11).AsByte();

                        Vector128<short> rowMask_A = V128Helper.UnpackLow(C_A, C_A).AsInt16();
                        Vector128<short> rowMask_B = V128Helper.UnpackLow(C_B, C_B).AsInt16();

                        d0_A = V128Helper.BlendVariable(Vector128<byte>.Zero, d0_A.AsByte(), V128.ShiftLeft(rowMask_A, 3).AsByte()).AsUInt16();
                        d1_A = V128Helper.BlendVariable(Vector128<byte>.Zero, d1_A.AsByte(), V128.ShiftLeft(rowMask_A, 2).AsByte()).AsUInt16();
                        d2_A = V128Helper.BlendVariable(Vector128<byte>.Zero, d2_A.AsByte(), V128.Add(rowMask_A, rowMask_A).AsByte()).AsUInt16();
                        d3_A = V128Helper.BlendVariable(Vector128<byte>.Zero, d3_A.AsByte(), rowMask_A.AsByte()).AsUInt16();

                        d0_B = V128Helper.BlendVariable(Vector128<byte>.Zero, d0_B.AsByte(), V128.ShiftLeft(rowMask_B, 3).AsByte()).AsUInt16();
                        d1_B = V128Helper.BlendVariable(Vector128<byte>.Zero, d1_B.AsByte(), V128.ShiftLeft(rowMask_B, 2).AsByte()).AsUInt16();
                        d2_B = V128Helper.BlendVariable(Vector128<byte>.Zero, d2_B.AsByte(), V128.Add(rowMask_B, rowMask_B).AsByte()).AsUInt16();
                        d3_B = V128Helper.BlendVariable(Vector128<byte>.Zero, d3_B.AsByte(), rowMask_B.AsByte()).AsUInt16();
                    }

                    // Test fast clear flag
                    if (hiZ != 1)
                    {
                        // Merge depth values
                        d0_A = V128.Max(V128.LoadAligned((ushort*)(@out + 0)), d0_A);
                        d0_B = V128.Max(V128.LoadAligned((ushort*)(@out + 1)), d0_B);
                        d1_A = V128.Max(V128.LoadAligned((ushort*)(@out + 2)), d1_A);
                        d1_B = V128.Max(V128.LoadAligned((ushort*)(@out + 3)), d1_B);

                        d2_A = V128.Max(V128.LoadAligned((ushort*)(@out + 4)), d2_A);
                        d2_B = V128.Max(V128.LoadAligned((ushort*)(@out + 5)), d2_B);
                        d3_A = V128.Max(V128.LoadAligned((ushort*)(@out + 6)), d3_A);
                        d3_B = V128.Max(V128.LoadAligned((ushort*)(@out + 7)), d3_B);
                    }

                    // Store back new depth
                    V128.StoreAligned(d0_A, (ushort*)(@out + 0));
                    V128.StoreAligned(d0_B, (ushort*)(@out + 1));
                    V128.StoreAligned(d1_A, (ushort*)(@out + 2));
                    V128.StoreAligned(d1_B, (ushort*)(@out + 3));

                    V128.StoreAligned(d2_A, (ushort*)(@out + 4));
                    V128.StoreAligned(d2_B, (ushort*)(@out + 5));
                    V128.StoreAligned(d3_A, (ushort*)(@out + 6));
                    V128.StoreAligned(d3_B, (ushort*)(@out + 7));

                    // Update HiZ
                    Vector128<ushort> newMinZ_A = V128.Min(V128.Min(d0_A, d1_A), V128.Min(d2_A, d3_A));
                    Vector128<ushort> newMinZ_B = V128.Min(V128.Min(d0_B, d1_B), V128.Min(d2_B, d3_B));
                    ushort newMinZ16 = V128Helper.MinHorizontal(V128.Min(newMinZ_A, newMinZ_B));

                    *pBlockRowHiZ = newMinZ16;
                }
            }
        }
    }
}