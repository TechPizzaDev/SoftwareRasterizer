using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

using static ScalarMath;

public unsafe class ScalarRasterizer : Rasterizer
{
    private const int Alignment = 128 / 8; // sizeof(float)

    public ScalarRasterizer(RasterizationTable rasterizationTable, uint width, uint height) :
        base(rasterizationTable, width, height, Alignment)
    {
    }

    public static ScalarRasterizer Create(RasterizationTable rasterizationTable, uint width, uint height)
    {
        bool success = false;
        rasterizationTable.DangerousAddRef(ref success);
        if (success)
        {
            return new ScalarRasterizer(rasterizationTable, width, height);
        }
        throw new ObjectDisposedException(rasterizationTable.GetType().Name);
    }

    public override unsafe void setModelViewProjection(float* matrix)
    {
        Vector4 mat0 = Unsafe.ReadUnaligned<Vector4>(matrix + 0);
        Vector4 mat1 = Unsafe.ReadUnaligned<Vector4>(matrix + 4);
        Vector4 mat2 = Unsafe.ReadUnaligned<Vector4>(matrix + 8);
        Vector4 mat3 = Unsafe.ReadUnaligned<Vector4>(matrix + 12);

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Store rows
        Unsafe.Write(m_modelViewProjectionRaw + 0, mat0);
        Unsafe.Write(m_modelViewProjectionRaw + 4, mat1);
        Unsafe.Write(m_modelViewProjectionRaw + 8, mat2);
        Unsafe.Write(m_modelViewProjectionRaw + 12, mat3);

        // Bake viewport transform into matrix and 6shift by half a block
        mat0 = ((mat0 + mat3) * new Vector4(m_width * 0.5f - 4.0f));
        mat1 = ((mat1 + mat3) * new Vector4(m_height * 0.5f - 4.0f));

        // Map depth from [-1, 1] to [bias, 0]
        mat2 = ((mat3 - mat2) * new Vector4(0.5f * floatCompressionBias));

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Store prebaked cols
        Unsafe.Write(m_modelViewProjection + 0, mat0);
        Unsafe.Write(m_modelViewProjection + 4, mat1);
        Unsafe.Write(m_modelViewProjection + 8, mat2);
        Unsafe.Write(m_modelViewProjection + 12, mat3);
    }

    public override void clear()
    {
        // Mark blocks as cleared by setting Hi Z to 1 (one unit separated from far plane). 
        // This value is extremely unlikely to occur during normal rendering, so we don't
        // need to guard against a HiZ of 1 occuring naturally. This is different from a value of 0, 
        // which will occur every time a block is partially covered for the first time.
        Vector128<int> clearValue = Vector128.Create((short)1).AsInt32();
        uint count = m_hiZ_Size / 8;
        Vector4I* pHiZ = (Vector4I*)m_hiZ;
        for (uint offset = 0; offset < count; ++offset)
        {
            Unsafe.Write((int*)pHiZ, clearValue);
            pHiZ++;
        }
    }

    public override bool queryVisibility(Vector4 boundsMin, Vector4 boundsMax, out bool needsClipping)
    {
        // Frustum cull
        Vector4 extents = (boundsMax - boundsMin);
        Vector4 center = (boundsMax + boundsMin); // Bounding box center times 2 - but since W = 2, the plane equations work out correctly
        Vector4 minusZero = new(-0.0f);

        Vector4 row0 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjectionRaw + 0);
        Vector4 row1 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjectionRaw + 4);
        Vector4 row2 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjectionRaw + 8);
        Vector4 row3 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjectionRaw + 12);

        needsClipping = false;

        // Compute distance from each frustum plane
        Vector4 plane0 = (row3 + row0);
        Vector4 offset0 = (center + Xor(extents, And(plane0, minusZero)));
        float dist0 = Vector4.Dot(plane0, offset0);
        if (float.IsNegative(dist0))
        {
            return false;
        }

        Vector4 plane1 = (row3 - row0);
        Vector4 offset1 = (center + Xor(extents, And(plane1, minusZero)));
        float dist1 = Vector4.Dot(plane1, offset1);
        if (float.IsNegative(dist1))
        {
            return false;
        }

        Vector4 plane2 = (row3 + row1);
        Vector4 offset2 = (center + Xor(extents, And(plane2, minusZero)));
        float dist2 = Vector4.Dot(plane2, offset2);
        if (float.IsNegative(dist2))
        {
            return false;
        }

        Vector4 plane3 = (row3 - row1);
        Vector4 offset3 = (center + Xor(extents, And(plane3, minusZero)));
        float dist3 = Vector4.Dot(plane3, offset3);
        if (float.IsNegative(dist3))
        {
            return false;
        }

        Vector4 plane4 = (row3 + row2);
        Vector4 offset4 = (center + Xor(extents, And(plane4, minusZero)));
        float dist4 = Vector4.Dot(plane4, offset4);
        if (float.IsNegative(dist4))
        {
            return false;
        }

        Vector4 plane5 = (row3 - row2);
        Vector4 offset5 = (center + Xor(extents, And(plane5, minusZero)));
        float dist5 = Vector4.Dot(plane5, offset5);
        if (float.IsNegative(dist5))
        {
            return false;
        }

        // Load prebaked projection matrix
        Vector4 col0 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjection + 0);
        Vector4 col1 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjection + 4);
        Vector4 col2 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjection + 8);
        Vector4 col3 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjection + 12);

        // Transform edges
        Vector4 egde0 = (col0 * new Vector4(extents.X));
        Vector4 egde1 = (col1 * Permute(extents, 0b01_01_01_01));
        Vector4 egde2 = (col2 * Permute(extents, 0b10_10_10_10));

        Vector4 corners0;
        Vector4 corners1;
        Vector4 corners2;
        Vector4 corners3;
        Vector4 corners4;
        Vector4 corners5;
        Vector4 corners6;
        Vector4 corners7;

        // Transform first corner
        corners0 =
          ((col0 * new Vector4(boundsMin.X)) +
            ((col1 * Permute(boundsMin, 0b01_01_01_01)) +
              ((col2 * Permute(boundsMin, 0b10_10_10_10)) +
                col3)));

        // Transform remaining corners by adding edge vectors
        corners1 = (corners0 + egde0);
        corners2 = (corners0 + egde1);
        corners4 = (corners0 + egde2);

        corners3 = (corners1 + egde1);
        corners5 = (corners4 + egde0);
        corners6 = (corners2 + egde2);

        corners7 = (corners6 + egde0);

        // Transpose into SoA
        _MM_TRANSPOSE4_PS(ref corners0, ref corners1, ref corners2, ref corners3);
        _MM_TRANSPOSE4_PS(ref corners4, ref corners5, ref corners6, ref corners7);

        // Even if all bounding box corners have W > 0 here, we may end up with some vertices with W < 0 to due floating point differences; so test with some epsilon if any W < 0.
        Vector4 maxExtent = Vector4.Max(extents, Permute(extents, 0b01_00_11_10));
        maxExtent = Vector4.Max(maxExtent, Permute(maxExtent, 0b10_11_00_01));
        Vector4 nearPlaneEpsilon = (maxExtent * (0.001f));
        Vector4I closeToNearPlane = (CompareLessThan(corners3, nearPlaneEpsilon) | CompareLessThan(corners7, nearPlaneEpsilon));
        if (!Vector4I.TestZ(closeToNearPlane))
        {
            needsClipping = true;
            return true;
        }

        // Perspective division
        corners3 = Reciprocal(corners3);
        corners0 *= corners3;
        corners1 *= corners3;
        corners2 *= corners3;

        corners7 = Reciprocal(corners7);
        corners4 *= corners7;
        corners5 *= corners7;
        corners6 *= corners7;

        // Vertical mins and maxes
        Vector4 minsX = Vector4.Min(corners0, corners4);
        Vector4 maxsX = Vector4.Max(corners0, corners4);

        Vector4 minsY = Vector4.Min(corners1, corners5);
        Vector4 maxsY = Vector4.Max(corners1, corners5);

        // Horizontal reduction, step 1
        Vector4 minsXY = Vector4.Min(UnpackLow(minsX, minsY), UnpackHigh(minsX, minsY));
        Vector4 maxsXY = Vector4.Max(UnpackLow(maxsX, maxsY), UnpackHigh(maxsX, maxsY));

        // Clamp bounds
        minsXY = Vector4.Max(minsXY, Vector4.Zero);
        maxsXY = Vector4.Min(maxsXY, new Vector4(m_width - 1f, m_height - 1f, m_width - 1f, m_height - 1f));

        // Negate maxes so we can round in the same direction
        maxsXY = Xor(maxsXY, minusZero);

        // Horizontal reduction, step 2
        Vector4 boundsF = Vector4.Min(UnpackLow(minsXY, maxsXY), UnpackHigh(minsXY, maxsXY));

        // Round towards -infinity and convert to int
        Vector4I boundsI = Vector4I.ConvertWithTruncation(Floor(boundsF));

        // Store as scalars
        uint minX = (uint)boundsI.X;
        uint maxX = (uint)boundsI.Y;
        uint minY = (uint)boundsI.Z;
        uint maxY = (uint)boundsI.W;

        // Revert the sign change we did for the maxes
        maxX = (uint)-(int)maxX;
        maxY = (uint)-(int)maxY;

        // No intersection between quad and screen area
        if (minX >= maxX || minY >= maxY)
        {
            return false;
        }

        Vector128<ushort> depth = packDepthPremultiplied(corners2, corners6);

        ushort maxZ = (ushort)(0xFFFFu ^ Sse41.MinHorizontal(Sse2.Xor(depth, Vector128.Create((short)-1).AsUInt16())).ToScalar());

        if (!query2D(minX, maxX, minY, maxY, maxZ))
        {
            return false;
        }

        return true;
    }

    public override bool query2D(uint minX, uint maxX, uint minY, uint maxY, uint maxZ)
    {
        ushort* pHiZBuffer = m_hiZ;
        Vector4I* pDepthBuffer = (Vector4I*)m_depthBuffer;

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
            Vector4I* pBlockDepth = pDepthBuffer + 8 * (blockY * m_blocksX + blockMinX) + startY;

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

                Vector4I* pRowDepth = pBlockDepth;

                for (uint y = startY; y <= endY; ++y)
                {
                    Vector4I rowDepth = *pRowDepth++;

                    Vector128<int> notVisible = Sse2.CompareEqual(Sse41.Min(rowDepth.AsUInt16(), maxZV), maxZV).AsInt32();

                    uint visiblePixelMask = (uint)~Sse2.MoveMask(notVisible.AsByte());

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

                Vector4 vBias = new(bias);
                Vector4 vOne = new(1.0f);
                Vector4 vDiv = new(100 * 256 * 2 * 0.25f);
                Vector4 vSt = new(0.25f + 1000.0f);
                Vector4 vSf = new(1000.0f - 0.25f);

                Vector4I* source = (Vector4I*)&m_depthBuffer[8 * (blockY * m_blocksX + blockX)];
                for (uint y = 0; y < 8; ++y)
                {
                    Vector4I depthI = Unsafe.Read<Vector4I>((int*)source++);

                    Vector4I depthILo = ((Vector4I)Sse41.ConvertToVector128Int32(depthI.AsUInt16()) << 12);
                    Vector4I depthIHi = ((Vector4I)Sse41.ConvertToVector128Int32(Sse2.Shuffle(depthI, 0b11_10).AsUInt16()) << 12);

                    Vector4 depthLo = (depthILo.AsSingle() * vBias);
                    Vector4 depthHi = (depthIHi.AsSingle() * vBias);

                    Vector4 linDepthLo = (vDiv / (vSt - ((vOne - depthLo) * vSf)));
                    Vector4 linDepthHi = (vDiv / (vSt - ((vOne - depthHi) * vSf)));

                    Unsafe.Write((float*)(linDepthA + y * 2 + 0), linDepthLo);
                    Unsafe.Write((float*)(linDepthA + y * 2 + 1), linDepthHi);
                }

                Vector4 vRcp100 = new(1.0f / 100.0f);
                Vector128<ushort> vZeroMax = Sse2.UnpackLow(Vector128<byte>.Zero, Vector128<byte>.AllBitsSet).AsUInt16();
                Vector128<ushort> vMask = Vector128.Create((ushort)0xff);

                for (uint y = 0; y < 8; y += 2)
                {
                    Vector4 depth0 = Unsafe.Read<Vector4>((float*)(linDepthA + y * 2 + 0));
                    Vector4 depth1 = Unsafe.Read<Vector4>((float*)(linDepthA + y * 2 + 1));
                    Vector4 depth2 = Unsafe.Read<Vector4>((float*)(linDepthA + y * 2 + 2));
                    Vector4 depth3 = Unsafe.Read<Vector4>((float*)(linDepthA + y * 2 + 3));

                    Vector4I vR32_0 = Vector4I.ConvertWithTruncation((depth0 * vRcp100));
                    Vector4I vR32_1 = Vector4I.ConvertWithTruncation((depth1 * vRcp100));
                    Vector4I vR32_2 = Vector4I.ConvertWithTruncation((depth2 * vRcp100));
                    Vector4I vR32_3 = Vector4I.ConvertWithTruncation((depth3 * vRcp100));

                    Vector128<ushort> vR16_0 = Sse2.And(Sse41.PackUnsignedSaturate(vR32_0, vR32_1), vMask);
                    Vector128<ushort> vR16_1 = Sse2.And(Sse41.PackUnsignedSaturate(vR32_2, vR32_3), vMask);
                    Vector128<byte> vR8 = Sse2.PackUnsignedSaturate(vR16_0.AsInt16(), vR16_1.AsInt16());

                    Vector4I vG32_0 = Vector4I.ConvertWithTruncation(depth0);
                    Vector4I vG32_1 = Vector4I.ConvertWithTruncation(depth1);
                    Vector4I vG32_2 = Vector4I.ConvertWithTruncation(depth2);
                    Vector4I vG32_3 = Vector4I.ConvertWithTruncation(depth3);

                    Vector128<ushort> vG16_0 = Sse2.And(Sse41.PackUnsignedSaturate(vG32_0, vG32_1), vMask);
                    Vector128<ushort> vG16_1 = Sse2.And(Sse41.PackUnsignedSaturate(vG32_2, vG32_3), vMask);
                    Vector128<byte> vG8 = Sse2.PackUnsignedSaturate(vG16_0.AsInt16(), vG16_1.AsInt16());

                    Vector128<ushort> vRG_Lo = Sse2.UnpackLow(vR8, vG8).AsUInt16();
                    Vector128<ushort> vRG_Hi = Sse2.UnpackHigh(vR8, vG8).AsUInt16();

                    Vector128<int> result1 = Sse2.UnpackLow(vRG_Lo, vZeroMax).AsInt32();
                    Vector128<int> result2 = Sse2.UnpackHigh(vRG_Lo, vZeroMax).AsInt32();
                    Vector128<int> result3 = Sse2.UnpackLow(vRG_Hi, vZeroMax).AsInt32();
                    Vector128<int> result4 = Sse2.UnpackHigh(vRG_Hi, vZeroMax).AsInt32();

                    Unsafe.Write((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 0))) + 0, result1);
                    Unsafe.Write((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 0))) + 4, result2);
                    Unsafe.Write((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 1))) + 0, result3);
                    Unsafe.Write((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 1))) + 4, result4);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void transpose256(
        Vector4 A,
        Vector4 B,
        Vector4 C,
        Vector4 D,
        int outOffset,
        Vector4* @out)
    {
        Vector128<float> _Tmp0 = Sse.Shuffle(A.AsVector128(), B.AsVector128(), 0x44);
        Vector128<float> _Tmp2 = Sse.Shuffle(A.AsVector128(), B.AsVector128(), 0xEE);
        Vector128<float> _Tmp1 = Sse.Shuffle(C.AsVector128(), D.AsVector128(), 0x44);
        Vector128<float> _Tmp3 = Sse.Shuffle(C.AsVector128(), D.AsVector128(), 0xEE);

        Vector128<float> tA = Sse.Shuffle(_Tmp0, _Tmp1, 0x88);
        Vector128<float> tB = Sse.Shuffle(_Tmp0, _Tmp1, 0xDD);
        Vector128<float> tC = Sse.Shuffle(_Tmp2, _Tmp3, 0x88);
        Vector128<float> tD = Sse.Shuffle(_Tmp2, _Tmp3, 0xDD);

        Unsafe.Write((float*)(@out + outOffset + 0), tA);
        Unsafe.Write((float*)(@out + outOffset + 2), tB);
        Unsafe.Write((float*)(@out + outOffset + 4), tC);
        Unsafe.Write((float*)(@out + outOffset + 6), tD);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void transpose256i(
        Vector4I A,
        Vector4I B,
        Vector4I C,
        Vector4I D,
        int outOffset,
        Vector4I* @out)
    {
        Vector128<long> _Tmp0 = Sse2.UnpackLow(A, B).AsInt64();
        Vector128<long> _Tmp1 = Sse2.UnpackLow(C, D).AsInt64();
        Vector128<long> _Tmp2 = Sse2.UnpackHigh(A, B).AsInt64();
        Vector128<long> _Tmp3 = Sse2.UnpackHigh(C, D).AsInt64();

        Vector128<int> tA = Sse2.UnpackLow(_Tmp0, _Tmp1).AsInt32();
        Vector128<int> tB = Sse2.UnpackHigh(_Tmp0, _Tmp1).AsInt32();
        Vector128<int> tC = Sse2.UnpackLow(_Tmp2, _Tmp3).AsInt32();
        Vector128<int> tD = Sse2.UnpackHigh(_Tmp2, _Tmp3).AsInt32();

        Unsafe.Write((int*)(@out + outOffset + 0), tA);
        Unsafe.Write((int*)(@out + outOffset + 2), tB);
        Unsafe.Write((int*)(@out + outOffset + 4), tC);
        Unsafe.Write((int*)(@out + outOffset + 6), tD);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void normalizeEdge<T>(ref Vector4 nx, ref Vector4 ny, Vector4I edgeFlipMask)
        where T : IPossiblyNearClipped
    {
        Vector4 invLen = Reciprocal(NotZeroAnd(nx) + NotZeroAnd(ny));

        const float maxOffset = -minEdgeOffset;
        Vector4 mul = new((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        if (T.PossiblyNearClipped)
        {
            mul = (mul.AsInt32() ^ edgeFlipMask).AsSingle();
        }

        invLen = (mul * invLen);
        nx *= invLen;
        ny *= invLen;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector4I quantizeSlopeLookup(Vector4 nx, Vector4 ny)
    {
        Vector4I yNeg = CompareLessThanOrEqual(ny, Vector4.Zero);

        // Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
        const float maxOffset = -minEdgeOffset;
        const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f / ((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        const float add = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f + 0.5f;

        Vector4I quantizedSlope = Vector4I.ConvertWithTruncation(((nx * mul) + new Vector4(add)));
        return (((quantizedSlope << 1) - yNeg) << OFFSET_QUANTIZATION_BITS);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> packDepthPremultiplied(Vector4 depthA, Vector4 depthB)
    {
        Vector4I x1 = (depthA.AsInt32() >> 12);
        Vector4I x2 = (depthB.AsInt32() >> 12);
        return Sse41.PackUnsignedSaturate(x1, x2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> packDepthPremultiplied(Vector4 depth)
    {
        Vector4I x = (depth.AsInt32() >> 12);
        return Sse41.PackUnsignedSaturate(x, Vector4I.Zero);
    }

    public override void rasterize<T>(in Occluder occluder)
    {
        Vector4I* vertexData = (Vector4I*)occluder.m_vertexData;
        uint packetCount = occluder.m_packetCount;

        Vector4I maskY = new(2047 << 10);
        Vector4I maskZ = new(1023);

        // Note that unaligned loads do not have a latency penalty on CPUs with SSE4 support
        Vector4 mat0 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjection + 0);
        Vector4 mat1 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjection + 4);
        Vector4 mat2 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjection + 8);
        Vector4 mat3 = Unsafe.ReadUnaligned<Vector4>(m_modelViewProjection + 12);

        Vector4 boundsMin = occluder.m_refMin;
        Vector4 boundsExtents = (occluder.m_refMax - boundsMin);

        // Bake integer => bounding box transform into matrix
        mat3 =
          ((mat0 * new Vector4(boundsMin.X)) +
            ((mat1 * Permute(boundsMin, 0b01_01_01_01)) +
              ((mat2 * Permute(boundsMin, 0b10_10_10_10)) +
                mat3)));

        mat0 *= (new Vector4(boundsExtents.X) * (1.0f / (2047ul << 21)));
        mat1 *= (Permute(boundsExtents, 0b01_01_01_01) * (1.0f / (2047 << 10)));
        mat2 *= (Permute(boundsExtents, 0b10_10_10_10) * (1.0f / 1023));

        // Bias X coordinate back into positive range
        mat3 = ((mat0 * ((float)(1024ul << 21))) + mat3);

        // Skew projection to correct bleeding of Y and Z into X due to lack of masking
        mat1 -= mat0;
        mat2 -= mat0;

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Due to linear relationship between Z and W, it's cheaper to compute Z from W later in the pipeline than using the full projection matrix up front
        float c0, c1;
        {
            float Za = mat2.W;
            float Zb = Vector4.Dot(mat2, new Vector4(1 << 21, 1 << 10, 1, 1));
            
            float Wa = mat3.W;
            float Wb = Vector4.Dot(mat3, new Vector4(1 << 21, 1 << 10, 1, 1));
            
            c0 = ((Za - Zb) / (Wa - Wb));
            c1 = (-(((Za - Zb) / (Wa - Wb)) * Wa) + Za);
        }

        const int alignment = 256 / 8;
        const int stackBufferSize =
            alignment - 1 +
            sizeof(uint) * 8 * 4 + // uint[8] x 4
            sizeof(float) * 4 * 8 * 4 + // Vector4[8] x 4
            sizeof(int) * 4 * 8 * 1 + // Vector4I[8] x 1
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

        Vector4* depthPlane = (Vector4*)alignedBuffer;
        alignedBuffer += sizeof(Vector4) * 8;

        Vector4* edgeNormalsX = (Vector4*)alignedBuffer;
        alignedBuffer += sizeof(Vector4) * 8;

        Vector4* edgeNormalsY = (Vector4*)alignedBuffer;
        alignedBuffer += sizeof(Vector4) * 8;

        Vector4* edgeOffsets = (Vector4*)alignedBuffer;
        alignedBuffer += sizeof(Vector4) * 8;

        Vector4I* slopeLookups = (Vector4I*)alignedBuffer;
        alignedBuffer += sizeof(Vector4I) * 8;

        ushort* depthBounds = (ushort*)alignedBuffer;
        alignedBuffer += sizeof(ushort) * 8;

        for (uint packetIdx = 0; packetIdx < packetCount; packetIdx += 4)
        {
            uint validMask = 0;

            int RenderPacketPart(int partIndex)
            {
                // Load data - only needed once per frame, so use streaming load
                Vector4I* vertexDataPacketPtr = vertexData + packetIdx * 2 + partIndex;
                Vector4I I0 = Unsafe.Read<Vector4I>((int*)(vertexDataPacketPtr + 0));
                Vector4I I1 = Unsafe.Read<Vector4I>((int*)(vertexDataPacketPtr + 2));
                Vector4I I2 = Unsafe.Read<Vector4I>((int*)(vertexDataPacketPtr + 4));
                Vector4I I3 = Unsafe.Read<Vector4I>((int*)(vertexDataPacketPtr + 6));

                // Vertex transformation - first W, then X & Y after camera plane culling, then Z after backface culling
                Vector4 Xf0 = (I0.ToSingle());
                Vector4 Xf1 = (I1.ToSingle());
                Vector4 Xf2 = (I2.ToSingle());
                Vector4 Xf3 = (I3.ToSingle());

                Vector4 Yf0 = ((I0 & maskY).ToSingle());
                Vector4 Yf1 = ((I1 & maskY).ToSingle());
                Vector4 Yf2 = ((I2 & maskY).ToSingle());
                Vector4 Yf3 = ((I3 & maskY).ToSingle());

                Vector4 Zf0 = ((I0 & maskZ).ToSingle());
                Vector4 Zf1 = ((I1 & maskZ).ToSingle());
                Vector4 Zf2 = ((I2 & maskZ).ToSingle());
                Vector4 Zf3 = ((I3 & maskZ).ToSingle());

                Vector4 mat00 = new(mat0.X);
                Vector4 mat01 = new(mat0.Y);
                Vector4 mat02 = new(mat0.Z);
                Vector4 mat03 = new(mat0.W);

                Vector4 X0 = ((Xf0 * mat00) + ((Yf0 * mat01) + ((Zf0 * mat02) + mat03)));
                Vector4 X1 = ((Xf1 * mat00) + ((Yf1 * mat01) + ((Zf1 * mat02) + mat03)));
                Vector4 X2 = ((Xf2 * mat00) + ((Yf2 * mat01) + ((Zf2 * mat02) + mat03)));
                Vector4 X3 = ((Xf3 * mat00) + ((Yf3 * mat01) + ((Zf3 * mat02) + mat03)));

                Vector4 mat10 = new(mat1.X);
                Vector4 mat11 = new(mat1.Y);
                Vector4 mat12 = new(mat1.Z);
                Vector4 mat13 = new(mat1.W);

                Vector4 Y0 = ((Xf0 * mat10) + ((Yf0 * mat11) + ((Zf0 * mat12) + mat13)));
                Vector4 Y1 = ((Xf1 * mat10) + ((Yf1 * mat11) + ((Zf1 * mat12) + mat13)));
                Vector4 Y2 = ((Xf2 * mat10) + ((Yf2 * mat11) + ((Zf2 * mat12) + mat13)));
                Vector4 Y3 = ((Xf3 * mat10) + ((Yf3 * mat11) + ((Zf3 * mat12) + mat13)));

                Vector4 mat30 = new(mat3.X);
                Vector4 mat31 = new(mat3.Y);
                Vector4 mat32 = new(mat3.Z);
                Vector4 mat33 = new(mat3.W);

                Vector4 W0 = ((Xf0 * mat30) + ((Yf0 * mat31) + ((Zf0 * mat32) + mat33)));
                Vector4 W1 = ((Xf1 * mat30) + ((Yf1 * mat31) + ((Zf1 * mat32) + mat33)));
                Vector4 W2 = ((Xf2 * mat30) + ((Yf2 * mat31) + ((Zf2 * mat32) + mat33)));
                Vector4 W3 = ((Xf3 * mat30) + ((Yf3 * mat31) + ((Zf3 * mat32) + mat33)));

                Vector4 invW0, invW1, invW2, invW3;
                // Clamp W and invert
                if (T.PossiblyNearClipped)
                {
                    Vector4 lowerBound = new((float)-maxInvW);
                    Vector4 upperBound = new((float)+maxInvW);
                    invW0 = Vector4.Min(upperBound, Vector4.Max(lowerBound, Reciprocal(W0)));
                    invW1 = Vector4.Min(upperBound, Vector4.Max(lowerBound, Reciprocal(W1)));
                    invW2 = Vector4.Min(upperBound, Vector4.Max(lowerBound, Reciprocal(W2)));
                    invW3 = Vector4.Min(upperBound, Vector4.Max(lowerBound, Reciprocal(W3)));
                }
                else
                {
                    invW0 = Reciprocal(W0);
                    invW1 = Reciprocal(W1);
                    invW2 = Reciprocal(W2);
                    invW3 = Reciprocal(W3);
                }

                // Round to integer coordinates to improve culling of zero-area triangles
                const float roundFactor = 0.125f;
                Vector4 x0 = (RoundToNearestInteger((X0 * invW0)) * roundFactor);
                Vector4 x1 = (RoundToNearestInteger((X1 * invW1)) * roundFactor);
                Vector4 x2 = (RoundToNearestInteger((X2 * invW2)) * roundFactor);
                Vector4 x3 = (RoundToNearestInteger((X3 * invW3)) * roundFactor);

                Vector4 y0 = (RoundToNearestInteger((Y0 * invW0)) * roundFactor);
                Vector4 y1 = (RoundToNearestInteger((Y1 * invW1)) * roundFactor);
                Vector4 y2 = (RoundToNearestInteger((Y2 * invW2)) * roundFactor);
                Vector4 y3 = (RoundToNearestInteger((Y3 * invW3)) * roundFactor);

                // Compute unnormalized edge directions
                Vector4 edgeNormalsX0 = (y1 - y0);
                Vector4 edgeNormalsX1 = (y2 - y1);
                Vector4 edgeNormalsX2 = (y3 - y2);
                Vector4 edgeNormalsX3 = (y0 - y3);

                Vector4 edgeNormalsY0 = (x0 - x1);
                Vector4 edgeNormalsY1 = (x1 - x2);
                Vector4 edgeNormalsY2 = (x2 - x3);
                Vector4 edgeNormalsY3 = (x3 - x0);

                Vector4 area0 = ((edgeNormalsX0 * edgeNormalsY1) - (edgeNormalsX1 * edgeNormalsY0));
                Vector4 area1 = ((edgeNormalsX1 * edgeNormalsY2) - (edgeNormalsX2 * edgeNormalsY1));
                Vector4 area2 = ((edgeNormalsX2 * edgeNormalsY3) - (edgeNormalsX3 * edgeNormalsY2));
                Vector4 area3 = ((area0 + area2) - area1);

                Vector4 minusZero128 = new(-0.0f);
                Vector4I minusZero128i = minusZero128.AsInt32();

                Vector4I wSign0, wSign1, wSign2, wSign3;
                if (T.PossiblyNearClipped)
                {
                    wSign0 = (invW0.AsInt32() & minusZero128i);
                    wSign1 = (invW1.AsInt32() & minusZero128i);
                    wSign2 = (invW2.AsInt32() & minusZero128i);
                    wSign3 = (invW3.AsInt32() & minusZero128i);
                }
                else
                {
                    wSign0 = Vector4I.Zero;
                    wSign1 = Vector4I.Zero;
                    wSign2 = Vector4I.Zero;
                    wSign3 = Vector4I.Zero;
                }

                // Compute signs of areas. We treat 0 as negative as this allows treating primitives with zero area as backfacing.
                Vector4I areaSign0, areaSign1, areaSign2, areaSign3;
                if (T.PossiblyNearClipped)
                {
                    Vector4 sign0 = wSign0.AsSingle();
                    Vector4 sign1 = wSign1.AsSingle();
                    
                    // Flip areas for each vertex with W < 0. This needs to be done before comparison against 0 rather than afterwards to make sure zero-are triangles are handled correctly.
                    areaSign0 = CompareLessThanOrEqual(Xor(Xor(area0, sign0), (wSign1 ^ wSign2).AsSingle()), Vector4.Zero);
                    areaSign1 = (minusZero128i & CompareLessThanOrEqual(Xor(Xor(area1, sign1), (wSign2 ^ wSign3).AsSingle()), Vector4.Zero));
                    areaSign2 = (minusZero128i & CompareLessThanOrEqual(Xor(Xor(area2, sign0), (wSign2 ^ wSign3).AsSingle()), Vector4.Zero));
                    areaSign3 = (minusZero128i & CompareLessThanOrEqual(Xor(Xor(area3, sign1), (wSign0 ^ wSign3).AsSingle()), Vector4.Zero));
                }
                else
                {
                    areaSign0 = CompareLessThanOrEqual(area0, Vector4.Zero);
                    areaSign1 = (minusZero128i & CompareLessThanOrEqual(area1, Vector4.Zero));
                    areaSign2 = (minusZero128i & CompareLessThanOrEqual(area2, Vector4.Zero));
                    areaSign3 = (minusZero128i & CompareLessThanOrEqual(area3, Vector4.Zero));
                }

                Vector4I config = (
                    ((areaSign3 >>> 28) | (areaSign2 >>> 29)) |
                    ((areaSign1 >>> 30) | (areaSign0 >>> 31)));

                if (T.PossiblyNearClipped)
                {
                    config |= (
                        ((wSign3 >>> 24) | (wSign2 >>> 25)) |
                        ((wSign1 >>> 26) | (wSign0 >>> 27)));
                }

                Vector4I modes;
                fixed (PrimitiveMode* modeTablePtr = modeTable)
                {
                    modes = new Vector4I(
                        (int)modeTablePtr[config.X],
                        (int)modeTablePtr[config.Y],
                        (int)modeTablePtr[config.Z],
                        (int)modeTablePtr[config.W]);
                }

                if (Vector4I.TestZ(modes))
                {
                    return 1;
                }

                Vector4I primitiveValid = (modes > Vector4I.Zero);

                Unsafe.Write((int*)(primModes + 4 * partIndex), modes);

                Vector4 minFx, minFy, maxFx, maxFy;

                if (T.PossiblyNearClipped)
                {
                    // Clipless bounding box computation
                    Vector4 infP = new(+10000.0f);
                    Vector4 infN = new(-10000.0f);

                    // Find interval of points with W > 0
                    Vector4 minPx0 = BlendVariable(x0, infP, wSign0);
                    Vector4 minPx1 = BlendVariable(x1, infP, wSign1);
                    Vector4 minPx2 = BlendVariable(x2, infP, wSign2);
                    Vector4 minPx3 = BlendVariable(x3, infP, wSign3);

                    Vector4 minPx = Vector4.Min(
                      Vector4.Min(minPx0, minPx1),
                      Vector4.Min(minPx2, minPx3));

                    Vector4 minPy0 = BlendVariable(y0, infP, wSign0);
                    Vector4 minPy1 = BlendVariable(y1, infP, wSign1);
                    Vector4 minPy2 = BlendVariable(y2, infP, wSign2);
                    Vector4 minPy3 = BlendVariable(y3, infP, wSign3);

                    Vector4 minPy = Vector4.Min(
                      Vector4.Min(minPy0, minPy1),
                      Vector4.Min(minPy2, minPy3));

                    Vector4 maxPx0 = (minPx0.AsInt32() ^ wSign0).AsSingle();
                    Vector4 maxPx1 = (minPx1.AsInt32() ^ wSign1).AsSingle();
                    Vector4 maxPx2 = (minPx2.AsInt32() ^ wSign2).AsSingle();
                    Vector4 maxPx3 = (minPx3.AsInt32() ^ wSign3).AsSingle();

                    Vector4 maxPx = Vector4.Max(
                      Vector4.Max(maxPx0, maxPx1),
                      Vector4.Max(maxPx2, maxPx3));

                    Vector4 maxPy0 = (minPy0.AsInt32() ^ wSign0).AsSingle();
                    Vector4 maxPy1 = (minPy1.AsInt32() ^ wSign1).AsSingle();
                    Vector4 maxPy2 = (minPy2.AsInt32() ^ wSign2).AsSingle();
                    Vector4 maxPy3 = (minPy3.AsInt32() ^ wSign3).AsSingle();

                    Vector4 maxPy = Vector4.Max(
                      Vector4.Max(maxPy0, maxPy1),
                      Vector4.Max(maxPy2, maxPy3));

                    // Find interval of points with W < 0
                    Vector4 minNx0 = BlendVariable(infP, x0, wSign0);
                    Vector4 minNx1 = BlendVariable(infP, x1, wSign1);
                    Vector4 minNx2 = BlendVariable(infP, x2, wSign2);
                    Vector4 minNx3 = BlendVariable(infP, x3, wSign3);

                    Vector4 minNx = Vector4.Min(
                      Vector4.Min(minNx0, minNx1),
                      Vector4.Min(minNx2, minNx3));

                    Vector4 minNy0 = BlendVariable(infP, y0, wSign0);
                    Vector4 minNy1 = BlendVariable(infP, y1, wSign1);
                    Vector4 minNy2 = BlendVariable(infP, y2, wSign2);
                    Vector4 minNy3 = BlendVariable(infP, y3, wSign3);

                    Vector4 minNy = Vector4.Min(
                      Vector4.Min(minNy0, minNy1),
                      Vector4.Min(minNy2, minNy3));

                    Vector4 maxNx0 = BlendVariable(infN, x0, wSign0);
                    Vector4 maxNx1 = BlendVariable(infN, x1, wSign1);
                    Vector4 maxNx2 = BlendVariable(infN, x2, wSign2);
                    Vector4 maxNx3 = BlendVariable(infN, x3, wSign3);

                    Vector4 maxNx = Vector4.Max(
                      Vector4.Max(maxNx0, maxNx1),
                      Vector4.Max(maxNx2, maxNx3));

                    Vector4 maxNy0 = BlendVariable(infN, y0, wSign0);
                    Vector4 maxNy1 = BlendVariable(infN, y1, wSign1);
                    Vector4 maxNy2 = BlendVariable(infN, y2, wSign2);
                    Vector4 maxNy3 = BlendVariable(infN, y3, wSign3);

                    Vector4 maxNy = Vector4.Max(
                      Vector4.Max(maxNy0, maxNy1),
                      Vector4.Max(maxNy2, maxNy3));

                    // Include interval bounds resp. infinity depending on ordering of intervals
                    Vector4 incAx = BlendVariable(minPx, infN, CompareGreaterThan(maxNx, minPx));
                    Vector4 incAy = BlendVariable(minPy, infN, CompareGreaterThan(maxNy, minPy));

                    Vector4 incBx = BlendVariable(maxPx, infP, CompareGreaterThan(maxPx, minNx));
                    Vector4 incBy = BlendVariable(maxPy, infP, CompareGreaterThan(maxPy, minNy));

                    minFx = Vector4.Min(incAx, incBx);
                    minFy = Vector4.Min(incAy, incBy);

                    maxFx = Vector4.Max(incAx, incBx);
                    maxFy = Vector4.Max(incAy, incBy);
                }
                else
                {
                    // Standard bounding box inclusion
                    minFx = Vector4.Min(Vector4.Min(x0, x1), Vector4.Min(x2, x3));
                    minFy = Vector4.Min(Vector4.Min(y0, y1), Vector4.Min(y2, y3));

                    maxFx = Vector4.Max(Vector4.Max(x0, x1), Vector4.Max(x2, x3));
                    maxFy = Vector4.Max(Vector4.Max(y0, y1), Vector4.Max(y2, y3));
                }

                // Clamp and round
                Vector4I minX, minY, maxX, maxY;
                minX = Vector4I.Max(Vector4I.ConvertWithTruncation((minFx + new Vector4(4.9999f / 8.0f))), Vector4I.Zero);
                minY = Vector4I.Max(Vector4I.ConvertWithTruncation((minFy + new Vector4(4.9999f / 8.0f))), Vector4I.Zero);
                maxX = Vector4I.Min(Vector4I.ConvertWithTruncation((maxFx + new Vector4(11.0f / 8.0f))), new Vector4I((int)m_blocksX));
                maxY = Vector4I.Min(Vector4I.ConvertWithTruncation((maxFy + new Vector4(11.0f / 8.0f))), new Vector4I((int)m_blocksY));
                
                // Check overlap between bounding box and frustum
                Vector4I inFrustum = ((maxX > minX) & (maxY > minY));
                Vector4I overlappedPrimitiveValid = (inFrustum & primitiveValid);

                if (Vector4I.TestZ(overlappedPrimitiveValid))
                {
                    return 2;
                }

                validMask |= MoveMask(overlappedPrimitiveValid.AsSingle()) << (4 * partIndex);

                // Convert bounds from [min, max] to [min, range]
                Vector4I rangeX = (maxX - minX);
                Vector4I rangeY = (maxY - minY);

                // Compute Z from linear relation with 1/W
                Vector4 C0 = new(c0);
                Vector4 C1 = new(c1);
                Vector4 z0, z1, z2, z3;
                z0 = ((invW0 * C1) + C0);
                z1 = ((invW1 * C1) + C0);
                z2 = ((invW2 * C1) + C0);
                z3 = ((invW3 * C1) + C0);

                Vector4 maxZ = Vector4.Max(Vector4.Max(z0, z1), Vector4.Max(z2, z3));

                // If any W < 0, assume maxZ = 1 (effectively disabling Hi-Z)
                if (T.PossiblyNearClipped)
                {
                    maxZ = BlendVariable(maxZ, new Vector4(1.0f), ((wSign0 | wSign1) | (wSign2 | wSign3)));
                }

                Vector128<ushort> packedDepthBounds = packDepthPremultiplied(maxZ);

                Sse2.Store(depthBounds + 4 * partIndex, packedDepthBounds);

                // Compute screen space depth plane
                Vector4I greaterArea = CompareLessThan(NotZeroAnd(area0), NotZeroAnd(area2));

                // Force triangle area to be picked in the relevant mode.
                Vector4I modeTriangle0 = (modes == new Vector4I((int)PrimitiveMode.Triangle0));
                Vector4I modeTriangle1 = (modes == new Vector4I((int)PrimitiveMode.Triangle1));
                greaterArea = (~modeTriangle0) & (modeTriangle1 | greaterArea);

                Vector4 invArea;
                if (T.PossiblyNearClipped)
                {
                    // Do a precise divison to reduce error in depth plane. Note that the area computed here
                    // differs from the rasterized region if W < 0, so it can be very small for large covered screen regions.
                    invArea = (new Vector4(1.0f) / BlendVariable(area0, area2, greaterArea));
                }
                else
                {
                    invArea = Reciprocal(BlendVariable(area0, area2, greaterArea));
                }

                Vector4 z12 = (z1 - z2);
                Vector4 z20 = (z2 - z0);
                Vector4 z30 = (z3 - z0);

                Vector4 edgeNormalsX4 = (y0 - y2);
                Vector4 edgeNormalsY4 = (x2 - x0);

                Vector4 depthPlane0, depthPlane1, depthPlane2;
                depthPlane1 = (invArea * BlendVariable(((z20 * edgeNormalsX1) - (z12 * edgeNormalsX4)), (-(z20 * edgeNormalsX3) + (z30 * edgeNormalsX4)), greaterArea));
                depthPlane2 = (invArea * BlendVariable(((z20 * edgeNormalsY1) - (z12 * edgeNormalsY4)), (-(z20 * edgeNormalsY3) + (z30 * edgeNormalsY4)), greaterArea));

                x0 -= minX.ToSingle();
                y0 -= minY.ToSingle();

                depthPlane0 = (-(x0 * depthPlane1) + (-(y0 * depthPlane2) + z0));

                // If mode == Triangle0, replace edge 2 with edge 4; if mode == Triangle1, replace edge 0 with edge 4
                edgeNormalsX2 = BlendVariable(edgeNormalsX2, edgeNormalsX4, modeTriangle0);
                edgeNormalsY2 = BlendVariable(edgeNormalsY2, edgeNormalsY4, modeTriangle0);
                edgeNormalsX0 = BlendVariable(edgeNormalsX0, Xor(minusZero128, edgeNormalsX4), modeTriangle1);
                edgeNormalsY0 = BlendVariable(edgeNormalsY0, Xor(minusZero128, edgeNormalsY4), modeTriangle1);

                // Flip edges if W < 0
                Vector4I edgeFlipMask0, edgeFlipMask1, edgeFlipMask2, edgeFlipMask3;
                if (T.PossiblyNearClipped)
                {
                    edgeFlipMask0 = (wSign0 ^ BlendVariable(wSign1, wSign2, modeTriangle1));
                    edgeFlipMask1 = (wSign1 ^ wSign2);
                    edgeFlipMask2 = (wSign2 ^ BlendVariable(wSign3, wSign0, modeTriangle0));
                    edgeFlipMask3 = (wSign0 ^ wSign3);
                }
                else
                {
                    edgeFlipMask0 = Vector4I.Zero;
                    edgeFlipMask1 = Vector4I.Zero;
                    edgeFlipMask2 = Vector4I.Zero;
                    edgeFlipMask3 = Vector4I.Zero;
                }

                // Normalize edge equations for lookup
                normalizeEdge<T>(ref edgeNormalsX0, ref edgeNormalsY0, edgeFlipMask0);
                normalizeEdge<T>(ref edgeNormalsX1, ref edgeNormalsY1, edgeFlipMask1);
                normalizeEdge<T>(ref edgeNormalsX2, ref edgeNormalsY2, edgeFlipMask2);
                normalizeEdge<T>(ref edgeNormalsX3, ref edgeNormalsY3, edgeFlipMask3);

                const float maxOffset = -minEdgeOffset;
                Vector4 add128 = new(0.5f - minEdgeOffset * (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
                Vector4 edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3;

                edgeOffsets0 = (-(x0 * edgeNormalsX0) + (-(y0 * edgeNormalsY0) + add128));
                edgeOffsets1 = (-(x1 * edgeNormalsX1) + (-(y1 * edgeNormalsY1) + add128));
                edgeOffsets2 = (-(x2 * edgeNormalsX2) + (-(y2 * edgeNormalsY2) + add128));
                edgeOffsets3 = (-(x3 * edgeNormalsX3) + (-(y3 * edgeNormalsY3) + add128));

                edgeOffsets1 = ((minX.ToSingle() * edgeNormalsX1) + edgeOffsets1);
                edgeOffsets2 = ((minX.ToSingle() * edgeNormalsX2) + edgeOffsets2);
                edgeOffsets3 = ((minX.ToSingle() * edgeNormalsX3) + edgeOffsets3);

                edgeOffsets1 = ((minY.ToSingle() * edgeNormalsY1) + edgeOffsets1);
                edgeOffsets2 = ((minY.ToSingle() * edgeNormalsY2) + edgeOffsets2);
                edgeOffsets3 = ((minY.ToSingle() * edgeNormalsY3) + edgeOffsets3);

                // Quantize slopes
                Vector4I slopeLookups0 = quantizeSlopeLookup(edgeNormalsX0, edgeNormalsY0);
                Vector4I slopeLookups1 = quantizeSlopeLookup(edgeNormalsX1, edgeNormalsY1);
                Vector4I slopeLookups2 = quantizeSlopeLookup(edgeNormalsX2, edgeNormalsY2);
                Vector4I slopeLookups3 = quantizeSlopeLookup(edgeNormalsX3, edgeNormalsY3);

                Vector128<int> firstBlockIdx = Sse2.Add(Sse2.MultiplyLow(minY.AsInt16(), Vector128.Create((int)m_blocksX).AsInt16()).AsInt32(), minX);

                Unsafe.Write((int*)(firstBlocks + 4 * partIndex), firstBlockIdx);

                Unsafe.Write((int*)(rangesX + 4 * partIndex), rangeX);

                Unsafe.Write((int*)(rangesY + 4 * partIndex), rangeY);

                // Transpose into AoS
                transpose256(depthPlane0, depthPlane1, depthPlane2, Vector4.Zero, partIndex, depthPlane);

                transpose256(edgeNormalsX0, edgeNormalsX1, edgeNormalsX2, edgeNormalsX3, partIndex, edgeNormalsX);

                transpose256(edgeNormalsY0, edgeNormalsY1, edgeNormalsY2, edgeNormalsY3, partIndex, edgeNormalsY);

                transpose256(edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3, partIndex, edgeOffsets);

                transpose256i(slopeLookups0, slopeLookups1, slopeLookups2, slopeLookups3, partIndex, slopeLookups);

                return 0;
            }

            int p0 = RenderPacketPart(0);
            int p1 = RenderPacketPart(1);
            if (p0 == p1 && p0 != 0)
            {
                continue;
            }

            rasterizeLoop<T>(
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
                primModes);
        }
    }

    private void rasterizeLoop<T>(
        uint validMask,
        ushort* depthBounds,
        Vector4* depthPlane,
        Vector4I* slopeLookups,
        Vector4* edgeNormalsX,
        Vector4* edgeNormalsY,
        Vector4* edgeOffsets,
        uint* firstBlocks,
        uint* rangesX,
        uint* rangesY,
        uint* primModes)
        where T : IPossiblyNearClipped
    {
        // Fetch data pointers since we'll manually strength-reduce memory arithmetic
        ulong* pTable = (ulong*)m_precomputedRasterTables.DangerousGetHandle();
        ushort* pHiZBuffer = m_hiZ;
        Vector4I* pDepthBuffer = (Vector4I*)m_depthBuffer;

        const float depthSamplePos = -0.5f + 1.0f / 16.0f;

        Vector4 depthSamplePosFactor1 = new(
            depthSamplePos + 0.0f, depthSamplePos + 0.125f, depthSamplePos + 0.25f, depthSamplePos + 0.375f);

        const float depthSamplePosFactor2A = (depthSamplePos);
        const float depthSamplePosFactor2B = (depthSamplePos + 0.125f);

        // Loop over set bits
        while (validMask != 0)
        {
            uint primitiveIdx = (uint)BitOperations.TrailingZeroCount(validMask);

            // Clear lowest set bit in mask
            validMask &= validMask - 1;

            uint primitiveIdxTransposed = ((primitiveIdx << 1) & 7) | (primitiveIdx >> 2);

            // Extract and prepare per-primitive data
            ushort primitiveMaxZ = depthBounds[primitiveIdx];

            Vector4 depthDx = new(Permute(Unsafe.Read<Vector4>((float*)(depthPlane + primitiveIdxTransposed)), 0b01_01_01_01).X);
            Vector4 depthDy = new(Permute(Unsafe.Read<Vector4>((float*)(depthPlane + primitiveIdxTransposed)), 0b10_10_10_10).X);

            Vector4 lineDepthTerm = new(*(float*)(depthPlane + primitiveIdxTransposed));

            Vector4 lineDepthA =
              ((depthDx * depthSamplePosFactor1) +
                ((depthDy * depthSamplePosFactor2A) +
                  lineDepthTerm));

            Vector4 lineDepthB =
              ((depthDx * depthSamplePosFactor1) +
                ((depthDy * depthSamplePosFactor2B) +
                  lineDepthTerm));

            Vector4I slopeLookup = Unsafe.Read<Vector4I>((int*)(slopeLookups + primitiveIdxTransposed));
            Vector4 edgeNormalX = Unsafe.Read<Vector4>((float*)(edgeNormalsX + primitiveIdxTransposed));
            Vector4 edgeNormalY = Unsafe.Read<Vector4>((float*)(edgeNormalsY + primitiveIdxTransposed));
            Vector4 lineOffset = Unsafe.Read<Vector4>((float*)(edgeOffsets + primitiveIdxTransposed));

            uint blocksX = m_blocksX;

            uint firstBlock = firstBlocks[primitiveIdx];
            uint blockRangeX = rangesX[primitiveIdx];
            uint blockRangeY = rangesY[primitiveIdx];

            ushort* pPrimitiveHiZ = pHiZBuffer + firstBlock;
            Vector4I* pPrimitiveOut = pDepthBuffer + 8 * firstBlock;

            uint primitiveMode = primModes[primitiveIdx];

            for (uint blockY = 0;
              blockY < blockRangeY;
              ++blockY,
              pPrimitiveHiZ += blocksX,
              pPrimitiveOut += 8 * blocksX,
              lineDepthA += depthDy,
              lineDepthB += depthDy,
              lineOffset += edgeNormalY)
            {
                ushort* pBlockRowHiZ = pPrimitiveHiZ;
                Vector4I* @out = pPrimitiveOut;

                Vector4 offset = lineOffset;
                Vector4 depthA = lineDepthA;
                Vector4 depthB = lineDepthB;

                bool anyBlockHit = false;
                for (uint blockX = 0;
                  blockX < blockRangeX;
                  ++blockX,
                  pBlockRowHiZ += 1,
                  @out += 8,
                  depthA = (depthDx + depthA),
                  depthB = (depthDx + depthB),
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
                        Vector4I anyOffsetOutsideMask = CompareGreaterThanOrEqual(offset, new Vector4(OFFSET_QUANTIZATION_FACTOR - 1));
                        if (!Vector4I.TestZ(anyOffsetOutsideMask))
                        {
                            if (anyBlockHit)
                            {
                                // Convexity implies we won't hit another block in this row and can skip to the next line.
                                break;
                            }
                            continue;
                        }

                        anyBlockHit = true;

                        Vector4I offsetClamped = Vector4I.Max(Vector4I.ConvertWithTruncation(offset), Vector4I.Zero);

                        Vector4I lookup = (slopeLookup | offsetClamped);

                        // Generate block mask
                        ulong A = pTable[(uint)lookup.X];
                        ulong B = pTable[(uint)lookup.Y];
                        ulong C = pTable[(uint)lookup.Z];
                        ulong D = pTable[(uint)lookup.W];

                        blockMask = A & B & C & D;

                        // It is possible but very unlikely that blockMask == 0 if all A,B,C,D != 0 according to the conservative test above, so we skip the additional branch here.
                    }
                    else
                    {
                        Vector4I offsetClamped = Vector4I.Min(Vector4I.Max(Vector4I.ConvertWithTruncation(offset), Vector4I.Zero), new Vector4I(OFFSET_QUANTIZATION_FACTOR - 1));
                        Vector4I lookup = (slopeLookup | offsetClamped);

                        // Generate block mask
                        ulong A = pTable[(uint)lookup.X];
                        ulong B = pTable[(uint)lookup.Y];
                        ulong C = pTable[(uint)lookup.Z];
                        ulong D = pTable[(uint)lookup.W];

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
                                if (T.PossiblyNearClipped)
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
                    Vector4 depth0_A = depthA;
                    Vector4 depth1_A = ((depthDx * 0.5f) + depth0_A);
                    Vector4 depth8_A = (depthDy + depth0_A);
                    Vector4 depth9_A = (depthDy + depth1_A);

                    Vector4 depth0_B = depthB;
                    Vector4 depth1_B = ((depthDx * 0.5f) + depth0_B);
                    Vector4 depth8_B = (depthDy + depth0_B);
                    Vector4 depth9_B = (depthDy + depth1_B);

                    // Pack depth
                    Vector128<ushort> d0_A = packDepthPremultiplied(depth0_A, depth1_A);
                    Vector128<ushort> d4_A = packDepthPremultiplied(depth8_A, depth9_A);

                    Vector128<ushort> d0_B = packDepthPremultiplied(depth0_B, depth1_B);
                    Vector128<ushort> d4_B = packDepthPremultiplied(depth8_B, depth9_B);

                    // Interpolate remaining values in packed space
                    Vector128<ushort> d2_A = Sse2.Average(d0_A, d4_A);
                    Vector128<ushort> d1_A = Sse2.Average(d0_A, d2_A);
                    Vector128<ushort> d3_A = Sse2.Average(d2_A, d4_A);

                    Vector128<ushort> d2_B = Sse2.Average(d0_B, d4_B);
                    Vector128<ushort> d1_B = Sse2.Average(d0_B, d2_B);
                    Vector128<ushort> d3_B = Sse2.Average(d2_B, d4_B);

                    // Not all pixels covered - mask depth 
                    if (blockMask != 0xffff_ffff_ffff_ffff)
                    {
                        Vector128<ushort> A = Vector128.CreateScalar((long)blockMask).AsUInt16();
                        Vector128<ushort> B = Sse2.ShiftLeftLogical(A.AsInt16(), 4).AsUInt16();

                        Vector128<byte> C_A = Sse41.Blend(A, B, 0b11_11_00_00).AsByte();
                        Vector128<byte> C_B = Sse41.Blend(A, B, 0b00_00_11_11).AsByte();

                        Vector128<short> rowMask_A = Sse2.UnpackLow(C_A, C_A).AsInt16();
                        Vector128<short> rowMask_B = Sse2.UnpackLow(C_B, C_B).AsInt16();

                        d0_A = Sse41.BlendVariable(Vector128<byte>.Zero, d0_A.AsByte(), Sse2.ShiftLeftLogical(rowMask_A, 3).AsByte()).AsUInt16();
                        d1_A = Sse41.BlendVariable(Vector128<byte>.Zero, d1_A.AsByte(), Sse2.ShiftLeftLogical(rowMask_A, 2).AsByte()).AsUInt16();
                        d2_A = Sse41.BlendVariable(Vector128<byte>.Zero, d2_A.AsByte(), Sse2.Add(rowMask_A, rowMask_A).AsByte()).AsUInt16();
                        d3_A = Sse41.BlendVariable(Vector128<byte>.Zero, d3_A.AsByte(), rowMask_A.AsByte()).AsUInt16();

                        d0_B = Sse41.BlendVariable(Vector128<byte>.Zero, d0_B.AsByte(), Sse2.ShiftLeftLogical(rowMask_B, 3).AsByte()).AsUInt16();
                        d1_B = Sse41.BlendVariable(Vector128<byte>.Zero, d1_B.AsByte(), Sse2.ShiftLeftLogical(rowMask_B, 2).AsByte()).AsUInt16();
                        d2_B = Sse41.BlendVariable(Vector128<byte>.Zero, d2_B.AsByte(), Sse2.Add(rowMask_B, rowMask_B).AsByte()).AsUInt16();
                        d3_B = Sse41.BlendVariable(Vector128<byte>.Zero, d3_B.AsByte(), rowMask_B.AsByte()).AsUInt16();
                    }

                    // Test fast clear flag
                    if (hiZ != 1)
                    {
                        // Merge depth values
                        d0_A = Sse41.Max(Sse2.LoadAlignedVector128((ushort*)(@out + 0)), d0_A);
                        d0_B = Sse41.Max(Sse2.LoadAlignedVector128((ushort*)(@out + 1)), d0_B);
                        d1_A = Sse41.Max(Sse2.LoadAlignedVector128((ushort*)(@out + 2)), d1_A);
                        d1_B = Sse41.Max(Sse2.LoadAlignedVector128((ushort*)(@out + 3)), d1_B);

                        d2_A = Sse41.Max(Sse2.LoadAlignedVector128((ushort*)(@out + 4)), d2_A);
                        d2_B = Sse41.Max(Sse2.LoadAlignedVector128((ushort*)(@out + 5)), d2_B);
                        d3_A = Sse41.Max(Sse2.LoadAlignedVector128((ushort*)(@out + 6)), d3_A);
                        d3_B = Sse41.Max(Sse2.LoadAlignedVector128((ushort*)(@out + 7)), d3_B);
                    }

                    // Store back new depth
                    Sse2.StoreAligned((ushort*)(@out + 0), d0_A);
                    Sse2.StoreAligned((ushort*)(@out + 1), d0_B);
                    Sse2.StoreAligned((ushort*)(@out + 2), d1_A);
                    Sse2.StoreAligned((ushort*)(@out + 3), d1_B);

                    Sse2.StoreAligned((ushort*)(@out + 4), d2_A);
                    Sse2.StoreAligned((ushort*)(@out + 5), d2_B);
                    Sse2.StoreAligned((ushort*)(@out + 6), d3_A);
                    Sse2.StoreAligned((ushort*)(@out + 7), d3_B);

                    // Update HiZ
                    Vector128<ushort> newMinZ_A = Sse41.Min(Sse41.Min(d0_A, d1_A), Sse41.Min(d2_A, d3_A));
                    Vector128<ushort> newMinZ_B = Sse41.Min(Sse41.Min(d0_B, d1_B), Sse41.Min(d2_B, d3_B));
                    Vector128<int> newMinZ16 = Sse41.MinHorizontal(Sse41.Min(newMinZ_A, newMinZ_B)).AsInt32();

                    *pBlockRowHiZ = (ushort)(uint)Sse2.ConvertToInt32(newMinZ16);
                }
            }
        }
    }
}