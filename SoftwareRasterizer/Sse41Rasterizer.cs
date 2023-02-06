using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

using static VectorMath;

public unsafe class Sse41Rasterizer : Rasterizer
{
    private const int Alignment = 256 / 8; // sizeof(Vector256<>)

    public Sse41Rasterizer(RasterizationTable rasterizationTable, uint width, uint height) :
        base(rasterizationTable, width, height, Alignment)
    {
    }

    public static Sse41Rasterizer Create(RasterizationTable rasterizationTable, uint width, uint height)
    {
        bool success = false;
        rasterizationTable.DangerousAddRef(ref success);
        if (success)
        {
            return new Sse41Rasterizer(rasterizationTable, width, height);
        }
        throw new ObjectDisposedException(rasterizationTable.GetType().Name);
    }

    public override unsafe void setModelViewProjection(float* matrix)
    {
        Vector128<float> mat0 = Sse.LoadVector128(matrix + 0);
        Vector128<float> mat1 = Sse.LoadVector128(matrix + 4);
        Vector128<float> mat2 = Sse.LoadVector128(matrix + 8);
        Vector128<float> mat3 = Sse.LoadVector128(matrix + 12);

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Store rows
        Sse.StoreAligned(m_modelViewProjectionRaw + 0, mat0);
        Sse.StoreAligned(m_modelViewProjectionRaw + 4, mat1);
        Sse.StoreAligned(m_modelViewProjectionRaw + 8, mat2);
        Sse.StoreAligned(m_modelViewProjectionRaw + 12, mat3);

        // Bake viewport transform into matrix and 6shift by half a block
        mat0 = Sse.Multiply(Sse.Add(mat0, mat3), Vector128.Create(m_width * 0.5f - 4.0f));
        mat1 = Sse.Multiply(Sse.Add(mat1, mat3), Vector128.Create(m_height * 0.5f - 4.0f));

        // Map depth from [-1, 1] to [bias, 0]
        mat2 = Sse.Multiply(Sse.Subtract(mat3, mat2), Vector128.Create(0.5f * floatCompressionBias));

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

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

    public override bool queryVisibility(Vector128<float> boundsMin, Vector128<float> boundsMax, out bool needsClipping)
    {
        // Frustum cull
        Vector128<float> extents = Sse.Subtract(boundsMax, boundsMin);
        Vector128<float> center = Sse.Add(boundsMax, boundsMin); // Bounding box center times 2 - but since W = 2, the plane equations work out correctly
        Vector128<float> minusZero = Vector128.Create(-0.0f);

        Vector128<float> row0 = Sse.LoadVector128(m_modelViewProjectionRaw + 0);
        Vector128<float> row1 = Sse.LoadVector128(m_modelViewProjectionRaw + 4);
        Vector128<float> row2 = Sse.LoadVector128(m_modelViewProjectionRaw + 8);
        Vector128<float> row3 = Sse.LoadVector128(m_modelViewProjectionRaw + 12);

        // Compute distance from each frustum plane
        Vector128<float> plane0 = Sse.Add(row3, row0);
        Vector128<float> offset0 = Sse.Add(center, Sse.Xor(extents, Sse.And(plane0, minusZero)));
        Vector128<float> dist0 = Sse41.DotProduct(plane0, offset0, 0xff);

        Vector128<float> plane1 = Sse.Subtract(row3, row0);
        Vector128<float> offset1 = Sse.Add(center, Sse.Xor(extents, Sse.And(plane1, minusZero)));
        Vector128<float> dist1 = Sse41.DotProduct(plane1, offset1, 0xff);

        Vector128<float> plane2 = Sse.Add(row3, row1);
        Vector128<float> offset2 = Sse.Add(center, Sse.Xor(extents, Sse.And(plane2, minusZero)));
        Vector128<float> dist2 = Sse41.DotProduct(plane2, offset2, 0xff);

        Vector128<float> plane3 = Sse.Subtract(row3, row1);
        Vector128<float> offset3 = Sse.Add(center, Sse.Xor(extents, Sse.And(plane3, minusZero)));
        Vector128<float> dist3 = Sse41.DotProduct(plane3, offset3, 0xff);

        Vector128<float> plane4 = Sse.Add(row3, row2);
        Vector128<float> offset4 = Sse.Add(center, Sse.Xor(extents, Sse.And(plane4, minusZero)));
        Vector128<float> dist4 = Sse41.DotProduct(plane4, offset4, 0xff);

        Vector128<float> plane5 = Sse.Subtract(row3, row2);
        Vector128<float> offset5 = Sse.Add(center, Sse.Xor(extents, Sse.And(plane5, minusZero)));
        Vector128<float> dist5 = Sse41.DotProduct(plane5, offset5, 0xff);

        // Combine plane distance signs
        Vector128<float> combined = Sse.Or(Sse.Or(Sse.Or(dist0, dist1), Sse.Or(dist2, dist3)), Sse.Or(dist4, dist5));

        // Can't use Avx.TestZ or _mm_comile_ss here because the OR's above created garbage in the non-sign bits
        if (Sse.MoveMask(combined) != 0)
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
        Vector128<float> egde0 = Sse.Multiply(col0, BroadcastScalarToVector128(extents));
        Vector128<float> egde1 = Sse.Multiply(col1, Permute(extents, 0b01_01_01_01));
        Vector128<float> egde2 = Sse.Multiply(col2, Permute(extents, 0b10_10_10_10));

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
          Fma.MultiplyAdd(col0, BroadcastScalarToVector128(boundsMin),
            Fma.MultiplyAdd(col1, Permute(boundsMin, 0b01_01_01_01),
              Fma.MultiplyAdd(col2, Permute(boundsMin, 0b10_10_10_10),
                col3)));

        // Transform remaining corners by adding edge vectors
        corners1 = Sse.Add(corners0, egde0);
        corners2 = Sse.Add(corners0, egde1);
        corners4 = Sse.Add(corners0, egde2);

        corners3 = Sse.Add(corners1, egde1);
        corners5 = Sse.Add(corners4, egde0);
        corners6 = Sse.Add(corners2, egde2);

        corners7 = Sse.Add(corners6, egde0);

        // Transpose into SoA
        _MM_TRANSPOSE4_PS(ref corners0, ref corners1, ref corners2, ref corners3);
        _MM_TRANSPOSE4_PS(ref corners4, ref corners5, ref corners6, ref corners7);

        // Even if all bounding box corners have W > 0 here, we may end up with some vertices with W < 0 to due floating point differences; so test with some epsilon if any W < 0.
        Vector128<float> maxExtent = Sse.Max(extents, Permute(extents, 0b01_00_11_10));
        maxExtent = Sse.Max(maxExtent, Permute(maxExtent, 0b10_11_00_01));
        Vector128<float> nearPlaneEpsilon = Sse.Multiply(maxExtent, Vector128.Create(0.001f));
        Vector128<float> closeToNearPlane = Sse.Or(Sse.CompareLessThan(corners3, nearPlaneEpsilon), Sse.CompareLessThan(corners7, nearPlaneEpsilon));
        if (!TestZ(closeToNearPlane, closeToNearPlane))
        {
            needsClipping = true;
            return true;
        }

        needsClipping = false;

        // Perspective division
        corners3 = Sse.Reciprocal(corners3);
        corners0 = Sse.Multiply(corners0, corners3);
        corners1 = Sse.Multiply(corners1, corners3);
        corners2 = Sse.Multiply(corners2, corners3);

        corners7 = Sse.Reciprocal(corners7);
        corners4 = Sse.Multiply(corners4, corners7);
        corners5 = Sse.Multiply(corners5, corners7);
        corners6 = Sse.Multiply(corners6, corners7);

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
        maxsXY = Sse.Xor(maxsXY, minusZero);

        // Horizontal reduction, step 2
        Vector128<float> boundsF = Sse.Min(Sse.UnpackLow(minsXY, maxsXY), Sse.UnpackHigh(minsXY, maxsXY));

        // Round towards -infinity and convert to int
        Vector128<int> boundsI = Sse2.ConvertToVector128Int32WithTruncation(Sse41.RoundToNegativeInfinity(boundsF));

        // Store as scalars
        int* bounds = stackalloc int[4];
        Sse2.Store(bounds, boundsI);

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

        ushort maxZ = (ushort)(0xFFFFu ^ Sse2.Extract(Sse41.MinHorizontal(Sse2.Xor(depth, Vector128.Create((short)-1).AsUInt16())), 0));

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
                    Vector128<int> depthI = Sse2.LoadAlignedVector128((int*)source++);

                    Vector256<int> depthI256 = Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(depthI.AsUInt16()), 12);
                    Vector256<float> depth = Avx.Multiply(depthI256.AsSingle(), vBias);

                    Vector256<float> linDepth = Avx.Divide(vDiv, Avx.Subtract(vSt, Avx.Multiply(Avx.Subtract(vOne, depth), vSf)));
                    Avx.StoreAligned((float*)(linDepthA + y), linDepth);
                }

                Vector256<float> vRcp100 = Vector256.Create(1.0f / 100.0f);
                Vector256<ushort> vZeroMax = Avx2.UnpackLow(Vector256<byte>.Zero, Vector256<byte>.AllBitsSet).AsUInt16();
                Vector256<ushort> vMask = Vector256.Create((ushort)0xff);

                for (uint y = 0; y < 8; y += 4)
                {
                    Vector256<float> depth0 = Avx.LoadAlignedVector256((float*)(linDepthA + y + 0));
                    Vector256<float> depth1 = Avx.LoadAlignedVector256((float*)(linDepthA + y + 1));
                    Vector256<float> depth2 = Avx.LoadAlignedVector256((float*)(linDepthA + y + 2));
                    Vector256<float> depth3 = Avx.LoadAlignedVector256((float*)(linDepthA + y + 3));

                    Vector256<int> vR32_0 = Avx.ConvertToVector256Int32WithTruncation(Avx.Multiply(depth0, vRcp100));
                    Vector256<int> vR32_1 = Avx.ConvertToVector256Int32WithTruncation(Avx.Multiply(depth1, vRcp100));
                    Vector256<int> vR32_2 = Avx.ConvertToVector256Int32WithTruncation(Avx.Multiply(depth2, vRcp100));
                    Vector256<int> vR32_3 = Avx.ConvertToVector256Int32WithTruncation(Avx.Multiply(depth3, vRcp100));

                    Vector256<ushort> vR16_0 = Avx2.And(Avx2.PackUnsignedSaturate(vR32_0, vR32_1), vMask);
                    Vector256<ushort> vR16_1 = Avx2.And(Avx2.PackUnsignedSaturate(vR32_2, vR32_3), vMask);
                    Vector256<byte> vR8 = Avx2.PackUnsignedSaturate(vR16_0.AsInt16(), vR16_1.AsInt16());

                    Vector256<int> vG32_0 = Avx.ConvertToVector256Int32WithTruncation(depth0);
                    Vector256<int> vG32_1 = Avx.ConvertToVector256Int32WithTruncation(depth1);
                    Vector256<int> vG32_2 = Avx.ConvertToVector256Int32WithTruncation(depth2);
                    Vector256<int> vG32_3 = Avx.ConvertToVector256Int32WithTruncation(depth3);

                    Vector256<ushort> vG16_0 = Avx2.And(Avx2.PackUnsignedSaturate(vG32_0, vG32_1), vMask);
                    Vector256<ushort> vG16_1 = Avx2.And(Avx2.PackUnsignedSaturate(vG32_2, vG32_3), vMask);
                    Vector256<byte> vG8 = Avx2.PackUnsignedSaturate(vG16_0.AsInt16(), vG16_1.AsInt16());

                    Vector256<byte> vRG_Lo = Avx2.UnpackLow(vR8, vG8);
                    Vector256<byte> vRG_Hi = Avx2.UnpackHigh(vR8, vG8);

                    Vector256<uint> result1 = Avx2.UnpackLow(vRG_Lo.AsUInt16(), vZeroMax).AsUInt32();
                    Vector256<uint> result2 = Avx2.UnpackHigh(vRG_Lo.AsUInt16(), vZeroMax).AsUInt32();
                    Vector256<uint> result3 = Avx2.UnpackLow(vRG_Hi.AsUInt16(), vZeroMax).AsUInt32();
                    Vector256<uint> result4 = Avx2.UnpackHigh(vRG_Hi.AsUInt16(), vZeroMax).AsUInt32();

                    Avx.StoreAligned((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 0))), result1);
                    Avx.StoreAligned((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 1))), result2);
                    Avx.StoreAligned((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 2))), result3);
                    Avx.StoreAligned((uint*)(target + 4 * (8 * blockX + m_width * (8 * blockY + y + 3))), result4);
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
    private static void transpose256(
        Vector128<float> A1,
        Vector128<float> A2,
        Vector128<float> B1,
        Vector128<float> B2,
        Vector128<float> C1,
        Vector128<float> C2,
        Vector128<float> D1,
        Vector128<float> D2,
        Vector128<float>* @out)
    {
        Vector256<float> A = A1.ToVector256Unsafe().WithUpper(A2);
        Vector256<float> B = B1.ToVector256Unsafe().WithUpper(B2);
        Vector256<float> C = C1.ToVector256Unsafe().WithUpper(C2);
        Vector256<float> D = D1.ToVector256Unsafe().WithUpper(D2);

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

        /*
        Vector128<float> _Tmp0 = Avx.Shuffle(A, B, 0x44);
        Vector128<float> _Tmp2 = Avx.Shuffle(A, B, 0xEE);
        Vector128<float> _Tmp1 = Avx.Shuffle(C, D, 0x44);
        Vector128<float> _Tmp3 = Avx.Shuffle(C, D, 0xEE);
        Vector128<float> _Tmp0b = Avx.Shuffle(A2, B2, 0x44);
        Vector128<float> _Tmp2b = Avx.Shuffle(A2, B2, 0xEE);
        Vector128<float> _Tmp1b = Avx.Shuffle(C2, D2, 0x44);
        Vector128<float> _Tmp3b = Avx.Shuffle(C2, D2, 0xEE);

        Vector128<float> tA = Avx.Shuffle(_Tmp0, _Tmp1, 0x88);
        Vector128<float> tB = Avx.Shuffle(_Tmp0, _Tmp1, 0xDD);
        Vector128<float> tC = Avx.Shuffle(_Tmp2, _Tmp3, 0x88);
        Vector128<float> tD = Avx.Shuffle(_Tmp2, _Tmp3, 0xDD);
        Vector128<float> tA2 = Avx.Shuffle(_Tmp0b, _Tmp1b, 0x88);
        Vector128<float> tB2 = Avx.Shuffle(_Tmp0b, _Tmp1b, 0xDD);
        Vector128<float> tC2 = Avx.Shuffle(_Tmp2b, _Tmp3b, 0x88);
        Vector128<float> tD2 = Avx.Shuffle(_Tmp2b, _Tmp3b, 0xDD);

        Avx.StoreAligned((float*)(@out + 0), tA);
        Avx.StoreAligned((float*)(@out + 1), tA2);
        Avx.StoreAligned((float*)(@out + 2), tB);
        Avx.StoreAligned((float*)(@out + 3), tB2);
        Avx.StoreAligned((float*)(@out + 4), tC);
        Avx.StoreAligned((float*)(@out + 5), tC2);
        Avx.StoreAligned((float*)(@out + 6), tD);
        Avx.StoreAligned((float*)(@out + 7), tD2);
        */
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void transpose256i(
        Vector128<int> A1,
        Vector128<int> A2,
        Vector128<int> B1,
        Vector128<int> B2,
        Vector128<int> C1,
        Vector128<int> C2,
        Vector128<int> D1,
        Vector128<int> D2,
        Vector128<int>* @out)
    {
        Vector256<int> A = A1.ToVector256Unsafe().WithUpper(A2);
        Vector256<int> B = B1.ToVector256Unsafe().WithUpper(B2);
        Vector256<int> C = C1.ToVector256Unsafe().WithUpper(C2);
        Vector256<int> D = D1.ToVector256Unsafe().WithUpper(D2);

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

        /*
        Vector128<long> _Tmp0 = Avx2.UnpackLow(A, B).AsInt64();
        Vector128<long> _Tmp1 = Avx2.UnpackLow(C, D).AsInt64();
        Vector128<long> _Tmp2 = Avx2.UnpackHigh(A, B).AsInt64();
        Vector128<long> _Tmp3 = Avx2.UnpackHigh(C, D).AsInt64();
        Vector128<long> _Tmp0b = Avx2.UnpackLow(A2, B2).AsInt64();
        Vector128<long> _Tmp1b = Avx2.UnpackLow(C2, D2).AsInt64();
        Vector128<long> _Tmp2b = Avx2.UnpackHigh(A2, B2).AsInt64();
        Vector128<long> _Tmp3b = Avx2.UnpackHigh(C2, D2).AsInt64();

        Vector128<int> tA = Avx2.UnpackLow(_Tmp0, _Tmp1).AsInt32();
        Vector128<int> tB = Avx2.UnpackHigh(_Tmp0, _Tmp1).AsInt32();
        Vector128<int> tC = Avx2.UnpackLow(_Tmp2, _Tmp3).AsInt32();
        Vector128<int> tD = Avx2.UnpackHigh(_Tmp2, _Tmp3).AsInt32();
        Vector128<int> tA2 = Avx2.UnpackLow(_Tmp0b, _Tmp1b).AsInt32();
        Vector128<int> tB2 = Avx2.UnpackHigh(_Tmp0b, _Tmp1b).AsInt32();
        Vector128<int> tC2 = Avx2.UnpackLow(_Tmp2b, _Tmp3b).AsInt32();
        Vector128<int> tD2 = Avx2.UnpackHigh(_Tmp2b, _Tmp3b).AsInt32();

        Avx.StoreAligned((int*)(@out + 0), tA);
        Avx.StoreAligned((int*)(@out + 1), tA2);
        Avx.StoreAligned((int*)(@out + 2), tB);
        Avx.StoreAligned((int*)(@out + 3), tB2);
        Avx.StoreAligned((int*)(@out + 4), tC);
        Avx.StoreAligned((int*)(@out + 5), tC2);
        Avx.StoreAligned((int*)(@out + 6), tD);
        Avx.StoreAligned((int*)(@out + 7), tD2);
        */
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void normalizeEdge<T>(ref Vector128<float> nx, ref Vector128<float> ny, Vector128<float> edgeFlipMask)
        where T : IPossiblyNearClipped
    {
        Vector128<float> minusZero = Vector128.Create(-0.0f);
        Vector128<float> invLen = Sse.Reciprocal(Sse.Add(Sse.AndNot(minusZero, nx), Sse.AndNot(minusZero, ny)));

        const float maxOffset = -minEdgeOffset;
        Vector128<float> mul = Vector128.Create((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        if (T.PossiblyNearClipped)
        {
            mul = Sse.Xor(mul, edgeFlipMask);
        }

        invLen = Sse.Multiply(mul, invLen);
        nx = Sse.Multiply(nx, invLen);
        ny = Sse.Multiply(ny, invLen);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> quantizeSlopeLookup(Vector128<float> nx, Vector128<float> ny)
    {
        Vector128<int> yNeg = Sse.CompareLessThanOrEqual(ny, Vector128<float>.Zero).AsInt32();

        // Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
        const float maxOffset = -minEdgeOffset;
        const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f / ((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
        const float add = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f + 0.5f;

        Vector128<int> quantizedSlope = Sse2.ConvertToVector128Int32WithTruncation(Fma.MultiplyAdd(nx, Vector128.Create(mul), Vector128.Create(add)));
        return Sse2.ShiftLeftLogical(Sse2.Subtract(Sse2.ShiftLeftLogical(quantizedSlope, 1), yNeg), OFFSET_QUANTIZATION_BITS);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> packDepthPremultiplied(Vector128<float> depthA, Vector128<float> depthB)
    {
        Vector128<int> x1 = Sse2.ShiftRightArithmetic(depthA.AsInt32(), 12);
        Vector128<int> x2 = Sse2.ShiftRightArithmetic(depthB.AsInt32(), 12);

        return Sse41.PackUnsignedSaturate(x1, x2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> packDepthPremultiplied(Vector256<float> depth)
    {
        Vector256<int> x = Avx2.ShiftRightArithmetic(depth.AsInt32(), 12);
        return Sse41.PackUnsignedSaturate(x.GetLower(), x.GetUpper());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<ushort> packDepthPremultiplied(Vector256<float> depthA, Vector256<float> depthB)
    {
        Vector256<int> x1 = Avx2.ShiftRightArithmetic(depthA.AsInt32(), 12);
        Vector256<int> x2 = Avx2.ShiftRightArithmetic(depthB.AsInt32(), 12);

        return Avx2.PackUnsignedSaturate(x1, x2);
    }

    public override void rasterize<T>(in Occluder occluder)
    {
        Vector128<int>* vertexData = (Vector128<int>*)occluder.m_vertexData;
        uint packetCount = occluder.m_packetCount;

        Vector128<int> maskY = Vector128.Create(2047 << 10);
        Vector128<int> maskZ = Vector128.Create(1023);

        // Note that unaligned loads do not have a latency penalty on CPUs with SSE4 support
        Vector128<float> mat0 = Sse.LoadVector128(m_modelViewProjection + 0);
        Vector128<float> mat1 = Sse.LoadVector128(m_modelViewProjection + 4);
        Vector128<float> mat2 = Sse.LoadVector128(m_modelViewProjection + 8);
        Vector128<float> mat3 = Sse.LoadVector128(m_modelViewProjection + 12);

        Vector128<float> boundsMin = occluder.m_refMin;
        Vector128<float> boundsExtents = Sse.Subtract(occluder.m_refMax, boundsMin);

        // Bake integer => bounding box transform into matrix
        mat3 =
          MultiplyAdd(mat0, BroadcastScalarToVector128(boundsMin),
            MultiplyAdd(mat1, Permute(boundsMin, 0b01_01_01_01),
              MultiplyAdd(mat2, Permute(boundsMin, 0b10_10_10_10),
                mat3)));

        mat0 = Sse.Multiply(mat0, Sse.Multiply(BroadcastScalarToVector128(boundsExtents), Vector128.Create(1.0f / (2047ul << 21))));
        mat1 = Sse.Multiply(mat1, Sse.Multiply(Permute(boundsExtents, 0b01_01_01_01), Vector128.Create(1.0f / (2047 << 10))));
        mat2 = Sse.Multiply(mat2, Sse.Multiply(Permute(boundsExtents, 0b10_10_10_10), Vector128.Create(1.0f / 1023)));

        // Bias X coordinate back into positive range
        mat3 = MultiplyAdd(mat0, Vector128.Create((float)(1024ul << 21)), mat3);

        // Skew projection to correct bleeding of Y and Z into X due to lack of masking
        mat1 = Sse.Subtract(mat1, mat0);
        mat2 = Sse.Subtract(mat2, mat0);

        _MM_TRANSPOSE4_PS(ref mat0, ref mat1, ref mat2, ref mat3);

        // Due to linear relationship between Z and W, it's cheaper to compute Z from W later in the pipeline than using the full projection matrix up front
        float c0, c1;
        {
            Vector128<float> Za = Permute(mat2, 0b11_11_11_11);
            Vector128<float> Zb = Sse41.DotProduct(mat2, Vector128.Create((float)(1 << 21), 1 << 10, 1, 1), 0xFF);

            Vector128<float> Wa = Permute(mat3, 0b11_11_11_11);
            Vector128<float> Wb = Sse41.DotProduct(mat3, Vector128.Create((float)(1 << 21), 1 << 10, 1, 1), 0xFF);

            Sse.StoreScalar(&c0, Sse.Divide(Sse.Subtract(Za, Zb), Sse.Subtract(Wa, Wb)));
            Sse.StoreScalar(&c1, MultiplyAddNegated(Sse.Divide(Sse.Subtract(Za, Zb), Sse.Subtract(Wa, Wb)), Wa, Za));
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
            Vector128<int>* vertexDataPacketPtr = vertexData + packetIdx * 2;
            Vector128<int> I0 = Sse41.LoadAlignedVector128NonTemporal((int*)(vertexDataPacketPtr + 0));
            Vector128<int> I0b = Sse41.LoadAlignedVector128NonTemporal((int*)(vertexDataPacketPtr + 1));
            Vector128<int> I1 = Sse41.LoadAlignedVector128NonTemporal((int*)(vertexDataPacketPtr + 2));
            Vector128<int> I1b = Sse41.LoadAlignedVector128NonTemporal((int*)(vertexDataPacketPtr + 3));
            Vector128<int> I2 = Sse41.LoadAlignedVector128NonTemporal((int*)(vertexDataPacketPtr + 4));
            Vector128<int> I2b = Sse41.LoadAlignedVector128NonTemporal((int*)(vertexDataPacketPtr + 5));
            Vector128<int> I3 = Sse41.LoadAlignedVector128NonTemporal((int*)(vertexDataPacketPtr + 6));
            Vector128<int> I3b = Sse41.LoadAlignedVector128NonTemporal((int*)(vertexDataPacketPtr + 7));

            // Vertex transformation - first W, then X & Y after camera plane culling, then Z after backface culling
            Vector128<float> Xf0 = Sse2.ConvertToVector128Single(I0);
            Vector128<float> Xf1 = Sse2.ConvertToVector128Single(I1);
            Vector128<float> Xf2 = Sse2.ConvertToVector128Single(I2);
            Vector128<float> Xf3 = Sse2.ConvertToVector128Single(I3);
            Vector128<float> Xf0b = Sse2.ConvertToVector128Single(I0b);
            Vector128<float> Xf1b = Sse2.ConvertToVector128Single(I1b);
            Vector128<float> Xf2b = Sse2.ConvertToVector128Single(I2b);
            Vector128<float> Xf3b = Sse2.ConvertToVector128Single(I3b);

            Vector128<float> Yf0 = Sse2.ConvertToVector128Single(Sse2.And(I0, maskY));
            Vector128<float> Yf1 = Sse2.ConvertToVector128Single(Sse2.And(I1, maskY));
            Vector128<float> Yf2 = Sse2.ConvertToVector128Single(Sse2.And(I2, maskY));
            Vector128<float> Yf3 = Sse2.ConvertToVector128Single(Sse2.And(I3, maskY));
            Vector128<float> Yf0b = Sse2.ConvertToVector128Single(Sse2.And(I0b, maskY));
            Vector128<float> Yf1b = Sse2.ConvertToVector128Single(Sse2.And(I1b, maskY));
            Vector128<float> Yf2b = Sse2.ConvertToVector128Single(Sse2.And(I2b, maskY));
            Vector128<float> Yf3b = Sse2.ConvertToVector128Single(Sse2.And(I3b, maskY));

            Vector128<float> Zf0 = Sse2.ConvertToVector128Single(Sse2.And(I0, maskZ));
            Vector128<float> Zf1 = Sse2.ConvertToVector128Single(Sse2.And(I1, maskZ));
            Vector128<float> Zf2 = Sse2.ConvertToVector128Single(Sse2.And(I2, maskZ));
            Vector128<float> Zf3 = Sse2.ConvertToVector128Single(Sse2.And(I3, maskZ));
            Vector128<float> Zf0b = Sse2.ConvertToVector128Single(Sse2.And(I0b, maskZ));
            Vector128<float> Zf1b = Sse2.ConvertToVector128Single(Sse2.And(I1b, maskZ));
            Vector128<float> Zf2b = Sse2.ConvertToVector128Single(Sse2.And(I2b, maskZ));
            Vector128<float> Zf3b = Sse2.ConvertToVector128Single(Sse2.And(I3b, maskZ));

            Vector128<float> mat00 = Vector128.Create(mat0.GetElement(0));
            Vector128<float> mat01 = Vector128.Create(mat0.GetElement(1));
            Vector128<float> mat02 = Vector128.Create(mat0.GetElement(2));
            Vector128<float> mat03 = Vector128.Create(mat0.GetElement(3));

            Vector128<float> X0 = MultiplyAdd(Xf0, mat00, MultiplyAdd(Yf0, mat01, MultiplyAdd(Zf0, mat02, mat03)));
            Vector128<float> X1 = MultiplyAdd(Xf1, mat00, MultiplyAdd(Yf1, mat01, MultiplyAdd(Zf1, mat02, mat03)));
            Vector128<float> X2 = MultiplyAdd(Xf2, mat00, MultiplyAdd(Yf2, mat01, MultiplyAdd(Zf2, mat02, mat03)));
            Vector128<float> X3 = MultiplyAdd(Xf3, mat00, MultiplyAdd(Yf3, mat01, MultiplyAdd(Zf3, mat02, mat03)));
            Vector128<float> X0b = MultiplyAdd(Xf0b, mat00, MultiplyAdd(Yf0b, mat01, MultiplyAdd(Zf0b, mat02, mat03)));
            Vector128<float> X1b = MultiplyAdd(Xf1b, mat00, MultiplyAdd(Yf1b, mat01, MultiplyAdd(Zf1b, mat02, mat03)));
            Vector128<float> X2b = MultiplyAdd(Xf2b, mat00, MultiplyAdd(Yf2b, mat01, MultiplyAdd(Zf2b, mat02, mat03)));
            Vector128<float> X3b = MultiplyAdd(Xf3b, mat00, MultiplyAdd(Yf3b, mat01, MultiplyAdd(Zf3b, mat02, mat03)));

            Vector128<float> mat10 = Vector128.Create(mat1.GetElement(0));
            Vector128<float> mat11 = Vector128.Create(mat1.GetElement(1));
            Vector128<float> mat12 = Vector128.Create(mat1.GetElement(2));
            Vector128<float> mat13 = Vector128.Create(mat1.GetElement(3));

            Vector128<float> Y0 = MultiplyAdd(Xf0, mat10, MultiplyAdd(Yf0, mat11, MultiplyAdd(Zf0, mat12, mat13)));
            Vector128<float> Y1 = MultiplyAdd(Xf1, mat10, MultiplyAdd(Yf1, mat11, MultiplyAdd(Zf1, mat12, mat13)));
            Vector128<float> Y2 = MultiplyAdd(Xf2, mat10, MultiplyAdd(Yf2, mat11, MultiplyAdd(Zf2, mat12, mat13)));
            Vector128<float> Y3 = MultiplyAdd(Xf3, mat10, MultiplyAdd(Yf3, mat11, MultiplyAdd(Zf3, mat12, mat13)));
            Vector128<float> Y0b = MultiplyAdd(Xf0b, mat10, MultiplyAdd(Yf0b, mat11, MultiplyAdd(Zf0b, mat12, mat13)));
            Vector128<float> Y1b = MultiplyAdd(Xf1b, mat10, MultiplyAdd(Yf1b, mat11, MultiplyAdd(Zf1b, mat12, mat13)));
            Vector128<float> Y2b = MultiplyAdd(Xf2b, mat10, MultiplyAdd(Yf2b, mat11, MultiplyAdd(Zf2b, mat12, mat13)));
            Vector128<float> Y3b = MultiplyAdd(Xf3b, mat10, MultiplyAdd(Yf3b, mat11, MultiplyAdd(Zf3b, mat12, mat13)));

            Vector128<float> mat30 = Vector128.Create(mat3.GetElement(0));
            Vector128<float> mat31 = Vector128.Create(mat3.GetElement(1));
            Vector128<float> mat32 = Vector128.Create(mat3.GetElement(2));
            Vector128<float> mat33 = Vector128.Create(mat3.GetElement(3));

            Vector128<float> W0 = MultiplyAdd(Xf0, mat30, MultiplyAdd(Yf0, mat31, MultiplyAdd(Zf0, mat32, mat33)));
            Vector128<float> W1 = MultiplyAdd(Xf1, mat30, MultiplyAdd(Yf1, mat31, MultiplyAdd(Zf1, mat32, mat33)));
            Vector128<float> W2 = MultiplyAdd(Xf2, mat30, MultiplyAdd(Yf2, mat31, MultiplyAdd(Zf2, mat32, mat33)));
            Vector128<float> W3 = MultiplyAdd(Xf3, mat30, MultiplyAdd(Yf3, mat31, MultiplyAdd(Zf3, mat32, mat33)));
            Vector128<float> W0b = MultiplyAdd(Xf0b, mat30, MultiplyAdd(Yf0b, mat31, MultiplyAdd(Zf0b, mat32, mat33)));
            Vector128<float> W1b = MultiplyAdd(Xf1b, mat30, MultiplyAdd(Yf1b, mat31, MultiplyAdd(Zf1b, mat32, mat33)));
            Vector128<float> W2b = MultiplyAdd(Xf2b, mat30, MultiplyAdd(Yf2b, mat31, MultiplyAdd(Zf2b, mat32, mat33)));
            Vector128<float> W3b = MultiplyAdd(Xf3b, mat30, MultiplyAdd(Yf3b, mat31, MultiplyAdd(Zf3b, mat32, mat33)));

            Vector128<float> invW0, invW1, invW2, invW3, invW0b, invW1b, invW2b, invW3b;
            // Clamp W and invert
            if (T.PossiblyNearClipped)
            {
                Vector128<float> lowerBound = Vector128.Create((float)-maxInvW);
                Vector128<float> upperBound = Vector128.Create((float)+maxInvW);
                invW0 = Sse.Min(upperBound, Sse.Max(lowerBound, Sse.Reciprocal(W0)));
                invW1 = Sse.Min(upperBound, Sse.Max(lowerBound, Sse.Reciprocal(W1)));
                invW2 = Sse.Min(upperBound, Sse.Max(lowerBound, Sse.Reciprocal(W2)));
                invW3 = Sse.Min(upperBound, Sse.Max(lowerBound, Sse.Reciprocal(W3)));
                invW0b = Sse.Min(upperBound, Sse.Max(lowerBound, Sse.Reciprocal(W0b)));
                invW1b = Sse.Min(upperBound, Sse.Max(lowerBound, Sse.Reciprocal(W1b)));
                invW2b = Sse.Min(upperBound, Sse.Max(lowerBound, Sse.Reciprocal(W2b)));
                invW3b = Sse.Min(upperBound, Sse.Max(lowerBound, Sse.Reciprocal(W3b)));
            }
            else
            {
                invW0 = Sse.Reciprocal(W0);
                invW1 = Sse.Reciprocal(W1);
                invW2 = Sse.Reciprocal(W2);
                invW3 = Sse.Reciprocal(W3);
                invW0b = Sse.Reciprocal(W0b);
                invW1b = Sse.Reciprocal(W1b);
                invW2b = Sse.Reciprocal(W2b);
                invW3b = Sse.Reciprocal(W3b);
            }

            // Round to integer coordinates to improve culling of zero-area triangles
            Vector128<float> roundFactor = Vector128.Create(0.125f);
            Vector128<float> x0 = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(X0, invW0)), roundFactor);
            Vector128<float> x1 = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(X1, invW1)), roundFactor);
            Vector128<float> x2 = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(X2, invW2)), roundFactor);
            Vector128<float> x3 = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(X3, invW3)), roundFactor);
            Vector128<float> x0b = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(X0b, invW0b)), roundFactor);
            Vector128<float> x1b = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(X1b, invW1b)), roundFactor);
            Vector128<float> x2b = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(X2b, invW2b)), roundFactor);
            Vector128<float> x3b = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(X3b, invW3b)), roundFactor);

            Vector128<float> y0 = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(Y0, invW0)), roundFactor);
            Vector128<float> y1 = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(Y1, invW1)), roundFactor);
            Vector128<float> y2 = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(Y2, invW2)), roundFactor);
            Vector128<float> y3 = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(Y3, invW3)), roundFactor);
            Vector128<float> y0b = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(Y0b, invW0b)), roundFactor);
            Vector128<float> y1b = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(Y1b, invW1b)), roundFactor);
            Vector128<float> y2b = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(Y2b, invW2b)), roundFactor);
            Vector128<float> y3b = Sse.Multiply(Sse41.RoundToNearestInteger(Sse.Multiply(Y3b, invW3b)), roundFactor);

            // Compute unnormalized edge directions
            Vector128<float> edgeNormalsX0 = Sse.Subtract(y1, y0);
            Vector128<float> edgeNormalsX1 = Sse.Subtract(y2, y1);
            Vector128<float> edgeNormalsX2 = Sse.Subtract(y3, y2);
            Vector128<float> edgeNormalsX3 = Sse.Subtract(y0, y3);
            Vector128<float> edgeNormalsX0b = Sse.Subtract(y1b, y0b);
            Vector128<float> edgeNormalsX1b = Sse.Subtract(y2b, y1b);
            Vector128<float> edgeNormalsX2b = Sse.Subtract(y3b, y2b);
            Vector128<float> edgeNormalsX3b = Sse.Subtract(y0b, y3b);

            Vector128<float> edgeNormalsY0 = Sse.Subtract(x0, x1);
            Vector128<float> edgeNormalsY1 = Sse.Subtract(x1, x2);
            Vector128<float> edgeNormalsY2 = Sse.Subtract(x2, x3);
            Vector128<float> edgeNormalsY3 = Sse.Subtract(x3, x0);
            Vector128<float> edgeNormalsY0b = Sse.Subtract(x0b, x1b);
            Vector128<float> edgeNormalsY1b = Sse.Subtract(x1b, x2b);
            Vector128<float> edgeNormalsY2b = Sse.Subtract(x2b, x3b);
            Vector128<float> edgeNormalsY3b = Sse.Subtract(x3b, x0b);

            Vector128<float> area0 = MultiplySubtract(edgeNormalsX0, edgeNormalsY1, Sse.Multiply(edgeNormalsX1, edgeNormalsY0));
            Vector128<float> area1 = MultiplySubtract(edgeNormalsX1, edgeNormalsY2, Sse.Multiply(edgeNormalsX2, edgeNormalsY1));
            Vector128<float> area2 = MultiplySubtract(edgeNormalsX2, edgeNormalsY3, Sse.Multiply(edgeNormalsX3, edgeNormalsY2));
            Vector128<float> area3 = Sse.Subtract(Sse.Add(area0, area2), area1);

            Vector128<float> area0b = MultiplySubtract(edgeNormalsX0b, edgeNormalsY1b, Sse.Multiply(edgeNormalsX1b, edgeNormalsY0b));
            Vector128<float> area1b = MultiplySubtract(edgeNormalsX1b, edgeNormalsY2b, Sse.Multiply(edgeNormalsX2b, edgeNormalsY1b));
            Vector128<float> area2b = MultiplySubtract(edgeNormalsX2b, edgeNormalsY3b, Sse.Multiply(edgeNormalsX3b, edgeNormalsY2b));
            Vector128<float> area3b = Sse.Subtract(Sse.Add(area0b, area2b), area1b);

            Vector128<float> minusZero128 = Vector128.Create(-0.0f);

            Vector128<float> wSign0, wSign1, wSign2, wSign3, wSign0b, wSign1b, wSign2b, wSign3b;
            if (T.PossiblyNearClipped)
            {
                wSign0 = Sse.And(invW0, minusZero128);
                wSign1 = Sse.And(invW1, minusZero128);
                wSign2 = Sse.And(invW2, minusZero128);
                wSign3 = Sse.And(invW3, minusZero128);
                wSign0b = Sse.And(invW0b, minusZero128);
                wSign1b = Sse.And(invW1b, minusZero128);
                wSign2b = Sse.And(invW2b, minusZero128);
                wSign3b = Sse.And(invW3b, minusZero128);
            }
            else
            {
                wSign0 = Vector128<float>.Zero;
                wSign1 = Vector128<float>.Zero;
                wSign2 = Vector128<float>.Zero;
                wSign3 = Vector128<float>.Zero;
                wSign0b = Vector128<float>.Zero;
                wSign1b = Vector128<float>.Zero;
                wSign2b = Vector128<float>.Zero;
                wSign3b = Vector128<float>.Zero;
            }

            // Compute signs of areas. We treat 0 as negative as this allows treating primitives with zero area as backfacing.
            Vector128<float> areaSign0, areaSign1, areaSign2, areaSign3, areaSign0b, areaSign1b, areaSign2b, areaSign3b;
            if (T.PossiblyNearClipped)
            {
                // Flip areas for each vertex with W < 0. This needs to be done before comparison against 0 rather than afterwards to make sure zero-are triangles are handled correctly.
                areaSign0 = Sse.CompareLessThanOrEqual(Sse.Xor(Sse.Xor(area0, wSign0), Sse.Xor(wSign1, wSign2)), Vector128<float>.Zero);
                areaSign1 = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(Sse.Xor(Sse.Xor(area1, wSign1), Sse.Xor(wSign2, wSign3)), Vector128<float>.Zero));
                areaSign2 = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(Sse.Xor(Sse.Xor(area2, wSign0), Sse.Xor(wSign2, wSign3)), Vector128<float>.Zero));
                areaSign3 = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(Sse.Xor(Sse.Xor(area3, wSign1), Sse.Xor(wSign0, wSign3)), Vector128<float>.Zero));

                areaSign0b = Sse.CompareLessThanOrEqual(Sse.Xor(Sse.Xor(area0b, wSign0b), Sse.Xor(wSign1b, wSign2b)), Vector128<float>.Zero);
                areaSign1b = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(Sse.Xor(Sse.Xor(area1b, wSign1b), Sse.Xor(wSign2b, wSign3b)), Vector128<float>.Zero));
                areaSign2b = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(Sse.Xor(Sse.Xor(area2b, wSign0b), Sse.Xor(wSign2b, wSign3b)), Vector128<float>.Zero));
                areaSign3b = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(Sse.Xor(Sse.Xor(area3b, wSign1b), Sse.Xor(wSign0b, wSign3b)), Vector128<float>.Zero));
            }
            else
            {
                areaSign0 = Sse.CompareLessThanOrEqual(area0, Vector128<float>.Zero);
                areaSign1 = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(area1, Vector128<float>.Zero));
                areaSign2 = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(area2, Vector128<float>.Zero));
                areaSign3 = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(area3, Vector128<float>.Zero));

                areaSign0b = Sse.CompareLessThanOrEqual(area0b, Vector128<float>.Zero);
                areaSign1b = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(area1b, Vector128<float>.Zero));
                areaSign2b = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(area2b, Vector128<float>.Zero));
                areaSign3b = Sse.And(minusZero128, Sse.CompareLessThanOrEqual(area3b, Vector128<float>.Zero));
            }

            Vector128<int> config = Sse2.Or(
              Sse2.Or(Sse2.ShiftRightLogical(areaSign3.AsInt32(), 28), Sse2.ShiftRightLogical(areaSign2.AsInt32(), 29)),
              Sse2.Or(Sse2.ShiftRightLogical(areaSign1.AsInt32(), 30), Sse2.ShiftRightLogical(areaSign0.AsInt32(), 31)));

            Vector128<int> configB = Sse2.Or(
              Sse2.Or(Sse2.ShiftRightLogical(areaSign3b.AsInt32(), 28), Sse2.ShiftRightLogical(areaSign2b.AsInt32(), 29)),
              Sse2.Or(Sse2.ShiftRightLogical(areaSign1b.AsInt32(), 30), Sse2.ShiftRightLogical(areaSign0b.AsInt32(), 31)));

            if (T.PossiblyNearClipped)
            {
                config = Sse2.Or(config,
                  Sse2.Or(
                    Sse2.Or(Sse2.ShiftRightLogical(wSign3.AsInt32(), 24), Sse2.ShiftRightLogical(wSign2.AsInt32(), 25)),
                    Sse2.Or(Sse2.ShiftRightLogical(wSign1.AsInt32(), 26), Sse2.ShiftRightLogical(wSign0.AsInt32(), 27))));

                configB = Sse2.Or(configB,
                  Sse2.Or(
                    Sse2.Or(Sse2.ShiftRightLogical(wSign3b.AsInt32(), 24), Sse2.ShiftRightLogical(wSign2b.AsInt32(), 25)),
                    Sse2.Or(Sse2.ShiftRightLogical(wSign1b.AsInt32(), 26), Sse2.ShiftRightLogical(wSign0b.AsInt32(), 27))));
            }

            Vector128<int> modes;
            Vector128<int> modesB;
            fixed (PrimitiveMode* modeTablePtr = modeTable)
            {
                modes = Sse2.And(Avx2.GatherVector128((int*)modeTablePtr, config, 1), Vector128.Create(0xff));
                modesB = Sse2.And(Avx2.GatherVector128((int*)modeTablePtr, configB, 1), Vector128.Create(0xff));
            }

            if (Sse41.TestZ(modes, modes) &&
                Sse41.TestZ(modesB, modesB))
            {
                continue;
            }

            Vector128<int> primitiveValid = Sse2.CompareGreaterThan(modes, Vector128<int>.Zero);
            Vector128<int> primitiveValidB = Sse2.CompareGreaterThan(modesB, Vector128<int>.Zero);

            Sse2.StoreAligned((int*)(primModes + 0), modes);
            Sse2.StoreAligned((int*)(primModes + 4), modesB);

            Vector128<float> minFx, minFy, maxFx, maxFy, minFxB, minFyB, maxFxB, maxFyB;

            if (T.PossiblyNearClipped)
            {
                // Clipless bounding box computation
                Vector128<float> infP = Vector128.Create(+10000.0f);
                Vector128<float> infN = Vector128.Create(-10000.0f);

                // Find interval of points with W > 0
                Vector128<float> minPx0 = Sse41.BlendVariable(x0, infP, wSign0);
                Vector128<float> minPx1 = Sse41.BlendVariable(x1, infP, wSign1);
                Vector128<float> minPx2 = Sse41.BlendVariable(x2, infP, wSign2);
                Vector128<float> minPx3 = Sse41.BlendVariable(x3, infP, wSign3);
                Vector128<float> minPx0b = Sse41.BlendVariable(x0b, infP, wSign0b);
                Vector128<float> minPx1b = Sse41.BlendVariable(x1b, infP, wSign1b);
                Vector128<float> minPx2b = Sse41.BlendVariable(x2b, infP, wSign2b);
                Vector128<float> minPx3b = Sse41.BlendVariable(x3b, infP, wSign3b);

                Vector128<float> minPx = Sse.Min(
                  Sse.Min(minPx0, minPx1),
                  Sse.Min(minPx2, minPx3));
                Vector128<float> minPxB = Sse.Min(
                  Sse.Min(minPx0b, minPx1b),
                  Sse.Min(minPx2b, minPx3b));

                Vector128<float> minPy0 = Sse41.BlendVariable(y0, infP, wSign0);
                Vector128<float> minPy1 = Sse41.BlendVariable(y1, infP, wSign1);
                Vector128<float> minPy2 = Sse41.BlendVariable(y2, infP, wSign2);
                Vector128<float> minPy3 = Sse41.BlendVariable(y3, infP, wSign3);
                Vector128<float> minPy0b = Sse41.BlendVariable(y0b, infP, wSign0b);
                Vector128<float> minPy1b = Sse41.BlendVariable(y1b, infP, wSign1b);
                Vector128<float> minPy2b = Sse41.BlendVariable(y2b, infP, wSign2b);
                Vector128<float> minPy3b = Sse41.BlendVariable(y3b, infP, wSign3b);

                Vector128<float> minPy = Sse.Min(
                  Sse.Min(minPy0, minPy1),
                  Sse.Min(minPy2, minPy3));
                Vector128<float> minPyB = Sse.Min(
                  Sse.Min(minPy0b, minPy1b),
                  Sse.Min(minPy2b, minPy3b));

                Vector128<float> maxPx0 = Sse.Xor(minPx0, wSign0);
                Vector128<float> maxPx1 = Sse.Xor(minPx1, wSign1);
                Vector128<float> maxPx2 = Sse.Xor(minPx2, wSign2);
                Vector128<float> maxPx3 = Sse.Xor(minPx3, wSign3);
                Vector128<float> maxPx0b = Sse.Xor(minPx0b, wSign0b);
                Vector128<float> maxPx1b = Sse.Xor(minPx1b, wSign1b);
                Vector128<float> maxPx2b = Sse.Xor(minPx2b, wSign2b);
                Vector128<float> maxPx3b = Sse.Xor(minPx3b, wSign3b);

                Vector128<float> maxPx = Sse.Max(
                  Sse.Max(maxPx0, maxPx1),
                  Sse.Max(maxPx2, maxPx3));
                Vector128<float> maxPxB = Sse.Max(
                  Sse.Max(maxPx0b, maxPx1b),
                  Sse.Max(maxPx2b, maxPx3b));

                Vector128<float> maxPy0 = Sse.Xor(minPy0, wSign0);
                Vector128<float> maxPy1 = Sse.Xor(minPy1, wSign1);
                Vector128<float> maxPy2 = Sse.Xor(minPy2, wSign2);
                Vector128<float> maxPy3 = Sse.Xor(minPy3, wSign3);
                Vector128<float> maxPy0b = Sse.Xor(minPy0b, wSign0b);
                Vector128<float> maxPy1b = Sse.Xor(minPy1b, wSign1b);
                Vector128<float> maxPy2b = Sse.Xor(minPy2b, wSign2b);
                Vector128<float> maxPy3b = Sse.Xor(minPy3b, wSign3b);

                Vector128<float> maxPy = Sse.Max(
                  Sse.Max(maxPy0, maxPy1),
                  Sse.Max(maxPy2, maxPy3));
                Vector128<float> maxPyB = Sse.Max(
                  Sse.Max(maxPy0b, maxPy1b),
                  Sse.Max(maxPy2b, maxPy3b));

                // Find interval of points with W < 0
                Vector128<float> minNx0 = Sse41.BlendVariable(infP, x0, wSign0);
                Vector128<float> minNx1 = Sse41.BlendVariable(infP, x1, wSign1);
                Vector128<float> minNx2 = Sse41.BlendVariable(infP, x2, wSign2);
                Vector128<float> minNx3 = Sse41.BlendVariable(infP, x3, wSign3);
                Vector128<float> minNx0b = Sse41.BlendVariable(infP, x0b, wSign0b);
                Vector128<float> minNx1b = Sse41.BlendVariable(infP, x1b, wSign1b);
                Vector128<float> minNx2b = Sse41.BlendVariable(infP, x2b, wSign2b);
                Vector128<float> minNx3b = Sse41.BlendVariable(infP, x3b, wSign3b);

                Vector128<float> minNx = Sse.Min(
                  Sse.Min(minNx0, minNx1),
                  Sse.Min(minNx2, minNx3));
                Vector128<float> minNxB = Sse.Min(
                  Sse.Min(minNx0b, minNx1b),
                  Sse.Min(minNx2b, minNx3b));

                Vector128<float> minNy0 = Sse41.BlendVariable(infP, y0, wSign0);
                Vector128<float> minNy1 = Sse41.BlendVariable(infP, y1, wSign1);
                Vector128<float> minNy2 = Sse41.BlendVariable(infP, y2, wSign2);
                Vector128<float> minNy3 = Sse41.BlendVariable(infP, y3, wSign3);
                Vector128<float> minNy0b = Sse41.BlendVariable(infP, y0b, wSign0b);
                Vector128<float> minNy1b = Sse41.BlendVariable(infP, y1b, wSign1b);
                Vector128<float> minNy2b = Sse41.BlendVariable(infP, y2b, wSign2b);
                Vector128<float> minNy3b = Sse41.BlendVariable(infP, y3b, wSign3b);

                Vector128<float> minNy = Sse.Min(
                  Sse.Min(minNy0, minNy1),
                  Sse.Min(minNy2, minNy3));
                Vector128<float> minNyB = Sse.Min(
                  Sse.Min(minNy0b, minNy1b),
                  Sse.Min(minNy2b, minNy3b));

                Vector128<float> maxNx0 = Sse41.BlendVariable(infN, x0, wSign0);
                Vector128<float> maxNx1 = Sse41.BlendVariable(infN, x1, wSign1);
                Vector128<float> maxNx2 = Sse41.BlendVariable(infN, x2, wSign2);
                Vector128<float> maxNx3 = Sse41.BlendVariable(infN, x3, wSign3);
                Vector128<float> maxNx0b = Sse41.BlendVariable(infN, x0b, wSign0b);
                Vector128<float> maxNx1b = Sse41.BlendVariable(infN, x1b, wSign1b);
                Vector128<float> maxNx2b = Sse41.BlendVariable(infN, x2b, wSign2b);
                Vector128<float> maxNx3b = Sse41.BlendVariable(infN, x3b, wSign3b);

                Vector128<float> maxNx = Sse.Max(
                  Sse.Max(maxNx0, maxNx1),
                  Sse.Max(maxNx2, maxNx3));
                Vector128<float> maxNxB = Sse.Max(
                  Sse.Max(maxNx0b, maxNx1b),
                  Sse.Max(maxNx2b, maxNx3b));

                Vector128<float> maxNy0 = Sse41.BlendVariable(infN, y0, wSign0);
                Vector128<float> maxNy1 = Sse41.BlendVariable(infN, y1, wSign1);
                Vector128<float> maxNy2 = Sse41.BlendVariable(infN, y2, wSign2);
                Vector128<float> maxNy3 = Sse41.BlendVariable(infN, y3, wSign3);
                Vector128<float> maxNy0b = Sse41.BlendVariable(infN, y0b, wSign0b);
                Vector128<float> maxNy1b = Sse41.BlendVariable(infN, y1b, wSign1b);
                Vector128<float> maxNy2b = Sse41.BlendVariable(infN, y2b, wSign2b);
                Vector128<float> maxNy3b = Sse41.BlendVariable(infN, y3b, wSign3b);

                Vector128<float> maxNy = Sse.Max(
                  Sse.Max(maxNy0, maxNy1),
                  Sse.Max(maxNy2, maxNy3));
                Vector128<float> maxNyB = Sse.Max(
                  Sse.Max(maxNy0b, maxNy1b),
                  Sse.Max(maxNy2b, maxNy3b));

                // Include interval bounds resp. infinity depending on ordering of intervals
                Vector128<float> incAx = Sse41.BlendVariable(minPx, infN, Sse.CompareGreaterThan(maxNx, minPx));
                Vector128<float> incAy = Sse41.BlendVariable(minPy, infN, Sse.CompareGreaterThan(maxNy, minPy));
                Vector128<float> incAxB = Sse41.BlendVariable(minPxB, infN, Sse.CompareGreaterThan(maxNxB, minPxB));
                Vector128<float> incAyB = Sse41.BlendVariable(minPyB, infN, Sse.CompareGreaterThan(maxNyB, minPyB));

                Vector128<float> incBx = Sse41.BlendVariable(maxPx, infP, Sse.CompareGreaterThan(maxPx, minNx));
                Vector128<float> incBy = Sse41.BlendVariable(maxPy, infP, Sse.CompareGreaterThan(maxPy, minNy));
                Vector128<float> incBxB = Sse41.BlendVariable(maxPxB, infP, Sse.CompareGreaterThan(maxPxB, minNxB));
                Vector128<float> incByB = Sse41.BlendVariable(maxPyB, infP, Sse.CompareGreaterThan(maxPyB, minNyB));

                minFx = Sse.Min(incAx, incBx);
                minFy = Sse.Min(incAy, incBy);
                minFxB = Sse.Min(incAxB, incBxB);
                minFyB = Sse.Min(incAyB, incByB);

                maxFx = Sse.Max(incAx, incBx);
                maxFy = Sse.Max(incAy, incBy);
                maxFxB = Sse.Max(incAxB, incBxB);
                maxFyB = Sse.Max(incAyB, incByB);
            }
            else
            {
                // Standard bounding box inclusion
                minFx = Sse.Min(Sse.Min(x0, x1), Sse.Min(x2, x3));
                minFy = Sse.Min(Sse.Min(y0, y1), Sse.Min(y2, y3));
                minFxB = Sse.Min(Sse.Min(x0b, x1b), Sse.Min(x2b, x3b));
                minFyB = Sse.Min(Sse.Min(y0b, y1b), Sse.Min(y2b, y3b));

                maxFx = Sse.Max(Sse.Max(x0, x1), Sse.Max(x2, x3));
                maxFy = Sse.Max(Sse.Max(y0, y1), Sse.Max(y2, y3));
                maxFxB = Sse.Max(Sse.Max(x0b, x1b), Sse.Max(x2b, x3b));
                maxFyB = Sse.Max(Sse.Max(y0b, y1b), Sse.Max(y2b, y3b));
            }

            // Clamp and round
            Vector128<int> minX, minY, maxX, maxY, minXB, minYB, maxXB, maxYB;
            minX = Sse41.Max(Sse2.ConvertToVector128Int32WithTruncation(Sse.Add(minFx, Vector128.Create(4.9999f / 8.0f))), Vector128<int>.Zero);
            minY = Sse41.Max(Sse2.ConvertToVector128Int32WithTruncation(Sse.Add(minFy, Vector128.Create(4.9999f / 8.0f))), Vector128<int>.Zero);
            maxX = Sse41.Min(Sse2.ConvertToVector128Int32WithTruncation(Sse.Add(maxFx, Vector128.Create(11.0f / 8.0f))), Vector128.Create((int)m_blocksX));
            maxY = Sse41.Min(Sse2.ConvertToVector128Int32WithTruncation(Sse.Add(maxFy, Vector128.Create(11.0f / 8.0f))), Vector128.Create((int)m_blocksY));
            minXB = Sse41.Max(Sse2.ConvertToVector128Int32WithTruncation(Sse.Add(minFxB, Vector128.Create(4.9999f / 8.0f))), Vector128<int>.Zero);
            minYB = Sse41.Max(Sse2.ConvertToVector128Int32WithTruncation(Sse.Add(minFyB, Vector128.Create(4.9999f / 8.0f))), Vector128<int>.Zero);
            maxXB = Sse41.Min(Sse2.ConvertToVector128Int32WithTruncation(Sse.Add(maxFxB, Vector128.Create(11.0f / 8.0f))), Vector128.Create((int)m_blocksX));
            maxYB = Sse41.Min(Sse2.ConvertToVector128Int32WithTruncation(Sse.Add(maxFyB, Vector128.Create(11.0f / 8.0f))), Vector128.Create((int)m_blocksY));

            // Check overlap between bounding box and frustum
            Vector128<int> inFrustum = Sse2.And(Sse2.CompareGreaterThan(maxX, minX), Sse2.CompareGreaterThan(maxY, minY));
            Vector128<int> overlappedPrimitiveValid = Sse2.And(inFrustum, primitiveValid);
            Vector128<int> inFrustumB = Sse2.And(Sse2.CompareGreaterThan(maxXB, minXB), Sse2.CompareGreaterThan(maxYB, minYB));
            Vector128<int> overlappedPrimitiveValidB = Sse2.And(inFrustumB, primitiveValidB);

            if (Sse41.TestZ(overlappedPrimitiveValid, overlappedPrimitiveValid) &&
                Sse41.TestZ(overlappedPrimitiveValidB, overlappedPrimitiveValidB))
            {
                continue;
            }

            uint validMask = (uint)Sse.MoveMask(overlappedPrimitiveValid.AsSingle());
            validMask |= (uint)Sse.MoveMask(overlappedPrimitiveValidB.AsSingle()) << 4;

            // Convert bounds from [min, max] to [min, range]
            Vector128<int> rangeX = Sse2.Subtract(maxX, minX);
            Vector128<int> rangeY = Sse2.Subtract(maxY, minY);
            Vector128<int> rangeXB = Sse2.Subtract(maxXB, minXB);
            Vector128<int> rangeYB = Sse2.Subtract(maxYB, minYB);

            // Compute Z from linear relation with 1/W
            Vector128<float> C0 = Vector128.Create(c0);
            Vector128<float> C1 = Vector128.Create(c1);
            Vector128<float> z0, z1, z2, z3, z0b, z1b, z2b, z3b;
            z0 = Fma.MultiplyAdd(invW0, C1, C0);
            z1 = Fma.MultiplyAdd(invW1, C1, C0);
            z2 = Fma.MultiplyAdd(invW2, C1, C0);
            z3 = Fma.MultiplyAdd(invW3, C1, C0);
            z0b = Fma.MultiplyAdd(invW0b, C1, C0);
            z1b = Fma.MultiplyAdd(invW1b, C1, C0);
            z2b = Fma.MultiplyAdd(invW2b, C1, C0);
            z3b = Fma.MultiplyAdd(invW3b, C1, C0);

            Vector128<float> maxZ = Sse.Max(Sse.Max(z0, z1), Sse.Max(z2, z3));
            Vector128<float> maxZB = Sse.Max(Sse.Max(z0b, z1b), Sse.Max(z2b, z3b));

            // If any W < 0, assume maxZ = 1 (effectively disabling Hi-Z)
            if (T.PossiblyNearClipped)
            {
                maxZ = Sse41.BlendVariable(maxZ, Vector128.Create(1.0f), Sse.Or(Sse.Or(wSign0, wSign1), Sse.Or(wSign2, wSign3)));
                maxZB = Sse41.BlendVariable(maxZB, Vector128.Create(1.0f), Sse.Or(Sse.Or(wSign0b, wSign1b), Sse.Or(wSign2b, wSign3b)));
            }

            Vector128<ushort> packedDepthBounds = packDepthPremultiplied(maxZ, maxZB);

            Sse2.StoreAligned(depthBounds, packedDepthBounds);

            // Compute screen space depth plane
            Vector128<float> greaterArea = Sse.CompareLessThan(Sse.AndNot(minusZero128, area0), Sse.AndNot(minusZero128, area2));
            Vector128<float> greaterAreaB = Sse.CompareLessThan(Sse.AndNot(minusZero128, area0b), Sse.AndNot(minusZero128, area2b));

            // Force triangle area to be picked in the relevant mode.
            Vector128<float> modeTriangle0 = Sse2.CompareEqual(modes, Vector128.Create((int)PrimitiveMode.Triangle0)).AsSingle();
            Vector128<float> modeTriangle1 = Sse2.CompareEqual(modes, Vector128.Create((int)PrimitiveMode.Triangle1)).AsSingle();
            greaterArea = Sse.AndNot(modeTriangle0, Sse.Or(modeTriangle1, greaterArea));

            Vector128<float> modeTriangle0b = Sse2.CompareEqual(modesB, Vector128.Create((int)PrimitiveMode.Triangle0)).AsSingle();
            Vector128<float> modeTriangle1b = Sse2.CompareEqual(modesB, Vector128.Create((int)PrimitiveMode.Triangle1)).AsSingle();
            greaterAreaB = Sse.AndNot(modeTriangle0b, Sse.Or(modeTriangle1b, greaterAreaB));

            Vector128<float> invArea, invAreaB;
            if (T.PossiblyNearClipped)
            {
                // Do a precise divison to reduce error in depth plane. Note that the area computed here
                // differs from the rasterized region if W < 0, so it can be very small for large covered screen regions.
                invArea = Sse.Divide(Vector128.Create(1.0f), Sse41.BlendVariable(area0, area2, greaterArea));
                invAreaB = Sse.Divide(Vector128.Create(1.0f), Sse41.BlendVariable(area0b, area2b, greaterAreaB));
            }
            else
            {
                invArea = Sse.Reciprocal(Sse41.BlendVariable(area0, area2, greaterArea));
                invAreaB = Sse.Reciprocal(Sse41.BlendVariable(area0b, area2b, greaterAreaB));
            }

            Vector128<float> z12 = Sse.Subtract(z1, z2);
            Vector128<float> z20 = Sse.Subtract(z2, z0);
            Vector128<float> z30 = Sse.Subtract(z3, z0);
            Vector128<float> z12b = Sse.Subtract(z1b, z2b);
            Vector128<float> z20b = Sse.Subtract(z2b, z0b);
            Vector128<float> z30b = Sse.Subtract(z3b, z0b);

            Vector128<float> edgeNormalsX4 = Sse.Subtract(y0, y2);
            Vector128<float> edgeNormalsY4 = Sse.Subtract(x2, x0);
            Vector128<float> edgeNormalsX4b = Sse.Subtract(y0b, y2b);
            Vector128<float> edgeNormalsY4b = Sse.Subtract(x2b, x0b);

            Vector128<float> depthPlane0, depthPlane1, depthPlane2, depthPlane0b, depthPlane1b, depthPlane2b;
            depthPlane1 = Sse.Multiply(invArea, Sse41.BlendVariable(Fma.MultiplySubtract(z20, edgeNormalsX1, Sse.Multiply(z12, edgeNormalsX4)), Fma.MultiplyAddNegated(z20, edgeNormalsX3, Sse.Multiply(z30, edgeNormalsX4)), greaterArea));
            depthPlane2 = Sse.Multiply(invArea, Sse41.BlendVariable(Fma.MultiplySubtract(z20, edgeNormalsY1, Sse.Multiply(z12, edgeNormalsY4)), Fma.MultiplyAddNegated(z20, edgeNormalsY3, Sse.Multiply(z30, edgeNormalsY4)), greaterArea));
            depthPlane1b = Sse.Multiply(invAreaB, Sse41.BlendVariable(MultiplySubtract(z20b, edgeNormalsX1b, Sse.Multiply(z12b, edgeNormalsX4b)), Fma.MultiplyAddNegated(z20b, edgeNormalsX3b, Sse.Multiply(z30b, edgeNormalsX4b)), greaterAreaB));
            depthPlane2b = Sse.Multiply(invAreaB, Sse41.BlendVariable(MultiplySubtract(z20b, edgeNormalsY1b, Sse.Multiply(z12b, edgeNormalsY4b)), Fma.MultiplyAddNegated(z20b, edgeNormalsY3b, Sse.Multiply(z30b, edgeNormalsY4b)), greaterAreaB));

            x0 = Sse.Subtract(x0, Sse2.ConvertToVector128Single(minX));
            y0 = Sse.Subtract(y0, Sse2.ConvertToVector128Single(minY));
            x0b = Sse.Subtract(x0b, Sse2.ConvertToVector128Single(minXB));
            y0b = Sse.Subtract(y0b, Sse2.ConvertToVector128Single(minYB));

            depthPlane0 = Fma.MultiplyAddNegated(x0, depthPlane1, Fma.MultiplyAddNegated(y0, depthPlane2, z0));
            depthPlane0b = Fma.MultiplyAddNegated(x0b, depthPlane1b, Fma.MultiplyAddNegated(y0b, depthPlane2b, z0b));

            // If mode == Triangle0, replace edge 2 with edge 4; if mode == Triangle1, replace edge 0 with edge 4
            edgeNormalsX2 = Sse41.BlendVariable(edgeNormalsX2, edgeNormalsX4, modeTriangle0);
            edgeNormalsY2 = Sse41.BlendVariable(edgeNormalsY2, edgeNormalsY4, modeTriangle0);
            edgeNormalsX0 = Sse41.BlendVariable(edgeNormalsX0, Sse.Xor(minusZero128, edgeNormalsX4), modeTriangle1);
            edgeNormalsY0 = Sse41.BlendVariable(edgeNormalsY0, Sse.Xor(minusZero128, edgeNormalsY4), modeTriangle1);
            edgeNormalsX2b = Sse41.BlendVariable(edgeNormalsX2b, edgeNormalsX4b, modeTriangle0b);
            edgeNormalsY2b = Sse41.BlendVariable(edgeNormalsY2b, edgeNormalsY4b, modeTriangle0b);
            edgeNormalsX0b = Sse41.BlendVariable(edgeNormalsX0b, Sse.Xor(minusZero128, edgeNormalsX4b), modeTriangle1b);
            edgeNormalsY0b = Sse41.BlendVariable(edgeNormalsY0b, Sse.Xor(minusZero128, edgeNormalsY4b), modeTriangle1b);

            // Flip edges if W < 0
            Vector128<float> edgeFlipMask0, edgeFlipMask1, edgeFlipMask2, edgeFlipMask3, edgeFlipMask0b, edgeFlipMask1b, edgeFlipMask2b, edgeFlipMask3b;
            if (T.PossiblyNearClipped)
            {
                edgeFlipMask0 = Sse.Xor(wSign0, Sse41.BlendVariable(wSign1, wSign2, modeTriangle1));
                edgeFlipMask1 = Sse.Xor(wSign1, wSign2);
                edgeFlipMask2 = Sse.Xor(wSign2, Sse41.BlendVariable(wSign3, wSign0, modeTriangle0));
                edgeFlipMask3 = Sse.Xor(wSign0, wSign3);

                edgeFlipMask0b = Sse.Xor(wSign0b, Sse41.BlendVariable(wSign1b, wSign2b, modeTriangle1b));
                edgeFlipMask1b = Sse.Xor(wSign1b, wSign2b);
                edgeFlipMask2b = Sse.Xor(wSign2b, Sse41.BlendVariable(wSign3b, wSign0b, modeTriangle0b));
                edgeFlipMask3b = Sse.Xor(wSign0b, wSign3b);
            }
            else
            {
                edgeFlipMask0 = Vector128<float>.Zero;
                edgeFlipMask1 = Vector128<float>.Zero;
                edgeFlipMask2 = Vector128<float>.Zero;
                edgeFlipMask3 = Vector128<float>.Zero;
                edgeFlipMask0b = Vector128<float>.Zero;
                edgeFlipMask1b = Vector128<float>.Zero;
                edgeFlipMask2b = Vector128<float>.Zero;
                edgeFlipMask3b = Vector128<float>.Zero;
            }

            // Normalize edge equations for lookup
            normalizeEdge<T>(ref edgeNormalsX0, ref edgeNormalsY0, edgeFlipMask0);
            normalizeEdge<T>(ref edgeNormalsX1, ref edgeNormalsY1, edgeFlipMask1);
            normalizeEdge<T>(ref edgeNormalsX2, ref edgeNormalsY2, edgeFlipMask2);
            normalizeEdge<T>(ref edgeNormalsX3, ref edgeNormalsY3, edgeFlipMask3);
            normalizeEdge<T>(ref edgeNormalsX0b, ref edgeNormalsY0b, edgeFlipMask0b);
            normalizeEdge<T>(ref edgeNormalsX1b, ref edgeNormalsY1b, edgeFlipMask1b);
            normalizeEdge<T>(ref edgeNormalsX2b, ref edgeNormalsY2b, edgeFlipMask2b);
            normalizeEdge<T>(ref edgeNormalsX3b, ref edgeNormalsY3b, edgeFlipMask3b);

            const float maxOffset = -minEdgeOffset;
            Vector128<float> add128 = Vector128.Create(0.5f - minEdgeOffset * (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
            Vector128<float> edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3, edgeOffsets0b, edgeOffsets1b, edgeOffsets2b, edgeOffsets3b;

            edgeOffsets0 = MultiplyAddNegated(x0, edgeNormalsX0, MultiplyAddNegated(y0, edgeNormalsY0, add128));
            edgeOffsets1 = MultiplyAddNegated(x1, edgeNormalsX1, MultiplyAddNegated(y1, edgeNormalsY1, add128));
            edgeOffsets2 = MultiplyAddNegated(x2, edgeNormalsX2, MultiplyAddNegated(y2, edgeNormalsY2, add128));
            edgeOffsets3 = MultiplyAddNegated(x3, edgeNormalsX3, MultiplyAddNegated(y3, edgeNormalsY3, add128));
            edgeOffsets0b = MultiplyAddNegated(x0b, edgeNormalsX0b, MultiplyAddNegated(y0b, edgeNormalsY0b, add128));
            edgeOffsets1b = MultiplyAddNegated(x1b, edgeNormalsX1b, MultiplyAddNegated(y1b, edgeNormalsY1b, add128));
            edgeOffsets2b = MultiplyAddNegated(x2b, edgeNormalsX2b, MultiplyAddNegated(y2b, edgeNormalsY2b, add128));
            edgeOffsets3b = MultiplyAddNegated(x3b, edgeNormalsX3b, MultiplyAddNegated(y3b, edgeNormalsY3b, add128));

            edgeOffsets1 = MultiplyAdd(Sse2.ConvertToVector128Single(minX), edgeNormalsX1, edgeOffsets1);
            edgeOffsets2 = MultiplyAdd(Sse2.ConvertToVector128Single(minX), edgeNormalsX2, edgeOffsets2);
            edgeOffsets3 = MultiplyAdd(Sse2.ConvertToVector128Single(minX), edgeNormalsX3, edgeOffsets3);
            edgeOffsets1b = MultiplyAdd(Sse2.ConvertToVector128Single(minXB), edgeNormalsX1b, edgeOffsets1b);
            edgeOffsets2b = MultiplyAdd(Sse2.ConvertToVector128Single(minXB), edgeNormalsX2b, edgeOffsets2b);
            edgeOffsets3b = MultiplyAdd(Sse2.ConvertToVector128Single(minXB), edgeNormalsX3b, edgeOffsets3b);

            edgeOffsets1 = MultiplyAdd(Sse2.ConvertToVector128Single(minY), edgeNormalsY1, edgeOffsets1);
            edgeOffsets2 = MultiplyAdd(Sse2.ConvertToVector128Single(minY), edgeNormalsY2, edgeOffsets2);
            edgeOffsets3 = MultiplyAdd(Sse2.ConvertToVector128Single(minY), edgeNormalsY3, edgeOffsets3);
            edgeOffsets1b = MultiplyAdd(Sse2.ConvertToVector128Single(minYB), edgeNormalsY1b, edgeOffsets1b);
            edgeOffsets2b = MultiplyAdd(Sse2.ConvertToVector128Single(minYB), edgeNormalsY2b, edgeOffsets2b);
            edgeOffsets3b = MultiplyAdd(Sse2.ConvertToVector128Single(minYB), edgeNormalsY3b, edgeOffsets3b);

            // Quantize slopes
            Vector128<int> slopeLookups0 = quantizeSlopeLookup(edgeNormalsX0, edgeNormalsY0);
            Vector128<int> slopeLookups1 = quantizeSlopeLookup(edgeNormalsX1, edgeNormalsY1);
            Vector128<int> slopeLookups2 = quantizeSlopeLookup(edgeNormalsX2, edgeNormalsY2);
            Vector128<int> slopeLookups3 = quantizeSlopeLookup(edgeNormalsX3, edgeNormalsY3);
            Vector128<int> slopeLookups0b = quantizeSlopeLookup(edgeNormalsX0b, edgeNormalsY0b);
            Vector128<int> slopeLookups1b = quantizeSlopeLookup(edgeNormalsX1b, edgeNormalsY1b);
            Vector128<int> slopeLookups2b = quantizeSlopeLookup(edgeNormalsX2b, edgeNormalsY2b);
            Vector128<int> slopeLookups3b = quantizeSlopeLookup(edgeNormalsX3b, edgeNormalsY3b);

            Vector128<int> firstBlockIdx = Sse2.Add(Sse2.MultiplyLow(minY.AsInt16(), Vector128.Create((int)m_blocksX).AsInt16()).AsInt32(), minX);
            Vector128<int> firstBlockIdxB = Sse2.Add(Sse2.MultiplyLow(minYB.AsInt16(), Vector128.Create((int)m_blocksX).AsInt16()).AsInt32(), minXB);

            Sse2.StoreAligned((int*)(firstBlocks + 0), firstBlockIdx);
            Sse2.StoreAligned((int*)(firstBlocks + 4), firstBlockIdxB);

            Sse2.StoreAligned((int*)(rangesX + 0), rangeX);
            Sse2.StoreAligned((int*)(rangesX + 4), rangeXB);

            Sse2.StoreAligned((int*)(rangesY + 0), rangeY);
            Sse2.StoreAligned((int*)(rangesY + 4), rangeYB);

            // Transpose into AoS
            transpose256(
                depthPlane0, depthPlane0b,
                depthPlane1, depthPlane1b,
                depthPlane2, depthPlane2b,
                Vector128<float>.Zero, Vector128<float>.Zero,
                depthPlane);

            transpose256(
                edgeNormalsX0, edgeNormalsX0b,
                edgeNormalsX1, edgeNormalsX1b,
                edgeNormalsX2, edgeNormalsX2b,
                edgeNormalsX3, edgeNormalsX3b,
                edgeNormalsX);

            transpose256(
                edgeNormalsY0, edgeNormalsY0b,
                edgeNormalsY1, edgeNormalsY1b,
                edgeNormalsY2, edgeNormalsY2b,
                edgeNormalsY3, edgeNormalsY3b,
                edgeNormalsY);

            transpose256(
                edgeOffsets0, edgeOffsets0b,
                edgeOffsets1, edgeOffsets1b,
                edgeOffsets2, edgeOffsets2b,
                edgeOffsets3, edgeOffsets3b,
                edgeOffsets);

            transpose256i(
                slopeLookups0, slopeLookups0b,
                slopeLookups1, slopeLookups1b,
                slopeLookups2, slopeLookups2b,
                slopeLookups3, slopeLookups3b,
                slopeLookups);

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
        Vector128<float>* depthPlane,
        Vector128<int>* slopeLookups,
        Vector128<float>* edgeNormalsX,
        Vector128<float>* edgeNormalsY,
        Vector128<float>* edgeOffsets,
        uint* firstBlocks,
        uint* rangesX,
        uint* rangesY,
        uint* primModes)
        where T : IPossiblyNearClipped
    {
        // Fetch data pointers since we'll manually strength-reduce memory arithmetic
        ulong* pTable = (ulong*)m_precomputedRasterTables.DangerousGetHandle();
        ushort* pHiZBuffer = m_hiZ;
        Vector128<int>* pDepthBuffer = m_depthBuffer;

        const float depthSamplePos = -0.5f + 1.0f / 16.0f;

        Vector128<float> depthSamplePosFactor1 = Vector128.Create(
            depthSamplePos + 0.0f, depthSamplePos + 0.125f, depthSamplePos + 0.25f, depthSamplePos + 0.375f);

        Vector128<float> depthSamplePosFactor2A = Vector128.Create(depthSamplePos);
        Vector128<float> depthSamplePosFactor2B = Vector128.Create(depthSamplePos + 0.125f);

        // Loop over set bits
        while (validMask != 0)
        {
            uint primitiveIdx = (uint)BitOperations.TrailingZeroCount(validMask);

            // Clear lowest set bit in mask
            validMask &= validMask - 1;

            uint primitiveIdxTransposed = ((primitiveIdx << 1) & 7) | (primitiveIdx >> 2);

            // Extract and prepare per-primitive data
            ushort primitiveMaxZ = depthBounds[primitiveIdx];

            Vector128<float> depthDx = BroadcastScalarToVector128(Permute(Sse.LoadAlignedVector128((float*)(depthPlane + primitiveIdxTransposed)), 0b01_01_01_01));
            Vector128<float> depthDy = BroadcastScalarToVector128(Permute(Sse.LoadAlignedVector128((float*)(depthPlane + primitiveIdxTransposed)), 0b10_10_10_10));

            Vector128<float> lineDepthTerm = Vector128.Create(*(float*)(depthPlane + primitiveIdxTransposed));

            Vector128<float> lineDepthA =
              MultiplyAdd(depthDx, depthSamplePosFactor1,
                MultiplyAdd(depthDy, depthSamplePosFactor2A,
                  lineDepthTerm));

            Vector128<float> lineDepthB =
              MultiplyAdd(depthDx, depthSamplePosFactor1,
                MultiplyAdd(depthDy, depthSamplePosFactor2B,
                  lineDepthTerm));

            Vector128<int> slopeLookup = Sse2.LoadAlignedVector128((int*)(slopeLookups + primitiveIdxTransposed));
            Vector128<float> edgeNormalX = Sse.LoadAlignedVector128((float*)(edgeNormalsX + primitiveIdxTransposed));
            Vector128<float> edgeNormalY = Sse.LoadAlignedVector128((float*)(edgeNormalsY + primitiveIdxTransposed));
            Vector128<float> lineOffset = Sse.LoadAlignedVector128((float*)(edgeOffsets + primitiveIdxTransposed));

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
              lineDepthA = Sse.Add(lineDepthA, depthDy),
              lineDepthB = Sse.Add(lineDepthB, depthDy),
              lineOffset = Sse.Add(lineOffset, edgeNormalY))
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
                  depthA = Sse.Add(depthDx, depthA),
                  depthB = Sse.Add(depthDx, depthB),
                  offset = Sse.Add(edgeNormalX, offset))
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
                        if (!TestZ(anyOffsetOutsideMask, anyOffsetOutsideMask))
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

                        Vector128<int> lookup = Sse2.Or(slopeLookup, offsetClamped);

                        // Generate block mask
                        ulong A = pTable[(uint)Sse2.ConvertToInt32(lookup)];
                        ulong B = pTable[(uint)Sse41.Extract(lookup, 1)];
                        ulong C = pTable[(uint)Sse41.Extract(lookup, 2)];
                        ulong D = pTable[(uint)Sse41.Extract(lookup, 3)];

                        blockMask = A & B & C & D;

                        // It is possible but very unlikely that blockMask == 0 if all A,B,C,D != 0 according to the conservative test above, so we skip the additional branch here.
                    }
                    else
                    {
                        Vector128<int> offsetClamped = Sse41.Min(Sse41.Max(Sse2.ConvertToVector128Int32WithTruncation(offset), Vector128<int>.Zero), Vector128.Create(OFFSET_QUANTIZATION_FACTOR - 1));
                        Vector128<int> lookup = Sse2.Or(slopeLookup, offsetClamped);

                        // Generate block mask
                        ulong A = pTable[(uint)Sse2.ConvertToInt32(lookup)];
                        ulong B = pTable[(uint)Sse41.Extract(lookup, 1)];
                        ulong C = pTable[(uint)Sse41.Extract(lookup, 2)];
                        ulong D = pTable[(uint)Sse41.Extract(lookup, 3)];

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
                    Vector128<float> depth0_A = depthA;
                    Vector128<float> depth1_A = MultiplyAdd(depthDx, Vector128.Create(0.5f), depth0_A);
                    Vector128<float> depth8_A = Sse.Add(depthDy, depth0_A);
                    Vector128<float> depth9_A = Sse.Add(depthDy, depth1_A);

                    Vector128<float> depth0_B = depthB;
                    Vector128<float> depth1_B = MultiplyAdd(depthDx, Vector128.Create(0.5f), depth0_B);
                    Vector128<float> depth8_B = Sse.Add(depthDy, depth0_B);
                    Vector128<float> depth9_B = Sse.Add(depthDy, depth1_B);

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return Sse.Add(Sse.Multiply(a, b), c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> MultiplySubtract(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return Sse.Subtract(Sse.Multiply(a, b), c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> MultiplyAddNegated(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return Sse.Subtract(c, Sse.Multiply(a, b));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TestZ(Vector128<float> a, Vector128<float> b)
    {
        Vector128<float> mask = Vector128.Create(0x80000000u).AsSingle();
        return Sse41.TestZ(Sse.And(a, mask).AsUInt32(), Sse.And(b, mask).AsUInt32());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Permute(Vector128<float> a, byte imm8)
    {
        return Sse.Shuffle(a, a, imm8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> BroadcastScalarToVector128(Vector128<float> a)
    {
        return Vector128.Create(a.ToScalar());
    }
}