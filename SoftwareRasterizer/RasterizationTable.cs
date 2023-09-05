using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

using static Rasterizer;

public sealed unsafe class RasterizationTable : SafeHandle
{
    private const uint angularResolution = 2000;
    private const uint offsetResolution = 2000;

    public override bool IsInvalid => handle == IntPtr.Zero;

    public RasterizationTable() : base(0, true)
    {
        uint precomputedRasterTablesByteCount = OFFSET_QUANTIZATION_FACTOR * SLOPE_QUANTIZATION_FACTOR * sizeof(ulong);
        ulong* precomputedRasterTables = (ulong*)NativeMemory.AlignedAlloc(
            byteCount: precomputedRasterTablesByteCount,
            alignment: sizeof(ulong));

        Unsafe.InitBlockUnaligned(precomputedRasterTables, 0, precomputedRasterTablesByteCount);

        if (Vector256.IsHardwareAccelerated)
        {
            ComputeRasterizationTableV256(precomputedRasterTables);
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            ComputeRasterizationTableV128(precomputedRasterTables);
        }
        else
        {
            ComputeRasterizationTableScalar(precomputedRasterTables);
        }

        handle = (IntPtr)precomputedRasterTables;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ComputeRasterizationTableV256(ulong* table)
    {
        for (uint i = 0; i < angularResolution; ++i)
        {
            float angle = -0.1f + 6.4f * i / (angularResolution - 1);

            (float ny, float nx) = MathF.SinCos(angle);
            float l = 1.0f / (MathF.Abs(nx) + MathF.Abs(ny));

            nx *= l;
            ny *= l;

            uint slopeLookup = (uint)quantizeSlopeLookup(nx, ny);

            Vector256<float> inc = Vector256.Create(
                (0 - 3.5f) / 8f,
                (1 - 3.5f) / 8f,
                (2 - 3.5f) / 8f,
                (3 - 3.5f) / 8f,
                (4 - 3.5f) / 8f,
                (5 - 3.5f) / 8f,
                (6 - 3.5f) / 8f,
                (7 - 3.5f) / 8f);

            Vector256<float> incX = (inc * Vector256.Create(nx));

            for (uint j = 0; j < offsetResolution; ++j)
            {
                float offset = -0.6f + 1.2f * j / (angularResolution - 1);

                uint offsetLookup = quantizeOffsetLookup(offset);

                uint lookup = slopeLookup | offsetLookup;

                ulong block = 0;

                for (int y = 0; y < 8; ++y)
                {
                    Vector256<float> o = Vector256.Create(offset + (y - 3.5f) / 8.0f * ny);
                    Vector256<float> edgeDistance = (o + incX);
                    Vector256<float> cmp = Vector256.LessThanOrEqual(edgeDistance, Vector256<float>.Zero);

                    uint mask = Vector256.ExtractMostSignificantBits(cmp);
                    block |= expandMask(mask) << y;
                }

                table[lookup] |= transposeMask(block);
            }

            // For each slope, the first block should be all ones, the last all zeroes
            Debug.Assert(table[slopeLookup] == 0xffff_ffff_ffff_ffff);
            Debug.Assert(table[slopeLookup + OFFSET_QUANTIZATION_FACTOR - 1] == 0);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ComputeRasterizationTableV128(ulong* table)
    {
        for (uint i = 0; i < angularResolution; ++i)
        {
            float angle = -0.1f + 6.4f * i / (angularResolution - 1);

            (float ny, float nx) = MathF.SinCos(angle);
            float l = 1.0f / (MathF.Abs(nx) + MathF.Abs(ny));

            nx *= l;
            ny *= l;

            uint slopeLookup = (uint)quantizeSlopeLookup(nx, ny);

            Vector128<float> inc1 = Vector128.Create(
                (0 - 3.5f) / 8f,
                (1 - 3.5f) / 8f,
                (2 - 3.5f) / 8f,
                (3 - 3.5f) / 8f);

            Vector128<float> inc2 = Vector128.Create(
                (4 - 3.5f) / 8f,
                (5 - 3.5f) / 8f,
                (6 - 3.5f) / 8f,
                (7 - 3.5f) / 8f);

            Vector128<float> incX1 = (inc1 * Vector128.Create(nx));
            Vector128<float> incX2 = (inc2 * Vector128.Create(nx));

            for (uint j = 0; j < offsetResolution; ++j)
            {
                float offset = -0.6f + 1.2f * j / (angularResolution - 1);

                uint offsetLookup = quantizeOffsetLookup(offset);

                uint lookup = slopeLookup | offsetLookup;

                ulong block = 0;

                for (int y = 0; y < 8; ++y)
                {
                    Vector128<float> o = Vector128.Create(offset + (y - 3.5f) / 8.0f * ny);
                    Vector128<float> edgeDistance1 = (o + incX1);
                    Vector128<float> edgeDistance2 = (o + incX2);
                    Vector128<float> cmp1 = Vector128.LessThanOrEqual(edgeDistance1, Vector128<float>.Zero);
                    Vector128<float> cmp2 = Vector128.LessThanOrEqual(edgeDistance2, Vector128<float>.Zero);

                    uint mask = Vector128.ExtractMostSignificantBits(cmp1) | (Vector128.ExtractMostSignificantBits(cmp2) << 4);
                    block |= expandMask(mask) << y;
                }

                table[lookup] |= transposeMask(block);
            }

            // For each slope, the first block should be all ones, the last all zeroes
            Debug.Assert(table[slopeLookup] == 0xffff_ffff_ffff_ffff);
            Debug.Assert(table[slopeLookup + OFFSET_QUANTIZATION_FACTOR - 1] == 0);
        }
    }

    private static void ComputeRasterizationTableScalar(ulong* table)
    {
        for (uint i = 0; i < angularResolution; ++i)
        {
            float angle = -0.1f + 6.4f * i / (angularResolution - 1);

            (float ny, float nx) = MathF.SinCos(angle);
            float l = 1.0f / (MathF.Abs(nx) + MathF.Abs(ny));

            nx *= l;
            ny *= l;

            uint slopeLookup = (uint)quantizeSlopeLookup(nx, ny);

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

                table[lookup] |= transposeMask(block);
            }

            // For each slope, the first block should be all ones, the last all zeroes
            Debug.Assert(table[slopeLookup] == 0xffff_ffff_ffff_ffff);
            Debug.Assert(table[slopeLookup + OFFSET_QUANTIZATION_FACTOR - 1] == 0);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong transposeMask(ulong mask)
    {
        if (Bmi2.X64.IsSupported)
        {
            ulong maskA = Bmi2.X64.ParallelBitDeposit(Bmi2.X64.ParallelBitExtract(mask, 0x5555555555555555ul), 0xF0F0F0F0F0F0F0F0ul);
            ulong maskB = Bmi2.X64.ParallelBitDeposit(Bmi2.X64.ParallelBitExtract(mask, 0xAAAAAAAAAAAAAAAAul), 0x0F0F0F0F0F0F0F0Ful);
            return maskA | maskB;
        }
        else
        {
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
            return maskA | maskB;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong expandMask(uint mask)
    {
        if (Bmi2.X64.IsSupported)
        {
            return Bmi2.X64.ParallelBitDeposit(mask, 0x101010101010101u);
        }
        else
        {
            uint a = 0;
            a |= (mask & 0b00000001u);
            a |= (mask & 0b00000010u) << 7;
            a |= (mask & 0b00000100u) << 14;
            a |= (mask & 0b00001000u) << 21;

            uint b = 0;
            b |= (mask & 0b00010000u) >> 4;
            b |= (mask & 0b00100000u) << 3;
            b |= (mask & 0b01000000u) << 10;
            b |= (mask & 0b10000000u) << 17;

            return ((ulong)b << 32) | a;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int quantizeSlopeLookup(float nx, float ny)
    {
        int yNeg = ny < 0 ? unchecked((int)0xFFFFFFFF) : 0;

        // Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
        const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f;
        const float add = mul + 0.5f;

        int quantizedSlope = (int)(nx * mul + add);
        return ((quantizedSlope << 1) - yNeg) << OFFSET_QUANTIZATION_BITS;
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

    protected override bool ReleaseHandle()
    {
        NativeMemory.AlignedFree((void*)handle);
        return true;
    }
}
