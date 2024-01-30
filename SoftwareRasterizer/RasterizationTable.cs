using System;
using System.Runtime.InteropServices;

namespace SoftwareRasterizer;

using static Rasterizer;

public sealed unsafe partial class RasterizationTable : SafeHandle
{
    public override bool IsInvalid => handle == IntPtr.Zero;

    private RasterizationTable(bool clear) : base(0, true)
    {
        uint precomputedRasterTablesByteCount = OFFSET_QUANTIZATION_FACTOR * SLOPE_QUANTIZATION_FACTOR * sizeof(ulong);
        ulong* precomputedRasterTables = (ulong*)NativeMemory.AlignedAlloc(
            byteCount: precomputedRasterTablesByteCount,
            alignment: 256 / 8);

        if (clear)
        {
            NativeMemory.Clear(precomputedRasterTables, precomputedRasterTablesByteCount);
        }

        handle = (IntPtr)precomputedRasterTables;
    }

    protected override bool ReleaseHandle()
    {
        NativeMemory.AlignedFree((void*)handle);
        return true;
    }
}