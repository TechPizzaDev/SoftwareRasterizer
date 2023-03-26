using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading;

namespace SoftwareRasterizer;

public abstract unsafe partial class Rasterizer : IRasterizer
{
    public readonly struct NearClipped : IPossiblyNearClipped
    {
        public static bool PossiblyNearClipped => true;
    }

    public readonly struct NotNearClipped : IPossiblyNearClipped
    {
        public static bool PossiblyNearClipped => false;
    }

    protected internal const float floatCompressionBias = 2.5237386e-29f; // 0xFFFF << 12 reinterpreted as float
    protected internal const float minEdgeOffset = -0.45f;
    protected internal const float maxInvW = 18446742974197923840; // MathF.Sqrt(float.MaxValue)

    protected internal const int OFFSET_QUANTIZATION_BITS = 6;
    protected internal const int OFFSET_QUANTIZATION_FACTOR = 1 << OFFSET_QUANTIZATION_BITS;

    protected internal const int SLOPE_QUANTIZATION_BITS = 6;
    protected internal const int SLOPE_QUANTIZATION_FACTOR = 1 << SLOPE_QUANTIZATION_BITS;

    private int _disposed;

    protected float* m_modelViewProjection;
    protected float* m_modelViewProjectionRaw;

    protected RasterizationTable m_precomputedRasterTables;
    protected Vector128<int>* m_depthBuffer;

    protected uint m_hiZ_Size;
    protected ushort* m_hiZ;

    protected uint m_width;
    protected uint m_height;
    protected uint m_blocksX;
    protected uint m_blocksY;

    protected Rasterizer(RasterizationTable rasterizationTable, uint width, uint height, nuint alignment)
    {
        m_precomputedRasterTables = rasterizationTable;
        m_width = width;
        m_height = height;
        m_blocksX = width / 8;
        m_blocksY = height / 8;

        Debug.Assert(width % 8 == 0 && height % 8 == 0);

        m_modelViewProjection = (float*)NativeMemory.AlignedAlloc(16 * sizeof(float), alignment);
        m_modelViewProjectionRaw = (float*)NativeMemory.AlignedAlloc(16 * sizeof(float), alignment);

        m_depthBuffer = (Vector128<int>*)NativeMemory.AlignedAlloc(width * height / 8 * (uint)sizeof(Vector128<int>), alignment);

        m_hiZ_Size = m_blocksX * m_blocksY + 8; // Add some extra padding to support out-of-bounds reads
        uint hiZ_Bytes = m_hiZ_Size * sizeof(ushort);
        m_hiZ = (ushort*)NativeMemory.AlignedAlloc(hiZ_Bytes, alignment);
        Unsafe.InitBlockUnaligned(m_hiZ, 0, hiZ_Bytes);
    }

    public abstract void setModelViewProjection(float* matrix);

    public abstract void clear();

    public abstract void rasterize<T>(in Occluder occluder) where T : IPossiblyNearClipped;

    public abstract bool queryVisibility(Vector4 boundsMin, Vector4 boundsMax, out bool needsClipping);

    public abstract bool query2D(uint minX, uint maxX, uint minY, uint maxY, uint maxZ);

    public abstract void readBackDepth(byte* target);

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        int state = Interlocked.Exchange(ref _disposed, 1);
        if (state == 0)
        {
            m_precomputedRasterTables.DangerousRelease();
            m_precomputedRasterTables = null!;

            NativeMemory.AlignedFree(m_modelViewProjection);
            m_modelViewProjection = null;

            NativeMemory.AlignedFree(m_modelViewProjectionRaw);
            m_modelViewProjectionRaw = null;

            NativeMemory.AlignedFree(m_depthBuffer);
            m_depthBuffer = null;

            NativeMemory.AlignedFree(m_hiZ);
            m_hiZ = null;
        }
    }

    ~Rasterizer()
    {
        Dispose(disposing: false);
    }
}