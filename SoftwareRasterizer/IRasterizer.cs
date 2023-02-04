using System;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

public unsafe interface IRasterizer : IDisposable
{
    void setModelViewProjection(float* matrix);
    
    void clear();

    void rasterize<T>(in Occluder occluder)
        where T : IPossiblyNearClipped;

	bool queryVisibility(Vector128<float> boundsMin, Vector128<float> boundsMax, out bool needsClipping);

    bool query2D(uint minX, uint maxX, uint minY, uint maxY, uint maxZ);

    void readBackDepth(byte* target);
}
