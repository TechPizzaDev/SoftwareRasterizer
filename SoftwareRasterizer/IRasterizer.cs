using System;
using System.Numerics;

namespace SoftwareRasterizer;

public unsafe interface IRasterizer : IDisposable
{
    void setModelViewProjection(float* matrix);
    
    void clear();

    void rasterize<T>(in Occluder occluder)
        where T : IPossiblyNearClipped;

	bool queryVisibility(Vector4 boundsMin, Vector4 boundsMax, out bool needsClipping);

    bool query2D(uint minX, uint maxX, uint minY, uint maxY, uint maxZ);

    void readBackDepth(byte* target);
}
