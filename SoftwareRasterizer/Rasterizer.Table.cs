using System;

namespace SoftwareRasterizer;

using static PrimitiveMode;

public abstract partial class Rasterizer
{
    protected static ReadOnlySpan<PrimitiveMode> modeTable => new PrimitiveMode[256] 
    {
        Convex,        Triangle1,     ConcaveLeft,   Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        ConcaveRight,  Triangle1,     Culled,        Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Convex,        Triangle1,     ConcaveLeft,   Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Culled,        Triangle1,     ConcaveCenter, Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Convex,        Culled,        ConcaveLeft,   Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        ConcaveRight,  Triangle1,     ConcaveCenter, Triangle1,     Triangle0,     Culled,        Culled,        Culled,
        Convex,        Triangle1,     ConcaveLeft,   Culled,        Triangle0,     Culled,        Triangle0,     Culled,
        ConcaveRight,  Triangle1,     ConcaveCenter, Triangle1,     Culled,        Culled,        Triangle0,     Culled,
        Convex,        Triangle1,     Culled,        Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        ConcaveRight,  Triangle1,     ConcaveCenter, Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Culled,        Triangle1,     ConcaveCenter, Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        ConcaveCenter, Triangle1,     ConcaveCenter, Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Convex,        Triangle1,     ConcaveLeft,   Triangle1,     Triangle0,     Culled,        Culled,        Culled,
        ConcaveRight,  Culled,        ConcaveCenter, Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Triangle1,     Triangle1,     Triangle1,     Triangle1,     Culled,        Culled,        Culled,        Culled,
        Triangle1,     Triangle1,     Triangle1,     Culled,        Culled,        Culled,        Culled,        Culled,
        Convex,        Triangle1,     ConcaveLeft,   Triangle1,     Culled,        Culled,        Triangle0,     Culled,
        ConcaveRight,  Triangle1,     ConcaveCenter, Culled,        Triangle0,     Culled,        Triangle0,     Culled,
        Convex,        Triangle1,     ConcaveLeft,   Triangle1,     Triangle0,     Culled,        Culled,        Culled,
        ConcaveRight,  Culled,        ConcaveCenter, Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Culled,        Triangle1,     ConcaveLeft,   Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        ConcaveRight,  Triangle1,     ConcaveCenter, Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        ConcaveRight,  Triangle1,     Culled,        Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        ConcaveRight,  Triangle1,     ConcaveCenter, Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Convex,        Triangle1,     ConcaveLeft,   Culled,        Triangle0,     Culled,        Triangle0,     Culled,
        ConcaveRight,  Triangle1,     ConcaveCenter, Triangle1,     Culled,        Culled,        Triangle0,     Culled,
        Triangle0,     Culled,        Triangle0,     Culled,        Triangle0,     Culled,        Triangle0,     Culled,
        Triangle0,     Culled,        Triangle0,     Culled,        Triangle0,     Culled,        Culled,        Culled,
        ConcaveLeft,   Triangle1,     ConcaveLeft,   Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Culled,        Triangle1,     ConcaveCenter, Triangle1,     Triangle0,     Culled,        Triangle0,     Culled,
        Culled,        Culled,        Culled,        Culled,        Culled,        Culled,        Culled,        Culled,
        Culled,        Culled,        Culled,        Culled,        Culled,        Culled,        Culled,        Culled,
    };
}