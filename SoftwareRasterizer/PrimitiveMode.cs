namespace SoftwareRasterizer;

public enum PrimitiveMode : byte
{
    Culled = 0,
    Triangle0,
    Triangle1,
    ConcaveRight,
    ConcaveLeft,
    ConcaveCenter,
    Convex
}
