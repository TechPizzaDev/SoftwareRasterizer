using System.Numerics;
using System.Runtime.CompilerServices;

namespace SoftwareRasterizer;

internal static class Vector4Extensions
{
    public static Vector4I AsInt32(this Vector4 value)
    {
        return Unsafe.As<Vector4, Vector4I>(ref value);
    }
}