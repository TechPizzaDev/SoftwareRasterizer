using System;
using System.Runtime.InteropServices;

namespace SoftwareRasterizer;

public static class RefExtensions
{
    public static Ref<T> AsRef<T>(this T[] array)
    {
        return new Ref<T>(ref MemoryMarshal.GetArrayDataReference(array));
    }

    public static Ref<T> AsRef<T>(this Span<T> span)
    {
        return new Ref<T>(ref MemoryMarshal.GetReference(span));
    }
}
