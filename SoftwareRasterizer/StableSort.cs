using System;

namespace SoftwareRasterizer;

public static unsafe class StableSort
{
    public static void stable_sort<T, TComparer>(T* start, T* end, TComparer comparer)
        where T : unmanaged
        where TComparer : IComparer<T>
    {
        long length = end - start;
        new Span<T>(start, (int)length).Sort((x, y) => comparer.Compare(x, y) ? -1 : 0);
    }

    public interface IComparer<T>
    {
        bool Compare(T x, T y);
    }
}