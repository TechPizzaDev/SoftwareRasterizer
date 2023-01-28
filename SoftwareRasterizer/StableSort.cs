using System;
using System.Collections.Generic;

namespace SoftwareRasterizer;

public static unsafe class StableSort
{
    public static void stable_sort<T, TComparer>(T* start, T* end, TComparer comparer)
        where T : unmanaged
        where TComparer : IComparer<T>
    {
        long length = end - start;
        new Span<T>(start, (int)length).Sort((x, y) => comparer.Compare(x, y));
    }

    //public interface IComparer<T>
    //{
    //    bool Compare(T x, T y);
    //}
}