using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SoftwareRasterizer;

public static unsafe class Algo
{
    public static void stable_sort<T, TComparer>(ref T start, ref T end, TComparer comparer)
        where T : unmanaged
        where TComparer : IComparer<T>
    {
        // TODO:
    }

    public static void sort<T, TComparer>(ref T start, ref T end, TComparer comparer)
        where TComparer : IComparer<T>
    {
        // TODO:
    }

    public static void sort<T, TComparer>(Span<T> span, TComparer comparer)
        where TComparer : IComparer<T>
    {
        ref T start = ref MemoryMarshal.GetReference(span);
        ref T end = ref Unsafe.Add(ref start, span.Length);
        sort(ref start, ref end, comparer);
    }

    public interface IComparer<T>
    {
        bool Compare(T x, T y);
    }
}