﻿using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

public readonly struct FmaIntrinsic : IFusedMultiplyAdd128, IFusedMultiplyAdd256
{
    public static bool IsHardwareAccelerated128 => Vector128.IsHardwareAccelerated;

    public static bool IsHardwareAccelerated256 => Vector256.IsHardwareAccelerated;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return ((a * b) + c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        return ((a * b) + c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MultiplyAddNegated(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return (c - (a * b));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MultiplyAddNegated(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        return (c - (a * b));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MultiplySubtract(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return ((a * b) - c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MultiplySubtract(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        return ((a * b) - c);
    }
}