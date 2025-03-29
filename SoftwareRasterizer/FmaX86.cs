using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

public readonly struct FmaX86 : IMultiplyAdd, IMultiplyAdd128, IMultiplyAdd256
{
    public static bool IsAcceleratedScalar => Fma.IsSupported;

    public static bool IsAccelerated128 => Fma.IsSupported;

    public static bool IsAccelerated256 => Fma.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float MulAdd(float a, float b, float c)
    {
        return Fma.MultiplyAddScalar(
            Vector128.CreateScalarUnsafe(a), 
            Vector128.CreateScalarUnsafe(b),
            Vector128.CreateScalarUnsafe(c)).ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MulAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c) => Fma.MultiplyAdd(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MulAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c) => Fma.MultiplyAdd(a, b, c);
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float MulAddNeg(float a, float b, float c)
    {
        return Fma.MultiplyAddNegatedScalar(
            Vector128.CreateScalarUnsafe(a), 
            Vector128.CreateScalarUnsafe(b),
            Vector128.CreateScalarUnsafe(c)).ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MulAddNeg(Vector128<float> a, Vector128<float> b, Vector128<float> c) => Fma.MultiplyAddNegated(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MulAddNeg(Vector256<float> a, Vector256<float> b, Vector256<float> c) => Fma.MultiplyAddNegated(a, b, c);
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float MulSub(float a, float b, float c)
    {
        return Fma.MultiplySubtractScalar(
            Vector128.CreateScalarUnsafe(a), 
            Vector128.CreateScalarUnsafe(b),
            Vector128.CreateScalarUnsafe(c)).ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MulSub(Vector128<float> a, Vector128<float> b, Vector128<float> c) => Fma.MultiplySubtract(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MulSub(Vector256<float> a, Vector256<float> b, Vector256<float> c) => Fma.MultiplySubtract(a, b, c);
}
