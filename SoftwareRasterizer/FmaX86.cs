using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

public readonly struct FmaX86 : IFusedMultiplyAdd128, IFusedMultiplyAdd256
{
    public static bool IsHardwareAccelerated128 => Fma.IsSupported;

    public static bool IsHardwareAccelerated256 => Fma.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return Fma.MultiplyAdd(a, b, c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        return Fma.MultiplyAdd(a, b, c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MultiplyAddNegated(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return Fma.MultiplyAddNegated(a, b, c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MultiplyAddNegated(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        return Fma.MultiplyAddNegated(a, b, c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MultiplySubtract(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return Fma.MultiplySubtract(a, b, c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MultiplySubtract(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        return Fma.MultiplySubtract(a, b, c);
    }
}
