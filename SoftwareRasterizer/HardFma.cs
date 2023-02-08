using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

public readonly struct HardFma : IFusedMultiplyAdd
{
    public static bool IsSupported => Fma.IsSupported;

    public static Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return Fma.MultiplyAdd(a, b, c);
    }

    public static Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        return Fma.MultiplyAdd(a, b, c);
    }

    public static Vector128<float> MultiplyAddNegated(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return Fma.MultiplyAddNegated(a, b, c);
    }

    public static Vector256<float> MultiplyAddNegated(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        return Fma.MultiplyAddNegated(a, b, c);
    }

    public static Vector128<float> MultiplySubtract(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        return Fma.MultiplySubtract(a, b, c);
    }

    public static Vector256<float> MultiplySubtract(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        return Fma.MultiplySubtract(a, b, c);
    }
}
