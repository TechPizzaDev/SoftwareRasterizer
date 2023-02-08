using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

public interface IFusedMultiplyAdd
{
    static abstract bool IsSupported { get; }

    static abstract Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c);

    static abstract Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c);

    static abstract Vector128<float> MultiplyAddNegated(Vector128<float> a, Vector128<float> b, Vector128<float> c);

    static abstract Vector256<float> MultiplyAddNegated(Vector256<float> a, Vector256<float> b, Vector256<float> c);

    static abstract Vector128<float> MultiplySubtract(Vector128<float> a, Vector128<float> b, Vector128<float> c);

    static abstract Vector256<float> MultiplySubtract(Vector256<float> a, Vector256<float> b, Vector256<float> c);
}