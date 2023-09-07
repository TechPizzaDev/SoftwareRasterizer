using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

public interface IFusedMultiplyAdd256
{
    static abstract bool IsHardwareAccelerated256 { get; }

    static abstract Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c);

    static abstract Vector256<float> MultiplyAddNegated(Vector256<float> a, Vector256<float> b, Vector256<float> c);

    static abstract Vector256<float> MultiplySubtract(Vector256<float> a, Vector256<float> b, Vector256<float> c);
}