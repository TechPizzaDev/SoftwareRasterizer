using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

public interface IMultiplyAdd256
{
    static abstract bool IsAccelerated256 { get; }

    static abstract Vector256<float> MulAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c);

    static abstract Vector256<float> MulAddNeg(Vector256<float> a, Vector256<float> b, Vector256<float> c);

    static abstract Vector256<float> MulSub(Vector256<float> a, Vector256<float> b, Vector256<float> c);
}