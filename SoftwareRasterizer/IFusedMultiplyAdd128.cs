using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

public interface IMultiplyAdd128
{
    static abstract bool IsAccelerated128 { get; }

    static abstract Vector128<float> MulAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c);

    static abstract Vector128<float> MulAddNeg(Vector128<float> a, Vector128<float> b, Vector128<float> c);

    static abstract Vector128<float> MulSub(Vector128<float> a, Vector128<float> b, Vector128<float> c);
}