using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

public interface IFusedMultiplyAdd128
{
    static abstract bool IsHardwareAccelerated128 { get; }

    static abstract Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c);

    static abstract Vector128<float> MultiplyAddNegated(Vector128<float> a, Vector128<float> b, Vector128<float> c);

    static abstract Vector128<float> MultiplySubtract(Vector128<float> a, Vector128<float> b, Vector128<float> c);
}