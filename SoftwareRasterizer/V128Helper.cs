using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

public static class V128Helper
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> DotProduct_x7F(Vector128<float> a, Vector128<float> b)
    {
        if (Sse41.IsSupported)
        {
            return Sse41.DotProduct(a, b, 0x7F);
        }
        else
        {
            Vector3 a3 = a.AsVector3();
            Vector3 b3 = b.AsVector3();
            return Vector128.Create(Vector3.Dot(a3, b3));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> ReciprocalSqrt(Vector128<float> value)
    {
        if (Sse.IsSupported)
        {
            return Sse.ReciprocalSqrt(value);
        }
        else if (AdvSimd.IsSupported)
        {
            return ReciprocalSqrtAdvSimd(value);
        }
        else
        {
            return Vector128.Divide(Vector128.Create(1f), Vector128.Sqrt(value));
        }
    }

    /// <summary>   
    /// <seealso href="https://github.com/DLTcollab/sse2neon/blob/fb160a53e5a4ba5bc21e1a7cb80d0bd390812442/sse2neon.h#L2313" />
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ReciprocalSqrtAdvSimd(Vector128<float> value)
    {
        // TODO: create non-normalized overload?

        Vector128<float> r = AdvSimd.ReciprocalSquareRootEstimate(value);

        // Generate masks for detecting whether input has any 0.0f/-0.0f
        // (which becomes positive/negative infinity by IEEE-754 arithmetic rules).
        Vector128<uint> pos_inf = Vector128.Create(0x7F800000u);
        Vector128<uint> neg_inf = Vector128.Create(0xFF800000u);
        Vector128<uint> has_pos_zero = AdvSimd.CompareEqual(pos_inf, r.AsUInt32());
        Vector128<uint> has_neg_zero = AdvSimd.CompareEqual(neg_inf, r.AsUInt32());

        r = AdvSimd.Multiply(r, AdvSimd.ReciprocalSquareRootStep(AdvSimd.Multiply(value, r), r));

        // Set output vector element to infinity/negative-infinity if
        // the corresponding input vector element is 0.0f/-0.0f.
        r = AdvSimd.BitwiseSelect(has_pos_zero.AsSingle(), pos_inf.AsSingle(), r);
        r = AdvSimd.BitwiseSelect(has_neg_zero.AsSingle(), neg_inf.AsSingle(), r);

        return r;
    }
}
