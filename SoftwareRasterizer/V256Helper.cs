using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

public static class V256Helper
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Reciprocal(Vector256<float> value)
    {
        if (Avx.IsSupported)
        {
            return Avx.Reciprocal(value);
        }
        else if (AdvSimd.IsSupported)
        {
            return Vector256.Create(
                V128Helper.ReciprocalAdvSimd(value.GetLower()),
                V128Helper.ReciprocalAdvSimd(value.GetUpper()));
        }
        else
        {
            return Vector256.Divide(Vector256.Create(1f), value);
        }
    }
}
