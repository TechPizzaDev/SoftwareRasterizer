using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
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
}
