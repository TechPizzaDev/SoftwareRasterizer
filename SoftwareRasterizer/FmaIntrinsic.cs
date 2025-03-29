using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

public readonly struct FmaIntrinsic : IMultiplyAdd, IMultiplyAdd128, IMultiplyAdd256
{
    public static bool IsAcceleratedScalar => true;

    public static bool IsAccelerated128 => Vector128.IsHardwareAccelerated;

    public static bool IsAccelerated256 => Vector256.IsHardwareAccelerated;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float MulAdd(float a, float b, float c) => (a * b) + c;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MulAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c) => (a * b) + c;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MulAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c) => (a * b) + c;
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float MulAddNeg(float a, float b, float c) => c - (a * b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MulAddNeg(Vector128<float> a, Vector128<float> b, Vector128<float> c) => c - (a * b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MulAddNeg(Vector256<float> a, Vector256<float> b, Vector256<float> c) => c - (a * b);
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float MulSub(float a, float b, float c) => (a * b) - c;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MulSub(Vector128<float> a, Vector128<float> b, Vector128<float> c) => (a * b) - c;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MulSub(Vector256<float> a, Vector256<float> b, Vector256<float> c) => (a * b) - c;
}