using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

public static class V128Helper
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<ushort> Average(Vector128<ushort> left, Vector128<ushort> right)
    {
        if (Sse2.IsSupported)
        {
            return Sse2.Average(left, right);
        }
        else if (AdvSimd.IsSupported)
        {
            return AdvSimd.FusedAddRoundedHalving(left, right);
        }
        else
        {
            (Vector128<uint> lo_l, Vector128<uint> hi_l) = Vector128.Widen(left);
            (Vector128<uint> lo_r, Vector128<uint> hi_r) = Vector128.Widen(right);
            Vector128<uint> lo = Vector128.ShiftRightLogical(Vector128.Add(lo_l, lo_r), 1);
            Vector128<uint> hi = Vector128.ShiftRightLogical(Vector128.Add(hi_l, hi_r), 1);
            return Vector128.Narrow(lo, hi);
        }
    }

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
    public static ushort MinHorizontal(Vector128<ushort> value)
    {
        // The rasterizers don't need the position of the min value, which simplifies the fallbacks.

        if (Sse41.IsSupported)
        {
            return Sse41.MinHorizontal(value).ToScalar();
        }
        else if (AdvSimd.Arm64.IsSupported)
        {
            return AdvSimd.Arm64.MinAcross(value).ToScalar();
        }
        else
        {
            return SoftwareFallback(value);
        }

        static ushort SoftwareFallback(Vector128<ushort> value)
        {
            ushort result = ushort.MaxValue;
            for (int i = 0; i < Vector128<ushort>.Count; i++)
            {
                result = ushort.Min(result, value.GetElement(i));
            }
            return result;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<ushort> PackUnsignedSaturate(Vector128<int> left, Vector128<int> right)
    {
        if (Sse41.IsSupported)
        {
            return Sse41.PackUnsignedSaturate(left, right);
        }
        else if (AdvSimd.IsSupported)
        {
            return Vector128.Create(
                AdvSimd.ExtractNarrowingSaturateUnsignedLower(left),
                AdvSimd.ExtractNarrowingSaturateUnsignedLower(right));
        }
        else
        {
            Vector128<int> max = Vector128.Create((int)ushort.MaxValue);
            Vector128<int> satLeft = Vector128.Max(Vector128<int>.Zero, Vector128.Min(left, max));
            Vector128<int> satRight = Vector128.Max(Vector128<int>.Zero, Vector128.Min(right, max));
            return Vector128.Narrow(satLeft.AsUInt32(), satRight.AsUInt32());
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> PermuteFrom0(Vector128<float> value)
    {
        return Vector128.Create(value.ToScalar());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> PermuteFrom1(Vector128<float> value)
    {
        return Vector128.Shuffle(value, Vector128.Create(1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> PermuteFrom2(Vector128<float> value)
    {
        return Vector128.Shuffle(value, Vector128.Create(2));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> PermuteFrom3(Vector128<float> value)
    {
        return Vector128.Shuffle(value, Vector128.Create(3));
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Reciprocal(Vector128<float> value)
    {
        if (Sse.IsSupported)
        {
            return Sse.Reciprocal(value);
        }
        else if (AdvSimd.IsSupported)
        {
            return ReciprocalAdvSimd(value);
        }
        else
        {
            return Vector128.Divide(Vector128.Create(1f), value);
        }
    }

    /// <summary>   
    /// <seealso href="https://github.com/DLTcollab/sse2neon/blob/fb160a53e5a4ba5bc21e1a7cb80d0bd390812442/sse2neon.h#L2292" />
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ReciprocalAdvSimd(Vector128<float> value)
    {
        Vector128<float> recip = AdvSimd.ReciprocalEstimate(value);
        recip = AdvSimd.Multiply(recip, AdvSimd.ReciprocalStep(recip, value));
        return recip;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool TestZ(Vector128<float> left, Vector128<float> right)
    {
        if (Avx.IsSupported)
        {
            return Avx.TestZ(left, right);
        }
        else
        {
            Vector128<float> mask = Vector128.Create(0x80000000u).AsSingle();
            Vector128<float> tmp = (left & right) & mask;
            return Vector128.EqualsAll(tmp.AsUInt32(), Vector128<uint>.Zero);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> UnpackHigh(Vector128<float> left, Vector128<float> right)
    {
        if (Sse.IsSupported)
        {
            return Sse.UnpackHigh(left, right);
        }
        else if (AdvSimd.Arm64.IsSupported)
        {
            return AdvSimd.Arm64.ZipHigh(left, right);
        }
        else
        {
            return SoftwareFallback(left, right);
        }

        static Vector128<float> SoftwareFallback(Vector128<float> left, Vector128<float> right)
        {
            return Vector128.Create(
                left.GetElement(2),
                right.GetElement(2),
                left.GetElement(3),
                right.GetElement(3));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> UnpackLow(Vector128<float> left, Vector128<float> right)
    {
        if (Sse.IsSupported)
        {
            return Sse.UnpackLow(left, right);
        }
        else if (AdvSimd.Arm64.IsSupported)
        {
            return AdvSimd.Arm64.ZipLow(left, right);
        }
        else
        {
            return SoftwareFallback(left, right);
        }

        static Vector128<float> SoftwareFallback(Vector128<float> left, Vector128<float> right)
        {
            return Vector128.Create(
                left.GetElement(0),
                right.GetElement(0),
                left.GetElement(1),
                right.GetElement(1));
        }
    }
}
