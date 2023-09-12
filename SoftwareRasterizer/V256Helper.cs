using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

public static class V256Helper
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<ushort> Average(Vector256<ushort> left, Vector256<ushort> right)
    {
        if (Avx2.IsSupported)
        {
            return Avx2.Average(left, right);
        }
        else
        {
            (Vector256<uint> l_lo, Vector256<uint> l_hi) = Vector256.Widen(left);
            (Vector256<uint> r_lo, Vector256<uint> r_hi) = Vector256.Widen(right);
            Vector256<uint> lo = Vector256.ShiftRightLogical(Vector256.Add(l_lo, r_lo), 1);
            Vector256<uint> hi = Vector256.ShiftRightLogical(Vector256.Add(l_hi, r_hi), 1);
            return Vector256.Narrow(lo, hi);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> BlendVariable(Vector256<float> left, Vector256<float> right, Vector256<float> mask)
    {
        if (Avx.IsSupported)
        {
            return Avx.BlendVariable(left, right, mask);
        }
        else
        {
            Vector256<float> c = Vector256.ShiftRightArithmetic(mask.AsInt32(), 31).AsSingle();
            return Vector256.ConditionalSelect(c, right, left);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<byte> BlendVariable(Vector256<byte> left, Vector256<byte> right, Vector256<byte> mask)
    {
        if (Avx2.IsSupported)
        {
            return Avx2.BlendVariable(left, right, mask);
        }
        else
        {
            Vector256<byte> c = Vector256.ShiftRightArithmetic(mask.AsSByte(), 7).AsByte();
            return Vector256.ConditionalSelect(c, right, left);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> ConvertToInt32(Vector128<ushort> value)
    {
        if (Avx2.IsSupported)
        {
            return Avx2.ConvertToVector256Int32(value);
        }
        else
        {
            (Vector128<uint> lo, Vector128<uint> hi) = Vector128.Widen(value);
            return Vector256.Create(lo, hi).AsInt32();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Vector256<int> GatherBy4(ref int baseAddress, Vector256<int> indices)
    {
        if (Avx2.IsSupported)
        {
            fixed (int* ptr = &baseAddress)
            {
                return Avx2.GatherVector256(ptr, indices, 4);
            }
        }
        else
        {
            return SoftwareFallback(ref baseAddress, indices);
        }

        static Vector256<int> SoftwareFallback(ref int baseAddress, Vector256<int> indices)
        {
            Unsafe.SkipInit(out Vector256<int> result);

            for (int i = 0; i < Vector256<int>.Count; i++)
            {
                result = result.WithElement(i, Unsafe.Add(ref baseAddress, indices.GetElement(i)));
            }

            return result;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<byte> PackUnsignedSaturate(Vector256<short> left, Vector256<short> right)
    {
        if (Avx2.IsSupported)
        {
            return Avx2.PackUnsignedSaturate(left, right);
        }
        else
        {
            return SoftwareFallback(left, right);
        }

        static Vector256<byte> SoftwareFallback(Vector256<short> left, Vector256<short> right)
        {
            Unsafe.SkipInit(out Vector256<byte> result);

            for (int i = 0; i < 8; i++)
            {
                result = result.WithElement(i + 0, byte.CreateSaturating(left.GetElement(i + 0)));
                result = result.WithElement(i + 8, byte.CreateSaturating(right.GetElement(i + 0)));
                result = result.WithElement(i + 16, byte.CreateSaturating(left.GetElement(i + 8)));
                result = result.WithElement(i + 24, byte.CreateSaturating(right.GetElement(i + 8)));
            }

            return result;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<ushort> PackUnsignedSaturate(Vector256<int> left, Vector256<int> right)
    {
        if (Avx2.IsSupported)
        {
            return Avx2.PackUnsignedSaturate(left, right);
        }
        else
        {
            return SoftwareFallback(left, right);
        }

        static Vector256<ushort> SoftwareFallback(Vector256<int> left, Vector256<int> right)
        {
            Unsafe.SkipInit(out Vector256<ushort> result);

            for (int i = 0; i < 4; i++)
            {
                result = result.WithElement(i + 0, ushort.CreateSaturating(left.GetElement(i + 0)));
                result = result.WithElement(i + 4, ushort.CreateSaturating(right.GetElement(i + 0)));
                result = result.WithElement(i + 8, ushort.CreateSaturating(left.GetElement(i + 4)));
                result = result.WithElement(i + 12, ushort.CreateSaturating(right.GetElement(i + 4)));
            }

            return result;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Reciprocal(Vector256<float> value)
    {
        if (Avx.IsSupported)
        {
            return Avx.Reciprocal(value);
        }
        else
        {
            return Vector256.Divide(Vector256.Create(1f), value);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool TestZ(Vector256<int> left, Vector256<int> right)
    {
        return Avx.TestZ(left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<byte> UnpackHigh(Vector256<byte> left, Vector256<byte> right)
    {
        return Avx2.UnpackHigh(left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<ushort> UnpackHigh(Vector256<ushort> left, Vector256<ushort> right)
    {
        return Avx2.UnpackHigh(left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> UnpackHigh(Vector256<int> left, Vector256<int> right)
    {
        return Avx2.UnpackHigh(left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<long> UnpackHigh(Vector256<long> left, Vector256<long> right)
    {
        return Avx2.UnpackHigh(left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<byte> UnpackLow(Vector256<byte> left, Vector256<byte> right)
    {
        return Avx2.UnpackLow(left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<ushort> UnpackLow(Vector256<ushort> left, Vector256<ushort> right)
    {
        return Avx2.UnpackLow(left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> UnpackLow(Vector256<int> left, Vector256<int> right)
    {
        return Avx2.UnpackLow(left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<long> UnpackLow(Vector256<long> left, Vector256<long> right)
    {
        return Avx2.UnpackLow(left, right);
    }
}
