using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

internal struct Vector4I
{
    public static Vector4I Zero => default;

    public int X;
    public int Y;
    public int Z;
    public int W;

    public Vector4I(int x, int y, int z, int w)
    {
        X = x;
        Y = y;
        Z = z;
        W = w;
    }

    public Vector4I(int value)
    {
        X = value;
        Y = value;
        Z = value;
        W = value;
    }

    public Vector4 AsSingle()
    {
        return Unsafe.As<Vector4I, Vector4>(ref this);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector4 ToSingle()
    {
        return new Vector4(X, Y, Z, W);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I Min(Vector4I a, Vector4I b)
    {
        return new Vector4I(
            Math.Min(a.X, b.X),
            Math.Min(a.Y, b.Y),
            Math.Min(a.Z, b.Z),
            Math.Min(a.W, b.W));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I Max(Vector4I a, Vector4I b)
    {
        return new Vector4I(
            Math.Max(a.X, b.X),
            Math.Max(a.Y, b.Y),
            Math.Max(a.Z, b.Z),
            Math.Max(a.W, b.W));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I ConvertWithTruncation(Vector4 value)
    {
        return new Vector4I(
            (int)value.X,
            (int)value.Y,
            (int)value.Z,
            (int)value.W);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I Convert(Vector4 value)
    {
        return new Vector4I(
            (int)MathF.Round(value.X),
            (int)MathF.Round(value.Y),
            (int)MathF.Round(value.Z),
            (int)MathF.Round(value.W));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool TestZ(Vector4I a)
    {
        UInt128 v = Unsafe.As<Vector4I, UInt128>(ref a);
        return v == 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator +(Vector4I a, Vector4I b)
    {
        return new Vector4I(
            a.X + b.X,
            a.Y + b.Y,
            a.Z + b.Z,
            a.W + b.W);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator -(Vector4I a, Vector4I b)
    {
        return new Vector4I(
            a.X - b.X,
            a.Y - b.Y,
            a.Z - b.Z,
            a.W - b.W);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator -(Vector4I a)
    {
        UInt128 res = -Unsafe.As<Vector4I, UInt128>(ref a);
        return Unsafe.As<UInt128, Vector4I>(ref res);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator ~(Vector4I a)
    {
        UInt128 res = ~Unsafe.As<Vector4I, UInt128>(ref a);
        return Unsafe.As<UInt128, Vector4I>(ref res);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator &(Vector4I a, Vector4I b)
    {
        UInt128 res = Unsafe.As<Vector4I, UInt128>(ref a) & Unsafe.As<Vector4I, UInt128>(ref b);
        return Unsafe.As<UInt128, Vector4I>(ref res);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator |(Vector4I a, Vector4I b)
    {
        UInt128 res = Unsafe.As<Vector4I, UInt128>(ref a) | Unsafe.As<Vector4I, UInt128>(ref b);
        return Unsafe.As<UInt128, Vector4I>(ref res);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator ^(Vector4I a, Vector4I b)
    {
        UInt128 res = Unsafe.As<Vector4I, UInt128>(ref a) ^ Unsafe.As<Vector4I, UInt128>(ref b);
        return Unsafe.As<UInt128, Vector4I>(ref res);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator <<(Vector4I a, int b)
    {
        return new Vector4I(
            a.X << b,
            a.Y << b,
            a.Z << b,
            a.W << b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator >>(Vector4I a, int b)
    {
        return new Vector4I(
            a.X >> b,
            a.Y >> b,
            a.Z >> b,
            a.W >> b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator >>>(Vector4I a, int b)
    {
        return new Vector4I(
            a.X >>> b,
            a.Y >>> b,
            a.Z >>> b,
            a.W >>> b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator >(Vector4I a, Vector4I b)
    {
        unchecked
        {
            return new Vector4I(
                a.X > b.X ? (int)0xFFFFFFFF : 0,
                a.Y > b.Y ? (int)0xFFFFFFFF : 0,
                a.Z > b.Z ? (int)0xFFFFFFFF : 0,
                a.W > b.W ? (int)0xFFFFFFFF : 0);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator <(Vector4I a, Vector4I b)
    {
        unchecked
        {
            return new Vector4I(
                a.X < b.X ? (int)0xFFFFFFFF : 0,
                a.Y < b.Y ? (int)0xFFFFFFFF : 0,
                a.Z < b.Z ? (int)0xFFFFFFFF : 0,
                a.W < b.W ? (int)0xFFFFFFFF : 0);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator ==(Vector4I a, Vector4I b)
    {
        unchecked
        {
            return new Vector4I(
                a.X == b.X ? (int)0xFFFFFFFF : 0,
                a.Y == b.Y ? (int)0xFFFFFFFF : 0,
                a.Z == b.Z ? (int)0xFFFFFFFF : 0,
                a.W == b.W ? (int)0xFFFFFFFF : 0);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4I operator !=(Vector4I a, Vector4I b)
    {
        unchecked
        {
            return new Vector4I(
                a.X != b.X ? (int)0xFFFFFFFF : 0,
                a.Y != b.Y ? (int)0xFFFFFFFF : 0,
                a.Z != b.Z ? (int)0xFFFFFFFF : 0,
                a.W != b.W ? (int)0xFFFFFFFF : 0);
        }
    }

    public Vector128<ushort> AsUInt16()
    {
        return Unsafe.As<Vector4I, Vector128<ushort>>(ref this);
    }

    public Vector128<short> AsInt16()
    {
        return Unsafe.As<Vector4I, Vector128<short>>(ref this);
    }

    public Vector128<byte> AsByte()
    {
        return Unsafe.As<Vector4I, Vector128<byte>>(ref this);
    }

    public static implicit operator Vector128<int>(Vector4I value)
    {
        return Unsafe.As<Vector4I, Vector128<int>>(ref value);
    }

    public static explicit operator Vector4I(Vector128<int> value)
    {
        return Unsafe.As<Vector128<int>, Vector4I>(ref value);
    }
}
