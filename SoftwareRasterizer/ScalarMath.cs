using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer
{
    internal static unsafe class ScalarMath
    {
        private const uint NegZeroVal = 2147483648u; // -0.0f bits

        public static void _MM_TRANSPOSE4_PS(
            ref Vector4 row0, ref Vector4 row1, ref Vector4 row2, ref Vector4 row3)
        {
            float t0_0 = row0.X;
            row0.X = t0_0;

            float t0_1 = row0.Y;
            float t0_2 = row1.X;
            row0.Y = t0_2;
            row1.X = t0_1;

            float t0_3 = row1.Y;
            row1.Y = t0_3;

            float t2_0 = row0.Z;
            float t1_0 = row2.X;
            row0.Z = t1_0;
            row2.X = t2_0;

            float t2_2 = row1.Z;
            float t1_1 = row2.Y;
            row1.Z = t1_1;
            row2.Y = t2_2;

            float t3_0 = row2.Z;
            row2.Z = t3_0;

            float t3_2 = row3.Z;
            float t3_1 = row2.W;
            row3.Z = t3_1;

            float t3_3 = row3.W;
            row3.W = t3_3;

            float t2_1 = row0.W;
            float t1_2 = row3.X;
            row0.W = t1_2;
            row3.X = t2_1;

            float t2_3 = row1.W;
            float t1_3 = row3.Y;
            row1.W = t1_3;
            row3.Y = t2_3;

            row2.W = t3_2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProduct_x7F(Vector4 left, Vector4 right)
        {
            Vector3 a = left.AsVector128().AsVector3();
            Vector3 b = right.AsVector128().AsVector3();
            return Vector3.Dot(a, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProduct_x7F(Vector4 left)
        {
            Vector3 v = left.AsVector128().AsVector3();
            return Vector3.Dot(v, v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 And(Vector4 left, Vector4 right)
        {
            UInt128 res = Unsafe.As<Vector4, UInt128>(ref left) & Unsafe.As<Vector4, UInt128>(ref right);
            return Unsafe.As<UInt128, Vector4>(ref res);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 AndNot(Vector4 left, Vector4 right)
        {
            UInt128 res = (~Unsafe.As<Vector4, UInt128>(ref left)) & Unsafe.As<Vector4, UInt128>(ref right);
            return Unsafe.As<UInt128, Vector4>(ref res);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 NotZeroAnd(Vector4 right)
        {
            // pack 4x -0.0f bits
            UInt128 left = new(
                ~(((ulong)NegZeroVal << 32) | NegZeroVal),
                ~(((ulong)NegZeroVal << 32) | NegZeroVal));

            UInt128 res = left & Unsafe.As<Vector4, UInt128>(ref right);
            return Unsafe.As<UInt128, Vector4>(ref res);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NotZeroAnd(float right)
        {
            uint res = (~NegZeroVal) & BitConverter.SingleToUInt32Bits(right);
            return BitConverter.UInt32BitsToSingle(res);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 UnpackLow(Vector4 a, Vector4 b)
        {
            return Sse.UnpackLow(a.AsVector128(), b.AsVector128()).AsVector4();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 UnpackHigh(Vector4 a, Vector4 b)
        {
            return Sse.UnpackHigh(a.AsVector128(), b.AsVector128()).AsVector4();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 Floor(Vector4 a)
        {
            return new Vector4(
                MathF.Floor(a.X),
                MathF.Floor(a.Y),
                MathF.Floor(a.Z),
                MathF.Floor(a.W));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 Reciprocal(Vector4 a)
        {
            return new Vector4(
                MathF.ReciprocalEstimate(a.X),
                MathF.ReciprocalEstimate(a.Y),
                MathF.ReciprocalEstimate(a.Z),
                MathF.ReciprocalEstimate(a.W));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 RoundToNearestInteger(Vector4 a)
        {
            return new Vector4(
                MathF.Round(a.X),
                MathF.Round(a.Y),
                MathF.Round(a.Z),
                MathF.Round(a.W));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 Xor(Vector4 a, Vector4 b)
        {
            return (a.AsVector128() ^ b.AsVector128()).AsVector4();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint MoveMask(Vector4 a)
        {
            return a.AsVector128().ExtractMostSignificantBits();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4I CompareLessThanOrEqual(Vector4 a, Vector4 b)
        {
            unchecked
            {
                return new Vector4I(
                    a.X <= b.X ? (int)0xFFFFFFFF : 0,
                    a.Y <= b.Y ? (int)0xFFFFFFFF : 0,
                    a.Z <= b.Z ? (int)0xFFFFFFFF : 0,
                    a.W <= b.W ? (int)0xFFFFFFFF : 0);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4I CompareLessThan(Vector4 a, Vector4 b)
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
        public static Vector4I CompareGreaterThanOrEqual(Vector4 a, Vector4 b)
        {
            unchecked
            {
                return new Vector4I(
                    a.X >= b.X ? (int)0xFFFFFFFF : 0,
                    a.Y >= b.Y ? (int)0xFFFFFFFF : 0,
                    a.Z >= b.Z ? (int)0xFFFFFFFF : 0,
                    a.W >= b.W ? (int)0xFFFFFFFF : 0);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4I CompareGreaterThan(Vector4 a, Vector4 b)
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
        public static Vector4I BlendVariable(Vector4I a, Vector4I b, Vector4I c)
        {
            return (Vector4I)Sse41.BlendVariable(a, b, c);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 Permute(Vector4 a, byte imm8)
        {
            return Sse.Shuffle(a.AsVector128(), a.AsVector128(), imm8).AsVector4();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 BlendVariable(Vector4 a, Vector4 b, Vector4I c)
        {
            return Sse41.BlendVariable(a.AsVector128(), b.AsVector128(), c.AsSingle().AsVector128()).AsVector4();
        }
    }
}
