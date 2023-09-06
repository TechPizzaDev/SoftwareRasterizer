using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

public static unsafe class VectorMath
{
    // Cross product
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> cross(Vector128<float> a, Vector128<float> b)
    {
        Vector128<float> a_yzx = Sse.Shuffle(a, a, 0b11_00_10_01);
        Vector128<float> b_yzx = Sse.Shuffle(b, b, 0b11_00_10_01);
        Vector128<float> c = Sse.Subtract(Sse.Multiply(a, b_yzx), Sse.Multiply(a_yzx, b));
        return Sse.Shuffle(c, c, 0b11_00_10_01);
    }

    // Normal vector of triangle
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> normal(Vector128<float> v0, Vector128<float> v1, Vector128<float> v2)
    {
        return cross(Sse.Subtract(v1, v0), Sse.Subtract(v2, v0));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> normalize(Vector128<float> v)
    {
        return Sse.Multiply(v, Sse.ReciprocalSqrt(Sse41.DotProduct(v, v, 0x7F)));
    }

    // Normal vector of triangle
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 normal(Vector4 v0, Vector4 v1, Vector4 v2)
    {
        return cross((v1 - v0).AsVector128(), (v2 - v0).AsVector128()).AsVector4();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 normalize(Vector4 v)
    {
        return Sse.Multiply(v.AsVector128(), Sse.ReciprocalSqrt(Sse41.DotProduct(v.AsVector128(), v.AsVector128(), 0x7F))).AsVector4();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void _MM_TRANSPOSE4_PS(
        ref Vector4 row0, ref Vector4 row1, ref Vector4 row2, ref Vector4 row3)
    {
        Vector128<float> _Tmp0 = Sse.Shuffle(row0.AsVector128(), row1.AsVector128(), 0x44);
        Vector128<float> _Tmp2 = Sse.Shuffle(row0.AsVector128(), row1.AsVector128(), 0xEE);
        Vector128<float> _Tmp1 = Sse.Shuffle(row2.AsVector128(), row3.AsVector128(), 0x44);
        Vector128<float> _Tmp3 = Sse.Shuffle(row2.AsVector128(), row3.AsVector128(), 0xEE);

        row0 = Sse.Shuffle(_Tmp0, _Tmp1, 0x88).AsVector4();
        row1 = Sse.Shuffle(_Tmp0, _Tmp1, 0xDD).AsVector4();
        row2 = Sse.Shuffle(_Tmp2, _Tmp3, 0x88).AsVector4();
        row3 = Sse.Shuffle(_Tmp2, _Tmp3, 0xDD).AsVector4();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void _MM_TRANSPOSE4_PS(
        ref Vector128<float> row0, ref Vector128<float> row1, ref Vector128<float> row2, ref Vector128<float> row3)
    {
        Vector128<float> _Tmp0 = Sse.Shuffle(row0, row1, 0x44);
        Vector128<float> _Tmp2 = Sse.Shuffle(row0, row1, 0xEE);
        Vector128<float> _Tmp1 = Sse.Shuffle(row2, row3, 0x44);
        Vector128<float> _Tmp3 = Sse.Shuffle(row2, row3, 0xEE);

        row0 = Sse.Shuffle(_Tmp0, _Tmp1, 0x88);
        row1 = Sse.Shuffle(_Tmp0, _Tmp1, 0xDD);
        row2 = Sse.Shuffle(_Tmp2, _Tmp3, 0x88);
        row3 = Sse.Shuffle(_Tmp2, _Tmp3, 0xDD);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix4x4 XMMatrixPerspectiveFovLH(
        float FovAngleY,
        float AspectRatio,
        float NearZ,
        float FarZ)
    {
        Debug.Assert(NearZ > 0f && FarZ > 0f);
        //Debug.Assert(!XMScalarNearEqual(FovAngleY, 0.0f, 0.00001f * 2.0f));
        //Debug.Assert(!XMScalarNearEqual(AspectRatio, 0.0f, 0.00001f));
        //Debug.Assert(!XMScalarNearEqual(FarZ, NearZ, 0.00001f));

        (float SinFov, float CosFov) = MathF.SinCos(0.5f * FovAngleY);

        float Height = CosFov / SinFov;
        float Width = Height / AspectRatio;
        float fRange = FarZ / (FarZ - NearZ);

        Matrix4x4 M;
        M.M11 = Width;
        M.M12 = 0.0f;
        M.M13 = 0.0f;
        M.M14 = 0.0f;

        M.M21 = 0.0f;
        M.M22 = Height;
        M.M23 = 0.0f;
        M.M24 = 0.0f;

        M.M31 = 0.0f;
        M.M32 = 0.0f;
        M.M33 = fRange;
        M.M34 = 1.0f;

        M.M41 = 0.0f;
        M.M42 = 0.0f;
        M.M43 = -fRange * NearZ;
        M.M44 = 0.0f;
        return M;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix4x4 XMMatrixLookToLH(
        Vector3 EyePosition,
        Vector3 EyeDirection,
        Vector3 UpDirection)
    {
        Debug.Assert(!(EyeDirection == Vector3.Zero));
        //Debug.Assert(!XMVector3IsInfinite(EyeDirection));
        Debug.Assert(!(UpDirection == Vector3.Zero));
        //Debug.Assert(!XMVector3IsInfinite(UpDirection));

        Vector3 R2 = Vector3.Normalize(EyeDirection);

        Vector3 R0 = Vector3.Cross(UpDirection, R2);
        R0 = Vector3.Normalize(R0);

        Vector3 R1 = Vector3.Cross(R2, R0);

        Vector3 NegEyePosition = Vector3.Negate(EyePosition);

        float D0 = Vector3.Dot(R0, NegEyePosition);
        float D1 = Vector3.Dot(R1, NegEyePosition);
        float D2 = Vector3.Dot(R2, NegEyePosition);

        Matrix4x4 M = new(
             R0.X, R0.Y, R0.Z, D0,
             R1.X, R1.Y, R1.Z, D1,
             R2.X, R2.Y, R2.Z, D2,
             0f, 0f, 0f, 1f);

        M = Matrix4x4.Transpose(M);

        return M;
    }
}

public struct Aabb
{
    public Aabb()
    {
        m_min = new Vector4(float.PositiveInfinity);
        m_max = new Vector4(float.NegativeInfinity);
    }

    public Vector4 m_min;
    public Vector4 m_max;

    public void include(in Aabb aabb)
    {
        m_min = Vector4.Min(m_min, aabb.m_min);
        m_max = Vector4.Max(m_max, aabb.m_max);
    }

    public void include(Vector4 point)
    {
        m_min = Vector4.Min(m_min, point);
        m_max = Vector4.Max(m_max, point);
    }

    public readonly Vector4 getCenter()
    {
        return (m_min + m_max);
    }

    public readonly Vector4 getExtents()
    {
        return (m_max - m_min);
    }

    public readonly Vector4 surfaceArea()
    {
        Vector128<float> extents = getExtents().AsVector128();
        Vector128<float> extents2 = Sse.Shuffle(extents, extents, 0b11_00_10_01);
        return Sse41.DotProduct(extents, extents2, 0x7F).AsVector4();
    }
}