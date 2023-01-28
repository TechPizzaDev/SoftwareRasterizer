/*
#pragma once

#include <smmintrin.h>
*/

using System.Runtime.Intrinsics;
using System.Runtime.CompilerServices;
using System.Numerics;
using System;
using System.Diagnostics;

namespace SoftwareRasterizer;

public static unsafe class VectorMath
{
    // Cross product
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> cross(Vector128<float> a, Vector128<float> b)
    {
        Vector128<float> a_yzx = _mm_shuffle_ps(a, a, 0b11_00_10_01);
        Vector128<float> b_yzx = _mm_shuffle_ps(b, b, 0b11_00_10_01);
        Vector128<float> c = _mm_sub_ps(_mm_mul_ps(a, b_yzx), _mm_mul_ps(a_yzx, b));
        return _mm_shuffle_ps(c, c, 0b11_00_10_01);
    }

    // Normal vector of triangle
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> normal(Vector128<float> v0, Vector128<float> v1, Vector128<float> v2)
    {
        return cross(_mm_sub_ps(v1, v0), _mm_sub_ps(v2, v0));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> normalize(Vector128<float> v)
    {
        return _mm_mul_ps(v, _mm_rsqrt_ps(_mm_dp_ps(v, v, 0x7F)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void _MM_TRANSPOSE4_PS(
        ref Vector128<float> row0, ref Vector128<float> row1, ref Vector128<float> row2, ref Vector128<float> row3)
    {
        Vector128<float> _Tmp0 = _mm_shuffle_ps((row0), (row1), 0x44);
        Vector128<float> _Tmp2 = _mm_shuffle_ps((row0), (row1), 0xEE);
        Vector128<float> _Tmp1 = _mm_shuffle_ps((row2), (row3), 0x44);
        Vector128<float> _Tmp3 = _mm_shuffle_ps((row2), (row3), 0xEE);

        (row0) = _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88);
        (row1) = _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD);
        (row2) = _mm_shuffle_ps(_Tmp2, _Tmp3, 0x88);
        (row3) = _mm_shuffle_ps(_Tmp2, _Tmp3, 0xDD);
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
        m_min = _mm_set1_ps(float.PositiveInfinity);
        m_max = _mm_set1_ps(float.NegativeInfinity);
    }

    public Vector128<float> m_min;
    public Vector128<float> m_max;

    public void include(in Aabb aabb)
    {
        m_min = _mm_min_ps(m_min, aabb.m_min);
        m_max = _mm_max_ps(m_max, aabb.m_max);
    }

    public void include(Vector128<float> point)
    {
        m_min = _mm_min_ps(m_min, point);
        m_max = _mm_max_ps(m_max, point);
    }

    public readonly Vector128<float> getCenter()
    {
        return _mm_add_ps(m_min, m_max);
    }

    public readonly Vector128<float> getExtents()
    {
        return _mm_sub_ps(m_max, m_min);
    }

    public readonly Vector128<float> surfaceArea()
    {
        Vector128<float> extents = getExtents();
        Vector128<float> extents2 = _mm_shuffle_ps(extents, extents, 0b11_00_10_01);
        return _mm_dp_ps(extents, extents2, 0x7F);
    }
}