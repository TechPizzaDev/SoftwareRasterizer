using System.Numerics;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

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
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<float> extents = getExtents().AsVector128();
            Vector128<float> extents2 = Vector128.Shuffle(extents, Vector128.Create(1, 2, 0, 3));
            return new Vector4(Vector128.Dot(extents, extents2));
        }
        else
        {
            Vector4 extents = getExtents();
            Vector4 extents2 = new(extents.Y, extents.Z, extents.X, extents.W);
            return new Vector4(ScalarMath.DotProduct_x7F(extents, extents2));
        }
    }
}