using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading;

namespace SoftwareRasterizer;

using static VectorMath;

public unsafe struct Occluder : IDisposable
{
    private IntPtr _vertexData;

    public Vector4 m_center;

    public Vector4 m_refMin;
    public Vector4 m_refMax;

    public Vector4 m_boundsMin;
    public Vector4 m_boundsMax;

    public Vector256<int>* m_vertexData => (Vector256<int>*)_vertexData;
    public uint m_packetCount;

    public static Occluder Bake(ReadOnlySpan<Vector4> vertices, Vector4 refMin, Vector4 refMax)
    {
        Debug.Assert(vertices.Length % 16 == 0);

        // Simple k-means clustering by normal direction to improve backface culling efficiency
        uint quadNormalsLength = (uint)vertices.Length / 4;
        Vector4* quadNormals = (Vector4*)NativeMemory.AlignedAlloc(
            byteCount: quadNormalsLength * (uint)sizeof(Vector4),
            alignment: (uint)sizeof(Vector4));
        for (int i = 0; i < vertices.Length; i += 4)
        {
            Vector4 v0 = vertices[i + 0];
            Vector4 v1 = vertices[i + 1];
            Vector4 v2 = vertices[i + 2];
            Vector4 v3 = vertices[i + 3];

            quadNormals[(uint)i / 4] = normalize((normal(v0, v1, v2) + normal(v0, v2, v3)));
        }

        const int centroidsLength = 6;
        Vector4* centroids = stackalloc Vector4[centroidsLength];
        centroids[0] = new Vector4(+1.0f, 0.0f, 0.0f, 0.0f);
        centroids[1] = new Vector4(0.0f, +1.0f, 0.0f, 0.0f);
        centroids[2] = new Vector4(0.0f, 0.0f, +1.0f, 0.0f);
        centroids[3] = new Vector4(0.0f, -1.0f, 0.0f, 0.0f);
        centroids[4] = new Vector4(0.0f, 0.0f, -1.0f, 0.0f);
        centroids[5] = new Vector4(-1.0f, 0.0f, 0.0f, 0.0f);

        uint* centroidAssignment = (uint*)NativeMemory.Alloc(quadNormalsLength, sizeof(uint));

        bool anyChanged = true;
        for (int iter = 0; iter < 10 && anyChanged; ++iter)
        {
            anyChanged = false;

            for (int j = 0; j < quadNormalsLength; ++j)
            {
                Vector4 normal = quadNormals[j];

                float bestDistance = float.NegativeInfinity;
                uint bestCentroid = 0;
                for (int k = 0; k < centroidsLength; ++k)
                {
                    float distance = ScalarMath.DotProduct_x7F(centroids[k], normal);
                    if (distance >= bestDistance)
                    {
                        bestDistance = distance;
                        bestCentroid = (uint)k;
                    }
                }

                if (centroidAssignment[j] != bestCentroid)
                {
                    centroidAssignment[j] = bestCentroid;
                    anyChanged = true;
                }
            }

            for (int k = 0; k < centroidsLength; ++k)
            {
                centroids[k] = Vector4.Zero;
            }

            for (int j = 0; j < quadNormalsLength; ++j)
            {
                int k = (int)centroidAssignment[j];

                centroids[k] = (centroids[k] + quadNormals[j]);
            }

            for (int k = 0; k < centroidsLength; ++k)
            {
                centroids[k] = normalize(centroids[k]);
            }
        }
        NativeMemory.AlignedFree(quadNormals);

        List<Vector4> orderedVertexList = new();
        for (uint k = 0; k < centroidsLength; ++k)
        {
            for (int j = 0; j < quadNormalsLength; ++j)
            {
                if (centroidAssignment[j] == k)
                {
                    orderedVertexList.Add(vertices[4 * j + 0]);
                    orderedVertexList.Add(vertices[4 * j + 1]);
                    orderedVertexList.Add(vertices[4 * j + 2]);
                    orderedVertexList.Add(vertices[4 * j + 3]);
                }
            }
        }
        NativeMemory.Free(centroidAssignment);

        Span<Vector4> orderedVertices = CollectionsMarshal.AsSpan(orderedVertexList);

        Vector4 invExtents = (new Vector4(1.0f) / (refMax - refMin));

        Vector4 scalingX = new(2047.0f);
        Vector4 scalingY = new(2047.0f);
        Vector4 scalingZ = new(1023.0f);

        Vector4 half = new(0.5f);

        uint packetCount = 0;
        Vector256<int>* vertexData = (Vector256<int>*)NativeMemory.AlignedAlloc((uint)orderedVertices.Length * 4, 32);

        for (int i = 0; i < orderedVertices.Length; i += 32)
        {
            for (int j = 0; j < 4; ++j)
            {
                // Transform into [0,1] space relative to bounding box
                Vector4 v0 = ((orderedVertices[i + j + 0] - refMin) * invExtents);
                Vector4 v1 = ((orderedVertices[i + j + 4] - refMin) * invExtents);
                Vector4 v2 = ((orderedVertices[i + j + 8] - refMin) * invExtents);
                Vector4 v3 = ((orderedVertices[i + j + 12] - refMin) * invExtents);
                Vector4 v4 = ((orderedVertices[i + j + 16] - refMin) * invExtents);
                Vector4 v5 = ((orderedVertices[i + j + 20] - refMin) * invExtents);
                Vector4 v6 = ((orderedVertices[i + j + 24] - refMin) * invExtents);
                Vector4 v7 = ((orderedVertices[i + j + 28] - refMin) * invExtents);

                // Transpose into [xxxx][yyyy][zzzz][wwww]
                _MM_TRANSPOSE4_PS(ref v0, ref v1, ref v2, ref v3);
                _MM_TRANSPOSE4_PS(ref v4, ref v5, ref v6, ref v7);

                // Scale and truncate to int
                v0 = ((v0 * scalingX) + half);
                v1 = ((v1 * scalingY) + half);
                v2 = ((v2 * scalingZ) + half);

                v4 = ((v4 * scalingX) + half);
                v5 = ((v5 * scalingY) + half);
                v6 = ((v6 * scalingZ) + half);

                Vector4I X0 = Vector4I.ConvertWithTruncation(v0) - new Vector4I(1024);
                Vector4I Y0 = Vector4I.ConvertWithTruncation(v1);
                Vector4I Z0 = Vector4I.ConvertWithTruncation(v2);

                Vector4I X1 = Vector4I.ConvertWithTruncation(v4) - new Vector4I(1024);
                Vector4I Y1 = Vector4I.ConvertWithTruncation(v5);
                Vector4I Z1 = Vector4I.ConvertWithTruncation(v6);

                // Pack to 11/11/10 format
                Vector4I XYZ0 = (X0 << 21) | ((Y0 << 10) | Z0);
                Vector4I XYZ1 = (X1 << 21) | ((Y1 << 10) | Z1);

                Unsafe.Write((int*)(vertexData + packetCount) + 0, XYZ0);
                Unsafe.Write((int*)(vertexData + packetCount) + 4, XYZ1);
                packetCount++;
            }
        }

        Vector4 min = new(float.PositiveInfinity);
        Vector4 max = new(float.NegativeInfinity);

        for (int i = 0; i < orderedVertices.Length; ++i)
        {
            min = Vector4.Min(vertices[i], min);
            max = Vector4.Max(vertices[i], max);
        }

        // Set W = 1 - this is expected by frustum culling code
        min.W = 1.0f;
        max.W = 1.0f;

        Occluder occluder = new()
        {
            m_packetCount = packetCount,
            _vertexData = (IntPtr)vertexData,

            m_refMin = refMin,
            m_refMax = refMax,

            m_boundsMin = min,
            m_boundsMax = max,

            m_center = ((max + min) * 0.5f)
        };

        return occluder;
    }

    public void Dispose()
    {
        IntPtr vertexData = Interlocked.Exchange(ref _vertexData, 0);
        if (vertexData != 0)
        {
            NativeMemory.AlignedFree((void*)vertexData);
        }
    }
}