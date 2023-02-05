using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;

namespace SoftwareRasterizer;

using static VectorMath;

public unsafe struct Occluder : IDisposable
{
    private nint _vertexData;

    public Vector128<float> m_center;

    public Vector128<float> m_refMin;
    public Vector128<float> m_refMax;

    public Vector128<float> m_boundsMin;
    public Vector128<float> m_boundsMax;

    public Vector256<int>* m_vertexData => (Vector256<int>*)_vertexData;
    public uint m_packetCount;

    public static Occluder bake(ReadOnlySpan<Vector128<float>> vertices, Vector128<float> refMin, Vector128<float> refMax)
    {
        Debug.Assert(vertices.Length % 16 == 0);

        // Simple k-means clustering by normal direction to improve backface culling efficiency
        uint quadNormalsLength = (uint)vertices.Length / 4;
        Vector128<float>* quadNormals = (Vector128<float>*)NativeMemory.AlignedAlloc(
            byteCount: quadNormalsLength * (uint)sizeof(Vector128<float>),
            alignment: (uint)sizeof(Vector128<float>));
        for (int i = 0; i < vertices.Length; i += 4)
        {
            Vector128<float> v0 = vertices[i + 0];
            Vector128<float> v1 = vertices[i + 1];
            Vector128<float> v2 = vertices[i + 2];
            Vector128<float> v3 = vertices[i + 3];

            quadNormals[(uint)i / 4] = normalize(Sse.Add(normal(v0, v1, v2), normal(v0, v2, v3)));
        }

        const int centroidsLength = 6;
        Vector128<float>* centroids = stackalloc Vector128<float>[centroidsLength];
        centroids[0] = Vector128.Create(+1.0f, 0.0f, 0.0f, 0.0f);
        centroids[1] = Vector128.Create(0.0f, +1.0f, 0.0f, 0.0f);
        centroids[2] = Vector128.Create(0.0f, 0.0f, +1.0f, 0.0f);
        centroids[3] = Vector128.Create(0.0f, -1.0f, 0.0f, 0.0f);
        centroids[4] = Vector128.Create(0.0f, 0.0f, -1.0f, 0.0f);
        centroids[5] = Vector128.Create(-1.0f, 0.0f, 0.0f, 0.0f);

        uint* centroidAssignment = (uint*)NativeMemory.Alloc(quadNormalsLength, sizeof(uint));

        bool anyChanged = true;
        for (int iter = 0; iter < 10 && anyChanged; ++iter)
        {
            anyChanged = false;

            for (int j = 0; j < quadNormalsLength; ++j)
            {
                Vector128<float> normal = quadNormals[j];

                Vector128<float> bestDistance = Vector128.Create(float.NegativeInfinity);
                uint bestCentroid = 0;
                for (int k = 0; k < centroidsLength; ++k)
                {
                    Vector128<float> distance = Sse41.DotProduct(centroids[k], normal, 0x7F);
                    if (Sse.CompareScalarOrderedGreaterThanOrEqual(distance, bestDistance))
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
                centroids[k] = Vector128<float>.Zero;
            }

            for (int j = 0; j < quadNormalsLength; ++j)
            {
                int k = (int)centroidAssignment[j];

                centroids[k] = Sse.Add(centroids[k], quadNormals[j]);
            }

            for (int k = 0; k < centroidsLength; ++k)
            {
                centroids[k] = normalize(centroids[k]);
            }
        }
        NativeMemory.AlignedFree(quadNormals);

        List<Vector128<float>> orderedVertexList = new();
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

        Span<Vector128<float>> orderedVertices = CollectionsMarshal.AsSpan(orderedVertexList);

        Vector128<float> invExtents = Sse.Divide(Vector128.Create(1.0f), Sse.Subtract(refMax, refMin));

        Vector128<float> scalingX = Vector128.Create(2047.0f);
        Vector128<float> scalingY = Vector128.Create(2047.0f);
        Vector128<float> scalingZ = Vector128.Create(1023.0f);

        Vector128<float> half = Vector128.Create(0.5f);

        uint packetCount = 0;
        Vector256<int>* vertexData = (Vector256<int>*)NativeMemory.AlignedAlloc((uint)orderedVertices.Length * 4, 32);

        Vector128<int>* v = stackalloc Vector128<int>[8];

        for (int i = 0; i < orderedVertices.Length; i += 32)
        {
            for (int j = 0; j < 4; ++j)
            {
                // Transform into [0,1] space relative to bounding box
                Vector128<float> v0 = Sse.Multiply(Sse.Subtract(orderedVertices[i + j + 0], refMin), invExtents);
                Vector128<float> v1 = Sse.Multiply(Sse.Subtract(orderedVertices[i + j + 4], refMin), invExtents);
                Vector128<float> v2 = Sse.Multiply(Sse.Subtract(orderedVertices[i + j + 8], refMin), invExtents);
                Vector128<float> v3 = Sse.Multiply(Sse.Subtract(orderedVertices[i + j + 12], refMin), invExtents);
                Vector128<float> v4 = Sse.Multiply(Sse.Subtract(orderedVertices[i + j + 16], refMin), invExtents);
                Vector128<float> v5 = Sse.Multiply(Sse.Subtract(orderedVertices[i + j + 20], refMin), invExtents);
                Vector128<float> v6 = Sse.Multiply(Sse.Subtract(orderedVertices[i + j + 24], refMin), invExtents);
                Vector128<float> v7 = Sse.Multiply(Sse.Subtract(orderedVertices[i + j + 28], refMin), invExtents);

                // Transpose into [xxxx][yyyy][zzzz][wwww]
                _MM_TRANSPOSE4_PS(ref v0, ref v1, ref v2, ref v3);
                _MM_TRANSPOSE4_PS(ref v4, ref v5, ref v6, ref v7);

                // Scale and truncate to int
                v0 = Fma.MultiplyAdd(v0, scalingX, half);
                v1 = Fma.MultiplyAdd(v1, scalingY, half);
                v2 = Fma.MultiplyAdd(v2, scalingZ, half);

                v4 = Fma.MultiplyAdd(v4, scalingX, half);
                v5 = Fma.MultiplyAdd(v5, scalingY, half);
                v6 = Fma.MultiplyAdd(v6, scalingZ, half);

                Vector128<int> X0 = Sse2.Subtract(Sse2.ConvertToVector128Int32WithTruncation(v0), Vector128.Create(1024));
                Vector128<int> Y0 = Sse2.ConvertToVector128Int32WithTruncation(v1);
                Vector128<int> Z0 = Sse2.ConvertToVector128Int32WithTruncation(v2);

                Vector128<int> X1 = Sse2.Subtract(Sse2.ConvertToVector128Int32WithTruncation(v4), Vector128.Create(1024));
                Vector128<int> Y1 = Sse2.ConvertToVector128Int32WithTruncation(v5);
                Vector128<int> Z1 = Sse2.ConvertToVector128Int32WithTruncation(v6);

                // Pack to 11/11/10 format
                Vector128<int> XYZ0 = Sse2.Or(Sse2.ShiftLeftLogical(X0, 21), Sse2.Or(Sse2.ShiftLeftLogical(Y0, 10), Z0));
                Vector128<int> XYZ1 = Sse2.Or(Sse2.ShiftLeftLogical(X1, 21), Sse2.Or(Sse2.ShiftLeftLogical(Y1, 10), Z1));

                v[2 * j + 0] = XYZ0;
                v[2 * j + 1] = XYZ1;
            }

            vertexData[packetCount++] = Avx.LoadVector256((int*)(v + 0));
            vertexData[packetCount++] = Avx.LoadVector256((int*)(v + 2));
            vertexData[packetCount++] = Avx.LoadVector256((int*)(v + 4));
            vertexData[packetCount++] = Avx.LoadVector256((int*)(v + 6));
        }

        Vector128<float> min = Vector128.Create(float.PositiveInfinity);
        Vector128<float> max = Vector128.Create(float.NegativeInfinity);

        for (int i = 0; i < orderedVertices.Length; ++i)
        {
            min = Sse.Min(vertices[i], min);
            max = Sse.Max(vertices[i], max);
        }

        // Set W = 1 - this is expected by frustum culling code
        min = Sse41.Blend(min, Vector128.Create(1.0f), 0b1000);
        max = Sse41.Blend(max, Vector128.Create(1.0f), 0b1000);

        Occluder occluder = new()
        {
            m_packetCount = packetCount,
            _vertexData = (nint)vertexData,

            m_refMin = refMin,
            m_refMax = refMax,

            m_boundsMin = min,
            m_boundsMax = max,

            m_center = Sse.Multiply(Sse.Add(max, min), Vector128.Create(0.5f))
        };

        return occluder;
    }

    public void Dispose()
    {
        nint vertexData = Interlocked.Exchange(ref _vertexData, 0);
        if (vertexData != 0)
        {
            NativeMemory.AlignedFree((void*)vertexData);
        }
    }
}