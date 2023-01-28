/*
#include "Occluder.h"

#include "VectorMath.h"

#include <cassert>
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

using static VectorMath;
using static Intrinsics;

public unsafe class Occluder
{
    public Vector128<float> m_center;

    public Vector128<float> m_refMin;
    public Vector128<float> m_refMax;

    public Vector128<float> m_boundsMin;
    public Vector128<float> m_boundsMax;

    public Vector256<int>* m_vertexData;
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

            quadNormals[(uint)i / 4] = (normalize(_mm_add_ps(normal(v0, v1, v2), normal(v0, v2, v3))));
        }
        
        const int centroidsLength = 6;
        Vector128<float>* centroids = stackalloc Vector128<float>[centroidsLength];
        centroids[0] = (_mm_setr_ps(+1.0f, 0.0f, 0.0f, 0.0f));
        centroids[1] = (_mm_setr_ps(0.0f, +1.0f, 0.0f, 0.0f));
        centroids[2] = (_mm_setr_ps(0.0f, 0.0f, +1.0f, 0.0f));
        centroids[3] = (_mm_setr_ps(0.0f, -1.0f, 0.0f, 0.0f));
        centroids[4] = (_mm_setr_ps(0.0f, 0.0f, -1.0f, 0.0f));
        centroids[5] = (_mm_setr_ps(-1.0f, 0.0f, 0.0f, 0.0f));

        uint* centroidAssignment = (uint*)NativeMemory.Alloc(quadNormalsLength, sizeof(uint));

        bool anyChanged = true;
        for (int iter = 0; iter < 10 && anyChanged; ++iter)
        {
            anyChanged = false;

            for (int j = 0; j < quadNormalsLength; ++j)
            {
                Vector128<float> normal = quadNormals[j];

                Vector128<float> bestDistance = _mm_set1_ps(float.NegativeInfinity);
                uint bestCentroid = 0;
                for (int k = 0; k < centroidsLength; ++k)
                {
                    Vector128<float> distance = _mm_dp_ps(centroids[k], normal, 0x7F);
                    if (_mm_comige_ss(distance, bestDistance))
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
                centroids[k] = _mm_setzero_ps();
            }

            for (int j = 0; j < quadNormalsLength; ++j)
            {
                int k = (int)centroidAssignment[j];

                centroids[k] = _mm_add_ps(centroids[k], quadNormals[j]);
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

        Occluder occluder = new();

        Vector128<float> invExtents = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sub_ps(refMax, refMin));

        Vector128<float> scalingX = _mm_set1_ps(2047.0f);
        Vector128<float> scalingY = _mm_set1_ps(2047.0f);
        Vector128<float> scalingZ = _mm_set1_ps(1023.0f);

        Vector128<float> half = _mm_set1_ps(0.5f);

        occluder.m_packetCount = 0;
        occluder.m_vertexData = (Vector256<int>*)(NativeMemory.AlignedAlloc((uint)orderedVertices.Length * 4, 32));

        Vector128<int>* v = stackalloc Vector128<int>[8];

        for (int i = 0; i < orderedVertices.Length; i += 32)
        {
            for (int j = 0; j < 4; ++j)
            {
                // Transform into [0,1] space relative to bounding box
                Vector128<float> v0 = _mm_mul_ps(_mm_sub_ps(orderedVertices[i + j + 0], refMin), invExtents);
                Vector128<float> v1 = _mm_mul_ps(_mm_sub_ps(orderedVertices[i + j + 4], refMin), invExtents);
                Vector128<float> v2 = _mm_mul_ps(_mm_sub_ps(orderedVertices[i + j + 8], refMin), invExtents);
                Vector128<float> v3 = _mm_mul_ps(_mm_sub_ps(orderedVertices[i + j + 12], refMin), invExtents);
                Vector128<float> v4 = _mm_mul_ps(_mm_sub_ps(orderedVertices[i + j + 16], refMin), invExtents);
                Vector128<float> v5 = _mm_mul_ps(_mm_sub_ps(orderedVertices[i + j + 20], refMin), invExtents);
                Vector128<float> v6 = _mm_mul_ps(_mm_sub_ps(orderedVertices[i + j + 24], refMin), invExtents);
                Vector128<float> v7 = _mm_mul_ps(_mm_sub_ps(orderedVertices[i + j + 28], refMin), invExtents);

                // Transpose into [xxxx][yyyy][zzzz][wwww]
                _MM_TRANSPOSE4_PS(ref v0, ref v1, ref v2, ref v3);
                _MM_TRANSPOSE4_PS(ref v4, ref v5, ref v6, ref v7);

                // Scale and truncate to int
                v0 = _mm_fmadd_ps(v0, scalingX, half);
                v1 = _mm_fmadd_ps(v1, scalingY, half);
                v2 = _mm_fmadd_ps(v2, scalingZ, half);

                v4 = _mm_fmadd_ps(v4, scalingX, half);
                v5 = _mm_fmadd_ps(v5, scalingY, half);
                v6 = _mm_fmadd_ps(v6, scalingZ, half);

                Vector128<int> X0 = _mm_sub_epi32(_mm_cvttps_epi32(v0), _mm_set1_epi32(1024));
                Vector128<int> Y0 = _mm_cvttps_epi32(v1);
                Vector128<int> Z0 = _mm_cvttps_epi32(v2);

                Vector128<int> X1 = _mm_sub_epi32(_mm_cvttps_epi32(v4), _mm_set1_epi32(1024));
                Vector128<int> Y1 = _mm_cvttps_epi32(v5);
                Vector128<int> Z1 = _mm_cvttps_epi32(v6);

                // Pack to 11/11/10 format
                Vector128<int> XYZ0 = _mm_or_si128(_mm_slli_epi32(X0, 21), _mm_or_si128(_mm_slli_epi32(Y0, 10), Z0));
                Vector128<int> XYZ1 = _mm_or_si128(_mm_slli_epi32(X1, 21), _mm_or_si128(_mm_slli_epi32(Y1, 10), Z1));

                v[2 * j + 0] = XYZ0;
                v[2 * j + 1] = XYZ1;
            }

            occluder.m_vertexData[occluder.m_packetCount++] = _mm256_loadu_si256((Vector256<int>*)(v + 0));
            occluder.m_vertexData[occluder.m_packetCount++] = _mm256_loadu_si256((Vector256<int>*)(v + 2));
            occluder.m_vertexData[occluder.m_packetCount++] = _mm256_loadu_si256((Vector256<int>*)(v + 4));
            occluder.m_vertexData[occluder.m_packetCount++] = _mm256_loadu_si256((Vector256<int>*)(v + 6));
        }

        occluder.m_refMin = refMin;
        occluder.m_refMax = refMax;

        Vector128<float> min = _mm_set1_ps(float.PositiveInfinity);
        Vector128<float> max = _mm_set1_ps(float.NegativeInfinity);

        for (int i = 0; i < orderedVertices.Length; ++i)
        {
            min = _mm_min_ps(vertices[i], min);
            max = _mm_max_ps(vertices[i], max);
        }

        // Set W = 1 - this is expected by frustum culling code
        min = _mm_blend_ps(min, _mm_set1_ps(1.0f), 0b1000);
        max = _mm_blend_ps(max, _mm_set1_ps(1.0f), 0b1000);

        occluder.m_boundsMin = min;
        occluder.m_boundsMax = max;

        occluder.m_center = _mm_mul_ps(_mm_add_ps(max, min), _mm_set1_ps(0.5f));

        return occluder;
    }
}