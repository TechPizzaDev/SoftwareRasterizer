/*
#include "SurfaceAreaHeuristic.h"

#include "VectorMath.h"

#include <algorithm>
#include <numeric>
*/

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

using static Intrinsics;

public static unsafe class SurfaceAreaHeuristic
{
    private readonly struct AabbComparer : IComparer<uint>
    {
        public readonly Aabb* aabbs;
        public readonly uint mask;

        public AabbComparer(Aabb* aabbs, uint mask)
        {
            this.aabbs = aabbs;
            this.mask = mask;
        }

        public int Compare(uint x, uint y)
        {
            Vector128<float> aabbX = aabbs[x].getCenter();
            Vector128<float> aabbY = aabbs[y].getCenter();

            if ((_mm_movemask_ps(_mm_cmplt_ps(aabbX, aabbY)) & mask) != 0)
            {
                return -1;
            }

            if ((_mm_movemask_ps(_mm_cmpgt_ps(aabbX, aabbY)) & mask) != 0)
            {
                return 1;
            }

            return 0;
        }
    }

    private static int sahSplit(Aabb* aabbsIn, uint splitGranularity, uint* indicesStart, uint* indicesEnd)
    {
        uint numIndices = (uint)(indicesEnd - indicesStart);

        Vector128<float> bestCost = _mm_set1_ps(float.PositiveInfinity);

        int bestAxis = -1;
        int bestIndex = -1;

        Vector128<float>* areasFromLeft = (Vector128<float>*)NativeMemory.AlignedAlloc(
            byteCount: numIndices * (uint)sizeof(Vector128<float>),
            alignment: (uint)sizeof(Vector128<float>));

        Vector128<float>* areasFromRight = (Vector128<float>*)NativeMemory.AlignedAlloc(
            byteCount: numIndices * (uint)sizeof(Vector128<float>),
            alignment: (uint)sizeof(Vector128<float>));

        for (int splitAxis = 0; splitAxis < 3; ++splitAxis)
        {
            // Sort along center position
            StableSort.stable_sort(indicesStart, indicesEnd, new AabbComparer(aabbsIn, 1u << bestAxis));

            Aabb fromLeft = new();
            for (uint i = 0; i < numIndices; ++i)
            {
                fromLeft.include(aabbsIn[indicesStart[i]]);
                areasFromLeft[i] = fromLeft.surfaceArea();
            }

            Aabb fromRight = new();
            for (int i = (int)(numIndices - 1); i >= 0; --i)
            {
                fromRight.include(aabbsIn[indicesStart[i]]);
                areasFromRight[i] = fromRight.surfaceArea();
            }

            for (uint splitIndex = splitGranularity; splitIndex < numIndices - splitGranularity; splitIndex += splitGranularity)
            {
                int countLeft = (int)(splitIndex);
                int countRight = (int)(numIndices - splitIndex);

                Vector128<float> areaLeft = areasFromLeft[splitIndex - 1];
                Vector128<float> areaRight = areasFromRight[splitIndex];
                Vector128<float> scaledAreaLeft = _mm_mul_ss(areaLeft, _mm_cvtsi32_ss(_mm_setzero_ps(), countLeft));
                Vector128<float> scaledAreaRight = _mm_mul_ss(areaRight, _mm_cvtsi32_ss(_mm_setzero_ps(), countRight));

                Vector128<float> cost = _mm_add_ss(scaledAreaLeft, scaledAreaRight);

                if (_mm_comilt_ss(cost, bestCost))
                {
                    bestCost = cost;
                    bestAxis = splitAxis;
                    bestIndex = (int)splitIndex;
                }
            }
        }

        NativeMemory.AlignedFree(areasFromLeft);
        NativeMemory.AlignedFree(areasFromRight);

        // Sort again according to best axis
        StableSort.stable_sort(indicesStart, indicesEnd, new AabbComparer(aabbsIn, 1u << bestAxis));

        return bestIndex;
    }

    private static void generateBatchesRecursive(
        Aabb* aabbsIn, uint targetSize, uint splitGranularity, uint* indicesStart, uint* indicesEnd, List<Vector> result)
    {
        int splitIndex = sahSplit(aabbsIn, splitGranularity, indicesStart, indicesEnd);

        uint** range = stackalloc uint*[] { indicesStart, indicesStart + splitIndex, indicesEnd };

        for (int i = 0; i < 2; ++i)
        {
            long batchSize = range[i + 1] - range[i];
            if (batchSize < targetSize)
            {
                result.Add(new Vector(range[i], range[i + 1]));
            }
            else
            {
                generateBatchesRecursive(aabbsIn, targetSize, splitGranularity, range[i], range[i + 1], result);
            }
        }
    }

    public static uint* generateBatches(ReadOnlySpan<Aabb> aabbs, uint targetSize, uint splitGranularity, List<Vector> result)
    {
        uint indexCount = (uint)aabbs.Length;
        uint* indices = (uint*)NativeMemory.Alloc(indexCount, sizeof(uint));
        for (uint i = 0; i < indexCount; i++)
        {
            indices[i] = i;
        }

        fixed (Aabb* aabbPtr = aabbs)
        {
            generateBatchesRecursive(aabbPtr, targetSize, splitGranularity, &indices[0], &indices[0] + indexCount, result);
        }
        return indices;
    }

    public static void freeBatches(uint* ptr)
    {
        NativeMemory.Free(ptr);
    }

    public readonly struct Vector
    {
        public readonly uint* Start;
        public readonly uint* End;

        public long Length => End - Start;

        public Vector(uint* start, uint* end)
        {
            Start = start;
            End = end;
        }
    }
}