using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SoftwareRasterizer;

public static unsafe class SurfaceAreaHeuristic
{
    private readonly struct V128AabbComparer : Algo.IComparer<uint>
    {
        public readonly Aabb* aabbs;
        public readonly uint mask;

        public V128AabbComparer(Aabb* aabbs, uint mask)
        {
            this.aabbs = aabbs;
            this.mask = mask;
        }

        public bool Compare(in uint x, in uint y)
        {
            Vector128<float> aabbX = aabbs[x].getCenter().AsVector128();
            Vector128<float> aabbY = aabbs[y].getCenter().AsVector128();
            return (Vector128.ExtractMostSignificantBits(Vector128.LessThan(aabbX, aabbY)) & mask) != 0;
        }
    }

    private readonly struct ScalarAabbComparer : Algo.IComparer<uint>
    {
        public readonly Aabb* aabbs;
        public readonly uint mask;

        public ScalarAabbComparer(Aabb* aabbs, uint mask)
        {
            this.aabbs = aabbs;
            this.mask = mask;
        }

        public bool Compare(in uint x, in uint y)
        {
            Vector4 aabbX = aabbs[x].getCenter();
            Vector4 aabbY = aabbs[y].getCenter();

            uint sum = 0;
            sum |= aabbX.X < aabbY.X ? (1u << 0) : 0;
            sum |= aabbX.Y < aabbY.Y ? (1u << 1) : 0;
            sum |= aabbX.Z < aabbY.Z ? (1u << 2) : 0;
            sum |= aabbX.W < aabbY.W ? (1u << 3) : 0;

            return (sum & mask) != 0;
        }
    }

    private static int sahSplit(Aabb* aabbsIn, uint splitGranularity, uint* indicesStart, uint* indicesEnd)
    {
        uint numIndices = (uint)(indicesEnd - indicesStart);

        float bestCost = float.PositiveInfinity;

        int bestAxis = -1;
        int bestIndex = -1;

        Vector4* areasFromLeft = (Vector4*)NativeMemory.AlignedAlloc(
            byteCount: numIndices * (uint)sizeof(Vector4),
            alignment: (uint)sizeof(Vector4));

        Vector4* areasFromRight = (Vector4*)NativeMemory.AlignedAlloc(
            byteCount: numIndices * (uint)sizeof(Vector4),
            alignment: (uint)sizeof(Vector4));

        Ref<uint> sortStart = Ref<uint>.FromPtr(indicesStart);
        Ref<uint> sortEnd = Ref<uint>.FromPtr(indicesEnd);

        for (int splitAxis = 0; splitAxis < 3; ++splitAxis)
        {
            // Sort along center position
            if (Vector128.IsHardwareAccelerated)
            {
                Algo.stable_sort(sortStart, sortEnd, new V128AabbComparer(aabbsIn, 1u << splitAxis));
            }
            else
            {
                Algo.stable_sort(sortStart, sortEnd, new ScalarAabbComparer(aabbsIn, 1u << splitAxis));
            }

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
                int countLeft = (int)splitIndex;
                int countRight = (int)(numIndices - splitIndex);

                Vector4 areaLeft = areasFromLeft[splitIndex - 1];
                Vector4 areaRight = areasFromRight[splitIndex];
                float scaledAreaLeft = areaLeft.X * countLeft;
                float scaledAreaRight = areaRight.X * countRight;

                float cost = scaledAreaLeft + scaledAreaRight;

                if (cost < bestCost)
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
        if (Vector128.IsHardwareAccelerated)
        {
            Algo.stable_sort(sortStart, sortEnd, new V128AabbComparer(aabbsIn, 1u << bestAxis));
        }
        else
        {
            Algo.stable_sort(sortStart, sortEnd, new ScalarAabbComparer(aabbsIn, 1u << bestAxis));
        }

        return bestIndex;
    }

    private static void generateBatchesRecursive(
        Aabb* aabbsIn, uint targetSize, uint splitGranularity, uint* indicesStart, uint* indicesEnd, List<Vector> result)
    {
        int splitIndex = sahSplit(aabbsIn, splitGranularity, indicesStart, indicesEnd);

        uint** range = stackalloc uint*[] { indicesStart, indicesStart + splitIndex, indicesEnd };

        for (int i = 0; i < 2; ++i)
        {
            uint* start = range[i];
            uint* end = range[i + 1];

            long batchSize = end - start;
            if (batchSize < targetSize)
            {
                result.Add(new Vector(start, end));
            }
            else
            {
                generateBatchesRecursive(aabbsIn, targetSize, splitGranularity, start, end, result);
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

    [DebuggerDisplay($"{{{nameof(GetDebuggerDisplay)}(),nq}}")]
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

        private string GetDebuggerDisplay()
        {
            return "Length = " + Length;
        }
    }
}