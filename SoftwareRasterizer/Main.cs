using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SoftwareRasterizer;

using static VectorMath;

public unsafe class Main
{
    public uint WindowWidth { get; }
    public uint WindowHeight { get; }

#if true
    protected const string SCENE = "Castle";
    protected const float FOV = 0.628f;
    protected Vector3 g_cameraPosition = new(27.0f, 2.0f, 47.0f);
    protected Vector3 g_cameraDirection = new(0.142582759f, 0.0611068942f, -0.987894833f);
    protected Vector3 g_upVector = new(0.0f, 1.0f, 0.0f);
#else
    protected const string SCENE = "Sponza";
    protected const float FOV = 1.04f;
    protected Vector3 g_cameraPosition = new(0.0f, 0.0f, 0.0f);
    protected Vector3 g_cameraDirection = new(1.0f, 0.0f, 0.0f);
    protected Vector3 g_upVector = new(0.0f, 0.0f, 1.0f);
#endif

    protected RasterizationTable g_rasterizationTable;
    protected Rasterizer g_rasterizer;

    protected List<Occluder> g_occluders = new();

    protected byte* g_rawData;

    protected Queue<double> samples = new();

    public Main(uint width, uint height)
    {
        WindowWidth = width;
        WindowHeight = height;

        uint[] inputIndices;
        {
            string fileName = Path.Combine(SCENE, "IndexBuffer.bin");
            using FileStream inFile = File.OpenRead(fileName);

            Console.WriteLine($"Opened {fileName}");

            long size = inFile.Length;
            long numIndices = size / sizeof(uint);

            inputIndices = new uint[numIndices];

            Span<byte> dst = MemoryMarshal.AsBytes(inputIndices.AsSpan());
            inFile.ReadAtLeast(dst, dst.Length);
        }

        Vector4* vertices;
        uint vertexCount;
        {
            string fileName = Path.Combine(SCENE, "VertexBuffer.bin");
            using FileStream inFile = File.OpenRead(fileName);

            Console.WriteLine($"Opened {fileName}");

            long size = inFile.Length;
            vertexCount = (uint)(size / sizeof(Vector4));

            uint vertexByteCount = vertexCount * (uint)sizeof(Vector4);
            vertices = (Vector4*)NativeMemory.Alloc(vertexByteCount);

            Span<byte> dst = new(vertices, (int)vertexByteCount);
            inFile.ReadAtLeast(dst, dst.Length);
        }

        List<uint> indexList = QuadDecomposition.decompose(
            inputIndices,
            new ReadOnlySpan<Vector4>(vertices, (int)vertexCount));

        g_rasterizationTable = new RasterizationTable();
        g_rasterizer = V128Rasterizer<FmaX86>.Create(g_rasterizationTable, WindowWidth, WindowHeight);

        // Pad to a multiple of 8 quads
        while (indexList.Count % 32 != 0)
        {
            indexList.Add(indexList[0]);
        }

        ReadOnlySpan<uint> indices = CollectionsMarshal.AsSpan(indexList);

        int quadAabbCount = indices.Length / 4;
        nuint quadAabbByteCount = (nuint)quadAabbCount * (uint)sizeof(Aabb);
        Aabb* quadAabbs = (Aabb*)NativeMemory.AlignedAlloc(quadAabbByteCount, (uint)sizeof(Vector4));
        for (int quadIndex = 0; quadIndex < quadAabbCount; ++quadIndex)
        {
            Aabb aabb = new();
            aabb.include(vertices[indices[4 * quadIndex + 0]]);
            aabb.include(vertices[indices[4 * quadIndex + 1]]);
            aabb.include(vertices[indices[4 * quadIndex + 2]]);
            aabb.include(vertices[indices[4 * quadIndex + 3]]);
            quadAabbs[quadIndex] = aabb;
        }

        List<SurfaceAreaHeuristic.Vector> batchAssignment = new();
        uint* batchAssignmentPtr = SurfaceAreaHeuristic.generateBatches(new ReadOnlySpan<Aabb>(quadAabbs, quadAabbCount), 512, 8, batchAssignment);

        Aabb refAabb = new();
        for (uint i = 0; i < vertexCount; i++)
        {
            refAabb.include(vertices[i]);
        }

        // Bake occluders
        foreach (SurfaceAreaHeuristic.Vector batch in batchAssignment)
        {
            Vector4[] batchVertices = new Vector4[batch.Length * 4];
            for (int i = 0; i < batch.Length; i++)
            {
                uint quadIndex = batch.Start[i];
                batchVertices[i * 4 + 0] = vertices[(int)indices[(int)(quadIndex * 4 + 0)]];
                batchVertices[i * 4 + 1] = vertices[(int)indices[(int)(quadIndex * 4 + 1)]];
                batchVertices[i * 4 + 2] = vertices[(int)indices[(int)(quadIndex * 4 + 2)]];
                batchVertices[i * 4 + 3] = vertices[(int)indices[(int)(quadIndex * 4 + 3)]];
            }

            g_occluders.Add(Occluder.Bake(batchVertices, refAabb.m_min, refAabb.m_max));
        }
        SurfaceAreaHeuristic.freeBatches(batchAssignmentPtr);

        g_rawData = (byte*)NativeMemory.Alloc(WindowWidth * WindowHeight * 4);
    }

    public void Rasterize()
    {
        Matrix4x4 projMatrix = XMMatrixPerspectiveFovLH(FOV, WindowWidth / (float)WindowHeight, 1.0f, 5000.0f);
        Matrix4x4 viewMatrix = XMMatrixLookToLH(g_cameraPosition, g_cameraDirection, g_upVector);
        Matrix4x4 viewProjection = Matrix4x4.Multiply(viewMatrix, projMatrix);

        float* mvp = stackalloc float[16];
        Unsafe.CopyBlockUnaligned(mvp, &viewProjection, 64);

        long raster_start = Stopwatch.GetTimestamp();
        g_rasterizer.clear();
        g_rasterizer.setModelViewProjection(mvp);

        // Sort front to back
        Algo.sort(CollectionsMarshal.AsSpan(g_occluders), new OccluderComparerV128(new Vector4(g_cameraPosition, 0)));

        int clips = 0;
        int notClips = 0;
        int misses = 0;
        foreach (ref readonly Occluder occluder in CollectionsMarshal.AsSpan(g_occluders))
        {
            if (g_rasterizer.queryVisibility(occluder.m_boundsMin, occluder.m_boundsMax, out bool needsClipping))
            {
                if (needsClipping)
                {
                    g_rasterizer.rasterize<Rasterizer.NearClipped>(occluder);
                    clips++;
                }
                else
                {
                    g_rasterizer.rasterize<Rasterizer.NotNearClipped>(occluder);
                    notClips++;
                }
            }
            else
            {
                misses++;
            }
        }

        long raster_end = Stopwatch.GetTimestamp();
        //Console.WriteLine(clips + " - " + notClips + " - " + misses);

        double rasterTime = Stopwatch.GetElapsedTime(raster_start, raster_end).TotalMilliseconds;

        samples.Enqueue(rasterTime);

        while (samples.Count > 60 * 10)
        {
            samples.Dequeue();
        }
    }

    public (double avgTime, double stDev, double median) GetTimings()
    {
        double avgRasterTime = samples.Sum() / samples.Count;
        double sqSum = samples.Select(x => x * x).Sum();
        double stDev = Math.Sqrt(sqSum / samples.Count - avgRasterTime * avgRasterTime);

        double median = samples.Order().ElementAt(samples.Count / 2);

        return (avgRasterTime, median, stDev);
    }

    public void CycleRasterizerImpl()
    {
        var previousRasterizer = g_rasterizer;
        if (previousRasterizer is Avx2Rasterizer<FmaIntrinsic> or Avx2Rasterizer<FmaX86>)
        {
            g_rasterizer = V128Rasterizer<FmaX86>.Create(g_rasterizationTable, WindowWidth, WindowHeight);
        }
        else if (previousRasterizer is V128Rasterizer<FmaIntrinsic> or V128Rasterizer<FmaX86>)
        {
            g_rasterizer = ScalarRasterizer.Create(g_rasterizationTable, WindowWidth, WindowHeight);
        }
        else
        {
            g_rasterizer = Avx2Rasterizer<FmaX86>.Create(g_rasterizationTable, WindowWidth, WindowHeight);
        }

        previousRasterizer.Dispose();
        Console.WriteLine($"Changed to {g_rasterizer}  (from {previousRasterizer})");
    }

    readonly struct OccluderComparerV128 : Algo.IComparer<Occluder>
    {
        public readonly Vector4 CameraPosition;

        public OccluderComparerV128(Vector4 cameraPosition)
        {
            CameraPosition = cameraPosition;
        }

        public bool Compare(in Occluder x, in Occluder y)
        {
            Vector128<float> dist1 = (x.m_center - CameraPosition).AsVector128();
            Vector128<float> dist2 = (y.m_center - CameraPosition).AsVector128();

            Vector128<float> a = V128Helper.DotProduct_x7F(dist1, dist1);
            Vector128<float> b = V128Helper.DotProduct_x7F(dist2, dist2);

            if (Sse.IsSupported)
            {
                return Sse.CompareScalarOrderedLessThan(a, b);
            }
            else
            {
                float sA = a.ToScalar();
                float sB = b.ToScalar();
                return !float.IsNaN(sA) && !float.IsNaN(sB) && sA < sB;
            }
        }
    }

    readonly struct ScalarOccluderComparer : Algo.IComparer<Occluder>
    {
        public readonly Vector4 CameraPosition;

        public ScalarOccluderComparer(Vector4 cameraPosition)
        {
            CameraPosition = cameraPosition;
        }

        public bool Compare(in Occluder x, in Occluder y)
        {
            Vector4 dist1 = (x.m_center - CameraPosition);
            Vector4 dist2 = (y.m_center - CameraPosition);

            float a = ScalarMath.DotProduct_x7F(dist1);
            float b = ScalarMath.DotProduct_x7F(dist2);

            return a < b;
        }
    }
}