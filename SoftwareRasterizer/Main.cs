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
using System.Runtime.Versioning;
using TerraFX.Interop.Windows;

namespace SoftwareRasterizer;

using static VectorMath;
using static Windows;
using static VK;

[SupportedOSPlatform("windows")]
public static unsafe class Main
{
    public const int WINDOW_WIDTH = 1280;
    public const int WINDOW_HEIGHT = 720;

#if true
    const string SCENE = "Castle";
    const float FOV = 0.628f;
    static Vector3 g_cameraPosition = new(27.0f, 2.0f, 47.0f);
    static Vector3 g_cameraDirection = new(0.142582759f, 0.0611068942f, -0.987894833f);
    static Vector3 g_upVector = new(0.0f, 1.0f, 0.0f);
#else
    const string SCENE = "Sponza";
    const float FOV = 1.04f;
    static Vector3 g_cameraPosition = new(0.0f, 0.0f, 0.0f);
    static Vector3 g_cameraDirection = new(1.0f, 0.0f, 0.0f);
    static Vector3 g_upVector = new(0.0f, 0.0f, 1.0f);
#endif

    static RasterizationTable g_rasterizationTable;
    static Rasterizer g_rasterizer;

    static HBITMAP g_hBitmap;
    static List<Occluder> g_occluders = new();

    static byte* g_rawData;

    public static int wWinMain(HINSTANCE hInstance)
    {
        uint[] inputIndices;
        {
            string fileName = Path.Combine(SCENE, "IndexBuffer.bin");
            using FileStream inFile = File.OpenRead(fileName);

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
        g_rasterizer = Avx2Rasterizer<HardFma>.Create(g_rasterizationTable, WINDOW_WIDTH, WINDOW_HEIGHT);

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

        g_rawData = (byte*)NativeMemory.Alloc(WINDOW_WIDTH * WINDOW_HEIGHT * 4);

        ushort windowClass;
        fixed (char* className = "RasterizerWindow")
        {
            WNDCLASSEXW wcex = new();
            wcex.cbSize = (uint)sizeof(WNDCLASSEXW);

            wcex.style = CS.CS_HREDRAW | CS.CS_VREDRAW;
            wcex.lpfnWndProc = &WndProc;
            wcex.hInstance = hInstance;
            wcex.hIcon = LoadIcon(default, IDI.IDI_APPLICATION);
            wcex.hCursor = LoadCursor(default, IDC.IDC_ARROW);
            wcex.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
            wcex.lpszClassName = (ushort*)className;

            windowClass = RegisterClassExW(&wcex);
        }

        HWND hWnd;
        fixed (char* windowName = "Rasterizer")
        {
            hWnd = CreateWindowW((ushort*)windowClass, (ushort*)windowName, WS.WS_SYSMENU,
              CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_WIDTH, WINDOW_HEIGHT, default, default, hInstance, default);
        }

        HDC hdc = GetDC(hWnd);
        g_hBitmap = CreateCompatibleBitmap(hdc, WINDOW_WIDTH, WINDOW_HEIGHT);
        ReleaseDC(hWnd, hdc);

        ShowWindow(hWnd, SW.SW_SHOW);
        UpdateWindow(hWnd);

        MSG msg;
        while (GetMessage(&msg, default, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        return 0;
    }

    static List<double> samples = new();

    static long lastPaint = Stopwatch.GetTimestamp();

    [UnmanagedCallersOnly]
    static LRESULT WndProc(HWND hWnd, uint message, WPARAM wParam, LPARAM lParam)
    {
        switch (message)
        {
            case WM.WM_PAINT:
            {
                Matrix4x4 projMatrix = XMMatrixPerspectiveFovLH(FOV, WINDOW_WIDTH / (float)WINDOW_HEIGHT, 1.0f, 5000.0f);
                Matrix4x4 viewMatrix = XMMatrixLookToLH(g_cameraPosition, g_cameraDirection, g_upVector);
                Matrix4x4 viewProjection = Matrix4x4.Multiply(viewMatrix, projMatrix);

                float* mvp = stackalloc float[16];
                Unsafe.CopyBlockUnaligned(mvp, &viewProjection, 64);

                long raster_start = Stopwatch.GetTimestamp();
                g_rasterizer.clear();
                g_rasterizer.setModelViewProjection(mvp);

                // Sort front to back
                Algo.sort(CollectionsMarshal.AsSpan(g_occluders), new Sse41OccluderComparer(new Vector4(g_cameraPosition, 0)));

                foreach (ref readonly Occluder occluder in CollectionsMarshal.AsSpan(g_occluders))
                {
                    if (g_rasterizer.queryVisibility(occluder.m_boundsMin, occluder.m_boundsMax, out bool needsClipping))
                    {
                        if (needsClipping)
                        {
                            g_rasterizer.rasterize<Rasterizer.NearClipped>(occluder);
                        }
                        else
                        {
                            g_rasterizer.rasterize<Rasterizer.NotNearClipped>(occluder);
                        }
                    }
                }

                long raster_end = Stopwatch.GetTimestamp();

                double rasterTime = Stopwatch.GetElapsedTime(raster_start, raster_end).TotalMilliseconds;

                samples.Add(rasterTime);

                double avgRasterTime = samples.Sum() / samples.Count;
                double sqSum = samples.Select(x => x * x).Sum();
                double stDev = Math.Sqrt(sqSum / samples.Count - avgRasterTime * avgRasterTime);

                //std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());
                double median = samples[samples.Count / 2];

                if (samples.Count > 100)
                    samples.Clear();

                int fps = (int)(1000.0f / avgRasterTime);

                string title = $"FPS: {fps}      Rasterization time: {avgRasterTime:0.000}±{stDev:0.000}ms stddev / {median:0.000}ms median";
                fixed (char* titlePtr = title)
                {
                    SetWindowText(hWnd, (ushort*)titlePtr);
                }

                g_rasterizer.readBackDepth(g_rawData);

                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hWnd, &ps);

                HDC hdcMem = CreateCompatibleDC(hdc);

                BITMAPINFO info = new();
                info.bmiHeader.biSize = (uint)sizeof(BITMAPINFOHEADER);
                info.bmiHeader.biWidth = WINDOW_WIDTH;
                info.bmiHeader.biHeight = WINDOW_HEIGHT;
                info.bmiHeader.biPlanes = 1;
                info.bmiHeader.biBitCount = 32;
                info.bmiHeader.biCompression = BI.BI_RGB;
                SetDIBits(hdcMem, g_hBitmap, 0, WINDOW_HEIGHT, g_rawData, &info, DIB_RGB_COLORS);

                BITMAP bm;
                HGDIOBJ hbmOld = SelectObject(hdcMem, g_hBitmap);

                GetObject(g_hBitmap, sizeof(BITMAP), &bm);

                BitBlt(hdc, 0, 0, bm.bmWidth, bm.bmHeight, hdcMem, 0, 0, SRCCOPY);

                SelectObject(hdcMem, hbmOld);
                DeleteDC(hdcMem);

                EndPaint(hWnd, &ps);

                long now = Stopwatch.GetTimestamp();

                Vector3 right = Vector3.Normalize(Vector3.Cross(g_cameraDirection, g_upVector));
                float deltaTime = (float)Stopwatch.GetElapsedTime(lastPaint, now).TotalMilliseconds;
                float translateSpeed = 0.01f * deltaTime;
                float rotateSpeed = 0.002f * deltaTime;

                lastPaint = now;

                if (GetAsyncKeyState(VK_SHIFT) != 0)
                    translateSpeed *= 3.0f;

                if (GetAsyncKeyState(VK_CONTROL) != 0)
                    translateSpeed *= 0.1f;

                if (GetAsyncKeyState('W') != 0)
                    g_cameraPosition = Vector3.Add(g_cameraPosition, Vector3.Multiply(g_cameraDirection, translateSpeed));

                if (GetAsyncKeyState('S') != 0)
                    g_cameraPosition = Vector3.Add(g_cameraPosition, Vector3.Multiply(g_cameraDirection, -translateSpeed));

                if (GetAsyncKeyState('A') != 0)
                    g_cameraPosition = Vector3.Add(g_cameraPosition, Vector3.Multiply(right, translateSpeed));

                if (GetAsyncKeyState('D') != 0)
                    g_cameraPosition = Vector3.Add(g_cameraPosition, Vector3.Multiply(right, -translateSpeed));

                if (GetAsyncKeyState(VK_UP) != 0)
                    g_cameraDirection = Vector3.Transform(g_cameraDirection, Quaternion.CreateFromAxisAngle(right, rotateSpeed));

                if (GetAsyncKeyState(VK_DOWN) != 0)
                    g_cameraDirection = Vector3.Transform(g_cameraDirection, Quaternion.CreateFromAxisAngle(right, -rotateSpeed));

                if (GetAsyncKeyState(VK_LEFT) != 0)
                    g_cameraDirection = Vector3.Transform(g_cameraDirection, Quaternion.CreateFromAxisAngle(g_upVector, -rotateSpeed));

                if (GetAsyncKeyState(VK_RIGHT) != 0)
                    g_cameraDirection = Vector3.Transform(g_cameraDirection, Quaternion.CreateFromAxisAngle(g_upVector, rotateSpeed));

                if ((GetAsyncKeyState('R') & 1) != 0)
                {
                    var previousRasterizer = g_rasterizer;
                    if (previousRasterizer is Avx2Rasterizer<SoftFma> or Avx2Rasterizer<HardFma>)
                    {
                        g_rasterizer = Sse41Rasterizer.Create(g_rasterizationTable, WINDOW_WIDTH, WINDOW_HEIGHT);
                    }
                    else if (previousRasterizer is Sse41Rasterizer)
                    {
                        g_rasterizer = ScalarRasterizer.Create(g_rasterizationTable, WINDOW_WIDTH, WINDOW_HEIGHT);
                    }
                    else
                    {
                        g_rasterizer = Avx2Rasterizer<HardFma>.Create(g_rasterizationTable, WINDOW_WIDTH, WINDOW_HEIGHT);
                    }

                    previousRasterizer.Dispose();
                    Console.WriteLine($"Changed to {g_rasterizer}  (from {previousRasterizer})");
                }

                InvalidateRect(hWnd, default, FALSE);
            }
            break;

            case WM.WM_DESTROY:
                PostQuitMessage(0);
                break;

            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
        }
        return 0;
    }

    readonly struct Sse41OccluderComparer : Algo.IComparer<Occluder>
    {
        public readonly Vector4 CameraPosition;

        public Sse41OccluderComparer(Vector4 cameraPosition)
        {
            CameraPosition = cameraPosition;
        }

        public bool Compare(in Occluder x, in Occluder y)
        {
            Vector128<float> dist1 = (x.m_center - CameraPosition).AsVector128();
            Vector128<float> dist2 = (y.m_center - CameraPosition).AsVector128();

            Vector128<float> a = Sse41.DotProduct(dist1, dist1, 0x7f);
            Vector128<float> b = Sse41.DotProduct(dist2, dist2, 0x7f);

            return Sse.CompareScalarOrderedLessThan(a, b);
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