using System;
using System.IO;
using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;

[assembly: SupportedOSPlatform("browser")]

namespace SoftwareRasterizer.Browser;

internal partial class Program
{
    static void Main(string[] args)
    {
        BrowserMain main = new(uint.Parse(args[0]), uint.Parse(args[1]));

        void callback(double value)
        {
            RequestAnimationFrame(callback);

            main.Rasterize();

            (double avgRasterTime, double stDev, double median) = main.GetTimings();
            int fps = (int)(1000.0f / avgRasterTime);
            string title = $"FPS: {fps}      Rasterization time: {avgRasterTime:0.000}±{stDev:0.000}ms stddev / {median:0.000}ms median";
        }

        RequestAnimationFrame(callback);
    }

    [JSImport("window.requestAnimationFrame", "main.js")]
    internal static partial void RequestAnimationFrame(
        [JSMarshalAs<JSType.Function<JSType.Number>>] Action<double> frameRequestCallback);
}