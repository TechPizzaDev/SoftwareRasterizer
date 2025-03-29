using System;
using System.Runtime.InteropServices;
using TerraFX.Interop.Windows;

namespace SoftwareRasterizer.Win32;

internal class Program
{
    static void Main(string[] args)
    {
        if (OperatingSystem.IsWindows())
        {
            Win32Main main = new(1280, 720);
            nint hinstance = Marshal.GetHINSTANCE(typeof(Main).Module);
            main.WinMain((HINSTANCE) hinstance);
        }
        else
        {
            throw new PlatformNotSupportedException();
        }
    }
}