using System.Runtime.InteropServices;
using TerraFX.Interop.Windows;

namespace SoftwareRasterizer;

internal class Program
{
    static void Main(string[] args)
    {
        nint hinstance = Marshal.GetHINSTANCE(typeof(Main).Module);
        SoftwareRasterizer.Main.wWinMain((HINSTANCE)hinstance);
    }
}
