
namespace SoftwareRasterizer;

public interface IMultiplyAdd
{
    static abstract bool IsAcceleratedScalar { get; }

    static abstract float MulAdd(float a, float b, float c);

    static abstract float MulAddNeg(float a, float b, float c);

    static abstract float MulSub(float a, float b, float c);
}