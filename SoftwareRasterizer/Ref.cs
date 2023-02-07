using System.Runtime.CompilerServices;

namespace SoftwareRasterizer;

#pragma warning disable CS0660 // Type defines operator == or operator != but does not override Object.Equals(object o)
#pragma warning disable CS0661 // Type defines operator == or operator != but does not override Object.GetHashCode()
public readonly ref struct Ref<T>
#pragma warning restore CS0661
#pragma warning restore CS0660
{
    public readonly ref T Value;

    public Ref(ref T value)
    {
        Value = ref value;
    }

    public static unsafe Ref<P> FromPtr<P>(P* ptr)
        where P : unmanaged
    {
        return new Ref<P>(ref *ptr);
    }

    public void Set(in T value)
    {
        Value = value;
    }

    public ref T Get()
    {
        return ref Value;
    }

    public static nuint Distance(Ref<T> start, Ref<T> end)
    {
        return (nuint)Unsafe.ByteOffset(ref start.Value, ref end.Value) / (nuint)Unsafe.SizeOf<T>();
    }

    public static bool operator ==(Ref<T> a, Ref<T> b)
    {
        return Unsafe.AreSame(ref a.Value, ref b.Value);
    }

    public static bool operator !=(Ref<T> a, Ref<T> b)
    {
        return !Unsafe.AreSame(ref a.Value, ref b.Value);
    }

    public static Ref<T> operator +(Ref<T> a, nint offset)
    {
        return new(ref Unsafe.Add(ref a.Value, offset));
    }

    public static Ref<T> operator +(Ref<T> a, nuint offset)
    {
        return new(ref Unsafe.Add(ref a.Value, offset));
    }

    public static Ref<T> operator ++(Ref<T> a)
    {
        return new(ref Unsafe.Add(ref a.Value, 1));
    }

    public static Ref<T> operator -(Ref<T> a, nint offset)
    {
        return new(ref Unsafe.Subtract(ref a.Value, offset));
    }

    public static Ref<T> operator -(Ref<T> a, nuint offset)
    {
        return new(ref Unsafe.Subtract(ref a.Value, offset));
    }

    public static Ref<T> operator --(Ref<T> a)
    {
        return new(ref Unsafe.Subtract(ref a.Value, 1));
    }
}