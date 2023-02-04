using System;
using System.Runtime.CompilerServices;

namespace SoftwareRasterizer;

public static unsafe class Algo
{
    public interface IComparer<T>
    {
        bool Compare(in T x, in T y);
    }

    private const int _ISORT_MAX = 32; // maximum size for insertion sort

    public static void stable_sort<TItem, TComparer>(
        Ref<TItem> start,
        Ref<TItem> end,
        TComparer comparer)
        where TComparer : IComparer<TItem>
    {
        // sort preserving order of equivalents
        _Adl_verify_range(start, end);

        nuint _Count = Ref<TItem>.Distance(start, end);
        if (_Count <= _ISORT_MAX)
        {
            _Insertion_sort_unchecked(start, end, comparer);
            return;
        }

        nuint capacity = _Count - _Count / 2;
        TItem[] _Temp_buf = new TItem[capacity];
        _Stable_sort_unchecked(start, end, _Count, _Temp_buf.AsRef(), capacity, comparer);
    }

    public static void sort<T, TComparer>(
        Ref<T> start,
        Ref<T> end,
        TComparer comparer)
        where TComparer : IComparer<T>
    {
        // TODO: other implementation?
        stable_sort(start, end, comparer);
    }

    public static void sort<TItem, TComparer>(
        Span<TItem> span,
        TComparer comparer)
        where TComparer : IComparer<TItem>
    {
        Ref<TItem> start = span.AsRef();
        Ref<TItem> end = span.AsRef() + span.Length;
        sort(start, end, comparer);
    }

    public static void _Stable_sort_unchecked<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Last,
        nuint _Count,
        Ref<TItem> _Temp_ptr,
        nuint _Capacity,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // sort preserving order of equivalents
        if (_Count <= _ISORT_MAX)
        {
            _Insertion_sort_unchecked(_First, _Last, _Pred); // small
            return;
        }

        // sort halves and merge
        nuint _Half_count = (_Count >> 1); // shift for codegen
        nuint _Half_count_ceil = (_Count - _Half_count);
        Ref<TItem> _Mid = _First + _Half_count_ceil;
        if (_Half_count_ceil <= _Capacity)
        {
            // temp buffer big enough, sort each half using buffer
            _Buffered_merge_sort_unchecked(_First, _Mid, _Half_count_ceil, _Temp_ptr, _Pred);
            _Buffered_merge_sort_unchecked(_Mid, _Last, _Half_count, _Temp_ptr, _Pred);
        }
        else
        {
            // temp buffer not big enough, divide and conquer
            _Stable_sort_unchecked(_First, _Mid, _Half_count_ceil, _Temp_ptr, _Capacity, _Pred);
            _Stable_sort_unchecked(_Mid, _Last, _Half_count, _Temp_ptr, _Capacity, _Pred);
        }

        _Buffered_inplace_merge_unchecked(
             _First, _Mid, _Last, _Half_count_ceil, _Half_count, _Temp_ptr, _Capacity, _Pred); // merge halves
    }

    private static void _Buffered_merge_sort_unchecked<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Last,
        nuint _Count,
        Ref<TItem> _Temp_ptr,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // sort using temp buffer for merges
        // pre: _Last - _First == _Count
        // pre: _Count <= capacity of buffer at _Temp_ptr; also allows safe narrowing to ptrdiff_t
        _Insertion_sort_isort_max_chunks(_First, _Last, _Count, _Pred);
        // merge adjacent pairs of chunks to and from temp buffer
        if (_Count <= _ISORT_MAX)
        {
            return;
        }

        // do the first merge, constructing elements in the temporary buffer
        _Uninitialized_chunked_merge_unchecked2(_First, _Last, _Temp_ptr, _Count, _Pred);
        nuint _Chunk = _ISORT_MAX;
        for (; ; )
        {
            // unconditionally merge elements back into the source buffer
            _Chunk <<= 1;
            _Chunked_merge_unchecked(_Temp_ptr, _Temp_ptr + _Count, _First, _Chunk, _Count, _Pred);
            _Chunk <<= 1;
            if (_Count <= _Chunk)
            {
                // if the input would be a single chunk, it's already sorted and we're done
                return;
            }

            // more merges necessary; merge to temporary buffer
            _Chunked_merge_unchecked(_First, _Last, _Temp_ptr, _Chunk, _Count, _Pred);
        }
    }

    private static void _Insertion_sort_isort_max_chunks<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Last,
        nuint _Count,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // insertion sort every chunk of distance _Isort_max<_BidIt> in [_First, _Last)
        // pre: _Count == distance(_First, _Last)
        for (; _ISORT_MAX < _Count; _Count -= _ISORT_MAX)
        {
            // sort chunks
            _First = _Insertion_sort_unchecked(_First, _First + _ISORT_MAX, _Pred);
        }

        _Insertion_sort_unchecked(_First, _Last, _Pred); // sort partial last chunk
    }

    private static void _Uninitialized_chunked_merge_unchecked2<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Last,
        Ref<TItem> _Dest,
        nuint _Count,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // move to uninitialized merging adjacent chunks of distance _Isort_max<_BidIt>
        // pre: _Count == distance(_First, _Last)
        // pre: _Chunk > 0
        _Uninitialized_backout<TItem> _Backout = new(_Dest);
        while (_Count > _ISORT_MAX)
        {
            _Count -= _ISORT_MAX;
            Ref<TItem> _Mid1 = _First + _ISORT_MAX;
            nuint _Chunk2 = Math.Min(_ISORT_MAX, _Count);
            _Count -= _Chunk2;
            Ref<TItem> _Mid2 = _Mid1 + _Chunk2;
            _Backout._Last = _Uninitialized_merge_move(_First, _Mid1, _Mid2, _Backout._Last, _Pred);
            _First = _Mid2;
        }

        _Uninitialized_move_unchecked(_First, _Last, _Backout._Last); // copy partial last chunk
        _Backout._Release();
    }

    private static Ref<TItem> _Uninitialized_merge_move<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Mid,
        Ref<TItem> _Last,
        Ref<TItem> _Dest,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // move merging ranges to uninitialized storage
        // pre: _First != _Mid && _Mid != _Last
        _Uninitialized_backout<TItem> _Backout = new(_Dest);
        Ref<TItem> _Next = _Mid;
        for (; ; )
        {
            if (_Pred.Compare(_Next.Get(), _First.Get()))
            {
                _Backout._Emplace_back(_Next.Get());
                ++_Next;

                if (_Next == _Last)
                {
                    _Backout._Last = _Uninitialized_move_unchecked(_First, _Mid, _Backout._Last);
                    return _Backout._Release();
                }
            }
            else
            {
                _Backout._Emplace_back(_First.Get());
                ++_First;

                if (_First == _Mid)
                {
                    _Backout._Last = _Uninitialized_move_unchecked(_Next, _Last, _Backout._Last);
                    return _Backout._Release();
                }
            }
        }
    }

    private static void _Chunked_merge_unchecked<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Last,
        Ref<TItem> _Dest,
        nuint _Chunk,
        nuint _Count,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // move merging adjacent chunks of distance _Chunk
        // pre: _Count == distance(_First, _Last)
        // pre: _Chunk > 0
        while (_Chunk < _Count)
        {
            _Count -= _Chunk;
            Ref<TItem> _Mid1 = _First + _Chunk;
            nuint _Chunk2 = Math.Min(_Chunk, _Count);
            _Count -= _Chunk2;
            Ref<TItem> _Mid2 = _Mid1 + _Chunk2;
            _Dest = _Merge_move(_First, _Mid1, _Mid2, _Dest, _Pred);
            _First = _Mid2;
        }

        _Move_unchecked(_First, _Last, _Dest); // copy partial last chunk
    }

    private static Ref<TItem> _Merge_move<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Mid,
        Ref<TItem> _Last,
        Ref<TItem> _Dest,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // move merging adjacent ranges [_First, _Mid) and [_Mid, _Last) to _Dest
        // pre: _First != _Mid && _Mid != _Last
        Ref<TItem> _Next = _Mid;
        for (; ; )
        {
            if (_Pred.Compare(_Next.Get(), _First.Get()))
            {
                _Dest.Set(_Next.Get());
                ++_Dest;
                ++_Next;

                if (_Next == _Last)
                {
                    return _Move_unchecked(_First, _Mid, _Dest);
                }
            }
            else
            {
                _Dest.Set(_First.Get());
                ++_Dest;
                ++_First;

                if (_First == _Mid)
                {
                    return _Move_unchecked(_Next, _Last, _Dest);
                }
            }
        }
    }

    private static void _Buffered_inplace_merge_unchecked<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Mid,
        Ref<TItem> _Last,
        nuint _Count1,
        nuint _Count2,
        Ref<TItem> _Temp_ptr,
        nuint _Capacity,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // merge sorted [_First, _Mid) with sorted [_Mid, _Last)
        // usual invariants *do not* apply; only sortedness applies
        // establish the usual invariants (explained in inplace_merge)
        if (_Mid == _Last)
        {
            return;
        }

        for (; ; )
        {
            if (_First == _Mid)
            {
                return;
            }

            if (_Pred.Compare(_Mid.Get(), _First.Get()))
            {
                break;
            }

            ++_First;
            --_Count1;
        }

        Ref<TItem> _Highest = _Mid - 1;
        do
        {
            --_Last;
            --_Count2;
            if (_Mid == _Last)
            {
                _Rotate_one_right(_First, _Mid, ++_Last);
                return;
            }
        }
        while (!_Pred.Compare(_Last.Get(), _Highest.Get()));

        ++_Last;
        ++_Count2;

        if (_Count1 == 1)
        {
            _Rotate_one_left(_First, _Mid, _Last);
            return;
        }

        _Buffered_inplace_merge_unchecked_impl(_First, _Mid, _Last, _Count1, _Count2, _Temp_ptr, _Capacity, _Pred);
    }

    private static void _Buffered_inplace_merge_unchecked_impl<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Mid,
        Ref<TItem> _Last,
        nuint _Count1,
        nuint _Count2,
        Ref<TItem> _Temp_ptr,
        nuint _Capacity,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // merge sorted [_First, _Mid) with sorted [_Mid, _Last)
        // usual invariants apply
        if (_Count1 <= _Count2 && _Count1 <= _Capacity)
        {
            _Inplace_merge_buffer_left(_First, _Mid, _Last, _Temp_ptr, _Pred);
        }
        else if (_Count2 <= _Capacity)
        {
            _Inplace_merge_buffer_right(_First, _Mid, _Last, _Temp_ptr, _Pred);
        }
        else
        {
            _Buffered_inplace_merge_divide_and_conquer(_First, _Mid, _Last, _Count1, _Count2, _Temp_ptr, _Capacity, _Pred);
        }
    }

    private static void _Inplace_merge_buffer_left<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Mid,
        Ref<TItem> _Last,
        Ref<TItem> _Temp_ptr,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // move the range [_First, _Mid) to _Temp_ptr, and merge it with [_Mid, _Last) to _First
        // usual invariants apply
        _Uninitialized_backout<TItem> _Backout = new(_Temp_ptr, _Uninitialized_move_unchecked(_First, _Mid, _Temp_ptr));
        Ref<TItem> _Left_first = _Temp_ptr;
        Ref<TItem> _Left_last = _Backout._Last - 1; // avoid a compare with the highest element
        _First.Set(_Mid.Get()); // the lowest element is now in position
        ++_First;
        ++_Mid;
        for (; ; )
        {
            if (_Pred.Compare(_Mid.Get(), _Left_first.Get()))
            {
                // take element from the right partition
                _First.Set(_Mid.Get());
                ++_First;
                ++_Mid;
                if (_Mid == _Last)
                {
                    _Move_unchecked(_Left_first, _Backout._Last, _First); // move any tail (and the highest element)
                    return;
                }
            }
            else
            {
                // take element from the left partition
                _First.Set(_Left_first.Get());
                ++_First;
                ++_Left_first;
                if (_Left_first == _Left_last)
                {
                    // move the remaining right partition and highest element, since *_Left_first is highest
                    _Move_unchecked(_Mid, _Last, _First).Set(_Left_last.Get());
                    return;
                }
            }
        }
    }

    private static void _Inplace_merge_buffer_right<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Mid,
        Ref<TItem> _Last,
        Ref<TItem> _Temp_ptr,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // move the range [_Mid, _Last) to _Temp_ptr, and merge it with [_First, _Mid) to _Last
        // usual invariants apply
        _Uninitialized_backout<TItem> _Backout = new(_Temp_ptr, _Uninitialized_move_unchecked(_Mid, _Last, _Temp_ptr));
        (--_Last).Set((--_Mid).Get()); // move the highest element into position
        Ref<TItem> _Right_first = _Temp_ptr;
        Ref<TItem> _Right_last = _Backout._Last - 1;
        --_Mid;
        for (; ; )
        {
            if (_Pred.Compare(_Right_last.Get(), _Mid.Get()))
            {
                // merge from the left partition
                (--_Last).Set(_Mid.Get());
                if (_First == _Mid)
                {
                    (--_Last).Set(_Right_last.Get()); // to make [_Right_first, _Right_last) a half-open range
                    _Move_backward_unchecked(_Right_first, _Right_last, _Last); // move any head (and lowest element)
                    return;
                }

                --_Mid;
            }
            else
            {
                // merge from the right partition
                (--_Last).Set(_Right_last.Get());
                --_Right_last;
                if (_Right_first == _Right_last)
                {
                    // we can't compare with *_Right_first, but we know it is lowest
                    (--_Last).Set(_Mid.Get()); // restore half-open range [_First, _Mid)
                    _Move_backward_unchecked(_First, _Mid, _Last);
                    _First.Set(_Right_first.Get());
                    return;
                }
            }
        }
    }

    private static void _Buffered_inplace_merge_divide_and_conquer<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Mid,
        Ref<TItem> _Last,
        nuint _Count1,
        nuint _Count2,
        Ref<TItem> _Temp_ptr,
        nuint _Capacity,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // merge sorted [_First, _Mid) with sorted [_Mid, _Last)
        // usual invariants apply
        if (_Count1 <= _Count2)
        {
            nuint _Count1n = _Count1 >> 1; // shift for codegen
            Ref<TItem> _Firstn = _First + _Count1n;
            Ref<TItem> _Lastn = lower_bound(_Mid, _Last, _Firstn.Get(), _Pred);
            _Buffered_inplace_merge_divide_and_conquer2(
                _First, _Mid, _Last, _Count1, _Count2, _Temp_ptr, _Capacity, _Pred,
                _Firstn, _Lastn, _Count1n, Ref<TItem>.Distance(_Mid, _Lastn));
        }
        else
        {
            nuint _Count2n = _Count2 >> 1; // shift for codegen
            Ref<TItem> _Lastn = _Mid + _Count2n;
            Ref<TItem> _Firstn = upper_bound(_First, _Mid, _Lastn.Get(), _Pred);
            _Buffered_inplace_merge_divide_and_conquer2(
                _First, _Mid, _Last, _Count1, _Count2, _Temp_ptr, _Capacity, _Pred,
                _Firstn, _Lastn, Ref<TItem>.Distance(_First, _Firstn), _Count2n);
        }
    }

    private static Ref<TItem> lower_bound<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Last,
        TItem _Val,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // find first element not before _Val
        _Adl_verify_range(_First, _Last);
        Ref<TItem> _UFirst = _First;
        nuint _Count = Ref<TItem>.Distance(_UFirst, _Last);

        while (0 < _Count)
        {
            // divide and conquer, find half that contains answer
            nuint _Count2 = _Count / 2;
            Ref<TItem> _UMid = _UFirst + _Count2;
            if (_Pred.Compare(_UMid.Get(), _Val))
            {
                // try top half
                _UFirst = _UMid + 1;
                _Count -= _Count2 + 1;
            }
            else
            {
                _Count = _Count2;
            }
        }

        _First = _UFirst;
        return _First;
    }

    private static Ref<TItem> upper_bound<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Last,
        TItem _Val,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // find first element that _Val is before
        _Adl_verify_range(_First, _Last);
        Ref<TItem> _UFirst = _First;
        nuint _Count = Ref<TItem>.Distance(_UFirst, _Last);

        while (0 < _Count)
        { 
            // divide and conquer, find half that contains answer
            nuint _Count2 = _Count / 2;
            Ref<TItem> _UMid = _UFirst + _Count2;
            if (_Pred.Compare(_Val, _UMid.Get()))
            {
                _Count = _Count2;
            }
            else
            { 
                // try top half
                _UFirst = _UMid + 1;
                _Count -= _Count2 + 1;
            }
        }

        _First = _UFirst;
        return _First;
    }

    private static void _Buffered_inplace_merge_divide_and_conquer2<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Mid,
        Ref<TItem> _Last,
        nuint _Count1,
        nuint _Count2,
        Ref<TItem> _Temp_ptr,
        nuint _Capacity,
        TComparer _Pred,
        Ref<TItem> _Firstn,
        Ref<TItem> _Lastn,
        nuint _Count1n,
        nuint _Count2n)
        where TComparer : IComparer<TItem>
    {
        // common block of _Buffered_inplace_merge_divide_and_conquer, below
        Ref<TItem> _Midn = _Buffered_rotate_unchecked(_Firstn, _Mid, _Lastn, _Count1 - _Count1n, _Count2n,
            _Temp_ptr, _Capacity); // rearrange middle

        _Buffered_inplace_merge_unchecked(
            _First, _Firstn, _Midn, _Count1n, _Count2n, _Temp_ptr, _Capacity, _Pred); // merge each new part

        _Buffered_inplace_merge_unchecked(_Midn, _Lastn, _Last, _Count1 - _Count1n,
            _Count2 - _Count2n, _Temp_ptr, _Capacity, _Pred);
    }

    private static Ref<T> _Buffered_rotate_unchecked<T>(
        Ref<T> _First,
        Ref<T> _Mid,
        Ref<T> _Last,
        nuint _Count1,
        nuint _Count2,
        Ref<T> _Temp_ptr,
        nuint _Capacity)
    {
        // rotate [_First, _Last) using temp buffer
        // precondition: _Count1 == distance(_First, _Mid)
        // precondition: _Count2 == distance(_Mid, _Last)
        if (_Count1 == 0)
        {
            return _Last;
        }

        if (_Count2 == 0)
        {
            return _First;
        }

        if (_Count1 <= _Count2 && _Count1 <= _Capacity)
        {
            // buffer left range, then copy parts
            _Uninitialized_backout<T> _Backout = new(
                _Temp_ptr, _Uninitialized_move_unchecked(_First, _Mid, _Temp_ptr));
            Ref<T> _New_mid = _Move_unchecked(_Mid, _Last, _First);
            _Move_unchecked(_Backout._First, _Backout._Last, _New_mid);
            return _New_mid; // _Backout destroys elements in temporary buffer
        }

        if (_Count2 <= _Capacity)
        {
            // buffer right range, then copy parts
            _Uninitialized_backout<T> _Backout = new(
                _Temp_ptr, _Uninitialized_move_unchecked(_Mid, _Last, _Temp_ptr));
            _Move_backward_unchecked(_First, _Mid, _Last);
            return _Move_unchecked(_Backout._First, _Backout._Last, _First); // ditto _Backout destroys elements
        }

        // buffer too small, rotate in place
        return rotate(_First, _Mid, _Last);
    }

    private static Ref<T> rotate<T>(
        Ref<T> _First,
        Ref<T> _Mid,
        Ref<T> _Last)
    {
        // exchange the ranges [_First, _Mid) and [_Mid, _Last)
        // that is, rotates [_First, _Last) left by distance(_First, _Mid) positions
        // returns the iterator pointing at *_First's new home
        _Adl_verify_range(_First, _Mid);
        _Adl_verify_range(_Mid, _Last);
        var _UFirst = _First;
        var _UMid = _Mid;
        var _ULast = _Last;
        if (_UFirst == _UMid)
        {
            return _Last;
        }

        if (_UMid == _ULast)
        {
            return _First;
        }

        reverse(_UFirst, _UMid);
        reverse(_UMid, _ULast);
        reverse(_UFirst, _ULast);
        _First = _UFirst + Ref<T>.Distance(_UMid, _ULast);

        return _First;
    }

    private static void reverse<T>(Ref<T> _First, Ref<T> _Last)
    {
        // reverse elements in [_First, _Last)
        _Adl_verify_range(_First, _Last);
        Ref<T> _UFirst = _First;
        Ref<T> _ULast = _Last;

        for (; _UFirst != _ULast && _UFirst != --_ULast; ++_UFirst)
        {
            swap(_UFirst, _ULast);
        }
    }

    private static void swap<T>(Ref<T> _Left, Ref<T> _Right)
    {
        T left = _Left.Get();
        _Left.Set(_Right.Get());
        _Right.Set(left);
    }

    private static Ref<T> _Uninitialized_move_unchecked<T>(Ref<T> _First, Ref<T> _Last, Ref<T> _Dest)
    {
        // move [_First, _Last) to raw [_Dest, ...)

        // TODO:
        //if (!RuntimeHelpers.IsReferenceOrContainsReferences<TItem>())
        //{
        //    return _Copy_memmove(_First, _Last, _Dest);
        //}

        _Uninitialized_backout<T> _Backout = new(_Dest);
        for (; _First != _Last; ++_First)
        {
            _Backout._Emplace_back(_First.Get());
        }

        return _Backout._Release();
    }

    private static void _Rotate_one_right<T>(Ref<T> _First, Ref<T> _Mid, Ref<T> _Last)
    {
        // exchanges the range [_First, _Mid) with [_Mid, _Last)
        // pre: distance(_Mid, _Last) is 1
        T _Temp = _Mid.Get();
        _Move_backward_unchecked(_First, _Mid, _Last);
        _First.Set(_Temp);
    }

    private static void _Rotate_one_left<T>(Ref<T> _First, Ref<T> _Mid, Ref<T> _Last)
    {
        // exchanges the range [_First, _Mid) with [_Mid, _Last)
        // pre: distance(_First, _Mid) is 1
        T _Temp = _First.Get();
        _Move_unchecked(_Mid, _Last, _First).Set(_Temp);
    }

    private static Ref<TItem> _Insertion_sort_unchecked<TItem, TComparer>(
        Ref<TItem> _First,
        Ref<TItem> _Last,
        TComparer _Pred)
        where TComparer : IComparer<TItem>
    {
        // insertion sort [_First, _Last)
        if (_First != _Last)
        {
            for (Ref<TItem> _Mid = _First; ++_Mid != _Last;)
            {
                // order next element
                Ref<TItem> _Hole = _Mid;
                TItem _Val = _Mid.Get();

                if (_Pred.Compare(_Val, _First.Get()))
                {
                    // found new earliest element, move to front
                    _Move_backward_unchecked(_First, _Mid, ++_Hole);
                    _First.Set(_Val);
                }
                else
                {
                    // look for insertion point after first
                    for (Ref<TItem> _Prev = _Hole; _Pred.Compare(_Val, (--_Prev).Get()); _Hole = _Prev)
                    {
                        _Hole.Set(_Prev.Get()); // move hole down
                    }

                    _Hole.Set(_Val); // insert element in hole
                }
            }
        }
        return _Last;
    }

    private static Ref<T> _Move_backward_unchecked<T>(Ref<T> _First, Ref<T> _Last, Ref<T> _Dest)
    {
        // move [_First, _Last) backwards to [..., _Dest)
        // note: _Move_backward_unchecked has callers other than the move_backward family

        // TODO:
        //if (!RuntimeHelpers.IsReferenceOrContainsReferences<T>())
        //{
        //    nuint distance = Ref<T>.Distance(_First, _Last);
        //    return Unsafe.CopyBlockUnaligned(ref _Dest.Value, ref _First.Value, distance);
        //}

        while (_First != _Last)
        {
            (--_Dest).Set((--_Last).Get());
        }

        return _Dest;
    }

    private static Ref<T> _Move_unchecked<T>(Ref<T> _First, Ref<T> _Last, Ref<T> _Dest)
    {
        // move [_First, _Last) to [_Dest, ...)
        // note: _Move_unchecked has callers other than the move family

        // TODO:
        //if (!RuntimeHelpers.IsReferenceOrContainsReferences<T>())
        //{
        //    return _Copy_memmove(_First, _Last, _Dest);
        //}

        for (; _First != _Last; ++_Dest, ++_First)
        {
            _Dest.Set(_First.Get());
        }

        return _Dest;
    }

    private static void _Adl_verify_range<T>(Ref<T> first, Ref<T> last)
    {
        if (Unsafe.IsAddressLessThan(ref last.Value, ref first.Value))
        {
            static void Throw()
            {
                throw new ArgumentException("The last reference is before the first.");
            }

            Throw();
        }
    }

    private ref struct _Uninitialized_backout<TItem>
    {
        // struct to undo partially constructed ranges in _Uninitialized_xxx algorithms
        public Ref<TItem> _First;
        public Ref<TItem> _Last;

        public _Uninitialized_backout(Ref<TItem> _Dest) : this(_Dest, _Dest)
        {
        }

        public _Uninitialized_backout(Ref<TItem> _First_, Ref<TItem> _Last_)
        {
            _First = _First_;
            _Last = _Last_;
        }

        public void _Emplace_back(TItem _Vals)
        {
            // construct a new element at *_Last and increment
            _Last.Set(_Vals);
            ++_Last;
        }

        public Ref<TItem> _Release()
        {
            // suppress any exception handling backout and return _Last
            _First = _Last;
            return _Last;
        }
    }
}
