from typing import Iterable, Optional
from itertools import islice


class GeneratorWithLen(object):
    """
    With generators that doesn't have support for the `len` method but the
    length can be calculated in advance, e.g. a yield in a for-loop, this
    class adds support for the `len` method.

    Keyword arguments:
    gen -- The generator or a callable that returns a generator.
    length -- The length of the iterable.
    """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        if callable(self.gen):
            return self.gen()
        else:
            return self.gen


class NamedIteratorSlice(object):
    """
    This class builds an iterator that has a custom name returned when str() or
    repr() is called on the instance. This is useful when trying to generate a
    unique signature that identifies a set of arguments through the use of the
    serialization.object_sig() method. Simply using Python`s islice() method would
    generate an output like `<itertools.islice object at 0x7fc20a5473b0>` when
    calling the str() or repr() methods.

    The benefit of such an iterator is when trying to save the state of an object
    based on a certain set of arguments that identifies it, one of them being an
    iterator. For example, if you are doing some expensive operations on a large
    file which you cannot load into memory, you are likely to use an iterable of
    some sort to process the files in chunks. To avoido repeating the same
    expensive operations, you might want to cache the result somewhere. To
    uniquely generate a key for caching, it is necessary to be able to uniquely
    identify the iterable, for which this class can be handy.
    """

    def __init__(self,
                 name: str,
                 iterable: Iterable,
                 start: Optional[int] = None,
                 stop: Optional[int] = None,
                 step: Optional[int] = None):
        self.iterable = islice(iterable, start, stop, step)
        self.name = name

    def __iter__(self):
        return self.iterable

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name


def named_islice(name, iterable, *args):
    """
    Like Python`s itertools.islice but attach a custom name to be returned
    when calling the str() or repr() on the returned instance. See the
    NamedIteratorSlice class for more information on how this is useful.

    Arguments:
    name -- the custom name to return when str() or repr() is called.
    iterable -- the iterable to slice.
    start, stop, step -- same as the start, stop, and step arguments of islice()

    Returns:
    An iterator that behaves exactly like islice() but with a custom name.
    """

    if len(args) == 1:
        stop, = args
        return NamedIteratorSlice(name, iterable, 0, stop)
    elif len(args) == 2:
        start, stop = args
        return NamedIteratorSlice(name, iterable, start, stop)
    elif len(args) == 3:
        start, stop, step = args
        return NamedIteratorSlice(name, iterable, start, stop, step)
    else:
        raise RuntimeError("""Invalid number of arguments. Either use:

 named_slice(name, iterable, stop)

 or:
 
 named_slice(name, iterable, start, stop[, step]""")
