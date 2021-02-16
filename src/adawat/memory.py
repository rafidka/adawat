from collections import deque
from itertools import chain
from reprlib import repr
from sys import getsizeof, stderr


# Copied from https://code.activestate.com/recipes/577504/ with minor modifications.
def gettotalsizeof(object, handlers={}, verbose=False):
    """
    Returns the approximate memory footprint of an object and its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses: tuple, list, deque, dict, set and frozenset.

    To support other containers, add handlers to iterate over their contents,
    for example:

    gettotalsizeof(my_object,
        handlers = {
            SomeContainerClass: iter,
            OtherContainerClass: OtherContainerClass.get_elements
        }
    )

    Keyword arguments:
    object -- The object.
    handlers -- (Optional) Custom handlers for iterating over the elements of
        containers. Use this if your object uses containers other than the ones
        mentioned above.
    verbose -- (Optional) Set to true if you want to print debugging info.

    Returns:
    Approximate memory footprint of the object and its contents.
    """

    # For iterating over the elements of a dictionary.
    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    # estimate sizeof object without __sizeof__
    default_size = getsizeof(0)

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(object)
