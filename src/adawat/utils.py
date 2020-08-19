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
