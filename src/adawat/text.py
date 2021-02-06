from typing import Generator, Iterator


def splittext(text_generator: Iterator[str], sep: str) -> Generator[str, None, None]:
    """
    Given a text generator, this method split it by the given separator in a
    streaming fashion, i.e. it doesn't load the whole text into memory. This
    is useful when the user wants to read huge text line by line without having
    to fit the whole text into memory.

    Keyword arguments:
    text_generator -- the text generator.
    sep -- the separator to split the text by.

    Returns:
    A generator that yields the lines of the text.
    """
    last_line = ""
    try:
        while True:
            chunk = "".join([last_line, str(next(text_generator))])
            chunk_lines = chunk.split(sep)

            # Remove the last line and save it because it could be that part of
            # it is in the next text block.
            last_line = chunk_lines.pop()

            # Iterate through the lines and yield them.
            for line in chunk_lines:
                yield line
    except StopIteration:
        # Yield the last line and finish the iteration.
        yield last_line
