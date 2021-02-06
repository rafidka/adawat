from adawat.text import splittext


def test_splittext():
    lines = list(splittext(iter([
        'this is the first line\nthis is the ',
        'second line\nthis is the third line'
    ]), sep='\n'))

    assert lines == [
        'this is the first line',
        'this is the second line',
        'this is the third line'
    ]
