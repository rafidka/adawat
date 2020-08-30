import pytest
from itertools import islice
from adawat.utils import named_islice


@pytest.mark.parametrize("test_list,start,stop,step", [
    (list(range(0, 10)), None, None, None),
    (list(range(0, 10)), 1, None, None),
    (list(range(0, 10)), 0, 5, None),
    (list(range(0, 10)), 0, 10, 2)
])
def test_named_islice(test_list, start, stop, step):
    test_name = 'test_name'

    test_slice = islice(test_list, start, stop, step)
    test_named_slice = named_islice(test_name, test_list, start, stop, step)

    assert list(test_slice) == list(test_named_slice)
    assert str(test_named_slice) == test_name
    assert repr(test_named_slice) == test_name
