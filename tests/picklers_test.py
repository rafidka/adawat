from adawat.picklers import MemoryPickler, FilePickler, ObjectNotFoundError


def test_MemoryPickler():
    pickler = MemoryPickler()

    some_obj = {
        'some_field': 'some_value'
    }

    some_obj_id = 'this-is-a-relatively-unique-id'

    # Dump the object
    pickler.dump(some_obj_id, some_obj)

    # Load the object
    some_obj_loaded = pickler.load(some_obj_id)

    assert some_obj == some_obj_loaded

    # Delete the object
    pickler.delete(some_obj_id)

    # Reload the object and
    try:
        pickler.load(some_obj_id)
        assert False
    except ObjectNotFoundError:
        # We are good.
        pass


def test_FilePickler():
    pickler = FilePickler()

    some_obj = {
        'some_field': 'some_value'
    }

    some_obj_id = 'this-is-a-relatively-unique-id'

    # Dump the object
    pickler.dump(some_obj_id, some_obj)

    # Load the object
    some_obj_loaded = pickler.load(some_obj_id)

    assert some_obj == some_obj_loaded

    # Delete the object
    pickler.delete(some_obj_id)

    # Reload the object and
    try:
        pickler.load(some_obj_id)
        assert False
    except ObjectNotFoundError:
        # We are good.
        pass
