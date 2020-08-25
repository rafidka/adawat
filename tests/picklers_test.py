from adawat.picklers import FilePickler


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
