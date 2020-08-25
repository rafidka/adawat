import pytest
from adawat.serialization import object_sig, object_id, get_state, \
    update_state, load_object, save_object, stateful
from adawat.picklers import FilePickler


class SomeClass():
    def __init__(self, *args, **kwargs):
        pass


class Test_object_sig:
    def test__object_sig(self):
        actual_sig = object_sig(SomeClass())
        expected_sig = "SomeClass;;;<no_args>;;;<no_kwargs>"
        assert actual_sig == expected_sig

    def test__object_sig_with_args(self):
        actual_sig = object_sig(SomeClass(), 1, 2, 3)
        expected_sig = "SomeClass;;;1;;2;;3;;;<no_kwargs>"
        assert actual_sig == expected_sig

    def test__object_sig_with_kwargs(self):
        actual_sig = object_sig(SomeClass(), key1='value1', key2='value2')
        expected_sig = "SomeClass;;;<no_args>;;;key1=value1;;key2=value2"
        assert actual_sig == expected_sig

    def test__object_sig_with_args_and_kwargs(self):
        actual_sig = object_sig(SomeClass(), 1, 2, 3,
                                key1='value1', key2='value2')
        expected_sig = "SomeClass;;;1;;2;;3;;;key1=value1;;key2=value2"
        assert actual_sig == expected_sig


class Test_object_id:
    def test__object_id(self):
        actual_id = object_id(SomeClass())
        expected_id = "5e132cc9e554c17346a190b6b97ce64f7a3f8e4570e5404f76ab06773885bb09"
        assert actual_id == expected_id

    def test__object_id_with_args(self):
        actual_id = object_id(SomeClass(), 1, 2, 3)
        expected_id = "d846994a24f5c07c3e65a9be30e06314feae2db268920b7228169f1c7118029a"
        assert actual_id == expected_id

    def test__object_id_with_kwargs(self):
        actual_id = object_id(
            SomeClass(), key1='value1', key2='value2')
        expected_id = "84b373381e136591a5f1d1d9be7ec8d088ae22a6181db475092e2347e6efa95a"
        assert actual_id == expected_id

    def test__object_id_with_args_and_kwargs(self):
        actual_id = object_id(SomeClass(), 1, 2, 3,
                              key1='value1', key2='value2')
        expected_id = "89d478b3e774453d3d6dda465d04767951c4a7f22996a4bbc692e353de7f3735"
        assert actual_id == expected_id


class Test_get_state:
    def test(self):
        obj = SomeClass()
        obj.name = 'Test Name'
        obj.value = 123

        state = get_state(obj, ['name', 'value'])
        assert state == ['Test Name', 123]


class Test_update_state:
    def test(self):
        obj = SomeClass()
        obj.name = 'Test Name'
        obj.value = 123

        update_state(obj, ['name', 'value'], ['Test Name - Updated', 12345])

        assert obj.name == 'Test Name - Updated'
        assert obj.value == 12345


class Test_save_load_object:
    def test(self):
        obj = SomeClass()
        obj.name = 'Test Name'
        obj.value = 123
        obj.unsaved_value = 102030

        filepickler = FilePickler()
        obj_id = object_id(obj, ['name', 'value'])
        save_object(obj, obj_id, ['name', 'value'], filepickler)

        obj2 = SomeClass()
        obj2.unsaved_value = 405060
        load_object(obj2, obj_id, ['name', 'value'], filepickler)

        assert obj.name == obj2.name
        assert obj.value == obj2.value
        assert obj2.unsaved_value == 405060


class Test_stateful_save_method:
    def test_conflicting_method_name(self):
        with pytest.raises(RuntimeError):
            @ stateful(save_state_method_name='save_state')
            class TestWithConflictingStateMethod:
                def __init__(self):
                    self.state_dict = {}

                def save_state(self):
                    pass
            TestWithConflictingStateMethod()  # to avoid unused class warning

    def test_nonconflicting_method_name(self):
        try:
            @ stateful(save_state_method_name='save_state')
            class TestWithoutConflictingStateMethod:
                def __init__(self):
                    self.state_dict = {}

                def save_state_other(self):
                    pass
            assert hasattr(TestWithoutConflictingStateMethod, 'save_state')
            assert callable(TestWithoutConflictingStateMethod.save_state)
        except Exception:
            self.fail()

    def test_method_behaviour(self):
        @ stateful(attrs=['state_dict'])
        class StatefulObject():
            def __init__(self):
                self.state_dict = {}

        some_state = {
            'test_key': 'test_value'
        }

        obj = StatefulObject()
        obj.state_dict = some_state
        obj.save_state()

        # re-instantiate the object and ensure the state is as expected
        obj = StatefulObject()
        assert obj.state_dict == some_state
