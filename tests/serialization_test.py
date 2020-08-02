import unittest
from adawat.serialization import object_sig, object_filepath, get_state, \
    update_state, load_object, save_object


class TestClass():
    def __init__(self, *args, **kwargs):
        pass


class Test_object_sig(unittest.TestCase):
    def test__object_sig(self):
        actual_sig = object_sig(TestClass())
        expected_sig = "TestClass;;;<no_args>;;;<no_kwargs>"
        self.assertEqual(actual_sig, expected_sig)

    def test__object_sig_with_args(self):
        actual_sig = object_sig(TestClass(), 1, 2, 3)
        expected_sig = "TestClass;;;1;;2;;3;;;<no_kwargs>"
        self.assertEqual(actual_sig, expected_sig)

    def test__object_sig_with_kwargs(self):
        actual_sig = object_sig(TestClass(), key1='value1', key2='value2')
        expected_sig = "TestClass;;;<no_args>;;;key1=value1;;key2=value2"
        self.assertEqual(actual_sig, expected_sig)

    def test__object_sig_with_args_and_kwargs(self):
        actual_sig = object_sig(TestClass(), 1, 2, 3,
                                key1='value1', key2='value2')
        expected_sig = "TestClass;;;1;;2;;3;;;key1=value1;;key2=value2"
        self.assertEqual(actual_sig, expected_sig)


class Test_object_filepath(unittest.TestCase):
    def test__object_filepath(self):
        actual_filepath = object_filepath(TestClass())
        expected_filename = "f4f5939d459b2a53ae64d0a4652f82c44a0df8dfd78279ab00fc047a1776d140b.state"
        self.assertTrue(actual_filepath.endswith(expected_filename),
                        f'Expecting f{actual_filepath} to end with {expected_filename}')

    def test__object_filepath_with_args(self):
        actual_filepath = object_filepath(TestClass(), 1, 2, 3)
        expected_filename = "f87e9eb2d2ceaf3b1cb18b5fa58a115634618f6080dadddf1dbb6651c64658d22.state"
        self.assertTrue(actual_filepath.endswith(expected_filename),
                        f'Expecting f{actual_filepath} to end with {expected_filename}')

    def test__object_filepath_with_kwargs(self):
        actual_filepath = object_filepath(
            TestClass(), key1='value1', key2='value2')
        expected_filename = "f2af8230ad0795fb4bb544987a0166191869d807d678e70c3c003448b0913e79f.state"
        self.assertTrue(actual_filepath.endswith(expected_filename),
                        f'Expecting f{actual_filepath} to end with {expected_filename}')

    def test__object_filepath_with_args_and_kwargs(self):
        actual_filepath = object_filepath(TestClass(), 1, 2, 3,
                                          key1='value1', key2='value2')
        expected_filename = "fa3bebec3ba7f68d58bec89a5a74fa27e0d27146377162f50c2d6b7637c5007d0.state"
        self.assertTrue(actual_filepath.endswith(expected_filename),
                        f'Expecting f{actual_filepath} to end with {expected_filename}')


class Test_get_state(unittest.TestCase):
    def test(self):
        obj = TestClass()
        obj.name = 'Test Name'
        obj.value = 123

        state = get_state(obj, ['name', 'value'])
        self.assertEqual(state, ['Test Name', 123])


class Test_update_state(unittest.TestCase):
    def test(self):
        obj = TestClass()
        obj.name = 'Test Name'
        obj.value = 123

        update_state(obj, ['name', 'value'], ['Test Name - Updated', 12345])

        self.assertEqual(obj.name, 'Test Name - Updated')
        self.assertEqual(obj.value, 12345)


class Test_save_load_state(unittest.TestCase):
    def test(self):
        obj = TestClass()
        obj.name = 'Test Name'
        obj.value = 123
        obj.unsaved_value = 102030

        save_object(obj, ['name', 'value'])

        obj2 = TestClass()
        obj2.unsaved_value = 405060
        load_object(obj2, ['name', 'value'])

        self.assertEqual(obj.name, obj2.name)
        self.assertEqual(obj.value, obj2.value)
        self.assertEqual(obj2.unsaved_value, 405060)
