import unittest
import numpy as np
from collections import namedtuple
from drirlearning import utils

Mockup = namedtuple('Mockup', 'data x y')
mu = Mockup(
    data={
        'features': np.linspace(0, 1, 120).reshape((10, 3, 4, 1)),
        'labels': np.linspace(0, 1, 81).reshape((9, 9))},
    x=np.arange(0, 1200).reshape((100, 3, 4, 1)),
    y=np.arange(0, 900).reshape((100, 9))
)


class TestLoadData(unittest.TestCase):
    def test_cut(self):
        # Check if errors are raised when necessary
        self.assertRaises(
            ValueError, utils.load_data, './_Input',
            cut_start=600,
            cut_end=400)


class TestCheckDataFormat(unittest.TestCase):
    def test_check_data_format(self):
        # Check if errors are raised when necessary
        data = {'features': 0}
        self.assertRaises(TypeError, utils.check_data_format, data)
        data = {'labels': 0}
        self.assertRaises(TypeError, utils.check_data_format, data)
        data = {'fetures': 0, 'labels': 0}
        self.assertRaises(TypeError, utils.check_data_format, data)
        self.assertRaises(TypeError, utils.check_data_format, mu.data)


class TestSplitData(unittest.TestCase):
    data = {
        'features': np.linspace(0, 1, 120).reshape((10, 3, 4, 1)),
        'labels': np.linspace(0, 1, 90).reshape((10, 9))}

    def test_value(self):
        # Check if errors are raised when necessary
        self.assertRaises(ValueError, utils.split_data, self.data, split=-2.0)
        self.assertRaises(ValueError, utils.split_data, self.data, split=2.0)

    def test_return_shape(self):
        # Check if shape of corresponding input and output is equal
        x_train, y_train, x_test, y_test = utils.split_data(self.data, split=0.8)
        self.assertEqual(x_train.shape[0], 8)
        self.assertEqual(y_train.shape[0], 8)
        self.assertEqual(x_test.shape[0], 2)
        self.assertEqual(y_test.shape[0], 2)


class TestShuffle(unittest.TestCase):
    def test_return_shape(self):
        # Check if shape of corresponding input and output is equal
        x = np.arange(0, 1200).reshape((100, 3, 4, 1))
        y = np.arange(0, 900).reshape((100, 9))
        x, y = utils.shuffle(x, y)
        self.assertEqual(x.shape, (100, 3, 4, 1))
        self.assertEqual(y.shape, (100, 9))


class TestNextBatch(unittest.TestCase):
    x = np.arange(0, 1200).reshape((100, 3, 4, 1))
    y = np.arange(0, 900).reshape((100, 9))

    def test_next_batch(self):
        # Check if next batch is returned
        x_batch, y_batch = utils.next_batch(5, mu.x, mu.y)
        self.assertTrue(np.array_equal(x_batch, mu.x[:5, :, :, :]))
        self.assertTrue(np.array_equal(y_batch, mu.y[:5, :]))
        x_batch, y_batch = utils.next_batch(5, mu.x, mu.y)
        self.assertTrue(np.array_equal(x_batch, mu.x[5:10, :, :, :]))
        self.assertTrue(np.array_equal(y_batch, mu.y[5:10, :]))
        pass

    def test_return_shape(self):
        # Check if shape of corresponding input and output is equal
        x_batch, y_batch = utils.next_batch(20, mu.x, mu.y)
        self.assertEqual(x_batch.shape, (20, 3, 4, 1))
        self.assertEqual(y_batch.shape, (20, 9))



if __name__ == '__main__':
    unittest.main()
