import unittest
import numpy as np
from collections import namedtuple
from drirlearning import utils

# Mockup = namedtuple('Mockup', 'x y')
# mu = Mockup(
#     x=[np.arange(0, 1200).reshape((100, 3, 4, 1))],
#     y=[np.arange(0, 900).reshape((100, 9))]
# )


class TestLoadData(unittest.TestCase):
    def test_cut(self):
        self.assertRaises(
            ValueError, utils.load_data, './_Input',
            cut_start=600,
            cut_end=400)


class TestCheckDataFormat(unittest.TestCase):
    def test_check_data_format(self):
        data = {'features': 0}
        self.assertRaises(TypeError, utils.check_data_format, data)
        data = {'labels': 0}
        self.assertRaises(TypeError, utils.check_data_format, data)
        data = {'fetures': 0, 'labels': 0}
        self.assertRaises(TypeError, utils.check_data_format, data)
        data = {
            'features': np.linspace(0, 1, 120).reshape((10, 3, 4, 1)),
            'labels': np.linspace(0, 1, 81).reshape((9, 9))}
        self.assertRaises(TypeError, utils.check_data_format, data)


class TestSplitData(unittest.TestCase):
    data = {
        'features': np.linspace(0, 1, 120).reshape((10, 3, 4, 1)),
        'labels': np.linspace(0, 1, 90).reshape((10, 9))}

    def test_value(self):
        self.assertRaises(ValueError, utils.split_data, self.data, split=-2.0)
        self.assertRaises(ValueError, utils.split_data, self.data, split=2.0)

    def test_return_shape(self):
        x_train, y_train, x_test, y_test = utils.split_data(self.data, split=0.8)
        self.assertEqual(x_train.shape[0], 8)
        self.assertEqual(y_train.shape[0], 8)
        self.assertEqual(x_test.shape[0], 2)
        self.assertEqual(y_test.shape[0], 2)


class TestShuffle(unittest.TestCase):
    def test_return_shape(self):
        x = np.arange(0, 1200).reshape((100, 3, 4, 1))
        y = np.arange(0, 900).reshape((100, 9))
        x, y = utils.shuffle(mu.x[0], mu.y[0])
        self.assertEqual(x.shape, (100, 3, 4, 1))
        self.assertEqual(y.shape, (100, 9))


class TestNextBatch(unittest.TestCase):
    def test_next_batch(self):
        x = np.arange(0, 1200).reshape((100, 3, 4, 1))
        y = np.arange(0, 900).reshape((100, 9))



if __name__ == '__main__':
    unittest.main()
