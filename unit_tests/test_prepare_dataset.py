import unittest
import numpy as np
from training_models.emotion_recognition_training import prepare_dataset

train_set, _, _, _ = prepare_dataset(7)
images, _ = train_set.next()
img = images[0]


class TestPrepareDatasetMethods(unittest.TestCase):

    def test_image_size(self):
        print('testing image size')
        self.assertEqual((224, 224, 3), img.shape)

    def test_pixel_range(self):
        print('testing pixel range')
        self.assertTrue(np.all((img <= 1) & (0 <= img)))

    def test_num_type(self):
        print('testing pixel type')
        self.assertTrue(img.dtype is np.dtype(np.float32))


if __name__ == '__main__':
    unittest.main()
