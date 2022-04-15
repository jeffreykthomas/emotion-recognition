import unittest
from training_models import emotion_recognition_training


class TestRecognitionTraining(unittest.TestCase):

    def test_training(self):
        print('Testing to ensure no errors in training')
        emotion_recognition_training.epochs = 1
        emotion_recognition_training.testing = True
        emotion_recognition_training.train_model()


if __name__ == '__main__':
    unittest.main()
