import pandas as pd

from training_models import emotion_recognition_training
from training_models.cyclegan import get_model, prepare_dataset
import numpy as np
import os

emotion_recognition_training.size = 64
emotion_recognition_training.classes = 8
recognizer_model = emotion_recognition_training.get_model()
gan_model, _, _ = get_model()

recognizer_model.load_weights('model_progression/train/32/model_checkpoint.h5')

emotion_dict = {
    0: 'anger',
    1: 'contempt',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'neutral',
    6: 'sadness',
    7: 'surprise'
}

folder_path = '/Volumes/Pegasus_R4i/cycle_gan'

results = []

for item in os.listdir(folder_path):
    emotion1 = item.split('_')[0]
    emotion2 = item.split('_')[1]
    if os.path.isdir(os.path.join(folder_path, item)) and \
            ((emotion1 == 'disgust') or (emotion1 == 'fear') or (emotion1 == 'contempt')):
        n = 1
        emotion1_test_generator, emotion2_test_generator, _, _ = prepare_dataset(emotion1, emotion2, testing=True)
        emotion1_images = emotion1_test_generator.next()
        emotion2_images = emotion2_test_generator.next()


        def evaluate_performance(predictions, emotion1, emotion2):
            yhat = recognizer_model.predict(predictions)
            predicted_classes = np.argmax(yhat, axis=1)
            true_classes = [list(emotion_dict.keys())[list(emotion_dict.values()).index(emotion1)]] * len(
                predicted_classes)
            percent_correct = np.sum(predicted_classes == true_classes) / len(predicted_classes)
            print(emotion1 + ' to ' + emotion2, n, percent_correct)
            results.append([emotion1 + ' to ' + emotion2, n, percent_correct])

        while True:
            gan_model.load_weights(os.path.join(folder_path, item, 'checkpoints', 'cyclegan_checkpoints_%03d' % n))
            predictions_emot1_to_emot2 = gan_model.gen_G.predict(emotion1_images)
            predictions_emot2_to_emot1 = gan_model.gen_F.predict(emotion2_images)
            predictions_emot1_to_emot2 = (predictions_emot1_to_emot2 * 127.5 + 127.5) / 255
            predictions_emot2_to_emot1 = (predictions_emot2_to_emot1 * 127.5 + 127.5) / 255

            evaluate_performance(predictions_emot1_to_emot2, emotion1, emotion2)
            evaluate_performance(predictions_emot2_to_emot1, emotion2, emotion1)

            n += 1
            if not os.path.isfile(os.path.join(folder_path, item, 'checkpoints', 'cyclegan_checkpoints_%03d.index' % n)):
                df_results = pd.DataFrame(data=results, columns=['folder', 'iteration', 'results'])
                df_results.to_pickle('df_results_%s.pkl' % item)
                break
