import pandas as pd

from training_models import emotion_recognition_training
from training_models.cyclegan import get_model, prepare_dataset
import numpy as np
import os

size = 128
classes = 8
recognizer_model = emotion_recognition_training.get_model(image_size=size, num_classes=classes)
gan_model, _, _ = get_model(input_img_size=(size, size, 3))

recognizer_model.load_weights('results/recognizer/128_reduced_False_augmented_False_8cat/model.h5')

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

folder_path = '/Volumes/Pegasus_R4i/cycle_gan/128_weights'

results = []

for item in os.listdir(folder_path):
    emotion1 = item.split('_')[0]
    emotion2 = item.split('_')[1]
    if os.path.isdir(os.path.join(folder_path, item)):
        n = 1
        emotion1_test_generator, emotion2_test_generator, _, _ = prepare_dataset(emotion1,
                                                                                 emotion2,
                                                                                 image_size=size,
                                                                                 testing=True)
        emotion1_images = emotion1_test_generator.next()
        emotion2_images = emotion2_test_generator.next()


        def evaluate_performance(predictions, emotion_1, emotion_2):
            yhat = recognizer_model.predict(predictions)
            predicted_classes = np.argmax(yhat, axis=1)
            transformed_classes = [list(emotion_dict.keys())[list(emotion_dict.values()).index(emotion_2)]] * len(
                predicted_classes)
            percent_correct = np.sum(predicted_classes == transformed_classes) / len(predicted_classes)
            print(emotion_1 + ' to ' + emotion_2, n, percent_correct)
            results.append([emotion_1 + ' to ' + emotion_2, n, percent_correct])

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
                df_results.to_pickle('pickles/gan_eval/df_results_%s.pkl' % item)
                results = []
                break
