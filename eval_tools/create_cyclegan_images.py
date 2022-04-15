import training_models.cyclegan as cyclegan
import numpy as np
import pandas as pd
import matplotlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

weights = {
    'anger from neutral': ('/Volumes/Pegasus_R4i/cycle_gan/anger_neutral_64/checkpoints/cyclegan_checkpoints_180', 'F'),
    'contempt from happiness': (
        '/Volumes/Pegasus_R4i/cycle_gan/contempt_happiness_64/checkpoints/cyclegan_checkpoints_731', 'F'),
    'disgust from happiness': (
        '/Volumes/Pegasus_R4i/cycle_gan/disgust_happiness_64/checkpoints/cyclegan_checkpoints_721', 'F'),
    'fear from happiness': ('/Volumes/Pegasus_R4i/cycle_gan/fear_happiness_64/checkpoints/cyclegan_checkpoints_981', 'F'),
    'sadness from happiness': (
        '/Volumes/Pegasus_R4i/cycle_gan/happiness_sadness_64/checkpoints/cyclegan_checkpoints_216', 'G'),
    'surprise from neutral': (
        '/Volumes/Pegasus_R4i/cycle_gan/surprise_neutral_64/checkpoints/cyclegan_checkpoints_176', 'F')
}


def create_gan_model(weight, model):
    gan_model, _, _ = cyclegan.get_model()
    gan_model.load_weights(weight).expect_partial()
    if model == 'G':
        return gan_model.gen_G
    else:
        return gan_model.gen_F


def prepare_dataset(emotion):

    df_train = pd.read_pickle('pickles/df_train_ambiguous_85.pkl')

    def get_dataframe(emotion_name):
        return {
            'happiness': df_train[df_train['y_col'] == 'Happiness'],
            'sadness': df_train[df_train['y_col'] == 'Sadness'],
            'neutral': df_train[df_train['y_col'] == 'Neutral'],
            'anger': df_train[df_train['y_col'] == 'Anger'],
            'surprise': df_train[df_train['y_col'] == 'Surprise'],
            'contempt': df_train[df_train['y_col'] == 'Contempt'],
            'fear': df_train[df_train['y_col'] == 'Fear'],
            'disgust': df_train[df_train['y_col'] == 'Disgust']
        }[emotion_name]

    first_train = get_dataframe(emotion)
    first_train = first_train[:100]

    def prep_fn(img):
        img = img.astype(np.float32)
        return (img / 127.5) - 1

    first_train_datagen = ImageDataGenerator(
        preprocessing_function=prep_fn
    )

    first_train_generator = first_train_datagen.flow_from_dataframe(
        first_train,
        x_col='x_col',
        y_col='y_col',
        shuffle=False,
        target_size=(64, 64),
        batch_size=500,
        color_mode='rgb',
        class_mode=None
    )

    return first_train_generator, first_train


results_array = []
j = 0

for (k, v) in weights.items():
    weight, model = v
    emotion1 = str(k).split(' ')[0]
    emotion2 = str(k).split(' ')[2]
    gan_model = create_gan_model(weight, model)
    first_train_dataframe, df_first_train = prepare_dataset(emotion2)
    prediction = gan_model.predict(first_train_dataframe.next())
    prediction = (prediction * 127.5 + 127.5) / 255

    for i, img in enumerate(prediction):
        print('Making Prediction ' + str(j))
        filename = 'data/cyclegan_images/cyclegan_additional_two_img_%03d.jpg' % j
        matplotlib.pyplot.imsave(filename, img)
        results_array.append([filename, df_first_train.iloc[i].x_col, emotion1.capitalize()])
        j += 1

df_results = pd.DataFrame(data=results_array, columns=['x_col', 'y_col', 'original_filename'])
df_results.to_pickle('pickles/df_cyclegan_images_additional_two.pkl')
