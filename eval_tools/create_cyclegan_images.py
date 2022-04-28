import training_models.cyclegan as cyclegan
import numpy as np
import pandas as pd
import matplotlib

from tensorflow.keras.preprocessing.image import ImageDataGenerator

weight_file_root = '/Volumes/Pegasus_R4i/cycle_gan/128_weights/'
weights = {
    'anger from neutral': (weight_file_root + 'anger_neutral/checkpoints/cyclegan_checkpoints_003', 'F'),
    'contempt from happiness': (weight_file_root + 'contempt_happiness/checkpoints/cyclegan_checkpoints_002', 'F'),
    'disgust from happiness': (weight_file_root + 'disgust_happiness/checkpoints/cyclegan_checkpoints_007', 'F'),
    'fear from happiness': (weight_file_root + 'fear_happiness/checkpoints/cyclegan_checkpoints_003', 'F'),
    'happiness from sadness': (weight_file_root + 'sadness_happiness/checkpoints/cyclegan_checkpoints_164', 'G'),
    'neutral from anger': (weight_file_root + 'anger_neutral/checkpoints/cyclegan_checkpoints_002', 'G'),
    'sadness from happiness': (weight_file_root + 'sadness_happiness/checkpoints/cyclegan_checkpoints_233', 'F'),
    'surprise from neutral': (weight_file_root + 'neutral_surprise/checkpoints/cyclegan_checkpoints_004', 'G')
}

image_size = 128


def create_gan_model(weight, model):
    gan_model, _, _ = cyclegan.get_model(input_img_size=(image_size, image_size, 3))
    gan_model.load_weights(weight).expect_partial()
    if model == 'G':
        return gan_model.gen_G
    else:
        return gan_model.gen_F


def prepare_dataset(emotion):

    df_train = pd.read_pickle('pickles/image_eval/df_train_ambiguous_85.pkl')

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
        target_size=(image_size, image_size),
        batch_size=5000,
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
    train_generator, df_train_dataframe = prepare_dataset(emotion2)
    prediction = gan_model.predict(train_generator.next())
    prediction = (prediction * 127.5 + 127.5) / 255

    for i, img in enumerate(prediction):
        print('Making Prediction ' + str(j))
        filename = 'data/cyclegan_images/cyclegan_img_%03d.jpg' % j
        matplotlib.pyplot.imsave(filename, img)
        results_array.append([filename, emotion1.capitalize(), df_train_dataframe.iloc[i].x_col])
        j += 1

df_results = pd.DataFrame(data=results_array, columns=['x_col', 'y_col', 'original_filename'])
df_results.to_pickle('pickles/df_cyclegan_images.pkl')
