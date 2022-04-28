# import the necessary packages
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from training_models import cyclegan

emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
weight_file_root = '/Volumes/Pegasus_R4i/cycle_gan/128_weights/'
weights = {
    'anger to happiness': (weight_file_root + 'anger_happiness/checkpoints/cyclegan_checkpoints_171', 'G'),
    'contempt to happiness': (weight_file_root + 'contempt_happiness/checkpoints/cyclegan_checkpoints_004', 'G'),
    'disgust to happiness': (weight_file_root + 'disgust_happiness/checkpoints/cyclegan_checkpoints_006', 'G'),
    'fear to happiness': (weight_file_root + 'fear_happiness/checkpoints/cyclegan_checkpoints_004', 'G'),
    'neutral to anger': (weight_file_root + 'anger_neutral/checkpoints/cyclegan_checkpoints_003', 'F'),
    'happiness to neutral': (weight_file_root + 'neutral_happiness/checkpoints/cyclegan_checkpoints_003', 'F'),
    'sadness to happiness': (weight_file_root + 'sadness_happiness/checkpoints/cyclegan_checkpoints_164', 'G'),
    'surprise to neutral': (weight_file_root + 'neutral_surprise/checkpoints/cyclegan_checkpoints_001', 'F')
}
threshold = 85


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title, display=False):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    if display:
        # setup the figure
        fig = plt.figure(title)
        plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap=plt.cm.gray)
        plt.axis("off")
        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap=plt.cm.gray)
        plt.axis("off")
        # show the images
        plt.show()
    else:
        return m, s


def prep_fn(img):
    img = img.astype(np.float32)
    return (img / 127.5) - 1


def prepare_dataset(emotion):
    df_train = pd.read_pickle('pickles/df_train.pkl')
    first_train_dataframe = df_train[df_train['y_col'] == emotion.capitalize()]

    df_val = pd.read_pickle('pickles/df_val.pkl')
    first_val_dataframe = df_val[df_val['y_col'] == emotion.capitalize()]

    first_train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        preprocessing_function=prep_fn
    )

    first_train_generator = first_train_datagen.flow_from_dataframe(
        first_train_dataframe,
        x_col='x_col',
        y_col='y_col',
        target_size=(128, 128),
        batch_size=256,
        color_mode='rgb',
        shuffle=False,
        class_mode=None
    )

    first_val_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        preprocessing_function=prep_fn
    )

    first_val_generator = first_val_datagen.flow_from_dataframe(
        first_val_dataframe,
        x_col='x_col',
        y_col='y_col',
        target_size=(128, 128),
        batch_size=256,
        color_mode='rgb',
        shuffle=False,
        class_mode=None
    )

    return first_train_generator, first_val_generator


def create_gan_model(weight, model):
    gan_model, _, _ = cyclegan.get_model(input_img_size=(128, 128, 3))
    gan_model.load_weights(weight).expect_partial()
    if model == 'G':
        return gan_model.gen_G
    else:
        return gan_model.gen_F


def test_image_transformation():
    df_train = pd.read_pickle('pickles/df_train.pkl')
    df_train['ambiguous'] = False
    df_val = pd.read_pickle('pickles/df_val.pkl')
    df_val['ambiguous'] = False
    for (k, v) in weights.items():
        weight, model = v
        emotion = str(k).split(' ')[0]
        gan_model = create_gan_model(weight, model)
        first_train_generator, first_val_generator = prepare_dataset(emotion)
        generators = [first_train_generator, first_val_generator]

        for generator in generators:
            i = 1
            for img in generator:
                if i > len(generator):
                    df_train.to_pickle('pickles/image_eval/df_train_ambiguous_' + str(threshold) + '.pkl')
                    df_val.to_pickle('pickles/image_eval/df_val_ambiguous_' + str(threshold) + '.pkl')
                    break

                print('Making Predictions')
                predictions = gan_model.predict(img)
                predictions = (predictions * 127.5 + 127.5)
                img = (img * 127.5 + 127.5)
                for n, (prediction, image) in enumerate(zip(predictions, img)):
                    print('Comparing image #' + str(n))
                    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    m, s = compare_images(image, prediction, 'versus generated')
                    if s > (threshold / 100):
                        print('Adding image to array')
                        idx = (generator.batch_index - 1) * generator.batch_size
                        filename = generator.filenames[idx: idx + generator.batch_size or None][n]
                        df_train.loc[df_train.x_col == filename, 'ambiguous'] = True
                        df_val.loc[df_val.x_col == filename, 'ambiguous'] = True
                i += 1
                print(emotion + ', i: ' + str(i))


if __name__ == '__main__':
    test_image_transformation()
