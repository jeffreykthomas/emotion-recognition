# import the necessary packages
import pandas as pd
from tensorflow.keras.preprocessing.image.image import ImageDataGenerator
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from training_models import cyclegan

emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
weights = {
    'anger to happiness': ('cycle_gan/weights/cyclegan_anger_happiness', 'G'),
    'contempt to happiness': ('cycle_gan/weights/cyclegan_contempt_happiness', 'G'),
    'disgust to happiness': ('cycle_gan/weights/cyclegan_disgust_happiness', 'G'),
    'fear to happiness': ('cycle_gan/weights/cyclegan_fear_happiness', 'G'),
    'neutral to surprise': ('cycle_gan/weights/cyclegan_surprise_neutral', 'F'),
    'happiness to sadness': ('cycle_gan/weights/cyclegan_happiness_sadness_00', 'G'),
    'sadness to happiness': ('cycle_gan/weights/cyclegan_happiness_sadness_01', 'F'),
    'surprise to happiness': ('cycle_gan/weights/cyclegan_surprise_happiness', 'G')
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
    df_train = pd.read_pickle('../src/df_train.pkl')
    first_train_dataframe = df_train[df_train['y_col'] == emotion.capitalize()]

    df_val = pd.read_pickle('../src/training_models/df_val.pkl')
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
        target_size=(64, 64),
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
        target_size=(64, 64),
        batch_size=256,
        color_mode='rgb',
        shuffle=False,
        class_mode=None
    )

    return first_train_generator, first_val_generator


def create_gan_model(weight, model):
    gan_model, _, _ = cyclegan.get_model()
    gan_model.load_weights(weight).expect_partial()
    if model == 'G':
        return gan_model.gen_G
    else:
        return gan_model.gen_F


def test_image_transformation():
    results_array = []
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
                    df_results = pd.DataFrame(data=results_array, columns=['filename'])
                    df_results.to_pickle('df_' + emotion.lower() + '_' + generator.filenames[0].split('/')[2].split('_')[0] + '_ambiguous_' + str(threshold) + '.pkl')
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
                        results_array.append(generator.filenames[idx: idx + generator.batch_size or None][n])
                i += 1
                print(emotion + ', i: ' + str(i))


if __name__ == '__main__':
    test_image_transformation()
