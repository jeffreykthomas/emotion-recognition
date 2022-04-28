from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--size", default=64, help="image size")
ap.add_argument("-classes", default=8, help="num of emotions")

size = ap.parse_args().size
classes = ap.parse_args().classes
"""
## Constants and hyperparameters
"""

batch_size = 32
num_channels = 3
num_classes = classes
image_size = size
latent_dim = 128
num_epoch = 400

"""
## Loading the AffectNet dataset and preprocessing it
"""


def prepare_dataset():

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    df_train = pd.read_pickle('pickles/df_train.pkl')
    df_val = pd.read_pickle('pickles/df_val.pkl')

    df = pd.concat([df_val, df_train])

    train_generator = train_datagen.flow_from_dataframe(
        df,
        x_col='x_col',
        y_col='y_col',
        target_size=(64, 64),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical'
    )

    num_train = len(df)

    dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, image_size, image_size, num_channels], [None, num_classes])
    )

    return dataset, train_generator, num_train

"""
## Calculating the number of input channel for the generator and discriminator
In a regular (unconditional) GAN, we start by sampling noise (of some fixed
dimension) from a normal distribution. In our case, we also need to account
for the class labels. We will have to add the number of classes to
the input channels of the generator (noise input) as well as the discriminator
(generated image input).
"""


def get_model(input_image, emotion_classes):
    generator_in_channels = latent_dim + emotion_classes
    discriminator_in_channels = num_channels + emotion_classes
    print(generator_in_channels, discriminator_in_channels)

    # Create the discriminator.
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(input_image, input_image, discriminator_in_channels)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            layers.Conv2D(256, kernel_size=3, strides=1, padding="same"),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

    # Create the generator.
    generator = keras.Sequential(
        [
            keras.Input(shape=(generator_in_channels,)),
            layers.Dense(8 * 8 * generator_in_channels),
            layers.Reshape((8, 8, generator_in_channels)),
            layers.UpSampling2D(),
            layers.Conv2D(128, kernel_size=3, padding="same"),
            layers.BatchNormalization(momentum=0.8),
            layers.ReLU(),
            layers.UpSampling2D(),
            layers.Conv2D(256, kernel_size=3, padding="same"),
            layers.BatchNormalization(momentum=0.8),
            layers.ReLU(),
            layers.UpSampling2D(),
            layers.Conv2D(512, kernel_size=3, padding="same"),
            layers.BatchNormalization(momentum=0.8),
            layers.ReLU(),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")
        ],
        name="generator",
    )

    """
    ## Creating a `ConditionalGAN` model
    """

    class ConditionalGAN(keras.Model):
        def __init__(self, discriminator, generator, latent_dim):
            super(ConditionalGAN, self).__init__()
            self.discriminator = discriminator
            self.generator = generator
            self.latent_dim = latent_dim
            self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
            self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

        @property
        def metrics(self):
            return [self.gen_loss_tracker, self.disc_loss_tracker]

        def compile(self, d_optimizer, g_optimizer, loss_fn):
            super(ConditionalGAN, self).compile()
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer
            self.loss_fn = loss_fn

        def train_step(self, data):
            # Unpack the data.
            real_images, one_hot_labels = data

            # Add dummy dimensions to the labels so that they can be concatenated with
            # the images. This is for the discriminator.
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = tf.repeat(
                image_one_hot_labels, repeats=[image_size * image_size]
            )
            image_one_hot_labels = tf.reshape(
                image_one_hot_labels, (-1, image_size, image_size, num_classes)
            )

            # Sample random points in the latent space and concatenate the labels.
            # This is for the generator.
            batch_size = tf.shape(real_images)[0]
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            random_vector_labels = tf.concat(
                [random_latent_vectors, one_hot_labels], axis=1
            )

            # Decode the noise (guided by labels) to fake images.
            generated_images = self.generator(random_vector_labels)

            # Combine them with real images. Note that we are concatenating the labels
            # with these images here.
            fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
            real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
            combined_images = tf.concat(
                [fake_image_and_labels, real_image_and_labels], axis=0
            )

            # Assemble labels discriminating real from fake images.
            labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
            )

            # Train the discriminator.
            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_images)
                d_loss = self.loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

            # Sample random points in the latent space.
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            random_vector_labels = tf.concat(
                [random_latent_vectors, one_hot_labels], axis=1
            )

            # Assemble labels that say "all real images".
            misleading_labels = tf.zeros((batch_size, 1))

            # Train the generator (note that we should *not* update the weights
            # of the discriminator)!
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_vector_labels)
                fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
                predictions = self.discriminator(fake_image_and_labels)
                g_loss = self.loss_fn(misleading_labels, predictions)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # Monitor loss.
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(d_loss)
            return {
                "g_loss": self.gen_loss_tracker.result(),
                "d_loss": self.disc_loss_tracker.result(),
            }

    cond_gan = ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim)

    return cond_gan


def train_model():

    cond_gan = get_model(image_size, num_classes)
    dataset, train_generator, num_train = prepare_dataset()

    class GANMonitor(keras.callbacks.Callback):
        def __init__(self, num_img=3, latent_dim=128):
            self.num_img = num_img
            self.latent_dim = latent_dim

        def on_epoch_end(self, epoch, logs=None):
            num_rows = 4
            num_cols = 5

            label_mapper = {v: k for k, v in train_generator.class_indices.items()}
            # label_array = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
            label_array = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

            plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))

            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    label = keras.utils.to_categorical([label_array[index]], num_classes)
                    label = tf.cast(label, tf.float32)
                    noise = tf.random.normal(shape=(1, latent_dim))
                    noise_and_label = tf.concat([noise, label], 1)
                    generated_image = self.model.generator(noise_and_label)
                    plt.gca().set_title(label_mapper[label_array[index]])
                    plt.imshow(generated_image[0])
                    plt.axis("off")
            plt.tight_layout()
            plt.savefig('results/conditional_gan/images/generated_img_%d.png' % epoch)
            plt.close()

    filepath = 'results/conditional_gan/checkpoints/model_checkpoint_{epoch:02d}'
    epochCheckpoint = keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='g_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=False,
        mode='min'
    )

    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    cond_gan.fit(
        dataset,
        epochs=num_epoch,
        steps_per_epoch=num_train // batch_size,
        callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim), epochCheckpoint]
    )


if __name__ == '__main__':
    train_model()
