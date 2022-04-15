from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import imageio
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="save/train")

mode = ap.parse_args().mode
"""
## Constants and hyperparameters
"""

batch_size = 32
num_channels = 3
num_classes = 8
image_size = 64
latent_dim = 128
num_epoch = 400

"""
## Loading the AffectNet dataset and preprocessing it
"""
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)
val_datagen = ImageDataGenerator(rescale=1. / 255)

df_train = pd.read_pickle('../src/df_train.pkl')
df_val = pd.read_pickle('../src/training_models/df_val.pkl')

df = pd.concat([df_val, df_train])
df = df[(df['y_col'] != 'Contempt') & (df['y_col'] != 'Disgust') & (df['y_col'] != 'Fear')]

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
# dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

"""
## Calculating the number of input channel for the generator and discriminator
In a regular (unconditional) GAN, we start by sampling noise (of some fixed
dimension) from a normal distribution. In our case, we also need to account
for the class labels. We will have to add the number of classes to
the input channels of the generator (noise input) as well as the discriminator
(generated image input).
"""

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

# Create the discriminator.
discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, discriminator_in_channels)),
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
        plt.savefig('conditional_gan/5cat/images/generated_img_%d.png' % (epoch + 44))
        plt.close()


filepath = 'conditional_gan/5cat/checkpoints/model_checkpoint_{epoch:02d}'
epochCheckpoint = keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='g_loss',
    verbose=1,
    save_weights_only=True,
    save_best_only=False,
    mode='min'
)
"""
## Training the Conditional GAN
"""

cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)


"""
## Interpolating between classes with the trained generator
"""


def test():
    # We first extract the trained generator from our Conditional GAN.
    trained_gen = cond_gan.generator

    num_rows = 4
    num_cols = 6

    label_mapper = {v: k for k, v in train_generator.class_indices.items()}
    label_array = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]

    plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))

    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)
            label = keras.utils.to_categorical([label_array[index]], num_classes)
            label = tf.cast(label, tf.float32)
            noise = tf.random.normal(shape=(1, latent_dim))
            noise_and_label = tf.concat([noise, label], 1)
            generated_image = trained_gen(noise_and_label)
            plt.gca().set_title(label_mapper[label_array[index]])
            image = keras.preprocessing.image.array_to_img(generated_image[0])
            plt.imshow(image)
            plt.axis("off")
    plt.tight_layout()
    plt.savefig('conditional_gan/images/generated_img_%d.png' % 25)
    plt.close()

    # Choose the number of intermediate images that would be generated in
    # between the interpolation + 2 (start and last images).
    # num_interpolation = 9  # @param {type:"integer"}
    #
    # # Sample noise for the interpolation.
    # interpolation_noise = tf.random.normal(shape=(1, latent_dim))
    # interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
    # interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))
    #
    # def interpolate_class(first_number, second_number):
    #     # Convert the start and end labels to one-hot encoded vectors.
    #     first_label = keras.utils.to_categorical([first_number], num_classes)
    #     second_label = keras.utils.to_categorical([second_number], num_classes)
    #     first_label = tf.cast(first_label, tf.float32)
    #     second_label = tf.cast(second_label, tf.float32)
    #
    #     # Calculate the interpolation vector between the two labels.
    #     percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    #     percent_second_label = tf.cast(percent_second_label, tf.float32)
    #     interpolation_labels = (
    #         first_label * (1 - percent_second_label) + second_label * percent_second_label
    #     )
    #
    #     # Combine the noise and the labels and run inference with the generator.
    #     noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    #     fake = trained_gen.predict(noise_and_labels)
    #     return fake
    #
    # start_class = 1  # @param {type:"slider", min:0, max:9, step:1}
    # end_class = 5  # @param {type:"slider", min:0, max:9, step:1}
    #
    # fake_images = interpolate_class(start_class, end_class)
    #
    # """
    # Here, we first sample noise from a normal distribution and then we repeat that for
    # `num_interpolation` times and reshape the result accordingly.
    # We then distribute it uniformly for `num_interpolation`
    # with the label indentities being present in some proportion.
    # """
    #
    # fake_images *= 255.0
    # converted_images = fake_images.astype(np.uint8)
    # converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
    # imageio.mimsave("conditional_gan/images/animation.gif", converted_images, fps=1)
    # for i in range(converted_images):
    #     img = keras.preprocessing.image.array_to_img(converted_images[i])
    #     img.save("conditional_gan/images/generated_img_%d.png" % i)


def train_gan():
    cond_gan.fit(
        dataset,
        epochs=num_epoch,
        steps_per_epoch=num_train // batch_size,
        callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim), epochCheckpoint]
    )


if mode == 'train':
    train_gan()
elif mode == 'grow':
    cond_gan.load_weights('conditional_gan/checkpoints/model_checkpoint_08')
    train_gan()
elif mode == 'create':
    label_mapper = {v: k for k, v in test_generator.class_indices.items()}
    label_array = [0, 1, 2, 3, 4, 5, 6, 7]
    num_images = 800
    array = []

    for n in range(num_images):
        label = keras.utils.to_categorical([label_array[n % 8]], num_classes)
        label = tf.cast(label, tf.float32)
        noise = tf.random.normal(shape=(1, latent_dim))
        noise_and_label = tf.concat([noise, label], 1)
        generated_image = model.generator(noise_and_label, training=False)
        image = keras.preprocessing.image.array_to_img(generated_image[0])
        prediction = fer_model.predict(generated_image)[0]
        predicted_class = np.argmax(prediction, axis=1)
        if predicted_class == (n % 8):
            filepath = 'data/gen_images/1/generated_img_%d.jpeg' % n
            image.save(filepath)
            label_name = label_mapper[label_array[n % 8]]
            array.append([filepath, label_name])

    df_generated = pd.DataFrame(data=array, columns=['x_col', 'y_col'])
    df_generated.to_pickle('df_generated_1.pkl')
else:
    cond_gan.load_weights('conditional_gan/checkpoints/model_checkpoint_19')
    test()
