import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scikitplot
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling2D, Dropout, BatchNormalization, GlobalMaxPooling2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, help="image size", default=224)
ap.add_argument("--classes", type=int, help="emotion classes", default=7)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--testing", type=bool, default=False)
ap.add_argument("--reduced", type=bool, default=False)
ap.add_argument("--augmented", type=bool, default=False)

size = ap.parse_args().size
classes = ap.parse_args().classes
epochs = ap.parse_args().epochs
testing = ap.parse_args().testing
reduced = ap.parse_args().reduced
augmented = ap.parse_args().augmented


# Load AffectNet Data from 'data/AffectNet'
def prepare_dataset(num_class):
    print('Dataset loading')
    batch_size = 64
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    if reduced:
        df_train = pd.read_pickle('pickles/image_eval/df_train_ambiguous_85.pkl')
        df_train = df_train[df_train['ambiguous'] == False]
        df_val = pd.read_pickle('pickles/image_eval/df_val_ambiguous_85.pkl')
        df_val = df_val[df_val['ambiguous'] == False]
        df_train = df_train.drop(df_train[df_train['y_col'] == 'Happiness'].sample(40000).index)
    else:
        df_train = pd.read_pickle('pickles/df_train.pkl')
        df_val = pd.read_pickle('pickles/df_val.pkl')
        df_train = df_train.drop(df_train[df_train['y_col'] == 'Neutral'].sample(50000).index)
        df_train = df_train.drop(df_train[df_train['y_col'] == 'Happiness'].sample(100000).index)

    if augmented:
        df_generated = pd.read_pickle('pickles/df_cyclegan_images.pkl')
        df_generated = df_generated.drop(df_generated[df_generated['y_col'] == 'Anger'].sample(2000).index)
        df_generated = df_generated.drop(df_generated[df_generated['y_col'] == 'Contempt'].sample(2000).index)
        df_generated = df_generated.drop(df_generated[df_generated['y_col'] == 'Disgust'].sample(2000).index)
        df_generated = df_generated.drop(df_generated[df_generated['y_col'] == 'Fear'].sample(2000).index)
        df_generated = df_generated.drop(df_generated[df_generated['y_col'] == 'Happiness'].sample(4000).index)
        df_generated = df_generated.drop(df_generated[df_generated['y_col'] == 'Neutral'].sample(4000).index)
        df_generated = df_generated.drop(df_generated[df_generated['y_col'] == 'Sadness'].sample(2000).index)
        df_generated = df_generated.drop(df_generated[df_generated['y_col'] == 'Surprise'].sample(2000).index)
        df_train = pd.concat([df_train, df_generated], ignore_index=True)

    if num_class == 5:
        df_train = df_train[
            (df_train['y_col'] != 'Contempt') & (df_train['y_col'] != 'Disgust') & (df_train['y_col'] != 'Fear')]
        df_val = df_val[
            (df_val['y_col'] != 'Contempt') & (df_val['y_col'] != 'Disgust') & (df_val['y_col'] != 'Fear')]
    elif num_class == 7:
        df_train = df_train[(df_train['y_col'] != 'Contempt')]
        df_val = df_val[(df_val['y_col'] != 'Contempt')]

    if testing:
        df_train = df_train.sample(1000)
        df_val = df_val.sample(100)

    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        x_col='x_col',
        y_col='y_col',
        target_size=(size, size),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_dataframe(
        df_val,
        x_col='x_col',
        y_col='y_col',
        shuffle=False,
        target_size=(size, size),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical'
    )

    class_weights = dict(enumerate(
        class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(df_train['y_col']),
                                          y=df_train['y_col'])))

    num_train = len(df_train)
    num_val = len(df_val)
    dataset_lengths = {'train': num_train, 'val': num_val}

    return train_generator, val_generator, class_weights, dataset_lengths


# Create the model
def get_model(image_size, num_classes):
    print('Creating recognizer model')
    input_shape = (image_size, image_size, 3)
    weight_decay = 0.01

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
                     kernel_regularizer=l1_l2(0, weight_decay),
                     bias_regularizer=l1_l2(0, weight_decay)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                     kernel_regularizer=l1_l2(0, weight_decay),
                     bias_regularizer=l1_l2(0, weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
                     kernel_regularizer=l1_l2(0, weight_decay),
                     bias_regularizer=l1_l2(0, weight_decay)))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu',
                     kernel_regularizer=l1_l2(0, weight_decay),
                     bias_regularizer=l1_l2(0, weight_decay)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu',
                     kernel_regularizer=l1_l2(0, weight_decay),
                     bias_regularizer=l1_l2(0, weight_decay)))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_regularizer=l1_l2(0, weight_decay),
                    bias_regularizer=l1_l2(0, weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu', kernel_regularizer=l1_l2(0, weight_decay),
                    bias_regularizer=l1_l2(0, weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(0, weight_decay),
                    bias_regularizer=l1_l2(0, weight_decay)))
    model.add(BatchNormalization())

    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model


# Save results of training in root folder
def save_results(model_type, model_history, val_gen, save_dir):
    # save progress and charts
    print('Saving results')
    epoch = model_history.epoch
    accuracy = model_history.history['accuracy']
    val_accuracy = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    sns.set()
    fig = plt.figure(0, (12, 4))

    ax = plt.subplot(1, 2, 1)
    sns.lineplot(x=epoch, y=accuracy, label='train')
    sns.lineplot(x=epoch, y=val_accuracy, label='valid')
    plt.title('Accuracy')
    plt.tight_layout()

    ax = plt.subplot(1, 2, 2)
    sns.lineplot(x=epoch, y=loss, label='train')
    sns.lineplot(x=epoch, y=val_loss, label='valid')
    plt.title('Loss')
    plt.tight_layout()

    plt.savefig(save_dir + '/epoch_history.png')
    plt.close(fig)

    # plot performance distribution
    df_accu = pd.DataFrame({'train': accuracy, 'valid': val_accuracy})
    df_loss = pd.DataFrame({'train': loss, 'valid': val_loss})

    dist_fig = plt.figure(1, (14, 4))
    ax = plt.subplot(1, 2, 1)
    sns.violinplot(x="variable", y="value", data=pd.melt(df_accu), showfliers=False)
    plt.title('Accuracy')
    plt.tight_layout()

    ax = plt.subplot(1, 2, 2)
    sns.violinplot(x="variable", y="value", data=pd.melt(df_loss), showfliers=False)
    plt.title('Loss')
    plt.tight_layout()

    plt.savefig(save_dir + '/performance_dist.png')
    plt.close(dist_fig)

    # create confusion matrix
    yhat_valid = model_type.predict(val_gen)

    predicted_classes = np.argmax(yhat_valid, axis=1)
    predicted_confidence = yhat_valid.max(axis=1)

    true_classes = val_gen.classes
    confuse_fig = scikitplot.metrics.plot_confusion_matrix(
        true_classes,
        predicted_classes,
        normalize='true',
        figsize=(7, 7))
    plt.savefig(save_dir + '/confusion_matrix.png')
    plt.close()

    with open(save_dir + '/results.txt', 'w') as f:
        results_string = classification_report(true_classes, predicted_classes)
        f.write(results_string)

    # save example faces, with true and predicted labels
    np.random.seed(2)
    random_indices = np.random.choice(range(len(val_gen.index_array)), size=18)

    face_fig = plt.figure(2, (18, 4))
    label_mapper = {v: k for k, v in val_gen.class_indices.items()}

    for j, i in enumerate(random_indices):
        ax = plt.subplot(3, 6, j + 1)
        sample_img_path = val_gen.filepaths[i]
        img = mpimg.imread(sample_img_path)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"true: {label_mapper[true_classes[i]]}, predict: {label_mapper[predicted_classes[i]]}")
        ax.set_xlabel("confidence: {0:.0%}".format(predicted_confidence[i]))
    plt.tight_layout()
    plt.subplots_adjust(wspace=1.7)
    plt.savefig(save_dir + '/example_faces.png')
    plt.close(face_fig)


# Train the model for around 70 epochs for best results
def train_model():
    print('Starting pipeline')
    batch_size = 64

    save_dir = 'results/recognizer/' + str(size) + '_' + 'reduced_' + str(reduced) + '_augmented_' + str(
        augmented) + '_' + str(classes) + 'cat'
    os.makedirs(save_dir, exist_ok=True)

    optimizer = Adam(learning_rate=0.0001, decay=1e-6)

    loss_function = CategoricalCrossentropy(label_smoothing=False)

    monitor_val_acc = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        verbose=1,
        restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    epochCheckpoint = ModelCheckpoint(
        save_dir + '/model.h5',
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        mode='min'
    )
    csv_logger = CSVLogger(
        save_dir + '/training.log',
        append=True
    )

    train_set, test_set, class_weights, dataset_lengths = prepare_dataset(classes)

    num_train = dataset_lengths['train']
    num_val = dataset_lengths['val']

    model = get_model(size, classes)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(
        train_set,
        class_weight=class_weights,
        steps_per_epoch=num_train // batch_size,
        epochs=epochs,
        validation_data=test_set,
        validation_steps=num_val // batch_size,
        callbacks=[monitor_val_acc, lr_scheduler, epochCheckpoint, csv_logger])

    if not testing:
        save_results(model, history, test_set, save_dir)


if __name__ == '__main__':
    train_model()
