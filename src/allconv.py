from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import numpy as np

K.set_image_dim_ordering('tf')

nb_train_samples = 299
nb_validation_samples = 32
batch_size = 32
nb_classes = 3
nb_epoch = 200

# data dir
train_data_dir = 'family/train'
validation_data_dir = 'family/test'

rows, cols = 64, 64

channels = 3

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                        activation='relu',
                        input_shape=(rows, cols, channels)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(nb_classes, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


print (model.summary())

lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=4, cooldown=0, verbose=1)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('doggy_result.csv')

train_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.2,
        rotation_range=25,
        horizontal_flip=True,
        vertical_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    # check this api for more into https://keras.io/preprocessing/image/

test_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        channel_shift_range=0.1,
        rotation_range=25)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(rows, cols),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(rows, cols),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[lr_reducer, csv_logger])

model.save('keras_allconv.h5')