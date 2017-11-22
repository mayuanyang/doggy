from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD

import numpy as np
import resnet
import simpleconv


nb_train_samples = 299
nb_validation_samples = 32
batch_size = 32
nb_classes = 3
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 256, 256
# images are RGB.
img_channels = 3

# data dir
train_data_dir = 'family/train'
validation_data_dir = 'family/test'

# model file name to be saved
model_path = 'doggy_model.hdf5'
model_weight_path = 'doggy_model_weight.h5'

print(np.sqrt(0.1))
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=4, cooldown=0, verbose=1)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('doggy_result.csv')

model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
# model = VGG16(include_top=True, weights=None, classes=3)
# model = simpleconv.SimpleConvBuilder.build((img_channels, img_rows, img_cols), nb_classes)

model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, save_weights_only=True, monitor='val_loss')

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              # optimizer='adam',
              optimizer=sgd,
              metrics=['accuracy'])

print (model.summary())

if not data_augmentation:
    print('Not using data augmentation.')
    print('Not implemented yet')
else:
    print('Using real-time data augmentation.')
    # this is the augmentation configuration we will use for training
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
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[lr_reducer, csv_logger])

# Save model and weights
model.save(model_path)
model.save_weights(model_weight_path)
print('Saved trained model at %s ' % model_path)
print('Saved trained model weights at %s ' % model_weight_path)
