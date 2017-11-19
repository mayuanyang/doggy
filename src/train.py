from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint

import numpy as np
import resnet


nb_train_samples = 8038
nb_validation_samples = 39
batch_size = 32
nb_classes = 5
nb_epoch = 10
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# images are RGB.
img_channels = 3

# data dir
train_data_dir = 'dog_dataset/train'
validation_data_dir = 'dog_dataset/test'

# model file name to be saved
model_path = 'doggy_model.h5'
model_weight_path = 'doggy_model_weight.h5'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('doggy_result.csv')

model = resnet.ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), nb_classes)
model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, save_weights_only=True, monitor='val_loss')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    print('Not implemented yet')
else:
    print('Using real-time data augmentation.')
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        zca_whitening=True,
        rotation_range=25,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    # check this api for more into https://keras.io/preprocessing/image/

    test_datagen = ImageDataGenerator(
        zca_whitening=True,
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
        callbacks=[lr_reducer, early_stopper, csv_logger, model_checkpoint])

# Save model and weights
model.save(model_path)
model.save_weights(model_weight_path)
print('Saved trained model at %s ' % model_path)
print('Saved trained model weights at %s ' % model_weight_path)
