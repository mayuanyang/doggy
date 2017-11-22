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

class SimpleConvBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                        activation='relu',
                        input_shape=input_shape))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(num_outputs, activation='relu'))
        model.add(Dense(num_outputs, activation='softmax'))
        return model