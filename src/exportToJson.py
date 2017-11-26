from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os, sys

# Load model 1
model1 = load_model('trained/family/model.hdf5')
with open('model.json', 'w') as f:
    f.write(model1.to_json())