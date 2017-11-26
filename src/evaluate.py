from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os, sys

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Load model 1
model1 = load_model('trained/family/model.hdf5')

dirlist = os.listdir('trained/family')
for f in dirlist:
    try:
        #print(f)
        if f.endswith("jpg") or f.endswith("png"):
            print(f)
            img1 = load_img('trained/family/' + f, target_size=(64, 64))
            e1 = img_to_array(img1)
            e1 = np.expand_dims(e1, axis=0)
            pred1 = model1.predict(e1)
            v1 = "{0:.0f}%".format(pred1[0][0] * 100)
            v2 = "{0:.0f}%".format(pred1[0][1] * 100)
            v3 = "{0:.0f}%".format(pred1[0][2] * 100)
            result = 'Predition for %s, Eddy: %s, Kristal: %s, Yanna: %s' % (f, v1, v2, v3)
            print (result)
    except Exception, e:
        # This is used to skip anything not an image.
        # Image.open will generate an exception if it cannot open a file.
        # Warning, this will hide other errors as well.
        pass