from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os, sys
from keras.applications.inception_v3 import preprocess_input

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Load model 1
model1 = load_model('trained/87.2_model.hdf5')
print (model1)

dirlist = os.listdir('trained/dog_actions')
dirlist.sort()
for f in dirlist:
    try:
        #print(f)
        if f.endswith("jpg") or f.endswith("png"):
            #print(f)
            img1 = load_img('trained/dog_actions/' + f, target_size=(299, 299))
            e1 = img_to_array(img1)
            e1 = np.expand_dims(e1, axis=0)
            e1 = preprocess_input(e1)
            pred1 = model1.predict(e1)
            v1 = "{0:.0f}%".format(pred1[0][0] * 100)
            v2 = "{0:.0f}%".format(pred1[0][1] * 100)
            v3 = "{0:.0f}%".format(pred1[0][2] * 100)
            v4 = "{0:.0f}%".format(pred1[0][3] * 100)
            v5 = "{0:.0f}%".format(pred1[0][4] * 100)
            result = 'Predition for %s, jumping: %s, laying: %s, rolling: %s, sitting: %s, standing: %s' % (f, v1, v2, v3, v4, v5)
            print (result)
            #print (pred1)
    except Exception, e:
        # This is used to skip anything not an image.
        # Image.open will generate an exception if it cannot open a file.
        # Warning, this will hide other errors as well.
        pass