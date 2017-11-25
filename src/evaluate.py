from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Load model 1
model1 = load_model('trained/family/model.hdf5')


img1 = load_img('trained/family/e1.jpg', target_size=(64, 64))
e1 = img_to_array(img1)
e1 = np.expand_dims(e1, axis=0)

img2 = load_img('trained/family/e2.jpg', target_size=(64, 64))
e2 = img_to_array(img2)
e2 = np.expand_dims(e2, axis=0)

img3 = load_img('trained/family/e3.png', target_size=(64, 64))
e3 = img_to_array(img3)
e3 = np.expand_dims(e3, axis=0)

img4 = load_img('trained/family/k1.png', target_size=(64, 64))
k1 = img_to_array(img4)
k1 = np.expand_dims(k1, axis=0)

img5 = load_img('trained/family/y1.png', target_size=(64, 64))
y1 = img_to_array(img5)
y1 = np.expand_dims(y1, axis=0)

pred1 = model1.predict(e1)
pred2 = model1.predict(e2)
pred3 = model1.predict(e3)
pred4 = model1.predict(k1)
pred5 = model1.predict(y1)
print('Predition:', pred1)
print('Predition:', pred2)
print('Predition:', pred3)
print('Predition:', pred4)
print('Predition:', pred5)
