import sys
import argparse
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input


target_size = (229, 229) #fixed size for InceptionV3 architecture


class Predict(object):
  @staticmethod
  def predict(model, img, target_size):
    """Run model prediction on image
    Args:
      model: keras model
      img: PIL format image
      target_size: (w,h) tuple
    Returns:
      list of predicted labels and their probabilities
    """
    print(img)
    if img.size != target_size:
      img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("jumping", "laying", "rolling", "sitting", "standing")
  plt.barh([0, 1, 2, 3, 4], preds, alpha=0.5)
  plt.yticks([0, 1, 2, 3, 4], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


if __name__=="__main__":
  root = Tk()
  root.withdraw() 
  filename = filedialog.askopenfilename()
  img2 = Image.open(filename)
  model = load_model("models/87.2_model.hdf5")
  blah = Predict()
  preds = blah.predict(model, img2, target_size)
  plot_preds(img2, preds)

