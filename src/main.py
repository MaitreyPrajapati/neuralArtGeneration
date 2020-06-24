import cv2
import numpy as np
import tensorflow as tf
from vgg_model import model

def get_model():
    vgg_m = model()
    actual_model = vgg_m.return_model()
    return actual_model

model = get_model()
print(model.summary())