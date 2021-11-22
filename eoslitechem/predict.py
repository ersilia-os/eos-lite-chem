import h5py
import os
import tensorflow as tf
import numpy as np

from .utils import Normalizer

from . import REFERENCE_H5
from . import OUTPUT_H5
from . import TFLITE_FILE

ROOT = os.path.dirname(os.path.abspath(__file__))


class Predictor(object):

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.reference_h5 = os.path.join(ROOT, "..", "data", REFERENCE_H5)

    def predict(self, idxs=None, head=None, tail=None):
        print("Getting X")
        if idxs is not None:
            with h5py.File(self.reference_h5, "r") as f:
                X = f["Values"][idxs]
        elif head is not None:
            with h5py.File(self.reference_h5, "r") as f:
                X = f["Values"][:head]
        elif tail is not None:
            with h5py.File(self.reference_h5, "r") as f:
                X = f["Values"][-tail:]
        else:
            with h5py.File(self.reference_h5, "r") as f:
                X = f["Values"][:]
        print("Loading model for prediction")
        interpreter = tf.lite.Interpreter(os.path.join(self.model_dir, TFLITE_FILE))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        output_data = []
        for x in X:
            interpreter.set_tensor(input_details[0]['index'], [x])
            interpreter.invoke()
            output_data += [interpreter.get_tensor(output_details[0]['index'])[0]]
        y = np.array(output_data)
        n = Normalizer()
        n.load(self.model_dir)
        return n.inverse_transform(y)
