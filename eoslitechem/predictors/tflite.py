import os
import numpy as np
import tensorflow as tf

from ..predict import BasePredictor
from ..utils import Normalizer
from .. import TFLITE_FILE


class Predictor(BasePredictor):
    def __init__(self, model_dir):
        BasePredictor.__init__(self, model_dir=model_dir)

    def predict(self, X=None, idxs=None, head=None, tail=None):
        X = self._get_X(X=X, idxs=idxs, head=head, tail=tail)
        print("Loading model for prediction")
        interpreter = tf.lite.Interpreter(os.path.join(self.model_dir, TFLITE_FILE))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        output_data = []
        for x in X:
            interpreter.set_tensor(input_details[0]["index"], [x])
            interpreter.invoke()
            output_data += [interpreter.get_tensor(output_details[0]["index"])[0]]
        y = np.array(output_data)
        print(y.shape)
        n = Normalizer()
        n.load(self.model_dir)
        return n.inverse_transform(y)
