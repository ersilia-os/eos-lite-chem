import h5py
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import autokeras as ak
import onnxruntime as rt
import numpy as np

from .utils import Normalizer

from . import REFERENCE_H5
from . import OUTPUT_H5
from . import TFLITE_FILE
from . import ONNX_FILE
from . import AUTOKERAS_PROJECT_NAME
from . import AUTOKERAS_MODEL_FOLDER



ROOT = os.path.dirname(os.path.abspath(__file__))


class Predictor(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.reference_h5 = os.path.join(ROOT, "..", "data", REFERENCE_H5)

    def _get_X(self, idxs=None, head=None, tail=None):
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
        return X

    def predict_tflite(self, idxs=None, head=None, tail=None):
        X = self._get_X(idxs=idxs, head=head, tail=tail)
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

    def predict_onnx(self, idxs=None, head=None, tail=None):
        X = self._get_X(idxs=idxs, head=head, tail=tail)
        sess = rt.InferenceSession(os.path.join(self.model_dir, ONNX_FILE))
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        output_data = sess.run([output_name],{input_name: X})
        y=np.array(output_data[0])
        n = Normalizer()
        n.load(self.model_dir)
        return n.inverse_transform(y)

    def predict_autokeras(self, idxs=None, head=None, tail=None):
        X = self._get_X(idxs=idxs, head=head, tail=tail)
        mdl_path=os.path.join(self.model_dir, AUTOKERAS_PROJECT_NAME, AUTOKERAS_MODEL_FOLDER)
        if not os.path.exists(mdl_path):
            print("not Found")
        mdl = load_model(mdl_path, custom_objects=ak.CUSTOM_OBJECTS)
        output_data = mdl.predict(X)
        print(output_data)
        y = np.array(output_data)
        n = Normalizer()
        n.load(self.model_dir)
        return n.inverse_transform(y)
