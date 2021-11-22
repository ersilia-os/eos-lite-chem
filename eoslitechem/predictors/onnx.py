import os
import numpy as np
import onnxruntime as rt


from ..predict import BasePredictor
from ..utils import Normalizer
from .. import ONNX_FILE


class Predictor(BasePredictor)
    def __init__(self, model_dir):
        BasePredictor.__init__(self, model_dir=model_dir)

    def predict(self, idxs=None, head=None, tail=None):
        X = self._get_X(idxs=idxs, head=head, tail=tail)
        sess = rt.InferenceSession(os.path.join(self.model_dir, ONNX_FILE))
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        output_data = sess.run([output_name],{input_name: X})
        y=np.array(output_data[0])
        n = Normalizer()
        n.load(self.model_dir)
        return n.inverse_transform(y)
