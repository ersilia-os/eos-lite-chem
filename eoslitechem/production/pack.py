from .service import OnnxInference
from .. import NORMALIZE_FILE, ONNX_FILE
import os
import json


class Packer(object):

    def __init__(self, model_path):
        self.model_path = os.path.abspath(model_path)

    def pack(self):
        srv = OnnxInference()
        srv.pack("model", os.path.join(self.model_path, ONNX_FILE))
        with open(os.path.join(self.model_path, NORMALIZE_FILE), "r") as f:
            norm = json.load(f)
        srv.pack("normalizer", norm)
        saved_path = srv.save()
        return saved_path