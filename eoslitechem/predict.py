import h5py
import os

from .utils import Normalizer
from . import REFERENCE_H5

ROOT = os.path.dirname(os.path.abspath(__file__))


class BasePredictor(object):
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
