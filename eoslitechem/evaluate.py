import os
import h5py
import matplotlib.pyplot as plt


from .utils import Normalizer

from .train import TRAIN_SIZE
from . import OUTPUT_H5
from . import REFERENCE_H5

ROOT = os.path.dirname(os.path.abspath(__file__))

class Evaluator(object):
    def __init__(self, predictor, model_dir):
        self.model_dir = os.path.abspath(model_dir)
        self.precalculated_h5 = os.path.join(self.model_dir, OUTPUT_H5)
        self.reference_h5 = os.path.join(ROOT, "..", "data", REFERENCE_H5)
        self.predictor = predictor(model_dir)
        norm = Normalizer()
        norm.load(self.model_dir)
        self.n = norm.n
        self.test_n = int(self.n*(1-TRAIN_SIZE))
        self.train_n = int(self.n*TRAIN_SIZE)

    def _get_X_train(self):
        with h5py.File(self.reference_h5, "r") as f:
            X_train = f["Values"][:self.train_n]
        return X_train     
    
    def _get_X_test(self):
        with h5py.File(self.reference_h5, "r") as f:
            X_train = f["Values"][self.train_n:self.n]
        return X_train       
    
    def _get_y_train(self):
        with h5py.File(self.precalculated_h5, "r") as f:
            y_train = f["Values"][:self.train_n]
        return y_train

    def _get_y_test(self):
        with h5py.File(self.precalculated_h5, "r") as f:
            y_test = f["Values"][self.train_n:self.n]
        return y_test

    def _predict_train(self):
        X = self._get_X_train()
        y_hat = self.predictor.predict(X=X)
        return y_hat

    def _predict_test(self):
        X = self._get_X_test()
        y_hat = self.predictor.predict(X=X)
        return y_hat

    def evaluate(self, test=True):
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        if test:
            y = self._get_y_test().ravel()
            y_hat = self._predict_test().ravel()
        else:
            y = self._get_y_train().ravel()
            y_hat = self._predict_train().ravel()
        ax.scatter(y, y_hat)
        plt.savefig(os.path.join(self.model_dir, "eval.png"), dpi=300)

