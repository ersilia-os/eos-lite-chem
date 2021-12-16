import os
import sys
import h5py
import tensorflow as tf

from . import ONNX_FILE

try:
    from .precalculate import PrecalculateErsilia
except:
    PrecalculateErsilia = None
from .trainers.tuner import TunerRegressorTrainer
from .trainers.autokeras import AutoKerasRegressorTrainer

from . import REFERENCE_H5
from . import OUTPUT_H5
from . import TFLITE_FILE

from .utils import Normalizer

import subprocess


ROOT = os.path.dirname(os.path.abspath(__file__))


class _Trainer(object):
    def __init__(
        self, reference_h5, precalculated_h5, output_dir, max_molecules, max_trials
    ):
        print("Initializing trainer")
        self.reference_h5 = reference_h5
        self.precalculated_h5 = precalculated_h5
        self.output_dir = output_dir
        self.max_molecules = max_molecules
        self.max_trials = max_trials
        self.cwd = os.getcwd()

    def _get_X_y(self):
        print("Getting X and y")
        with h5py.File(self.reference_h5, "r") as f:
            X = f["Values"][: self.max_molecules]
        with h5py.File(self.precalculated_h5, "r") as f:
            y = f["Values"][: self.max_molecules]
        return X, y

    def _normalize(self, y):
        n = Normalizer()
        n.fit(y)
        n.save(self.output_dir)
        return n.transform(y)

    def _train_autokeras(self, X, y):
        print("Training with {0} trials".format(self.max_trials))
        os.chdir(self.output_dir)
        mdl = AutoKerasRegressorTrainer(X, y)
        mdl.fit(max_trials=self.max_trials)
        os.chdir(self.cwd)
        return mdl

    def _train_tuner(self, X, y):
        print("Training naively (with tuner, not with AutoKeras)")
        os.chdir(self.output_dir)
        mdl = TunerRegressorTrainer(X, y)
        mdl.fit()
        os.chdir(self.cwd)
        return mdl

    def _export(self, mdl):
        print("Exporting model")
        os.chdir(self.output_dir)
        input_model = mdl.export_model()
        print(input_model.summary())
        print("Converting to TFLITE")
        output_model = os.path.join(self.output_dir, TFLITE_FILE)
        converter = tf.lite.TFLiteConverter.from_keras_model(input_model)
        tflite_quant_model = converter.convert()
        with open(output_model, "wb") as o_:
            o_.write(tflite_quant_model)
        print("Converting to ONNX")
        cmd = "{0} -m tf2onnx.convert --opset 13 --tflite {1} --output {2}".format(
            sys.executable, TFLITE_FILE, ONNX_FILE
        )
        print(cmd)
        subprocess.Popen(cmd, shell=True).wait()
        print("Conversions done!")
        os.chdir(self.cwd)

    def run(self):
        X, y = self._get_X_y()
        print("X shape: {0}".format(X.shape))
        print("y shape: {0}".format(y.shape))
        y = self._normalize(y)
        if self.max_trials == 0:
            mdl = self._train_tuner(X, y)
        else:
            mdl = self._train_autokeras(X, y)
        self._export(mdl)


class Trainer(object):
    def __init__(
        self, model_id, output_dir=".", max_molecules=1000000000, max_trials=1000
    ):
        assert model_id is not None and output_dir is not None

        self.model_id = model_id
        self.output_dir = os.path.join(os.path.abspath(output_dir), model_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_molecules = max_molecules
        self.max_trials = max_trials
        self.reference_h5 = os.path.join(ROOT, "..", "data", REFERENCE_H5)
        os.makedirs(self.output_dir, exist_ok=True)
        self.precalculated_h5 = os.path.join(self.output_dir, OUTPUT_H5)

    def _precalculate_ersilia(self):
        p = PrecalculateErsilia(
            model_id=self.model_id,
            reference_h5=self.reference_h5,
            output_h5=self.precalculated_h5,
            max_molecules=self.max_molecules,
        )
        p.run()

    def _train_lite_model(self):
        t = _Trainer(
            reference_h5=self.reference_h5,
            precalculated_h5=self.precalculated_h5,
            output_dir=self.output_dir,
            max_molecules=self.max_molecules,
            max_trials=self.max_trials,
        )
        t.run()

    def _get_size_reference_h5(self):
        with h5py.File(self.reference_h5, "r") as f:
            n = f["Values"].shape[0]
        return n

    def _precalculations_available(self):
        if not os.path.exists(self.precalculated_h5):
            return False
        try:
            r = self._get_size_reference_h5()
            with h5py.File(self.precalculated_h5, "r") as f:
                n = f["Values"].shape[0]
            if n < r and n < self.max_molecules:
                return False
            else:
                return True
        except:
            os.remove(self.precalculated_h5)
            return False

    def fit(self):
        if not self._precalculations_available():
            assert (
                PrecalculateErsilia is not None
            ), "Ersilia needs to be installed if no precalculations are available!"
            self._precalculate_ersilia()
        else:
            print("Precalculations are available!")
        self._train_lite_model()
