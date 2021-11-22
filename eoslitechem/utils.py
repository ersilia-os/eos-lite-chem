import os
import json
import numpy as np

from . import NORMALIZE_FILE


class Normalizer(object):

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, data):
        self.means = []
        self.stds = []
        for j in range(data.shape[1]):
            self.means += [np.nanmean(data[:,j])]
            self.stds += [np.nanstd(data[:,j])]
        self.means = [float(x) for x in self.means]
        self.stds = [float(x) for x in self.stds]

    def transform(self, data):
        data_ = np.zeros(data.shape, dtype=data.dtype)
        for j in range(data.shape[1]):
            mask = np.isnan(data[:,j])
            data[mask,j] = self.means[j]
            data_[:,j] = (data[:,j] - self.means[j]) / self.stds[j]
        return data_

    def inverse_transform(self, data):
        data_ = np.zeros(data.shape, dtype=data.dtype)
        for j in range(data.shape[1]):
            data_[:,j] = (data[:,j]*self.stds[j]) + self.means[j]
        return data_

    def save(self, dir_name):
        with open(os.path.join(dir_name, NORMALIZE_FILE), "w") as f:
            data = {
                "means": self.means,
                "stds": self.stds
            }
            json.dump(data, f, indent=4)

    def load(self, dir_name):
        with open(os.path.join(dir_name, NORMALIZE_FILE), "r") as f:
            data = json.load(f)
            self.means = data["means"]
            self.stds = data["stds"]
