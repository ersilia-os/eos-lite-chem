import numpy as np

from bentoml import env, artifacts, api
from bentoml import BentoService
from bentoml.frameworks.onnx import OnnxModelArtifact
from bentoml.service.artifacts.common import JSONArtifact
from bentoml.adapters import DataframeInput


@env(infer_pip_packages=True)
@artifacts([OnnxModelArtifact("model"), JSONArtifact("normalizer")])
class OnnxInference(BentoService):

    def denormalize(self, y, norm_json):
        means = norm_json["means"]
        stds = norm_json["stds"]
        y_ = np.zeros(y.shape, dtype=y.dtype)
        for j in range(y.shape[1]):
            y_[:, j] = (y[:, j] * stds[j]) + means[j]
        return y_

    @api(input=DataframeInput(), batch=True)
    def run(self, df):
        input_data = df.to_numpy().astype(np.float32)
        input_name = self.artifacts.model.get_inputs()[0].name
        output_name = self.artifacts.model.get_outputs()[0].name
        output_data = self.artifacts.model.run([output_name],{input_name: input_data})
        y = np.array(output_data[0])
        norm_json = self.artifacts.normalizer
        return self.denormalize(y, norm_json)
