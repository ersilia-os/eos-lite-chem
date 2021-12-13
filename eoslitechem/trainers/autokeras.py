import autokeras as ak
from .. import AUTOKERAS_PROJECT_NAME


class AutoKerasRegressorTrainer(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fit(self, max_trials):
        self.mdl = ak.StructuredDataRegressor(
            overwrite=False,
            max_trials=max_trials,
            project_name=AUTOKERAS_PROJECT_NAME,
        )
        self.mdl.fit(self.X, self.y)

    def export_model(self):
        return self.mdl.export_model()