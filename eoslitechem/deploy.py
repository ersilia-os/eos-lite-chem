from .production.pack import Packer


class Deployer(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def deploy(self):
        p = Packer(self.model_path)
        saved_path = p.pack()
        return saved_path
