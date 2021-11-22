import h5py
from ersilia.serve.api import Api
from ersilia.serve.autoservice import AutoService
from ersilia.cli.commands.utils.utils import tmp_pid_file


class PrecalculateErsilia(object):
    def __init__(self, model_id, reference_h5, output_h5, max_molecules, api_name=None):
        self.model_id = model_id
        self.reference_h5 = reference_h5
        self.output_h5 = output_h5
        self.max_molecules = max_molecules
        self.api_name = api_name

    def _read_smiles(self):
        print("Reading SMILES")
        with h5py.File(self.reference_h5, "r") as f:
            smiles_list = [x.decode("utf-8") for x in f["Inputs"][: self.max_molecules]]
        print("{0} SMILES read".format(len(smiles_list)))
        return smiles_list

    def _calculate(self, smiles_list):
        input = smiles_list
        output = self.output_h5
        batch_size = 100
        print("Calculating")
        mdl = AutoService(self.model_id)
        mdl.serve()
        if self.api_name is None:
            api_names = mdl.get_apis()
            assert len(api_names) == 1
            self.api_name = api_names[0]
        tmp_file = tmp_pid_file(self.model_id)
        with open(tmp_file, "r") as f:
            for l in f:
                url = l.rstrip().split()[1]
        api = Api(self.model_id, url, self.api_name)
        for result in api.post(input=input, output=output, batch_size=batch_size):
            continue
        mdl.close()

    def run(self):
        smiles_list = self._read_smiles()
        self._calculate(smiles_list)
