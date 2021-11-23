import h5py
from ersilia import ErsiliaModel


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
        with ErsiliaModel(self.model_id) as em:
            em.api(input=input, output=output, batch_size=batch_size)

    def run(self):
        smiles_list = self._read_smiles()
        self._calculate(smiles_list)
