# Lite Ersilia Models based on neural networks
This repository contains a pipeline to train a lighter version of the models available in the Ersilia Model Hub, based on pre-trained chemical language models. 

# Installation
We recommend creating a conda environment and cloning this repository:
```
conda create -n eoslitechem python=3.7
conda activate eoslitechem
git clone https://github.com/ersilia-os/eos-lite-chem.git
cd eos-lite-chem
```

### Packages
Install the following packages in the conda environment:\
autokeras==1.0.16.post1\
onnxruntime\
tf2onnx\
rdkit2021.03\
ersilia

## Reference Library
The chemical library used for model training must be an .h5 file containing two datasets:
- "Inputs": smiles strings
- "Values": molecular descriptors for each SMILES

In the examples below, we use a reference library of 2M compounds downloaded from CHEMBL29. The featurization of the reference compounds is available for [GROVER fingerprints](https://github.com/ersilia-os/groverfeat) and [MolBERT fingerprints](https://github.com/ersilia-os/molbertfeat).

## Precalculate
Optional: predict the activity of the reference library compounds for the relevant Ersilia Model prior to model training. This option is provided separately to facilitate division of the pipeline. It can be skipped as it is integrated in the train.
```python
from eoslitechem.train import Trainer

Trainer("<ersilia model id>", max_trials=1)._precalculate_ersilia()
```

## Train
Precalculate the activities of the reference library (if not existing) and train a neural network using autokeras (if max_trials =  1 or above) or a simple one_layer Keras-based neural network (if max_trials = -1)

```python
from eoslitechem.train import Trainer

Trainer("<ersilia model id>", max_molecules=100, max_trials=1).fit()
```

## Predict
Predict the activity of new molecules using the .onnx version of the Light Model. Please note that the predictor accepts only the same .h5 format as the reference library, with a dataset for "Inputs" and a dataset for "Values". Molecular representation must be done accordingly to the chosen for the reference library.
```python
from eoslitechem.predictors.onnx import Predictor

y = Predictor("<ersilia model id>").predict(head=100)
print(y)
```

# About us
The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization incorporated with the Charity Commission for England and Wales (number 1192266). Our mission is to reduce the imbalance in biomedical research productivity between countries by supporting research in underfunded settings.

You can support us via our [Open Collective](https:/opencollective.com/ersilia).
