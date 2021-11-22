# Lite Ersilia Models based on neural networks
A light version of Ersilia Models, based on pre-trained chemical language models

TODO: Documentation

### Packages
autokeras==1.0.16.post1
onnxruntime
tf2onnx
rdkit2021.03
ersilia

## Train

```python
from eoslitechem.train import Trainer

Trainer("eos2r5a", max_molecules=100, max_trials=1).fit()
```

## Predict

```python
from eoslitechem.predictors.onnx import Predictor

y = Predictor("eos2r5a").predict(head=100)
print(y)
```
