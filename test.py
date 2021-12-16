import h5py

ref = "data/reference_library.h5"

with h5py.File(ref, "r") as f:
    X = f["Values"][:2]

from eoslitechem.services.awslambda import AwsLambdaService

srv = AwsLambdaService("eos4e40")
print(srv.post(X))
