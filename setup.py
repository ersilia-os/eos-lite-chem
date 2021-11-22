import io
import os
import re

from setuptools import find_packages, setup

# Get the version from /__init__.py
# Adapted from https://stackoverflow.com/a/39671214
this_directory = os.path.dirname(os.path.realpath(__file__))
version_matches = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open(f"{this_directory}/eoslitechem/__init__.py", encoding="utf_8_sig").read(),
)
if version_matches is None:
    raise Exception("Could not determine version from __init__.py")
__version__ = version_matches.group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="eoslitechem",
    version=__version__,
    author="Ersilia Open Source Initiative",
    author_email="hello@ersilia.io",
    description="Build lite models for any model in the Ersilia Model Hub",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ersilia-os/eos-lite-chem/",
    packages=find_packages(),
    install_requires=["h5py"],
    include_package_data=True,
    zip_safe=True,
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ),
)
