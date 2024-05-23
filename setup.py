#!/usr/bin/env python3
"""Metadata for package to allow installation with pip."""

import os

import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

# Use same version from code
# See 3 from
# https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open(os.path.join("multimodalecho", "__version__.py")) as f:
    exec(f.read(), version)  # pylint: disable=W0122

setuptools.setup(
    name="multimodalecho",
    description="Multi-Modal Learning for Classification on Echocardiograms",
    version=version["__version__"],
    url="unavailable",
    packages=setuptools.find_packages(),
    install_requires=[
        "click",
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "opencv-python",
        "scikit-image",
        "tqdm",
        "sklearn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "multimodalecho=multimodalecho:main",
        ],
    }

)
