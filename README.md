M(otion)-mode based prediction of cardiac function on echocardiograms
------------------------------------------------------------------------------

This project (called multimodalecho for short) implements a deep learning pipeline for
  1) M-mode extraction from B-mode videos
  2) Prediction of ejection fraction using generated M-modes
  3) Assessment of cardiomyopathy with reduced ejection fraction

Dataset
-------
The EchoNet-Dynamic dataset, consisting of over 10,000 labeled echocardiograms, was used in this project.
Additional information is available at https://echonet.github.io/dynamic/. 
The workflow introduced by the EchoNet authors was used as a baseline for our pipeline.

Installation
------------

First, clone this repository and enter the directory by running:

    git clone https://gitlab.inf.ethz.ch/eoezkan/multimodalecho.git
    cd multimodalecho

You will find the complete source code of both echonet and multimodalecho in the git repo.
Both directories include another README with more specific details.
For convenience, you can use a venv to install all dependencies by calling:

    python3 -m venv env

where env is the name of your environment directory.
You can then activate the environment and install the newest version of pip:
    
    source env/bin/activate
    pip install --upgrade pip

Finally, you can enter the desired directory (echonet or multimodalecho) and install the dependencies for the respective pipeline using:

    cd multimodalecho
    pip install --user -e .

where the -e flag ensures that you enter editable mode and do not need to run pip install after every change you make to the source code. In case you do not plan on changing anything, you can omit this flag.

For more information on echonet and multimodalecho, consult the README file in the respective directory.


