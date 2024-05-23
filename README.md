# M(otion)-mode based prediction of cardiac function on echocardiograms
------------------------------------------------------------------------------

This project (called multimodalecho for short) implements a deep learning pipeline for
  1) M-mode extraction from B-mode videos
  2) Prediction of ejection fraction using generated M-modes
  3) Assessment of cardiomyopathy with reduced ejection fraction

## Dataset
-------
The EchoNet-Dynamic dataset, consisting of over 10,000 labeled echocardiograms, was used in this project.
Additional information is available at https://echonet.github.io/dynamic/. 
The workflow introduced by the EchoNet authors was used as a baseline for our pipeline.

## Installation
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
--------------------------------------------------------------------------------------------


## ReadMe for the MultimodalEcho part
---------------------------------------------------------------------------------------------

This project (called multimodalecho for short) implements a deep learning pipeline for
  1) M-mode extraction from B-mode videos
  2) Prediction of ejection fraction using generated M-modes
  3) Assessment of cardiomyopathy with reduced ejection fraction

Dataset
-------
The EchoNet-Dynamic dataset, consisting of over 10,000 labeled echocardiograms, was used in this project.
Additional information is available at https://echonet.github.io/dynamic/. 
The workflow introduced by the EchoNet authors was used as a baseline for our pipeline.

Usage
-----
### YAML files

This project makes use of quite a few command line parameters.<br>
To provide users with an overview, all parameters can alternatively be set in YAML files.
Copy the provided `default.yaml` into a new `config.yaml` and configure the parameters as needed in your favourite text editor.

    scp default.yaml config.yaml
    nano config.yaml

Inside the YAML files, each parameter and its default value are described.<br>
In case you prefer using command line parameters, you can do so without changing anything, any parameter value passed per command line overwrites the available value in the YAML file.

### Extracting videos

A first (optional) step is the extraction of videos by length, i.e. number of frames.
As each frame in the video represents a one-pixel-wide vertical line in the resulting M-mode image, the video length dictates the M-mode image width.<br>
This step is optional and simply leads to the creation of a "new" dataset containing videos of the desired length, with videos of shorter length discarded.
As the same dataset is generally used for many (or all) experiments, this can be a sensible step to reduce future computational cost.<br>
If, for example, you would like to work with 112x112 M-mode images, you could run:

    python multimodalecho extract_videos

which will create the directory `../../extract_videos_112` storing the new `Video` directory, `FileList.csv` and `VolumeTracings.csv`.<br>
Make sure to update the `data_dir` in your `config.yaml` file!

### Running experiments

As soon as you're happy with your parameters and dataset, you can start running any individual experiment with this command:

    python multimodalecho run_model

Alternatively, if you want to run several jobs on a cluster, a job submission script is available (for LSF).
Define your desired value ranges for parameters you want to experiment with, select appropriate resources and you can submit a (large) number of jobs simultaneously: 

    ./submit_jobs.sh

### Collecting results

After running some experiments, you can collect your results and produce some plots using:

    python multimodalecho collect_results

This function will produce a table of all results (can be suppressed passing the `--no_table` flag) and will produce a number of plots (can be suppressed passing the `--no_plot` flag).
