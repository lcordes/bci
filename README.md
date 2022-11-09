# Transfer Learning for Motor Imagery Classification using Low-Cost Brain-computer Interfaces
[![Pipeline tests](https://github.com/lcordes/bci/actions/workflows/run-tests.yml/badge.svg)](https://github.com/lcordes/bci/actions/workflows/run-tests.yml)
![Overview of the Pipeline](https://github.com/lcordes/bci/blob/main/architecture/pipeline.jpg)

This repository contains the code accompanying my master's thesis on the use of transfer learning for motor imagery classification. Two data sets were collected using a low-cost EEG device (the OpenBCI Ultracortex headset) and can be downloaded [here](https://osf.io/fb9pu/). See the figure above for an overview of the overall BCI pipeline.



## Setup
Scripts should be run from the project root directory.
- First, install the necessary requirements (preferably in a virtual environment):

        pip install -r requirements.txt

- Rename the '.example_env' file in the root directory to '.env' and update its environment variables.

## Downloading data

You can automatically download the collected OpenBCI data sets to the correct folders:

    python3 src/data_acquisition/data_download.py

To assess the BCI on medical-grade EEG, I used data set 2a of the fourth BCI competition as a benchmark. You can download it [here](https://www.bbci.de/competition/iv/#download). 

## Collecting data 

Run the graphical interface used for collecting data:

    python3 src/apps/training_generator.py

If you do not have an OpenBCI headset at hand you can simulate incoming EEG data:

    python3 src/apps/training_generator.py --board synthetic

## Evaluating models

Use the scripts in `src/analysis` to train and test models. For online classification using the trained models you can run the evaluation GUI:

    python3 src/apps/evaluator.py
