# Utilities and Apps for Motor Imagery Classification using a Brain Computer Interface.
Scripts should be run from the project root directory. You can view optional parameters using -h:

    python3 script.py -h

## Setup
- First, install the necessary requirements (preferably in a virtual environment):

        pip3 install -r requirements.txt

- Rename the '.example_env' file in the root directory to '.env' and update its environment variables


## Generating training data and training models
Use the experiment interface to create a recording of motor imagery trials:

    python3 src/apps/experiment.py --log --train

Train a feature extraction and a classifier model, specifying the recording from which you want to train and the name with which the models should be saved:

    python3 src/classification/train_model.py recording_name model_name

## Online Prediction using the BCI server
Start the BCI server for data acquisition and motor imagery prediction, specifying the models which should be used for prediction:

    python3 src/bci_server.py --model model_name

Alternatively, run the server without connecting to a headset using simulated EEG data instead:
    
    python3 src/bci_server.py --sim

Then start one of the following applications in a seperate terminal.

### Assessing model accuracy using the experiment interface
Use the experiment interface to get feedback whether your motor imagery is classified correctly by the current model:

    python3 src/apps/experiment.py

### Using BCI predictions for movement control in a 2D game
Start the game environment with obstacles:

    python3 src/apps/game.py

Or in sandbox mode without obstacles:

    python3 src/apps/game.py --sandbox

## Measuring concentration
To get a visual measure of how concentrated the current user is start the BCI server in concentration mode instead:

    python3 src/bci_server.py --concentration

Then start the interface in a separate terminal:

    python3 src/apps/concentration.py
