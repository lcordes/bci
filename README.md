# Utilities and Apps for Motor Imagery Prediction using a Brain Computer Interface.
Scripts should be run from the project root directory. You can view optional parameters using -h:

    python3 script.py -h

## Setup
- First, install the necessary requirements (preferably in a virtual environment):

        pip install -r requirements.txt

- Rename the '.example_env' file in the root directory to '.env' and update its environment variables


![Overview of the Pipeline](https://github.com/lcordes/bci/architecture/pipeline.png)


## Generating training data and training models
Use the experiment interface to create a recording of motor imagery trials:

    python3 src/apps/train_generator.py

Train a feature extraction and a classifier model, specifying the recording from which you want to train and the name with which the models should be saved:

    python3 src/classification/train_model.py recording_name model_name

## Online Prediction using the BCI server
Start the BCI server for data acquisition and motor imagery prediction, specifying the models which should be used for prediction:

    python3 src/bci_server.py --model model_name

Alternatively, run the server without connecting to a headset using simulated EEG data instead:
    
    python3 src/bci_server.py --board synthetic

Then start one of the following applications in a seperate terminal.

### Using BCI predictions for movement control in a 2D game
Start the game environment with obstacles:

    python3 src/apps/game.py

Or in sandbox mode without obstacles:

    python3 src/apps/game.py --sandbox


### Assessing model accuracy using the experiment interface

Start the BCI server in client mode:

    python3 src/bci_server.py --client

Use the experiment interface to get feedback whether your motor imagery is classified correctly by the current model:

    python3 src/apps/accuracy_checker.py
