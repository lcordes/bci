# Demo pipeline for a BCI controlled 2D game.

## Setup
- First, install the necessary requirements (preferably in a virtual environment):

        pip3 install -r requirements.txt

- Rename the '.example_env' file in the root directory to '.env' and update its environment variables

## Running the pipeline
Start the BCI server from the project root directory for data acquisition and processing:

    python3 src/bci_server.py

Alternatively, run the server without connecting to a headset using simulated EEG data instead:
    
    python3 src/bci_server.py --sim

Then start the game environment with obstacles:

    python3 src/apps/game.py

Or in sandbox mode without obstacles:

    python3 src/apps/game.py --sandbox

