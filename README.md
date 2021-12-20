# Demo pipeline for a BCI controlled 2D game.

## Installing requirements
First, install the necessary requirements (preferably in a virtual environment):

    pip3 install -r requirements.txt

## Running the pipeline
Start the BCI server for data acquisition and processing from the project root directory:

    python3 src/bci_server.py

Then start the game environment:

    python3 src/game/ladders.py

