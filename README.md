# Demo pipeline for a BCI controlled 2D game.

## Installing requirements
First, install the necessary requirements (preferably in a virtual environment):
    ```
    pip3 install -r requirements.txt
    ```
## Running the pipeline
To start the BCI server for data acquisition and processing from the root directory:
    ```
    python3 src/bci_server.py
    ```
To start the game environment:
    ```
    python3 src/game/sandbox.py
    ```
