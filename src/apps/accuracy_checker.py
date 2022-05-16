from experiment_gui import *
from datetime import datetime
import argparse
import pygame
import pandas as pd
import numpy as np
from random import choice
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from communication.client_server import Client


class Evaluator(ExperimentGUI):
    def __init__(self, keep_log):
        super().__init__(keep_log)
        pygame.display.set_caption("Evaluator")
        self.client = Client()
        if self.client.connected():
            print("Connected to BCI server")
        else:
            print(
                "Couldn't connect to BCI server. Did you start it in client mode (--client)?"
            )
            self.running = False

    def exit(self):
        if self.log:
            data = pd.DataFrame.from_dict(self.log)
            acc = (data["instruction"] == data["prediction"]).mean().round(2)
            print(f"Session accuracy: {acc}")

            if self.keep_log:
                session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
                data.to_csv(f"data/recordings/Evaluation_trials_{session_time}.csv")

    def run(self):
        while self.running:
            if self.pause:
                self.display_text("Press spacebar to resume", FRONT_COL)
                while not self.key_pressed("space"):
                    pass
                self.pause = False
                self.state = "arrow"

            elif self.state == "start":
                self.display_text("Start session with spacebar", FRONT_COL)

                while not self.key_pressed("space"):
                    pass
                self.state = "arrow"

            elif self.state == "fixdot":
                self.draw_circle()
                self.state = "imagine"
                pygame.time.delay(1500)

            elif self.state == "arrow":
                self.current_class = choice(CLASSES)
                self.draw_arrow()
                self.state = "fixdot"
                pygame.time.delay(2000)

            elif self.state == "imagine":
                self.display_text("", FRONT_COL)
                self.state = "feedback"
                pygame.time.delay(IMAGERY_PERIOD)

            elif self.state == "feedback":

                self.client.request_command()
                prob_string = self.client.get_command()
                probs = prob_string.split(";")
                probs = [float(prob) for prob in probs]
                self.command = CLASSES[np.argmax(probs)]
                probs = [np.round(prob, 2) for prob in probs]
                self.display_feedback(probs)
                self.state = "arrow"

                data = {
                    "trial": self.trial,
                    "instruction": self.current_class,
                    "prediction": self.command,
                }
                print(data)
                self.log.append(data)
                self.trial += 1
                pygame.time.delay(3000)

            self.check_events()

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        help="write trial data to csv",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--board",
        dest="board_type",
        choices=["synthetic", "cyton", "daisy"],
        const="daisy",
        nargs="?",
        type=str,
        help="Use synthetic or cyton board instead of daisy.",
    )
    args = parser.parse_args()

    evaluator = Evaluator(keep_log=args.log)
    evaluator.run()
