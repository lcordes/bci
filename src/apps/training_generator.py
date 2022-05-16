from experiment_gui import *
from datetime import datetime
import argparse
import pygame
import pandas as pd
from random import shuffle
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from data_acquisition.data_handler import OpenBCIHandler


class DataGenerator(ExperimentGUI):
    def __init__(self, keep_log, board_type):
        super().__init__(keep_log)
        pygame.display.set_caption("Evaluator")
        self.data_handler = OpenBCIHandler(board_type=board_type)
        if self.data_handler.status == "no_connection":
            print(
                "\n",
                "No headset connection, use '--train synthetic' for simulated data",
            )
            self.running = False
        # Create shuffled list containing n TRIALS_PER_CLASS examples of all classes
        self.trial_classes = CLASSES * TRIALS_PER_CLASS
        shuffle(self.trial_classes)

    def exit(self):
        if self.log:
            self.data_handler.merge_trials_and_exit()
            if self.keep_log:
                data = pd.DataFrame.from_dict(self.log)
                session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
                data.to_csv(f"data/recordings/Training_trials_{session_time}.csv")

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
                self.current_class = self.trial_classes[self.trial - 1]
                self.draw_arrow()
                self.state = "fixdot"
                pygame.time.delay(2000)

            elif self.state == "imagine":
                self.data_handler.insert_marker(CLASSES.index(self.current_class) + 1)
                self.display_text("", FRONT_COL)
                self.state = "trial_end"
                pygame.time.delay(IMAGERY_PERIOD)
                self.data_handler.insert_marker(TRIAL_END_MARKER)

            elif self.state == "trial_end":
                self.draw_circle()
                self.state = "arrow"
                data = {
                    "trial": self.trial,
                    "instruction": self.current_class,
                }
                print(data)
                self.log.append(data)
                self.trial += 1
                self.data_handler.save_trial()

                if self.trial > len(self.trial_classes):
                    self.exit()
                    self.running = False
                else:
                    pygame.time.delay(1000)

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

    generator = DataGenerator(keep_log=args.log, board_type=args.board_type)
    generator.run()
