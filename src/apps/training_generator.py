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
    def __init__(self, keep_log, board_type, name, testing):
        super().__init__(keep_log)
        pygame.display.set_caption("Evaluator")
        self.testing = testing
        self.data_handler = OpenBCIHandler(board_type=board_type, recording_name=name)
        if not self.testing:
            self.data_handler.load_demographics()
        if self.data_handler.status == "no_connection":
            print(
                "\n",
                "No headset connection, use '--board synthetic' for simulated data",
            )
            self.running = False
        elif self.data_handler.status == "invalid_demographics":
            print("Couldn't load demographics information.")
            self.running = False

        # Create shuffled list containing n TRIALS_PER_CLASS examples of all classes
        self.trials = CLASSES * TRIALS_PER_CLASS
        self.practice_trials = CLASSES * PRACTICE_TRIALS
        shuffle(self.practice_trials)
        shuffle(self.trials)
        n_trials = len(self.trials)
        self.n_practice = len(self.practice_trials)
        self.trials = self.practice_trials + self.trials
        self.break_trials = [
            n_trials // 3 + self.n_practice,
            (n_trials // 3) * 2 + self.n_practice,
        ]

    def exit(self):
        metadata = {
            "trials_per_class": TRIALS_PER_CLASS,
            "practice_trials": PRACTICE_TRIALS,
            "trials": self.trials,
            "break_trials": self.break_trials,
        }
        self.data_handler.add_metadata(metadata)
        self.data_handler.merge_trials_and_exit()

        if self.log:
            if self.keep_log:
                data = pd.DataFrame.from_dict(self.log)
                session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
                data.to_csv(f"data/recordings/Training_trials_{session_time}.csv")

    def run(self):
        while self.running:
            if self.state == "pause":
                self.wait_for_space("Press spacebar to resume.")
                self.pause = False
                self.state = "arrow"

            elif self.state == "start":
                self.wait_for_space(
                    "Welcome to the experiment! Press spacebar to begin the practice trials."
                )
                self.state = "arrow"

            elif self.state == "practive_over":
                self.data_handler.insert_marker(PRACTICE_END_MARKER)
                self.wait_for_space(
                    "You finished the practice trials! Press spacebar to begin with the experiment trials."
                )
                self.state = "arrow"

            elif self.state == "break":
                self.wait_for_space(
                    "Block done! Take a breather and press spacebar to resume when you feel ready.",
                )
                self.state = "arrow"

            elif self.state == "arrow":
                self.current_class = self.trials[self.trial - 1]
                self.draw_arrow()
                self.state = "imagine"
                pygame.time.delay(2000)

            elif self.state == "imagine":
                self.data_handler.insert_marker(CLASSES.index(self.current_class) + 1)
                self.draw_cross()
                self.state = "trial_end"
                pygame.time.delay(IMAGERY_PERIOD)
                self.data_handler.insert_marker(TRIAL_END_MARKER)

            elif self.state == "trial_end":
                self.draw_circle()
                # self.display_text(":)", FRONT_COL)
                if self.trial == self.n_practice and self.n_practice > 0:
                    self.state = "practive_over"
                elif self.trial in self.break_trials:
                    self.state = "break"
                else:
                    self.state = "arrow"

                data = {
                    "trial": self.trial,
                    "instruction": self.current_class,
                }
                print(data)
                self.log.append(data)
                self.trial += 1
                self.data_handler.save_trial()

                if self.trial > len(self.trials):
                    self.exit()
                    self.running = False
                    self.display_text(
                        "Experiment done! Thank you for your participation."
                    )

                pygame.time.delay(2000)

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
        default="daisy",
        nargs="?",
        type=str,
        help="Use synthetic or cyton board instead of daisy.",
    )

    parser.add_argument(
        "--name",
        const=None,
        nargs="?",
        type=str,
        help="Give explicit name for recording file to be created (e.g. for testing).",
    )

    parser.add_argument(
        "--testing",
        help="Skip demographics import and practice trials.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    generator = DataGenerator(
        keep_log=args.log,
        board_type=args.board_type,
        name=args.name,
        testing=args.testing,
    )
    generator.run()
