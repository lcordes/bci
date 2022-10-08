from experiment_gui import *
from datetime import datetime
import argparse
import pygame
from random import shuffle
import sys
from pathlib import Path

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from data_acquisition.data_handler import OpenBCIHandler, RecordingHandler
from pipeline.utilities import load_model, train_model, create_config
from pipeline.preprocessing import preprocess_recording
from pipeline.transfer_learning import get_align_mat, align

instructions = [
            "In the following trials you will be shown an arrow which indicates what movement you should imagine.",
            "",
            "Pointing left: squeeze with your left hand",
            "Pointing right: squeeze with your right hand",
            "Pointing down: curl your feet",
            "",
            "You will then be presented with a fixation cross.",
            "While it is on screen, keep your eyes centered on the cross and perform the imagined movement.",
            "Once it disappears you can relax until the next trial starts.",
            "",
            "We will now begin with some practical trials to get you familiar with the experiment.",
            "Press spacebar to continue.",
        ]

class Evaluator(ExperimentGUI):
    def __init__(self, board_type, testing):
        super().__init__(fullscreen=(not testing))
        pygame.display.set_caption("Evaluator")
        self.testing = testing

        if self.board_type == "recording":
            config = create_config({"data_set": "training", "simulate": True})
            self.data_handler = RecordingHandler("u16", config)
            self.base_model = load_model("u16_base")
            self.tf_model = load_model("u16_tf")
        else:
            self.data_handler = OpenBCIHandler(board_type=board_type)
            self.base_model = load_model("training_all_base")
            self.tf_model = load_model("training_all_tf")
                
        if self.data_handler.status == "no_connection":
            print("\n No headset connection, use '--board synthetic' for simulated data")
            self.running = False
        elif self.data_handler.status == "invalid_demographics":
            print("Couldn't load demographics information.")
            self.running = False


    def get_trial_sequence(trials_per_class):
        sequence = CLASSES * trials_per_class
        shuffle(sequence)
        return sequence


    def run_welcome(self):
        intro = (
            "Welcome to the experiment! Please press spacebar to begin with a short questionnaire."
            if not self.data_handler.get_board_id() == -1
            else "Warning: Collection synthetic data. Start with spacebar."
        )
        self.wait_for_space(intro)


    def run_demographics(self): 

        for category in ["participant number", "age", "gender"]:
            if self.running:
                response = self.display_text_input(
                    f"Please enter your {category} and confirm with enter:"
                )
                self.data_handler.add_metadata({category: response})
                
                # Determine counterbalance order based on participant number
                if category == "participant number":
                    self.baseline_first = int(response) % 2 == 0 #TODO test if this works


    def get_current_recording(self, calibration):
        config = create_config({"data_set": "evaluation"})
        session_data = self.data_handler.combine_trial_data()
        metadata = self.data_handler.get_metadata()
        user = (session_data, metadata, calibration) 
        X, y = preprocess_recording(user, config)
        return X, y, config
    

    def calculate_align_mat(self): 
        X, _ , _ = self.get_current_recording(calibration=True)
        self.align_mat = get_align_mat(X)


    def train_within_model(self): 
        X, y, config = self.get_current_recording(calibration=False)
        self.witin_model = train_model(X, y, config)

    def text_break(self, text):
        self.wait_for_space(text)

    def run_trial(self, label, feedback): 
        run_trial = True
        while self.running and run_trial:
            if self.state == "pause":
                self.pause_menu()
                self.pause = False
                self.state = "fixdot"

            elif self.state == "fixdot":
                self.draw_circle()
                self.state = "arrow"
                pygame.time.delay(1000)
                self.play_sound("on_beep.wav")
                pygame.time.delay(500)

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
                self.play_sound("off_beep.wav")
                pygame.time.delay(500)
                self.display_text("")

                data = {
                    "trial": self.trial,
                    "instruction": self.current_class,
                }
                print(data)
                self.data_handler.save_trial()
                self.run_trial = False
                 

                pygame.time.delay(2500)

            self.check_events()


    def run(self):
        while self.running: #TODO this currently doesnt get checked during the entire experiment
            self.run_welcome()
            self.run_demographics()
            self.text_break(instructions)

            # Calibration/practice trials
            trial_sequence = self.get_trial_sequence(trials_per_class=4)
            for label in trial_sequence:
                self.run_trial(label, feedback=None)
            self.calculate_align_mat()
            self.data_handler.insert_marker({PRACTICE_END_MARKER})
            self.text_break("You finished the calibration trials! Press spacebar to begin with the experiment trials.")

            # Determine counterbalance order
            if self.baseline_first:
                block1_feedback, block2_feedback = "baseline", "tf"
            else:
                block1_feedback, block2_feedback = "tf", "baseline"

            # Block 1
            trial_sequence = self.get_trial_sequence(trials_per_class=10)
            for label in trial_sequence:
                self.run_trial(label, feedback=block1_feedback)
            self.text_break("Block done! Take a breather and press spacebar to resume when you feel ready.")

            # Block 2
            trial_sequence = self.get_trial_sequence(trials_per_class=10)
            for label in trial_sequence:
                self.run_trial(label, feedback=block2_feedback)
            self.text_break("Block done! Relax a bit and then start the last block by pressing spacebar.")

            # Block 3 (Within classifier trained on block 1 and 2)
            self.train_within_model()
            trial_sequence = self.get_trial_sequence(trials_per_class=10)
            for label in trial_sequence:
                self.run_trial(label, feedback="within")

            # Experiment end 
            text = ["Experiment done! Thank you for your participation.",
            "Please press spacebar to finish and then call the experimenter."]       
            self.text_break(text)
            self.exit(quit_early=False)

        pygame.quit()


    def exit(self, quit_early=True): #TODO update 
        metadata = {
            "trials_per_class": TRIALS_PER_CLASS,
            "practice_trials": PRACTICE_TRIALS,
            "trial_sequence": self.trials,
            "break_trials": self.break_trials,
            "quit_early": quit_early,
        }
        self.data_handler.add_metadata(metadata)
        self.data_handler.merge_trials_and_exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--board",
        dest="board_type",
        choices=["synthetic", "cyton", "daisy", "recording"],
        default="daisy",
        nargs="?",
        type=str,
        help="Use synthetic or cyton board instead of daisy, or simulate from an existing recording",
    )

    parser.add_argument(
        "--testing",
        help="Skip demographics import and practice trials.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    evaluator = Evaluator(
        board_type=args.board_type,
        testing=args.testing,
    )
    evaluator.run()
