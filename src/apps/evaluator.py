from experiment_gui import *
import argparse
import pygame
from random import shuffle
import numpy as np
import pandas as pd
import sys
from pathlib import Path

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from data_acquisition.data_handler import OpenBCIHandler
from pipeline.utilities import load_model, train_model, create_config
from pipeline.preprocessing import preprocess_recording, preprocess_trial
from pipeline.transfer_learning import get_align_mat, align
from analysis.between_users_transfer import train_full_models

PRACTICE_TRIALS_PER_CLASS = 4
CLASS_TRIALS_PER_BLOCK = 10

INSTRUCTIONS = [
            "In the following trials you will be shown an arrow which indicates what movement you should imagine.",
            "",
            "Pointing left: Imagine squeezing with your left hand",
            "Pointing right: Imagine squeezing with your right hand",
            "Pointing down: Imagine curling your feet",
            "",
            "You will then be presented with a fixation cross.",
            "While it is on screen, keep your eyes centered on the cross and perform the imagined movement.",
            "Once it disappears you can relax until the next trial starts.",
            "",
            "We will now begin with some calibration trials to adapt the experiment to your individual characteristics.",
            "Press spacebar to continue.",
        ]

AFTER_CALIBRATION = [
    "You finished the calibration trials!",  
    "From now on you will see how your imagined movement was classified after each trial.",
    "Remember that wrong classifications are the classifier's fault, not yours.",
    "Please press spacebar to begin with the experiment trials."
]

class Evaluator(ExperimentGUI):
    def __init__(self, board_type, testing):
        super().__init__(fullscreen=(not testing))
        pygame.display.set_caption("Evaluator")
        self.testing = testing
        self.speed = 3 if testing else 1
        self.block ={"1": "baseline", "2": "tf"}
        channels = ["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"] #["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"]
        self.config = create_config({"data_set": "evaluation", "channels": channels}) 
        
        if len(self.config["channels"]) < 8:
            self.state = "retrain"
        elif self.testing:
            self.state = "calibration"
        else:
            self.state = "intro"
        
        self.trials = {"instruction": [], "prediction": [], "classifier": []}

        self.data_handler = OpenBCIHandler(board_type=board_type)
        self.base_model = load_model("training_base")
        self.tf_model = load_model("training_tf")

        if self.data_handler.status == "no_connection":
            print("\n No headset connection, use '--board synthetic' for simulated data")
            self.running = False
        elif self.data_handler.status == "invalid_demographics":
            print("Couldn't load demographics information.")
            self.running = False

        self.channel_names = self.data_handler.get_metadata()["channel_names"]
        self.sampling_rate = 125 if board_type == "synthetic" else self.data_handler.get_sampling_rate()
        print(f"Sampling rate: {self.sampling_rate}")


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
                
                # Switch block order based on participant number for counterbalancing
                if category == "participant number":
                    if int(response) % 2 == 0:
                        self.block = {"1": "tf", "2": "baseline"}


    def get_current_recording(self, calibration):
        session_data = self.data_handler.combine_trial_data()
        metadata = self.data_handler.get_metadata()
        user = (session_data, metadata, calibration) 
        X, y = preprocess_recording(user, self.config)
        return X, y


    def get_trial_sequence(self, trials_per_class):
        sequence = CLASSES * trials_per_class
        shuffle(sequence)
        return sequence


    def calculate_align_mat(self): 
        try:
            X, y  = self.get_current_recording(calibration=True)
            self.align_mat = get_align_mat(X)
        except Exception as e:
            print("Calculating alignment matrix failed, got error", e)


    def train_within_model(self): 
        X, y = self.get_current_recording(calibration=False)
        self.within_model = train_model(X, y, self.config)


    def get_prediction(self, model_type):
        data = self.data_handler.get_current_data()
        processed = preprocess_trial(data, self.sampling_rate, self.channel_names, self.config)
        if model_type == "baseline":
            extractor = self.base_model[0]
            predictor = self.base_model[1]
        elif model_type == "tf":
            extractor = self.tf_model[0]
            predictor = self.tf_model[1]
            processed = align(processed, self.align_mat)
        elif model_type == "within":
            extractor = self.within_model[0]
            predictor = self.within_model[1]

        features = extractor.transform(processed)
        num_pred = int(predictor.predict(features))
        return CLASSES[num_pred -1]


    def retrain_classifiers(self):
        config = self.config.copy()
        config["data_set"] = "training"
        train_full_models(config, title="retrain")
        self.base_model = load_model("retrain_base")
        self.tf_model = load_model("retrain_tf")
        print(f"Retrained classifiers to use only {len(self.config['channels'])} channels ({self.config['channels']})")
        self.play_sound("on_beep.wav")

    def text_break(self, text):
        if self.running:
            self.wait_for_space(text)


    def run_trial(self, label, feedback): 
        run_trial = True
        self.trial_state = "fixdot"

        while self.running and run_trial:
            if self.pause:
                self.pause_menu()
                self.pause = False
                self.trial_state = "fixdot"

            elif self.trial_state == "fixdot":
                self.draw_circle()
                self.trial_state = "arrow"
                pygame.time.delay(1000//self.speed)
                if not self.testing:
                    self.play_sound("on_beep.wav")
                pygame.time.delay(500//self.speed)

            elif self.trial_state == "arrow":
                self.draw_arrow(label)
                self.trial_state = "imagine"
                pygame.time.delay(2000//self.speed)

            elif self.trial_state == "imagine":
                self.data_handler.insert_marker(CLASSES.index(label) + 1)
                self.draw_cross()
                self.trial_state = "feedback"
                pygame.time.delay(IMAGERY_PERIOD//self.speed)
                self.data_handler.insert_marker(TRIAL_END_MARKER)
                try:
                    pred = self.get_prediction(model_type=feedback) if feedback else None
                    clf = feedback
                except Exception as e:
                    pred = self.get_prediction(model_type="baseline") if feedback else None
                    print(f"Trial {len(self.trials['instruction']) + 1}: Getting prediction failed, defaulting to baseline model. Got error", e)
                    clf = "baseline due to error"
            elif self.trial_state == "feedback":
                if not self.testing:
                    self.play_sound("off_beep.wav")
                pygame.time.delay(500//self.speed)
                
                if not feedback:
                    self.display_text("")
                else:
                    col = GREEN if label == pred else RED
                    self.draw_cross(col, move=pred)

                self.trials["instruction"].append(label)
                self.trials["prediction"].append(pred)
                self.trials["classifier"].append(clf)

                print(f"Trial {len(self.trials['instruction'])}: Instruction: {label}, prediction: {pred}, classifier: {clf}")
                self.data_handler.save_trial()
                run_trial = False
                 
                pygame.time.delay(2500//self.speed)

            self.check_events()


    def run(self):
        quit_early=True
        while self.running: 
            
            if self.state == "retrain":
                self.display_text(f"Retrained classifiers to use only {len(self.config['channels'])} channels ({self.config['channels']})")
                self.retrain_classifiers()
                self.state = "intro"
            
            
            elif self.state == "intro":
                self.run_welcome()
                self.run_demographics()
                self.text_break(INSTRUCTIONS)
                self.state = "calibration"

            elif self.state == "calibration":
                trial_sequence = self.get_trial_sequence(trials_per_class=PRACTICE_TRIALS_PER_CLASS)
                for label in trial_sequence:
                    self.run_trial(label, feedback=None)

                if self.running:
                    self.calculate_align_mat()
                    self.data_handler.insert_marker(PRACTICE_END_MARKER)
                    self.text_break(AFTER_CALIBRATION)
                    self.state = "block1"
                    print("Calibration done")

            elif self.state == "block1":
                # Either tf or baseline classifer
                trial_sequence = self.get_trial_sequence(trials_per_class=CLASS_TRIALS_PER_BLOCK)
                for label in trial_sequence:
                    self.run_trial(label, feedback=self.block["1"])
                self.text_break("Block done! Take a breather and press spacebar to resume when you feel ready.")
                self.state = "block2"
                print("Block 1 done")

            elif self.state == "block2":
                # Either tf or baseline classifer
                trial_sequence = self.get_trial_sequence(trials_per_class=CLASS_TRIALS_PER_BLOCK)
                for label in trial_sequence:
                    self.run_trial(label, feedback=self.block["2"])
                self.text_break("Block done! Relax a bit and then start the last block by pressing spacebar.")
                self.state = "block3"
                print("Block 2 done")


            elif self.state == "block3":
            # Within classifier trained on block 1 and 2
                try:
                    self.train_within_model()
                except Exception as e:
                    print("Training within model failed, got error:", e)
                trial_sequence = self.get_trial_sequence(trials_per_class=CLASS_TRIALS_PER_BLOCK)
                for label in trial_sequence:
                    self.run_trial(label, feedback="within")

                print("Block 3 done")
                text = ["Experiment done! Thank you for your participation.",
                "Please press spacebar to finish and then call the experimenter."]       
                self.text_break(text)
                quit_early=False
                self.running=False
        
        # Save to disk if at least one experiment trial was done
        if len(self.trials["instruction"]) > PRACTICE_TRIALS_PER_CLASS * 3:
            self.exit(quit_early=quit_early)
            print(f"Total trials: {len(self.trials['instruction'])}")
        pygame.quit()


    def exit(self, quit_early=True): 
        metadata = {
            "instruction_hist": np.array(self.trials["instruction"], dtype="S"), 
            "prediction_hist": np.array(self.trials["prediction"], dtype="S"), 
            "classifier_hist": np.array(self.trials["classifier"], dtype="S"), 
            "used_channels": np.array(self.config['channels'], dtype="S"), 
            "quit_early": quit_early
        }
        self.data_handler.add_metadata(metadata)
        self.data_handler.merge_trials_and_exit()
        if self.trials:
            df = pd.DataFrame(self.trials)
            df["correct"] = df["instruction"] == df["prediction"]
            print(df.groupby("classifier")["correct"].mean())


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
        help="Skip demographics and run experiment at faster speed",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    evaluator = Evaluator(
        board_type=args.board_type,
        testing=args.testing,
    )
    evaluator.run()
