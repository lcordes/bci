from random import choice
import sys
from pathlib import Path
from natsort import natsorted
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from data_acquisition.data_handler import RecordingHandler
from data_acquisition.preprocessing import preprocess_trial, preprocess_openbci
from classification.train_test_model import load_model
from sklearn.metrics import confusion_matrix, accuracy_score


class Simulator:
    def __init__(self, model_name, recording):
        self.extractor, self.predictor = load_model(model_name)
        self.config = self.predictor.model.config
        self.data_handler = RecordingHandler(
            recording_name=recording,
            config=self.config,
        )
        self.recording = recording
        self.sampling_rate = self.data_handler.get_sampling_rate()

    def simulate_user(self, n=1000):
        labels = {1: "left", 2: "right", 3: "down"}
        label_hist = []
        pred_hist = []
        for _ in range(n):
            label = choice(list(labels.values()))
            pred = self.get_prediction(label)
            label_hist.append(label)
            pred_hist.append(labels[pred])
        return accuracy_score(label_hist, pred_hist)

    def get_prediction(self, command):
        raw = self.data_handler.get_current_data(label=command)
        processed = preprocess_trial(raw, self.sampling_rate, self.config)
        features = self.extractor.transform(processed)
        prediction = int(self.predictor.predict(features))
        return prediction

    def predict_offline(self):
        X, y = preprocess_openbci(self.recording, self.config)
        X_transformed = self.extractor.transform(X)
        acc = self.predictor.score(X_transformed, y)
        return acc


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/training_data_collection")
    users = natsorted([path.stem for path in dir.glob("*.hdf5")])
    n_iter = 1000
    for user in users:
        sim = Simulator(f"LDA_optimal_{user}", user)
        sim_acc = sim.simulate_user(n=n_iter)
        acc = sim.predict_offline()
        print(f"{user} recording acc: {acc:.3f}, sim acc: {sim_acc:.3f}")
