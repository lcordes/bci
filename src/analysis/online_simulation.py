import numpy as np
from random import choice
from datetime import datetime
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)
from data_acquisition.data_handler import RecordingHandler
from pipeline.preprocessing import preprocess_recording, preprocess_trial, get_users
from pipeline.utilities import create_config, load_model
from pipeline.transfer_learning import get_align_mat, align
from sklearn.metrics import accuracy_score


class Simulator:
    def __init__(self, user, config, model_name, alignment=False):
        self.user = user
        self.config = config
        self.alignment = alignment
        self.extractor, self.predictor = load_model(model_name)
        self.data_handler = RecordingHandler(user, config)
        self.sampling_rate = 250 if config["data_set"] == "benchmark" else 125
        self.labels = [1, 2, 3]
        if alignment:
            print("Simulating calibration trials for euclidean alignment")
            self.simulate_calibration()


    def simulate_calibration(self):
        trials = []
        for _ in range(15):
            data = self.data_handler.get_current_data(choice(self.labels))
            trials.append(preprocess_trial(data, self.sampling_rate, self.config))
        X = np.concatenate(trials, axis=0)
        self.align_mat = get_align_mat(X)


    def simulate_user(self, n=1000):
        label_hist = []
        pred_hist = []
        pred_times = []
        for _ in range(n):
            label = choice(self.labels)
            start = datetime.now()
            pred = self.get_prediction(label)
            pred_times.append((datetime.now() - start).total_seconds())
            label_hist.append(label)
            pred_hist.append(pred)
        return accuracy_score(label_hist, pred_hist), np.mean(pred_times)

    def get_prediction(self, label):
        data = self.data_handler.get_current_data(label)
        processed = preprocess_trial(data, self.sampling_rate, self.config)
        if self.alignment:
            processed = align(processed, self.align_mat)
        features = self.extractor.transform(processed)
        prediction = int(self.predictor.predict(features))
        return prediction

    def predict_offline(self):
        offline_config = create_config({"data_set": self.config["data_set"]})
        X, y = preprocess_recording(self.user, offline_config)
        if self.alignment:
            X = align(X, get_align_mat(X))
        X_transformed = self.extractor.transform(X)
        acc = self.predictor.score(X_transformed, y)
        return acc


if __name__ == "__main__":
    config = create_config({"data_set": "training", "simulate": True})
    n_iter = 1000
    users = get_users(config)
    accs, sim_accs, pred_times = [], [], []
    for user in users:
        sim = Simulator(user, config, f"{user}_tf", alignment=True)
        sim_acc, pred_time = sim.simulate_user(n=n_iter)
        acc = sim.predict_offline() 
        accs.append(acc)
        sim_accs.append(sim_acc)
        pred_times.append(pred_time)
        print(f"{user}: recording acc: {acc:.3f}, sim acc: {sim_acc:.3f}\n")
    print(f"Overall recording acc: {np.mean(accs):.3f}, overall sim acc: {np.mean(sim_accs):.3f}")
    print(f"Overall prediction processing time {np.mean(pred_times):.5f} seconds")