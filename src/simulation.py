import numpy as np
from random import choice
from data_acquisition.data_handler import RecordingHandler
from data_acquisition.preprocessing import preprocess_trial, preprocess_recording
from feature_extraction.extractors import CSPExtractor
from classification.classifiers import SVMClassifier, LDAClassifier
from sklearn.metrics import confusion_matrix


model_name = "optimal_16_LDA"
recording = "Training_session_#646445_30-06-2022_14-29-27"

extractor = CSPExtractor()
extractor.load_model(model_name)
predictor = LDAClassifier()  # Try LDA instead
predictor.load_model(model_name)
config = predictor.model.config
data_handler = RecordingHandler(
    recording_name=recording,
    config=config,
)
sampling_rate = data_handler.get_sampling_rate()


def get_prediction(command):
    raw = data_handler.get_current_data(label=command)
    processed = preprocess_trial(raw, sampling_rate, config)
    features = extractor.transform(processed)
    prediction = int(predictor.predict(features))
    return prediction


def predict_offline():
    X, y = preprocess_recording(recording, config)
    X = X[:, :, 1:]
    print(X.shape)
    X_transformed = extractor.transform(X)
    acc = predictor.score(X_transformed, y)
    return np.round(acc, 3)


# acc = predict_offline()
# print((acc))
labels = {1: "left", 2: "right", 3: "down"}
label_hist = []
pred_hist = []
for _ in range(1000):
    label = choice(list(labels.values()))
    pred = get_prediction(label)
    label_hist.append(label)
    pred_hist.append(labels[pred])

conf = confusion_matrix(label_hist, pred_hist, labels=list(labels.values()))
acc = np.mean([l == p for l, p in zip(label_hist, pred_hist)])
print(acc, "\n", conf)
