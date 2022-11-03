import joblib
from pathlib import Path
import sys
pipeline_dir = str(Path(__file__).parents[0].resolve())
sys.path.append(pipeline_dir)
from feature_extraction import CSPExtractor
from classification import CLASSIFIERS
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
IMAGERY_PERIOD = float(os.environ["IMAGERY_PERIOD"])
ONLINE_FILTER_LENGTH = float(os.environ["ONLINE_FILTER_LENGTH"])

def create_config(params):
    """Create a configuration object containing the given parameters and add default values for other
    parameters based on the data set used."""
    config = dict(
        model_name="",
        model_type="LDA",
        clf_specific={"shrinkage": "auto", "solver": "eigen"},
        n_classes=3,
        csp_components=8,
        csp_reg=None,
        simulate=False,
        discard_railed=True
    )
    if ("data_set", "benchmark") in params.items():
        data_set_specific = dict(
            channels=['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'],
            imagery_window=3, 
            bandpass=(8, 30),
            notch=None
        )
    else:
        data_set_specific = dict(
            data_set="training",
            channels=["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"],
            imagery_window=4, 
            bandpass=(10, 12),
            notch=(50),
            notch_width=0.5,
        )
    config.update(data_set_specific)
    config.update(params)

    config["description"] = ";".join(f"{param}={value}" for param, value in params.items())
    config["tmin"] = TRIAL_OFFSET if not config["simulate"] else - (ONLINE_FILTER_LENGTH - IMAGERY_PERIOD)
    trial_length = config["imagery_window"] if not config["simulate"] else ONLINE_FILTER_LENGTH
    config["tmax"] = config["tmin"] + trial_length
    
    return config

      
def train_model(X, y, config):
    extractor = CSPExtractor(config)
    X_transformed = extractor.fit_transform(X, y)
    classifier = CLASSIFIERS[config["model_type"]](config)
    classifier.fit(X_transformed, y)

    return extractor, classifier


def test_model(X, y, model):
    extractor, predictor = model
    X_transformed = extractor.transform(X)
    acc = predictor.score(X_transformed, y)
    return acc


def load_model(model_name):
    path = f"{DATA_PATH}/models/{model_name}.pkl"
    try:
        return joblib.load(path)
    except:
        print("Model not found.")


def save_model(model, config, model_name):
    model[1].model.config = config
    path = f"{DATA_PATH}/models/{model_name}.pkl"
    joblib.dump(model, path)

