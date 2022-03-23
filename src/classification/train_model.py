import argparse
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from feature_extraction.extractors import MIExtractor, prepare_trials
from classifiers import Classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "recording_name",
        help="Name of the recording to train from (without extension).",
    )
    parser.add_argument(
        "model_name", help="Name for saving the trained model (without extension)."
    )
    args = parser.parse_args()
    recording_name = args.recording_name.replace(".pkl", "")
    model_name = args.model_name.replace(".pkl", "")

    # Prepare the trial data
    X, y = prepare_trials(recording_name)

    # Train and save the extractor
    extractor = MIExtractor(type="CSP")
    X_transformed = extractor.fit_transform(X, y)
    extractor.save_model(model_name)
    print("Extractor trained and saved successfully.")

    # Train and save the extractor
    classifier = Classifier(type="LDA")
    classifier.fit(X_transformed, y)
    classifier.save_model(model_name)
    print("Classifier trained and saved successfully.")
