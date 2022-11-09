import requests
import zipfile
import io
import os
from dotenv import load_dotenv

load_dotenv()
TRAINING_URL = os.environ["TRAINING_URL"]
EVALUATION_URL = os.environ["EVALUATION_URL"]
DATA_PATH = os.environ["DATA_PATH"]


def download_data(data_set):
    if data_set == "training":
        url = TRAINING_URL
        dir=f"{DATA_PATH}/recordings/training"
    elif data_set == "evaluation":
        url = EVALUATION_URL
        dir=f"{DATA_PATH}/recordings/evaluation"

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dir)


if __name__ == "__main__":
    try:
        print("Downloading and extracting training data...")
        download_data("training")
        print("Downloading and extracting evaluation data...")
        download_data("evaluation")
        print("All data successfully downloaded.")
    except:
        print("Something went wrong. please try again.")