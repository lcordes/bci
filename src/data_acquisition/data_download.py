import requests
import zipfile
import io
import os
from dotenv import load_dotenv

load_dotenv()
DATA_URL = os.environ["DATA_URL"]
DATA_PATH = os.environ["DATA_PATH"]


def download_data(url, dir):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dir)


if __name__ == "__main__":
    dir = f"{DATA_PATH}/recordings/training"
    download_data(DATA_URL, dir)
