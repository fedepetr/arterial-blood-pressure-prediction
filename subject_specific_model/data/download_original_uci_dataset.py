"""
Script to download the original dataset "Cuff-Less Blood Pressure Estimation"
by UCI (University of California, Irvine).
"""


from os import getcwd, remove
from os.path import isdir, join as path_join
from shutil import rmtree
from sys import path
from zipfile import ZipFile

from requests import get

path.append(path_join(getcwd(), 'modeling'))
from modeling.constants import (
    ORIGINAL_UCI_DATASET_DIR,
    ORIGINAL_UCI_DATASET_URL
)


def download_uci_dataset(dataset_url: str, destination_dir: str) -> None:
    """
    Download the original UCI dataset from the given URL, saving it to the
    specified destination folder.
    """
    # path to temporarily save the downloaded .zip file before extracting it:
    temp_zip_dataset_path = path_join(destination_dir, 'temp.zip')

    # downloading and saving the original dataset, in its .zip format:
    response = get(url=dataset_url, stream=True)
    with open(temp_zip_dataset_path, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=128):
            fd.write(chunk)

    # extracting the original UCI files within the downloaded .zip dataset:
    with ZipFile(file=temp_zip_dataset_path, mode='r') as zip_file:
        zip_file.extractall(destination_dir)
    
    # deleting the .zip version of the dataset:
    remove(path=temp_zip_dataset_path)

    # deleting any potential decompression artifacts:
    decompression_artifacts_dir = path_join(destination_dir, '__MACOSX')
    if isdir(s=decompression_artifacts_dir):
        rmtree(path=decompression_artifacts_dir)


if __name__ == '__main__':
    download_uci_dataset(
        dataset_url=ORIGINAL_UCI_DATASET_URL,
        destination_dir=ORIGINAL_UCI_DATASET_DIR
    )
