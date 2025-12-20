import pandas as pd
import os
import requests
import logging
from typing import NamedTuple, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
import zipfile

logger = logging.getLogger(__name__)

def __maybe_download(link: str, filename: str, force: bool = False) -> None:
    
    if os.path.exists(filename) and not force:
        logger.debug(f'file {filename} already exists')
    else:
        logger.debug(f'downloading {link} to {filename}')
        response = requests.get(link)
        with open(filename, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=128):
                fd.write(chunk)


def get_tensorflow_ECG5000(
        test_size: float = 0.2,
        random_state: int = 21,
        keep_original_labels: bool = False, 
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download ECG5000 postprocessed dataset

    Download dataset used in Tensorflow How-To 
    
    LINK: https://www.tensorflow.org/tutorials/generative/autoencoder
    
    :param test_size: Fraction of test data
    :param random_state: Random state for dataset splitting
    :param keep_original_labels: default False
        if True, labels are modified so to adhere custom approach:
            - nominal -> 0
            - anomalous -> 1
        if False, original mapping is retained
            - nominal -> 1
            - anomalous -> 0

    :return: 
        train_data, test_data, train_labels, test_labels
    
    """

    LINK = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'
    TEMP_FOLDER = '../data/temp/ECG5000'
    CSV_FILENAME = os.path.join(TEMP_FOLDER, 'ecg.csv')            
    
    # prepare temp folder
    logger.debug(f'creating temp file in {TEMP_FOLDER}')
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    # download data
    __maybe_download(LINK, CSV_FILENAME)

    # read data
    logger.debug(f'reading {CSV_FILENAME}')
    dataframe = pd.read_csv(CSV_FILENAME, header=None)
    raw_data = dataframe.values
    
    # The last element contains the labels
    labels = raw_data[:, -1]

    # labels are originally mapped in 0 -> Anomalous, 1 -> Nominal
    if not keep_original_labels:
        # changing to 0 -> Nominal, 1 -> Anomalous
        logger.debug('modifying labels')
        labels = np.where(labels == 0, 1, 0)

    # The other data points are the electrocadriogram data
    data = raw_data[:, 0:-1]
    
    return train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )


def get_ECG5000(
    keep_original_labels: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ECG5000 dataset
    
    LINK: https://www.timeseriesclassification.com/description.php?Dataset=ECG5000
    
    :param keep_original_labels: default False
        if True, labels are modified so to adhere custom approach:
            - nominal -> 0
            - anomalous -> 1
        if False, original mapping is retained
            - nominal -> 1
            - anomalous -> 2,3,4,5

    :return: 
        train_data, test_data, train_labels, test_labels
    
    
    """
    LINK = 'https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip'
    TEMP_FOLDER = '../data/temp/ECG5000'
    ZIP_FILENAME = os.path.join(TEMP_FOLDER, 'ECG5000.zip')

    # prepare temp folder
    logger.debug(f'creating temp file in {TEMP_FOLDER}')
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    # download data
    __maybe_download(LINK, ZIP_FILENAME)
    
    # extract
    logger.debug(f'extracting {ZIP_FILENAME} to {TEMP_FOLDER}')
    with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall(TEMP_FOLDER)
    
    # reformat to numpy ndarray
    TEST_FILE = os.path.join(TEMP_FOLDER, 'ECG5000_TEST.arff')
    TRAIN_FILE = os.path.join(TEMP_FOLDER, 'ECG5000_TRAIN.arff')
    
    def _read_arff(filename: str) -> Tuple[np.ndarray, np.ndarray]:

        logger.debug(f'reading {filename}')
        df = pd.read_csv(filename, skiprows=145, sep=',', header=None)
        raw_data = df.values

        data = raw_data[:,0:-1]
        labels = raw_data[:,-1]

        return data, labels

    train_data, train_labels = _read_arff(TRAIN_FILE)
    test_data, test_labels = _read_arff(TEST_FILE)

    # labels are originally mapped in 1 -> Nominal, 2...5 -> Anomalous
    if not keep_original_labels:
        logger.debug('modifying labels')
        # changing to 0 -> Nominal, 1 -> Anomalous
        train_labels = np.where(train_labels == 1, 0, 1)
        test_labels = np.where(test_labels == 1, 0, 1)

    return train_data, train_labels, test_data, test_labels
