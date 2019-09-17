import os
import sys

import numpy as np
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from config import config
from utils.utils import read_tarfile
import json

def load_trainset():
    """
    Load training dataset from a .tar.gz file
    :returns: training data, ground truth
    :rtype: numpy.ndarray, numpy.ndarray
    :raise ValueError: No data
    """
    trainset_path = config.get("dataset", "trainset")
    train_set_fh = read_tarfile(trainset_path)
    if len(train_set_fh) >= 1:
        database = None
        for f in train_set_fh:
            database = json.loads(f.read())
    else:
        raise ValueError("No training dataset.")

    data = database['data']
    row_index = database['row_index']
    column_index = database['column_index']
    labels = np.array(database['labels'])  # convert to numpy.array
    shape = database['shape']

    feature_rpst = np.zeros(shape, dtype=np.float32)
    for i in range(len(row_index)):
        feature_rpst[row_index[i], column_index[i]] = data[i]

    return feature_rpst, labels

def load_testset():
    """
    load test set
    :returns: feature, ground truth labels, attack indicator, non_attack indicator
    :rtype: numpy.ndarray, list, list, list
    """
    testdata_path = config.get("dataset", "test_data")
    testdata_content = read_tarfile(testdata_path)
    if len(testdata_content) >= 1:
        data_schema = None
        for f in testdata_content:
            data_schema = json.loads(f.read())
    else:
        raise ValueError("No test data.")

    data = data_schema['data']
    row_index = data_schema['row_index']
    column_index = data_schema['column_index']
    shape = data_schema['shape']

    feature_rpst = np.zeros(shape, dtype=np.float32)
    for i in range(len(row_index)):
        feature_rpst[row_index[i], column_index[i]] = data[i]

    gt_path = config.get('dataset', 'test_ground_truth')
    gt_path_fh = read_tarfile(gt_path)
    if len(gt_path_fh) == 1:
        gt_schema = json.loads(gt_path_fh[0].read())
    else:
        raise ValueError("No test ground truth.")

    labels = np.array(gt_schema['labels'])
    attack_idx = gt_schema['attack_index']
    non_attack_idx = gt_schema['benign_index']

    return feature_rpst, labels, non_attack_idx, attack_idx
        
def _main():
    train_features, train_labels = load_trainset()

    test_features, test_labels, non_attr_idx, attr_idx = load_testset()
    return 0

if __name__ == "__main__":
    sys.exit(_main())

