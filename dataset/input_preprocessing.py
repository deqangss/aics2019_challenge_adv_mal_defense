import os
import sys
import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import utils
from config import config

def normalize_data(X, is_fitting = False):
    """Normalize data using minmaxscalar"""
    if not os.path.exists(config.get('dataset', 'normalizer')) and is_fitting:
        minmax_norm = MinMaxScaler()
        normalizer = minmax_norm.fit(X)
        utils.dump_joblib(normalizer, config.get('dataset', 'normalizer'),)
    normalizer = utils.read_joblib(config.get('dataset', 'normalizer'))
    x_clipped = np.clip(X, a_min=normalizer.data_min_, a_max=normalizer.data_max_)
    X_normlized = normalizer.transform(x_clipped)
    return X_normlized

def normalize_inverse(X, normalizer = None):
    try:
        if normalizer is None:
            normalizer = utils.read_joblib(config.get('dataset', 'normalizer'))
        if np.min(X) < 0 and np.max(X) > 1.:
            warnings.warn("The data is not within the range [0, 1]")
    except IOError as e:
        raise IOError("Unable to load normalizer.")
    return normalizer.inverse_transform(X)

def get_min_max_bound(normalizer=None):
    '''
    get the min and max contraints for data,
    :param normalizer: the normalizer, if None, load it from default location
    :return: minimum value and maximum value for each dimension
    '''
    if normalizer is not None:
        return normalizer.data_min_, normalizer.data_max_
    else:
        raise ValueError("No normalizer exists!")

def normalize_transform(X, normalizer):
    """Normalize feature into [0,1]"""
    if normalizer is not None:
        scale_data = normalizer.transform(X)
        return scale_data
    else:
        raise ValueError("No normalizer exists!")

def get_median():
    if not os.path.exists(config.get('dataset', 'threshold')):
        trainX, _, _ = utils.read_joblib(config.get('dataset', 'dataX'))
        threshold = np.median(trainX, axis=0)
        utils.dumpdata_np(threshold, config.get('dataset', 'threshold'))
    threshold = utils.readdata_np(config.get('dataset', 'threshold'))
    return threshold

def random_over_sampling(X, y, ratio = None):
    """
    over sampling
    :param X: data
    :type 2D numpy array
    :param y: label
    :type 1D numpy.ndarray
    :param ratio: proportion
    :type float
    :return: X, y
    """
    if ratio is None:
        return X, y
    if not isinstance(ratio, float):
        raise TypeError("{}".format(type(ratio)))
    if ratio > 1.:
        ratio = 1.
    if ratio < 0.:
        ratio = 0.

    if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
        raise TypeError

    count_array = np.bincount(y)
    max_count_num = np.max(count_array)
    curr_count = np.rint(max_count_num * ratio).astype(np.int64)
    X_amended_list = [X]
    y_amended_list = [y]
    for l in range(len(count_array)):
        if count_array[l] < curr_count:
            # extend the corresponding data
            random_indices = np.random.choice(
                np.where(y == l)[0], curr_count - count_array[l]
            )
            X_amended_list.append(X[random_indices])
            y_amended_list.append(y[random_indices])
        else:
            warnings.warn("The data labelled by {} is not conducted by over sampling ({} vs {}).".format(
                l, count_array[l], curr_count
            ), stacklevel= 4)

    def random_shuffle(x, random_seed = 0):
        np.random.seed(random_seed)
        np.random.shuffle(x)
    X_amended = np.concatenate(X_amended_list)
    random_shuffle(X_amended)
    y_amended = np.concatenate(y_amended_list)
    random_shuffle(y_amended)

    return X_amended, y_amended