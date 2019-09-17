"""
feature binarization
"""

import os
import sys

import numpy as np
import tensorflow as tf

proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)

from defenses.basic_dnn import BasicDNN, graph
from dataset.input_preprocessing import get_median

def binarization(x_tf, threshold = None):
    """
    feature binarization
    :param x_tf: feature
    :type: tf.Tensor, tf.float32
    :param threshold: threshold value corresponds the shape of x_tf
    :type: float, 1D-array, 2D-array
    :return: binarized input
    :rtype: tf.Tensor
    """
    if isinstance(threshold, np.ndarray):
        assert x_tf.get_shape().as_list()[-1] == threshold.shape[-1]

    if threshold is None:
        return tf.rint(x_tf)
    else:
        # binary_mapping = tf.logical_and(tf.greater_equal(x_tf, threshold),
        #                                 tf.greater(x_tf, 0.))
        binary_mapping = tf.greater(x_tf, threshold)
        return tf.cast(binary_mapping, tf.float32)

class FeatureBinarizationDNN(BasicDNN):
    def __init__(self,
                 hyper_params = None,
                 reuse = False,
                 is_saving = True,
                 init_graph = True,
                 mode = 'train',
                 name = 'Feature_BNRZ_DNN'):
        super(FeatureBinarizationDNN, self).__init__(hyper_params, reuse, is_saving, init_graph, mode, name)

    def forward(self, x_tensor, y_tensor, reuse = False):
        # graph
        self.nn = graph
        self.threshold = None # get_median()
        self.input_transform = binarization(x_tensor, threshold= self.threshold)

        # debug
        # self.input_transform = tf.Print(self.input_transform,
        #                                [self.input_transform],'input_trans:', summarize=7)

        _1, _2, logits = graph(
            self.input_transform, self.hidden_layers, self.output_dim,
            self.is_training, name=self.model_name, reuse=reuse
        )

        return logits, y_tensor


def _test_binarization():
    r, c = 5, 3

    x = np.random.rand(5,3)
    x_tf = tf.constant(x, dtype = tf.float32)
    x_trans = binarization(x_tf)

    x_zeros = np.where(x < 0.5)
    x_ones = np.where(x >= 0.5)

    sess = tf.Session()
    x_proc = sess.run(x_trans)
    sess.close()
    assert (x_proc[x_zeros] == 0.).all()
    assert (x_proc[x_ones] == 1.).all()

    threshold = np.random.rand(3,)
    x_trans = binarization(x_tf, threshold)
    x_zeros = np.where(x <= threshold)
    x_ones = np.where(x > threshold)
    sess = tf.Session()
    x_proc = sess.run(x_trans)
    sess.close()
    assert (x_proc[x_zeros] == 0.).all()
    assert (x_proc[x_ones] == 1.).all()


def _main():
    _test_binarization()
    feat_bnrz_dnn = FeatureBinarizationDNN()
    feat_bnrz_dnn.train()
    feat_bnrz_dnn.mode = 'test'
    feat_bnrz_dnn.test()

if __name__ == "__main__":
    _main()