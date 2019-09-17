"""
adversarial training using small perturbations
"""

import os
import sys
import warnings
from datetime import datetime
from timeit import default_timer

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)

from defenses.basic_dnn import BasicDNN, graph, DNN_HP
from utils import utils
from dataset.input_preprocessing import get_min_max_bound, normalize_transform, normalize_data, normalize_inverse
from utils.adam_optimizer import TensorAdam
from config import config

# For regularization
AUG_PARAM = {
    'learning_rate': 0.01,
    'max_iteration': 60,
    'batch_size': 128
}

ADV_TRAIN_HP = {
    'trials': 5,
    'eta': 0.1,
    'lambda_':0.5
}
ADV_TRAIN_HP.update(DNN_HP)

class PGDAdam(object):
    """The perturbation direction is calculated in the adam optimizer rather than the l-infinity space"""
    def __init__(self, targeted_model, input_dim, normalizer, verbose = False, **kwargs):
        self.lr = AUG_PARAM['learning_rate']
        self.batch_size = AUG_PARAM['batch_size']
        self.iterations = AUG_PARAM['max_iteration']
        self.optimizer = TensorAdam(lr = self.lr)
        self.is_init_graph = False
        self.parse(**kwargs)
        self.model = targeted_model
        self.input_dim = input_dim
        self.verbose = verbose

        self.normalizer = normalizer

        self.clip_min = None
        self.clip_max = None
        self.scaled_clip_min = None
        self.scaled_clip_max = None

        self.clip_min, self.clip_max = get_min_max_bound(normalizer=self.normalizer)
        self.scaled_clip_min = normalize_transform(np.reshape(self.clip_min, (1, -1)), normalizer=self.normalizer)
        self.scaled_clip_max = normalize_transform(np.reshape(self.clip_max, (1, -1)), normalizer=self.normalizer)

    def project_pertubations(self, x_input, perturbations):
        # boundary
        return tf.clip_by_value(x_input + perturbations,
                                clip_value_min= self.scaled_clip_min,
                                clip_value_max= self.scaled_clip_max)

    def init_attack_graph(self):
        # TF tensor
        self.x_adv_batch = self.attack_graph(self.model.x_input, self.model.y_input)
        self.is_init_graph = True

    def attack_graph(self, x_input, y_input, using_normalizer = True):
        def _cond(i, *_):
            return tf.less(i, self.iterations)

        init_state = self.optimizer.init_state([tf.zeros_like(x_input, dtype = tf.float32)])
        nest = tf.contrib.framework.nest

        def _body(i, x_adv_tmp, flat_optim_state):
            curr_state = nest.pack_sequence_as(structure = init_state,
                                               flat_sequence = flat_optim_state)

            def _loss_fn_wrapper(x_):
                return -1 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model.get_logits(x_),
                                                                           labels=y_input)
            x_adv_tmp_list, new_optim_state = self.optimizer.minimize(_loss_fn_wrapper, [x_adv_tmp], curr_state)

            x_adv_tmp_clip = tf.clip_by_value(x_adv_tmp_list[0],
                                              clip_value_min= self.scaled_clip_min,
                                              clip_value_max= self.scaled_clip_max)

            return i + 1, x_adv_tmp_clip, nest.flatten(new_optim_state)

        flat_init_state = nest.flatten(init_state)
        _, adv_x_batch, _ = tf.while_loop(_cond,
                                          _body,
                                          (tf.zeros([]), x_input, flat_init_state),
                                          maximum_iterations=self.iterations,
                                          back_prop=False #
                                          )

        # map to discrete domain
        if using_normalizer:
            x_adv = tf.rint(tf.divide(adv_x_batch - self.normalizer.min_,
                                      self.normalizer.scale_)) # projection in the discrete domain with the determinstic threshold:0.5
            # project back
            x_adv_normalized = tf.multiply(x_adv, self.normalizer.scale_) + self.normalizer.min_
        else:
            x_adv_normalized = tf.rint(adv_x_batch)

        return x_adv_normalized

    def parse(self, learning_rate = 0.01, max_iteration = 55, batch_size = 50, **kwargs):
        self.lr = learning_rate
        self.iterations = max_iteration
        self.batch_size = batch_size
        self.optimizer = TensorAdam(lr=self.lr)
        if len(kwargs) > 0:
            warnings.warn("Unused hyper parameters.")

    def perturb(self, dataX, ground_truth_labels, sess = None):
        # TF tensor
        if not self.is_init_graph:
            self.x_adv_batch = self.attack_graph(self.model.x_input, self.model.y_input)

        try:
            input_data = utils.DataProducer(dataX, ground_truth_labels, batch_size= self.batch_size, name = 'test')

            # load baseline model parameters
            sess_close_flag = False
            if sess is None:
                cur_checkpoint = tf.train.latest_checkpoint(self.model.save_dir)
                config_gpu = tf.ConfigProto(log_device_placement=True)
                config_gpu.gpu_options.allow_growth = True
                sess = tf.Session(config=config_gpu)
                saver = tf.train.Saver()
                saver.restore(sess, cur_checkpoint)
                sess_close_flag = True
        except IOError as ex:
            raise IOError("PGD adam attack: Failed to load data and model parameters.")

        with sess.as_default():
            x_adv = []
            for idx, X_batch, y_batch in input_data.next_batch():

                _x_adv_tmp = sess.run(self.x_adv_batch, feed_dict={
                    self.model.x_input: X_batch,
                    self.model.y_input: y_batch,
                    self.model.is_training: False
                })

                x_adv.append(_x_adv_tmp)
            x_adv_normalized = np.concatenate(x_adv)
            if sess_close_flag:
                sess.close()

        return x_adv_normalized, ground_truth_labels

    

class AdversarialTrainingRegDNN(BasicDNN):
    def __init__(self,
                 hyper_params = None,
                 reuse = False,
                 is_saving = True,
                 init_graph = True,
                 mode = 'train',
                 name = 'ADV_TRAINING_DNN'):
        self.is_saving = is_saving
        self.init_graph = init_graph
        self.mode = mode

        if hyper_params is None:
            hyper_params = ADV_TRAIN_HP
        self.hp_params = utils.ParamWrapper(hyper_params)

        # initilization
        if not (os.path.exists(config.get('dataset', 'dataX')) and
                os.path.exists(config.get('dataset', 'datay')) and
                os.path.exists(config.get('dataset', 'normalizer'))):
            dataX, datay = self.data_preprocess()
            utils.dump_joblib(dataX, config.get('dataset', 'dataX'))
            utils.dump_joblib(datay, config.get('dataset', 'datay'))

        self.normalizer = utils.read_joblib(config.get('dataset', 'normalizer'))
        input_dim = len(self.normalizer.data_min_)
        self.inner_maximizer = PGDAdam(self, input_dim, self.normalizer, verbose=False, **AUG_PARAM)

        super(AdversarialTrainingRegDNN, self).__init__(hyper_params, reuse,
                                                        self.is_saving, self.init_graph, self.mode, name)

    def data_augment_graph(self, x_tensor, y_tensor, trials=0):
        """
        static graph for enhancing attack
        :param x_tensor: batch of input data
        :param y_tensor: batch of ground truths
        :param trials: number of trials
        :return: perturbed x_tensor, y_tensor
        """

        def filter(aug_x):
            """filter the examples that are predicted correctly"""
            logits = self.get_logits(aug_x)
            y_pred = tf.argmax(logits, axis=1)
            incorrect_case = tf.reshape(tf.to_float(tf.logical_not(
                tf.equal(y_pred, y_tensor))), (-1, 1))
            return tf.stop_gradient((aug_x - x_tensor) * incorrect_case + x_tensor)

        if trials == 0:
            aug_x = tf.stop_gradient(self.inner_maximizer.attack_graph(x_tensor, y_tensor))
            return filter(aug_x), y_tensor
        elif trials >= 1:
            x_shape = x_tensor.get_shape().as_list()
            x_batch_ext = tf.tile(x_tensor, [trials, 1])
            eta = tf.random_uniform([1, ], 0, self.hp_params.eta)
            init_perturbations = tf.random_uniform(tf.shape(x_batch_ext),
                                                   minval=-1.,
                                                   maxval=1.,
                                                   dtype=tf.float32)
            init_perturbations = tf.multiply(
                tf.sign(init_perturbations),
                tf.to_float(
                    tf.abs(init_perturbations) > 1. - eta),
            )

            init_x_batch_ext = self.inner_maximizer.project_pertubations(
                x_batch_ext,
                init_perturbations
            )
            init_x_batch_ext = tf.Print(init_x_batch_ext, [tf.reduce_mean(tf.reduce_sum(
                tf.abs(init_x_batch_ext - x_batch_ext), axis=-1))], message="initial perturbations:", summarize=5)
            trials = trials + 1
            x_batch_ext = tf.concat([x_tensor, init_x_batch_ext], axis=0)
            y_batch_ext = tf.tile(y_tensor, [trials, ])

            adv_x_batch_ext = self.inner_maximizer.attack_graph(
                x_batch_ext,
                y_batch_ext
            )

            def _loss_fn(x, y):
                _1, _2, logits = self.nn(x, self.hidden_layers, self.output_dim, False, name=self.model_name,
                                         reuse=True)
                return -1 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=y)

            adv_losses = tf.stop_gradient(_loss_fn(adv_x_batch_ext, y_batch_ext))

            adv_x_pool = tf.reshape(adv_x_batch_ext, [trials, -1, x_shape[1]])
            adv_losses = tf.reshape(adv_losses, [trials, -1])

            idx_selected = tf.stack([tf.argmin(adv_losses),
                                     tf.range(self.hp_params.batch_size, dtype=tf.int64)], axis=1)

            aug_x = tf.gather_nd(adv_x_pool, idx_selected)
            return filter(aug_x), y_tensor
        else:
            raise ValueError("trial is a positive integer.")

    def get_logits(self, dataX):
        """called after model graph defined"""
        return self.nn(dataX, self.hidden_layers, self.output_dim, is_training=False, name=self.model_name, reuse=True)[-1]

    def forward(self, x_tensor, y_tensor, reuse=False):
        # graph
        self.nn = graph
        _1, _2, _3 = graph(
            x_tensor, self.hidden_layers, self.output_dim,
            is_training=False, name=self.model_name, reuse=reuse
        )
        if self.mode == 'train':
            aug_x, aug_y = self.data_augment_graph(x_tensor, y_tensor, self.hp_params.trials)
            # debug info
            aug_x = tf.Print(aug_x,
                             [tf.reduce_mean(tf.reduce_sum(tf.abs(aug_x - x_tensor), axis=-1))],
                             message="Debug info: the average perturbations:",
                             summarize=self.hp_params.batch_size)
            self.aug_x = tf.cond(self.is_training,
                                 lambda : tf.concat([x_tensor, aug_x], axis=0),
                                 lambda : x_tensor
                                 )
            self.aug_y = tf.cond(self.is_training,
                                 lambda : tf.concat([y_tensor, aug_y], axis=0),
                                 lambda : y_tensor
                                 )
        elif self.mode == 'test':
            self.aug_x = x_tensor
            self.aug_y = y_tensor
        else:
            pass

        _1, _2, logits = graph(
            self.aug_x, self.hidden_layers, self.output_dim,
            is_training=self.is_training, name=self.model_name, reuse=True
        )
        y_tensor = self.aug_y

        return logits, y_tensor

    def model_inference(self):
        """
        define model inference
        :return: None
        """
        # loss definition
        cross_entropy_orig = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_tensor[:self.hp_params.batch_size],
            logits=self.logits[:self.hp_params.batch_size]
        )
        cross_entropy_aug = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_tensor[self.hp_params.batch_size:],
            logits=self.logits[self.hp_params.batch_size:]
        )
        self.cross_entropy = self.hp_params.lambda_ * cross_entropy_aug + \
                             (1. - self.hp_params.lambda_) * cross_entropy_orig
        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_tensor,
            logits=self.logits
        )

        # prediction
        self.y_proba = tf.nn.softmax(self.logits)
        self.y_pred = tf.argmax(self.logits, axis=1)

        # some information
        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(self.y_pred, self.y_tensor))
        )

def _main():
    adv_train_dnn = AdversarialTrainingRegDNN()
    adv_train_dnn.train()
    adv_train_dnn.mode = 'test'
    adv_train_dnn.test()

if __name__ == "__main__":
    _main()