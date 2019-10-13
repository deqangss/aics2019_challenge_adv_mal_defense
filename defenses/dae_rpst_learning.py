"""
Denoise Autoencoder (DAE) using small perturbations
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

from dataset.input_preprocessing import random_over_sampling
from defenses.basic_dnn import BasicDNN, graph, DNN_HP
from defenses.adv_regularization import PGDAdam, ADV_TRAIN_HP, AUG_PARAM
from utils import utils
from config import config

DAE_TRAIN_HP = ADV_TRAIN_HP
DAE_TRAIN_HP.update(DNN_HP)

def graph_decoder(bottleneck, output_dim, is_training=True, hidden_units = 160, name="DECODER", reuse=False):
    with tf.variable_scope("{}".format(name), reuse=reuse):
        # decoder layer #1
        # the neuron unit is 160
        # input Tensor shape: [batch_size, 160]
        # output Tensor shape:[batch_size, 160]
        decoder1 = tf.layers.dense(inputs=bottleneck, units=hidden_units, activation=tf.nn.relu,
                                   name='DECODER1')

        # decoder layer #2
        # the neuron unit is output_dim
        # input Tensor shape: [batch_size, 160]
        # output Tensor shape:[batch_size, output_dim]
        output = tf.layers.dense(inputs=decoder1, units=output_dim, activation=None, name="DECODER2")
    return output

class DAE_RPST_DNN(BasicDNN):
    def __init__(self,
                 hyper_params = None,
                 reuse = False,
                 is_saving = True,
                 init_graph = True,
                 mode = 'train',
                 name = 'DAE_RPST_LEARN_DNN'):
        self.is_saving = is_saving
        self.init_graph = init_graph
        self.mode = mode

        if hyper_params is None:
            hyper_params = DAE_TRAIN_HP

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

        super(DAE_RPST_DNN, self).__init__(hyper_params, reuse,
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

            init_x_batch_ext = self.inner_maximizer.project_perturbations(
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

    def model_graph(self, reuse = False):
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='X')
        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None, ], name='Y')
        self.is_training = tf.placeholder(tf.bool, name="TRAIN")

        tf.set_random_seed(self.hp_params.random_seed)
        self.logits, self.y_tensor, self.decoder_logits, self.y_dae_tensor = \
            self.forward(self.x_input, self.y_input, self.x_input, reuse = reuse)
        self.model_inference()

    def forward(self, x_tensor, y_tensor, y_tensor_dae, reuse=False):
        # graph
        self.nn = graph
        _1, _2, _logits = graph(x_tensor,
                               self.hidden_layers,
                               self.output_dim,
                               is_training=False,
                               name=self.model_name,
                               reuse=reuse
                               )
        if self.mode == 'train':
            aug_x, aug_y = self.data_augment_graph(x_tensor, y_tensor, self.hp_params.trials)
            # debug info
            aug_x = tf.Print(aug_x,
                             [tf.reduce_mean(tf.reduce_sum(tf.abs(aug_x - x_tensor), axis=-1))],
                             message="Debug info: the average perturbations:",
                             summarize=self.hp_params.batch_size)
            self.aug_x = tf.cond(self.is_training,
                                 lambda : tf.concat([aug_x, x_tensor], axis=0),
                                 lambda : x_tensor
                                 )
            self.aug_y = tf.cond(self.is_training,
                                 lambda : tf.concat([aug_y, y_tensor], axis=0),
                                 lambda : y_tensor
                                 )
            #forward
            _1, _hl, _logits_all = graph(self.aug_x,
                                         self.hidden_layers,
                                         self.output_dim,
                                         self.is_training,
                                         name=self.model_name,
                                         reuse=True
                                         )
            adv_rpst = _hl[:self.hp_params.batch_size]

            # dae decoder graph
            _decoder_logits = tf.cond(self.is_training,
                                      lambda: graph_decoder(adv_rpst,
                                                            x_tensor.get_shape().as_list()[1],
                                                            self.is_training,
                                                            self.hidden_layers[0],
                                                            self.model_name,
                                                            reuse=reuse
                                                            ),
                                      lambda: tf.zeros(0))
            _y_tensor_dae = tf.cond(self.is_training,
                                    lambda: y_tensor_dae,
                                    lambda: tf.zeros(0))

            _logits = tf.cond(self.is_training,
                              lambda : _logits_all[self.hp_params.batch_size:],
                              lambda : _logits_all)
            return _logits, y_tensor, _decoder_logits, _y_tensor_dae
        elif self.mode == 'test':
            return _logits, y_tensor, tf.zeros(0), tf.zeros(0)
        else:
            raise ValueError("Mode supports 'train' and 'test'.")

    def model_inference(self):
        """
        define model inference
        :return: None
        """
        # loss definition
        self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_tensor,
            logits=self.logits
        )
        self.mse_dae = tf.losses.mean_squared_error(labels=self.y_dae_tensor,
                                                    predictions=tf.nn.sigmoid(self.decoder_logits))

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

    def get_logits(self, dataX):
        """get logits after static model graph defined"""
        return self.nn(dataX, self.hidden_layers, self.output_dim, is_training=False, name=self.model_name, reuse=True)[-1]

    def train(self, trainX = None, trainy = None, is_sampling = False):
        """train dnn based malware detector"""
        if trainX is None and trainy is None:
            trainX, _, _ = utils.read_joblib(config.get('dataset', 'dataX'))
            trainy, _, _ = utils.read_joblib(config.get('dataset', 'datay'))

        if is_sampling:
            trainX, trainy = random_over_sampling(trainX, trainy, ratio=0.3)

        train_input_supervised = utils.DataProducer(trainX, trainy,
                                                    self.hp_params.batch_size,
                                                    n_epochs=self.hp_params.n_epochs)

        saver = tf.train.Saver(max_to_keep=10)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.cross_entropy)
        merged_summaries = tf.summary.merge_all()
        global_train_step = tf.train.get_or_create_global_step()

        # optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer_clf = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.cross_entropy,
                                                                                          global_step=global_train_step)
            optimizer_dae = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.mse_dae)

        tf_cfg = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        tf_cfg.gpu_options.allow_growth = True
        tf_cfg.gpu_options.per_process_gpu_memory_fraction = 1.
        sess = tf.Session(config=tf_cfg)

        with sess.as_default():
            summary_writer = tf.summary.FileWriter(self.save_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            training_time = 0.0
            train_input_supervised.reset_cursor()
            output_steps = 100
            for step_idx, X_batch, y_batch in train_input_supervised.next_batch():
                train_dict = {
                    self.x_input: X_batch,
                    self.y_input: y_batch,
                    self.is_training: True
                }

                if (step_idx + 1) % output_steps == 0:
                    print('Step {}/{}:{}'.format(step_idx + 1, train_input_supervised.steps, datetime.now()))
                    _acc = sess.run(self.accuracy, feed_dict=train_dict)
                    print("The Accuracy on training batch:{:.5f}%".format(_acc * 100))
                    if step_idx != 0:
                        print('    {} samples per second'.format(
                            output_steps * self.hp_params.batch_size / training_time))
                        training_time = 0.

                    summary = sess.run(merged_summaries, feed_dict=train_dict)
                    summary_writer.add_summary(summary, global_train_step.eval(sess))
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    saver.save(sess, os.path.join(self.save_dir, 'checkpoint'),
                               global_step=global_train_step)

                start = default_timer()
                sess.run(optimizer_dae, feed_dict=train_dict)
                sess.run(optimizer_clf, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
        sess.close()

def _main():
    dae_rpst_dnn = DAE_RPST_DNN()
    dae_rpst_dnn.train()
    dae_rpst_dnn.mode = 'test'
    dae_rpst_dnn.test()

if __name__ == "__main__":
    _main()