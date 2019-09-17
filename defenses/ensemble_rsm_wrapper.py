"""random subspace method, using homomorphic base classifier"""

import os
import sys
from datetime import datetime
from timeit import default_timer
import random

import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)

from dataset.input_preprocessing import random_over_sampling
from defenses.basic_dnn import BasicDNN, DNN_HP
from defenses.feature_binarization_dnn import FeatureBinarizationDNN
from defenses.adv_regularization import AdversarialTrainingRegDNN, PGDAdam, AUG_PARAM
from defenses.joint_defense import JointDefense

from utils import utils
from config import config

ENS_HP = {
    'base_module_symbol':3, # invalid number: [0, 1, 2, 3]
    'base_module_count': 5,
    'training_sample_ratio':.8,
    'feature_sample_ratio': .5
}
ENS_HP.update(DNN_HP)

module_symbol_method_mapping = {
    0: BasicDNN,
    1: AdversarialTrainingRegDNN,
    2: FeatureBinarizationDNN,
    3: JointDefense
}

class RandomSubspaceMethod(BasicDNN):
    def __init__(self,
                 hyper_params = None,
                 reuse=False,
                 is_saving = True,
                 init_graph = True,
                 mode = 'train',
                 name = 'RANDOM_SUBSPACE'):
        self.is_saving = is_saving
        self.init_graph = init_graph
        self.mode = mode
        if hyper_params is None:
            hyper_params = ENS_HP
        self.hp_params = utils.ParamWrapper(hyper_params)
        self.model_name = name

        self.sub_model_method = module_symbol_method_mapping[self.hp_params.base_module_symbol]
        sub_model_count = self.hp_params.base_module_count
        self.sub_model_names = ["{}_{}".format(self.model_name, k) for k in range(sub_model_count)]
        self.training_sample_ratio = self.hp_params.training_sample_ratio
        self.feature_ratio = self.hp_params.feature_sample_ratio

        # input distributing
        np.random.seed(self.hp_params.random_seed)
        self.random_seeds = np.random.choice(23456, sub_model_count, replace=False)

        super(RandomSubspaceMethod, self).__init__(hyper_params, reuse = reuse,
                                                   is_saving=self.is_saving, init_graph= self.init_graph,
                                                   mode = self.mode, name = name)

    def model_graph(self, reuse=False):
        self.sub_models = [
            self.sub_model_method(name=self.sub_model_names[k], is_saving=False, init_graph=True, reuse=False,
                                  mode=self.mode) \
            for k in range(self.hp_params.base_module_count)]

        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='ENS_X')
        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None, ], name='ENS_Y')
        self.is_training = tf.placeholder(tf.bool, name="ENS_TRAIN")

        # transmit some useful information to base models.
        for m in range(self.hp_params.base_module_count):
            self.sub_models[m].x_input = self.x_input
            self.sub_models[m].y_input = self.y_input
            self.sub_models[m].is_training = self.is_training

        self.logits, self.y_tensor = self.forward(self.x_input, self.y_input, self.x_input, reuse = True)
        self.model_inference()

    def forward(self, x_tensor, y_tensor, y_dae_tensor=None, reuse=False):
        # graph
        for sub_m in range(self.hp_params.base_module_count):
            masker = utils.zero_masking(self.input_dim, self.feature_ratio, random_seed=self.random_seeds[sub_m])
            x_input_masked = tf.multiply(x_tensor, masker)

            # different models
            self.sub_models[sub_m].mode = self.mode
            self.sub_models[sub_m].logits, self.sub_models[sub_m].y_tensor = \
                    self.sub_models[sub_m].forward(x_input_masked, y_tensor, reuse)

        logits_list = []
        for sub_m in range(self.hp_params.base_module_count):
            logits_list.append(self.sub_models[sub_m].logits)
        logits = tf.reduce_mean(tf.stack(logits_list, axis=0), axis=0)
        y_tensor = self.sub_models[0].y_tensor # work for homogeneous ensemble
        return logits, y_tensor

    def model_inference(self):
        # combination
        for sub_m in range(self.hp_params.base_module_count):
            self.sub_models[sub_m].model_inference()

        self.y_proba_comb = tf.reduce_mean(
            tf.stack([self.sub_models[m].y_proba for m in range(self.hp_params.base_module_count)], axis=0),
            axis=0
        )

        self.y_proba = self.y_proba_comb
        self.y_pred = tf.argmax(self.y_proba_comb, axis=1)

        # some information
        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(self.y_pred, self.y_tensor))
        )

    def train(self, trainX=None, trainy=None, is_sampling = True):
        """train dnn based malware detector"""
        if trainX is None and trainy is None:
            trainX, _, _ = utils.read_joblib(config.get('dataset', 'dataX'))
            trainy, _, _ = utils.read_joblib(config.get('dataset', 'datay'))
        if is_sampling:
            trainX, trainy = random_over_sampling(trainX, trainy, ratio=0.3)

        # train submodel subsequently per mini-batch
        global_train_step = tf.train.get_or_create_global_step()
        saver = tf.train.Saver()

        # optimizers
        from collections import defaultdict
        optimizers_dict = defaultdict(list)
        for sub_m in range(self.hp_params.base_module_count):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer_clf = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(
                    self.sub_models[sub_m].cross_entropy,
                    global_step=global_train_step)
                optimizers_dict[sub_m] = [optimizer_clf]

        tf_cfg = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        tf_cfg.gpu_options.allow_growth = True
        tf_cfg.gpu_options.per_process_gpu_memory_fraction = 1.
        sess = tf.Session(config=tf_cfg)

        with sess.as_default():
            # summary_writer = tf.summary.FileWriter(self.save_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            training_time = 0.0
            output_steps = 200
            for epoch_idx in range(self.hp_params.n_epochs):

                for sub_m in range(self.hp_params.base_module_count):
                    train_idx = range(len(trainX))
                    random.seed(self.random_seeds[sub_m])
                    sub_train_idx = random.sample(train_idx, int(len(train_idx) * self.training_sample_ratio))
                    train_input_supervised = utils.DataProducer(trainX[sub_train_idx], trainy[sub_train_idx],
                                                                self.hp_params.batch_size, n_epochs=1)
                    train_input_supervised.reset_cursor()

                    for step_idx, X_batch, y_batch in train_input_supervised.next_batch():

                        train_dict = {
                            self.x_input: X_batch,
                            self.y_input: y_batch,
                            self.is_training: True
                        }

                        start = default_timer()
                        if len(optimizers_dict[sub_m]) == 1:
                            sess.run(optimizers_dict[sub_m][0], feed_dict=train_dict)
                        else:
                            raise ValueError("Optimizer needs to be changed.")
                        end = default_timer()
                        training_time = training_time + end - start
                        iterations = epoch_idx * train_input_supervised.mini_batches + step_idx + 1

                        if iterations % output_steps == 0:
                            print("Sub model: ", sub_m)
                            print('Epoch {}/{},Step {}/{}:{}'.format(epoch_idx, self.hp_params.n_epochs,
                                                                     step_idx + 1, train_input_supervised.steps,
                                                                     datetime.now()))

                            _acc = sess.run(self.accuracy, feed_dict=train_dict)
                            print('    training accuracy {:.5}%'.format(_acc * 100))

                            if not os.path.exists(self.save_dir):
                                os.makedirs(self.save_dir)
                            saver.save(sess, os.path.join(self.save_dir, 'checkpoint'),
                                       global_step=global_train_step)

        sess.close()

def _main():
    rsm_dnn = RandomSubspaceMethod()
    rsm_dnn.train()
    rsm_dnn.mode = 'test'
    rsm_dnn.test()

if __name__ == "__main__":
    _main()