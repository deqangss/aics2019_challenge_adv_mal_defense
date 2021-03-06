"""joint feature binarization and adversarial training"""

import os
import sys

import tensorflow as tf

proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)

from defenses.basic_dnn import BasicDNN, graph, DNN_HP
from defenses.adv_regularization import PGDAdam
from defenses.feature_binarization_dnn import binarization
from utils import utils
from config import config

AUG_PARAM = {
    'learning_rate': 0.01,
    'max_iteration': 60,
    'batch_size': 128
}

ADV_TRAIN_HP = {
    'trials':5,
    'eta': 0.1,
    'lambda_':0.5
}
ADV_TRAIN_HP.update(DNN_HP)

class JointDefense(BasicDNN):
    def __init__(self,
                 hyper_params = None,
                 reuse = False,
                 is_saving = True,
                 init_graph = True,
                 mode = 'train',
                 name = 'JOINT_DEFENSE'):
        self.is_saving = is_saving
        self.init_graph = init_graph
        self.mode = mode

        if hyper_params is None:
            hyper_params = ADV_TRAIN_HP
        self.hp_params = utils.ParamWrapper(hyper_params)
        self.threshold = None # get_median()

        # attack initilization
        if not (os.path.exists(config.get('dataset', 'dataX')) and
                os.path.exists(config.get('dataset', 'datay')) and
                os.path.exists(config.get('dataset', 'normalizer'))):
            dataX, datay = self.data_preprocess()
            utils.dump_joblib(dataX, config.get('dataset', 'dataX'))
            utils.dump_joblib(datay, config.get('dataset', 'datay'))


        self.normalizer = utils.read_joblib(config.get('dataset', 'normalizer'))
        input_dim = len(self.normalizer.data_min_)
        self.inner_maximizer = PGDAdam(self, input_dim, self.normalizer, verbose=False, **AUG_PARAM)
        super(JointDefense, self).__init__(hyper_params, reuse,
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
            aug_x = tf.stop_gradient(self.inner_maximizer.attack_graph(x_tensor,
                                                                       y_tensor,
                                                                       using_normalizer=False))
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
                y_batch_ext,
                using_normalizer= False
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
        """called after static model graph called"""
        return self.nn(dataX, self.hidden_layers, self.output_dim, is_training=False, name=self.model_name, reuse=True)[-1]

    def forward(self, x_tensor, y_tensor, reuse=False):
        # graph
        self.input_transform = binarization(x_tensor, threshold=self.threshold)
        self.nn = graph
        _1, _2, _3 = graph(
            self.input_transform, self.hidden_layers, self.output_dim,
            is_training=False, name=self.model_name, reuse=reuse
        )
        if self.mode == 'train':
            aug_x, aug_y = self.data_augment_graph(self.input_transform, y_tensor, self.hp_params.trials)
            # debug info
            aug_x = tf.Print(aug_x,
                             [tf.reduce_mean(tf.reduce_sum(tf.abs(aug_x - self.input_transform), axis=-1))],
                             message="Debug info: the average perturbations:",
                             summarize=self.hp_params.batch_size)
            self.aug_x = tf.cond(self.is_training,
                                 lambda: tf.concat([self.input_transform, aug_x], axis=0),
                                 lambda: x_tensor
                                 )
            self.aug_y = tf.cond(self.is_training,
                                 lambda: tf.concat([y_tensor, aug_y], axis=0),
                                 lambda: y_tensor
                                 )

        elif self.mode == 'test':
            self.aug_x = self.input_transform
            self.aug_y = y_tensor
        else:
            pass

        _1, _2, logits = graph(
            self.aug_x,
            self.hidden_layers, self.output_dim,
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
    joint_dnn = JointDefense()
    joint_dnn.train()
    joint_dnn.mode = 'test'
    joint_dnn.test()

if __name__ == "__main__":
    _main()