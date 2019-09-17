"""
Basic DNN for malware classification
"""

import os
import sys
from datetime import datetime
from timeit import default_timer
import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from defenses.classification import *
from dataset.dataset import load_trainset, load_testset
from dataset.input_preprocessing import normalize_data, random_over_sampling
from utils import utils
from config import config, logging

logger = logging.getLogger("basic_dnn")


def graph(x_input, hidden_neurons=[160, 160], output_dim=5, is_training=True, name="BASIC_DNN",
          reuse=False):
    '''
    the defined architectures of nerual network
    :param x_input: Tensor
    :param hidden_neurons: neurons for hidden layers
    :param output_dim: int
    :param is_training: training or not
    :param name: string, nets name
    :param reuse: reuse or not
    :return: the defined graph of neural networks
    '''
    with tf.variable_scope("{}".format(name), reuse=reuse):

        # dense layer #1 ~ #len(hidden_layers)
        # the neuron unit is layer_neurons[0]
        # input Tensor shape: [batch_size, input_dim]
        # output Tensor shape:[batch_size, hidden_neurons[0]]
        dense1 = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                 name="DENSE1")

        # dense layer #2
        # the neuron unit is layer_neurons[1]
        # input Tensor shape: [batch_size, hidden_neurons[0]]
        # output Tensor shape:[batch_size, hidden_neurons[1]]
        dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[1], activation=tf.nn.relu, name="DENSE2")

        # bottlenect
        # dense layer #3
        # the neron unit output_dim
        # input Tensor shape: [batch_size, hidden_neurons[1]]
        # output Tensor shape: [batch_size, output_dim]
        dense3 = tf.layers.dense(inputs=dense2, units=output_dim, activation=None, name="DENSE3")

        return dense1, dense2, dense3

def tester(sess, testX, testy, model, eval_dir):
    """
    model testing
    :param sess: tf.Session
    :param testX: test data, type: 2-D float np.ndarry
    :param testy: test groud truth label, type: 1-D int np.ndarray
    :param model: trained model inherited from classifier
    :param save_dir: result save path
    :return:
    """
    test_input = utils.DataProducer(testX, testy, batch_size=20, name='test')

    with sess.as_default():
        _accuracy = 0
        _preds = []
        for _, x, y in test_input.next_batch():
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            _y_pred, _acc= sess.run([model.y_pred, model.accuracy], feed_dict=test_dict)
            _preds.append(_y_pred)
            _accuracy += _acc
        accuracy = _accuracy / test_input.mini_batches
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='accuracy', simple_value=accuracy)
        ])
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        summary_writer = tf.summary.FileWriter(eval_dir)
        summary_writer.add_summary(summary)
        preds = np.concatenate(_preds)
        from sklearn.metrics import f1_score
        return accuracy, f1_score(testy, preds, average= 'macro')

DNN_HP = {
    'random_seed': 23456,
    'hidden_units': [160, 160], # number of layers is fixed
    'output_dim': 5,
    'n_epochs': 30,
    'batch_size': 128,
    'learning_rate':0.001,
    'optimizer': 'adam' # others are not supported
}

class BasicDNN(Classifier):
    def __init__(self,
                 hyper_params=None,
                 reuse=False,
                 is_saving=True,
                 init_graph=True,
                 mode='train',
                 name='BASIC_DNN'):
        super(BasicDNN, self).__init__()
        self.is_saving = is_saving
        self.init_graph = init_graph
        self.reuse = reuse
        self.model_name = name

        try:
            assert mode == 'train' or mode == 'test'
        except:
            raise AssertionError("Two modes: 'train' or 'test', not both.")
        self.mode = mode
        if hyper_params is not None:
            self.hp_params_dict = hyper_params
            self.hp_params = utils.ParamWrapper(hyper_params)
        else:
            self.hp_params_dict = DNN_HP
            self.hp_params = utils.ParamWrapper(DNN_HP)

        if self.is_saving:
            self.save_dir = config.get("experiments", self.model_name.lower())

        if not (os.path.exists(config.get('dataset', 'dataX')) and
                os.path.exists(config.get('dataset', 'datay')) and
                os.path.exists(config.get('dataset', 'normalizer'))):
            dataX, datay = self.data_preprocess()
            utils.dump_joblib(dataX, config.get('dataset', 'dataX'))
            utils.dump_joblib(datay, config.get('dataset', 'datay'))
        self.normalizer = utils.read_joblib(config.get('dataset', 'normalizer'))

        # DNN based model
        self.input_dim = len(self.normalizer.data_min_)

        self.hidden_layers = self.hp_params.hidden_units
        self.output_dim = self.hp_params.output_dim
        tf.set_random_seed(self.hp_params.random_seed)
        if self.init_graph:
            self.model_graph(reuse=reuse)

    def data_preprocess(self):
        '''
        normalize the data
        :returns: two tuples: X and y
        '''
        train_data, train_label = load_trainset()
        # normalization
        trainX = normalize_data(train_data, is_fitting=True)

        test_data, test_label, prist_data_idx, adv_data_idx = \
            load_testset()
        testX = normalize_data(test_data)
        prist_testX = testX[prist_data_idx]
        prist_testy = test_label[prist_data_idx]
        adv_testX = testX[adv_data_idx]
        adv_testy = test_label[adv_data_idx]
        return (trainX, prist_testX, adv_testX), (train_label, prist_testy, adv_testy)

    def model_graph(self, reuse=False):
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='X')
        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None, ], name='Y')
        self.is_training = tf.placeholder(tf.bool, name="TRAIN")

        tf.set_random_seed(self.hp_params.random_seed)
        self.logits, self.y_tensor = self.forward(self.x_input, self.y_input, reuse=reuse)
        self.model_inference()

    def forward(self, x_tensor, y_tensor, reuse = False):
        """
        define model inference
        :param x_tensor: input data
        :type: Tensor.float32
        :param y_tensor: label
        :type: Tensor.int64
        :param reuse: Boolean
        :return: Null
        """
        self.nn = graph
        _1, _2, logits = graph(
            x_tensor, self.hidden_layers, self.output_dim,
            self.is_training, name=self.model_name, reuse=reuse
        )

        return logits, y_tensor

    def model_inference(self):
        # loss definition
        self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(
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

    def train(self, trainX = None, trainy = None, is_sampling = False):
        """train dnn based malware detector"""
        if trainX is None and trainy is None:
            trainX, _, _ = utils.read_joblib(config.get('dataset', 'dataX'))
            trainy, _, _ = utils.read_joblib(config.get('dataset', 'datay'))

        if is_sampling:
            trainX, trainy = random_over_sampling(trainX, trainy, ratio=0.3)

        train_input_supervised = utils.DataProducer(trainX, trainy,
                                                    self.hp_params.batch_size, n_epochs=self.hp_params.n_epochs)

        saver = tf.train.Saver(max_to_keep=10)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.cross_entropy)
        merged_summaries = tf.summary.merge_all()
        global_train_step = tf.train.get_or_create_global_step()

        # optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.cross_entropy,
                                                                                      global_step=global_train_step)
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
                sess.run(optimizer, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
        sess.close()

    def test(self, testX = None, testy = None):
        prist_testX = None
        adv_testX = None
        if testX is None and testy is None:
            _, prist_testX, adv_testX = utils.read_joblib(config.get('dataset', 'dataX'))
            _, prist_testy, adv_testy = utils.read_joblib(config.get('dataset', 'datay'))

            testX = np.concatenate((prist_testX, adv_testX))
            testy = np.concatenate((prist_testy, adv_testy))
        if len(testX) == 0:
            print("No test data.")
            return

        self.mode = 'test'

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()

        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        if cur_checkpoint is None:
            print("No saved parameters")
            return

        saver = tf.train.Saver()
        eval_dir = os.path.join(self.save_dir, 'eval')
        sess = tf.Session()
        with sess:
            saver.restore(sess, cur_checkpoint)
            accuracy, macro_f1_score = tester(sess, testX, testy, self, eval_dir)
            MSG = "The accuracy on the test dataset is {:.5f}%"
            print(MSG.format(accuracy * 100))
            logger.info(MSG.format(accuracy * 100))

            if prist_testX is not None and adv_testX is not None:
                print("Other evaluation metrics we may need:")
                prist_acc, prist_f1_socre = tester(sess, prist_testX, prist_testy, self, eval_dir)
                adv_acc, adv_f1_score = tester(sess, adv_testX, adv_testy, self, eval_dir)
                harmonic_f1_score = utils.harmonic_mean(prist_f1_socre, adv_f1_score)
                MSG = "The accuracy on pristine test datasest is {:.5f}% vs. {:.5f}% on adversarial data."
                print(MSG.format(prist_acc * 100, adv_acc * 100))
                logger.info(MSG.format(prist_acc * 100, adv_acc * 100))
                MSG = "The macro f1 score on pristine test datasest is {:.5f}% vs. {:.5f}% on adversarial data."
                print(MSG.format(prist_f1_socre * 100, adv_f1_score * 100))
                logger.info(MSG.format(prist_f1_socre * 100, adv_f1_score * 100))
                MSG = "Harmonic macro F1 score is {:.5f}%"
                print(MSG.format(harmonic_f1_score * 100))
                logger.info(MSG.format(harmonic_f1_score * 100))

            sess.close()
        return accuracy

def _main():
    basic_dnn_model = BasicDNN()
    basic_dnn_model.train()
    basic_dnn_model.mode = 'test'
    basic_dnn_model.test()

if __name__ == "__main__":
    sys.exit(_main())