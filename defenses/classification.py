"""classifier abstract class"""
from abc import ABCMeta, abstractmethod
import tensorflow as tf

class Classifier(object):
    """Abstract base class for all classifier classes."""
    __metaclass__ = ABCMeta
    def __init__(self,
                 hyper_params=None,
                 reuse=False,
                 is_saving=True,
                 init_graph=True,
                 mode='train',
                 name='BASIC_DNN'):
        self.is_saving = is_saving
        self.init_graph = init_graph
        self.reuse = reuse
        self.model_name = name

        try:
            assert mode == 'train' or mode == 'test'
        except:
            raise AssertionError("Two modes: 'train' or 'test', not both.")
        self.mode = mode
        self.hp_params_dict = hyper_params

    @abstractmethod
    def data_preprocess(self):
        """data preprocess such as minmax normalization"""
        raise NotImplementedError

    @abstractmethod
    def model_graph(self, reuse=False):
        "model definition"
        raise NotImplementedError
    @abstractmethod
    def forward(self, x_tensor, y_tensor, reuse = False):
        """forward procedure of DNN"""
        raise  NotImplementedError

    @abstractmethod
    def model_inference(self):
        """define loss function, accuracy, prediction, etc"""
        raise NotImplementedError

    @abstractmethod
    def train(self, trainX = None, trainy = None):
        """
        train a model upon (trainX, trainy), if value is none, default data will be leveraged
        :param trainX: np.2darray
        :param trainy: np.1darray
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, testX, gt_labels):
        """testing"""
        raise NotImplementedError