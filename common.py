from time import sleep
from defenses import BasicDNN, FeatureBinarizationDNN, AdversarialTrainingRegDNN, \
    DAE_RPST_DNN, JointDefense, RandomSubspaceMethod

from config import logging

logger = logging.getLogger("common")

class Defender(object):
    def __init__(self, defense_method_name = 'adv_training_dnn'):
        self.defense_method_name = defense_method_name
        self.defense = None
        if self.defense_method_name == 'basic_dnn':
            self.defense = BasicDNN()
        elif self.defense_method_name == 'feature_bnrz_dnn':
            self.defense = FeatureBinarizationDNN()
        elif self.defense_method_name == 'adv_training_dnn':
            self.defense = AdversarialTrainingRegDNN()
        elif self.defense_method_name == 'dae_rpst_dnn':
            self.defense = DAE_RPST_DNN()
        elif self.defense_method_name == 'joint_defense':
            self.defense = JointDefense()
        elif self.defense_method_name == 'random_subspace':
            self.defense = RandomSubspaceMethod()
        else:
            raise ValueError(
                "Please choose method from 'basic_dnn', 'feature_bnrz_dnn', 'adv_training_dnn', 'dae_rpst_dnn', 'joint_defense', and 'random_subspace'.")

    def train(self):
        self.defense.mode = 'train'
        MSG = "Train {}.".format(self.defense_method_name)
        print(MSG)
        logger.info(MSG)
        MSG = "The hyper-parameters are defined as:\n {}".format(self.defense.hp_params_dict)
        print(MSG)
        logger.info(MSG)
        sleep(10)
        self.defense.train(is_sampling = True)

    def predict(self, apks=None, gt_lables=None):
        self.defense.mode = 'test'
        if apks is None and gt_lables is None:
            self.defense.test()
        else:
            self.defense.test(apks, gt_lables)
