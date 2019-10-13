from collections import defaultdict

from .basic_dnn import BasicDNN
from .feature_binarization_dnn import FeatureBinarizationDNN
from .adv_regularization import AdversarialTrainingRegDNN
from .joint_defense import JointDefense
from .dae_rpst_learning import DAE_RPST_DNN
from .ensemble_rsm_wrapper import RandomSubspaceMethod

defense_model_scope_dict = {
    'basic_dnn' : BasicDNN,
    'feature_bnrz_dnn' : FeatureBinarizationDNN,
    'adv_training_dnn' : AdversarialTrainingRegDNN,
    'joint_defense' : JointDefense,
    'dae_rpst_learn_dnn' : DAE_RPST_DNN,
    'random_subspace' : RandomSubspaceMethod
}

defense_model_scope_dict = defaultdict(**defense_model_scope_dict)