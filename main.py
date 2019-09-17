from __future__ import print_function
import sys
from argparse import ArgumentParser

from common import Defender

def _main():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(title= 'Role selection')

    # creat arguments for defender
    defender_parser = subparsers.add_parser('defender', help = 'Defend the classifier (learner).')
    defender_parser.add_argument('-d', '--defense', dest='defense', type = str, default= 'basic_dnn',
                                 choices=['basic_dnn', 'feature_bnrz_dnn', 'adv_training_dnn', 'joint_defense',
                                          'random_subspace'], required= False)
    defender_parser.add_argument('-t', '--train', help = 'Training defense model', action= 'store_true', default=False)
    defender_parser.add_argument('-p', '--prediction', help = 'Predict labels for test samples (including adversarial version).',
                                action= 'store_true', default= False)

    defender_parser.set_defaults(action = 'defender')

    args = parser.parse_args()

    if args.action == 'defender':
        defender = Defender(args.defense)

        if args.train:
            defender.train()
        if args.prediction:
            defender.predict()


if __name__ == "__main__":
    sys.exit(_main())