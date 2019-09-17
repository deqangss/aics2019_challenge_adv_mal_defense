from __future__ import print_function
import os
import sys
import logging

if sys.version_info[0] < 3:
    import ConfigParser as configparser
else:
    import configparser

config = configparser.SafeConfigParser()
config_dir = os.path.dirname(__file__)
get = config.get

def parser_config():
    config_file = os.path.join(config_dir, "conf")

    if not os.path.exists(config_file):
        sys.stderr.write("Error: Unable to find the config file!")
        sys.exit(1)

    #parse the configuration
    global config
    config.readfp(open(config_file))

parser_config()

os.environ["CUDA_VISIBLE_DEVICES"] = config.get('common', 'gpu_idx')

logging.basicConfig(level=logging.INFO,filename=os.path.join(config_dir, "log"),filemode="w",
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
ErrorHandler = logging.StreamHandler()
ErrorHandler.setLevel(logging.WARNING)
ErrorHandler.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'))