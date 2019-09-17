This code repository for the paper [Enhancing Robustness of Deep Neural Networks Against Adversarial Malware Samples: Principles, Framework, and AICS'2019 Challenge](https://arxiv.org/abs/1812.08108), Li et al., AICS 2019

## Overview
Four kinds of defenses are implemented against adversarial evasion attacks produced by the AICS 2019 organiser. The descriptions of the challenge problem are put in the folder `challenge`. More information can be found [here](http://www-personal.umich.edu/~arunesh/AICS2019/challenge.html).  

## Dependencies:
* python 2.7 or python 3.6
* Codes have been tested on tensorflow-gpu==1.9.0 and Tensorflow-gpu==1.14.0
* numpy >= 1.13.3
* scikit-Learn >= 0.18.2

## Usage && Files Descriptions
  We are required to change the `project_root` in the file of `conf` to accommodate the current absolute path.

## Run

1. Execute `main.py`
```
python main.py defender -tp
```
2. To reproduce the experiment results reported in the paper, we can execute `main.py`:
```
python main.py defender random_subspace -tp
```

Please follow the helper function in `main.py`, if you'd like to execute other defenses. All learned model will be saved into the current directory under `save` folder which can be reset in the file of `conf`


## Citation

If you'd like to cite us, please consider the following:

```
@inproceedings{li2019enhancing,
  title={Enhancing Robustness of Deep Neural Networks against Adversarial Malware Samples: Principles, Framework, and Application to AICS’2019 Challenge},
  author={Li, Deqiang and Li, Qianmu and Ye, Yanfang and Xu, Shouhuai},
  booktitle={The AAAI-19 Workshop on Artificial Intelligence for Cyber Security (AICS), 2019},
  year={2019}
}
```