# Learning-to-Defend-by-Learning-to-Attack

This repository shares the code for the paper *[Learning to Defend by Learning to Attack](http://proceedings.mlr.press/v130/jiang21a/jiang21a.pdf)* in AISTATS 2021, by Haoming Jiang, Zhehui Chen, Yuyang Shi, Bo Dai and Tuo Zhao. 

- **train_l2l_1_cifar10.py** and **train_l2l_2_cifar10.py** are used for training the Grad-L2L and 2-Step L2L models over CIFAR10, respectively;
- **models** includes several network architectures for the classifier network;
- **attacker.py** includes the network architectures for the attacker network;
- **pgd_attack_cifar10.py** and **cw_attack_cifar10.py** perform two types or adversarial attack using PGD and CW method, respectively.

## Prerequisite
- Python3
- Pytorch
- CUDA
- numpy
