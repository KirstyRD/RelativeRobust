# Relative Robustness of Quantized Neural Networks Against Adversarial Attack

The scripts in this repository can be used to reproduce the results presented in [the paper](https://www.semanticscholar.org/paper/Relative-Robustness-of-Quantized-Neural-Networks-Duncan-Komendantskaya/d16e076610a8f329f8de557158d45d52d057562b) which was published in ICJNN 2020. 

**get_results.sh** simultaneously produces results for the relative robustness of a quantized neural network and the transfer of adversarial attacks from a full precision to a quantized network. Input parameters are:
1. Index of test dataset to begin results,
2. Index of test dataset to end,
3. Manipulation step size for robustness, out of 128.
4. Search radius, i.e. the maximum Manhattan distance between original image and perturbed images.

The original DLV tool and theory paper can be found [here](https://github.com/VeriDeep/DLV).

Requirements to run this portion of code are:
- Python 2.7 
- opencv>=2.4.x
- numpy==1.13
- skimage==0.14.x
- cvxopt>=1.2.x
- stopit>=1.1.x
- theano==0.9.0
- keras==1.2.2
- pySMT
- z3

[Z3 installation instructions](https://github.com/Z3Prover/z3)

Requirements for the quantization code are as follows:
- Python 3
- numpy==1.18.1
- tensorflow==1.14.0
- keras>=2.2.4
- skimage>=0.14.x

