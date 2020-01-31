#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""

from network_configuration import *
from usual_configuration import *


#######################################################
#
#  The following are parameters to indicate how to work
#   with a problem
#
#######################################################

# which dataset to work with
#dataset = "twoDcurve"
#dataset = "mnist"
#dataset = "gtsrb"
dataset = "cifar10"
#dataset = "imageNet"

# the network is trained from scratch
#  or read from the saved files
whichMode = "read"
#whichMode = "train"

# work with a single image or a batch of images
dataProcessing = "single"
#dataProcessing = "batch"
dataProcessingBatchNum = 1


#######################################################
#
#   1. parameters related to the networks
#
#######################################################


(span,numSpan,errorBounds,boundOfPixelValue,NN,dataBasics,directory_model_string,directory_statistics_string,directory_pic_string,filterSize) = network_parameters(dataset)


#######################################################
#
#  2. parameters related to the experiments
#
#######################################################


(featureDims,startIndexOfImage,startLayer, maxLayer,numOfFeatures,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,enumerationMethod,checkingMode,exitWhen) = usual_configuration(dataset)


############################################################
#
#  3. other parameters that are believed to be shared among all cases
#
################################################################


# timeout for z3 to handle a run
timeout = 600

# the error bound for manipulation refinement
# between layers
epsilon = 0.1
