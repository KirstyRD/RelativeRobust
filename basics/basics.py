#!/usr/bin/env python


import numpy as np
import math
import time
import os
import copy
from keras import backend as K

def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

# to check whether a specific point is 
# inconsistent for network and curve
def checkCex(model,x):
    y_predicted = model.predict(np.array([x]))
    y_p = [chooseResult(y) for y in y_predicted]
    y_actual = mapping(x)

    if (y_p[0] == 1) and (y_actual[0] == False): 
        result = True
    elif (y_p[0] == 2) and (y_actual[0] == True): 
        result = True
    else: 
        result = False
    if result == True: 
        print "the point " + str(x) + " is a counterexample!"
    else: 
        print "error: the point " + str(x) + " is NOT a counterexample! please check ... "
    return result
    
def current_milli_time():
    return int(round(time.time() * 1000) % 4294967296)


def diffImage(image1,image2):
    return zip (*np.nonzero(np.subtract(image1,image2)))
    
    
def diffPercent(image1,image2): 
        return len(diffImage(image1,image2)) / float(image1.size)
        
def numDiffs(image1,image2): 
        return len(diffImage(image1,image2))
    

    
def euclideanDistance(image1,image2):
    return math.sqrt(np.sum(np.square(np.subtract(image1,image2))))
    
def l1Distance(image1,image2):
    return np.sum(np.absolute(np.subtract(image1,image2)))

def l0Distance(image1,image2):
    return np.count_nonzero(np.absolute(np.subtract(image1,image2)))

def normalisation(y):
    for k in range(len(y)): 
        if y[k] < 0: y[k] = 0 
    return [y[0]/(y[0]+y[1]),y[1]/(y[0]+y[1])]


def chooseResult(y): 
    [y0,y1] = normalisation(y)
    if y0 >= y1: return 1
    else: return 2
    
def addPlotBoxes(plt,boxes,c):
    if len(boxes) > 0: 
        for bb in boxes: 
            addPlotBox(plt,bb,c)
    
def addPlotBox(plt,bb,c): 
        x = [bb[0][0],bb[1][0],bb[1][0],bb[0][0],bb[0][0]]
        y = [bb[0][1],bb[0][1],bb[1][1],bb[1][1],bb[0][1]]
        plt.plot(x,y,c)
        
def equalActivations(activation1,activation2, pk):
    if activation1.shape == activation2.shape :
        if isinstance(activation1, np.float32) or isinstance(activation1, np.float64):
            return abs(activation1 - activation2) < pk
        else: 
            bl = True
            for i in range(len(activation1)):
                bl = bl and equalActivations(activation1[i],activation2[i], pk)
            return bl
    else: print("not the same shape of two activations.")
    
def mergeTwoDicts(x,y):
    z = x.copy()
    z.update(y)
    return z
    
    
############################################################
#
#  SIFT auxiliary functions
#
################################################################

def withKL(dist,const,image1,image2):

    import scipy
    dist1, dist2 = GMM(image1), GMM(image2)
    return dist + const * scipy.stats.entropy(dist1.flatten(),dist2.flatten())

    
 

def getDistribution(image, kp):

    import matplotlib.pyplot as plt
    import scipy
    from scipy.stats import multivariate_normal
    import scipy.stats
    import numpy.linalg
    
    dist = np.zeros(image.shape[:2])
    i = 1
    for  k in kp: 
        #print(i)
        i += 1
        a = np.array((k.pt[0],k.pt[1]))
        for i in range(len(dist)): 
            for j in range(len(dist[0])): 
                b = np.array((i,j))
                dist2 = numpy.linalg.norm(a - b)
                dist[i][j] += scipy.stats.norm.pdf(dist2, loc=0.0, scale=k.size) * k.response
                    
    return dist / np.sum(dist)
    
############################################################
#
#  auxiliary functions
#
################################################################
    
def getWeight(wv,bv,layerIndex):
    wv = [ (a,(p,c),w) for (a,(p,c),w) in wv if p == layerIndex ]
    bv = [ (p,c,w) for (p,c,w) in bv if p == layerIndex ]
    return (wv,bv)
    
def numberOfFilters(wv):
    return 1#np.amax((zip (*((zip (*wv))[1])))[1])

#  the features of the last layer
def numberOfFeatures(wv):
    return 1#np.amax((zip (*((zip (*wv))[0])))[1])
    
def otherPixels(image, ps):
    ops = []
    if len(image.shape) == 2: 
          for i in range(len(image)): 
              for j in range(len(image[0])): 
                  if (i,j) not in ps: ops.append((i,j))
    return ops
 
    
#######################################################
#
#  show detailedInformation or not
#  FIXME: check to see if they are really needed/used
#
#######################################################

def nprint(str):
    return      
        
    
