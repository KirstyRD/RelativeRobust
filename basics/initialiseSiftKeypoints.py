#!/usr/bin/env python

"""
author: Xiaowei Huang

"""

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from keras import backend as K
from scipy.stats import truncnorm, norm

from basics import *
from networkBasics import *
from configuration import *
import collections

############################################################
#
#  initialise SIFT-based manipulations for two-player game
#
################################################################

if dataset == "imageNet":
    imageEnlargeProportion = 1
    maxNumOfPointPerKeyPoint = 100
else:
    imageEnlargeProportion = 2
    maxNumOfPointPerKeyPoint = 0

# should be local_featureDims, but make it 1
local_featureDims = 1


def initialiseSiftKeypointsTwoPlayer(model,image,manipulated):

    image1 = copy.deepcopy(image)
    if np.max(image1) <= 1:
        image1 = (image1*255).astype(np.uint8)
    else:
        image1 = image1.astype(np.uint8)
    #if len(image1.shape) > 2:
    #    image1 = (image1*255).transpose(1, 2, 0)
    if max(image1.shape) < 100 and K.backend() == 'theano' and len(image1.shape) == 3 :
        image1 = image1.reshape(image1.shape[1],image1.shape[2],image1.shape[0])
        image1 = cv2.resize(image1, (0,0), fx=imageEnlargeProportion, fy=imageEnlargeProportion)
        kp = SIFT_Filtered_twoPlayer(image1)
        image1 = image1.reshape(image1.shape[2],image1.shape[0],image1.shape[1])
    #elif max(image1.shape) < 100 and (K.backend() == 'tensorflow' or (K.backend() == 'theano' and len(image1.shape) == 2)):
    #    image1 = cv2.resize(image1, (0,0), fx=imageEnlargeProportion, fy=imageEnlargeProportion)
    #    kp = SIFT_Filtered_twoPlayer(image1)
    elif K.backend() == 'theano' and len(image1.shape) == 3:
        #kp, des = SIFT_Filtered(image1)
        image1 = image1.reshape(image1.shape[1],image1.shape[2],image1.shape[0])
        kp = SIFT_Filtered_twoPlayer(image1)
        image1 = image1.reshape(image1.shape[2],image1.shape[0],image1.shape[1])
    else:
        #kp, des = SIFT_Filtered(image1)
        kp = SIFT_Filtered_twoPlayer(image1)

    print("%s keypoints are found. "%(len(kp)))

    actions = {}
    actions[0] = kp
    s = 1
    kp2 = []
    if len(image1.shape) == 2:
        image0 = np.zeros(image1.shape)
    #elif K.backend() == 'tensorflow':
    #    image0 = np.zeros(image1.shape[:2])
    else:
        image0 = np.zeros(image1.shape[1:])
    numOfmanipulations = 0
    points_all = getPoints_twoPlayer(image0, kp)
    print("The pixels are partitioned with respect to keypoints.")
    for k, points in points_all.iteritems():
        allRegions = []
        for i in range(len(points)):
        #     print kp[i].pt
            points[i] = (points[i][0]/imageEnlargeProportion, points[i][1]/imageEnlargeProportion)
        points = list(set(points))
        num = len(points)/local_featureDims  # numOfFeatures
        i = 0
        while i < num :
            nextSpan = {}
            nextNumSpan = {}
            ls = []
            for j in range(local_featureDims):
                x = int(points[i*local_featureDims + j][0])
                y = int(points[i*local_featureDims + j][1])
                if image0[x][y] == 0 and len(image1.shape) == 2:
                    ls.append((x,y))
                #elif image0[x][y] == 0 and K.backend() == 'tensorflow':
                #    ls.append((x,y,0))
                #    ls.append((x,y,1))
                #    ls.append((x,y,2))
                elif image0[x][y] == 0 and K.backend() == 'theano':
                    ls.append((0,x,y))
                    ls.append((1,x,y))
                    ls.append((2,x,y))
                image0[x][y] = 1

            if len(ls) > 0:
                for j in ls:
                    nextSpan[j] = span
                    nextNumSpan[j] = numSpan
                oneRegion = (nextSpan,nextNumSpan,local_featureDims)
                allRegions.append(oneRegion)
            i += 1
        actions[s] = allRegions
        kp2.append(kp[s-1])
        s += 1
        print("%s manipulations have been initialised for keypoint (%s,%s), whose response is %s."%(len(allRegions), int(kp[k-1].pt[0]/imageEnlargeProportion), int(kp[k-1].pt[1]/imageEnlargeProportion),kp[k-1].response))
        numOfmanipulations += len(allRegions)
    actions[0] = kp2
    print("the number of all manipulations initialised: %s\n"%(numOfmanipulations))
    return actions

def SIFT_Filtered_twoPlayer(image): #threshold=0.0):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = image.astype(np.uint8)
    #equ = cv2.equalizeHist(image)
    #image = np.hstack((image,equ))
    sift = cv2.xfeatures2d.SIFT_create()  #cv2.SURF(400) #

    kp, des = sift.detectAndCompute(image,None)
    if len(kp) == 0 :
        print("cannot find keypoint, please try other approach ... ")
        sys.exit()
    else:
        return  kp

def getPoints_twoPlayer(image, kps):
    import operator
    import random
    points = {}
    if dataset != "imageNet":
        for x in range(max(image.shape)):
            for y in range(max(image.shape)):
                ps = 0
                maxk = -1
                for i in range(1, len(kps)+1):
                   k = kps[i-1]
                   dist2 = np.linalg.norm(np.array([x,y]) - np.array([k.pt[0],k.pt[1]]))
                   ps2 = norm.pdf(dist2, loc=0.0, scale=k.size)
                   if ps2 > ps:
                       ps = ps2
                       maxk = i
                #maxk = max(ps.iteritems(), key=operator.itemgetter(1))[0]
                if maxk in points.keys():
                    points[maxk].append((x,y))
                else: points[maxk] = [(x,y)]
        if maxNumOfPointPerKeyPoint > 0:
            for mk in points.keys():
                beginingNum = len(points[mk])
                for i in range(beginingNum - maxNumOfPointPerKeyPoint):
                    points[mk].remove(random.choice(points[mk]))
        return points
    else:
        kps = kps[:200]
        eachNum = max(image.shape) ** 2 / len(kps)
        maxk = 1
        points[maxk] = []
        for x in range(max(image.shape)):
            for y in range(max(image.shape)):
                if len(points[maxk]) <= eachNum:
                    points[maxk].append((x,y))
                else:
                    maxk += 1
                    points[maxk] = [(x,y)]
        return points





'''

def getPoints_twoPlayer(image, dist, kps, n):
    dist1 = np.zeros(dist.shape)
    for k in kps:
        a = np.array((k.pt[0],k.pt[1]))
        for i in range(len(dist)):
            for j in range(len(dist[0])):
                b = np.array((i,j))
                dist2 = np.linalg.norm(a - b)
                dist2 = scipy.stats.norm.pdf(dist2, loc=0.0, scale=k.size)
                if dist2 < k.size:
                    dist1[i][j] = dist[i][j]
    dist1 = dist1 / np.sum(dist1)

    indices = [] #np.zeros(dist.shape)
    for i in range(len(dist)):
        for j in range(len(dist[0])):
            indices.append(i*len(dist)+j) # [i][j] = (i,j)
    l =  np.random.choice(indices, n, p = dist1.flatten())
    l2 = []
    for ind in l:
        l2.append(getPixelLoc(ind,image))
    return list(set(l2))
'''

############################################################
#
#  initialise SIFT-based manipulations for single-player game
#
################################################################


def initialiseSiftKeypoints(model,image,manipulated):

    if len(image.shape) == 3:
        numOfPoints = image.size / min(image.shape)
    else:
        numOfPoints = image.size

    image1 = copy.deepcopy(image)
    if np.max(image1) <= 1:
        image1 = (image1*255).astype(np.uint8)
    else:
        image1 = image1.astype(np.uint8)
    #if len(image1.shape) > 2:
    #    image1 = (image1*255).transpose(1, 2, 0)
    if max(image1.shape) < 100:
        image1 = cv2.resize(image1, (0,0), fx=imageEnlargeProportion, fy=imageEnlargeProportion)
        #kp, des = SIFT_Filtered(image1)
        dist, kp = SIFT_Filtered(image1,numOfPoints)
        for i in range(len(kp)):
        #     print kp[i].pt
            kp[i] = (kp[i][0]/imageEnlargeProportion, kp[i][1]/imageEnlargeProportion)
            #print "%s:%s"%(i,des[i])
    else:
        dist, kp = SIFT_Filtered(image1,numOfPoints)

    print("%s keypoints are found. "%(len(kp)))

    allRegions = []
    num = len(kp)/local_featureDims  # numOfFeatures
    i = 0
    while i < num :
        nextSpan = {}
        nextNumSpan = {}
        ls = []
        for j in range(local_featureDims):
            x = int(kp[i*local_featureDims + j][0])
            y = int(kp[i*local_featureDims + j][1])
            if len(image1.shape) == 2:
                ls.append((x,y))
            ##elif K.backend() == 'tensorflow':
            #    ls.append((x,y,0))
            #    ls.append((x,y,1))
            #    ls.append((x,y,2))
            else:
                ls.append((0,x,y))
                ls.append((1,x,y))
                ls.append((2,x,y))

        for j in ls:
            nextSpan[j] = span
            nextNumSpan[j] = numSpan

        oneRegion = (nextSpan,nextNumSpan,local_featureDims)
        allRegions.append(oneRegion)
        i += 1
    print("%s manipulations have been initialised."%(len(allRegions)))
    return allRegions


def SIFT_Filtered(image, numOfPoints): #threshold=0.0):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = image.astype(np.uint8)
    #equ = cv2.equalizeHist(image)
    #image = np.hstack((image,equ))
    sift = cv2.SIFT() # cv2.SURF(400) #    cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    return  getPoints(image, kp, numOfPoints)

    #print kp[0], kp[0].response, kp[0].pt, kp[0].class_id, kp[0].octave, kp[0].size, len(des[0])
    '''
    if len(kp) == 0:
        print("There is no keypont found in the image. \nPlease try approaches other than SIFT in processing this image. ")
        sys.exit()

    actions = sorted(zip(kp,des), key=lambda x: x[0].response)

    return zip(*actions)
    '''


def getPoints(image, kp, n):
    dist = getDistribution(image, kp)
    indices = [] #np.zeros(dist.shape)
    for i in range(len(dist)):
        for j in range(len(dist[0])):
            indices.append(i*len(dist)+j) # [i][j] = (i,j)
    l =  np.random.choice(indices, n, p = dist.flatten())
    l2 = []
    for ind in l:
        l2.append(getPixelLoc(ind,image))
    return dist, list(set(l2))
    #print("value = %s"%(dist))
    #print np.max(dist)
    #print dist.flatten().reshape(dist.shape)


def getPixelLoc(ind, image):
    return (ind/len(image), ind%len(image))

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


    '''
        mean = [k.pt[0], k.pt[1]]
        cov = [0, k.size], [k.size, 0]

        var = scipy.stats.multivariate_normal(mean,cov)
        print var.pdf([1,1])
        x, y = np.random.multivariate_normal(mean, cov, 5000).T


        numpy.linalg.norm()
        stats.norm.pdf(0, loc=0.0, scale=k.size)

        plt.plot(x, y, 'x')
        plt.axis('equal')
        plt.show()\
    '''


def CreateGaussianMixtureModel(image, kp, dimension=0):


    if(len(kp) == 0):
        myclip_a = 0
        myclip_b = 28
        my_mean = 10
        my_std = 3

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        x_range = np.linspace(myclip_a,myclip_b,28)
        sampled = truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std)

        return sampled, 28

    shape = image.shape
    observations = 0
    if (dimension == 0):
        observations = shape[1]
        index_to_use = 1
    else:
        observations = shape[0]
        index_to_use = 0
    distributions = []
    sum_of_weights = 0
    for k in kp:
        mu, sigma = int(round(k.pt[index_to_use])), k.size

        myclip_a = 0
        myclip_b = observations
        my_mean = mu
        my_std = sigma

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        x_range = np.linspace(myclip_a,myclip_b,observations)
        lamb = truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std/2)

        distributions.append(lamb)
        sum_of_weights += k.response
    gamma = []
    for k in kp:
        gamma.append(k.response/sum_of_weights)
    A = []
    sum_of_densitys = 0
    #print("observations: %s distributions: %s "%(observations, len(distributions)))
    # here we assume that the shape returns a size and not a highest index... may be problematic
    for i in range(observations-1):
        prob_of_observation = 0
        for d in distributions:
            prob_of_observation = prob_of_observation + d[i]
        A.append(prob_of_observation)
        sum_of_densitys = sum_of_densitys + prob_of_observation

    A = np.divide(A, np.sum(A))
    if(np.sum(A) != 1):
        val_to_add = 1 - np.sum(A)
        #NEED TO GET MAX INDEX HERE
        A[np.argmax(A)] = A[np.argmax(A)] + val_to_add
    return A, observations


def GMM(image):

    import cv2
    sift = cv2.SIFT() # cv2.SURF(400) #    cv2.xfeatures2d.SIFT_create()
    image1 = copy.deepcopy(image)
    if np.max(image1) <= 1:
        image1 = (image1*255).astype(np.uint8)
    else:
        image1 = image1.astype(np.uint8)

    if dataset == "imageNet":
        imageEnlargeProportion = 1
    else:
        imageEnlargeProportion = 2

    if max(image1.shape) < 100 and K.backend() == 'theano' and len(image1.shape) == 3 :
        image1 = image1.reshape(image1.shape[1],image1.shape[2],image1.shape[0])
        image1 = cv2.resize(image1, (0,0), fx=imageEnlargeProportion, fy=imageEnlargeProportion)
        kp, des = sift.detectAndCompute(image1,None)
        image1 = image1.reshape(image1.shape[2],image1.shape[0],image1.shape[1])
    #elif max(image1.shape) < 100 and (K.backend() == 'tensorflow' or (K.backend() == 'theano' and len(image1.shape) == 2)):
    #    image1 = cv2.resize(image1, (0,0), fx=imageEnlargeProportion, fy=imageEnlargeProportion)
    #    kp, des = sift.detectAndCompute(image1,None)
    elif K.backend() == 'theano' and (image1.shape) == 3:
        #kp, des = SIFT_Filtered(image1)
        image1 = image1.reshape(image1.shape[1],image1.shape[2],image1.shape[0])
        kp, des = sift.detectAndCompute(image1,None)
        image1 = image1.reshape(image1.shape[2],image1.shape[0],image1.shape[1])
    else:
        #kp, des = SIFT_Filtered(image1)
        kp, des = sift.detectAndCompute(image1,None)

    return  getDistribution(image1, kp)
