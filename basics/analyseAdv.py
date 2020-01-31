from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import imp
from scipy.optimize import least_squares
from basics import l1Distance
from sklearn.cluster import KMeans
import copy
from basics import diffImage

class analyseAdv:

    def __init__(self,img):    
        self.image = img
        self.adv = []
        self.localdist = ("L1",0.03)
        self.numKmeans = 10
        self.advSets = []
        
        
    def addAdv(self,img): 
        self.adv.append(img)
        
    def analyse(self):
        print("%s adversarial examples are found."%(len(self.adv)))
        self.removeDuplicatedImages()
        print("%s adversarial examples are found."%(len(self.adv)))
        #self.splitImages(copy.deepcopy(self.adv))
        self.kmean(copy.deepcopy(self.adv))
        print("%s clusters are found."%(len(self.advSets)))
        typicalImages = []
        for set in self.advSets: 
            typicalImages.append((len(set),meanPoint(set)))
        return typicalImages
        
    def removeDuplicatedImages(self):
        i = 0
        while i < len(self.adv):  
            j = i + 1
            while j < len(self.adv): 
                if diffImage(self.adv[i],self.adv[j]) == []: 
                    self.adv.pop(j)    
                else: j += 1
            i += 1    
        
    def splitImages(self,remainImages):
        if remainImages != []: 
            newSet = [remainImages[0]]
            remainImages.pop(0)
            similarIndex = []
            for i in range(len(remainImages)): 
                if l1Distance(remainImages[i],self.image) < self.localdist[1] :
                    newSet.append(remainImages[i])
                    similarIndex.append(i)
            similarIndex = sorted(similarIndex, reverse=True)
            for i in similarIndex: 
                remainImages.pop(i)
            self.advSets.append(newSet)
            self.splitImages(remainImages)
            
    def kmean(self,imageSet):
        imageSet2 = []
        for set in imageSet: 
            imageSet2.append(list(set.flatten()))
        imageSet2 = np.array(imageSet2)
        kmeans = KMeans(n_clusters=min(self.numKmeans,len(imageSet)), random_state=0).fit(imageSet2)
        for i in range(max(kmeans.labels_)+1): 
            newSet = []
            for j in range(len(imageSet)): 
                if kmeans.labels_[j] == i: newSet.append(imageSet[j])
            self.advSets.append(newSet)


def meanPoint(imageSet): 
    return reduce((lambda x, y: np.add(x,y)), imageSet) / len(imageSet)

def argMMSE(imageSet):
            
    # define a function 
    code = "def f(x):\n"
    code += "    import math\n"
    code += "    return "
     
    str_i = ""
    for i in range(len(imageSet)): 
        str_j = ""
        img = imageSet[i].flatten()
        for j in range(img.size): 
            if str_j == "": 
                str_j += "(x[%s] - %s)**2"%(j,img[j])
            else: 
                str_j += " + (x[%s] - %s)**2"%(j,img[j])    
        str_j = "math.sqrt(%s)"%str_j
        if str_i == "": 
            str_i += str_j
        else: 
            str_i += "+ %s"%str_j
    code += str_i
    #print code

    # execute code
    argMMSEModule = imp.new_module('argMMSE')
    exec code in argMMSEModule.__dict__

    # compute least_squares
    current_opt = 1e30000
    res = least_squares(argMMSEModule.f, imageSet[0].flatten())
    if res.optimality < current_opt: 
        current_opt = res.optimality
        current_x = res.x
    
    print("First-order optimality measure: %s"%(current_opt))
    return np.reshape(current_x,imageSet[0].shape)
