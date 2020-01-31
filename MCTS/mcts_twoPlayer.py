#!/usr/bin/env python

"""
A data structure for organising search
author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy
import sys
import operator
import random
import math

from configuration import *

from inputManipulation import applyManipulation
from basics import *

from decisionTree import decisionTree
from initialiseSiftKeypoints import initialiseSiftKeypointsTwoPlayer

from re_training import re_training


effectiveConfidenceWhenChanging = 0
explorationRate = math.sqrt(2)


class mcts_twoPlayer:

    def __init__(self, model, autoencoder, image, activations, layer, player_mode):
        self.image = image
        self.activations = activations
        self.model = model
        self.autoencoder = autoencoder

        # current layer
        self.layer = layer
        self.manipulationType = "sift_twoPlayer"
        self.player_mode = player_mode

        (self.originalClass,self.originalConfident) = self.predictWithActivations(self.activations)

        self.collectImages = 1
        self.collectedImages = []

        self.cost = {}
        self.numberOfVisited = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}

        self.indexToNow = 0
        # current root node
        self.rootIndex = 0

        self.decisionTree = 0
        self.re_training = re_training(model,self.image.shape)

        self.spans = {}
        self.numSpans = {}
        # initialise root node
        self.spans[-1] = {}
        self.numSpans[-1] = {}
        self.initialiseLeafNode(0,-1,[],[])

        # record all the keypoints: index -> kp
        self.keypoints = {}
        # mapping nodes to keyponts
        self.keypoint = {}
        self.keypoint[0] = 0

        # local actions
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

        # best case
        self.bestCase = (0,{},{})
        self.numConverge = 0

        # number of adversarial exmaples
        self.numAdv = 0
        #self.analyseAdv = analyseAdv(activations)

        # useless points
        self.usefulPixels = {}

        # temporary variables for sampling
        self.spansPath = []
        self.numSpansPath = []
        self.depth = 0
        self.availableActionIDs = []
        self.usedActionIDs = []
        self.accDims = []
        self.d =0

    def predictWithActivations(self,activations):
        if self.layer > -1:
            output = np.squeeze(self.autoencoder.predict(np.expand_dims(activations,axis=0)))
            return NN.predictWithImage(self.model,output)
        else:
            return NN.predictWithImage(self.model,activations)

    def visualizationMCTS(self):
        for k in range(len(self.activations)):
            activations1 = copy.deepcopy(self.activations)
            # use a random node to replace the feature node
            emptyNode = np.zeros_like(self.activations[0])
            activations1[k] = emptyNode
            output = np.squeeze(self.autoencoder.predict(np.expand_dims(activations1,axis=0)))
            path0="%s/%s_autoencoder_%s.png"%(directory_pic_string,startIndexOfImage,k)
            dataBasics.save(-1,output, path0)

    def setManipulationType(self,typeStr):
        self.manipulationType = typeStr

    def initialiseActions(self):
        # initialise actions according to the type of manipulations
        if self.manipulationType == "sift_twoPlayer":
            actions = initialiseSiftKeypointsTwoPlayer(self.autoencoder,self.activations,[])
            self.keypoints[0] = 0
            i = 1
            for k in actions[0]:
                self.keypoints[i] = k
                i += 1
            #print self.keypoints
        else:
            print("unknown manipulation type")
            exit

        #print("actions=%s"%(actions.keys()))
        for i in range(len(actions)):
            ast = {}
            for j in range(len(actions[i])):
                ast[j] = actions[i][j]
            self.actions[i] = ast
        nprint("%s actions have been initialised. "%(len(self.actions)))
        # initialise decision tree
        #self.decisionTree = decisionTree(self.model, self.actions, self.activations, "decision")

    def initialiseLeafNode(self,index,parentIndex,newSpans,newNumSpans):
        nprint("initialising a leaf node %s from the node %s"%(index,parentIndex))
        self.spans[index] = mergeTwoDicts(self.spans[parentIndex],newSpans)
        self.numSpans[index] = mergeTwoDicts(self.numSpans[parentIndex],newNumSpans)
        self.cost[index] = 0
        self.parent[index] = parentIndex
        self.children[index] = []
        self.fullyExpanded[index] = False
        self.numberOfVisited[index] = 0
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        self.re_training.addDatum(activations1,self.originalClass,self.originalClass)


    def destructor(self):
        self.image = 0
        self.activations = 0
        self.model = 0
        self.autoencoder = 0
        self.spans = {}
        self.numSpans = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}

        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

    # move one step forward
    # it means that we need to remove children other than the new root
    def makeOneMove(self,newRootIndex):
        print("making a move into the new root %s, whose value is %s and visited number is %s"%(newRootIndex,self.cost[newRootIndex],self.numberOfVisited[newRootIndex]))
        self.removeChildren(self.rootIndex,[newRootIndex])
        self.rootIndex = newRootIndex

    def removeChildren(self,index,indicesToAvoid):
        if self.fullyExpanded[index] == True:
            for childIndex in self.children[index]:
                if childIndex not in indicesToAvoid: self.removeChildren(childIndex,[])
        self.spans.pop(index,None)
        self.numSpans.pop(index,None)
        self.cost.pop(index,None)
        self.parent.pop(index,None)
        self.keypoint.pop(index,None)
        self.children.pop(index,None)
        self.fullyExpanded.pop(index,None)
        self.numberOfVisited.pop(index,None)

    def bestChild(self,index):
        allValues = {}
        for childIndex in self.children[index]:
            allValues[childIndex] = self.cost[childIndex] / float(self.numberOfVisited[childIndex])
        nprint("finding best children from %s"%(allValues))
        return max(allValues.iteritems(), key=operator.itemgetter(1))[0]

    def treeTraversal(self,index):
        if self.fullyExpanded[index] == True:
            nprint("tree traversal on node %s"%(index))
            allValues = {}
            for childIndex in self.children[index]:
                allValues[childIndex] = (self.cost[childIndex] / float(self.numberOfVisited[childIndex])) + explorationRate * math.sqrt(math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex]))
            #nextIndex = max(allValues.iteritems(), key=operator.itemgetter(1))[0]
            if self.player_mode == "adversary" and self.keypoint[index] == 0 :
                allValues2 = {}
                for k,v in allValues.iteritems():
                     allValues2[k] = 1 / float(allValues[k])
                nextIndex = np.random.choice(allValues.keys(), 1, p = [ x/sum(allValues2.values()) for x in allValues2.values()])[0]
            else:
                nextIndex = np.random.choice(allValues.keys(), 1, p = [ x/sum(allValues.values()) for x in allValues.values()])[0]

            if self.keypoint[index] in self.usedActionsID.keys() and self.keypoint[index] != 0 :
                self.usedActionsID[self.keypoint[index]].append(self.indexToActionID[index])
            elif self.keypoint[index] != 0 :
                self.usedActionsID[self.keypoint[index]] = [self.indexToActionID[index]]
            return self.treeTraversal(nextIndex)
        else:
            nprint("tree traversal terminated on node %s"%(index))
            availableActions = copy.deepcopy(self.actions)
            for k in self.usedActionsID.keys():
                for i in self.usedActionsID[k]:
                    availableActions[k].pop(i, None)
            return (index,availableActions)

    def initialiseExplorationNode(self,index,availableActions):
        nprint("expanding %s"%(index))
        if self.keypoint[index] != 0:
            for (actionId, (span,numSpan,_)) in availableActions[self.keypoint[index]].iteritems() : #initialisePixelSets(self.model,self.image,list(set(self.spans[index].keys() + self.usefulPixels))):
                self.indexToNow += 1
                self.keypoint[self.indexToNow] = 0
                self.indexToActionID[self.indexToNow] = actionId
                self.initialiseLeafNode(self.indexToNow,index,span,numSpan)
                self.children[index].append(self.indexToNow)
        else:
            for kp in self.keypoints.keys() : #initialisePixelSets(self.model,self.image,list(set(self.spans[index].keys() + self.usefulPixels))):
                self.indexToNow += 1
                self.keypoint[self.indexToNow] = kp
                self.indexToActionID[self.indexToNow] = 0
                self.initialiseLeafNode(self.indexToNow,index,{},{})
                self.children[index].append(self.indexToNow)
        self.fullyExpanded[index] = True
        self.usedActionsID = {}
        return self.children[index]

    def backPropagation(self,index,value):
        self.cost[index] += value
        self.numberOfVisited[index] += 1
        if self.parent[index] in self.parent :
            #nprint("start backPropagating the value %s from node %s, whose parent node is %s"%(value,index,self.parent[index]))
            self.backPropagation(self.parent[index],value)
        #else:
            #nprint("backPropagating ends on node %s"%(index))


    def analysePixels(self):
        #self.usefulPixels = self.decisionTree.collectUsefulPixels()
        usefulPixels = []
        for index in self.usefulPixels:
            usefulPixels.append(self.actions[index])
        #print("%s useful pixels = %s"%(len(self.usefulPixels),self.usefulPixels))
        values = self.usefulPixels.values()
        images = []
        for v in values:
            pixels = [ dim for key, value in self.usefulPixels.iteritems() for dim in self.actions[key][0].keys() if value >= v ]
            ndim = self.image.ndim
            usefulimage = copy.deepcopy(self.image)
            span = {}
            numSpan = {}
            for p in pixels:
                span[p] = 1
                numSpan[p] = 1
            #usefulimage = applyManipulation(usefulimage,span,numSpan)
            #'''
            if ndim == 2:
                for x in range(len(usefulimage)):
                    for y in range(len(usefulimage[0])):
                        if (x,y) not in pixels:
                            usefulimage[x][y] = 0
            elif ndim == 3:
                for x in range(len(usefulimage)):
                    for y in range(len(usefulimage[0])):
                        for z in range(len(usefulimage[0][0])):
                            if (x,y,z) not in pixels:
                                usefulimage[x][y][z] = 0
            #'''
            images.append((v,usefulimage))
        return images

    def addUsefulPixels(self,dims):
        for dim in dims:
            if dim in self.usefulPixels.keys():
                self.usefulPixels[dim] += 1
            else:
                self.usefulPixels[dim] = 1

    def getUsefulPixels(self,accDims,d):
        import operator
        sorted_accDims = sorted(self.accDims, key=operator.itemgetter(1), reverse=True)
        needed_accDims = sorted_accDims[:d-1]
        self.addUsefulPixels([x for (x,y) in needed_accDims])

    # start random sampling and return the Euclidean value as the value
    def sampling(self,index,availableActions):
        #nprint("start sampling node %s"%(index))
        availableActions2 = copy.deepcopy(availableActions)
        #print(availableActions,self.keypoint[index],self.indexToActionID[index])
        availableActions2[self.keypoint[index]].pop(self.indexToActionID[index], None)
        sampleValues = []
        i = 0
        for i in range(MCTS_multi_samples):
            self.spansPath = self.spans[index]
            self.numSpansPath = self.numSpans[index]
            self.depth = 0
            self.availableActionIDs = {}
            for k in self.keypoints.keys():
                self.availableActionIDs[k] = availableActions2[k].keys()
            self.usedActionIDs = {}
            for k in self.keypoints.keys():
                self.usedActionIDs[k] = []
            self.accDims = []
            self.d = 2
            (childTerminated, val) = self.sampleNext(self.keypoint[index])
            sampleValues.append(val)
            #if childTerminated == True: break
            i += 1
        return (childTerminated, max(sampleValues))

    def sampleNext(self,k):
        #print("k=%s"%k)
        #for j in self.keypoints:
        #    print(len(self.availableActionIDs[j]))
        #print("oooooooo")

        activations1 = applyManipulation(self.activations,self.spansPath,self.numSpansPath)
        (newClass,newConfident) = self.predictWithActivations(activations1)
        (distMethod,distVal) = controlledSearch
        if distMethod == "euclidean":
            dist = euclideanDistance(activations1,self.activations)
            termValue = 0.0
            termByDist = dist > distVal
        elif distMethod == "L1":
            dist = l1Distance(activations1,self.activations)
            termValue = 0.0
            termByDist = dist > distVal
        elif distMethod == "Percentage":
            dist = diffPercent(activations1,self.activations)
            termValue = 0.0
            termByDist = dist > distVal
        elif distMethod == "NumDiffs":
            dist =  diffPercent(activations1,self.activations) * self.activations.size
            termValue = 0.0
            termByDist = dist > distVal

        #if termByDist == False and newConfident < 0.5 and self.depth <= 3:
        #    termByDist = True

        #self.re_training.addDatum(activations1,self.originalClass,newClass)


        if newClass != self.originalClass and newConfident > effectiveConfidenceWhenChanging:
            # and newClass == dataBasics.next_index(self.originalClass,self.originalClass):
            nprint("sampling a path ends in a terminal node with self.depth %s... "%self.depth)

            #print("L1 distance: %s"%(l1Distance(self.activations,activations1)))
            #print(self.activations.shape)
            #print(activations1.shape)
            #print("L1 distance with KL: %s"%(withKL(l1Distance(self.activations,activations1),self.activations,activations1)))

            (self.spansPath,self.numSpansPath) = self.scrutinizePath(self.spansPath,self.numSpansPath,newClass)

            #self.decisionTree.addOnePath(dist,self.spansPath,self.numSpansPath)
            self.numAdv += 1
            #self.analyseAdv.addAdv(activations1)
            self.getUsefulPixels(self.accDims,self.d)

            self.re_training.addDatum(activations1,self.originalClass,newClass)
            if self.bestCase[0] < dist:
                self.numConverge += 1
                self.bestCase = (dist,self.spansPath,self.numSpansPath)
                path0="%s/%s_currentBest_%s_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,self.numConverge,dataBasics.LABELS(int(newClass)),newConfident)
                dataBasics.save(-1,activations1,path0)

            return (self.depth == 0, dist)
        elif termByDist == True:
            nprint("sampling a path ends by controlled search with self.depth %s ... "%self.depth)
            self.re_training.addDatum(activations1,self.originalClass,newClass)
            return (self.depth == 0, termValue)
        elif list(set(self.availableActionIDs[k])-set(self.usedActionIDs[k])) == []:
            nprint("sampling a path ends with self.depth %s because no more actions can be taken ... "%self.depth)
            return (self.depth == 0, termValue)
        else:
            #print("continue sampling node ... ")
            #allChildren = initialisePixelSets(self.model,self.activations,self.spansPath.keys())
            randomActionIndex = random.choice(list(set(self.availableActionIDs[k])-set(self.usedActionIDs[k]))) #random.randint(0, len(allChildren)-1)
            if k == 0:
                span = {}
                numSpan = {}
            else:
                (span,numSpan,_) = self.actions[k][randomActionIndex]
                self.availableActionIDs[k].remove(randomActionIndex)
                self.usedActionIDs[k].append(randomActionIndex)
            newSpanPath = self.mergeSpan(self.spansPath,span)
            newNumSpanPath = self.mergeNumSpan(self.numSpansPath,numSpan)
            activations2 = applyManipulation(self.activations,newSpanPath,newNumSpanPath)
            (newClass2,newConfident2) = self.predictWithActivations(activations2)
            confGap2 = newConfident - newConfident2
            if newClass2 == newClass:
                self.accDims.append((randomActionIndex,confGap2))
            else: self.accDims.append((randomActionIndex,1.0))

            self.spansPath = newSpanPath
            self.numSpansPath = newNumSpanPath
            self.depth = self.depth+1
            self.accDims = self.accDims
            self.d = self.d
            if k == 0:
                return self.sampleNext(randomActionIndex)
            else:
                return self.sampleNext(0)

    def scrutinizePath(self,spanPath,numSpanPath,changedClass):
        lastSpanPath = copy.deepcopy(spanPath)
        for i in self.actions.keys():
            if i != 0:
                for key, (span,numSpan,_) in self.actions[i].iteritems():
                    if set(span.keys()).issubset(set(spanPath.keys())):
                        tempSpanPath = copy.deepcopy(spanPath)
                        tempNumSpanPath = copy.deepcopy(numSpanPath)
                        for k in span.keys():
                            tempSpanPath.pop(k)
                            tempNumSpanPath.pop(k)
                        activations1 = applyManipulation(self.activations,tempSpanPath,tempNumSpanPath)
                        (newClass,newConfident) = self.predictWithActivations(activations1)
                        #if changedClass == newClass:
                        if newClass != self.originalClass and newConfident > effectiveConfidenceWhenChanging:
                            for k in span.keys():
                                spanPath.pop(k)
                                numSpanPath.pop(k)
        if len(lastSpanPath.keys()) != len(spanPath.keys()):
            return self.scrutinizePath(spanPath,numSpanPath,changedClass)
        else:
            return (spanPath,numSpanPath)

    def terminalNode(self,index):
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        (newClass,_) = self.predictWithActivations(activations1)
        return newClass != self.originalClass

    def terminatedByControlledSearch(self,index):
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        (distMethod,distVal) = controlledSearch
        if distMethod == "euclidean":
            dist = euclideanDistance(activations1,self.activations)
        elif distMethod == "L1":
            dist = l1Distance(activations1,self.activations)
        elif distMethod == "Percentage":
            dist = diffPercent(activations1,self.activations)
        elif distMethod == "NumDiffs":
            dist = diffPercent(activations1,self.activations)
        nprint("terminated by controlled search")
        return dist > distVal

    def applyManipulationToGetImage(self,spans,numSpans):
        activations1 = applyManipulation(self.activations,spans,numSpans)
        if self.layer > -1:
            return np.squeeze(self.autoencoder.predict(np.expand_dims(activations1,axis=0)))
        else:
            return activations1

    def euclideanDist(self,index):
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return euclideanDistance(self.activations,activations1)

    def l1Dist(self,index):
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return l1Distance(self.activations,activations1)

    def l0Dist(self,index):
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return l0Distance(self.activations,activations1)

    def diffImage(self,index):
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return diffImage(self.activations,activations1)

    def diffPercent(self,index):
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return diffPercent(self.activations,activations1)

    def mergeSpan(self,spansPath,span):
        return mergeTwoDicts(spansPath, span)

    def mergeNumSpan(self,numSpansPath,numSpan):
        return mergeTwoDicts(numSpansPath, numSpan)

    def showDecisionTree(self):
        self.decisionTree.show()
        self.decisionTree.outputTree()
