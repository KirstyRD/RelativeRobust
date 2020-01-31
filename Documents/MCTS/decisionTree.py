#!/usr/bin/env python


import sys
from PIL import Image
import numpy as np
import imp
from basics import *
from networkBasics import *
from configuration import * 
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib as mpl


class decisionTree:

    def __init__(self, actions):
        self.actions = actions
        self.indexUpToNow = 1
        self.tree = {}
        self.tree[0] = []

    def addOnePath(self,dist,spansPath,numSpansPath): 
        alldims = spansPath.keys()
        currentNode = 0
        while alldims != []: 
            (act, alldims) = self.getAction(alldims,currentNode)
            nodeExists = False
            for (act2,nextNode) in self.tree[currentNode]: 
                if act == act2: 
                    currentNode = nextNode
                    nodeExists = True
                    break
                      
            if nodeExists == False: 
                self.indexUpToNow += 1
                self.tree[self.indexUpToNow] = []
                self.tree[currentNode].append((act,self.indexUpToNow))
                currentNode = self.indexUpToNow


    def getAction(self,dims,currentNode):
        existingDims = []
        if self.tree[currentNode] != []:
            for act in (zip (*self.tree[currentNode]))[0]: 
                existingDims += self.actions[act][0].keys()
        
        intersectSet = list(set(dims).intersection(set(existingDims)))
        if intersectSet == []: 
            e = dims[0]
        else: 
            e = intersectSet[0]
        for k,ls in self.actions.iteritems():
            if e in ls[1].keys() : 
                return (k, [ e2 for e2 in dims if e2 not in ls[1].keys() ])
        print "decisionTree: getAction: cannot find action "
        
    def show(self):
        print("decision tree: %s"%self.tree)
        
        import graphviz as gv
        import networkx as nx
        
        graph = gv.Digraph(format='svg')
        vNodes = {}
        for node in self.tree.keys():
            graph.node('%s'%(node))
            
        for node in self.tree.keys():
            for (act, nextNode) in self.tree[node]: 
                graph.edge('%s'%node, '%s'%nextNode)
        graph.render('data/decisionTree.gv')
        
