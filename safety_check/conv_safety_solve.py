#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

import numpy as np
import math
import ast
import copy
import random
import time
import multiprocessing
import stopit

from z3 import *

#import display
import mnist as mm

from scipy import ndimage

from configuration import *
from basics import *

inverseFunction = "point"


def conv_safety_solve(layer2Consider,nfeatures,nfilters,filters,bias,input,activations,prevSpan,prevNumSpan,span,numSpan,pk):

    random.seed(time.time())

    # number of clauses
    c = 0
    # number of variables
    d = 0

    # variables to be used for z3
    variable={}

    if nfeatures == 1: images = np.expand_dims(input, axis=0)
    else: images = input

    if len(activations.shape) == 3:
        avg = np.sum(activations)/float(len(activations)*len(activations[0])*len(activations[0][0]))
    elif len(activations0.shape) == 2:
        avg = np.sum(activations)/float(len(activations)*len(activations[0]))
    else: avg = 0

    s = Tactic('qflra').solver()
    s.reset()

    #print("%s\n%s\n%s\n%s"%(prevSpan,prevNumSpan,span,numSpan))

    toBeChanged = []
    if inverseFunction == "point":
        if nfeatures == 1:
            #print("%s\n%s"%(nfeatures,prevSpan.keys()))
            ks = [ (0,x,y) for (x,y) in prevSpan.keys() ]
        else: ks = copy.deepcopy(prevSpan.keys())
        toBeChanged = toBeChanged + ks
    elif inverseFunction == "area":
        for (k,x,y) in span.keys():
             toBeChanged = toBeChanged + [(l,x1,y1) for l in range(nfeatures) for x1 in range(x,x+filterSize) for y1 in range(y,y+filterSize) if x1 >= 0 and y1 >= 0 and x1 < images.shape[1] and y1 < images.shape[2]]
        toBeChanged = list(set(toBeChanged))

    for (l,x,y) in toBeChanged:
        variable[1,0,l+1,x,y] = Real('1_x_%s_%s_%s' % (l+1,x,y))
        d += 1
        if not(boundOfPixelValue == [0,0]) and (layer2Consider == 0):
            pstr = eval("variable[1,0,%s,%s,%s] <= %s"%(l+1,x,y,boundOfPixelValue[1]))
            pstr = And(eval("variable[1,0,%s,%s,%s] >= %s"%(l+1,x,y,boundOfPixelValue[0])), pstr)
            pstr = And(eval("variable[1,0,%s,%s,%s] != %s"%(l+1,x,y,images[l][x][y])), pstr)
            s.add(pstr)
            c += 1

    maxterms = ""
    for (k,x,y) in span.keys():
        variable[1,1,k+1,x,y] = Real('1_y_%s_%s_%s' % (k+1,x,y))
        d += 1
        string = "variable[1,1,%s,%s,%s] == "%(k+1,x,y)
        for l in range(nfeatures):
           for x1 in range(filterSize):
                for y1 in range(filterSize):
                    if (l,x+x1,y+y1) in toBeChanged:
                        newstr1 = " variable[1,0,%s,%s,%s] * %s + "%(l+1,x+x1,y+y1,filters[l,k][x1][y1])
                    elif x+x1 < images.shape[1] and y+y1 < images.shape[2] :
                        newstr1 = " %s + "%(images[l][x+x1][y+y1] * filters[l,k][x1][y1])
                    string += newstr1
        string += str(bias[l,k])
        s.add(eval(string))
        #print("***")
        #print(string)
        c += 1

        if enumerationMethod == "line":
            pstr = eval("variable[1,1,%s,%s,%s] < %s" %(k+1,x,y,activations[k][x][y] + span[(k,x,y)] * numSpan[(k,x,y)] + epsilon))
            pstr = And(eval("variable[1,1,%s,%s,%s] > %s "%(k+1,x,y,activations[k][x][y] + span[(k,x,y)] * numSpan[(k,x,y)] - epsilon)), pstr)
        elif enumerationMethod == "convex" or enumerationMethod == "point":
            if activations[k][x][y] + span[(k,x,y)] * numSpan[(k,x,y)] >= 0:
                upper = activations[k][x][y] + span[(k,x,y)] * numSpan[(k,x,y)] + pk
                lower = -1 * (activations[k][x][y] + span[(k,x,y)] * numSpan[(k,x,y)]) - pk
            else:
                upper = -1 * (activations[k][x][y] + span[(k,x,y)] * numSpan[(k,x,y)]) + pk
                lower = activations[k][x][y] + span[(k,x,y)] * numSpan[(k,x,y)] - pk

            #if span[(k,x,y)] > 0 :
            #    upper = activations[k][x][y] + span[(k,x,y)] * numSpan[(k,x,y)] + epsilon
            #    lower = activations[k][x][y] - span[(k,x,y)] * numSpan[(k,x,y)] - epsilon
            #else:
            #    upper = activations[k][x][y] - span[(k,x,y)] * numSpan[(k,x,y)] + epsilon
            #    lower = activations[k][x][y] + span[(k,x,y)] * numSpan[(k,x,y)] - epsilon

            #if span[(k,x,y)] > 0 :
            #    upper = activations[k][x][y] + span[(k,x,y)]  + epsilon
            #    lower = activations[k][x][y] - epsilon - span[(k,x,y)]
            #else:
            #    upper = activations[k][x][y] + epsilon - span[(k,x,y)]
            #    lower = activations[k][x][y] + span[(k,x,y)] - epsilon
            pstr = eval("variable[1,1,%s,%s,%s] < %s"%(k+1,x,y,upper))
            pstr = And(eval("variable[1,1,%s,%s,%s] > %s"%(k+1,x,y,lower)), pstr)
            pstr = And(eval("variable[1,1,%s,%s,%s] != %s"%(k+1,x,y,activations[k][x][y])), pstr)
        s.add(pstr)
        c += 1

        if activations[k][x][y] > 0 :
            maxterm = "- variable[1,1,%s,%s,%s]"%(k+1,x,y)
        else:
            maxterm = "variable[1,1,%s,%s,%s]"%(k+1,x,y)

        if maxterms == "":
            maxterms = "(%s)"%maxterm
        else:
            maxterms = " %s + (%s) "%(maxterms, maxterm)

    nprint("Number of variables: " + str(d))
    nprint("Number of clauses: " + str(c))

    p = multiprocessing.Process(target=s.check)
    p.start()

    # Wait for timeout seconds or until process finishes
    p.join(timeout)

    # If thread is still active
    if p.is_alive():
        print "Solver running more than timeout seconds (default="+str(timeout)+"s)! Skip it"
        p.terminate()
        p.join()
    else:
        s_return = s.check()

    if 's_return' in locals():
        if s_return == sat:
            inputVars = [ (l,x,y,eval("variable[1,0,"+ str(l+1) +"," + str(x) +"," + str(y)+ "]")) for (l,x,y) in toBeChanged ]
            cex = copy.deepcopy(images)
            for (l,x,y,v) in inputVars:
                #if cex[l][x][y] != v: print("different dimension spotted ... ")
                cex[l][x][y] = getDecimalValue(s.model()[v])
                #print("%s    %s"%(images[l][x][y],cex[l][x][y]))
            cex = np.squeeze(cex)

            #cex2 = copy.deepcopy(activations)
            #nextVars = [ (k,x,y,eval("variable[1,1,"+ str(k+1) +"," + str(x) +"," + str(y)+ "]")) for (k,x,y) in span.keys() ]
            #for (k,x,y,v) in nextVars:
                #if cex[l][x][y] != v: print("different dimension spotted ... ")
            #    cex2[k][x][y] = getDecimalValue(s.model()[v])
            #    print("%s       %s"%(activations[k][x][y],cex2[k][x][y]))

            nprint("satisfiable!")
            return (True, cex)
        else:
            nprint("unsatisfiable!")
            #***
            #return (True,cex)
            return (False, input)
    else:
        print "timeout! "
        return (False, input)

def getDecimalValue(v0):
    v = RealVal(str(v0))
    return float(v.numerator_as_long())/v.denominator_as_long()
