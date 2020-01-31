#!/usr/bin/env python

"""
main file

author: Xiaowei Huang
"""

import sys
sys.path.append('networks')
sys.path.append('safety_check')
sys.path.append('configuration')
sys.path.append('basics')
sys.path.append('MCTS')

import time
import numpy as np
import copy
import random
#import matplotlib as mlp
#mlp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

from loadData import loadData
from regionSynth import regionSynth, initialiseRegion
from precisionSynth import precisionSynth
from safety_analysis import safety_analysis

from configuration import *
from basics import *
from networkBasics import *

from searchTree import searchTree
from searchMCTS import searchMCTS
from mcts_twoPlayer import mcts_twoPlayer
from initialiseSiftKeypoints import GMM
from dataCollection import dataCollection

from inputManipulation import applyManipulation,assignManipulationSimple

from keras import backend as K

def main():

	model = loadData()
#    sys.exit()
	dc = dataCollection()

    #file name number
	picnum = 0



		#if originalClass != realClass:

		#row = str(sys.argv[1]) + "," + str(originalClass) + "," + str(sys.argv[2]) + "," + str(sys.argv[3]) + "," + str(sys.argv[4]) + "\n"
		#csv.write(row)
		#csv.close()

    # handle a set of inputs starting from an index
	succNum = 0
	for whichIndex in range(startIndexOfImage,startIndexOfImage + dataProcessingBatchNum):
		#check original image correct
		image = NN.getImage(model,startIndexOfImage)
		(originalClass,originalConfident) = NN.predictWithImage(model,image)
		realClass = NN.getClass(model,startIndexOfImage)

		#only process if classified correctly
		if realClass == originalClass:
			print "\n\nprocessing input of index %s in the dataset: " %(str(whichIndex))
			(succ,picnum) = handleOne(model,dc,whichIndex,picnum)
			if succ == True: succNum += 1
		else:
			print "\n\n Original image misclassified."
	dc.addSuccPercent(succNum/float(dataProcessingBatchNum))
	dc.provideDetails()
	dc.summarise()
	dc.close()

	#output="Results.csv"
	#csv = open(output, "a")


###########################################################################
#
# safety checking
# starting from the a specified hidden layer
#
############################################################################

## how many branches to expand
numOfPointsAfterEachFeature = 1

#mcts_mode  = "sift_twoPlayer"
mcts_mode  = "singlePlayer"


def handleOne(model,dc,startIndexOfImage,picnum):

	# get an image to interpolate
	global np
	image = NN.getImage(model,startIndexOfImage)
#    print("the shape of the input is "+ str(image.shape))

    #if dataset == "twoDcurve": image = np.array([3.58747339,1.11101673])
	if dataset == "twoDcurve": image = np.array([3.0,1.11101673])

	dc.initialiseIndex(startIndexOfImage)
	originalImage = copy.deepcopy(image)

	if checkingMode == "stepwise":
		k = startLayer
	elif checkingMode == "specificLayer":
		k = maxLayer

	while k <= maxLayer:

		layerType = getLayerType(model, k)
		re = False
		start_time = time.time()

        # only these layers need to be checked
		if layerType in ["Convolution2D","Conv2D", "Dense", "InputLayer"] and k >= 0 :

			dc.initialiseLayer(k)
			st = searchTree(image,k)
			st.addImages(model,[image])

#            print("\n================================================================")
#            print "\nstart checking the safety of layer "+str(k)

			(originalClass,originalConfident) = NN.predictWithImage(model,image)
			origClassStr = dataBasics.LABELS(int(originalClass))

			path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,originalConfident)
			dataBasics.saveim(-1,originalImage, path0)

            # for every layer
			f = 0
			while f < numOfFeatures :

				f += 1
#                print("\n================================================================")
#                print("Round %s of layer %s for image %s"%(f,k,startIndexOfImage))
				index = st.getOneUnexplored()
				imageIndex = copy.deepcopy(index)

				# for every image
                # start from the first hidden layer
				t = 0
				re = False
				while True and index != (-1,-1):

                    # pick the first element of the queue
#                    print "(1) get a manipulated input ..."
					(image0,span,numSpan,numDimsToMani,_) = st.getInfo(index)

#                    print "current layer: %s."%(t)
#                    print "current index: %s."%(str(index))

					path2 = directory_pic_string+"/temp"+str(index)+".png"
#                    print "current operated image is saved into %s"%(path2)
					dataBasics.saveim(index[0],image0,path2)

#                    print "(2) synthesise region from %s..."%(span.keys())
                     # ne: next region, i.e., e_{k+1}
                    #print "manipulated: %s"%(st.manipulated[t])
					(nextSpan,nextNumSpan,numDimsToMani) = regionSynth(model,dataset,image0,st.manipulated[t],t,span,numSpan,numDimsToMani)
					st.addManipulated(t,nextSpan.keys())

                    #print "nextSpan.keys(): %s"%(nextSpan.keys())

					(nextSpan,nextNumSpan,npre) = precisionSynth(model,image0,t,span,numSpan,nextSpan,nextNumSpan)

#                    print "dimensions to be considered: %s"%(nextSpan)
#                    print "spans for the dimensions: %s"%(nextNumSpan)

					if t == k:

                        # only after reaching the k layer, it is counted as a pass
#                        print "(3) safety analysis ..."
                        # wk for the set of counterexamples
                        # rk for the set of images that need to be considered in the next precision
                        # rs remembers how many input images have been processed in the last round
                        # nextSpan and nextNumSpan are revised by considering the precision npre
						(nextSpan,nextNumSpan,rs,wk,rk,picnum) = safety_analysis(model,dataset,t,startIndexOfImage,st,index,nextSpan,nextNumSpan,npre,picnum)
                        #print"span: "+str(nextSpan)+" numspan: "+str(nextNumSpan)
						if len(rk) > 0:
							rk = (zip (*rk))[0]

#                            print "(4) add new images ..."
							random.seed(time.time())
							if len(rk) > numOfPointsAfterEachFeature:
								rk = random.sample(rk, numOfPointsAfterEachFeature)
							diffs = diffImage(image0,rk[0])
#                            print("the dimensions of the images that are changed in the this round: %s"%diffs)
							if len(diffs) == 0:
								st.clearManipulated(k)
								return

							st.addImages(model,rk)
							st.removeProcessed(imageIndex)
							(re,percent,eudist,l1dist,l0dist) = reportInfo(image,wk)
#                            print "euclidean distance %s"%(euclideanDistance(image,rk[0]))
#                            print "L1 distance %s"%(l1Distance(image,rk[0]))
#                            print "L0 distance %s"%(l0Distance(image,rk[0]))
#                            print "manipulated percentage distance %s\n"%(diffPercent(image,rk[0]))
							break
						else:
							st.removeProcessed(imageIndex)
							break
					else:
#                        print "(3) add new intermediate node ..."
						index = st.addIntermediateNode(image0,nextSpan,nextNumSpan,npre,numDimsToMani,index)
						re = False
						t += 1
				if re == True:
					dc.addManipulationPercentage(percent)
#                    print "euclidean distance %s"%(eudist)
#                    print "L1 distance %s"%(l1dist)
#                    print "L0 distance %s"%(l0dist)
#                    print "manipulated percentage distance %s\n"%(percent)
					dc.addEuclideanDistance(eudist)
					dc.addl1Distance(l1dist)
					dc.addl0Distance(l0dist)
					(ocl,ocf) = NN.predictWithImage(model,wk[0])
					dc.addConfidence(ocf)
					break

#				if f == numOfFeatures:
#                print "(6) no adversarial example is found in this layer within the distance restriction."
				st.destructor()


		elif layerType in ["Input"]  and k < 0 and mcts_mode  == "singlePlayer" :

#            print "directly handling the image ... "

			dc.initialiseLayer(k)

			(originalClass,originalConfident) = NN.predictWithImage(model,image)
			origClassStr = dataBasics.LABELS(int(originalClass))
			path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,originalConfident)
			dataBasics.save(-1,originalImage, path0)

			'''************************************************************************'''

            # initialise a search tree
			st = searchMCTS(model,image,k)
			st.initialiseActions()

			start_time_all = time.time()
			runningTime_all = 0
			numberOfMoves = 0
			while st.terminalNode(st.rootIndex) == False and st.terminatedByControlledSearch(st.rootIndex) == False and runningTime_all <= MCTS_all_maximal_time:
#                print("the number of moves we have made up to now: %s"%(numberOfMoves))
				eudist = st.euclideanDist(st.rootIndex)
				l1dist = st.l1Dist(st.rootIndex)
				l0dist = st.l0Dist(st.rootIndex)
				percent = st.diffPercent(st.rootIndex)
				diffs = st.diffImage(st.rootIndex)
#                print "euclidean distance %s"%(eudist)
#                print "L1 distance %s"%(l1dist)
#                print "L0 distance %s"%(l0dist)
#                print "manipulated percentage distance %s"%(percent)
#                print "manipulated dimensions %s"%(diffs)

				start_time_level = time.time()
				runningTime_level = 0
				childTerminated = False
				while runningTime_level <= MCTS_level_maximal_time:
#                    print(runningTime_level)
					(leafNode,availableActions) = st.treeTraversal(st.rootIndex)
					newNodes = st.initialiseExplorationNode(leafNode,availableActions)
					for node in newNodes:
						(childTerminated, value) = st.sampling(node,availableActions)
						if childTerminated == True: break
						st.backPropagation(node,value)
					if childTerminated == True: break
					runningTime_level = time.time() - start_time_level
#                    print("best possible one is %s"%(st.showBestCase()))
				bestChild = st.bestChild(st.rootIndex)
                #st.collectUselessPixels(st.rootIndex)
				st.makeOneMove(bestChild)

				image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
				diffs = st.diffImage(st.rootIndex)
				path0="%s/%s_temp_%s.png"%(directory_pic_string,startIndexOfImage,len(diffs))
				pathK="%s/%s/%s.png"%(directory_pic_string,startIndexOfImage,numberOfMoves)
				dataBasics.saveim(-1,image1,path0)
				(newClass,newConfident) = NN.predictWithImage(model,image1)
#                print "confidence: %s"%(newConfident)

				if childTerminated == True: break

                # store the current best
				(_,bestSpans,bestNumSpans) = st.bestCase
				image1 = applyManipulation(st.image,bestSpans,bestNumSpans)
				path0="%s/%s/currentBest.png"%(directory_pic_string,startIndexOfImage)
				dataBasics.saveim(-1,image1,path0)

				runningTime_all = time.time() - runningTime_all
				numberOfMoves += 1

			(_,bestSpans,bestNumSpans) = st.bestCase
           #image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
			image1 = applyManipulation(st.image,bestSpans,bestNumSpans)
			(newClass,newConfident) = NN.predictWithImage(model,image1)
			newClassStr = dataBasics.LABELS(int(newClass))
			re = newClass != originalClass
			path0="%s/%sa/%s_modified_into_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,newClassStr)
			dataBasics.saveim(-1,image1,path0)
            #print np.max(image1), np.min(image1)
#            print("difference between images: %s"%(diffImage(image,image1)))
            #plt.imshow(image1 * 255, cmap=mpl.cm.Greys)
            #plt.show()

			if re == True:
				eudist = euclideanDistance(st.image,image1)
				l1dist = l1Distance(st.image,image1)
				l0dist = l0Distance(st.image,image1)
				percent = diffPercent(st.image,image1)
#                print "euclidean distance %s"%(eudist)
#                print "L1 distance %s"%(l1dist)
#                print "L0 distance %s"%(l0dist)
#                print "manipulated percentage distance %s"%(percent)
#                print "class is changed into %s with confidence %s\n"%(newClassStr, newConfident)
				dc.addEuclideanDistance(eudist)
				dc.addl1Distance(l1dist)
				dc.addl0Distance(l0dist)
				dc.addManipulationPercentage(percent)

			#else:
					#row = str(sys.argv[1]) + "," + str(originalClass) + "," + str(sys.argv[2]) + "," + str(sys.argv[3]) + "," + str(sys.argv[4]) + "\n"
					#csv.write(row)
					#csv.close()

			st.destructor()


#		else:
#            print("layer %s is of type %s, skipping"%(k,layerType))
            #return

		runningTime = time.time() - start_time
		dc.addRunningTime(runningTime)
		if re == True and exitWhen == "foundFirst":
			break
		k += 1

#    print("Please refer to the file %s for statistics."%(dc.fileName))
	if re == True:
		return (True,picnum)
	else: return (False,picnum)


def reportInfo(image,wk):

    # exit only when we find an adversarial example
	if wk == []:
		print "(5) no adversarial example is found in this round."
		return (False,0,0,0,0)
	else:
		print "(5) an adversarial example has been found."
		image0 = wk[0]
		eudist = euclideanDistance(image,image0)
		l1dist = l1Distance(image,image0)
		l0dist = l0Distance(image,image0)
		percent = diffPercent(image,image0)
		return (True,percent,eudist,l1dist,l0dist)

if __name__ == "__main__":

	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))
