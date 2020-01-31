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
import scipy.io as sio
import matplotlib as mpl


class re_training:

    def __init__(self,model,dataShapeX):
        self.originalModel = model
        self.xtrain = []
        # temporary setting for classification
        # the class the image should belong to
        self.ytrain = []
        # the class the image is misclassified into
        self.ztrain = []
        # re-training parameters
        self.batch_size = 128
        self.nb_epoch = 12
        self.reTrainedModelName = ""

        # save model
        #ae =  "_retrained_%s"%(self.reTrainedModelName)
        #json_string = self.originalModel.to_json()
        #open('%s/%s%s.json'%(directory_model_string,dataset,ae), 'w').write(json_string)
        #self.originalModel.save_weights('%s/%s%s.h5'%(directory_model_string,dataset,ae), overwrite=True)
        #sio.savemat('%s/%s%s.mat'%(directory_model_string,dataset,ae), {'weights': self.originalModel.get_weights()})

    def addDatum(self,xdata,ydata,newdata):
        if len(xdata.shape) == 2:
            self.xtrain.append(np.expand_dims(xdata,axis=0))
            self.ytrain.append(ydata)
            self.ztrain.append(newdata)
        else:
            self.xtrain.append(xdata)
            self.ytrain.append(ydata)
            self.ztrain.append(newdata)


    def addData(self,xdata,ydata):
        self.xtrain += xdata
        self.ytrain += ydata

    def returnData(self):
        return self.xtrain, self.ytrain

    def numberOfNewExamples(self):
        return len(self.xtrain)

    def setReTrainedModelName(self, str):
        self.reTrainedModelName = str

    def training(self):

        xtrain = np.array(self.xtrain)
        ytrain = np.array(self.ytrain)

        if dataset == "mnist":
            (X_train, Y_train, X_test, Y_test, batch_size, nb_epoch) = NN.read_dataset()
        elif dataset == "cifar10":
            (X_train,Y_train,X_test,Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()


        nb_classes = Y_train.shape[1]
        ytrain2 = []
        for y in ytrain:
            temp_y = np.zeros_like(Y_train[0])
            temp_y[y] = 1
            ytrain2.append(temp_y)

        xtrain2 = np.append(X_train, np.array(xtrain), axis=0)
        ytrain2 = np.append(Y_train, np.array(ytrain2), axis=0)

        print xtrain2.shape, ytrain2.shape, X_train.shape, Y_train.shape

        ae =  "" #"_retrained_%s"%(self.reTrainedModelName)
        if dataset == "mnist":
            model = NN.read_model_from_file('%s/%s%s.mat'%(directory_model_string,dataset,ae),'%s/%s%s.json'%(directory_model_string,dataset,ae))
        elif dataset == "cifar10":
            model = NN.read_model_from_file(img_channels, img_rows, img_cols, nb_classes, '%s/cifar10%s.mat'%(directory_model_string,ae),'%s/cifar10%s.json'%(directory_model_string,ae))


        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        model.fit(xtrain2, ytrain2, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=1)

        # save model
        ae =  "_retrained_%s"%(self.reTrainedModelName)
        json_string = model.to_json()
        open('%s/%s%s.json'%(directory_model_string,dataset,ae), 'w').write(json_string)
        model.save_weights('%s/%s%s.h5'%(directory_model_string,dataset,ae), overwrite=True)
        sio.savemat('%s/%s%s.mat'%(directory_model_string,dataset,ae), {'weights': model.get_weights()})

        return model

    def evaluateWithOriginalModel(self):

        if dataset == "mnist":
            (X_train, Y_train, X_test, Y_test, batch_size, nb_epoch) = NN.read_dataset()
        elif dataset == "cifar10":
            (X_train,Y_train,X_test,Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()

        self.originalModel.compile(loss='categorical_crossentropy',
                                   optimizer='adadelta',
                                   metrics=['accuracy'])
        score = self.originalModel.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
        scoreReport = '%s %s'%(score,self.originalModel.metrics_names)
        return scoreReport

    def evaluateWithUpdatedModel(self):
        if dataset == "mnist":
            (X_train, Y_train, X_test, Y_test, batch_size, nb_epoch) = NN.read_dataset()
            ae =  "_retrained_%s"%(self.reTrainedModelName)
            model = NN.read_model_from_file('%s/%s%s.mat'%(directory_model_string,dataset,ae),'%s/%s%s.json'%(directory_model_string,dataset,ae))

        elif dataset == "cifar10":
            (X_train,Y_train,X_test,Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()
            ae =  "_retrained_%s"%(self.reTrainedModelName)
            model = NN.read_model_from_file(img_channels, img_rows, img_cols, nb_classes, '%s/%s%s.mat'%(directory_model_string,dataset,ae),'%s/%s%s.json'%(directory_model_string,dataset,ae))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
        scoreReport = '%s %s'%(score,model.metrics_names)
        return scoreReport

    def saveToMat(self,path):
        import scipy.io as sio
        data = []
        label = []
        for n in range(len(self.xtrain)):
            data.append(self.xtrain[n].flatten())
            label.append(self.ztrain[n].flatten())
        data = np.array(data)
        label = np.array(label)
        #(label,data) = self.sortData(label,data)
        data_complete = {}
        data_complete["X_test_adv"] = data
        data_complete["L_test_adv"] = label
        sio.savemat(path,data_complete)

    def sortData(self,label,data):
        sortedLabel = []
        sortedData = []
        for l in sorted(label):
            for i in range(len(label)):
                if label[i] == l:
                     sortedLabel.append(label[i])
                     sortedData.append(data[i])
        return (sortedLabel,sortedData)
                
