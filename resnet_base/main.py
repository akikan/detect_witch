# -*- coding: utf-8 -*-
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
import keras
from ResNetModel import ResNet
import os
import glob
import time
import sys
import cv2
import numpy as np
from keras import optimizers

# 入力画像の次元とチャンネル
img_rows, img_cols, img_channels = 48, 48, 3

batch_size = 33
num_classes = 11
epochs = 50000

def getData(pathsAndLabels,shuffle):
    allData = []
    labelDevide = []
    c=1
    min_size = 1000000
    #get mini size
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + "*.jpg")
        if min_size > len(imagelist):
            min_size = len(imagelist)

    #
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + "*.jpg")
        if shuffle==1:
            imagelist = np.random.permutation(imagelist)
        #テストデータと学習データに分ける
        for imgName in imagelist:
            if c<min_size:
                labelDevide.append([imgName, label])
            c += 1
        c=1
        allData.append(labelDevide)
        labelDevide = []
    return allData

def getAllData(Data, Label):
    retD=[]
    retL=[]
    for i in range(len(Label)):
        retD.extend(Data[i])
        retL.extend(Label[i])
    return np.asarray(retD),np.asarray(retL)

def forwardCalc(allData,width=0,height=0, par = 0.8):

    imageDatas=[]
    labelDatas= []
    unk_imageDatas=[]
    unk_labelDatas= []
    for labelDevide in allData:
        length = len(labelDevide)
        c=1
        imageData=[]
        labelData= []
        unk_imageData=[]
        unk_labelData= []
        for pathAndLabel in labelDevide:
            tempLabel= np.zeros(num_classes)
            filepath = pathAndLabel[0].translate(str.maketrans('\\', '/'))

            # image_r = tf.read_file(filepath)
            # images = tf.image.decode_image(image_r, channels=3)
            
            img2 = cv2.imread(filepath)
            img=cv2.resize(img2,(img_rows,img_rows))
            img=img.astype(np.float32)/255.0
            for x in range(len(img)):
                img[x] = 1-img[x]
            tempLabel[int(pathAndLabel[1])] = 1 

            if par > (float(c/length)):
                imageData.append(img)#18チャンネルimgdata
                labelData.append(tempLabel)
                # labelData.append(np.float32(pathAndLabel[1]))
            else:
                unk_imageData.append(img)# = np.asarray(image)
                unk_labelData.append(tempLabel)
                # unk_labelData.append(np.float32(pathAndLabel[1]))
                # unk_tempLabel[(c-1)] = np.int32(pathAndLabel[1])
                
            c+=1
        imageDatas.append(imageData)
        labelDatas.append(labelData)
        unk_imageDatas.append(unk_imageData)
        unk_labelDatas.append(unk_labelData)
        # labelData.append(tempLabel)
        # unk_labelData.append(unk_tempLabel)
    imageDatas = np.asarray(imageDatas)
    labelDatas = np.asarray(labelDatas, dtype=np.int32)
    unk_imageDatas = np.asarray(unk_imageDatas)
    unk_labelDatas = np.asarray(unk_labelDatas, dtype=np.int32)
    return imageDatas,labelDatas,unk_imageDatas,unk_labelDatas

pathsAndLabels=[]
for i in range(0,11):
    pathsAndLabels.append(["./"+str(i+1)+"/",i])

allData = getData(pathsAndLabels,1)
x_train, y_train, x_test, y_test = forwardCalc(allData)
x_train, y_train = getAllData(x_train, y_train)
x_test, y_test = getAllData(x_test, y_test)


model=ResNet(img_rows,img_cols,img_channels, num_classes, x_train)
model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            shuffle=True)