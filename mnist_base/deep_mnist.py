# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import os
import glob
import time
from tensorflow.contrib import learn

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
# def getBatch(Data,Label,count):
#     retData=[]
#     retLabel=[]
#     flag=False
#     l=0
#     for label in Label:
#         length = len(label)
#         size=int(np.sqrt(length))

#         li=[]
#         for i in range(length):
#             li.append(i)

#         if (count+1)*size > length:
#             count=0
#             flag=True
#         index = size*count
#         for i in range(size):
#             retData.append(Data[l][index+i])
#             retLabel.append(Label[l][index+i])
        
        
#         l+=1
#     count+=1
#     if flag:
#         count=0
#     return retData, retLabel, count
def getBatch(Data,Label,count):
    retData=[]
    retLabel=[]
    flag=False
    l=0

    for label in Label:
        length = len(label)
        size=int(np.sqrt(length))
        li=[]
        for i in range(length):
            li.append(i)

        if (count+1)*size > length:
            count=0
            flag=True

        for i in range(size):
            index = int(np.random.rand()*(len(li)-1))
            retData.append(Data[l][index])
            retLabel.append(Label[l][index])
            li.pop(index)
        count+=1
        if flag:
    	    count=0
        l+=1
    return retData, retLabel, count

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
            tempLabel= np.zeros(NUM_CLASSES)
            filepath = pathAndLabel[0].translate(str.maketrans('\\', '/'))

            # image_r = tf.read_file(filepath)
            # images = tf.image.decode_image(image_r, channels=3)
            
            img2 = cv2.imread(filepath)
            img=cv2.resize(img2,(IMAGE_SIZE,IMAGE_SIZE))
            img=img.flatten().astype(np.float32)/255.0
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

def getAllData(Data, Label):
    retD=[]
    retL=[]
    for i in range(len(Label)):
        retD.extend(Data[i])
        retL.extend(Label[i])
    return np.asarray(retD),np.asarray(retL)

def batch_normalization(shape, input, gamma, beta):
  eps = 1e-5

  mean, variance = tf.nn.moments(input, [0])
  return gamma * (input - mean) / tf.sqrt(variance + eps) + beta

NUM_CLASSES = 11
IMAGE_SIZE = 48
CHANNEL = 3
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*CHANNEL

pathsAndLabels=[]
for i in range(0,11):
    pathsAndLabels.append(["./"+str(i+1)+"/",i])

allData = getData(pathsAndLabels,1)
trainData, trainLabel, testData, testLabel = forwardCalc(allData)
allD, allL = getAllData(trainData, trainLabel)

def bulid_graph(is_training):
    def batch_norm_wrapper(inputs, is_training, decay = 0.999):
        epsilon = 1e-5
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
        rank = len(inputs.get_shape())
        axes = []  # nn:[0], conv:[0,1,2]
        for i in range(rank - 1):
            axes.append(i)
        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,axes)
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL])
    #normal
    # W_conv1 = weight_variable([5, 5, CHANNEL, 32])
    # b_conv1 = bias_variable([32])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)

    #bacth
    W_conv1 = weight_variable([5, 5, CHANNEL, 32])
    h_conv1 = conv2d(x_image, W_conv1)
    gamma1 = weight_variable([32])
    beta1 = weight_variable([32])
    bn1 = batch_norm_wrapper(h_conv1,is_training)#batch_normalization(32, h_conv1,gamma1, beta1)
    h_pool1 = max_pool_2x2(tf.nn.relu(bn1))



    W_conv2 = weight_variable([5, 5, 32, 64])
    # b_conv2 = bias_variable([64])
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    h_conv2 = conv2d(h_pool1, W_conv2)
    gamma2 = weight_variable([64])
    beta2 = weight_variable([64])
    bn2 = batch_norm_wrapper(h_conv2,is_training) #batch_normalization(64, h_conv2,gamma2,beta2)
    h_pool2 = max_pool_2x2(tf.nn.relu(bn2))


    W_fc1 = weight_variable([12 * 12 * 64, 1024])
    # b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    bn3 = batch_norm_wrapper(tf.matmul(h_pool2_flat, W_fc1),is_training) #batch_normalization(1024, tf.matmul(h_pool2_flat, W_fc1), gamma3, beta3)
    h_fc1 = tf.nn.relu(bn3)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean( y )
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    prediction = tf.argmax(y_conv, 1)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return (x,y_), accuracy, y_conv, prediction, train_step, keep_prob, tf.train.Saver()

(x,y_), accuracy, y_conv, prediction, train_step, keep_prob, saver= bulid_graph(is_training=True)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  count = 1
  # 元データをバッチで使用できる形式に変更する
  # label, image = tf.train.slice_input_producer([trainLabel,trainData], shuffle=True, seed=1)
        
  
  for i in range(10000):
    batchX, batchY ,count = getBatch(trainData, trainLabel,count)
    # データをエンキューしてバッチ化する
    # batchY, batchX = tf.train.batch([label, image], batch_size=33)
    if i % 1001 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: allD, y_: allL, keep_prob: 1.0})
      # print(batchY)
      print('step %d, training accuracy %g' % (i, train_accuracy))
      saver.save(sess, "C:/TensorFlow/image/devide_original_resize/model2.ckpt")
    # print(batchX[0])
    train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})
  batchX, batchY ,count = getBatch(testData, testLabel,0)
  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: batchX, y_: batchY, keep_prob: 1.0}))




sess.close()
tf.reset_default_graph()
batchX, batchY ,count = getBatch(testData, testLabel,0)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  (x,y_), accuracy, y_conv, prediction, train_step, keep_prob, saver= bulid_graph(is_training=False)
  saver.restore(sess, "C:/TensorFlow/image/devide_original_resize/model2.ckpt")
  for bx in batchX:
          # print(bx)
    print("result: %g"%prediction.eval(feed_dict={x: [bx], keep_prob: 1.0}))
  print(batchY)




# batch_normalization
# http://qiita.com/sergeant-wizard/items/052c98c6e712a4a8df6a