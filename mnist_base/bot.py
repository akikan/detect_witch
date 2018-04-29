# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import os
import glob
from tensorflow.contrib import learn
import os
import tweepy
from urllib.request import urlopen
import time

CK = " "
CS = ""
AT = ""
AS = ""
# Twitterオブジェクトの生成
auth = tweepy.OAuthHandler(CK, CS)
auth.set_access_token(AT, AS)
api = tweepy.API(auth)




class Listener(tweepy.StreamListener):
    sess = tf.InteractiveSession()
    NUM_CLASSES = 11
    IMAGE_SIZE = 48
    CHANNEL = 3
    IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*CHANNEL
    
    def bulid_graph(is_training):
        NUM_CLASSES = 11
        IMAGE_SIZE = 48
        CHANNEL = 3
        IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*CHANNEL
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

    sess.run(tf.global_variables_initializer())
    (x,y_), accuracy, y_conv, prediction, train_step, keep_prob, saver= bulid_graph(is_training=False)

    saver.restore(sess, "C:/TensorFlow/image/devide_original_resize/model2.ckpt")

    # Haar-like特徴分類器
    cascadePath = "lbpcascade_animeface.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)


    def saveFace(self, filename):
        frame = cv2.imread(filename)
        image = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        # Haar-like特徴分類器で顔を検知
        faces = self.faceCascade.detectMultiScale(image)
        # 検出した顔画像の処理
        ret = []
        for (x, y, w, h) in faces:
            img = frame[y: y + h, x: x + w]
            ret.append(img)
        return ret




    def on_status(self, status):
        try:
            print('------------------------------')
            print(status.text)
            print(u"{name}({screen}) {created} via {src}\n".format(
            	name=status.author.name, 
            	screen=status.author.screen_name,
                created=status.created_at,
                src=status.source))
            status_id = status.id
            screen_name = status.author.screen_name
            print(status.entities['media'])
            tweet_part = status.text.split("@open_cans ")
            
            if status.entities['media']!=[]:
                for media in status.entities['media']:
                    url = media['media_url_https']
                    url_orig = '%s:orig' %  url
                    filename = url.split('/')[-1]
                    savepath = "C:/TensorFlow/image/devide_original_resize/img/" + filename
                    
                    response = urlopen(url_orig)
                    with open(savepath, "wb") as f:
                        f.write(response.read())
                    faces = self.saveFace(savepath)
                    if status.in_reply_to_screen_name == "@open_cans" or len(tweet_part) != 1:
                        text=""
                        for img in faces:
                            imageData=[]   
                            img2=cv2.resize(img,(self.IMAGE_SIZE,self.IMAGE_SIZE))
                            imageData.append(img2.flatten().astype(np.float32)/255.0)#18チャンネルimgdata
                            for x in range(len(img2)):
                                img2[x] = 1-img2[x]
                            prediction = tf.argmax(self.y_conv,1)
                            print(self.y_conv)
                            print(img2)
                            ret = prediction.eval(feed_dict={self.x: np.asarray([img2.flatten().astype(np.float32)/255.0]), self.keep_prob: 1.0}, session=self.sess)
                            
                            if ret == 0:
                                text += u",ペリーヌ"
                            elif ret == 1:
                                text += u",エイラ"
                            elif ret == 2:
                                text += u",リーネ"
                            elif ret == 3:
                                text += u",バルクホルン"
                            elif ret == 4:
                                text += u",サーニャ"
                            elif ret == 5:
                                text += u",シャーリー"
                            elif ret == 6:
                                text += u",エーリカ"
                            elif ret == 7:
                                text += u",宮藤"
                            elif ret == 8:
                                text += u",もっさん"
                            elif ret == 9:
                                text += u",ルッキーニ"
                            elif ret == 10:
                                text += u",ミーナ"
                            
                        if faces==[]:
                            print("no faces") 
                            text = u"顔が見当たりません　"+ str(int(time.time()))
                        else:
                            text = text[1:]
                            text = text + u"がいます　" + str(int(time.time()))
                            print(text)

                        text = u"@"+screen_name+u" "+text
                        api.update_status(status = text,in_reply_to_status_id=status_id)
        except Exception as e:
            print("[-] Error: ", e)
        return True
     
    def on_error(self, status_code):
        print('Got an error with status code: ' + str(status_code))
        return True
     
    def on_timeout(self):
        print('Timeout...')
        return True
 


#参考サイト
#userstreamの使いかた
# http://ha1f-blog.blogspot.jp/2015/02/tweepypythonpip-tweepymac.html

    


try:
    listener = Listener()
    stream = tweepy.Stream(auth, listener)
    stream.userstream()
except:
    pass