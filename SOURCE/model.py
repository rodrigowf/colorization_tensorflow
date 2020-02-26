# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import os
import tensorflow as tf
import config
import neural_network
import numpy as np
import cv2


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):
    for i in range(config.BATCH_SIZE):
        result = np.concatenate((batchX[i], predictedY[i]), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR, filelist[i] + "_reconstructed.jpg")
        cv2.imwrite(save_path, result)


class MODEL():

    def __init__(self):
        self.inputs = tf.compat.v1.placeholder(shape=[config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 1], dtype=tf.float32)
        self.labels = tf.compat.v1.placeholder(shape=[config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 2], dtype=tf.float32)
        self.loss = None
        self.output = None

    def build(self):
        input_data = self.inputs
        low_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 1, 64], stddev=0.1, value=0.1)
        h = low_level_conv1.feed_forward(input_data=input_data, stride=[1, 2, 2, 1])

        low_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 64, 128], stddev=0.1, value=0.1)
        h = low_level_conv2.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        low_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 128, 128], stddev=0.1, value=0.1)
        h = low_level_conv3.feed_forward(input_data=h, stride=[1, 2, 2, 1])

        low_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 128, 256], stddev=0.1, value=0.1)
        h = low_level_conv4.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        low_level_conv5 = neural_network.Convolution_Layer(shape=[3, 3, 256, 256], stddev=0.1, value=0.1)
        h = low_level_conv5.feed_forward(input_data=h, stride=[1, 2, 2, 1])

        low_level_conv6 = neural_network.Convolution_Layer(shape=[3, 3, 256, 512], stddev=0.1, value=0.1)
        h = low_level_conv6.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        mid_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h1 = mid_level_conv1.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        mid_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 512, 256], stddev=0.1, value=0.1)
        h1 = mid_level_conv2.feed_forward(input_data=h1, stride=[1, 1, 1, 1])

        global_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h2 = global_level_conv1.feed_forward(input_data=h, stride=[1, 2, 2, 1])

        global_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h2 = global_level_conv2.feed_forward(input_data=h2, stride=[1, 1, 1, 1])

        global_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h2 = global_level_conv3.feed_forward(input_data=h2, stride=[1, 2, 2, 1])

        global_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h2 = global_level_conv4.feed_forward(input_data=h2, stride=[1, 1, 1, 1])

        h2_flat = tf.reshape(h2, [config.BATCH_SIZE, -1])
        dim = h2_flat.get_shape()[1].value
        global_level_FC1 = neural_network.FullyConnected_Layer(shape=[dim, 1024], stddev=0.04, value=0.1)
        h2 = global_level_FC1.feed_forward(input_data=h2_flat)

        global_level_FC2 = neural_network.FullyConnected_Layer(shape=[1024, 512], stddev=0.04, value=0.1)
        h2 = global_level_FC2.feed_forward(input_data=h2)

        global_level_FC3 = neural_network.FullyConnected_Layer(shape=[512, 256], stddev=0.04, value=0.1)
        h2 = global_level_FC3.feed_forward(input_data=h2)

        fusion_layer = neural_network.Fusion_Layer(shape=[1, 1, 512, 256], stddev=0.1, value=0.1)
        h = fusion_layer.feed_forward(h1, h2, stride=[1, 1, 1, 1])

        colorization_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 256, 128], stddev=0.1, value=0.1)
        h = colorization_level_conv1.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        h = tf.image.resize(h, [56, 56], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        colorization_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 128, 64], stddev=0.1, value=0.1)
        h = colorization_level_conv2.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        colorization_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 64, 64], stddev=0.1, value=0.1)
        h = colorization_level_conv3.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        h = tf.image.resize(h, [112, 112], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        colorization_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 64, 32], stddev=0.1, value=0.1)
        h = colorization_level_conv4.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        output_layer = neural_network.Output_Layer(shape=[3, 3, 32, 2], stddev=0.1, value=0.1)
        logits = output_layer.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        self.output = tf.image.resize(logits, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.loss = tf.reduce_mean(tf.math.squared_difference(self.labels, self.output))

    def train(self, data, log):
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(self.loss)
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')
            if config.USE_PRETRAINED:
                saver.restore(session, os.path.join(config.MODEL_DIR, config.PRETRAINED))
                print('Pretrained weights loaded')
            for epoch in range(config.NUM_EPOCHS):
                avg_cost = 0
                for batch in range(int(data.size/config.BATCH_SIZE)):
                    batchX, batchY, _ = data.generate_batch()
                    feed_dict = {self.inputs: batchX, self.labels: batchY}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                    print("batch:", batch, " loss: ", loss_val)
                    avg_cost += loss_val / int(data.size/config.BATCH_SIZE)
                print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))
                log.write("Epoch: " + str(epoch + 1) + " Average Cost: " + str(avg_cost) + "\n")

            save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            print("Model saved in path: %s" % save_path)
            log.write("Model saved in path: " + save_path + "\n")

    def test(self, data, log):
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            avg_cost = 0
            total_batch = int(data.size/config.BATCH_SIZE)
            for _ in range(total_batch):
                batchX, batchY, filelist = data.generate_batch()
                feed_dict = {self.inputs: batchX, self.labels: batchY}
                predY, loss = session.run([self.output, self.loss], feed_dict=feed_dict)
                reconstruct(deprocess(batchX), deprocess(predY), filelist)
                avg_cost += loss/total_batch
            print("cost =", "{:.3f}".format(avg_cost))
            log.write("Average Cost: " + str(avg_cost) + "\n")
