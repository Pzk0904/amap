import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow.contrib.rnn import BasicLSTMCell,BasicRNNCell,static_bidirectional_rnn
import ImageProcessing
import LabelProcessing
import json

batch_size = 32
sequence_length = 5
classes = 3
save_path = './save_model/my_model'
epoches = 5
hidden_size = 128
json_path = "./data/amap_traffic_annotations_test.json"
out_path = "./data/amap_traffic_annotations_test_result.json"

class ResNetLSTM():
    def __init__(self):
        self.training = tf.placeholder(tf.bool,name='training')
        self.inputs = tf.placeholder(dtype = tf.float32,shape = [None,5,224,224,3])
        self.inputs = tf.unstack(self.inputs,axis = 1)
        self.sequence_length = tf.placeholder(dtype = tf.int32,shape = [None])

        LSTM_inputs = []
        for i in self.inputs:
            LSTM_inputs.append(self.get_features(i))
        self.LSTM_inputs = LSTM_inputs    #   seq_length*32*128
        print('Image feature extraction is successful')

        lstm_f_cell = BasicLSTMCell(num_units = hidden_size)
        lstm_b_cell = BasicLSTMCell(num_units = hidden_size)
        init_fw = lstm_f_cell.zero_state(batch_size, dtype=tf.float32)
        init_bw = lstm_b_cell.zero_state(batch_size, dtype=tf.float32)
        outputs,output_state_fw,output_state_bw = static_bidirectional_rnn(lstm_f_cell, lstm_b_cell, self.LSTM_inputs,initial_state_fw=init_fw,initial_state_bw=init_bw,sequence_length = self.sequence_length)
        self.predict = tf.layers.dense(outputs[-1],classes)
        self.finally_pre = tf.nn.softmax(self.predict)
        self.finally_pre = tf.argmax(self.predict)
        self.targets = tf.placeholder(dtype = tf.int32,shape = [None])

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.targets,logits = self.predict))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


    def get_features(self, x):
        with tf.variable_scope('get_features',reuse = tf.AUTO_REUSE):
            x = tf.layers.conv2d(x,64,kernel_size=7,strides=2,padding='same',name='conv1') # 112 112 64
            x = tf.layers.batch_normalization(x,training=self.training)
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling2d(x,3,strides=2,padding='same')

            filters = 64
            n = 1
            for repeats in (3,4,6,3):
                for _ in range(repeats):
                    x = self.resnet_block(x,filters,name='resnet_%d' % n)
                    n += 1
                filters *= 2
            #  7*7*2048
            x = tf.layers.average_pooling2d(x,7,1)  # 1*1*2048
            x = tf.reshape(x,[-1,2048])
            x = tf.layers.dense(x,hidden_size,name = 'fc')
            return x

    def resnet_block(self,x,filters,name):
        with tf.variable_scope(name):
            with tf.variable_scope('branch1'):
                x1 = tf.layers.conv2d(x,filters,1,padding='same',name='conv1')
                x1 = tf.layers.batch_normalization(x1,training=self.training)
                x1 = tf.nn.relu(x1)
                x1 = tf.layers.conv2d(x1,filters,3,padding='same',name='conv2')
                x1 = tf.layers.batch_normalization(x1,training=self.training)
                x1 = tf.nn.relu(x1)
                x1 = tf.layers.conv2d(x1,4 * filters,1,padding='same',name='conv3')

            with tf.variable_scope('shortcut1'):
                x2 = tf.layers.conv2d(x,4 * filters,1,padding='same',name='conv1')

            x = x1 + x2
            x = tf.layers.batch_normalization(x,training=self.training)
            x = tf.nn.relu(x)
        return x

class Operations():
    def __init__(self):
        self.graph1 = tf.Graph()
        with self.graph1.as_default():
            self.tensor1 = ResNetLSTM()
            self.saver = tf.train.Saver()
            with tf.Session(config = tf.ConfigProto(allow_soft_placement=True)) as self.sess:
                try:
                    self.saver.restore(self.sess,save_path)
                except:
                    self.sess.run(tf.global_variables_initializer())

    def train(self):
        samples = ImageProcessing.dataprocessing('train')
        targets = LabelProcessing.get_labels()
        steps_per_epoch = samples.samples_num // batch_size
        for i in range(epoches):
            for j in range(steps_per_epoch):
                feed_dict = {
                    self.tensor1.training: True,
                    self.tensor1.inputs:samples.next_batch_samples(batch_size),
                    self.tensor1.sequence_length:samples.next_batch_sequence_length(batch_size),
                    self.tensor1.targets:targets.next_batch_labels(batch_size)
                }
                self.sess.run(self.tensor1.train_op,feed_dict)
                if j // 50 == 0:
                    print('已训练%d次' % (i * 1500 + j + 1),flush=True)
            self.saver.save(self.sess,save_path)

    def predict(self):
        samples = ImageProcessing.dataprocessing('test')
        test_num = samples.samples_num
        results = []
        for i in range(test_num):
            feed_dict = {
                self.tensor1.training:False,
                self.tensor1.inputs:samples.next_batch_samples(1),
                self.tensor1.sequence_length: samples.next_batch_sequence_length(batch_size)
            }
            predict = self.sess.run(self.tensor1.finally_pre,feed_dict)
            results.append(predict)

        # result 是你的结果, key是id, value是status
        with open(json_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
            json_dict = json.load(f)
            data_arr = json_dict["annotations"]
            new_data_arr = []
            for i,data in enumerate(data_arr):
                id_ = data["id"]
                data["status"] = int(results[i])
                new_data_arr.append(data)
            json_dict["annotations"] = new_data_arr
            json.dump(json_dict, w)



if __name__ == '__main__':
    a = Operations()

