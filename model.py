import tensorflow as tf
import numpy as np 
import os
import argparse
import scipy.misc
import time
import keras
import json
import random
import nltk
import word2vec

class Model:
    def __init__(self, batch_size=50, lr = 0.0001, keep_prob=0.4, is_training = True):
        self.layer ={}
        self.batch_size = batch_size
        self.lr = lr 
        self.keep_prob = keep_prob
        self.is_training = is_training

        self.image_input = tf.placeholder(tf.float32, [self.batch_size, None, None, 5], name='input_image') 
        self.sentence_input = tf.placeholder(tf.float32, [self.batch_size, None, None, 1], name='input_sentence')
        self.label = tf.placeholder(tf.int64, [self.batch_size])
        
    def image_convolutional(self):
       with tf.name_scope('image_conv_layer'):
            with tf.variable_scope('image_conv1', reuse=tf.AUTO_REUSE):
                self.layer['image_conv1_layer'] = tf.nn.relu(
                    tf.nn.conv2d(
                        input = self.image_input, 
                        filter = tf.get_variable('c1w', [5,5, 5, 128], trainable=self.is_training), 
                        strides=[1,2,2,1],
                        padding='SAME'), name = 'image_conv_layer_1')

            with tf.variable_scope('image_batch_norm_1', reuse = tf.AUTO_REUSE):
                self.layer['image_bn_1'] = tf.keras.layers.BatchNormalization(trainable=self.is_training, name = 'image_bn_1')(self.layer['image_conv1_layer'])

            with tf.variable_scope('image_pool1', reuse=tf.AUTO_REUSE):
                self.layer['image_pool1_layer'] = tf.nn.pool(self.layer['image_bn_1'],[3,3], pooling_type='AVG', padding='SAME', name = 'image_pooling_layer_1')
            
            with tf.variable_scope('image_conv2', reuse=tf.AUTO_REUSE):
                self.layer['image_conv2_layer'] = tf.nn.relu(
                    tf.nn.conv2d(
                        input = self.layer['image_pool1_layer'],
                        filter = tf.get_variable('c2w', [3,3,128,256], trainable=self.is_training),
                        strides = [1,2,2,1],
                        padding='SAME'), name = 'image_conv_layer_2')
            
            with tf.variable_scope('image_batch_norm_2', reuse = tf.AUTO_REUSE):
                self.layer['image_bn_2'] = tf.keras.layers.BatchNormalization(trainable=self.is_training, name = 'image_bn_2')(self.layer['image_conv2_layer'])

            with tf.variable_scope('image_pool2', reuse=tf.AUTO_REUSE):
                self.layer['image_pool2_layer'] = tf.nn.pool(self.layer['image_bn_2'],[3,3], pooling_type='AVG', padding='SAME', name = 'image_pooling_layer_2')
            
    # def sentence_convolutional(self):
    #    with tf.name_scope('sentence_conv_layer'):
    #         with tf.variable_scope('sentence_conv1', reuse=tf.AUTO_REUSE):
    #             self.layer['sentence_conv1_layer'] = tf.nn.relu(
    #                 tf.nn.conv2d(
    #                     input = self.sentence_input, 
    #                     filter = tf.get_variable('c1w', [5,5, 1, 128], trainable=self.is_training), 
    #                     strides=[1,2,2,1],
    #                     padding='SAME'), name = 'sentence_conv_layer_1')

    #         with tf.variable_scope('sentence_batch_norm_1', reuse = tf.AUTO_REUSE):
    #             self.layer['sentence_bn_1'] = tf.keras.layers.BatchNormalization(axis = 0, trainable=self.is_training, name = 'sentence_bn_1')(self.layer['sentence_conv1_layer'])

    #         with tf.variable_scope('sentence_pool1', reuse=tf.AUTO_REUSE):
    #             self.layer['sentence_pool1_layer'] = tf.nn.pool(self.layer['sentence_bn_1'],[3,3], pooling_type='AVG', padding='SAME', name = 'sentence_pooling_layer_1')
            
    #         with tf.variable_scope('sentence_conv2', reuse=tf.AUTO_REUSE):
    #             self.layer['sentence_conv2_layer'] = tf.nn.relu(
    #                 tf.nn.conv2d(
    #                     input = self.layer['sentence_pool1_layer'],
    #                     filter = tf.get_variable('c2w', [3,3,128,256], trainable=self.is_training),
    #                     strides = [1,2,2,1],
    #                     padding='SAME'), name = 'sentence_conv_layer_2')
            
    #         with tf.variable_scope('sentence_batch_norm_2', reuse = tf.AUTO_REUSE):
    #             self.layer['sentence_bn_2'] = tf.keras.layers.BatchNormalization(axis = 0, trainable=self.is_training, name = 'sentence_bn_2')(self.layer['sentence_conv2_layer'])

    #         with tf.variable_scope('sentence_pool2', reuse=tf.AUTO_REUSE):
    #             self.layer['sentence_pool2_layer'] = tf.nn.pool(self.layer['sentence_bn_2'],[3,3], pooling_type='AVG', padding='SAME', name = 'sentence_pooling_layer_2')

    def fc(self):
        flatten_image_feature = tf.reshape(self.layer['image_pool2_layer'], (self.batch_size, 160000))
        # flatten_sentence_feature = tf.reshape(self.layer['sentence_pool2_layer'], (self.batch_size, 320000))
        # feature = tf.concat([flatten_image_feature, flatten_sentence_feature],1)
        with tf.variable_scope('fc1',reuse=tf.AUTO_REUSE):
            self.layer['fc1_layer'] = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)(flatten_image_feature)
        
        with tf.variable_scope('drop_out', reuse=tf.AUTO_REUSE):
            self.layer['dropout_1'] = tf.layers.dropout(inputs = self.layer['fc1_layer'], rate = self.keep_prob)
        
        with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
            self.layer['fc2_layer'] = tf.layers.dense(self.layer['dropout_1'], units = 64, activation = tf.nn.relu, reuse=tf.AUTO_REUSE)
        
        with tf.variable_scope('fc_output', reuse=tf.AUTO_REUSE):
            self.layer['logits'] = tf.layers.dense(self.layer['fc2_layer'], units = 10)
        
        with tf.variable_scope('softmax'):
            self.layer['softmax'] = tf.nn.softmax(self.layer['logits'])

    def loss(self):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label,
                logits=self.layer['logits'],
                name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy)
        # tf.summary.scalar("train_loss", loss)
        return loss

    def accuracy(self):
        with tf.name_scope('accuracy'):
            prediction = tf.argmax(self.layer['logits'], axis=1)
            correct_prediction = tf.equal(prediction, self.label)
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), 
            name = 'result')
        # tf.summary.scalar("train_acc", accuracy)
        return accuracy
    
    def optimizer(self):
        return tf.train.AdamOptimizer(self.lr)
    
    # def cal_gradient(self):
    #     return tf.train.AdamOptimizer(self.lr).compute_gradients(self.layer['softmax'], tf.trainable_variables())



class Trainer:
    def __init__(self, input_dir, model, epoch = 10, class_num = 3):
        self.input_dir = input_dir 
        self.model = model 
        self.epoch = epoch
        self.class_num = class_num

        self.lr = self.model.lr
        self.batch_size = self.model.batch_size
        
        self.model.image_convolutional()
        # self.model.sentence_convolutional()
        self.model.fc()
        self.loss = self.model.loss()
        self.accuracy = self.model.accuracy()
        self.optimizer = self.model.optimizer()

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss)

        self.positive_train = json.load(open(self.input_dir+"train_data_positive.json", 'r'))[:-300]
        self.positive_val = json.load(open(self.input_dir+"train_data_negative.json", 'r'))[-300:]
        self.negative_train = json.load(open(self.input_dir+"train_data_negative.json", 'r'))[:-300]
        self.negative_val = json.load(open(self.input_dir+"train_data_negative.json", 'r'))[-300:]
        

        self.word2vec_model = word2vec.load('./word2vec.bin')
        

    def train(self, sess, writer):

        min_len_train = np.min(np.array([len(self.positive_train), len(self.negative_train)]))
        # print(min_len_train, min_len_val)

        category_batch_num = int(self.batch_size/self.class_num)
        idx = int(min_len_train // category_batch_num)

        global_step = 0
        for ep in range(self.epoch):
            if ep == int(self.epoch //3):
                self.lr = self.lr / 10
            if ep == int(self.epoch*2//3):
                self.lr = self.lr / 10

            for idi in range(idx):
                
                batch_positive_list = random.sample(self.positive_train, category_batch_num)
                batch_positive = data_processing(batch_positive_list, self.word2vec_model,1)

                batch_negative_list = random.sample(self.negative_train, category_batch_num)
                batch_negative = data_processing(batch_negative_list, self.word2vec_model,0)

                batch_fn = batch_positive + batch_negative
                random.shuffle(batch_fn)

                batch_image_data = []
                # batch_sentence_data = []
                batch_label = []
                for i in range(len(batch_fn)):
                    data_d = np.concatenate((np.array(batch_fn[i][0][0]), np.array(batch_fn[i][0][1])), axis = -1)
                    batch_image_data.append(data_d)
                    batch_label.append(batch_fn[i][0][2])

                # print(np.array(batch_image_data).shape, np.array(batch_sentence_data).shape,np.array(batch_label).shape)
                _, loss_val, acc_val= sess.run(
                    (self.train_op, self.loss, self.accuracy),
                    feed_dict={self.model.image_input: batch_image_data, self.model.label: batch_label}
                )

                print("Epoch:[{}]===Step:[{}/{}]===Learning Rate:{}\nTrain_loss:[{:.4f}], Train_acc[{:.4f}]".format(ep, idi, idx, self.lr, loss_val, acc_val))
                
                manual_summary = tf.Summary(
                    value = [
                        tf.Summary.Value(tag='train_acc', simple_value = acc_val * 1.), 
                        tf.Summary.Value(tag='train_loss', simple_value = loss_val * 1.)
                        ]
                )
                writer.add_summary(manual_summary, global_step)


                global_step += 1

            self.validation(sess, writer, global_step)

    def validation(self, sess, writer, global_step):
        min_len_val = np.min(np.array([len(self.positive_val), len(self.negative_val)]))
        val_category_batch = int(self.batch_size/self.class_num)
        val_idx = int(min_len_val // val_category_batch)

        val_loss_sum = 0
        val_acc_sum = 0

        for val_idi in range(val_idx):
            val_batch_positive_list = random.sample(self.positive_val, val_category_batch)
            val_batch_positive = data_processing(val_batch_positive_list, self.word2vec_model,1)

            val_batch_negative_list = random.sample(self.negative_val, val_category_batch)
            val_batch_negative = data_processing(val_batch_negative_list, self.word2vec_model,0)

            val_batch_fn = val_batch_positive + val_batch_negative
            random.shuffle(val_batch_fn)

            val_batch_image_data = []
            # val_batch_sentence_data = []
            val_batch_label = []
            for i in range(len(val_batch_fn)):
                val_data_d = np.concatenate((np.array(val_batch_fn[i][0][0]), np.array(val_batch_fn[i][0][1])), axis = -1)
                val_batch_image_data.append(val_data_d)
                val_batch_label.append(val_batch_fn[i][0][2])

            val_loss_val, val_acc_val= sess.run(
                    (self.loss, self.accuracy),
                    feed_dict={self.model.image_input: val_batch_image_data, self.model.label: val_batch_label}
                )
            val_loss_sum += val_loss_val
            val_acc_sum += val_acc_val

        print("\n===\nValidation Loss: {:.4f}, Validation Acc: {:.4f}\n===\n".format(val_loss_sum/val_idx, val_acc_sum/val_idx))

        val_manual_summary = tf.Summary(
                value = [
                    tf.Summary.Value(tag='val_acc', simple_value = val_acc_sum * 1. /val_idx), 
                    tf.Summary.Value(tag='val_loss', simple_value = val_loss_sum * 1. /val_idx)
                    ]
            )
        writer.add_summary(val_manual_summary, global_step)
            

def data_processing(dic_list, word2vec_model, category):
    batch_list = []
    for dic in dic_list:
        token_list = nltk.word_tokenize(dic["Caption"])
        img_path = dic["Figure_path"]

        sample = []

        sent_matrix = []

        for token in token_list:
            word_vec = word2vec_model[token].tolist()
            
            sent_matrix.append(word_vec)
        
        sent_matrix += [[0]*100]*(200-len(sent_matrix))
        sent_matrix = np.reshape(np.expand_dims(np.array(sent_matrix), axis = -1),(100,100,-1)).tolist()
        
        img = scipy.misc.imresize(scipy.misc.imread(img_path),(100,100)).tolist()
        sample.append([img, sent_matrix, category])
        batch_list.append(sample)
    
    return batch_list





        # self.category = self.preprocess.to_category()

        # lens = []
        # num_category_batch = int(self.batch_size / self.class_num)
        # for category in self.category:
        #     lens.append(len(os.listdir(self.input_dir+'/'+category)))

        # idx = min(lens) // num_category_batch

        # # merged = tf.summary.merge_all()

        # global_step = 0





        # for ep in range(self.epoch):
        #     if ep == int(self.epoch //3):
        #         self.lr = self.lr / 10
        #     if ep == int(self.epoch*2//3):
        #         self.lr = self.lr / 10
        #     for idi in range(idx):
        #         batch_img = []
        #         img_list = []
        #         label = []
        #         i = 0

        #         for category in self.category:
        #             all_image = os.listdir(self.input_dir+'/'+category)
        #             _label=[int(i)]* num_category_batch
        #             label = label + _label
        #             i += 1
        #             img_list = all_image[idi*num_category_batch:(idi+1)*num_category_batch]
        #             _batch_img = [
        #                 scipy.misc.imresize(scipy.misc.imread(self.input_dir+'/'+category+'/'+img).astype(np.float), (64,64)) for img in img_list
        #                 ]
        #             batch_img = batch_img+_batch_img
                    
        #             # print(np.array([scipy.misc.imresize(scipy.misc.imread(self.input_dir+'/'+category+'/'+img).astype(np.float), (64,64)) for img in img_list]).shape)
                
        #         # print(np.array(batch_img).shape)
        #         # print(np.array(label).shape)

                
        #         # with tf.Session() as sess:
        #         _, loss_val, acc_val= sess.run(
        #             (self.train_op, self.loss, self.accuracy),
        #             feed_dict={self.model.input:batch_img, self.model.label:label}
        #         )

        #         # s = tf.Summary().value.add(acc_val)
        #         # writer.add_summary(summary, global_step)
        #         manual_summary = tf.Summary(
        #             value = [
        #                 tf.Summary.Value(tag='train_acc', simple_value = acc_val), 
        #                 tf.Summary.Value(tag='train_loss', simple_value = loss_val)
        #                 ]
        #         )
        #         writer.add_summary(manual_summary, global_step)

        #         global_step += 1

        #         # print("Epoch:[{}]===Step:[{}/{}]".format(ep, idi, idx))
        #         print("Epoch:[{}]===Step:[{}/{}]===Learning Rate:{}\nTrain_loss:[{:.4f}], Train_acc[{:.4f}]".format(ep, idi, idx, self.lr, loss_val, acc_val))






# if __name__ == "__main__":
#     # a= Train("./USPSdata/Train/", batch_size=50, lr=0.001)
#     train_model = Model(batch_size=100)
#     # train_model_conv = train_model.convolutional()
#     # train_model_fc = train_model.fc()
#     # loss = train_model.loss()
#     # accuracy = train_model.accuracy()
#     # optimizer = train_model.optimizer()
#     with tf.Session() as sess:
#         train = Trainer("./USPSdata/Train/", model = train_model, epoch=1)
#         writer = tf.summary.FileWriter("./log/")
#         writer.add_graph(sess.graph)
#         sess.run(tf.global_variables_initializer())
#         train.train(sess, writer)
#         writer.close()

