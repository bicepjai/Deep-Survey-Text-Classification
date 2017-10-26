
import sys
import os


import re
import collections
import itertools
import bcolz
import pickle
sys.path.append('../../../lib')

import numpy as np
import pandas as pd
import gc
import random
import smart_open
import h5py
import csv
import json
import functools
import time

import datetime as dt
from tqdm import tqdm_notebook as tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

random_state_number = 967898

import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

# GENERATORS ===============================================================================

def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        # print("epoch >> " + str(epoch + 1), "num_batches_per_epoch: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # x_batch, y_batch = get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index)
            # batch = list(zip(x_batch, y_batch))
            yield [x_shuffled[start_index:end_index], y_shuffled[start_index:end_index]]


# DECORATORS ===============================================================================

# http://danijar.com/structuring-your-tensorflow-models/
def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    # name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


# MODEL ===============================================================================

class MedCNN():
    """
    architecture proposed in the paper
    """

    def __init__(self, n_cnn2_pool_pair_layers=2, fc_layer_len=128, n_filters=128, kernel_size=3,
                     dropout_porb=0.5, input_sentence_len=None, output_label_size=None, word_vectors=None):

        # model settings
        self.n_cnn2_pool_pair_layers = n_cnn2_pool_pair_layers
        self.fc_layer_len = fc_layer_len
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dropout_porb = dropout_porb

        # input data
        self.input_sentence_len = input_sentence_len
        self.output_label_size = output_label_size
        self.word_vectors = word_vectors

        assert input_sentence_len != None, "set proper length for each sentence considered"
        assert output_label_size != None, "set proper length for each label considered"

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        self.logs_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

        # tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.Placeholders
            self.Embeddings
            self.MainGraph
            self.FinalLayers
            self.Evaluation
            self.LossAndOptimizer
            self.Summaries

    @define_scope
    def Placeholders(self):
        with tf.variable_scope("Placeholders"):
            self.batch_inputs = tf.placeholder(tf.int32, [None, self.input_sentence_len], name="batch_inputs")
            self.batch_labels = tf.placeholder(tf.int32, [None, self.output_label_size], name="batch_labels")
            self.is_training = tf.placeholder(tf.bool, name="is_training")

    @define_scope
    def Embeddings(self):
        """
        refer https://goo.gl/6SgW7P
        """
        with tf.variable_scope("Embeddings"):
            embedding_matrix = tf.Variable(tf.zeros(list(self.word_vectors.shape), tf.float32),name="embedding_matrix")
            self.embedding_placeholder = tf.placeholder(tf.float32, shape=list(self.word_vectors.shape))
            self.embedding_init = embedding_matrix.assign(self.embedding_placeholder)
            self.batch_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.batch_inputs)

    @define_scope
    def MainGraph(self):
        with tf.variable_scope("MainGraph"):
            input_layer = self.batch_embeddings
            for i in range(self.n_cnn2_pool_pair_layers):
                input_layer = self.TwoConvOnePool(input_layer, i)
            self.input2fc = input_layer

    @define_scope
    def FinalLayers(self):
        with tf.variable_scope("FinalLayers"):
            flattened = tf.contrib.layers.flatten(self.input2fc)
            dropout_layer = tf.nn.dropout(flattened, self.dropout_porb)
            fc_layer = tf.contrib.layers.fully_connected(dropout_layer, 128)
            dropout_layer = tf.nn.dropout(fc_layer, self.dropout_porb)
            self.final_scores = tf.contrib.layers.fully_connected(fc_layer, self.output_label_size)

    @define_scope
    def Evaluation(self):
        with tf.variable_scope("Evaluation"):
            predicted_classes = tf.argmax(self.final_scores, axis=1, name="predictions")
            label_classes = tf.argmax(self.batch_labels, axis=1, name="y_labels")
            correct_predictions = tf.equal(predicted_classes, label_classes)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    @define_scope
    def LossAndOptimizer(self):
        with tf.variable_scope("LossAndOptimizer"):
            # To update the computation of moving_mean & moving_var,
            # we must put it on the parent graph of minimizing loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.final_scores, labels=self.batch_labels)
                self.loss = tf.reduce_mean(losses)
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_step")
                self.learning_rate = tf.Variable(0.01, trainable=False, dtype=tf.float32, name="learning_rate")
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

    @define_scope
    def Summaries(self):
        with tf.variable_scope("Summaries"):
            self.accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
            self.loss_summary     = tf.summary.scalar("loss",     self.loss)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in self.grads_and_vars:

               if g is not None:
                  v_name = v.name.replace(":","_")
                  grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v_name), g)
                  sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v_name), tf.nn.zero_fraction(g))
                  grad_summaries.append(grad_hist_summary)
                  grad_summaries.append(sparsity_summary)
            self.grad_summaries_merged = tf.summary.merge(grad_summaries, name="gradient")


    def TwoConvOnePool(self, input_layer, local_id=None):
        """
        creating one set of 2 conv followed by maxpool layer
        """
        with tf.variable_scope("TwoConvOnePool_"+str(local_id)):

            conv1 = tf.layers.conv1d(input_layer, filters=self.n_filters, kernel_size=self.kernel_size, padding="valid", name="conv_1")

            relu1 = tf.nn.relu(conv1, name="relu_1")

            conv2 =  tf.layers.conv1d(relu1, filters=self.n_filters, kernel_size=self.kernel_size,
                                      padding="valid", name="conv_2")

            relu2 = tf.nn.relu(conv2, name="relu_2")

            maxpool =  tf.layers.max_pooling1d(relu2, pool_size=self.kernel_size, strides=1,
                                      padding="valid", name="maxpool")

            return maxpool
        return None


    def train(self,
              x_train_y_train,
              x_test_y_test,
              num_epochs=1,
              checkpoint_every_n_epoch=1,
              evaluate_every_n_epoch=1,
              learning_rate=0.01, batch_size=32):
        """
        this can be called multiple times
        """

        (x_train, y_train) = x_train_y_train
        (x_test, y_test)   = x_test_y_test
        print("Writing logs to {}\n".format(self.logs_dir))

        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            session_conf = tf.ConfigProto( allow_soft_placement=True,
                                           log_device_placement=True)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # train summaries
                self.train_summary_op = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.grad_summaries_merged])
                train_summary_dir = os.path.join(self.logs_dir, "summaries","train")
                self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # test/val summaries
                self.test_summary_op = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.grad_summaries_merged])
                test_summary_dir = os.path.join(self.logs_dir, "summaries","test")
                self.test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(self.logs_dir, "checkpoints"))
                model_checkpoint_path = os.path.join(checkpoint_dir, "model")
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                # if that checkpoint exists, restore from checkpoint
                if latest_checkpoint:
                    saver.restore(sess, latest_checkpoint)
                    # print "restored checkpoint ..."
                else:
                    if not os.path.exists(model_checkpoint_path):
                        os.makedirs(model_checkpoint_path)

                # initializers
                sess.run(tf.global_variables_initializer())
                sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.word_vectors})

                # Training loop. For each batch...
                test_details = train_details = ""
                for epoch in range(num_epochs):
                    total_batches = len(x_train)/batch_size
                    train_batch_gen = batch_iter(x_train, y_train, batch_size, 1, shuffle=True)
                    train_details = self.train_step(train_batch_gen, total_batches, sess)

                    current_epoch = tf.train.global_step(sess, self.global_step)
                    if current_epoch % checkpoint_every_n_epoch == 0:
                        path = saver.save(sess, model_checkpoint_path,
                                            global_step=current_epoch,
                                            meta_graph_suffix='meta', write_meta_graph=True)
                        print("Saved model checkpoint to {}\n".format(path))

                    if current_epoch % evaluate_every_n_epoch == 0:
                        test_batch_gen = batch_iter(x_test, y_test, batch_size, 1, shuffle=True)
                        test_details = self.test_step(test_batch_gen, sess)#, writer=self.test_summary_writer)

                    print ("epoch " + str(epoch) + " " + train_details + " " + test_details)


    # defining training_step
    def train_step(self, batch_gen, tqdm_total, sess):
        """
        Runs one full epocj on the batch provided
        """

        acc, losses = [], []
        for batch in tqdm(batch_gen, total=tqdm_total):
            feed_dict = {
              self.batch_inputs: batch[0],
              self.batch_labels: batch[1],
              self.is_training: True # Update moving_mean, moving_var
            }
            _, step, summaries, loss, accuracy = sess.run(
                [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy], feed_dict)
            acc.append(accuracy)
            losses.append(loss)

            self.train_summary_writer.add_summary(summaries, step)
        return ("train_loss:{} train_accuracy:{}".format(str(sum(losses)/len(losses)),str(sum(acc)/len(acc))))


    # defining validation step
    def test_step(self, batch_gen, sess, writer=None):
        """
        Evaluates model on a test/val set, providng the batch generator
        """
        acc, losses = [], []
        for i, batch in enumerate(batch_gen):
            feed_dict = {
              self.batch_inputs: batch[0],
              self.batch_labels: batch[1],
              self.is_training: False # Use converged (fixed) updated moving_mean, moving_var
            }
            step, summaries, loss, accuracy = sess.run(
                [self.global_step, self.test_summary_op, self.loss, self.accuracy], feed_dict)
            acc.append(accuracy)
            losses.append(loss)

            # time_str = datetime.datetime.now().isoformat()
            # print("batch " + str(i + 1) + " in dev >>" +
            #      " {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
        return ("test_loss:{} test_accuracy:{}".format(str(sum(losses)/len(losses)),str(sum(acc)/len(acc))))
