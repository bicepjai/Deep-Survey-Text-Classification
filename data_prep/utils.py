# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import re

import numpy as np
import pandas as pd

#===========================================================================
# Custom Tokenizer
# donot forget to update the notebook if changes are made here
#===========================================================================

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import STOPWORDS

nltk_stopwords = set(stopwords.words('english'))
gensim_stopwords = STOPWORDS
stop_words = nltk_stopwords | set(gensim_stopwords)


custom_regular_expressions = [
    r"[\(]\s*fig[s]*\s*[.]*\s*\w*\s*\w*\s*[\)]", # (figure xxx)
    r"[\(]\s*fig[ure]*[s]*[.]*\s*\w*\s*and\s*\w*\s*[\)]", # (figure xxx and xxx)
    r"[\(]\s*fig[ure]*[s]*[.]*\s*\w*\s*and\s*fig[ure]*[s]*[.]*\s*\w*\s*[\)]", # (figure xxx and figure xxx)

    r"[\(]\s*supplementary\s*fig[s]*\s*[.]*\s*\w*\s*\w*\s*[\)]", # (supplementary figure xxx)
    r"[\(]\s*supplementary\s*fig[ure]*[s]*[.]*\s*\w*\s*and\s*\w*\s*[\)]",
    r"[\(]\s*supplementary\s*fig[ure]*[s]*[.]*\s*\w*\s*and\s*fig[ure]*[s]*[.]*\s*\w*\s*[\)]",

    r"[\(]\s*supplementary\s*table[.]*[s]*\s*\w*\s*\w*\s*[\)]", # (supplementary table xxx)
    r"[\(]\s*supplementary\s*table[.]*[s]*\s*\w*\s*and\s*\w*\s*[\)]",
    r"[\(]\s*supplementary\s*table[.]*[s]*\s*\w*\s*and\s*table[.]*[s]*\s*\w*\s*[\)]",

    r"[\(]\s*table[.]*[s]*\s*\w*\s*\w*\s*[\)]", # table
    r"[\(]\s*table[.]*[s]*\s*\w*\s*and\s*\w*\s*[\)]",
    r"[\(]\s*table[.]*[s]*\s*\w*\s*and\s*table[.]*[s]*\s*\w*\s*[\)]",

    r"\sfigureopen\s",
    r"\sfigure[s]*\s",
    r"\sfig[s*][.]*\s",
    r"\stable[.]*[s]*\s",

    r"\W+fig[s]*\s*[.]*\s*\w*\s*\w*\s*\W+",
    r"\W+\s*table[.]*[s]*\s*\w*\s*\w*\s*\W+",
]

def apply_custom_regx(text):
    re_text = text
    for regx in custom_regular_expressions:
        re_text = re.sub(regx, '', re_text)
    return re_text


nltk_tokenizer = RegexpTokenizer(r'\w+[-]*\w*')
def custom_word_tokenizer(text):
    if not text or pd.isnull(text): return ["null"]
    tokens = nltk_tokenizer.tokenize(text)
    return tokens


#===========================================================================
# Tensor board
#===========================================================================
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

def visualize_embeddings_in_tensorboard(final_embedding_matrix, metadata_path, dir_path):
    """
    view the tensors in tensorboard with PCA/TSNE
    final_embedding_matrix: embedding vector
    metadata_path: path to the vocabulary indexing the final_embedding_matrix
    """
    with tf.Session() as sess:
        embedding_var = tf.Variable(final_embedding_matrix, name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = metadata_path

        visual_summary_writer = tf.summary.FileWriter(dir_path)

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(visual_summary_writer, config)

        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, dir_path+'/visual_embed.ckpt', 1)

        visual_summary_writer.close()

#===========================================================================
# SIMPLE BATCH GENERATOR
#===========================================================================

def convert_words_to_index(words, dictionary):
    """
    Replace each word in the dataset with its index in the dictionary
    """
    return [dictionary[word] if word in dictionary else 0 for word in words]


def generate_sample(index_words, context_window_size):
    """
    Form training pairs according to the skip-gram model.
    """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """
    Group a numerical stream into batches and yield them as Numpy arrays.
    """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def get_batch_forever(index_words, context_window_size, batch_size):
    """
    get batched which doesnt depend on the index words size
    """
    generator = generate_sample(index_words, context_window_size)
    while True:
        try:
            center_batch = np.zeros(batch_size, dtype=np.int32)
            target_batch = np.zeros([batch_size, 1])
            for index in range(batch_size):
                center_batch[index], target_batch[index] = next(generator)
        except StopIteration:
            generator = generate_sample(index_words, context_window_size)

        yield center_batch, target_batch


def process_data_limited(words, dictionary, batch_size, skip_window):
    index_words = convert_words_to_index(words, dictionary)
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size)


def process_data_unlimited(words, dictionary, batch_size, skip_window):
    index_words = convert_words_to_index(words, dictionary)
    return get_batch_forever(index_words, skip_window, batch_size)


#===========================================================================
# TF BATCH GENERATOR
#===========================================================================


def tf_batch_gen(filenames, batch_size=2):
    """ filenames is the list of files you want to read from.
    In this case, it contains only heart.csv
    """
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=1) # skip the first line in the file
    _, value = reader.read(filename_queue)

    # the maximum number of elements in the queue
    capacity = 20 * batch_size

    # shuffle the data to generate batch_size sample pairs
    data_batch, label_batch = tf.train.batch([value, 1], batch_size=batch_size,
                                        capacity=capacity)

    return data_batch, label_batch

def tf_get_batch(data_batch, label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(10): # generate 10 batches
            features, labels = sess.run([data_batch, label_batch])
            print (features, labels)
        coord.request_stop()
        coord.join(threads)


#===========================================================================
# MAIN
#===========================================================================


# def main():
#     data_batch, label_batch = tf_batch_gen(["../input/local/sample_corpus_text.txt"])
#     tf_get_batch(data_batch, label_batch)

# if __name__ == '__main__':
#     main()
#
