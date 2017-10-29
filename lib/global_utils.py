import sys
import os

import re
import collections
import itertools
import bcolz
import pickle

import numpy as np
import pandas as pd
import gc
import random
import smart_open
import h5py
import csv
import tensorflow as tf
import gensim

import datetime as dt
from tqdm import tqdm_notebook as tqdm

# import multiprocessing as mp
# from itertools import repeat, product
# from functools import partial

# to be able to pickle class methods for multi processing
# https://stackoverflow.com/questions/27318290/why-can-i-pass-an-instance-method-to-multiprocessing-process-but-not-a-multipro

def _instance_method_alias(obj, arg):
    """
    Alias for instance method that allows the method to be called in a
    multiprocessing pool
    """
    return obj.convertSent2WordIds(arg)



def get_embeddings_from_ft(fasttext_vec_file, dim, vocab_words):
   """
   convert fast text .vec file to numpy array
   created embedding will be in order of words in vocab_words
   """

   # gathering words from fasttext vec file--------------------
   ft_lines = None

   with open(fasttext_vec_file, "r") as f:
      ft_lines = f.readlines()

   ft_shape = tuple([int(i.strip()) for i in ft_lines[0].split()])
   ft_vocab_size = ft_shape[0]

   ft_wvs_dict = {}

   for i, line in enumerate(ft_lines[1:]):
      str_list = line.split()
      word = str(str_list[0].strip())
      vec = np.array([np.float(f) for f in str_list[1:]])
      assert dim == len(vec), "fast text some vectors doesn't match dimensions "+str(dim)+" != "+str(len(vec))
      ft_wvs_dict[word] = vec

   assert ft_vocab_size == len(ft_wvs_dict), "fast text vectors file read issue "+str(ft_vocab_size)+" != "+str(len(ft_wvs_dict))

   # creating embedding matrix from the file --------------------
   wvs_embedding = np.random.randn(len(vocab_words), dim)
   for i,word in enumerate(vocab_words):
      if word in ft_wvs_dict:
         wvs_embedding[i] = ft_wvs_dict[word]

   return wvs_embedding


#=============================================================
#     DOCUMENT PREPROCESSING
#=============================================================

CHAR_ALPHABETS = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n "
char_start_tag_idx         = len(CHAR_ALPHABETS) + 0
char_end_tag_idx           = len(CHAR_ALPHABETS) + 1
char_unknown_tag_idx       = len(CHAR_ALPHABETS) + 2

# when sentences are converted to characters
# these are appended to signal end of sentences
char_sent_start_tag_idx    = len(CHAR_ALPHABETS) + 3
char_sent_end_tag_idx      = len(CHAR_ALPHABETS) + 4

CHAR_ALPHABETS_LEN = len(CHAR_ALPHABETS) + 4

class GenerateDataset(object):
   """
   This class takes in preprocessed data frame and
   generated datasets as necessary
   """

   def __init__(self, data_frame, vocab_idx):
      self.data_frame = data_frame
      self.vocab_idx = vocab_idx
      self.vocab_size = len(vocab_idx)

      # constants ================================================================================
      self.sentence_start_tag_idx = self.vocab_idx["<SOSent>"]
      self.sentence_end_tag_idx   = self.vocab_idx["<EOSent>"]
      self.word_unknown_tag_idx        = self.vocab_idx["<UNK>"]

      self.default_unit_dict = {
         "gene_unit"      : "words",
         "variation_unit" : "words",
         "doc_unit"       : "words",
         "doc_form"       : "text",
         "doc_cntx_dir"   : "forward",
         "divide_document": "single_unit"
      }


   def convertSent2WordIds(self, sentence, add_start_end_tag=False):
      """
      sentence is a list of word.
      It is converted to list of ids based on vocab_idx
      """

      sent2id = []
      if add_start_end_tag:
         sent2id = [self.sentence_start_tag_idx]

      try:
         sent2id = sent2id + [self.vocab_idx[word] if self.vocab_idx[word]<self.vocab_size else self.word_unknown_tag_idx for word in sentence]
      except KeyError as e:
         print(e)
         print (sentence)
         raise ValueError('Fix this issue dude')

      if add_start_end_tag:
         sent2id = sent2id + [self.sentence_end_tag_idx]

      return sent2id



   def convertDoc2Sent2WordIds(self, document, add_start_end_tag=False):
      """
      document is a list of sentence.
      sentence is a list of word.
      so given sent_list will be converted to list of list of ids based on vocab_idx
      """

      return [self.convertSent2WordIds(sentence, add_start_end_tag) for sentence in document]



   def convertWord2Char2Ids(self, word, add_start_end_tag=False):
      """
      word is a char sequence or list of characters,
      return list of ids in word or char sequence
      """
      char2id = []
      if add_start_end_tag:
         char2id = [char_start_tag_idx]

      char2id = char2id + [CHAR_ALPHABETS.find(char) for char in word]

      if add_start_end_tag:
         char2id = char2id + [char_end_tag_idx]

      return char2id



   def convertSent2Word2Char2Ids(self, sentence, add_start_end_tag=False, unit="chars"):
      """
      sentence is list of words
      word is list of characters
      returns list of list of ids
      """

      sent2words2char2id = []
      if unit == "chars":
         """
         all the words are grouped as list of chars with pre-post added tags
         """
         if add_start_end_tag:
            sent2words2char2id = [[char_sent_start_tag_idx]]

         sent2words2char2id = sent2words2char2id + [self.convertWord2Char2Ids(word, add_start_end_tag) if self.vocab_idx[word] < self.vocab_size else [char_unknown_tag_idx] for word in sentence]

         if add_start_end_tag:
            sent2words2char2id = sent2words2char2id + [[char_sent_end_tag_idx]]
      elif unit == "raw_chars":
         """
         just a stream of characters
         """
         if add_start_end_tag:
            sent2words2char2id = [char_sent_start_tag_idx]

         for word in sentence:
            if self.vocab_idx[word] < self.vocab_size:
               sent2words2char2id += [charid for charid in self.convertWord2Char2Ids(word, add_start_end_tag)]
            else:
               sent2words2char2id += [char_unknown_tag_idx]

         if add_start_end_tag:
            sent2words2char2id = sent2words2char2id + [char_sent_end_tag_idx]
      else:
         assert False, "give valid doc_unit argument"

      return sent2words2char2id



   def convertDoc2Sent2Word2Char2Ids(self, document, doc_form="sentences", add_start_end_tag=False, unit="chars"):
      """
      document is a list of sentence.
      sentence is a list of word.
      so given sent_list will be converted to list of list of ids based on vocab_idx

      returns list of list if doc_form == "text"
      returns list of list of list if doc_form == "sentences"
      """
      doc2word2char2ids = []

      if doc_form == "sentences":
         doc2word2char2ids = [self.convertSent2Word2Char2Ids(sentence, add_start_end_tag, unit) for sentence in document]

      elif doc_form == "text":
         doc2word2char2ids = [list_or_charid for list_or_charid in self.convertSent2Word2Char2Ids(sentence, add_start_end_tag, unit)]
      else:
         assert False, "give valid doc_form argument"

      return doc2word2char2ids



   def generate_data(self, unit_dict=None, has_class=False, add_start_end_tag=False):
      """
      dataframe expects to have Sentences, Variations, Genes, Class(has_class)

      Sentences Text attribute converted to list of sentences which in turn converted to list of words
      Variations just one sentence which is a list of words
      Genes just one sentence which is a list of words

      returns information based on request

      unit_dict contains these 5 keys that can have
      gene_unit      can be ["words", "chars", "raw_chars"]
      variation_unit can be ["words", "chars", "raw_chars"]
      doc_unit       can be ["words", "word_list", chars", "raw_chars"]
      doc_form       can be ["sentences", "text"]
      doc_cntx_dir   can be ["forward", "backward"]
      divide_document can be ["single_unit", "multiple_units"]

      """
      if not unit_dict:
         unit_dict = self.default_unit_dict

      try:
         unit_dict["doc_cntx_dir"]
      except KeyError as e:
         unit_dict["doc_cntx_dir"] = "forward"

      ids_document   = []
      ids_labels     = []
      ids_genes      = []
      ids_variations = []

      # since sometimes the data will be shuffled in the frame
      # during train test split
      for index in self.data_frame.index:
         document = self.data_frame.Sentences[index]
         if unit_dict["divide_document"] == "single_unit": #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

            # doc units --------------------------------------------------------------
            if unit_dict["doc_unit"] == "words" or unit_dict["doc_unit"] == "word_list":

               if unit_dict["doc_form"] == "sentences":
                  ids_document.append(self.convertDoc2Sent2WordIds(document, add_start_end_tag))

               else: # unit_dict["doc_form"] == "text"

                  # using multiprocess to process each sentence in document and concatenate them to a single sentence
                  # get_wordid_list = lambda d, setag : [wid for s in d for wid in self.convertSent2WordIds(s, setag)]
                  # text_word_list = []
                  # with mp.Pool(processes = 5) as pool:
                  #    # text_word_list = pool.starmap(get_wordid_list, product(document, [add_start_end_tag]*len(document)))
                  #    # text_word_list = pool.starmap(get_wordid_list, zip(document, repeat(add_start_end_tag)))
                  #    text_word_list = pool.map(partial(get_wordid_list, setag=add_start_end_tag), document)


                  # without multiprocessing
                  if unit_dict["doc_unit"] == "words":
                     text_word_list = [word_id for sentence in document for word_id in self.convertSent2WordIds(sentence, add_start_end_tag)]

                     if unit_dict["doc_cntx_dir"] == "backward":
                        text_word_list = text_word_list[::-1]

                  else: # unit_dict["doc_unit"] == "word_list": sentence form a list
                     text_word_list = [self.convertSent2WordIds(sentence, add_start_end_tag) for sentence in document]

                     if unit_dict["doc_cntx_dir"] == "backward":
                        text_word_list = [self.convertSent2WordIds(sentence, add_start_end_tag)[::-1] for sentence in document]

                  ids_document.append(text_word_list)

            elif unit_dict["doc_unit"] == "chars" or unit_dict["doc_unit"] == "raw_chars":

               if unit_dict["doc_form"] == "sentences":

                  for sentence in document:
                     ids_document.append(self.convertDoc2Sent2Word2Char2Ids(document,
                        doc_form=unit_dict["doc_form"], unit=unit_dict["doc_unit"], add_start_end_tag=add_start_end_tag))

               else: # unit_dict["doc_form"] == "text"
                  text_char_list = [word_as_char_list_id for sentence in document for word_as_char_list_id in self.convertSent2Word2Char2Ids(sentence, add_start_end_tag, unit=unit_dict["doc_unit"])]

                  ids_document.append(text_char_list)

            else:
               assert False, "give valid doc_unit key-value"

            # others --------------------------------------------------------------
            if has_class:
               ids_labels.append(self.data_frame.Class[index])

            if unit_dict["gene_unit"] == "words":
               ids_genes.append(self.convertSent2WordIds(self.data_frame.Gene[index], add_start_end_tag))
            else:
               ids_genes.append(self.convertSent2Word2Char2Ids(self.data_frame.Gene[index],
                                 add_start_end_tag, unit=unit_dict["doc_unit"]))

            if unit_dict["variation_unit"] == "words":
               ids_variations.append(self.convertSent2WordIds(self.data_frame.Variation[index], add_start_end_tag))
            else:
               ids_variations.append(self.convertSent2Word2Char2Ids(self.data_frame.Variation[index],
                                       add_start_end_tag, unit=unit_dict["doc_unit"]))

         else: # unit_dict["divide_document"] == "multiple_unit" #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
            for sentence in document:

               # doc units --------------------------------------------------------------
               if unit_dict["doc_unit"] == "words":

                  # doesnt matter if
                  # unit_dict["doc_form"] == "sentences"
                  # unit_dict["doc_form"] == "text"

                  try:
                     sentence_list = self.convertSent2WordIds(sentence, add_start_end_tag)
                     if unit_dict["doc_cntx_dir"] == "backward":
                        text_word_list = self.convertSent2WordIds(sentence, add_start_end_tag)[::-1]

                     ids_document.append(sentence_list)

                  except ValueError as e:
                     print(e)
                     print (index)
                     raise ValueError('Fix this issue dude !')

               elif unit_dict["doc_unit"] == "chars" or unit_dict["doc_unit"] == "raw_chars":

                  # doesnt matter if
                  # unit_dict["doc_form"] == "sentences"
                  # unit_dict["doc_form"] == "text"

                  ids_document.append(self.convertSent2Word2Char2Ids(sentence, add_start_end_tag,
                                    unit=unit_dict["doc_unit"]))


               # others --------------------------------------------------------------
               if has_class:
                  ids_labels.append(self.data_frame.Class[index])

               if unit_dict["gene_unit"] == "words":
                  ids_genes.append(self.convertSent2WordIds(self.data_frame.Gene[index], add_start_end_tag))
               else:
                  ids_genes.append(self.convertSent2Word2Char2Ids(self.data_frame.Gene[index],
                                    add_start_end_tag, unit=unit_dict["gene_unit"]))

               if unit_dict["variation_unit"] == "words":
                  ids_variations.append(self.convertSent2WordIds(self.data_frame.Variation[index], add_start_end_tag))
               else:
                  ids_variations.append(self.convertSent2Word2Char2Ids(self.data_frame.Variation[index],
                                          add_start_end_tag, unit=unit_dict["variation_unit"]))


      return ids_document, ids_genes, ids_variations, ids_labels



   def placeholder_function(self, unit_dict=None, limit_dict=None,
      has_class=False, add_start_end_tag=False):
      """
      dataframe expects to have Sentences, Variations, Genes, Class(has_class)

      Sentences Text attribute converted to list of sentences which in turn converted to list of words
      Variations just one sentence which is a list of words
      Genes just one sentence which is a list of words

      returns information based on request

      unit_dict contains these 5 keys that can have
      gene_unit      can be ["words", "chars"]
      variation_unit can be ["words", "chars"]
      doc_unit       can be ["words", "chars"]
      doc_form       can be ["sentences", "text"]
      divide_document can be ["single_unit", "multiple_units"]

      limit_dict contains max sequence len to form valid matrices
      Text attribute options
      max_text_seq_len       => maximum number of words in a text
      max_text_document_len   => maximum number of sentences in a document
      max_text_sentence_len   => maximum number of words in a sentence
      max_text_word_len       => maximum number of chars in a word

      Gene Attribute options
      max_gene_sentence_len        => maximum number of words in a sentence
      max_gene_word_len            => maximum number of chars in a word

      Variation Attribute options
      max_variation_sentence_len   => maximum number of words in a sentence
      max_variation_word_len       => maximum number of chars in a word

      """

      ids_document, ids_genes, ids_variations, ids_labels = self.generate_dataset(unit_dict, has_class, add_start_end_tag)


# testing ======================================================================================

def test_class():
   document = [
         ['beautiful', 'is', 'better', 'than', 'ugly.'],
         ['explicit', 'is', 'better', 'than', 'implicit.'],
         ['simple', 'is', 'better', 'than', 'complex.'],
         ['complex', 'is', 'better', 'than', 'complicated.'],
         ['flat', 'is', 'better', 'than', 'nested.'],
         # ['sparse', 'is', 'better', 'than', 'dense.'],
         # ['readability', 'counts.'],
         # ['special', 'cases', "aren't", 'special', 'enough', 'to', 'break', 'the', 'rules.'],
         # ['although', 'practicality', 'beats', 'purity.'],
         # ['errors', 'should', 'never', 'pass', 'silently.'],
         # ['unless', 'explicitly', 'silenced.'],
         # ['in', 'the', 'face', 'of', 'ambiguity,', 'refuse', 'the', 'temptation', 'to', 'guess.'],
         # ['there', 'should', 'be', 'one--', 'and', 'preferably', 'only', 'one', '--obvious', 'way', 'to', 'do', 'it.'],
         # ['although', 'that', 'way', 'may', 'not', 'be', 'obvious', 'at', 'first', 'unless', "you're", 'Dutch.'],
         # ['now', 'is', 'better', 'than', 'never.'], ['Although', 'never', 'is', 'often', 'better', 'than', '*right*', 'now.'],
         # ['if', 'the', 'implementation', 'is', 'hard', 'to', 'explain,', "it's", 'a', 'bad', 'idea.'],
         # ['if', 'the', 'implementation', 'is', 'easy', 'to', 'explain,', 'it', 'may', 'be', 'a', 'good', 'idea.'],
         # ['namespaces', 'are', 'one', 'honking', 'great', 'idea', '--', "let's", 'do', 'more', 'of', 'those!'],
      ]

   data_dict = {
      "ID"        : 0,
      "Gene"      : [["beautiful"]],
      "Variation" : [["complex", "simple"]],
      "Class"     : 0,
      "Sentences" : [document[:]]
   }

   custom_unit_dict = {
         "gene_unit"          : "raw_chars",
         "variation_unit"     : "raw_chars",
         # text transformed to sentences attribute
         "doc_unit"           : "raw_chars",
         "doc_form"           : "sentences",
         # "doc_cntx_dir"       : "forward",
         "divide_document"    : "single_unit"
      }

   df = pd.DataFrame(data=data_dict)
   corpus = sorted(list(set([word for sentence in document for word in sentence])))
   corpus_wordidx = {word:i for i,word in enumerate(corpus)}
   corpus_wordidx["<SOSent>"] = len(corpus)
   corpus_wordidx["<EOSent>"] = len(corpus) + 1
   corpus_wordidx["<UNK>"]    = len(corpus) + 2

   gen_data = GenerateDataset(df, corpus_wordidx)
   x_T, x_G, x_V, x_C = gen_data.generate_data(custom_unit_dict, has_class=True, add_start_end_tag=True)

   print("data", df.Sentences[0], "\n")
   print(corpus_wordidx)
   index = 0
   print("text",np.array(x_T).shape, x_T[index])
   print("gene",np.array(x_G).shape, x_G[index])
   print("variation",np.array(x_V).shape, x_V[index])
   print("classes",np.array(x_C).shape, x_C[index])


if __name__ == "__main__":
   test_class()

