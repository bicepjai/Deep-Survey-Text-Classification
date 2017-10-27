# [MGNC-CNN: A Simple Approach to Exploiting Multiple Word Embeddings for Sentence Classification](https://arxiv.org/abs/1603.00968)


Ye Zhang et al proposed MG(NC)-CNN captures multiple features from multiple sets of embeddings which are concatenated at the penultimate layer. MG(NC)-CNN is very similar to MV-CNN but address some drawbacks such as model complexity and requirement for the dimension of embeddings to be the same.

MG-CNN uses off the shelf embeddings and treats them as distinct groups for performing convolutions following up with max-pooling layer. Because of its simplicity, the model requires training time in the order of hours. MGNC-CNN differs just in the regularization strategy. It imposes grouped regularization constraints independently regularizing the sub components from each separate groups (embeddings). Intuitively this captures discriminative properties of the text by penalizing weight estimates for features derived from less discriminative embeddings. Different sets of trained embeddings will be used.

Keras model summary is presented below.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 60)            0
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 60, 300)       105666000   input_1[0][0]
____________________________________________________________________________________________________
embedding_2 (Embedding)          (None, 60, 200)       70444000    input_1[0][0]
____________________________________________________________________________________________________
embedding_3 (Embedding)          (None, 60, 100)       35222000    input_1[0][0]
____________________________________________________________________________________________________
zero_padding1d_1 (ZeroPadding1D) (None, 64, 300)       0           embedding_1[0][0]
____________________________________________________________________________________________________
zero_padding1d_2 (ZeroPadding1D) (None, 68, 300)       0           embedding_1[0][0]
____________________________________________________________________________________________________
zero_padding1d_3 (ZeroPadding1D) (None, 64, 200)       0           embedding_2[0][0]
____________________________________________________________________________________________________
zero_padding1d_4 (ZeroPadding1D) (None, 68, 200)       0           embedding_2[0][0]
____________________________________________________________________________________________________
zero_padding1d_5 (ZeroPadding1D) (None, 64, 100)       0           embedding_3[0][0]
____________________________________________________________________________________________________
zero_padding1d_6 (ZeroPadding1D) (None, 68, 100)       0           embedding_3[0][0]
____________________________________________________________________________________________________
conv1d_1 (Conv1D)                (None, 64, 16)        14416       zero_padding1d_1[0][0]
____________________________________________________________________________________________________
conv1d_2 (Conv1D)                (None, 68, 16)        24016       zero_padding1d_2[0][0]
____________________________________________________________________________________________________
conv1d_3 (Conv1D)                (None, 64, 16)        9616        zero_padding1d_3[0][0]
____________________________________________________________________________________________________
conv1d_4 (Conv1D)                (None, 68, 16)        16016       zero_padding1d_4[0][0]
____________________________________________________________________________________________________
conv1d_5 (Conv1D)                (None, 64, 16)        4816        zero_padding1d_5[0][0]
____________________________________________________________________________________________________
conv1d_6 (Conv1D)                (None, 68, 16)        8016        zero_padding1d_6[0][0]
____________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalMa (None, 16)            0           conv1d_1[0][0]
____________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalMa (None, 16)            0           conv1d_2[0][0]
____________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalMa (None, 16)            0           conv1d_3[0][0]
____________________________________________________________________________________________________
global_max_pooling1d_4 (GlobalMa (None, 16)            0           conv1d_4[0][0]
____________________________________________________________________________________________________
global_max_pooling1d_5 (GlobalMa (None, 16)            0           conv1d_5[0][0]
____________________________________________________________________________________________________
global_max_pooling1d_6 (GlobalMa (None, 16)            0           conv1d_6[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 96)            0           global_max_pooling1d_1[0][0]
                                                                   global_max_pooling1d_2[0][0]
                                                                   global_max_pooling1d_3[0][0]
                                                                   global_max_pooling1d_4[0][0]
                                                                   global_max_pooling1d_5[0][0]
                                                                   global_max_pooling1d_6[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           12416       concatenate_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 9)             1161        dense_1[0][0]
====================================================================================================
Total params: 211,422,473
Trainable params: 211,422,473
Non-trainable params: 0
____________________________________________________________________________________________________
```
