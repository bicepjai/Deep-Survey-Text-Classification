# [Multichannel Variable-Size Convolution for Sentence Classification](https://arxiv.org/abs/1603.04513)

Yin et al (2016) [20] proposed MV-CNN which combines diverse versions of pre trained word embeddings and extracts features from multi-granular phrases with variable-size convolutions. Using multiple embeddings of same dimension from different sets of word vectors should contain more information that can be leveraged during training.

Paper describes maintaining 3 dimensional embedding matrix with channels are multiple embeddings. This multi channel initialization might help unknown words across different embeddings. Frequent words can have multiple representations and rare word (partially known word) can be made up by other words. The model then has 2 sets of convolution layer and dynamic k-max pooling layers followed by a fully connected layer with softmax (or logistic) as the last layer.

The paper describes 2 tricks for model enhancement. One is called mutual learning which is implemented in this project. Same vocabulary is maintained across the channels and they help each other tune parameters while training. The other trick is to enhance the embeddings with pretraining just like a skip-gram model or autoencoder using noise-contrastive estimation. This is not implemented in this project.


Different sets of embeddings used for this model comes from glove, word2vec, custom trained vectors on the train and test corpus using fast text tool

Keras model summary is presented below.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 60)            0
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 60, 200)       70444000    input_1[0][0]
____________________________________________________________________________________________________
embedding_2 (Embedding)          (None, 60, 200)       70444000    input_1[0][0]
____________________________________________________________________________________________________
zero_padding1d_1 (ZeroPadding1D) (None, 64, 200)       0           embedding_1[0][0]
____________________________________________________________________________________________________
zero_padding1d_2 (ZeroPadding1D) (None, 68, 200)       0           embedding_1[0][0]
____________________________________________________________________________________________________
zero_padding1d_3 (ZeroPadding1D) (None, 64, 200)       0           embedding_2[0][0]
____________________________________________________________________________________________________
zero_padding1d_4 (ZeroPadding1D) (None, 68, 200)       0           embedding_2[0][0]
____________________________________________________________________________________________________
conv1d_1 (Conv1D)                (None, 64, 128)       76928       zero_padding1d_1[0][0]
____________________________________________________________________________________________________
conv1d_2 (Conv1D)                (None, 68, 128)       128128      zero_padding1d_2[0][0]
____________________________________________________________________________________________________
conv1d_3 (Conv1D)                (None, 64, 128)       76928       zero_padding1d_3[0][0]
____________________________________________________________________________________________________
conv1d_4 (Conv1D)                (None, 68, 128)       128128      zero_padding1d_4[0][0]
____________________________________________________________________________________________________
k_max_pooling_1 (KMaxPooling)    (None, 30, 128)       0           conv1d_1[0][0]
____________________________________________________________________________________________________
k_max_pooling_2 (KMaxPooling)    (None, 30, 128)       0           conv1d_2[0][0]
____________________________________________________________________________________________________
k_max_pooling_3 (KMaxPooling)    (None, 30, 128)       0           conv1d_3[0][0]
____________________________________________________________________________________________________
k_max_pooling_4 (KMaxPooling)    (None, 30, 128)       0           conv1d_4[0][0]
____________________________________________________________________________________________________
zero_padding1d_5 (ZeroPadding1D) (None, 34, 128)       0           k_max_pooling_1[0][0]
____________________________________________________________________________________________________
zero_padding1d_6 (ZeroPadding1D) (None, 38, 128)       0           k_max_pooling_2[0][0]
____________________________________________________________________________________________________
zero_padding1d_7 (ZeroPadding1D) (None, 34, 128)       0           k_max_pooling_3[0][0]
____________________________________________________________________________________________________
zero_padding1d_8 (ZeroPadding1D) (None, 38, 128)       0           k_max_pooling_4[0][0]
____________________________________________________________________________________________________
conv1d_5 (Conv1D)                (None, 34, 128)       49280       zero_padding1d_5[0][0]
____________________________________________________________________________________________________
conv1d_6 (Conv1D)                (None, 38, 128)       82048       zero_padding1d_6[0][0]
____________________________________________________________________________________________________
conv1d_7 (Conv1D)                (None, 34, 128)       49280       zero_padding1d_7[0][0]
____________________________________________________________________________________________________
conv1d_8 (Conv1D)                (None, 38, 128)       82048       zero_padding1d_8[0][0]
____________________________________________________________________________________________________
k_max_pooling_5 (KMaxPooling)    (None, 4, 128)        0           conv1d_5[0][0]
____________________________________________________________________________________________________
k_max_pooling_6 (KMaxPooling)    (None, 4, 128)        0           conv1d_6[0][0]
____________________________________________________________________________________________________
k_max_pooling_7 (KMaxPooling)    (None, 4, 128)        0           conv1d_7[0][0]
____________________________________________________________________________________________________
k_max_pooling_8 (KMaxPooling)    (None, 4, 128)        0           conv1d_8[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 16, 128)       0           k_max_pooling_5[0][0]
                                                                   k_max_pooling_6[0][0]
                                                                   k_max_pooling_7[0][0]
                                                                   k_max_pooling_8[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2048)          0           concatenate_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           262272      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 9)             1161        dense_1[0][0]
====================================================================================================
Total params: 141,824,201
Trainable params: 141,824,201
Non-trainable params: 0
____________________________________________________________________________________________________
```
