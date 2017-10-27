# [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

Yoon et al (2014) proposed CNN models on top of pre-trained word vectors which achieved excellent results on multiple benchmarks. The model architecture as shown in the figure maintains multiple channels of input such as different types of pre-trained vectors or vectors that are kept static during training. Then they are convolved with different kernels/filters to create sets of features which are then max pooled. These features form penultimate layer and are passed to fully connected softmax layer whose output is the probability distribution over labels.

The paper presents several variants of the model such as

1. CNN-rand (a baseline model with randomly initialized word vectors)
2. CNN-static (model with pre-trained word vectors)
3. CNN-non-static (same as above but pre-trained fine tuned)
4. CNN-multichannel (model with 2 sets of word vectors)

Keras model summary is also presented below for reference.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 5000)          0
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 5000, 200)     70444000    input_1[0][0]
____________________________________________________________________________________________________
conv1d_1 (Conv1D)                (None, 5000, 128)     76928       embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_2 (Conv1D)                (None, 5000, 128)     102528      embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_3 (Conv1D)                (None, 5000, 128)     128128      embedding_1[0][0]
____________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)   (None, 1666, 128)     0           conv1d_1[0][0]
____________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)   (None, 1250, 128)     0           conv1d_2[0][0]
____________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)   (None, 1000, 128)     0           conv1d_3[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 3916, 128)     0           max_pooling1d_1[0][0]
                                                                   max_pooling1d_2[0][0]
                                                                   max_pooling1d_3[0][0]
____________________________________________________________________________________________________
conv1d_4 (Conv1D)                (None, 3912, 128)     82048       concatenate_1[0][0]
____________________________________________________________________________________________________
max_pooling1d_4 (MaxPooling1D)   (None, 39, 128)       0           conv1d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4992)          0           max_pooling1d_4[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           639104      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 9)             1161        dense_1[0][0]
====================================================================================================
Total params: 71,473,897
Trainable params: 71,473,897
Non-trainable params: 0
____________________________________________________________________________________________________
```
