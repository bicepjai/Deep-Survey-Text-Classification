# [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882), Yoon Kim

Yoon et al (2014) proposed CNN models on top of pre-trained word vectors which achieved excellent results on multiple benchmarks. The model architecture as shown in the figure maintains multiple channels of input such as different types of pre-trained vectors or vectors that are kept static during training. Then they are convolved with different kernels/filters to create sets of features which are then max pooled. These features form penultimate layer and are passed to fully connected softmax layer whose output is the probability distribution over labels.

The paper presents several variants of the model such as

1. CNN-rand (a baseline model with randomly initialized word vectors)
2. CNN-static (model with pre-trained word vectors)
3. CNN-non-static (same as above but pre-trained fine tuned)
4. CNN-multichannel (model with 2 sets of word vectors)

Keras model summary is also presented below for reference.

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 5000, 200)         70444000
_________________________________________________________________
zero_padding1d_1 (ZeroPaddin (None, 5098, 200)         0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 5098, 64)          640064
_________________________________________________________________
k_max_pooling_1 (KMaxPooling (None, 5, 64)             0
_________________________________________________________________
activation_1 (Activation)    (None, 5, 64)             0
_________________________________________________________________
zero_padding1d_2 (ZeroPaddin (None, 53, 64)            0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 53, 64)            102464
_________________________________________________________________
folding_1 (Folding)          (None, 53, 32)            0
_________________________________________________________________
k_max_pooling_2 (KMaxPooling (None, 5, 32)             0
_________________________________________________________________
activation_2 (Activation)    (None, 5, 32)             0
_________________________________________________________________
flatten_1 (Flatten)          (None, 160)               0
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 1449
=================================================================
Total params: 71,187,977
Trainable params: 71,187,977
Non-trainable params: 0
_________________________________________________________________
