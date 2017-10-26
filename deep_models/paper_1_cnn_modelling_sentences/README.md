# [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882), Yoon Kim

Yoon et al (2014) proposed CNN models on top of pre-trained word vectors which achieved excellent results on multiple benchmarks. The model architecture as shown in the figure maintains multiple channels of input such as different types of pre-trained vectors or vectors that are kept static during training. Then they are convolved with different kernels/filters to create sets of features which are then max pooled. These features form penultimate layer and are passed to fully connected softmax layer whose output is the probability distribution over labels.

The paper presents several variants of the model such as

1. CNN-rand (a baseline model with randomly initialized word vectors)
2. CNN-static (model with pre-trained word vectors)
3. CNN-non-static (same as above but pre-trained fine tuned)
4. CNN-multichannel (model with 2 sets of word vectors)


