# [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188)

Kalchbrenner et al (2014) presented Dynamic Convolutional Neural Network for semantic modelling of sentences. This model handles sentences of varying length and uses dynamic k-max pooling over linear sequences. This helps the model induce a feature graph that is capable of capturing short and long range relations. K-max pooling is difference from local max pooling and outputs k max values from the necessary dimension of the previous convolutional layer. For smooth extraction of higher order features, paper introduces Dynamic k-max pooling where the k in the k-max pooling operation is a function of the length of the input sentences.



where  is the num,ber of convolutional layer to which pooling is applied,  is total number of convolutional layers,  is the fixed pooling parameter of the top most convolutional layer.  The model also has a folding layer which sums over every two rows in the feature-map component wise. This folding operation is valid since feature detectors in different rows are independent before the fully connected layers.

Wide convolutions are preferred for the model instead of narrow convolutions, this is achieved in the code using appropriate zero padding.

This network models performance is related to its ability to capture the word and n-gram order in the sentences and to tell the relative position of the most relevant n-grams. The model also has the advantage of inducing a connected, directed acyclic graph with weighted edges and a root node as shown below.

Folding and K-max pooling layers are not readily available and has to be created using keras functional apis. Keras model summary is also presented below for reference.

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
dense_2 (Dense)                  (None, 9)             44937       flatten_1[0][0]
====================================================================================================
Total params: 70,878,569
Trainable params: 70,878,569
Non-trainable params: 0
____________________________________________________________________________________________________
```
