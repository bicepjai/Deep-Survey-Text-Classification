# [Combination of Convolutional and Recurrent Neural Network for Sentiment Analysis of Short Texts](https://pdfs.semanticscholar.org/a0c3/b9083917b6c2368ebf09483a594821c5018a.pdf)

Wang et al proposed another model for sentiment analysis for shorter texts which is a combination of CNN and RNN. The model helps capturing coarse grained local features from CNN and long distance dependencies from RNN.
The model is very similar to C-LSTM, but it employs a max pool layer with multiple feature maps that reduces the output of convolution by half.

Keras model summary presented below

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 60)            0
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 60, 200)       70444000    input_1[0][0]
____________________________________________________________________________________________________
conv1d_1 (Conv1D)                (None, 58, 64)        38464       embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_2 (Conv1D)                (None, 57, 64)        51264       embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_3 (Conv1D)                (None, 56, 64)        64064       embedding_1[0][0]
____________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)   (None, 19, 64)        0           conv1d_1[0][0]
____________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)   (None, 14, 64)        0           conv1d_2[0][0]
____________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)   (None, 11, 64)        0           conv1d_3[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 44, 64)        0           max_pooling1d_1[0][0]
                                                                   max_pooling1d_2[0][0]
                                                                   max_pooling1d_3[0][0]
____________________________________________________________________________________________________
lstm_1 (LSTM)                    (None, 64)            33024       concatenate_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           8320        lstm_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 9)             1161        dense_1[0][0]
====================================================================================================
Total params: 70,640,297
Trainable params: 70,640,297
Non-trainable params: 0
____________________________________________________________________________________________________
```
