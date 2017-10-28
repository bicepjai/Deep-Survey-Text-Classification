# [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9745/9552)

Lai et al proposed RCNN model that addresses the limitation of unbiased CNN model with shorter window sizes and biased RNN models. The model has bi-directional recurrent structure that reduces noise and captures semantic information to the greatest extent possible. Max pool layer on top of recurrent structure judges the features role in capturing key components necessary for classification.


The model has bidirectional LSTM that captures context information from nearby words in both the directions. Then the max pool layer captures the key features that are fed as input to the softmax layer that classifies the text provided.

Keras model summary presented below


```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_2 (InputLayer)             (None, None)          0
____________________________________________________________________________________________________
input_3 (InputLayer)             (None, None)          0
____________________________________________________________________________________________________
embedding_2 (Embedding)          (None, 5000, 200)     70444000    input_2[0][0]
____________________________________________________________________________________________________
input_1 (InputLayer)             (None, None)          0
____________________________________________________________________________________________________
embedding_3 (Embedding)          (None, 5000, 200)     70444000    input_3[0][0]
____________________________________________________________________________________________________
lstm_1 (LSTM)                    (None, 5000, 100)     120400      embedding_2[0][0]
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 5000, 200)     70444000    input_1[0][0]
____________________________________________________________________________________________________
lstm_2 (LSTM)                    (None, 5000, 100)     120400      embedding_3[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 5000, 400)     0           lstm_1[0][0]
                                                                   embedding_1[0][0]
                                                                   lstm_2[0][0]
____________________________________________________________________________________________________
time_distributed_1 (TimeDistribu (None, 5000, 200)     80200       concatenate_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 200)           0           time_distributed_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 9)             1809        lambda_1[0][0]
====================================================================================================
Total params: 211,654,809
Trainable params: 211,654,809
Non-trainable params: 0
____________________________________________________________________________________________________
```
