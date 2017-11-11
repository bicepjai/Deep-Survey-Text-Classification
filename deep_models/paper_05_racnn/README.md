# [Rationale-Augmented Convolutional Neural Networks for Text Classification](https://arxiv.org/abs/1605.04469)

Ye Zhang et al presents RA-CNN model that jointly exploits labels on documents and their constituent sentences. The model tries to estimate the probability that a given sentence is rationales and then scale the contribution of each sentence to aggregate a document representation in proportion to the estimates. Rationales are sentences that directly support document classification.

To make the understanding of RA-CNN simpler, authors explain Doc-CNN model. In this model, a CNN model is applied over each sentence in a document and then all the generated sentence level vectors are added to form a document vector. As before, we add a softmax layer to perform document classification. Regularization is applied to both the document and sentence level vector output.

RA-CNN model is same as Doc-CNN but document vector is created as weighted sum of its constituent sentence. There are 2 stages in training this architecture, sentence level training and document level training.

For the former stage, each sentence is provided with 3 classes positive rationales, negative rationales and neutral rationales. Then with a softmax layer parametrized with its own weights  (will contain 3 vectors, one for each class) over the sentences, we fit this sub-model to maximize the probabilities of the sentences being one of the rationales class. This would provide the conditional probability estimates regarding whether the sentence is a positive or negative rationale.

For the document level training, the document vector is estimated using the weighted sum of the constituent sentence vectors. The weights are set to the estimated probabilities that corresponding sentences are rationales in the most likely direction. The probabilities considered for the weights are maximum of 2 classes positive and negative rationale (neutral class is omitted). The intuition is that sentences likely to be rationales will have greater influence on the resultant document vector. The final document vector is used to perform classification on the document labels. When performing document level training, we freeze the sentence weights $W_{sen}$  and initialize the embeddings and other conv layer parameters tuned during sentence level training.

Keras model summary for RA-CNN

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 500, 80)       0
____________________________________________________________________________________________________
reshape_1 (Reshape)              (None, 40000)         0           input_1[0][0]
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 40000, 200)    70444000    reshape_1[0][0]
____________________________________________________________________________________________________
reshape_2 (Reshape)              (None, 1, 500, 16000) 0           embedding_1[0][0]
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 32, 500, 79)   12832       reshape_2[0][0]
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 32, 500, 78)   19232       reshape_2[0][0]
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 32, 500, 1)    0           conv2d_1[0][0]
____________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)   (None, 32, 500, 1)    0           conv2d_2[0][0]
____________________________________________________________________________________________________
permute_1 (Permute)              (None, 500, 32, 1)    0           max_pooling2d_1[0][0]
____________________________________________________________________________________________________
permute_2 (Permute)              (None, 500, 32, 1)    0           max_pooling2d_2[0][0]
____________________________________________________________________________________________________
reshape_3 (Reshape)              (None, 500, 32)       0           permute_1[0][0]
____________________________________________________________________________________________________
reshape_4 (Reshape)              (None, 500, 32)       0           permute_2[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 500, 64)       0           reshape_3[0][0]
                                                                   reshape_4[0][0]
____________________________________________________________________________________________________
sentence_predictions (TimeDistri (None, 500, 9)        585         concatenate_1[0][0]
____________________________________________________________________________________________________
time_distributed_1 (TimeDistribu (None, 500)           0           sentence_predictions[0][0]
____________________________________________________________________________________________________
reshape_5 (Reshape)              (None, 1, 500)        0           time_distributed_1[0][0]
____________________________________________________________________________________________________
dot_1 (Dot)                      (None, 64, 1)         0           concatenate_1[0][0]
                                                                   reshape_5[0][0]
____________________________________________________________________________________________________
reshape_6 (Reshape)              (None, 64)            0           dot_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 64)            0           reshape_6[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 9)             585         dropout_1[0][0]
====================================================================================================
Total params: 70,477,234
Trainable params: 70,477,234
Non-trainable params: 0
____________________________________________________________________________________________________

```
