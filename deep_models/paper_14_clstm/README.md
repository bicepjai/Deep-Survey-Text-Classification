# [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630)

Zhou et al proposed C-LSTM Neural Networks which uses CNN to capture local features of phrases and RNN to capture global and temporal sentence semantics.

The model extracts high level correlations from n-gram features with shorter windows in CNN model and are fed as a sequence to the following LSTM layer which helps in creating an embedding for the document for classification. The output of each convolution (single feature map) has semantic information from the whole sentence as a sequence. From these multiple feature maps, without using a max-pool layer, the vectors representing positional information are grouped or concatenated which are fed to RNN layer as a sequence.

Keras model summary presented below

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 5000)          0
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 5000, 200)     70444000    input_1[0][0]
____________________________________________________________________________________________________
conv1d_1 (Conv1D)                (None, 4991, 64)      128064      embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_2 (Conv1D)                (None, 4981, 64)      256064      embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_3 (Conv1D)                (None, 4971, 64)      384064      embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_4 (Conv1D)                (None, 4961, 64)      512064      embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_5 (Conv1D)                (None, 4951, 64)      640064      embedding_1[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 24855, 64)     0           conv1d_1[0][0]
                                                                   conv1d_2[0][0]
                                                                   conv1d_3[0][0]
                                                                   conv1d_4[0][0]
                                                                   conv1d_5[0][0]
____________________________________________________________________________________________________
lstm_1 (LSTM)                    (None, 64)            33024       concatenate_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           8320        lstm_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 9)             1161        dense_1[0][0]
====================================================================================================
Total params: 72,406,825
Trainable params: 72,406,825
Non-trainable params: 0
____________________________________________________________________________________________________
```
