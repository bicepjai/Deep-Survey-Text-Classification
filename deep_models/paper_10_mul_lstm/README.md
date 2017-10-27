# [Multiplicative LSTM for sequence modelling](https://arxiv.org/abs/1609.07959)

Krause et al (2016) [30] proposed Multiplicative LSTM which is a hybrid RNN that gives the model more flexibility in choosing recurrent transitions for each possible input which makes it more expressive in autoregressive density estimation. This model is proposed for sequential language modelling, but I chose the model to see how it helps in text classification task. The authors argue that current RNNs has hard time to recover from mistakes when predicting sequences. If RNN hidden state remembers erroneous information then it might take more time steps to recover and will have snowball effect. Some proposed solutions such as introducing latent variables and increasing memory will only result in complex intractable distribution over the hidden states and reinterpreting stored inputs. The multiplicative model provides more flexibility for transitions in these kind of situations.

Since this is a modification to existing RNN unit (LSTM or GRU), this has to be implemented as a keras Layer similar to keras LSTM layer. The equations for mLSTM are

$m_t = (W_{mx}.x_t) \odot W_{mh}.h_t$
$\hat h_t = (W_{hx}.x_t) + W_{hm}.m_t$
$i_t = (W_{ix}.x_t) + W_{im}.m_t$
$o_t = (W_{ox}.x_t) + W_{om}.m_t$
$f_t = (W_{fx}.x_t) + W_{fm}.m_t$

Keras model summary is presented below

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_2 (Embedding)      (None, 60, 200)           70444000
_________________________________________________________________
multiplicative_lstm_1 (Multi (None, 32)                37280
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 297
=================================================================
Total params: 70,481,577
Trainable params: 70,481,577
Non-trainable params: 0
_________________________________________________________________
```

