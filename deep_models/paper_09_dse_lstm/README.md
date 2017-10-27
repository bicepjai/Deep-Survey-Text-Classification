# [Deep Sentence Embedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval](https://arxiv.org/abs/1502.06922)

Palangi et al proposed LSTM-RNN for text classification. The model has shown to perform better than Paragraph Vectors for document/sentence embedding. As the model reads to the end of the sentence, the topic activation accumulate and the hidden state representation at the last word encodes the rich contextual information of the entire sentence. LSTM-RNN are effective against noise and can be robust in scenarios such as when every word in the document is not equally important and only salient words needs to be remembered using limited memory.

This is a very simple model that takes all the words in the document sequentially and the final output gives the document embedding. The model does not use max pool layers to capture global contextual information.

Keras model summary presented below.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_2 (Embedding)      (None, 60, 200)           70444000
_________________________________________________________________
lstm_1 (LSTM)                (None, 32)                29824
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 297
=================================================================
Total params: 70,474,121
Trainable params: 70,474,121
Non-trainable params: 0
_________________________________________________________________
