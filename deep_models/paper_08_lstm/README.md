# [Generative and Discriminative Text Classification with Recurrent Neural Networks](https://arxiv.org/abs/1703.01898)

Yogatama et al (2017) [28] studies generative and discriminative LSTM models for text classification. Suggested generative models approach asymptotic errors faster than than the discriminative models.

In the discriminative model the LSTM “reads” the document and uses its hidden representation to model the class posterior. Discriminative models are known to suffer from catastrophic forgetting when learning sequentially. Discriminative model uses LSTM with “peephole” connections to encode a document and build a classifier on top of the encoder by using the average of the LSTM hidden representations as the document representation.

Generative models are robust to shifts in data distribution. generative models for text classification,proposed network can model unbounded (conditional) dependencies among words in each document. Here, documents are generated word by word and conditioned on a learned class embedding. This model takes advantage of the setup and maximizes the training objective for a new class, while decoupling other classes more easily. To types of generative models proposed are
In shared LSTM, to predict a single word, the LSTM’s hidden representations are concatenated with a separate label class embedding and another softmax layer over vocabulary is added.
In independant LSTM, the sharing of parameters are removed which also results in more parameters in the model.


In this project discriminative model is implemented. Keras model summary presented below.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_11 (Embedding)     (None, 60, 200)           70444000
_________________________________________________________________
lstm_5 (LSTM)                (None, 60, 32)            29824
_________________________________________________________________
flatten_2 (Flatten)          (None, 1920)              0
_________________________________________________________________
dense_5 (Dense)              (None, 9)                 17289
=================================================================
Total params: 70,491,113
Trainable params: 70,491,113
Non-trainable params: 0
_________________________________________________________________
```
