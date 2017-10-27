# [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

Yang et al proposed Hierarchical Attention Networks for document classification which took advantage of the hierarchical structures that mirrors the hierarchical structure of the documents and two levels of attention mechanism applied to apply differentially more and less important content when constructing document representation.

First lets look into what attention mechanism means, for this let's look into the feedforward attention mechanism proposed by raffel et al (2015) [32]. Attention mechanism allows more direct dependencies between the states of the model at different points in time.
From the picture we can see vectors in the hidden state sequence are fed into a learnable function that produces a probability vector which in turn is used to find weighted average of the output hidden states.

Similarly Hierarchical Attention Networks proposes 2 stage attention mechanism to encode documents. The intuition behind the model is different words and sentences in a document are differentially informative. The document representations are constructed from each sentences and each sentence representation are constructed from words. Each sentence in a document can be considered as a word sequence which is fed into a generative LSTM layer with attention on top and again the same procedure is followed for sentences with attention on top. Finally a softmax layer on top to classifying the document labels.

Keras does not provide apis for attention mechanism, hence custom Attention layer is created to handle different levels of attentions. Keras model summary presented below.

Word encoder model for each eentence
```
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 80, 200)           70444000
_________________________________________________________________
bidirectional_1 (Bidirection (None, 80, 50)            33900
_________________________________________________________________
attention_with_context_1 (At (None, 50)                210
=================================================================
Total params: 70,478,110
Trainable params: 70,478,110
Non-trainable params: 0
_________________________________________________________________
```

Sentence encoder model for each document
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 500, 80)           0
_________________________________________________________________
time_distributed_1 (TimeDist (None, 500, 50)           70478110
_________________________________________________________________
bidirectional_2 (Bidirection (None, 500, 50)           11400
_________________________________________________________________
attention_with_context_2 (At (None, 50)                1050
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 459
=================================================================
Total params: 70,491,019
Trainable params: 70,491,019
Non-trainable params: 0
_________________________________________________________________
```
