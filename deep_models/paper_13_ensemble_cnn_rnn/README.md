# [Ensemble Application of Convolutional and Recurrent Neural Networks for Multi-label Text Categorization](http://sentic.net/convolutional-and-recurrent-neural-networks-for-text-categorization.pdf)

Chen et al proposed a model that is an ensemble of CNN and RNN for multi label text categorization which helps capturing both the global and local semantics. It employs a CNN model as proposed in section CNN for sentence classification and a RNN model that returns output as sequence which are classified for word labelling.

The model is modified to be used for text classification. The RNN structure is used to create sentence embedding with the same input sentence provided for CNN. The output of the CNN model is fed to the first hidden state of the RNN that helps with better information retrieval.
