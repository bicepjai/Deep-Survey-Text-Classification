# Deep-Survey-on-Text-Classification

This is a survey on deep learning models for text classification and will be updated frequently with testing and evaluation on different datasets.


Natural Language Processing tasks ( part-of-speech tagging, chunking, named entity recognition, text classification, etc .) has gone through tremendous amount of research over decades. Text Classification has been the most competed NLP task in kaggle and other similar competitions. Count based models are being phased out with new deep learning models emerging almost every month. This project is an attempt to survey most of the neural based models for text classification task. Models selected, based on CNN and RNN, are explained with code (keras and tensorflow) and block diagrams. The models are evaluated on one of the  kaggle competition medical dataset.

Update:
Non stop training and power issues in my geographic location burned my motherboard. By the time i had to do 2 RMAs with ASROCK and get the system up and running, the competition was over :( but still i learned a lot.

## Project setup

1. Download and install anaconda3 and create a virtual environment using `conda create`
2. Install `pip` using `conda` from the virtual anaconda3 environment (pip installing in current env)
3. Use this [requirements](https://github.com/bicepjai/Deep-Survey-Text-Classification/blob/master/requirements.txt) file for installing dependencies using pip
4. make sure jupyter notebooks work and I have some extensions enabled for eas of view and navigation

Now we should be ready to run this project. The details regarding the machine used for training can be found [here](https://bicepjai.github.io/machine-learning/2015/05/25/machine-learning-rig.html)

Version Reference on some important packages used

1. Keras==2.0.8
2. tensorflow-gpu==1.3.0
3. tensorflow-tensorboard==0.1.8


## Data

Details regarding the data used can be found  [here](https://github.com/bicepjai/Deep-Survey-Text-Classification/blob/master/data_prep/dataset/README.md)

## Content

This project is completed and now the write up is in the process. It can be viewed [here](https://docs.google.com/document/d/1zAh2lUjweKR8o5OZkv-48NWMVW_Pvvy5O953A-9KcNM/edit?usp=sharing). The papers explored in this project

1. [Convolutional Neural Networks for Sentence Classification, Yoon Kim (2014)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_01_cnn_sent_class)
2. [A Convolutional Neural Network for Modelling Sentences, Nal Kalchbrenner, Edward Grefenstette, Phil Blunsom (2014)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_02_cnn_sent_model)
3. [Medical Text Classification using Convolutional Neural Networks, Mark Hughes, Irene Li, Spyros Kotoulas, Toyotaro Suzumura (2017)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_03_med_cnn)
4. [Very Deep Convolutional Networks for Text Classification, Alexis Conneau, Holger Schwenk, Loïc Barrault, Yann Lecun (2016)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_04_vdcnn)
5. [Rationale-Augmented Convolutional Neural Networks for Text Classification, Ye Zhang, Iain Marshall, Byron C. Wallace (2016)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_05_racnn)
6. [Multichannel Variable-Size Convolution for Sentence Classification, Wenpeng Yin, Hinrich Schütze (2016)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_06_mvcnn)
7. [MGNC-CNN: A Simple Approach to Exploiting Multiple Word Embeddings for Sentence Classification Ye Zhang, Stephen Roller, Byron Wallace (2016)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_07_mgnccnn)
8. [Generative and Discriminative Text Classification with Recurrent Neural Networks, Dani Yogatama, Chris Dyer, Wang Ling, Phil Blunsom (2017)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_08_lstm)
9. [Deep Sentence Embedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval, Hamid Palangi, Li Deng, Yelong Shen, Jianfeng Gao, Xiaodong He, Jianshu Chen, Xinying Song, Rabab Ward](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_09_dse_lstm)
10. [Multiplicative LSTM for sequence modelling, Ben Krause, Liang Lu, Iain Murray, Steve Renals (2016)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_10_mul_lstm)
11. [Hierarchical Attention Networks for Document Classification, Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy (2016)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_11_hier_att_net)
12. [Recurrent Convolutional Neural Networks for Text Classification, Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao (2015)](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_x12_rcnn)
14. more paper-implementations on the way ...

