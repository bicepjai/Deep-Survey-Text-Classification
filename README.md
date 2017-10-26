# Deep-Survey-on-Text-Classification

This is a survey on deep learning models for text classification and will be updated frequently with testing and evaluation on different datasets.


Natural Language Processing tasks ( part-of-speech tagging, chunking, named entity recognition, text classification, etc .) has gone through tremendous amount of research over decades. Text Classification has been the most competed NLP task in kaggle and other similar competitions. Count based models are being phased out with new deep learning models emerging almost every month. This project is an attempt to survey most of the neural based models for text classification task. Models selected, based on CNN and RNN, are explained with code (keras and tensorflow) and block diagrams. The models are evaluated on one of the active kaggle competition medical dataset.


## Project setup

1. Download and install anaconda3 and create a virtual environment using `conda create`
2. Install `pip` using `conda` from the virtual anaconda3 environment (pip installing in current env)
3. Use this [requirements](https://github.com/bicepjai/Deep-Survey-Text-Classification/blob/master/requirements.txt) file for installing dependencies using pip
4. make sure jupyter notebooks work and I have some extensions enabled for eas of view and navigation

Now we should be ready to run this project. The details regarding the machine used for training can be found [here](https://bicepjai.github.io/machine-learning/2015/05/25/machine-learning-rig.html)

## Data

The data used for evaluating all the models are taken from an active competition [Personalized Medicine: Redefining Cancer Treatment](https://www.kaggle.com/c/msk-redefining-cancer-treatment) launched by Kaggle along with Memorial Sloan Kettering Cancer Center (MSKCC). This has been accepted by the NIPS 2017 Competition Track, because they need data scientists to help take personalized medicine to its full potential. Once sequenced, a cancer tumor can have thousands of genetic mutations. But the challenge is distinguishing the mutations that contribute to tumor growth (drivers) from the neutral mutations (passengers). Currently, this interpretation of genetic mutations is being done manually. This is a very time-consuming task where a clinical pathologist has to manually review and classify every single genetic mutation based on evidence from text-based clinical literature. For this competition MSKCC is making available an expert-annotated knowledge base where world-class researchers and oncologists have manually annotated thousands of mutations. One needs to develop Machine Learning algorithms that, using this knowledge base as a baseline, automatically classifies genetic variations.

Data can be downloaded [here](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data)

## Content

The papers explored in this  project

1. [Convolutional Neural Networks for Sentence Classification](https://github.com/bicepjai/Deep-Survey-Text-Classification/tree/master/deep_models/paper_1_cnn_modelling_sentences), Yoon Kim

2.
