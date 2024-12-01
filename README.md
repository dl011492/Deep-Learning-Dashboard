# Deep Learning Dashboard

**Deep Learning Dashboard** is a Python and TensorFlow project that allows one to select between standard machine learning
datasets like: MNIST, Fashion MNIST, Kaggle cats and dogs and Oxford-IIIT pets.
It is designed to easily display samples of selected datasets, select a specific neural network model (MLP 512, CNN Small,
CNN Medium with data augmentation, VGG16-DA, mini Xception, Segmenattion, sentiment analysis) monitor training and evaluate
the results. The script is by no means complete. Feel free to try to finish it, implement new models and improve it.

![Deep learning dashboard example 1](https://github.com/dl011492/Deep-Learning-Dashboard/blob/main/figures/dld_v02.png)

![Deep learning dashboard example 2](https://github.com/dl011492/Deep-Learning-Dashboard/blob/main/figures/dld_v02_b.png)

## Datasets
MNIST and Fashion MNIST are already part of TensorFlow and one should not have any problems using these. Kaggle is not the
original dataset (available at https://www.kaggle.com/datasets/tongpython/cat-and-dog) but a smaller dataset. 
It has been organized and made smaller acording to the example described in Chollet's book, p. 212.
Oxford-IIIT pets (www.robots.ox.ac.uk/~vgg/data/pets/) p. 240.

---
## Requirements
This simple GUI is created using Tkinter. It works fine with the following minnimum requirements: 
- ## Python 3.9
- ## TensorFlow 2.10 (to be able to use CUDA 11.2 cuDNN 8.1 on Windows 10)
- ## It will work with newer TensorFlow version under Linux

## Features

- **Model Training Tracking**: Monitor training progress with dynamic loss and accuracy diagrams.
- **Data Visualization**: Analyze datasets and outputs using built-in graphing tools.
- **Cross-Platform Compatibility**: Runs seamlessly on both Windows and Linux systems.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Deep-Learning-Dashboard.git
   cd Deep-Learning-Dashboard

## Credits
Most of the deep learning examples were taken from the excellent book:
Deep Learning with Python, 2nd Ed. Francois Chollet, Manning Publications Co. 2021.
