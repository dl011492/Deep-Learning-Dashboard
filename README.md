# Deep Learning Dashboard

**Deep Learning Dashboard** is a Python and TensorFlow project that allows one to select between standard machine learning
datasets like: MNIST, Fashion MNIST, Kaggle cats and dogs and Oxford-IIIT pets.  
It is designed to easily display samples of selected datasets, select a specific neural network model (MLP 512, CNN Small,
CNN Medium with data augmentation, VGG16-DA, mini Xception, Segmentation, sentiment analysis) monitor their training and
evaluate the results. The scripts are by no means complete. Feel free to try to finish them, implement new models and
improve them.  
**Warning** The datasets in this repository are compressed, but still large (1.2 GB). Be aware of this when cloning or
downloading files.

![Deep learning dashboard example 1](https://github.com/dl011492/Deep-Learning-Dashboard/blob/main/figures/dld_v02.png)

![Deep learning dashboard example 2](https://github.com/dl011492/Deep-Learning-Dashboard/blob/main/figures/dld_v02_b.png)

## Datasets
MNIST and Fashion MNIST are already part of TensorFlow Keras and one should not have any problems using them.
[Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog) is not the original dataset but a smaller dataset organized
and made smaller according to the example described in Chollet's book, p. 212.
For the [Oxford-IIIT pets](www.robots.ox.ac.uk/~vgg/data/pets/) dataset, I used the book's example on p. 240. A minor speedup
in loading and pre-processing is achieved by accessing the provided ./data/pets/oxford-IIIT_pets.npz file directly.

---
## Requirements
This simple GUI is created using Tkinter. It works fine with the following minimum requirements: 
- Python 3.9
- TensorFlow 2.10 (to be able to use CUDA 11.2 cuDNN 8.1 on Windows 10)
- The script should also work on newer TensorFlow versions under Linux
- At least 1.1 GB of free space

## Features

- **Model Training Tracking**: Monitor training progress with dynamic loss and accuracy diagrams.
- **Data Visualization**: Analyze datasets and outputs using built-in graphing tools.
- **Cross-Platform Compatibility**: Runs seamlessly on both Windows and Linux systems.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dl011492/Deep-Learning-Dashboard.git
   cd Deep-Learning-Dashboard
2. Unzip the Kaggle files:
   ```bash
   cd data/cats_vs_dogs_small
   gzip -d train.zip validation.zip test.zip 
4. Run the script:
   ```bash
   python dld.py 

## Credits
Most of the deep learning models and how to obtain the Kaggle and Oxford-IIIT datasets were taken from
the excellent book:
[Deep Learning with Python, 2nd Ed. F. Chollet, Manning Publicatiiosn Co. 2021](https://www.manning.com/books/deep-learning-with-python-second-edition)
. Jupyter notebooks of the book examples are available on GitHub at
[github.com/fchollet/deep-learning-with-pythonnotebooks.](https://github.com/fchollet/deep-learning-with-python-notebooks)
