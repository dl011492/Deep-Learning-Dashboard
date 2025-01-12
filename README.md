# Deep Learning Dashboard

**Deep Learning Dashboard** is a Python and TensorFlow project that allows one to select between standard machine learning
datasets like: MNIST, Fashion MNIST, Kaggle cats and dogs, Oxford-IIIT pets, imdb and aclImdb.  
It is designed to easily display samples of selected datasets, select a specific neural network model (for example: MLP 512,
CNN Medium with data augmentation, mini Xception, Segmentation, Transformer Encoderetc), monitor their training and
evaluate the results.  
The scripts are by no means complete. Feel free to try to finish them, implement new models and improve them.  
**Warning** The datasets in this repository are compressed, but still large (2.6 GB). Be aware of this when cloning or
downloading files.

![Deep learning dashboard example 1](https://github.com/dl011492/Deep-Learning-Dashboard/blob/main/figures/dld_v02.png)

![Deep learning dashboard example 2](https://github.com/dl011492/Deep-Learning-Dashboard/blob/main/figures/dld_v02_b.png)

## Datasets
MNIST and Fashion MNIST are already part of TensorFlow Keras and one should not have any problems using them.
[Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog) is not the original dataset but a smaller dataset organized
and made smaller according to the example described in Chollet's book, p. 212.  
For the [Oxford-IIIT pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset, I used the book's example on p. 240. A minor speedup
in loading and pre-processing is achieved by accessing the provided ./data/pets/oxford-IIIT_pets.npz file directly.  
For text models, I am using the data from A. Maas [aclImdb](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) from the
book's example on p.  320.

## Features
All the models are taken from F. Chollet's book (see [Credits](#credits)).
- **Models for computer vision**: MLP 512, MLP-2L-96, MLP-128-Dropout, CNN Small for the datasets MNIST and Fashion MNIST.
- **Models for computer vision II**: CNN Medium-DA, VGG16-DA, mini Xception (Kaggle dataset), Segmentation (Oxford-IIIT).
- **Models for sentiment analysis**: MLP sent. analysis (imdb), bag-of-words 2g, 2LSTM-wembd, Transf encoder (aclImdb).
- **Model Training Tracking**: Monitor training progress with dynamic loss and accuracy diagrams.
- **Data Visualization**: Analyze datasets and outputs using built-in graphing tools.
- **Cross-Platform Compatibility**: Runs on both Windows and Linux systems.

---

## Requirements
This simple GUI is created using Tkinter.  
It has been tested under Windows 10 Pro, Win 10 IOT Enterprise, Win 11 Pro and Ubuntu 22.04.  
Unfortunately, TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows 
([Caution note](https://www.tensorflow.org/install/pip#windows-native)).     
If one wants to use a newer TensorFlow version on a Windows system, one needs to install
[WSL2](https://www.tensorflow.org/install/pip#windows-wsl2).  
 
For a comprehensive list of tested systems and their corresponding CUDA and cuDNN dependencies, look in
https://www.tensorflow.org/install/source#gpu.

Tested systems:  
- Windows 10: TensorFlow 2.10 with [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive) and 
  [cuDNN 8.1](https://developer.nvidia.com/rdp/cudnn-archive)
- Ubuntu 22.04: TensorFlow 2.18 with GPU support. No separate CUDA or cuDNN installation needed

Other requirements:
- At least 3.0 GB of free disk space and 16 GB of RAM

## TensorFlow Installation
For the Deep Learning Dashboard (DLD) script to work, you need a few other dependencies.  
On Linux systems (even on a Raspberry Pi 5!), the latest TensorFlow version, tf-2.18, and needed dependencies are very easy to install:

```bash
sudo apt-get install python3-pil.imagetk
pip install matplotlib
pip install tensorflow
pip install tqdm
```

## DLD Installation Example for version 11

1. Clone the repository using Git LFS to manage large dataset files:
   ```bash
   sudo apt update
   sudo apt install git-lfs
   git lfs install
   git clone https://github.com/dl011492/Deep-Learning-Dashboard.git
   mv Deep-Learning-Dashboard dld_v11 

2. Unzip Kaggle files:
   ```bash
   cd dld_v11/data/cats_vs_dogs_small
   unzip -q train.zip
   unzip -q validation.zip
   unzip -q test.zip
   rm *.zip 

3. Unzip aclImdb files:
   ```bash
   cd ../aclImdb
   unzip -q train.zip
   unzip -q validation.zip
   unzip -q test.zip  
   rm *.zip 

4. Place the data directory at the right position:
   ```bash
   cd ../..
   mv data .. 
 
5. Run the script:
   ```bash
   cd dld_v11
   python dld.py 

Eventually, you can rename the directory Deep-Learning-Dashboard to dld_v11. In this way, the
script will reflect your code version number in the Tkinter window.

---

## Credits
Most of the deep learning models and how to obtain the Kaggle and Oxford-IIIT datasets were taken from
the excellent book:
[Deep Learning with Python, 2nd Ed. F. Chollet, Manning Publications Co. 2021](https://www.manning.com/books/deep-learning-with-python-second-edition).  
Jupyter notebooks of the book examples are available on GitHub at
[github.com/fchollet/deep-learning-with-pythonnotebooks.](https://github.com/fchollet/deep-learning-with-python-notebooks)
