import os
import pathlib
import random
import numpy as np
import tqdm as tq
import tensorflow as tf

# Set the logger to display only errors or more critical messages
#tf.get_logger().setLevel('ERROR')

class Dataset:
    def __init__(self, dataset):
        # Initialize attributes for TensorFlow datasets as None by default
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.train = None
        self.train_labels = None
        self.val = None
        self.val_labels = None
        self.test = None
        self.test_labels = None
        
        if dataset == "MNIST":
            # Load MNIST dataset
            mnist = tf.keras.datasets.mnist.load_data()        
            (self.train, self.train_labels), (self.test, self.test_labels) = mnist

        if dataset == "Fashion MNIST":
            # Load Fashion MNIST dataset
            fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()        
            (self.train, self.train_labels), (self.test, self.test_labels) = fashion_mnist
     
        if dataset == "Kaggle":
            # Load Kaggle dataset
            new_base_dir = pathlib.Path("./data/cats_vs_dogs_small")
            self.train_dataset = tf.keras.utils.image_dataset_from_directory(
                new_base_dir / "train",
                image_size = (180, 180),
                batch_size = 32)
            self.validation_dataset = tf.keras.utils.image_dataset_from_directory(
                new_base_dir / "validation",
                image_size = (180, 180),
                batch_size = 32)
            self.test_dataset = tf.keras.utils.image_dataset_from_directory(
                new_base_dir / "test",
                image_size = (180, 180),
                batch_size = 32)

            # Convert the datasets to numpy arrays
            self.train, self.train_labels = self.dataset_to_numpy(self.train_dataset)
            self.val, self.val_labels = self.dataset_to_numpy(self.validation_dataset)
            self.test, self.test_labels = self.dataset_to_numpy(self.test_dataset)
            
        if dataset == "Oxford-IIIT":
            # Load Oxford-IIIT dataset
            data_path = pathlib.Path("./data/pets/oxford-IIIT_pets.npz")

            def load_arrays_with_progress(file_path):
                pets_data = np.load(file_path)
                arrays = {}
                for key in tq.tqdm(pets_data.keys(), desc = "Loading data...."):
                    arrays[key] = pets_data[key]
                return arrays
            pets_arrays = load_arrays_with_progress(data_path)
            
            self.train = pets_arrays["train"]
            self.train_labels = pets_arrays["train_labels"]
            self.val = pets_arrays["val"]
            self.val_labels = pets_arrays["val_labels"]
            self.test = pets_arrays["test"]
            self.test_labels = pets_arrays["test_labels"]
            
        # Normalize the pixel values to be between 0 and 1
        if self.train is not None:
            self.train = self.train.astype("float32") / 255.0
        if self.val is not None:
            self.val = self.val.astype("float32") / 255.0
        if self.test is not None:
            self.test = self.test.astype("float32") / 255.0
            
        if dataset == "imdb":
            def vectorize_seq(seqs, dim = 10000):
                results = np.zeros((len(seqs), dim))
                for i, seq in enumerate(seqs):
                    results[i, seq] = 1.
                return results
        
            imdb = tf.keras.datasets.imdb.load_data(num_words = 10000)
            (self.train, self.train_labels), (self.test, self.test_labels) = imdb

            self.train = vectorize_seq(self.train)
            self.train_labels = np.asarray(self.train_labels).astype("float32")
            self.test = vectorize_seq(self.test)
            self.test_labels = np.asarray(self.test_labels).astype("float32")

            # Creating a validation set
            self.train = self.train[10000:]
            self.train_labels = self.train_labels[10000:]            
            self.val = self.train[:10000]
            self.val_labels = self.train_labels[:10000]

    # Function to convert tf.data.Dataset to NumPy arrays
    def dataset_to_numpy(self, dataset):
        images = []
        labels = []    
        for image_batch, label_batch in dataset:
            images.append(image_batch.numpy())  # Convert images to NumPy
            labels.append(label_batch.numpy())  # Convert labels to NumPy
        # Concatenate list of batches into full arrays
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)    
        return images, labels
            
    def get_train_data(self):
        return self.train, self.train_labels

    def get_train_dataset(self):
        return self.train_dataset
    
    def get_val_data(self):
        return self.val, self.val_labels

    def get_val_dataset(self):
        return self.validation_dataset
    
    def get_test_data(self):
        return self.test, self.test_labels

    def get_test_dataset(self):
        return self.test_dataset 

    def get_train_info(self):
        return self.train.shape, self.train.dtype
