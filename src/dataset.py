import os
import time
import pathlib
import random
from collections import Counter
import numpy as np
import tqdm as tq
import tensorflow as tf

class Dataset:
    def __init__(self, dataset):
        # Initialize attributes for TensorFlow datasets as None by default
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.vec_test_dataset = None
        
        self.train = None
        self.train_labels = None
        self.val = None
        self.val_labels = None
        self.test = None
        self.test_labels = None

        # Datasets for computer vision
        print(f"Loading {dataset} ...")
        start_util = time.time()
        if dataset == "MNIST":                    # Load MNIST data            
            #start_util = time.time()
            mnist = tf.keras.datasets.mnist.load_data()        
            (self.train, self.train_labels), (self.test, self.test_labels) = mnist

        if dataset == "Fashion MNIST":            # Load Fashion MNIST data           
            #start_util = time.time()
            fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()        
            (self.train, self.train_labels), (self.test, self.test_labels) = fashion_mnist
     
        if dataset == "Kaggle":                   # Load Kaggle cats_vs_dogs dataset
            #start_util = time.time()
            base_dir = pathlib.Path("../data/cats_vs_dogs_small")
            batch_size = 32            
            self.train_dataset = tf.keras.utils.image_dataset_from_directory(
                base_dir / "train",
                image_size = (180, 180),
                batch_size = batch_size)
            self.validation_dataset = tf.keras.utils.image_dataset_from_directory(
                base_dir / "validation",
                image_size = (180, 180),
                batch_size = batch_size)
            self.test_dataset = tf.keras.utils.image_dataset_from_directory(
                base_dir / "test",
                image_size = (180, 180),
                batch_size = batch_size)

            # Convert the datasets to numpy arrays
            self.train, self.train_labels = self.dataset_to_numpy(self.train_dataset)
            self.val, self.val_labels = self.dataset_to_numpy(self.validation_dataset)
            self.test, self.test_labels = self.dataset_to_numpy(self.test_dataset)
            
        if dataset == "Oxford-IIIT":              # Load Oxford-IIIT data          
            #start_util = time.time()
            base_dir = pathlib.Path("../data/pets/oxford-IIIT_pets.npz")

            def load_arrays_with_progress(file_path):
                pets_data = np.load(file_path)
                arrays = {}
                for key in tq.tqdm(pets_data.keys(), desc = "Processing npz file...."):
                    arrays[key] = pets_data[key]
                return arrays
            pets_arrays = load_arrays_with_progress(base_dir)

            print("Storing and normalizing...")
            self.train = pets_arrays["train"]
            self.train_labels = pets_arrays["train_labels"]
            self.val = pets_arrays["val"]
            self.val_labels = pets_arrays["val_labels"]
            self.test = pets_arrays["test"]
            self.test_labels = pets_arrays["test_labels"]
            
        # Normalize pixel values to be between 0 and 1 for visualization.
        # This applies to: MNIST, Fashion MNIST, Kaggle (NumPy arrays), Oxford-IIIT
        if self.train is not None:
            self.train = self.train.astype("float32") / 255.0
        if self.val is not None:
            self.val = self.val.astype("float32") / 255.0
        if self.test is not None:
            self.test = self.test.astype("float32") / 255.0

        # Datasets for natural language processing    
        if dataset == "imdb":                     # Load imdb data 
            #start_util = time.time()

            voc = 10000    # vocabulary size
            def build_vocab(train_data, voc = voc):
                # Flatten the training data into a single list of tokens
                all_tokens = [token for seq in train_data for token in seq]
                word_counts = Counter(all_tokens)
                most_common = word_counts.most_common(voc - 1)  # Reserve 1 spot for "unknown"

                # Create mappings
                word_to_index = {word: i+1 for i, (word, _) in enumerate(most_common)}  # Index 0 reserved
                word_to_index["<UNK>"] = 0  # Unknown words map to 0
                index_to_word = {i: word for word, i in word_to_index.items()}
    
                return word_to_index, index_to_word

            def words_to_indices(seqs, word_to_index):
                return [
                    [word_to_index.get(word, 0) for word in seq]  # Map to index or <UNK>
                    for seq in seqs
                    ]
            
            def vectorize_seq(seqs, dim = voc):
                results = np.zeros((len(seqs), dim))
                for i, seq in enumerate(seqs):
                    results[i, seq] = 1.
                return results
        
            imdb = tf.keras.datasets.imdb.load_data(num_words = voc)
            (self.train, self.train_labels), (self.test, self.test_labels) = imdb

            # build the vocabulary
            self.word_to_index, self.index_to_word = build_vocab(self.train, voc)
            # Convert words to indices
            sequences_as_indices = words_to_indices(self.train, self.word_to_index)
            # Vectorize sequences
            self.train = vectorize_seq(self.train)
            
            self.train_labels = np.asarray(self.train_labels).astype("float32")
            self.test = vectorize_seq(self.test)
            self.test_labels = np.asarray(self.test_labels).astype("float32")

            # Creating a validation set
            self.train = self.train[10000:]
            self.train_labels = self.train_labels[10000:]            
            self.val = self.train[:10000]
            self.val_labels = self.train_labels[:10000]

        if dataset == "aclImdb":                  # Load aclImdb dataset            
            base_dir = pathlib.Path("../data/aclImdb")
            batch_size = 32
            self.train_dataset = tf.keras.utils.text_dataset_from_directory(
                base_dir / "train",
                batch_size = batch_size)
            self.validation_dataset = tf.keras.utils.text_dataset_from_directory(
                base_dir / "validation",
                batch_size = batch_size)
            self.test_dataset = tf.keras.utils.text_dataset_from_directory(
                base_dir / "test",
                batch_size = batch_size)

        end_util = time.time()
        print(f"Loading time: {(end_util - start_util):.1f} s")

            
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

    def get_train_dataset_info(self):
        base_dir = pathlib.Path("../data/aclImdb")
        for self.inputs, self.targets in self.train_dataset:
            break
        total_samples = sum(len(files) for _, _, files in os.walk(base_dir / "train")) - 5
        return (total_samples,), self.inputs.dtype.name

    def get_vocabulary(self):
        return self.word_to_index, self.index_to_word
        
