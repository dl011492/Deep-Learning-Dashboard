import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import tensorflow as tf
import src.textWinFrame as tw
from src.models import TransformerEncoder, PositionalEmbedding

fg, bg = "white", "navy"

class PredictApp(tk.Frame):
    def __init__(self, parent, shared_data, shared_settings):
        super().__init__(parent)
        self.shared_data = shared_data
        self.shared_settings = shared_settings
        
        self.predictFrame = tk.Frame(parent, bg = bg, width = 250, height = 500)
        self.predictFrame.pack_propagate(False)
        self.predictFrame.grid(row = 1, column = 2, rowspan = 2, sticky = "nsew")

        # Frame title
        self.lbl = tk.Label(self.predictFrame, text = "Model Prediction", fg = fg, bg = bg)
        self.lbl.pack(anchor = "n", pady = 10)
        
        # Button to run a test
        self.button = tk.Button(self.predictFrame, width = 20, text = "Visualize Test Sample",
                                fg = fg, bg = bg, command = self.on_visTest_click)
        self.button.pack(pady = (10, 20))

        # Label to display the prediction
        self.lbl_pred = tk.Label(self.predictFrame, text = "Predicted Value:", fg = fg, bg = bg)
        self.lbl_pred.pack(pady = (0, 10))

        # Label to display the test label
        self.lbl_label = tk.Label(self.predictFrame, text = "Test Sample Label:", fg = fg, bg = bg)
        self.lbl_label.pack(pady = (0, 20))
        
    def on_visTest_click(self):        
        # Access the test data and labels from the shared data
        test_data = self.shared_data.get('test_data')
        test_labels = self.shared_data.get('test_labels')
        
        # Access the test datasets from the shared data
        #train_dataset = self.shared_data.get('train_dataset')   # needed for vocabularies
        test_dataset = self.shared_data.get('test_dataset')
        
        dataset = self.shared_data.get('dataset')
        if test_data is None and test_dataset is None:
            self.lbl_pred.config(text = "Please select a dataset")
        else:
            ext = self.shared_settings['extension']
            if test_data is not None:
                max_test = int(len(test_data))
                index = random.randint(0, max_test)
            if dataset == "MNIST":
                # Select a test sample
                test_image = tf.reshape(test_data[index,:], (28, 28))
                self.generatePlot(test_image)

                # Make a prediction
                model = tf.keras.models.load_model('./cache/mnist_model' + ext)
                y_proba = model.predict(test_data[index:index+1], verbose = 0)
                class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]
                pred = y_proba[0].round(2)
                pred_index = pred.argmax()
                self.lbl_pred.config(text = f"Predicted Value: {class_names[pred_index]}")
                
                # Display the test label               
                self.lbl_label.config(text = f"Test Sample Label: {test_labels[index]}")
                         
            if dataset == "Fashion MNIST":
                # Select a test sample
                test_image = tf.reshape(test_data[index,:], (28, 28))
                self.generatePlot(test_image)

                # Make a prediction
                model = tf.keras.models.load_model('./cache/fashion_mnist_model' + ext)
                y_proba = model.predict(test_data[index:index+1], verbose = 0)                
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                pred = y_proba[0].round(2)
                pred_index = pred.argmax()
                self.lbl_pred.config(text = f"Predicted Value: {class_names[pred_index]}")

                # Display the test label
                test_ind = test_labels[index]
                self.lbl_label.config(text = f"Test Sample Label: {class_names[test_ind]}")
                
            if dataset == "Kaggle":
                class_names = ['cat', 'dog']
                # Select first test image from the batch and expand dims to simulate a batch of 1 image
                for i, element in enumerate(test_dataset):
                    test_images, test_labels = element                # Unpack the tuple
                    test_image_reshaped = tf.reshape(test_images[0], (180, 180, 3)) / 255.0
                    if i >= 0:
                        break                
                self.generateColorPlot(test_image_reshaped)
                test_image = np.expand_dims(test_images[0], axis = 0)
                
                # Make a prediction                        
                model = tf.keras.models.load_model('./cache/kaggle_model' + ext)                
                y_proba = model.predict(test_image, verbose = 0)
                pred_index = (y_proba >= 0.5).astype(int)
                self.lbl_pred.config(text = f"Predicted Value: {class_names[pred_index[0][0]]}")

                # Display the test label
                ind = test_labels[0]
                self.lbl_label.config(text = f"Test Sample Label: {class_names[ind]}")

            if dataset == "Oxford-IIIT":
                # Select a random test sample and plot it
                test_data = self.shared_data.get('val_data')
                index = random.randint(0, int(len(test_data)))
                test_image = tf.reshape(test_data[index,:], (200, 200, 3))
                self.generateColorPlot(test_image)
                                                       
                # Make a prediction
                model = tf.keras.models.load_model('./cache/oxford_model' + ext)        
                mask = model.predict(np.expand_dims(test_image, 0), verbose = 0)[0]
                mask_reshaped = np.argmax(mask, axis = -1)
                mask_reshaped *= 127
                self.generateColorPlot(mask_reshaped)
                
            if dataset == "imdb":
                print("Genearate imdb sample text")

            if dataset == "aclImdb":
                if self.shared_settings["model"] == "Transf Encoder":
                    model = tf.keras.models.load_model(
                        './cache/aclImdb_model' + ext,
                        custom_objects = {"TransformerEncoder": TransformerEncoder,
                                          "PositionalEmbedding": PositionalEmbedding})
                else:
                    model = tf.keras.models.load_model('./cache/aclImdb_model' + ext)
                #max_tokens = 10000
                #seq_length = 600
                max_tokens = self.shared_settings["max_tokens"]
                seq_length = self.shared_settings["seq_length"]

                # Using the training vocabulary
                if self.shared_settings["model"] == "bag-of-words 2g":
                    #max_tokens = 20000
                    text_vectorization = tf.keras.layers.TextVectorization(
                        ngrams = 2,
                        max_tokens = max_tokens,
                        output_mode = "multi_hot",
                        vocabulary = self.shared_data["vocab"])
                    
                if self.shared_settings["model"] == "2LSTM-wembd" or self.shared_settings["model"] == "Transf Encoder":
                    text_vectorization = tf.keras.layers.TextVectorization(
                        max_tokens = max_tokens,
                        output_mode = "int",
                        output_sequence_length = seq_length,
                        vocabulary = self.shared_data["vocab"])
                    
                # Extract a single test sample and its label                             
                for text_batch, label_batch in test_dataset.take(1):
                    text_sample = text_batch[0].numpy().decode("utf-8")    # Decoding text
                    test_label = label_batch[0].numpy()  

                    # Vectorize the text sample
                    vectorized_text = text_vectorization([text_sample]) 
                    
                    # Make a prediction
                    class_names = ['negative', 'positive']
                    y_proba = model.predict(vectorized_text, verbose = 0)[0][0]
                    pred_index = (y_proba >= 0.5).astype(int)
                    self.lbl_pred.config(text = f"Predicted Value: {class_names[pred_index]}")

                    # Display the test label
                    ind = label_batch[0]
                    self.lbl_label.config(text = f"Test Sample Label: {class_names[ind]}")

                    # Display the test review
                    textWin = tw.textWinApp(self.predictFrame, text_batch[0], self.shared_data)
                    textWin.wait_window()

    def generatePlot(self, image):
        fig = plt.figure(figsize = (0.5,0.5))
        plt.imshow(image, cmap = plt.cm.binary)
        plt.axis("off")           
        self.canvas = FigureCanvasTkAgg(fig, master = self.predictFrame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def generateColorPlot(self, image):
        fig = plt.figure(figsize = (1.0,1.0))
        plt.imshow(image)
        plt.axis("off")          
        self.canvas = FigureCanvasTkAgg(fig, master = self.predictFrame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()    
