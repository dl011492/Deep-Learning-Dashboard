import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import tensorflow as tf
import src.dataset as ds
import src.lossDiagFrame as ld
import src.textWinFrame as tw

fg, bg = "white", "navy" 

class DatasetApp(tk.Frame):
    def __init__(self, parent, shared_data, shared_settings):
        super().__init__(parent)
        self.shared_data = shared_data
        self.shared_settings = shared_settings
        self.parent = parent
        
        self.dsFrame = tk.Frame(parent, bg = bg, width = 250, height = 800, name = "dsFrame")
        self.dsFrame.pack_propagate(False)                            # Prevent the frame from shrinking below the set size
        self.dsFrame.grid(row = 0, column = 0, rowspan = 2, sticky = "nsew")
        
        # Frame title
        self.lbl = tk.Label(self.dsFrame, text = "Data Sets", fg = fg, bg = bg)
        self.lbl.pack(anchor = "n", pady = 10)                        # Label centered horizontally

        # ListBox for the available datasets
        self.listbox = tk.Listbox(self.dsFrame, fg = fg, bg = bg, height = 5)
        datasets = self.shared_settings["datasets"]
        for dataset in datasets:
            self.listbox.insert(tk.END, dataset)
        self.listbox.pack(pady = 5)
        self.listbox.bind('<<ListboxSelect>>', self.on_DS_select)     # Bind the selected event
       
        # Label to display the selected dataset
        self.label = tk.Label(self.dsFrame, text = "Select an option from the list", fg = fg, bg = bg)
        self.label.pack(pady = (0, 20))

        # Label to display number of train data samples
        self.lbl_t_samples = tk.Label(self.dsFrame, text = "Training samples:", fg = fg, bg = bg)
        self.lbl_t_samples.pack(pady = (0, 10))

        # Label to display train data dtype
        self.lbl_t_dtype = tk.Label(self.dsFrame, text = "Training dtype:", fg = fg, bg = bg)
        self.lbl_t_dtype.pack(pady = (0, 10))

        # Label to display number of test data samples
        self.lbl_tst_samples = tk.Label(self.dsFrame, text = "Test samples:", fg = fg, bg = bg)
        self.lbl_tst_samples.pack(pady = (0, 40))
        
        # Button to run visualization
        self.button = tk.Button(self.dsFrame, width = 20, text = "Visualize Train Sample",
                                fg = fg, bg = bg,
                                command = self.on_visualize_click )
        self.button.pack(pady = (10, 20))

        self.selected_item = None                           # Variable to store the selected dataset
        
    def on_DS_select(self, event):        
        widget = event.widget                               # Get the selected item from the listbox
        selection_index = widget.curselection()
        if selection_index:
            self.selected_item = widget.get(selection_index)
            self.label.config(text = f"Selected: {self.selected_item}")
            
            # Load the dataset using the Dataset class
            ds_obj = ds.Dataset(self.selected_item, self.shared_settings)            
            self.shared_data['dataset'] = self.selected_item

            # Resetting past data calls. Solves Issue #3
            self.shared_data['train_dataset'] = None

            # Setting the shared_data for each dataset
            if self.selected_item == "MNIST" or self.selected_item == "Fashion MNIST" \
               or self.selected_item == "Kaggle" or self.selected_item == "Oxford-IIIT" \
               or self.selected_item == "imdb":
                self.shared_data['train_data'], self.shared_data['train_labels'] = ds_obj.get_train_data()
                self.shared_data['val_data'], self.shared_data['val_labels'] = ds_obj.get_val_data()            
                self.shared_data['test_data'], self.shared_data['test_labels'] = ds_obj.get_test_data()            
                self.shared_data['train_shape'], self.shared_data['train_dtype'] = ds_obj.get_train_info()
                self.shared_data['test_shape'], self.shared_data['test_dtype'] = ds_obj.get_test_info()

            if self.selected_item == "Kaggle" or self.selected_item == "aclImdb":
                self.shared_data['train_dataset'] = ds_obj.get_train_dataset()
                self.shared_data['val_dataset'] = ds_obj.get_val_dataset()
                self.shared_data['test_dataset'] = ds_obj.get_test_dataset()
                
            # Setting the models for each dataset
            if self.selected_item == "MNIST" or self.selected_item == "Fashion MNIST":
                self.shared_settings["models"] = ["MLP-512", "MLP-2L-96", "MLP-128-Dropout", "CNN Small"]
               
            if self.selected_item == "Kaggle":
                self.shared_settings["models"] = ["CNN Medium-DA", "VGG16-DA", "mini Xception"]

            if self.selected_item == "Oxford-IIIT":
                self.shared_settings["models"] = ["Segmentation"]
                
            if self.selected_item == "imdb":
                self.shared_settings["models"] = ["MLP sent. analysis"]
                
            if self.selected_item == "aclImdb":
                self.shared_settings["models"] = ["bag-of-words 2g", "2LSTM-wembd", "Transf Encoder"]                 
                self.shared_data['train_shape'], self.shared_data['train_dtype'] = ds_obj.get_train_dataset_info()
                self.shared_data['test_shape'], self.shared_data['test_dtype'] = ds_obj.get_test_dataset_info()
                
            self.lbl_t_samples.config(text = f"Training samples: {self.shared_data['train_shape'][0]}")
            self.lbl_t_dtype.config(text = f"Train dtype: {self.shared_data['train_dtype']}")
            self.lbl_tst_samples.config(text = f"Test samples: {self.shared_data['test_shape'][0]}")

            #print(self.shared_data['train_dataset'])
            
            # Instantiate the lossDiag framne to update the models settings
            lossFrame = ld.LossDiagApp(self.parent, self.shared_data, self.shared_settings)

        else:
            self.selected_item = None

    def on_visualize_click(self):        
        # Access the train data from the shared data
        train_data = self.shared_data.get('train_data')
        train_labels = self.shared_data.get('train_labels')

        # Access the train dataset from the shared data
        train_dataset = self.shared_data.get('train_dataset')
        
        if self.selected_item is None:
            self.label.config(text = "Please select a dataset")
        else:
            max_train = 32
            if train_dataset is None:             
                max_train = int(len(train_data))
            index = random.randint(0, max_train)
            
            if self.selected_item == "MNIST" or self.selected_item == "Fashion MNIST":  
                train_image = tf.reshape(train_data[index,:], (28, 28))
                self.generatePlot(train_image)

            elif self.selected_item == "Kaggle":
                train_image = tf.reshape(train_data[index,:], (180, 180, 3))
                self.generateColorPlot(train_image)

            elif self.selected_item == "Oxford-IIIT":  
                train_image = tf.reshape(train_data[index,:], (200, 200, 3))
                self.generateColorPlot(train_image)

            elif self.selected_item == "imdb":
                train_text = train_data[index]
                textWin = tw.textWinApp(self.dsFrame, train_text, self.shared_data)
                textWin.wait_window()

            elif self.selected_item == "aclImdb":
                sample_text = None
                textWin = tw.textWinApp(self.dsFrame, sample_text, self.shared_data)
                textWin.wait_window()
                
    def generatePlot(self, image):
        fig = plt.figure(figsize = (0.5,0.5))
        plt.imshow(image, cmap = plt.cm.binary)
        plt.axis("off")           
        self.canvas = FigureCanvasTkAgg(fig, master = self.dsFrame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def generateColorPlot(self, image):
        fig = plt.figure(figsize = (1.0,1.0))
        plt.imshow(image)
        plt.axis("off")          
        self.canvas = FigureCanvasTkAgg(fig, master = self.dsFrame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()        
