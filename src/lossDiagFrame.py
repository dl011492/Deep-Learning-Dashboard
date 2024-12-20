import os
import time
import tkinter as tk
import src.trainSettingsFrame as ts
import src.modelParamsFrame as mp
import src.models as md
from tkinter import ttk
from src.plotCallback import PlotCallback
import tensorflow as tf

# Set the logger to display only errors or more critical messages
#tf.get_logger().setLevel('ERROR')

fg, bg = "black", "lightgrey"

sel_model = ""

class LossDiagApp(ttk.Frame):
    def __init__(self, parent, shared_data, shared_settings):
        super().__init__(parent)
        self.shared_data = shared_data
        self.shared_settings = shared_settings

        if self.shared_settings["epochs"] != "":            
            max_epochs = self.shared_settings["epochs"]
        else:
            max_epochs = 20 
        max_loss = 1.0

        # Frame creation and title
        self.lossDiagFrame = tk.Frame(parent, bg = bg, width = 700, height = 800)
        self.lossDiagFrame.pack_propagate(False)
        self.lossDiagFrame.grid(row = 0, column = 1, rowspan = 2, sticky = "nsew")
        self.lbl = tk.Label(self.lossDiagFrame, text = "Accuracy and Loss Diagrams", fg = fg, bg = bg)
        self.lbl.pack(anchor = "n", pady = 10)              # label centered horizontally

        # Create a frame to hold the callback plot
        self.plotFrame = tk.Frame(self.lossDiagFrame)
        self.plotFrame.pack(anchor = "n", pady = (0, 10))
        
        # Initialize the PlotCallback within plotFrame and epoch count
        self.plot_callback = PlotCallback(self.plotFrame,
                                          epochs = max_epochs,
                                          max_loss = max_loss)

        # Menus and buttons
        btn_width = 16
        if os.name == "nt":
            btn_width = 20
        # Menu for models        
        placeholder = "Models"            
        self.sel_model = tk.StringVar(value = placeholder)
        options = self.shared_settings["models"]
        
        def on_selected_model(*args):
            selected = self.sel_model.get()
            print(f"Selected model: {selected}")
            self.model = md.Model(selected, self.shared_settings).get_model()
            if selected != "MLP sent. analysis":
                self.model.summary()
            shared_settings["duration"] = 0
            self.update_settings()
            self.plot_callback = PlotCallback(self.plotFrame,
                                              epochs = self.shared_settings["epochs"],
                                              max_loss = 1.0)
                   
        self.models_menu = tk.OptionMenu(self.lossDiagFrame, self.sel_model, *options)
        self.models_menu.config(width = btn_width - 1)
        if os.name == "nt":
            self.models_menu.config(width = 16)
        self.models_menu.place(x = 500, y = 550)
        self.sel_model.trace("w", on_selected_model)

        # Button for model parameters
        self.button_mp = tk.Button(self.lossDiagFrame, width = btn_width,
                                   text = "Model Parameters", command = self.on_mp_click)
        self.button_mp.place(x = 500, y = 605)
        
        # Button for training settings parameters
        self.button_ts = tk.Button(self.lossDiagFrame, width = btn_width,
                                   text = "Training Settings", command = self.on_ts_click)
        self.button_ts.place(x = 500, y = 660)
        
        # Button to start training
        self.button = tk.Button(self.lossDiagFrame, width = btn_width,
                                text = "Train", command = self.on_train_click)                               
        self.button.place(x = 500, y = 715)
        
        # Settings labels
        self.set_labels()

        # Update Settings
        self.update_settings()


    def set_labels(self):
        py = 3
        self.lbl_ep = tk.Label(self.lossDiagFrame, text = "Epochs:", fg = fg, bg = bg)
        self.lbl_ep.pack(side = "top", anchor = "w", padx = 20, pady = (10, 1))
        
        self.lbl_bs = tk.Label(self.lossDiagFrame, text = "Batch size:", fg = fg, bg = bg)
        self.lbl_bs.pack(side = "top", anchor = "w", padx = 20, pady = py)

        self.lbl_vs = tk.Label(self.lossDiagFrame, text = "Validation split:", fg = fg, bg = bg)
        self.lbl_vs.pack(side = "top", anchor = "w", padx = 20, pady = py)

        self.lbl_opt = tk.Label(self.lossDiagFrame, text = "Optimizer:", fg = fg, bg = bg)
        self.lbl_opt.pack(side = "top", anchor = "w", padx = 20, pady = py)

        self.lbl_loss = tk.Label(self.lossDiagFrame, text = "Loss function:", fg = fg, bg = bg)
        self.lbl_loss.pack(side = "top", anchor = "w", padx = 20, pady = py)

        self.lbl_sta = tk.Label(self.lossDiagFrame, text = "Status:", fg = fg, bg = bg)
        self.lbl_sta.pack(side = "top", anchor = "w", padx = 20, pady = py)
        
        self.lbl_dur = tk.Label(self.lossDiagFrame, text = "Training duration: ", fg = fg, bg = bg)
        self.lbl_dur.pack(side = "top", anchor = "w", padx = 20, pady = py)

        self.lbl_ver = tk.Label(self.lossDiagFrame, text = "Version: ", font = ("Tahoma", 8), fg = fg, bg = bg)
        self.lbl_ver.pack(side = "top", anchor = "w", padx = 21, pady = (15, 5))
        self.lbl_ver.config(text = f"{self.shared_settings['version']}")
        
    def update_settings(self):
        self.lbl_ep.config(text = f"Epochs: {self.shared_settings['epochs']}")
        self.lbl_bs.config(text = f"Batch size: {self.shared_settings['batch_size']}")
        self.lbl_vs.config(text = f"Validation split: {self.shared_settings['validation_split']}")
        self.lbl_opt.config(text = f"Optimizer: {self.shared_settings['optimizer']}")
        self.lbl_loss.config(text = f"Loss function: {self.shared_settings['loss']}")     
        self.lbl_sta.config(text = f"Status: {self.shared_settings['status']}")
        if self.shared_settings['duration'] != "":
            self.lbl_dur.config(text = f"Training duration: {self.shared_settings['duration']} s")
        
    def on_mp_click(self):
        modParams = mp.ModelParamsFrame(self.lossDiagFrame, self.shared_settings)
        modParams.wait_window()
        self.update_settings()
        
    def on_ts_click(self):
        trainSettings = ts.trainSettingsFrame(self.lossDiagFrame, self.shared_settings)
        trainSettings.wait_window()
        self.update_settings()
        self.plot_callback = PlotCallback(self.plotFrame,
                                          epochs = self.shared_settings['epochs'],
                                          max_loss = 1.0)
 
    def on_train_click(self):
        dataset = self.shared_data.get("dataset")
        # Clear the plot before starting training  (I need this!)
        self.plot_callback.clear_plot()
                
        if dataset is None:
            self.lbl_ep.config(text = "Please select a dataset")
            return
                    
        # Access the train and validation data from the shared data
        train_data = self.shared_data.get("train_data")
        train_labels = self.shared_data.get("train_labels")
        val_data = self.shared_data.get("val_data")
        val_labels = self.shared_data.get("val_labels")

        # Access the train and validation datasets from the shared data
        train_dataset = self.shared_data.get("train_dataset")
        val_dataset = self.shared_data.get("val_dataset")    

        # Update the status in shared_settings                                  
        self.shared_settings['status'] = "...running....."
        self.update_settings()

        # Define callbacks
        if dataset == "MNIST":
            dataset_name = "mnist"
        elif dataset == "Fashion MNIST":
            dataset_name = "fashion_mnist"
        elif dataset == "imdb":
            dataset_name = "imdb"
        elif dataset == "Kaggle":
            dataset_name = "kaggle"
        elif dataset == "Oxford-IIIT":
            dataset_name = "oxford" 
        else:
            dataset_name = "unknown"
        filepath = f"./cache/{dataset_name}_model.h5"
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                    filepath = filepath,  
                                    monitor = "val_loss",
                                    save_best_only = True )
        callbacks = [self.plot_callback, checkpoint_callback]
            
        # Set up validation data or validation split
        validation_args = {}
        if val_data is not None and val_labels is not None:
            validation_args["validation_data"] = (val_data, val_labels)    
        else:
            validation_args["validation_split"] = self.shared_settings["validation_split"]
            
        # Train the model
        start_time = time.time()
        if train_dataset is not None:
            self.model.fit(train_dataset,
                           epochs = self.shared_settings["epochs"],
                           batch_size = self.shared_settings["batch_size"],
                           callbacks = callbacks,
                           validation_data = val_dataset,)
        else:
            self.model.fit(train_data, train_labels,
                           epochs = self.shared_settings["epochs"],
                           batch_size = self.shared_settings["batch_size"],
                           callbacks = callbacks,
                           **validation_args)                     
        end_time = time.time()
        train_time = f"{(end_time - start_time):.1f}"

        self.shared_settings['status'] = "finished"
        self.shared_settings['duration'] = train_time
        self.update_settings()
        
        # Reset some status labels
        self.shared_settings['status'] = "idle"
        self.shared_settings['duration'] = ""
             
