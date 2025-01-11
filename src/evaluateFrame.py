import sys
import tkinter as tk
import tensorflow as tf
from src.models import TransformerEncoder, PositionalEmbedding

fg, bg = "white", "navy"

class EvaluateApp(tk.Frame):
    def __init__(self, parent, shared_data, shared_settings):
        super().__init__(parent)
        self.shared_data = shared_data
        self.shared_settings = shared_settings
        
        self.evalFrame = tk.Frame(parent, bg = bg, width = 250, height = 300)
        self.evalFrame.pack_propagate(False)
        self.evalFrame.grid(row = 0, column = 2, rowspan = 2, sticky = "nsew")

        # Frame title
        self.lbl = tk.Label(self.evalFrame, text = "Model Evaluation", fg = fg, bg = bg)
        self.lbl.pack(anchor = "n", pady = 10)       

        # Button to evaluate the model
        self.eval_button = tk.Button(self.evalFrame, text = "Evaluate Model", fg = fg, bg = bg,
                                     command = self.eval)
        self.eval_button.pack(pady = (15,5))

        # Label to display the evaluated accuracy
        self.label = tk.Label(self.evalFrame, text = "Test Accuracy:", fg = fg, bg = bg)
        self.label.pack(pady = 40)

        # Quit button to stop the application
        quit_button = tk.Button(self.evalFrame, text= "Quit", fg = fg, bg = bg,
                                command = self.quit_app)
        quit_button.pack(pady = 10)

    def eval(self):
        # Access the test data and labels from the shared data
        test_data = self.shared_data.get("test_data")
        test_labels = self.shared_data.get("test_labels")
        
        # Access the test datasets from the shared data
        test_dataset = self.shared_data.get("test_dataset")
        vec_test_dataset = self.shared_data.get("vec_test_dataset")
        
        dataset = self.shared_data.get("dataset")        
        if test_data is None and test_dataset is None:
            self.label.config(text = "Please select a dataset")
        else:
            # Load the pre-trained model
            ext = self.shared_settings['extension']
            if dataset == "MNIST":
                model = tf.keras.models.load_model('./cache/mnist_model' + ext)
                test_loss, test_acc = model.evaluate(test_data, test_labels)                          
            elif dataset == "Fashion MNIST":
                model = tf.keras.models.load_model('./cache/fashion_mnist_model' + ext)
                test_loss, test_acc = model.evaluate(test_data, test_labels)                       
            elif dataset == "Kaggle":
                model = tf.keras.models.load_model('./cache/kaggle_model' + ext)
                test_loss, test_acc = model.evaluate(test_dataset)
            elif dataset == "Oxford-IIIT":
                model = tf.keras.models.load_model('./cache/oxford_model' + ext)
                test_loss, test_acc = model.evaluate(test_data, test_labels)           
            elif dataset == "imdb":
                test_henc_data = self.shared_data['test_henc_data']
                test_henc_labels = self.shared_data['test_henc_labels']                
                model = tf.keras.models.load_model('./cache/imdb_model' + ext)
                test_loss, test_acc = model.evaluate(test_henc_data, test_henc_labels)           
            elif dataset == "aclImdb":
                if vec_test_dataset is None:
                    self.label.config(text = "Please train a model")                
                else:
                    if self.shared_settings["model"] == "Transf Encoder":
                        model = tf.keras.models.load_model(
                            './cache/aclImdb_model' + ext,
                            custom_objects = {"TransformerEncoder": TransformerEncoder,
                                              "PositionalEmbedding": PositionalEmbedding})
                    else:
                        model = tf.keras.models.load_model('./cache/aclImdb_model' + ext)
                    test_loss, test_acc = model.evaluate(vec_test_dataset)
                    
            self.label.config(text = f"Test Accuracy: {100*test_acc:.1f}")
            
    def quit_app(self):
        # Shutting down TensorFlow properly
        tf.keras.backend.clear_session()
        
        # Quit or destroy the root application to close the program
        self.master.destroy()
        sys.exit(0)
