import sys
import tkinter as tk
import tensorflow as tf

# Set the logger to display only errors or more critical messages
#tf.get_logger().setLevel('ERROR')

fg, bg = "white", "navy"

class EvaluateApp(tk.Frame):
    def __init__(self, parent, shared_data):
        super().__init__(parent)
        self.shared_data = shared_data
        
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
        
        dataset = self.shared_data.get("dataset")        
        if test_data is None:
            self.label.config(text = "Please select a dataset")
        else:
            # Load the pre-trained model
            if dataset == "MNIST":
                model = tf.keras.models.load_model('./cache/mnist_model.h5')
            if dataset == "Fashion MNIST":
                model = tf.keras.models.load_model('./cache/fashion_mnist_model.h5')
            if dataset == "Kaggle":
                model = tf.keras.models.load_model('./cache/kaggle_model.h5')
            if dataset == "Oxford-IIIT":
                model = tf.keras.models.load_model('./cache/oxford_model.h5')                
            if dataset == "imdb":
                model = tf.keras.models.load_model('./cache/imdb_model.h5')

            if test_dataset is not None:
                test_loss, test_acc = model.evaluate(test_dataset)
            else:
                test_loss, test_acc = model.evaluate(test_data, test_labels)
            
            self.label.config(text = f"Test Accuracy: {100*test_acc:.1f}")

    def quit_app(self):
        # Shutting down TensorFlow properly
        tf.keras.backend.clear_session()
        
        # Quit or destroy the root application to close the program
        self.master.destroy()
        sys.exit(0)
