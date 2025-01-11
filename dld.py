# Entry point program
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow INFO and WARNING messages
warnings.filterwarnings('ignore')         # Suppresses all warnings

import tensorflow as tf
# Set the logger to display only errors and critical messages
tf.get_logger().setLevel('ERROR')

import re
import tkinter as tk
import src.datasetsFrame as ds
import src.lossDiagFrame as ld
import src.evaluateFrame as ev
import src.predictFrame as pr

# To run with CPU only
#import tensorflow as tf
# Disable all GPUs
#tf.config.set_visible_devices([], 'GPU')

# If available, force GPU initialization by performing a dummy operation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    with tf.device('/GPU:0'):             # Replace '/GPU:0' with specific GPU device ID if needed
        dummy = tf.constant([1.0])  
        tf.print("GPU initialized successfully.")
    print()

# Setup GUI 
sw_version = os.getcwd()[-2:]
root = tk.Tk()
root.title(f"Deep Learning Dashboard v.{sw_version}")
root.minsize(1200, 800)
root.option_add("*Font", "Tahoma 11")   # Global font style and size for Windows10

# Finding the OS type 
if os.name == "nt":
    os_type = "windows"
if os.name == "posix":
    os_type = "linux"

# Finding the TF version to set the keras model extension type
tf_ver = int(tf.__version__.split('.')[1])
if int(tf.__version__.split('.')[1]) <= 14:   # *.keras format available for TF > 2.14
    ext = ".h5"
else:
    ext = ".keras"
    
# TensorFlow and CUDA versions
version = f"TF version: {tf.__version__} without CUDA"
if gpus:                                # Checking CUDA
    if os_type == "linux":
        #cuda_ver_pattern = re.compile(r'cuda-([\d.]+)')  
        #cuda_version = cuda_ver_pattern.findall(os.environ.get("LD_LIBRARY_PATH"))
        cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
        version = f"TF version: {tf.__version__} with CUDA: {cuda_version}"
    if os_type == "windows":
        cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
        version = f"TF version: {tf.__version__} with CUDA: {cuda_version[-3:]}"

shared_settings = {
    "datasets" : ["MNIST", "Fashion MNIST", "Kaggle", "Oxford-IIIT", "aclImdb", "imdb"],
    "models" : ["MLP-512", "MLP-2L-96", "MLP-128-Dropout", "CNN Small",
                "CNN Medium-DA", "VGG16-DA", "mini Xception",
                "Segmentation",
                "MLP sent. analysis",
                "bag-of-words 2g", "2LSTM-wembd", "Transf Encoder"],
    "layers" : "",
    "layers_config" : {},
    "optimizer" : "",
    "loss" : "",
    "metrics" : ["accuracy"],
    "epochs" : "",
    "batch_size" : "",
    "validation_split" : "",
    "max_tokens" : 7500,
    "seq_length" : 600,
    "status" : "idle",
    "duration" : "",
    "version" : version,
    "os" : os_type,
    "extension" : ext,
}

# Shared data and settings structures (e.g., a dictionary)
shared_data = {}

# Datasets frame
dsFrame = ds.DatasetApp(root, shared_data, shared_settings)

# Loss diagrams frame
lossFrame = ld.LossDiagApp(root, shared_data, shared_settings)

# Model evaluation frame
evalFrame = ev.EvaluateApp(root, shared_data, shared_settings)

# Model prediction frame
predFrame = pr.PredictApp(root, shared_data, shared_settings)

root.protocol("Closing...", evalFrame.quit_app)
root.mainloop()
