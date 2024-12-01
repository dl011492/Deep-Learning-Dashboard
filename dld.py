# Entry point program
import os
import tkinter as tk
import tensorflow as tf
import src.datasetsFrame as ds
import src.lossDiagFrame as ld
import src.evaluateFrame as ev
import src.predictFrame as pr

# To run with CPU only
#import tensorflow as tf
# Disable all GPUs
#tf.config.set_visible_devices([], 'GPU')

# Set the logger to display only errors and critical messages
tf.get_logger().setLevel('ERROR')

# Setup GUI 
sw_version = os.getcwd()[-2:]
root = tk.Tk()
root.title(f"Deep Learning Dashboard v. {sw_version}")
root.minsize(1200, 800)
root.option_add("*Font", "Tahoma 11")   # Global font style and size for Windows10

# TensorFlow and CUDA versions
tf_version = f"TF version: {tf.__version__} without CUDA"
if tf.test.is_built_with_cuda():        # Checking CUDA
    cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
    tf_version = f"TF version: {tf.__version__} with CUDA: {cuda_version[-3:]}"

# Shared data and settings structures (e.g., a dictionary)
shared_data = {}

shared_settings = {
    "models" : ["MLP-512", "MLP-2L-96", "MLP-128-Dropout", "CNN Small",
                "CNN Medium-DA", "VGG16-DA", "mini Xception",
                "Segmentation",
                "MLP sent. analysis"],
    "layers" : "",
    "layers_config" : {},
    "optimizer" : "",
    "loss" : "",
    "metrics" : ["accuracy"],
    "epochs" : "",
    "batch_size" : "",
    "validation_split" : "",
    "status" : "idle",
    "duration" : "",
    "version" : tf_version
}

# Datasets frame
dsFrame = ds.DatasetApp(root, shared_data, shared_settings)

# Loss diagrams frame
lossFrame = ld.LossDiagApp(root, shared_data, shared_settings)

# Model evaluation frame
evalFrame = ev.EvaluateApp(root, shared_data)

# Model prediction frame
predFrame = pr.PredictApp(root, shared_data)

root.protocol("Closing...", evalFrame.quit_app)
root.mainloop()
