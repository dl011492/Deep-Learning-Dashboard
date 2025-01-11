Dec 4, 2024.

I keep the best model results in this folder. 

For MNIST is CNN Small.
For Fashion MNIST is also CNN Small.
For Kaggle is VGG16-DA.
For Oxford-IIIT is Segmentation (the only one implemented).
For imdb is MLP sent. analysis (the only one implemented)..
For aclImdb is bag-of-words 2g

I usually rename these files so that I know which model I used to create them.

One can use these files to make model predictions (with sample visualizations) and measure the test accuracy
of the model without having to train the model again.
The only exception are the models for aclImdb dataset, where one needs to build the vocabulary during training
in order to make predictions and visualizations.

The file extension for these files depend on your TensorFlow version. From TensorFlow 2.12 on,
the new model saving format is *.keras. So the script will use this extension. Since the files in this folder were
created under Windows with Tf-2.10, they have the *.h5 extension.


