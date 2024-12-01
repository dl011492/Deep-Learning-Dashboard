import string
import tensorflow as tf

# Set the logger to display only errors or more critical messages
#tf.get_logger().setLevel('ERROR')

# Several model layer configurations

sent_cfg = {0: {'type': 'Dense', 'neurons':  16, 'activation': 'relu'},   # not working (not Sequential)
            1: {'type': 'Dense', 'neurons':  16, 'activation': 'relu'},
            2: {'type': 'Dense', 'neurons':   1, 'activation': 'sigmoid'}}

class Model:
    def __init__(self, model_name, shared_settings):
        # Default model. MLP with 512 neurons. DL with Python p. 28, p. 139
        if model_name == "MLP-512":
            shared_settings["layers"] = 4
            shared_settings["layers_config"] = {0: {'type': 'Input'},
                                                1: {'type': 'Flatten'},
                                                2: {'type': 'Dense', 'neurons': 512, 'activation': 'relu'},
                                                3: {'type': 'Dense', 'neurons':  10, 'activation': 'softmax'}}
            self.model = self.build_model(shared_settings.get('layers_config', {}))
            shared_settings["optimizer"] = "rmsprop"
            shared_settings["loss"] = "sparse_categorical_crossentropy"
            shared_settings["epochs"] = 5
            shared_settings["batch_size"] = 128
            shared_settings["validation_split"] = 0.2

        # MLP with increased model capacity. 2 Layers 96 neurons. DL with Python p. 141
        elif model_name == "MLP-2L-96":
            shared_settings["layers"] = 5
            shared_settings["layers_config"] = {0: {'type': 'Input'},
                                                1: {'type': 'Flatten'},
                                                2: {'type': 'Dense', 'neurons': 96, 'activation': 'relu'},
                                                3: {'type': 'Dense', 'neurons': 96, 'activation': 'relu'},
                                                4: {'type': 'Dense', 'neurons':  10, 'activation': 'softmax'}}
            self.model = self.build_model(shared_settings.get('layers_config', {}))
            shared_settings["optimizer"] = "rmsprop"
            shared_settings["loss"] = "sparse_categorical_crossentropy"
            shared_settings["epochs"] = 20
            shared_settings["batch_size"] = 128
            shared_settings["validation_split"] = 0.2
            
        # Small Multilayer Perceptron with droput for MNIST or Fashion MNIST
        elif model_name == "MLP-128-Dropout":
            shared_settings["layers"] = 5
            shared_settings["layers_config"] = {0: {'type': 'Input'},
                                                1: {'type': 'Flatten'},
                                                2: {'type': 'Dense', 'neurons': 128, 'activation': 'relu'},
                                                3: {'type': 'Dropout', 'dropout': 0.2},
                                                4: {'type': 'Dense', 'neurons':  10, 'activation': 'softmax'}}
            self.model = self.build_model(shared_settings.get('layers_config', {}))            
            shared_settings["optimizer"] = "adam"
            shared_settings["loss"] = "sparse_categorical_crossentropy"
            shared_settings["epochs"] = 20
            shared_settings["batch_size"] = 128
            shared_settings["validation_split"] = 0.4

        # Small Convolution Network for MNIST and Fashion MNIST. DL with Python p. 202
        elif model_name == "CNN Small":
            inputs = tf.keras.layers.Input(shape = (28, 28, 1))
            x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,
                                       activation = "relu")(inputs)
            x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
            x = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3,
                                       activation = "relu")(x)
            x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
            x = tf.keras.layers.Conv2D(filters = 128, kernel_size = 3,
                                       activation = "relu")(x)
            x = tf.keras.layers.Flatten()(x)
            outputs = tf.keras.layers.Dense(10, activation = "softmax")(x)
            self.model = tf.keras.Model(inputs = inputs, outputs = outputs)

            shared_settings["optimizer"] = "rmsprop"
            shared_settings["loss"] = "sparse_categorical_crossentropy"
            shared_settings["epochs"] = 20
            shared_settings["batch_size"] = 64
            shared_settings["validation_split"] = 0.2

        # Medium Convolution Network with data augmentation for Kaggle. DL with Python p. 223
        elif model_name == "CNN Medium-DA":
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.2),])            
            inputs = tf.keras.layers.Input(shape = (180, 180, 3))
            x = data_augmentation(inputs)
            x = tf.keras.layers.Rescaling(1./255)(x)
            x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,
                                       activation = "relu")(x)
            x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
            x = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3,
                                       activation = "relu")(x)
            x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
            x = tf.keras.layers.Conv2D(filters = 128, kernel_size = 3,
                                       activation = "relu")(x)
            x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
            x = tf.keras.layers.Conv2D(filters = 256, kernel_size = 3,
                                       activation = "relu")(x)
            x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
            x = tf.keras.layers.Conv2D(filters = 256, kernel_size = 3,
                                       activation = "relu")(x)            
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(1, activation = "sigmoid")(x)
            self.model = tf.keras.Model(inputs = inputs, outputs = outputs)

            shared_settings["optimizer"] = "rmsprop"
            shared_settings["loss"] = "binary_crossentropy"
            shared_settings["epochs"] = 100
            shared_settings["batch_size"] = 32

        # VGG16 convolutional base with data augmentation. DL with Ptyhon p. 232
        elif model_name == "VGG16-DA":
            conv_base = tf.keras.applications.vgg16.VGG16(
                weights = "imagenet",
                include_top = False,)
            conv_base.trainable = False
            conv_base.summary()
            
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.2),])
            
            inputs = tf.keras.Input(shape = (180, 180, 3))
            x = data_augmentation(inputs)
            x = tf.keras.applications.vgg16.preprocess_input(x)
            x = conv_base(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256)(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(1, activation = "sigmoid")(x)
            self.model = tf.keras.Model(inputs, outputs)

            shared_settings["optimizer"] = "rmsprop"
            shared_settings["loss"] = "binary_crossentropy"
            shared_settings["epochs"] = 50
            shared_settings["batch_size"] = 32     # not needed. Just to disply it on the main window

        # mini Xception like model. DL with Ptyhon p. 259
        elif model_name == "mini Xception":
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.2),])
            inputs = tf.keras.layers.Input(shape = (180, 180, 3))
            x = data_augmentation(inputs)
            
            x = tf.keras.layers.Rescaling(1./255)(x)
            x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 5,
                                       use_bias = False)(x)
            
            for size in [32, 64, 128, 256, 512]:
                residual = x

                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation("relu")(x)
                x = tf.keras.layers.SeparableConv2D(size, 3, padding = "same",
                                                    use_bias = False)(x)

                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation("relu")(x)
                x = tf.keras.layers.SeparableConv2D(size, 3, padding = "same",
                                                    use_bias = False)(x)

                x = tf.keras.layers.MaxPooling2D(3, strides = 2, padding = "same")(x)

                residual = tf.keras.layers.Conv2D(size, 1, strides = 2, padding = "same",
                                                  use_bias = False)(residual)
                x = tf.keras.layers.add([x, residual])
                
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(1, activation = "sigmoid")(x)
            self.model = tf.keras.Model(inputs, outputs)

            shared_settings["optimizer"] = "rmsprop"
            shared_settings["loss"] = "binary_crossentropy"
            shared_settings["epochs"] = 100
            shared_settings["batch_size"] = 32

        # Segmentation. DL with Ptyhon p. 244
        elif model_name == "Segmentation":
            inputs = tf.keras.Input(shape = (200, 200) + (3,))
            #x = tf.keras.layers.Rescaling(1./255)(inputs)
            x = tf.keras.layers.Conv2D(64, 3, strides = 2, activation = "relu",
                                       padding = "same")(inputs)
            x = tf.keras.layers.Conv2D(64, 3, activation = "relu", padding = "same")(x)
            x = tf.keras.layers.Conv2D(128, 3, strides=2, activation = "relu",
                                       padding = "same")(x)
            x = tf.keras.layers.Conv2D(128, 3, activation = "relu", padding = "same")(x)
            x = tf.keras.layers.Conv2D(256, 3, strides = 2, activation = "relu",
                                       padding = "same",)(x)
            x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
            x = tf.keras.layers.Conv2DTranspose(256, 3, activation = "relu", padding = "same")(x)
            x = tf.keras.layers.Conv2DTranspose(256, 3, activation = "relu",
                                                padding = "same", strides = 2)(x)
            x = tf.keras.layers.Conv2DTranspose(128, 3, activation = "relu", padding = "same")(x)
            x = tf.keras.layers.Conv2DTranspose(128, 3, activation = "relu",
                                                padding = "same", strides = 2)(x)
            x = tf.keras.layers.Conv2DTranspose(64, 3, activation = "relu", padding = "same")(x)
            x = tf.keras.layers.Conv2DTranspose(64, 3, activation = "relu",
                                                padding = "same", strides = 2)(x)
            outputs = tf.keras.layers.Conv2D(3, 3, activation = "softmax", padding="same")(x)
            self.model = tf.keras.Model(inputs, outputs)

            shared_settings["optimizer"] = "rmsprop"
            shared_settings["loss"] = "sparse_categorical_crossentropy"
            shared_settings["epochs"] = 50
            shared_settings["batch_size"] = 64

        # Small Multilayer Perceptron for imdb sentiment analysis. DL with Python p. 99
        elif model_name == "MLP sent. analysis":
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(16, activation = "relu"),
                tf.keras.layers.Dense(16, activation = "relu"),
                tf.keras.layers.Dense( 1, activation = "sigmoid")
            ])
            shared_settings["optimizer"] = "rmsprop"
            shared_settings["loss"] = "binary_crossentropy"
            shared_settings["epochs"] = 20
            shared_settings["batch_size"] = 512
        
        # Compile the model
        self.model.compile(
            optimizer = shared_settings["optimizer"],
            loss = shared_settings["loss"],
            metrics = shared_settings["metrics"],
        )

    def build_model(self, layers_config):  
        # Initialize the model
        model = tf.keras.Sequential()

        # Add hidden layers with varying types
        for layer, layer_info in layers_config.items():
            layer_type = layer_info.get('type')
            neurons = layer_info.get('neurons')
            activation = str(layer_info.get('activation')).lower()
            dropout = layer_info.get('dropout')
            if layer_type == "Input":
                model.add(tf.keras.layers.Input(shape = (28, 28)))
            if layer_type == "Flatten":
                model.add(tf.keras.layers.Flatten())                      
            if layer_type == "Dense":
                model.add(tf.keras.layers.Dense(neurons, activation = activation))
            if layer_type == "Dropout":
                model.add(tf.keras.layers.Dropout(dropout))               
        return model

    def build_model2(self, layers_config):
        # This is NOT a sequential model
        for layer, layer_info in layers_config.items():
            layer_type = layer_info.get('type')
            if layer_type == "Input":
                inputs = tf.keras.layers.Input(shape = (28, 28, 1))
            if layer_type == "Conv2D":
                x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,
                                           activation = "relu")(inputs)
            if layer_type == "MaxPooling":
                x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
            if layer_type == "Flatten":
                x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10, activation = "softmax")(x)
        self.model = tf.keras.Model(inputs = inputs, outputs = outputs)
        return model
        
    def get_model(self):
        return self.model
