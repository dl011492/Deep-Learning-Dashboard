import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import tensorflow as tf

class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, master_frame, epochs, max_loss):
        super(PlotCallback, self).__init__()
            
        # Store the Tkinter frame and number of epochs
        self.master_frame = master_frame
        self.epochs = epochs
        self.max_loss = max_loss

        # Initialize data containers
        self.train_acc = []
        self.train_losses = []
        self.epoch_train_acc = []
        self.epoch_train_losses = []
        self.val_acc = []
        self.val_losses = []

        # Set up the figure and axis for the plot
        self.fig, self.ax = plt.subplots()
        self.train_acc_line, = self.ax.plot([], [], label = 'Training Accuracy', color = 'm')
        self.train_line, = self.ax.plot([], [], label = 'Training Loss', color = 'red')
        self.val_acc_line, = self.ax.plot([], [], label = 'Validation Accuracy', color = '#006400')
        self.val_line, = self.ax.plot([], [], label = 'Validation Loss', color = 'blue')
        
        # Configure plot limits and labels
        self.ax.set_xlim(0, self.epochs)
        self.ax.set_ylim(0, self.max_loss)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Accuracy, Loss')
        self.ax.legend(loc = "center right")
        # Major and minor tics
        self.ax.xaxis.set_major_locator(MultipleLocator(epochs/4))
        self.ax.xaxis.set_minor_locator(MultipleLocator(epochs/4/5))

        self.ax.yaxis.set_major_locator(MultipleLocator(0.2))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.05))

        # Integrate with Tkinter via FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.master_frame)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row = 0, column = 0, padx = (0,0), pady = (0, 0)) 
        
        # Background to restore for blitting
        self.background = None

    def on_train_batch_end(self, batch, logs = None):
        # Capture the batch-level training acuracy
        train_acc = logs.get('accuracy')
        if train_acc is not None:
            self.train_acc.append(train_acc)
            
        # Capture the batch-level training loss
        train_loss = logs.get('loss')
        if train_loss is not None:
            self.train_losses.append(train_loss)

    def on_epoch_end(self, epoch, logs = None):
        # Calculate and store the average training accuracy for the epoch
        if self.train_acc:
            epoch_train_acc = sum(self.train_acc) / len(self.train_acc)
            self.epoch_train_acc.append(epoch_train_acc)
            self.train_acc.clear()

            # Update training accuracy line data
            train_acc_xs = list(range(1, len(self.epoch_train_acc) + 1))
            self.train_acc_line.set_data(train_acc_xs, self.epoch_train_acc) 

        # Calculate and store the average training loss for the epoch
        if self.train_losses:
            epoch_train_loss = sum(self.train_losses) / len(self.train_losses)
            self.epoch_train_losses.append(epoch_train_loss)
            self.train_losses.clear()

            # Update training loss line data
            train_xs = list(range(1, len(self.epoch_train_losses) + 1))
            self.train_line.set_data(train_xs, self.epoch_train_losses)         

        # Capture and update validation accuracy line data
        val_acc = logs.get('val_accuracy')        
        if val_acc is not None:
            self.val_acc.append(val_acc)
            val_acc_xs = list(range(1, len(self.val_acc) + 1))        
            self.val_acc_line.set_data(val_acc_xs, self.val_acc)
           
        # Capture and update validation loss line data
        val_loss = logs.get('val_loss')        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            val_xs = list(range(1, len(self.val_losses) + 1))        
            self.val_line.set_data(val_xs, self.val_losses)

        # Draw and blit for performance
        self.ax.draw_artist(self.train_acc_line)
        self.ax.draw_artist(self.train_line)
        self.ax.draw_artist(self.val_acc_line)
        self.ax.draw_artist(self.val_line)
        self.fig.canvas.blit(self.ax.bbox)
        self.canvas.flush_events()

    def clear_plot(self):
        """Clear the loss plot and reset the losses."""
        self.epoch_train_acc.clear()                        # Clear the list of losses
        self.epoch_train_losses.clear()
        self.train_losses.clear()
        self.val_acc.clear()
        self.val_losses.clear()

        self.train_acc_line.remove()
        self.train_line.remove()
        self.val_acc_line.remove()
        self.val_line.remove()
        self.train_acc_line, = self.ax.plot([], [], label = 'Training Accuracy', color = 'm')
        self.train_line, = self.ax.plot([], [], label = 'Training Loss', color = 'red')
        self.val_acc_line, = self.ax.plot([], [], label = 'Validation Accuracy', color = '#006400')
        self.val_line, = self.ax.plot([], [], label = 'Validation Loss', color = 'blue')
        
        self.background = None
        self.canvas.draw() 
