import tkinter as tk

class trainSettingsFrame(tk.Toplevel):    
    def __init__(self, parent, shared_settings):
        super().__init__(parent)
        self.parent = parent                      # Store a reference to the parent window
        self.shared_settings = shared_settings
        self.layer_widgets = {}                   # Dictionary to store widgets for each layer

        self.title("Training Settings")        
        self.optimizer_controls()
    
    def optimizer_controls(self):
        height = 5
        # Optimizer
        optimizers = ["Adam", "SGD", "RMSprop"]                       # optimizer options
        optimizer_map = {opt: opt.lower() for opt in optimizers}      # Mapping dictionary
        def on_opt_select(event):
            selected_display = self.optimizer_listbox.get(            # Get the selected item
                self.optimizer_listbox.curselection())
            selected_optimizer = optimizer_map[selected_display]      # Retrieve lowercase
            self.shared_settings["optimizer"] = selected_optimizer    # Update shared_settings

        opt_ind = list(filter(lambda i: i[1].lower() == self.shared_settings["optimizer"],   # Getting the default
                              enumerate(optimizers)))[0][0]
        optimizer_label = tk.Label(self, text = "Optimizer:")
        optimizer_label.grid(row = 0, column = 0, sticky = "w", padx = 10, pady = 5)

        self.optimizer_listbox = tk.Listbox(self, height = height, exportselection = False)
        self.optimizer_listbox.grid(row = 0, column = 1, padx = 10, pady = 5)        
        for opt in optimizers:
            self.optimizer_listbox.insert(tk.END, opt)
        self.optimizer_listbox.bind("<<ListboxSelect>>", self.on_optimizer_select)
        self.optimizer_listbox.selection_set(opt_ind)                 # Setting the default


        # Loss function
        losses = ["MSE", "Cat. Crossentropy", "Sparse Cat. Crossentropy",       # loss functions
                  "Binary Crossentropy"] 
        losses_map = {"MSE": "mse",  "Cat. Crossentropy": "categorical_crossentropy",
                      "Sparse Cat. Crossentropy": "sparse_categorical_crossentropy",
                      "Binary Crossentropy": "binary_crossentropy"}
        def on_loss_select(event):
            selected_display = self.loss_listbox.get(                 # Get the selected item
                self.loss_listbox.curselection())
            selected_loss = loss_map[selected_display]                # Retrieve lowercase
            self.shared_settings["loss"] = selected_loss              # Update shared_settings
        
        # Getting the default index for the loss
        loss_key = next((k for k, v in losses_map.items() if v == self.shared_settings['loss']), None)
        if loss_key is not None:
            loss_ind = losses.index(loss_key)     # Find the index of the loss key in the losses list
        else:
            loss_ind = 0                          # Default to the first index if not found
        
        loss_label = tk.Label(self, text = "Loss Function:")
        loss_label.grid(row = 0, column = 2, sticky = "w", padx = 10, pady = 5)

        self.loss_listbox = tk.Listbox(self, height = height, exportselection = False)
        self.loss_listbox.grid(row = 0, column = 3, padx = 10, pady = 5)        
        for loss in losses:
            self.loss_listbox.insert(tk.END, loss)
        self.loss_listbox.bind("<<ListboxSelect>>", self.on_loss_select)
        self.loss_listbox.selection_set(loss_ind)                     # Setting the default


        # Epochs
        epochs_options = [1, 5, 10, 20, 50, 100]                      # epoch options
        ep_ind = epochs_options.index(self.shared_settings["epochs"]) # Getting the default
        epochs_label = tk.Label(self, text = "Epochs:")
        epochs_label.grid(row = 1, column = 0, sticky = "w", padx = 10, pady = 5)

        self.epochs_listbox = tk.Listbox(self, height = height, exportselection = False)
        self.epochs_listbox.grid(row = 1, column = 1, padx = 10, pady = 5)       
        for ep in epochs_options:
            self.epochs_listbox.insert(tk.END, ep)
        self.epochs_listbox.bind("<<ListboxSelect>>", self.on_epochs_select)
        self.epochs_listbox.selection_set(ep_ind)                     # Setting the default


        # Batch size
        batch_sizes = [32, 64, 128, 256, 512]                         # batch size options                    
        bs_ind = batch_sizes.index(self.shared_settings["batch_size"])# Getting the default
        batch_size_label = tk.Label(self, text = "Batch Size:")
        batch_size_label.grid(row = 1, column = 2, sticky = "w", padx = 10, pady = 5)

        self.batch_size_listbox = tk.Listbox(self, height = height, exportselection = False)
        self.batch_size_listbox.grid(row = 1, column = 3, padx = 10, pady = 5)    
        for bs in batch_sizes:
            self.batch_size_listbox.insert(tk.END, bs)
        self.batch_size_listbox.bind("<<ListboxSelect>>", self.on_batch_size_select)
        self.batch_size_listbox.selection_set(bs_ind)                 # Setting the default


        # Done Button: print the current settings configuration and exit
        self.done_button = tk.Button(self, text = "Done", command = self.done)
        self.done_button.grid(row = 0, column = 5,  padx = 10, pady = 50)

        
    def on_optimizer_select(self, event):
        selection = self.optimizer_listbox.get(self.optimizer_listbox.curselection())
        self.shared_settings["optimizer"] = selection

    def on_loss_select(self, event):
        selection = self.loss_listbox.get(self.loss_listbox.curselection())
        self.shared_settings["loss"] = selection

    def on_epochs_select(self, event):
        selection = self.epochs_listbox.get(self.epochs_listbox.curselection())
        self.shared_settings["epochs"] = int(selection)

    def on_batch_size_select(self, event):
        selection = self.batch_size_listbox.get(self.batch_size_listbox.curselection())
        self.shared_settings["batch_size"] = int(selection)

    def done(self):
        #print("Current Model Configuration:")
        #for key, value in self.shared_settings.items():
        #    print(f"{key}: {value}")
        self.destroy()
