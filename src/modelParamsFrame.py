import tkinter as tk

class ModelParamsFrame(tk.Toplevel):    
    def __init__(self, parent, shared_settings):
        super().__init__(parent)
        self.parent = parent                      # Store a reference to the parent window
        self.shared_settings = shared_settings
        self.layer_widgets = {}                   # Dictionary to store widgets for each layer

        self.title("Model Parameters")                                # Frame title
           
        # Done Button: print the current settings configuration and exit
        self.done_button = tk.Button(self, text = "Done", command = self.done)
        self.done_button.grid(row = 0, column = 6,  padx = 10, pady = 20)

        # Number of layers
        #layers = range(1, max_layers + 1)        
        self.layer_count = tk.StringVar(self)
        self.layer_count.set(self.shared_settings["layers"])          # Getting the default
        #num_layers = int(self.layer_count.get())
        num_layers = self.shared_settings["layers"]
        layers = range(1, num_layers + 1)
        layers_label = tk.Label(self, text = "Number of Layers:")
        layers_label.grid(row = 0, column = 0, sticky = "w", padx = 10, pady = 20)

        self.layers_menu = tk.OptionMenu(self, self.layer_count, *layers, )
                                         #command = self.on_layer_select)  # on_layer_select is de-activated
        self.layers_menu.grid(row = 0, column = 1, pady = 5)
        #self.layer_count.set(num_layers)

        # Layer parameters (dynamic, added below based on the number of layers selected)
        self.layer_start_row = 1                  # Starting row for layer options

        # Get selected number of layers and update shared_settings        
        #num_layers = int(self.layer_count.get())
        #self.shared_settings["layers"] = num_layers

        # Add settings options for each layer dynamically
        layer_types = ["Input", "Flatten", "Dense", "Dropout", "Conv2D", "LSTM"]

        for layer_index in range(num_layers):
            # Label for Layer Type                                    # Getting the default
            layer_type_ind = layer_types.index(self.shared_settings['layers_config'][layer_index].get('type', layer_types[0]))
            #layer_type_ind = layer_types.index(self.shared_settings['layers_config'][layer_index]['type'])
            layer_type_label = tk.Label(self, text = f"Layer {layer_index + 1}:")
            layer_type_label.grid(row = self.layer_start_row + layer_index,
                                  column = 0, sticky = "w", padx = 10, pady = 5)
                        
            # ListBox for Layer Type selection
            layer_type_listbox = tk.Listbox(self, height = 6, exportselection = False)
            layer_type_listbox.grid(row = self.layer_start_row + layer_index,
                                    column = 1, padx = 10, pady = 5)
            for lt in layer_types:
                layer_type_listbox.insert(tk.END, lt)                
            layer_type_listbox.bind("<<ListboxSelect>>", lambda e, i = layer_index:
                                    self.on_layer_type_select(e, i))
            layer_type_listbox.selection_set(layer_type_ind)          # Setting the default
            
            # Store the widgets to be able to clear them later
            self.layer_widgets[layer_index] = [layer_type_label, layer_type_listbox]

            # Label and ListBox for Neurons in Layer
            neurons = [10, 16, 32, 64, 128, 256, 512]                 # Getting the default
            neurons_ind = neurons.index(self.shared_settings['layers_config'][layer_index].get('neurons', neurons[0]))
            #neurons_ind = neurons.index(self.shared_settings['layers_config'][layer_index]['neurons'])           
            neurons_label = tk.Label(self, text = f"Neurons:")
            #neurons_label.grid(row=self.layer_start_row + layer_index, column=2, sticky="w", padx=10, pady=5)

            neurons_listbox = tk.Listbox(self, height = 6, exportselection = False)
            #neurons_listbox.grid(row=self.layer_start_row + layer_index, column=3, padx=10, pady=5)
            for n in neurons:
                neurons_listbox.insert(tk.END, n)
            neurons_listbox.bind("<<ListboxSelect>>", lambda e, i = layer_index:
                     self.on_neurons_select(e, i))
            neurons_listbox.selection_set(neurons_ind)                # Setting the default

            # Label and ListBox for Activation Functions
            activations = ["relu", "sigmoid", "tanh", "softmax"]
            activations_ind = activations.index(self.shared_settings['layers_config'][layer_index].get('activation', activations[0]))            
            activation_label = tk.Label(self, text=f"Activation:")
            #activation_label.grid(row=self.layer_start_row + layer_index, column=4, sticky="w", padx=10, pady=5)

            activation_listbox = tk.Listbox(self, height = 6, exportselection = False)
            #activation_listbox.grid(row=self.layer_start_row + layer_index, column=5, padx=10, pady=5)
            for act in activations:
                activation_listbox.insert(tk.END, act)
            #activation_listbox.bind("<<ListboxSelect>>", lambda e, i=layer_index:
            #            self.on_activation_select(e, i))
            activation_listbox.selection_set(activations_ind)  # Setting the default

            # Store neurons and activation widgets for dynamic updating later
            self.layer_widgets[layer_index].extend([neurons_label, neurons_listbox, activation_label, activation_listbox])

            # Label and ListBox for dropout rates
            rates = [0.1, 0.2, 0.3, 0.4, 0.5]                         # Getting the default
            rates_ind = rates.index(self.shared_settings['layers_config'][layer_index].get('dropout', rates[0]))
            dropout_label = tk.Label(self, text = f"Dropout Rate:")

            dropout_listbox = tk.Listbox(self, height = 6, exportselection = False)
            #dropout_listbox.grid(row = self.layer_start_row + layer_index, column = 3, padx = 10, pady = 5)
            for rate in rates:  # Dropout rates
                dropout_listbox.insert(tk.END, rate)
            #dropout_listbox.bind("<<ListboxSelect>>", lambda e, i = layer_index:
            #                     self.on_dropout_select(e, i))
            dropout_listbox.selection_set(rates_ind)                  # Setting the default
            

            # Check the default layer type and only grid neurons/activation widgets if required
            default_layer_type = self.shared_settings['layers_config'][layer_index]['type']
            if default_layer_type in ["Dense", "Conv2D", "LSTM"]:
                # Place neurons and activation widgets on the grid
                neurons_label.grid(     row = self.layer_start_row + layer_index,
                                        column = 2, sticky = "w", padx = 10, pady = 5)
                neurons_listbox.grid(   row = self.layer_start_row + layer_index,
                                        column = 3, padx = 10, pady = 5)
                activation_label.grid(  row = self.layer_start_row + layer_index,
                                        column = 4, sticky = "w", padx = 10, pady = 5)
                activation_listbox.grid(row = self.layer_start_row + layer_index,
                                        column = 5, padx = 10, pady = 5)
            if default_layer_type in ["Dropout"]:
                dropout_label.grid(     row = self.layer_start_row + layer_index,
                                        column = 2, sticky = "w", padx = 10, pady = 5)                
                dropout_listbox.grid(   row = self.layer_start_row + layer_index,
                                        column = 3, padx = 10, pady = 5)                
                
    # Function to handle layer type selection and toggle visibility of widgets
    def on_layer_type_select(self, event, layer_index):
        selection = event.widget.get(event.widget.curselection())
        self.shared_settings.setdefault("layer_settingss", {})[layer_index] = {"type": selection}

        # Remove existing neuron and activation widgets if they exist
        #for widget in self.layer_widgets[layer_index][2:]:
        #    widget.destroy()
        #self.layer_widgets[layer_index] = self.layer_widgets[layer_index][:2]

        #activations = ["ReLU", "Sigmoid", "Tanh", "Softmax"]
        #activations = ["relu", "sigmoid", "tanh", "softmax"]
        #neurons = [10, 16, 32, 64, 128, 256]

        # Update visibility or content of neurons and activation widgets based on layer type
        if selection in ["Dense", "Conv2D", "LSTM"]:
            # Show neurons and activation options
            self.layer_widgets[layer_index][2].grid(row = self.layer_start_row + layer_index,
                                                    column = 2, sticky = "w", padx = 10, pady = 5)
            self.layer_widgets[layer_index][3].grid(row=self.layer_start_row + layer_index,
                                                    column = 3, padx = 10, pady = 5)
            self.layer_widgets[layer_index][4].grid(row=self.layer_start_row + layer_index,
                                                    column = 4, sticky = "w", padx = 10, pady = 5)
            self.layer_widgets[layer_index][5].grid(row=self.layer_start_row + layer_index,
                                                    column = 5, padx = 10, pady = 5)
            #self.layer_widgets[layer_index][2].grid()  # neurons_label
            #self.layer_widgets[layer_index][3].grid()  # neurons_listbox
            #self.layer_widgets[layer_index][4].grid()  # activation_label
            #self.layer_widgets[layer_index][5].grid()  # activation_listbox
        else:
            # Hide neurons and activation options
            self.layer_widgets[layer_index][2].grid_remove()
            self.layer_widgets[layer_index][3].grid_remove()
            self.layer_widgets[layer_index][4].grid_remove()
            self.layer_widgets[layer_index][5].grid_remove()

        # Add relevant widgets based on layer type
        #if selection == "Dense" or selection == "Conv2D" or selection == "LSTM":
            # Label and ListBox for Neurons in Layer                  # Getting the default
            #neurons_ind = neurons.index(self.shared_settings['layers_config'][layer_index]['neurons'])
            #neurons_label = tk.Label(self, text = f"Neurons:")
            #neurons_label.grid(row = self.layer_start_row + layer_index,
            #                   column = 2, sticky = "w", padx = 10, pady = 5)
            
            #neurons_listbox = tk.Listbox(self, height = 6, exportselection = False)
            #neurons_listbox.grid(row = self.layer_start_row + layer_index,
            #                     column = 3, padx = 10, pady = 5)
            #for n in neurons:
            #    neurons_listbox.insert(tk.END, n)
            #neurons_listbox.bind("<<ListboxSelect>>", lambda e, i = layer_index:
            #                     self.on_neurons_select(e, i))
            #neurons_listbox.selection_set(neurons_ind)                # Setting the default

            # Label and ListBox for Activation Function               # Getting the default
            #act_ind = activations.index(self.shared_settings['layers_config'][layer_index]['activation'])
            #activation_label = tk.Label(self, text = f"Activation:")
            #activation_label.grid(row = self.layer_start_row + layer_index,
            #                      column = 4, sticky = "w", padx = 10, pady = 5)

            #activation_listbox = tk.Listbox(self, height = 4, exportselection = False)
            #activation_listbox.grid(row=self.layer_start_row + layer_index,
            #                        column = 5, padx = 10, pady = 5)            
            #for act in activations:                
            #    activation_listbox.insert(tk.END, act)
            #activation_listbox.bind("<<ListboxSelect>>", lambda e, i = layer_index:
             #                       self.on_activation_select(e, i))
            #activation_listbox.selection_set(act_ind)                 # Setting the default

            # Store widgets for cleanup
            #self.layer_widgets[layer_index].extend([neurons_label, neurons_listbox,
            #                                        activation_label, activation_listbox])

        #elif selection == "Dropout":
            # Label and ListBox for Dropout Rate 
            #rates = [0.1, 0.2, 0.3, 0.4, 0.5]                         # Getting the default
            #rates_ind = rates.index(self.shared_settings['layers_config'][layer_index]['dropout'])
            #dropout_label = tk.Label(self, text = f"Dropout Rate:")
            #dropout_label.grid(row = self.layer_start_row + layer_index,
            #                   column = 2, sticky = "w", padx = 10, pady = 5)

            #dropout_listbox = tk.Listbox(self, height = 4, exportselection = False)
            #dropout_listbox.grid(row = self.layer_start_row + layer_index,
            #                     column = 3, padx = 10, pady = 5)
            #for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:  # Dropout rates
            #    dropout_listbox.insert(tk.END, rate)
            #dropout_listbox.bind("<<ListboxSelect>>", lambda e, i = layer_index:
            #                     self.on_dropout_select(e, i))

            # Store widgets for cleanup
            #self.layer_widgets[layer_index].extend([dropout_label, dropout_listbox])
            #dropout_listbox.selection_set(rates_ind)                  # Setting the default

    def on_neurons_select(self, event, layer_index):
        selection = event.widget.get(event.widget.curselection())
        self.shared_settings["layer_settingss"][layer_index]["neurons"] = int(selection)

    def on_activation_select(self, event, layer_index):
        selection = event.widget.get(event.widget.curselection())
        self.shared_settings["layer_settingss"][layer_index]["activation"] = selection

    def on_dropout_select(self, event, layer_index):
        selection = event.widget.get(event.widget.curselection())
        self.shared_settings["layer_settingss"][layer_index]["dropout"] = float(selection)

    def done(self):
        print("Current Model Configuration:")
        for key, value in self.shared_settings.items():
            print(f"{key}: {value}")
        #print(self.shared_settings.get("layers_config"))
        #self.parent.destroy()
        self.destroy()

