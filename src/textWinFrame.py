import re
import tkinter as tk

class textWinApp(tk.Toplevel):
    def __init__(self, parent, sample_text, shared_data):
        super().__init__(parent)
        self.parent = parent                      # Store a reference to the parent window
        self.shared_data = shared_data

        self.title("Text Sample")

        # Replace <br> or <br /> with newline characters
        def strip_html_tags_with_breaks(text):
            if isinstance(text, bytes):                # Convert bytes to string using UTF-8 encoding
                text = text.decode('utf-8')  
            text = re.sub(r'<br\s*/?>', '\n', text)    # Handles <br>, <br/>, and <br />
            # Remove all other HTML tags
            text = re.sub(r'<[^>]*>', '', text)
            return text

        # Visualizing sample reviews from the train_dataset only if sample_text is none        
        if sample_text is None:
            # Get the first element from the batch
            for inputs, targets in self.shared_data['train_dataset']:
                sample_text = inputs[0]
                break
            
        paragraph = sample_text.numpy().decode("utf-8")
        clean_text = strip_html_tags_with_breaks(paragraph)
        cropped_paragraph = "\n".join(clean_text.splitlines()[:10])
                        
        label = tk.Label(self, text = cropped_paragraph, wraplength = 900,
                         justify = "left")
        label.pack()

        # Done Button
        self.done_button = tk.Button(self, text = "Done", command = self.done)
        self.done_button.pack(pady = 10)
        
    def done(self):
        self.destroy()
