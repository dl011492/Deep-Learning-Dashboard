import re
import tkinter as tk
import tensorflow as tf

class textWinApp(tk.Toplevel):
    def __init__(self, parent, sample_text, shared_data):
        super().__init__(parent)
        self.parent = parent                      # Store a reference to the parent window
        self.shared_data = shared_data

        if self.parent.winfo_name() == "dsFrame":
            self.title("Train Sample")
        elif self.parent.winfo_name() == "predictFrame":
            self.title("Test Sample")            

        # Decoding function for a sequence
        def decode_review(encoded_review):    
            words = [reverse_word_index.get(i, "?") for i in encoded_review]
            cleaned_words = [word for word in words if word not in ("<START>", "br")]
            return " ".join(cleaned_words)

        # Stripping function: replaces <br> or <br /> with newline characters
        def strip_html_tags_with_breaks(text):
            if isinstance(text, bytes):                # Convert bytes to string using UTF-8 encoding
                text = text.decode('utf-8')  
            text = re.sub(r'<br\s*/?>', '\n', text)    # Handles <br>, <br/>, and <br />
            # Remove all other HTML tags
            text = re.sub(r'<[^>]*>', '', text)
            return text

        if isinstance(sample_text, list):         # For imdb seqs of integers train and test samples 
            word_index = tf.keras.datasets.imdb.get_word_index()       # Load the word index
            word_index = {k: (v + 3) for k, v in word_index.items()}
            word_index["<PAD>"] = 0
            word_index["<START>"] = 1
            word_index["<UNK>"] = 2
            word_index["<UNUSED>"] = 3

            # Reverse the word index to decode reviews back to text
            reverse_word_index = {value: key for key, value in word_index.items()}
            paragraph = decode_review(sample_text)
            cropped_paragraph = "\n".join(paragraph.splitlines()[:10])

        elif sample_text is None:                 # For aclImdb train samples                        
            # Get the first element from the batch
            for inputs, targets in self.shared_data['train_dataset']:
                sample_text = inputs[0]
                break
            paragraph = sample_text.numpy().decode("utf-8")
            clean_text = strip_html_tags_with_breaks(paragraph)
            cropped_paragraph = "\n".join(clean_text.splitlines()[:10])

        elif isinstance(sample_text, tf.Tensor):  # For aclImdb test samples
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
