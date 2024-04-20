# Assignment: API Library Setups

import gpt_2_simple as gpt2
import nltk
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import tkinter as tk
from tkinter import scrolledtext

nltk.download('stopwords')

# Download the GPT-2 model if not already downloaded
model_name = "124M"
models_dir = "F:/AI School/MS Adv Prog/GPT-2-Simple/GPT-2-Simple/models"

# Check if the model is already downloaded
checkpoint_dir = os.path.join(models_dir, model_name)
if not os.path.exists(checkpoint_dir):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name, model_dir=models_dir)
    print(f"Downloaded {model_name} model.")

# Use the model
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, model_name=model_name, checkpoint_dir=checkpoint_dir)


def generate_text(prompt, sess, checkpoint_dir, temperature=0.5):
    if not os.path.exists(checkpoint_dir):
        print(f"Model {checkpoint_dir} not found. Downloading...")
        gpt2.download_gpt2(model_name=model_name, model_dir=models_dir)
        print(f"Downloaded {checkpoint_dir} model.")

    gpt2.load_gpt2(sess, model_name=model_name, checkpoint_dir=checkpoint_dir, reuse=True)

    # Generate text with the specified temperature
    text = gpt2.generate(sess, model_name=model_name, checkpoint_dir=checkpoint_dir, prefix=prompt, temperature=temperature, return_as_list=True)[0]
    return text.strip()

def process_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    return ' '.join(tokens)

def make_request(url):
    response = requests.get(url)
    return response.text

def generate_text_gui():
    prompt = text_entry.get("1.0", tk.END).strip()
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "Generating text...")

    # Use after() to schedule the text generation after a short delay
    root.after(100, lambda: generate_text_and_display(prompt))

def generate_text_and_display(prompt):
    sess = gpt2.start_tf_sess()
    checkpoint_dir = os.path.join(models_dir, model_name)

    generated_text = generate_text(prompt, sess, checkpoint_dir)
    processed_text = process_text(generated_text)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, processed_text)

    # Reset the TensorFlow session to release the variables
    gpt2.reset_session(sess)


# Create the main window
root = tk.Tk()
root.title("Text Generator")

# Create a text entry widget for the user to enter a prompt
text_entry = scrolledtext.ScrolledText(root, width=120, height=3, wrap=tk.WORD)
text_entry.pack(padx=10, pady=10)

# Create a button to trigger the text generation
generate_button = tk.Button(root, text="Generate Text", command=generate_text_gui)
generate_button.pack(pady=10)

# Create a text widget to display the generated and processed text
output_text = scrolledtext.ScrolledText(root, width=120, height=30, wrap=tk.WORD)
output_text.pack(padx=10, pady=10)

# Start the GUI event loop
root.mainloop()

if __name__ == "__main__":
    url = "https://jsonplaceholder.typicode.com/posts/1"
    response_text = make_request(url)
    print("\nResponse Text:")
    print(response_text)

    # Reset the TensorFlow session to release the variables
    gpt2.reset_session(sess)


