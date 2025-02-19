import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import datasets
import pandas as pd
from datasets import Dataset

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate a funny joke
response = generator("Tell me a funny joke about cats:", max_length=50, num_return_sequences=1)
print(response[0]['generated_text'])


def load_jokes_dataset(word):
   # Load the dataset
    df = pd.read_csv("short-jokes.csv")
    
    # Filter jokes that contain the input word (case-insensitive)
    filtered_jokes = df[df['joke'].str.contains(word, case=False, na=False)]
    
    # Return a random joke if there are matches, otherwise return a message
    if not filtered_jokes.empty:
        return filtered_jokes.sample(n=1).iloc[0]['joke']
    else:
        return "No jokes found with that word! Try another word 😊"



with gr.Blocks(css=".button-custom {background-color: #FF5733; color: white; font-size: 18px; border-radius: 12px; padding: 12px 20px; border: none; cursor: pointer;}") as demo:
    gr.Markdown("**Dad Joke Generator**")
    gr.Markdown("just type a word and ill make a dad joke out of it")
    word_input = gr.Textbox(label="Enter a word")
    output = gr.Textbox(" ")
    button = gr.Button("Submit")
    button.click(fn=load_jokes_dataset,inputs=word_input, outputs=output)
demo.launch(share=True)
