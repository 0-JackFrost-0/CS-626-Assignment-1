import gradio as gr
import nltk
from nltk import pos_tag
from hmmforapp import get_POS
from nltk.tokenize import word_tokenize

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

# Define the function that gets POS tags
# def get_POS(text):
#     words = word_tokenize(text)
#     pos_tags = pos_tag(words, tagset='universal')
#     return pos_tags

# Define a function that will be linked with the Gradio UI
def pos_tagger(text):
    pos_result = get_POS(text)
    return pos_result

# Create the Gradio interface
iface = gr.Interface(
    fn=pos_tagger,
    inputs="text",
    outputs="text",
    title="POS Tagger",
    flagging_options=None,
    description="Enter a sentence to get Part-of-Speech tags."
)

# Launch the interface
iface.launch()