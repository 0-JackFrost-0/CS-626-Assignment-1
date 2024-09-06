import gradio as gr
import nltk
from hmm import get_POS

nltk.download('universal_tagset')

def pos_tagger(text):
    pos_result = get_POS(text)
    return pos_result

iface = gr.Interface(
    fn=pos_tagger,
    inputs="text",
    outputs="text",
    title="POS Tagger",
    flagging_options=None,
    description="Enter a sentence to get Part-of-Speech tags."
)

iface.launch()