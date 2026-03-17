from llama_cpp import Llama
import gradio as gr

llm = Llama(
    model_path="model.gguf",
    n_ctx=2048
)

def chat(prompt):
    output = llm(prompt, max_tokens=200)
    return output["choices"][0]["text"]

gr.Interface(fn=chat, inputs="text", outputs="text").launch(server_name="0.0.0.0", server_port=7860)
