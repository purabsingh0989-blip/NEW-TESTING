from llama_cpp import Llama
import gradio as gr

# Load local GGUF model (NO downloading)
llm = Llama(
    model_path="model.gguf",
    n_ctx=2048,
    n_threads=4,
    n_batch=512,
    use_mmap=True,
    use_mlock=False,
    n_gpu_layers=0
)

def chat(message, history):
    prompt = ""

    for user, bot in history:
        prompt += f"User: {user}\nAssistant: {bot}\n"

    prompt += f"User: {message}\nAssistant:"

    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        stop=["User:", "</s>"]
    )

    return output["choices"][0]["text"].strip()

demo = gr.ChatInterface(fn=chat)

demo.launch(server_name="0.0.0.0", server_port=7860)
