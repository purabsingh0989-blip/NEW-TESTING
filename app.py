from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import gradio as gr

# डाउनलोड मॉडल (auto डाउनलोड होगा पहली बार)
model_path = hf_hub_download(
    repo_id="mradermacher/Qwen2.5-Coder-1.5B-Unsensored-DPO-i1-GGUF",
    filename="Qwen2.5-Coder-1.5B-Unsensored-DPO-i1.Q4_K_M.gguf"
)

# मॉडल लोड (optimized settings)
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    n_batch=512,
    use_mmap=True,
    use_mlock=False,
    n_gpu_layers=0
)

# चैट फ़ंक्शन
def chat(message, history):
    prompt = ""

    # history build
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

    response = output["choices"][0]["text"].strip()
    return response

# ग्रेडियो UI
demo = gr.ChatInterface(fn=chat)

demo.launch(server_name="0.0.0.0", server_port=7860)
