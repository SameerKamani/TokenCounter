import gradio as gr
import tiktoken
from transformers import AutoTokenizer

# =========================
# Load tokenizers (once)
# =========================

# --- OpenAI / GPT-style tokenizers ---
enc_o200k = tiktoken.get_encoding("o200k_base")   # GPT-4o, GPT-4.1
enc_cl100k = tiktoken.get_encoding("cl100k_base") # GPT-4, GPT-3.5, Grok, Claude
enc_p50k = tiktoken.get_encoding("p50k_base")     # Codex (legacy)
enc_r50k = tiktoken.get_encoding("r50k_base")     # GPT-2 (legacy)

# --- Open-source model tokenizers ---
llama_tokenizer = AutoTokenizer.from_pretrained(
    "hf-internal-testing/llama-tokenizer",
    use_fast=True
)

mistral_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    use_fast=True
)

# Gemini tokenizer approximation (SentencePiece-based)
gemini_tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base",
    use_fast=True
)

# =========================
# Tokenization logic
# =========================

def tokenize(text, model):
    if not text.strip():
        return 0, ""

    # ---- GPT-style (tiktoken) ----
    if model == "GPT-4o / GPT-4.1":
        tokens = enc_o200k.encode(text)
        token_strings = [enc_o200k.decode([t]) for t in tokens]

    elif model in ["GPT-4 / GPT-3.5", "Claude", "Grok"]:
        tokens = enc_cl100k.encode(text)
        token_strings = [enc_cl100k.decode([t]) for t in tokens]

    elif model == "Codex (p50k)":
        tokens = enc_p50k.encode(text)
        token_strings = [enc_p50k.decode([t]) for t in tokens]

    elif model == "GPT-2 (r50k)":
        tokens = enc_r50k.encode(text)
        token_strings = [enc_r50k.decode([t]) for t in tokens]

    # ---- HuggingFace tokenizers ----
    elif model == "LLaMA":
        tokens = llama_tokenizer.encode(text, add_special_tokens=False, legacy=False)
        token_strings = llama_tokenizer.convert_ids_to_tokens(tokens)

    elif model == "Mistral":
        tokens = mistral_tokenizer.encode(text, add_special_tokens=False)
        token_strings = mistral_tokenizer.convert_ids_to_tokens(tokens)

    elif model == "Gemini":
        tokens = gemini_tokenizer.encode(text, add_special_tokens=False)
        token_strings = gemini_tokenizer.convert_ids_to_tokens(tokens)

    else:
        return 0, "Unknown model"

    return len(tokens), " | ".join(token_strings)


# =========================
# Gradio UI
# =========================

with gr.Blocks(title="LLM Token Counter") as demo:
    gr.Markdown("## LLM Token Counter (Tokenizer-Accurate)")

    model_choice = gr.Dropdown(
        choices=[
            "GPT-4o / GPT-4.1",
            "GPT-4 / GPT-3.5",
            "Claude",
            "Grok",
            "LLaMA",
            "Mistral",
            "Gemini",
            "Codex (p50k)",
            "GPT-2 (r50k)"
        ],
        value="GPT-4o / GPT-4.1",
        label="Select Model"
    )

    text_input = gr.Textbox(
        label="Input Text",
        lines=6,
        placeholder="Type or paste your text here..."
    )

    submit_btn = gr.Button("Count Tokens")

    token_count = gr.Number(label="Token Count")
    token_view = gr.Textbox(
        label="Token Breakdown",
        lines=6
    )

    submit_btn.click(
        fn=tokenize,
        inputs=[text_input, model_choice],
        outputs=[token_count, token_view]
    )

if __name__ == "__main__":
    demo.launch()
