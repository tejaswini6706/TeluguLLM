import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

# --- Configuration ---
MODEL_ID = "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0"
FINE_TUNED_MODEL_PATH = "./fine_tuned_telugu_llm"

# --- Model Loading ---
print(f"Loading tokenizer for model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully.")

print("Configuring BitsAndBytes for quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("BitsAndBytes configuration complete.")

print(f"Loading model: {MODEL_ID} with quantization...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model with quantization: {e}")
    print("Attempting to load model without quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("Model loaded successfully without quantization.")

model.eval()

# --- Inference Function ---
def generate_telugu_response(user_input: str) -> str:
    """Generates a Telugu response from the LLM based on user input."""
    prompt = f"### Instruction:\n{user_input}\n\n### Response:"
    
    print(f"\nUser Input: {user_input}")
    print(f"Formatted Prompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=250,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Raw Generated Text: {generated_text}")

    response_start_marker = "### Response:"
    if response_start_marker in generated_text:
        response_only = generated_text.split(response_start_marker, 1)[1].strip()
    else:
        response_only = generated_text.strip()

    response_only = response_only.replace("### Instruction:", "").replace("### Response:", "").strip()
    print(f"Cleaned Response: {response_only}")
    return response_only

# --- Gradio Interface ---
print("Setting up Gradio interface...")
iface = gr.Interface(
    fn=generate_telugu_response,
    inputs=gr.Textbox(
        lines=3,
        placeholder="తెలుగులో మీ ప్రశ్నను ఇక్కడ టైప్ చేయండి...",
        label="మీ ప్రశ్న (Your Question)"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="LLM సమాధానం (LLM Response)"
    ),
    title="తెలుగు LLM చాట్‌బాట్ (Telugu LLM Chatbot)",
    description=(
        "ఇది ఒక ముందే శిక్షణ పొందిన తెలుగు LLM. మీరు తెలుగులో ప్రశ్నలు అడగవచ్చు "
        "మరియు అది మీకు సమాధానం ఇస్తుంది. మీ ప్రత్యేక అవసరాల కోసం మీరు దీన్ని "
        "మీ స్వంత డేటాతో ఫైన్-ట్యూన్ చేయవచ్చు."
        "\n\n(This is a pre-trained Telugu LLM. You can ask questions in Telugu "
        "and it will provide answers. You can fine-tune it with your own data "
        "for your specific needs.)"
    ),
    theme="soft"
)

if __name__ == "__main__":
    print("Launching Gradio interface...")
    iface.launch(share=True, server_name="0.0.0.0", server_port=7860)