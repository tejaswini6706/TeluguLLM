import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset # Used for creating a dummy dataset or loading real data

# --- Configuration ---
# Define the pre-trained model ID from Hugging Face Hub.
# This model is instruction-tuned for Indian languages, including Telugu.
MODEL_ID = "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0"

# Path to save/load fine-tuned adapters (if fine-tuning is performed)
FINE_TUNED_MODEL_PATH = "./fine_tuned_telugu_llm"

# --- Model Loading ---
# Initialize tokenizer
print(f"Loading tokenizer for model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Set pad_token to eos_token for proper generation behavior, especially with some models.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully.")

# Configure BitsAndBytes for 4-bit quantization to reduce memory usage.
# This is crucial for running large models (like 7B parameters) on consumer GPUs.
# If you have a very powerful GPU (e.g., A100), you might set load_in_4bit=False.
print("Configuring BitsAndBytes for quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # NormalFloat 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for computation if your GPU supports it (Ampere+ architecture)
                                          # Otherwise, use torch.float16
)
print("BitsAndBytes configuration complete.")

# Load the base model with quantization
print(f"Loading model: {MODEL_ID} with quantization...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto", # Automatically distributes the model across available GPUs
        torch_dtype=torch.bfloat16 # Ensure model is loaded with bfloat16 if using it for compute
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model with quantization: {e}")
    print("Attempting to load model without quantization (might require more VRAM)...")
    # Fallback to float16 if bfloat16 causes issues or is not supported
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("Model loaded successfully without quantization.")

# Set model to evaluation mode for inference
model.eval()

# --- Inference Function ---
def generate_telugu_response(user_input: str) -> str:
    """
    Generates a Telugu response from the LLM based on user input.
    The prompt format is crucial for instruction-tuned models like Navarasa 2.0.
    """
    # The model expects a specific instruction format.
    # Check the model card on Hugging Face Hub for the exact format if you switch models.
    prompt = f"### Instruction:\n{user_input}\n\n### Response:"

    print(f"\nUser Input: {user_input}")
    print(f"Formatted Prompt: {prompt}")

    # Encode the prompt to input IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generate the response
    # Parameters for generation can be tuned for different output styles:
    # max_new_tokens: Maximum length of the generated response.
    # num_beams: For beam search, generally produces more coherent and less repetitive text.
    # do_sample: If True, enables sampling (temperature, top_k, top_p) for more creative outputs.
    # temperature: Controls randomness. Lower values (e.g., 0.1-0.5) for more deterministic,
    #              higher values (e.g., 0.7-1.0) for more creative.
    # top_k: Limits sampling to the top_k most probable tokens.
    # top_p: Filters tokens based on cumulative probability (nucleus sampling).
    # no_repeat_ngram_size: Prevents repetition of n-grams.
    # early_stopping: Stops generation when all beam hypotheses have met a stopping criterion.
    outputs = model.generate(
        input_ids,
        max_new_tokens=250, # Generate up to 250 new tokens
        num_beams=5,        # Use beam search with 5 beams
        no_repeat_ngram_size=2, # Avoid repeating 2-grams
        early_stopping=True,
        do_sample=True,      # Enable sampling
        temperature=0.7,     # Moderate creativity
        top_k=50,            # Consider top 50 tokens
        top_p=0.9,           # Nucleus sampling
        pad_token_id=tokenizer.pad_token_id, # Use the defined pad token
        eos_token_id=tokenizer.eos_token_id # Use the defined end-of-sequence token
    )

    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Raw Generated Text: {generated_text}")

    # Post-process the generated text to extract only the model's response.
    # The model might echo the prompt as part of its output.
    response_start_marker = "### Response:"
    if response_start_marker in generated_text:
        response_only = generated_text.split(response_start_marker, 1)[1].strip()
    else:
        # Fallback if the marker isn't found (e.g., if the model generates differently)
        response_only = generated_text.strip()

    # Remove any lingering instruction/response markers that might appear due to model quirks
    response_only = response_only.replace("### Instruction:", "").replace("### Response:", "").strip()

    print(f"Cleaned Response: {response_only}")
    return response_only

# --- Placeholder for Fine-tuning (Uncomment and modify to use) ---
"""
# If you want to fine-tune the model on your custom Telugu data,
# uncomment the code below and follow these steps:

# 1. Prepare your custom Telugu dataset.
#    It should be a list of dictionaries, where each dictionary has a 'text' key
#    containing your instruction-response pairs formatted like:
#    "### Instruction:\n{your_instruction_in_telugu}\n\n### Response:\n{your_response_in_telugu}"
#    Example:
# my_custom_telugu_data = [
#     {"text": "### Instruction:\n'రాముడు మంచి బాలుడు' అనే వాక్యాన్ని వివరించండి.\n\n### Response:\n'రాముడు మంచి బాలుడు' అంటే 'Rama is a good boy' అని అర్థం. ఇది ఒక సాధారణ తెలుగు వాక్యం."},
#     {"text": "### Instruction:\nనాకు హైదరాబాద్ గురించి కొన్ని ఆసక్తికరమైన విషయాలు చెప్పండి.\n\n### Response:\nహైదరాబాద్ ముత్యాలకు మరియు బిర్యానీకి ప్రసిద్ధి. ఇది చార్మినార్ మరియు గోల్కొండ కోట వంటి చారిత్రక కట్టడాలకు నిలయం."},
#     # Add more of your specific Telugu data here
# ]
# custom_dataset = Dataset.from_list(my_custom_telugu_data)

# 2. Prepare the model for k-bit training (essential for LoRA/QLoRA)
# model = prepare_model_for_kbit_training(model)

# 3. Configure LoRA (Low-Rank Adaptation)
#    LoRA allows efficient fine-tuning by only training a small number of new parameters.
# lora_config = LoraConfig(
#     r=16, # Rank of the update matrices (common values: 8, 16, 32, 64)
#     lora_alpha=32, # Scaling factor for LoRA weights
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM" # Indicates it's a generative language model task
# )
# model = get_peft_model(model, lora_config)
# print("Trainable parameters after PEFT:")
# model.print_trainable_parameters()

# 4. Define training arguments
# training_args = TrainingArguments(
#     output_dir=FINE_TUNED_MODEL_PATH, # Directory to save checkpoints and final model
#     num_train_epochs=3, # Number of passes over the training data
#     per_device_train_batch_size=2, # Adjust based on your GPU memory
#     gradient_accumulation_steps=4, # Accumulate gradients to simulate a larger batch size
#     optim="paged_adamw_8bit", # Optimizer, good for QLoRA
#     save_strategy="epoch", # Save model checkpoint after each epoch
#     logging_dir="./logs", # Directory for logs
#     logging_steps=10, # Log training metrics every 10 steps
#     learning_rate=2e-4, # Learning rate for fine-tuning
#     bf16=True, # Use bfloat16 if your GPU supports it, otherwise set to False and fp16=True
#     group_by_length=True, # Improves training efficiency for varying sequence lengths
#     lr_scheduler_type="cosine", # Learning rate scheduler
#     warmup_ratio=0.05, # Warmup ratio for learning rate scheduler
#     report_to="none" # Disable reporting to Weights & Biases if not used
# )

# 5. Initialize and run the SFTTrainer
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=custom_dataset,
#     peft_config=lora_config,
#     dataset_text_field="text", # The column in your dataset containing the text
#     max_seq_length=512, # Max sequence length for tokenizer
#     tokenizer=tokenizer,
#     args=training_args,
#     packing=False, # Set to True to pack multiple short examples into one sequence for efficiency
# )
# print("Starting fine-tuning...")
# trainer.train()
# print("Fine-tuning complete. Saving fine-tuned model...")
# trainer.save_model(FINE_TUNED_MODEL_PATH)
# print(f"Fine-tuned model saved to {FINE_TUNED_MODEL_PATH}")

# # Optional: Merge LoRA adapters with the base model for easier deployment
# # This creates a full model checkpoint that doesn't require the base model separately
# base_model_reloaded = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )
# merged_model = PeftModel.from_pretrained(base_model_reloaded, FINE_TUNED_MODEL_PATH)
# merged_model = merged_model.merge_and_unload()
# merged_model.save_pretrained(f"{FINE_TUNED_MODEL_PATH}_merged")
# tokenizer.save_pretrained(f"{FINE_TUNED_MODEL_PATH}_merged")
# print(f"Merged model saved to {FINE_TUNED_MODEL_PATH}_merged")
"""

# --- Gradio Interface ---
# Create the Gradio interface for easy interaction with the LLM.
print("Setting up Gradio interface...")
iface = gr.Interface(
    fn=generate_telugu_response,
    inputs=gr.Textbox(
        lines=3,
        placeholder="తెలుగులో మీ ప్రశ్నను ఇక్కడ టైప్ చేయండి...", # Type your question in Telugu here...
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
    theme="soft" # A pleasant theme for the UI
)

# Launch the Gradio app
print("Launching Gradio interface...")
if __name__ == "__main__":
    iface.launch(share=True)
