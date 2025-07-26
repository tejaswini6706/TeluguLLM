from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)
CORS(app)

# --- Configuration ---
# Using Facebook's BART model which is good for summarization
MODEL_ID = "facebook/bart-large-cnn"

# --- Model Loading ---
print(f"Loading tokenizer and model: {MODEL_ID}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    
    # Create a summarization pipeline for easier use
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# --- Inference Function ---
def generate_summary_backend(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """
    Generates a summary from the input text using the BART model.
    """
    print(f"Received input for summarization (length: {len(text)})")
    
    try:
        # Use the pipeline for summarization
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        
        result = summary[0]['summary_text']
        print(f"Generated summary: {result}")
        return result
    except Exception as e:
        print(f"Error during summarization: {e}")
        raise e

# --- API Endpoint ---
@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    API endpoint to receive text and return summarized version.
    Expects a JSON payload with 'text' key and optional 'max_length'/'min_length'.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    try:
        max_length = data.get('max_length', 130)
        min_length = data.get('min_length', 30)
        
        summary = generate_summary_backend(text, max_length, min_length)
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({"error": "Internal server error during summarization"}), 500

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": MODEL_ID})

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)