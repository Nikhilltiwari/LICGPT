from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re

app = Flask(__name__)

# Ensure the path to the trained model directory is correct
model_path = r"D:\ML\LICGPT\trained_model"

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def preprocess_input(prompt):
    prompt = re.sub(r'((I think it\'s a good policy\. )+)', '', prompt)  # Remove repetitive "I think it's a good policy."
    prompt = re.sub(r' +', ' ', prompt)  # Remove extra spaces
    prompt = prompt.replace('The government has also announced that it will introduce a new online banking system, which will allow people to pay for their own online banking.', '')
    return prompt.strip()

def postprocess_response(response):
    # Clean up repetitive or irrelevant responses
    response = re.sub(r'\bi think it\'s a good policy\b', '', response, flags=re.IGNORECASE)
    response = re.sub(r'\s+', ' ', response).strip()
    return response

def generate_response(prompt, model, tokenizer, max_length=512):
    prompt = preprocess_input(prompt)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if len(inputs[0]) > max_length:
        inputs = inputs[:, :max_length]  # Truncate inputs if too long
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return postprocess_response(response)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if "prompt" not in data:
        return jsonify({"error": "Please provide a prompt"}), 400

    prompt = data["prompt"]
    response = generate_response(prompt, model, tokenizer)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
