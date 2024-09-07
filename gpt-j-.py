import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model name (GPT-J model)
model_name = "EleutherAI/gpt-j-6B"

# Load the model and tokenizer
try:
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model, tokenizer = None, None

# Function to generate a response using the model
def get_response(prompt, max_new_tokens=50):
    if model and tokenizer:
        try:
            # Tokenize the input prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate output from the model
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,  # Controls randomness; lower values make outputs more deterministic
                do_sample=True,   # Enable sampling to use temperature
                pad_token_id=tokenizer.eos_token_id  # Set padding token ID
            )

            # Decode the generated tokens to a string
            response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return response
        except Exception as e:
            return f"Error generating response: {e}"
    else:
        return "Model not loaded."

# Example usage
prompt = "She is"
response = get_response(prompt, max_new_tokens=50)
print(f"Response: {response}")
