import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# [1] Load the trained model and tokenizer
model_path = "./gpt2-medical-model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# [2] Test with an example input
sample_input = "What is Huntington's disease?"
print(f"Input: {sample_input}")

# Tokenize the input
input_ids = tokenizer.encode(sample_input, return_tensors='pt')

# [3] Generate prediction from the model
with torch.no_grad():
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# [4] Decode the generated output and print it
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated Output: {generated_text}")
