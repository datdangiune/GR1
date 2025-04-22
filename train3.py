# INFERENCE
from transformers import pipeline
# Inference: Load the fine-tuned model using a text-generation pipeline.
chatbot = pipeline(
    "text-generation",
model="./trained_model2",
tokenizer="./trained_model2"

)

# Interactive loop to test the chatbot.
print("Chatbot is ready. Type 'exit' or 'quit' to stop.")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    # Format the input to indicate patient dialogue, and prompt the model for a doctor's reply.
    formatted_input = "Patient: " + user_input + "\nDoctor:"
    response = chatbot(
        formatted_input,
        max_length=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    # Extract doctor's reply by taking the text after "Doctor:" if present.
    generated_text = response[0]['generated_text']
    if "Doctor:" in generated_text:
        doctor_reply = generated_text.split("Doctor:")[-1].strip()
    else:
        doctor_reply = generated_text.strip()
    print("Doctor:", doctor_reply)