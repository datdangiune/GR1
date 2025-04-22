from transformers import pipeline, AutoTokenizer

# Load tokenizer thủ công để lấy eos_token_id
tokenizer = AutoTokenizer.from_pretrained("./trained_model2")

# Load model và tokenizer từ thư mục đã fine-tune
chatbot = pipeline(
    "text-generation",
    model="./trained_model2",
    tokenizer=tokenizer
)

# Interactive test
print("Chatbot is ready. Type 'exit' or 'quit' to stop.")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    formatted_input = "Patient: " + user_input + "\nDoctor:"
    response = chatbot(
        formatted_input,
        max_length=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id  # giờ thì ổn
    )
    generated_text = response[0]['generated_text']
    if "Doctor:" in generated_text:
        doctor_reply = generated_text.split("Doctor:")[-1].strip()
    else:
        doctor_reply = generated_text.strip()
    print("Doctor:", doctor_reply)
