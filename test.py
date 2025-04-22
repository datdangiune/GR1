import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn model đã fine-tune
MODEL_PATH = "./trained_model2"

# Load tokenizer và model đã huấn luyện
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

def chat_with_model(question: str, max_new_tokens=256, temperature=0.7, top_p=0.95, top_k=50):
    """Sinh câu trả lời từ model dựa trên câu hỏi"""
    prompt = f"Question: {question.strip()} Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    # Giải mã và trích xuất phần trả lời
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()

    if not answer.endswith(('.', '?', '!')):
        answer += '.'

    return answer

def run_cli():
    """Giao diện trò chuyện đơn giản trong terminal"""
    print("🤖 Chatbot Y Tế (gõ 'exit' để thoát)\n")
    while True:
        try:
            user_input = input("👤 Bạn: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("👋 Tạm biệt!")
                break

            if not user_input:
                print("⚠️ Vui lòng nhập câu hỏi.")
                continue

            reply = chat_with_model(user_input)
            print(f"🤖 Bác sĩ AI: {reply}\n")
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break

if __name__ == "__main__":
    run_cli()
