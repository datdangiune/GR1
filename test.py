import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn tới model đã huấn luyện
MODEL_PATH = "./trained_model2"  # Thay đổi đường dẫn nếu cần

# Load tokenizer và model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# Hàm để hỏi mô hình
def chat_with_model(question, max_new_tokens=256, temperature=0.7, top_p=0.95):
    # Tạo prompt với mẫu câu tự nhiên
    prompt = f"Question: {question.strip()} Answer:"

    # Token hóa, với attention_mask rõ ràng
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Sinh văn bản
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.get("attention_mask")  # Đảm bảo attention mask được truyền
        )

    # Decode kết quả
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Tách phần trả lời sau từ "Answer:"
    answer = response.split("Answer:")[-1].strip()

    # Đảm bảo câu trả lời kết thúc hợp lý (thêm dấu câu nếu thiếu)
    if not answer.endswith(('.', '!', '?')):
        answer += '.'

    return answer

# Giao diện CLI đơn giản
if __name__ == "__main__":
    print("🤖 Chatbot Y Tế (gõ 'exit' để thoát)")
    while True:
        user_input = input("👤 Bạn: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Tạm biệt!")
            break
        reply = chat_with_model(user_input)
        print("🤖 Bác sĩ AI:", reply)
