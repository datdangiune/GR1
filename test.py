import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Sử dụng GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn tới mô hình đã huấn luyện
MODEL_PATH = "./trained_model2"

# Load tokenizer và mô hình
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# Prompt hướng dẫn chuyên biệt
INSTRUCTION = (
    "You are a helpful, concise medical assistant. You specialize in diagnosing conditions "
    "based on user-described symptoms. Always try to guess possible conditions and give advice "
    "on what to do next. Avoid repeating or unrelated information. Answer clearly and professionally.\n"
)

# Hàm để hỏi mô hình
def chat_with_model(question, max_new_tokens=256, temperature=0.7, top_p=0.9):
    prompt = f"{INSTRUCTION}Patient: {question.strip()}\nDoctor:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.get("attention_mask")
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Chỉ giữ phần trả lời của bác sĩ sau "Doctor:"
    if "Doctor:" in response:
        answer = response.split("Doctor:")[-1].strip()
    else:
        answer = response.strip()

    # Kết thúc bằng dấu chấm nếu thiếu
    if not answer.endswith(('.', '!', '?')):
        answer += '.'

    return answer

# Giao diện dòng lệnh
if __name__ == "__main__":
    print("🤖 Bác sĩ AI sẵn sàng (gõ 'exit' để thoát)")
    while True:
        user_input = input("👤 Bạn: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Tạm biệt!")
            break
        reply = chat_with_model(user_input)
        print("🤖 Bác sĩ AI:", reply)
