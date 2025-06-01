from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 1. Tải model và tokenizer
checkpoint_dir = "./llama_finetuned"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Cải tiến hàm chat
def improved_chat_with_doctor(question):
    # Sử dụng prompt đơn giản hơn
    prompt = (
        "You are a helpful medical assistant. Answer clearly and professionally.\n\n"
        f"Patient: {question}\n"
        "Doctor:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    
    # Tạo response với cấu hình chặt chẽ hơn
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.5,  # Giảm nhiệt độ để ít sáng tạo hơn
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Giải mã và làm sạch response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.replace(prompt, "").strip()
    
    # Loại bỏ các phần không cần thiết
    stop_phrases = ["Patient:", "Doctor:", "<|endoftext|>"]
    for phrase in stop_phrases:
        response = response.split(phrase)[0]
    
    return response

# 3. Test lại với các câu hỏi
medical_questions = [
    "What are the common symptoms of diabetes?",
    "How can I prevent the flu?",
    "What should I do if I have a severe headache?",
    "Can you explain what hypertension is?",
    "What are the first aid steps for a burn?"
]

print("\n" + "="*50)
print("IMPROVED MEDICAL CHATBOT TEST")
print("="*50 + "\n")

for i, question in enumerate(medical_questions, 1):
    print(f"Question {i}: {question}")
    response = improved_chat_with_doctor(question)
    print(f"\nResponse:\n{response}\n")
    print("-"*80 + "\n")