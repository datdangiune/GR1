from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from nltk.translate.bleu_score import sentence_bleu

# Hàm đánh giá BLEU score
def evaluate_bleu(predictions, references):
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        bleu_scores.append(sentence_bleu([ref.split()], pred.split()))
    return sum(bleu_scores) / len(bleu_scores)

# Khởi tạo tokenizer và model GPT-2 đã huấn luyện
model = GPT2LMHeadModel.from_pretrained("./gpt2-medical-model")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-medical-model")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 không có pad token

# Kiểm thử dự đoán mẫu và đánh giá
def test_model(sample_input, sample_output):
    input_ids = tokenizer.encode(sample_input, return_tensors='pt')

    # Dự đoán câu trả lời từ mô hình
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Giải mã đầu ra
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Input: {sample_input}")
    print(f"Generated Output: {generated_text}")
    
    # Đánh giá BLEU score
    bleu_score = evaluate_bleu([generated_text], [sample_output])
    print(f"BLEU Score: {bleu_score:.4f}\n")
    
    return generated_text, bleu_score

# Kiểm thử với một mẫu từ tập huấn luyện
sample_input = "What is Huntington's disease?"
sample_output = "Huntington's disease is a genetic disorder that causes progressive loss of mental functions, including memory, judgment, and speech."

test_model(sample_input, sample_output)

# Cho phép người dùng nhập câu hỏi
while True:
    user_input = input("Nhập câu hỏi (hoặc 'exit' để thoát): ")
    if user_input.lower() == 'exit':
        break
    
    # Dự đoán câu trả lời từ mô hình
    input_ids = tokenizer.encode(f"Question: {user_input}", return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Đọc và in kết quả
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Câu trả lời: {generated_text}\n")
