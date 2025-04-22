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
model = GPT2LMHeadModel.from_pretrained("./checkpoint-24612")
tokenizer = GPT2Tokenizer.from_pretrained("./checkpoint-24612")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 không có pad token

# Hàm dự đoán và đánh giá một mẫu
def test_model(sample_input, sample_output):
    input_ids = tokenizer.encode(sample_input, return_tensors='pt')

    # Sinh câu trả lời với các tham số tránh lặp
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True  # Cho kết quả đa dạng hơn
    )
    
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

# Chatbox: cho phép người dùng nhập câu hỏi tự do
while True:
    user_input = input("Nhập câu hỏi (hoặc 'exit' để thoát): ")
    if user_input.lower() == 'exit':
        break
    
    input_ids = tokenizer.encode(f"Question: {user_input}", return_tensors='pt')

    # Sinh câu trả lời từ mô hình với cấu hình tương tự
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )
    
    # In kết quả
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Câu trả lời: {generated_text}\n")
