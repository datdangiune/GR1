import time
import pandas as pd
from datasets import Dataset, load_metric
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def print_time(step_name, start_time):
    elapsed = time.time() - start_time
    print(f"[{step_name}] Hoàn thành sau: {pd.to_timedelta(elapsed, unit='s')}")

# [1] Load model đã fine-tune
start_time = time.time()
print("[1] Đang load model đã fine-tune...")

model_path = "./gpt2-medical-model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

print_time("[1] Load model", start_time)

# [2] Đánh giá mô hình trên 100 mẫu từ medquad
start_time = time.time()
print("[2] Đang đánh giá mô hình...")

# Đọc lại dữ liệu
df = pd.read_csv("data/medquad.csv").dropna(subset=["question", "answer"])
df = df.rename(columns={"question": "input", "answer": "output"})

# Lấy 100 mẫu test
eval_samples = df.sample(n=100, random_state=42)

# Load metric
bleu = load_metric("bleu")
rouge = load_metric("rouge")

references = []
predictions = []

for _, row in eval_samples.iterrows():
    prompt = f"Question: {row['input']}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    output_ids = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        do_sample=False
    )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_answer = generated.split("Answer:")[-1].strip()

    references.append([row["output"].split()])
    predictions.append(generated_answer.split())

# Tính điểm BLEU và ROUGE
bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=[" ".join(p) for p in predictions],
                            references=[" ".join(r[0]) for r in references],
                            rouge_types=["rouge1", "rouge2", "rougeL"])

print("\n=== ĐÁNH GIÁ MÔ HÌNH ===")
print("BLEU score:", bleu_score["bleu"])
for k, v in rouge_score.items():
    print(f"{k}: {v.mid.fmeasure:.4f}")

print_time("[2] Đánh giá", start_time)

# [3] Chat QnA
print("\n=== CHAT HỎI ĐÁP VỚI MÔ HÌNH ===")
while True:
    try:
        user_input = input("Bạn hỏi gì? (gõ 'exit' để thoát): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Tạm biệt 👋")
            break

        prompt = f"Question: {user_input}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        output_ids = model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_answer = generated.split("Answer:")[-1].strip()
        print(f"🧠 GPT trả lời: {generated_answer}\n")

    except KeyboardInterrupt:
        print("\nĐã dừng.")
        break
