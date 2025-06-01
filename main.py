from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from torch.utils.data import Dataset
import torch
import time
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import pipeline, set_seed
import os



# 1. Đọc file token IDs
tokenized_sequences = []
with open('/data/llama_training_data_token_ids_30k.txt', 'r') as f:
    for line in f:
        tokens = list(map(int, line.strip().split()))
        if tokens:  # Bỏ qua dòng trống
            tokenized_sequences.append(tokens)

print(f"Đã đọc {len(tokenized_sequences)} sequences")

# 2. Chia dữ liệu: 90% train, 10% validation
train_data, val_data = train_test_split(tokenized_sequences, test_size=0.1, random_state=42)
print(f"Train sequences: {len(train_data)}, Validation sequences: {len(val_data)}")

# 3. Kiểm tra hỗ trợ bfloat16
bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"BF16 supported: {bf16_supported}")

# 4. Tải tokenizer và model
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Sử dụng EOS token làm pad token

# Xác định kiểu dữ liệu phù hợp
torch_dtype = torch.bfloat16 if bf16_supported else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)

# 5. Tạo Dataset - SỬA LẠI
class CompleteSampleDataset(Dataset):
    def __init__(self, sequences):
        self.samples = []
        for tokens in sequences:
            # Chỉ thêm các mẫu có độ dài hợp lý
            if len(tokens) > 10:
                self.samples.append(torch.tensor(tokens, dtype=torch.long))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# Tạo datasets
train_dataset = CompleteSampleDataset(train_data)
val_dataset = CompleteSampleDataset(val_data)



# 6. Data Collator - Xử lý padding tự động
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Sử dụng causal language modeling
)

# 7. Callback theo dõi thời gian
class TimeCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start
        print(f"⏱️ Epoch {state.epoch} hoàn thành trong {epoch_time/60:.2f} phút")
        
# Hàm kiểm tra dữ liệu
def verify_data_samples(tokenizer, dataset, num_samples=3):
    print("\n" + "="*50)
    print("VERIFYING DATA SAMPLES")
    print("="*50 + "\n")
    
    for i in range(min(num_samples, len(dataset))):
        token_ids = dataset[i].tolist()
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        parts = full_text.split("<|eot_id|>")
        
        if len(parts) >= 3:
            prompt = "<|eot_id|>".join(parts[:2]) + "<|eot_id|>"
            actual_response = parts[2].replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
            
            print(f"--- Sample {i+1} ---")
            print(f"Full text: {full_text[:200]}...")
            print(f"\nPrompt: {prompt[:200]}...")
            print(f"\nActual Response: {actual_response[:200]}...")
            print("="*100)
        else:
            print(f"--- Sample {i+1} INVALID ---")
            print(f"Only {len(parts)} parts found")
            print(f"Content: {full_text[:200]}...")
            print("="*100)

# Kiểm tra dữ liệu trước khi huấn luyện
verify_data_samples(tokenizer, train_dataset)
verify_data_samples(tokenizer, val_dataset)

# 8. Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,             # Số epoch huấn luyện
    per_device_train_batch_size=16,   # Batch size cho training
    per_device_eval_batch_size=16,    # Batch size cho evaluation
    gradient_accumulation_steps=4,   # Tích lũy gradient để tăng effective batch size
    learning_rate=2e-5,              # Tốc độ học
    weight_decay=0.01,               # Trọng số giảm
    eval_strategy="steps",      # Đánh giá sau mỗi số bước
    eval_steps=500,                  # Số bước giữa các lần đánh giá
    save_strategy="steps",            # Lưu checkpoint sau mỗi số bước
    save_steps=50000,                 # Số bước giữa các lần lưu
    logging_steps=50,                # Log sau mỗi 50 bước
    fp16=not bf16_supported,         # Sử dụng float16 nếu không hỗ trợ bfloat16
    bf16=bf16_supported,             # Sử dụng bfloat16 nếu được hỗ trợ
    optim="adamw_torch_fused",       # Sử dụng phiên bản fused optimizer
    report_to="none",                # Không gửi báo cáo
    warmup_steps=100,                # Số bước warmup
    logging_dir="./logs",            # Thư mục log
    load_best_model_at_end=True,     # Tải model tốt nhất cuối cùng
    greater_is_better=False,         # Loss nhỏ hơn là tốt hơn
    metric_for_best_model="eval_loss",
    save_total_limit=1,              # Chỉ giữ 1 checkpoint
    gradient_checkpointing=True,     # Kích hoạt gradient checkpointing
)

# 9. Khởi tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[TimeCallback()],
)

# 10. Huấn luyện
print("🚀 Bắt đầu huấn luyện...")
start_time = time.time()
trainer.train()
end_time = time.time()

# 11. Lưu model
trainer.save_model("./llama_finetuned")
tokenizer.save_pretrained("./llama_finetuned")
print(f"✅ Huấn luyện hoàn tất! Tổng thời gian: {(end_time - start_time)/60:.2f} phút")





# 1. Trực quan hóa lịch sử huấn luyện
def plot_training_history(log_history):
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []
    
    for log in log_history:
        if 'loss' in log:
            train_losses.append(log['loss'])
            train_steps.append(log['step'])
        if 'eval_loss' in log:
            eval_losses.append(log['eval_loss'])
            eval_steps.append(log['step'])
    
    plt.figure(figsize=(12, 6))
    
    # Biểu đồ train loss
    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot(train_steps, train_losses, 'b-', label='Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Steps')
        plt.grid(True)
        plt.legend()
    
    # Biểu đồ eval loss
    plt.subplot(1, 2, 2)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, 'r-', label='Validation Loss')
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Loss')
        plt.title('Validation Loss Over Steps')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# 2. Tính perplexity
def calculate_perplexity(eval_loss):
    perplexity = np.exp(eval_loss)
    print(f"Validation Loss: {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    # Lưu kết quả vào file
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Final Validation Loss: {eval_loss:.4f}\n")
        f.write(f"Perplexity: {perplexity:.4f}\n")
    
    return perplexity

# 3. Tạo các dự đoán mẫu
def generate_sample_predictions(model, tokenizer, val_dataset, num_samples=3):
    set_seed(42)  # Đảm bảo kết quả tái lặp
    
    # Tạo pipeline sinh văn bản
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    # Chọn ngẫu nhiên các mẫu từ tập validation
    sample_indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    results = []
    for i, idx in enumerate(sample_indices):
        # Lấy input_ids từ tập validation
        input_ids = val_dataset[idx].tolist()
        
        # Chia thành prompt và completion thực tế
        split_point = len(input_ids) // 2
        prompt_ids = input_ids[:split_point]
        actual_completion_ids = input_ids[split_point:]
        
        # Giải mã prompt
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        
        # Sinh văn bản từ mô hình
        generated = text_generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        generated_text = generated[0]['generated_text']
        
        # Chỉ lấy phần sinh ra (bỏ qua prompt)
        generated_completion = generated_text[len(prompt):].strip()
        
        # Giải mã completion thực tế
        actual_completion = tokenizer.decode(actual_completion_ids, skip_special_tokens=True)
        
        results.append({
            "prompt": prompt,
            "actual": actual_completion,
            "generated": generated_completion
        })
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"\nActual Response: {actual_completion}")
        print(f"\nGenerated Response: {generated_completion}")
        print("=" * 100)
    
    return results

# 4. Tính toán độ chính xác cơ bản
def calculate_accuracy(generated_samples):
    correct = 0
    for sample in generated_samples:
        # So sánh cơ bản - trong thực tế cần phương pháp phức tạp hơn
        if sample['actual'].lower() in sample['generated'].lower():
            correct += 1
    
    accuracy = correct / len(generated_samples)
    print(f"\nBasic Accuracy: {accuracy:.2%} ({correct}/{len(generated_samples)})")
    
    # Thêm vào file kết quả
    with open("evaluation_results.txt", "a") as f:
        f.write(f"\nBasic Accuracy: {accuracy:.2%}\n")
    
    return accuracy

# 5. Chạy toàn bộ quy trình đánh giá
def evaluate_model(model, tokenizer, val_dataset, trainer=None):
    print("\n" + "="*50)
    print("Starting Model Evaluation")
    print("="*50 + "\n")
    
    eval_results = {}
    
    # 5.1. Vẽ biểu đồ lịch sử huấn luyện (nếu có trainer)
    if trainer is not None:
        print("Visualizing training history...")
        plot_training_history(trainer.state.log_history)
    
    # 5.2. Tính perplexity (nếu có trainer)
    if trainer is not None:
        print("\nCalculating perplexity using trainer...")
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics['eval_loss']
        perplexity = calculate_perplexity(eval_loss)
        eval_results['eval_loss'] = eval_loss
        eval_results['perplexity'] = perplexity
    else:
        print("\nCalculating perplexity manually...")
        # Tính toán thủ công nếu không có trainer
        eval_loss = calculate_manual_eval_loss(model, val_dataset)
        if eval_loss is not None:
            perplexity = calculate_perplexity(eval_loss)
            eval_results['eval_loss'] = eval_loss
            eval_results['perplexity'] = perplexity
    
    # 5.3. Tạo các dự đoán mẫu
    print("\nGenerating sample predictions...")
    generated_samples = generate_sample_predictions(model, tokenizer, val_dataset)
    
    # 5.4. Tính toán độ chính xác cơ bản
    print("\nCalculating basic accuracy...")
    accuracy = calculate_accuracy(generated_samples)
    eval_results['accuracy'] = accuracy
    eval_results['samples'] = generated_samples
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50 + "\n")
    
    return eval_results

# Hàm tính eval_loss thủ công (nếu không có trainer)
def calculate_manual_eval_loss(model, val_dataset, batch_size=4):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling
    
    try:
        # Chuẩn bị data collator
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Tạo data loader
        eval_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False
        )
        
        # Tính toán loss
        model.eval()
        total_loss = 0
        total_batches = 0
        
        for batch in eval_loader:
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            total_batches += 1
        
        avg_loss = total_loss / total_batches
        return avg_loss
    
    except Exception as e:
        print(f"Error calculating manual eval loss: {e}")
        return None

# 6. Chạy đánh giá
# Kiểm tra xem trainer có tồn tại không
try:
    trainer_exists = trainer is not None
except NameError:
    trainer_exists = False

# Chạy đánh giá
eval_results = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    val_dataset=val_dataset,
    trainer=trainer if trainer_exists else None
)

# 7. Hiển thị kết quả tổng quan
print("\nFinal Evaluation Summary:")
if 'eval_loss' in eval_results:
    print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"Perplexity: {eval_results['perplexity']:.4f}")
print(f"Basic Accuracy: {eval_results['accuracy']:.2%}")

# Lưu các mẫu vào file
with open("sample_predictions.txt", "w") as f:
    for i, sample in enumerate(eval_results['samples']):
        f.write(f"--- Sample {i+1} ---\n")
        f.write(f"Prompt: {sample['prompt']}\n")
        f.write(f"Actual Response: {sample['actual']}\n")
        f.write(f"Generated Response: {sample['generated']}\n")
        f.write("=" * 100 + "\n\n")

print("\nSample predictions saved in sample_predictions.txt")