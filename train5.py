# Cell 1: Cài đặt thư viện
#!pip install -q transformers accelerate evaluate datasets torchmetrics rouge_score pandas numpy tqdm matplotlib

# Cell 2: Import thư viện
import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import evaluate
from accelerate import Accelerator
import matplotlib.pyplot as plt
from datetime import datetime

# Cell 3: Kiểm tra GPU
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Khởi tạo Accelerator (hỗ trợ multi-GPU)
accelerator = Accelerator()
device = accelerator.device
print(f"Using device: {device}")

# Cell 4: Load dataset từ Kaggle
def load_and_combine_datasets():
    # MedQuad dataset
    medquad = pd.read_csv('data/medquad.csv')[['question', 'answer']].dropna()
    medquad = medquad[medquad['answer'].str.len() > 20]  # Lọc câu trả lời ngắn
    medquad = medquad.sample(frac=0.7, random_state=42)  # 70% dữ liệu
    
    # Medical Chatbot dataset
    chatbot = pd.read_csv('data/ai-medical-chatbot.csv')[['Patient', 'Doctor']].dropna()
    chatbot.columns = ['question', 'answer']
    chatbot = chatbot[chatbot['answer'].str.len() > 10]  # Lọc câu trả lời quá ngắn
    chatbot = chatbot.sample(frac=0.3, random_state=42)  # 30% dữ liệu
    
    # Kết hợp và xáo trộn
    combined = pd.concat([medquad, chatbot]).sample(frac=1, random_state=42)
    return combined

# Cell 5: Tiền xử lý dữ liệu
def preprocess_data(df):
    processed = []
    disclaimer = "IMPORTANT: This is not medical advice. Consult a doctor. "
    
    for _, row in tqdm(df.iterrows(), desc="Processing"):
        q = ' '.join(str(row['question']).strip().split()[:150])  # Giới hạn độ dài
        a = ' '.join(str(row['answer']).strip().split()[:300])
        
        # Format chuẩn cho model
        text = f"""### Role: You are an AI medical assistant.
### Question: {' '.join(q)}
### Answer: {disclaimer}{' '.join(a)}"""
        
        processed.append({'text': text})
    return processed

# Cell 6: Load và xử lý dữ liệu
print("📊 Loading data...")
df = load_and_combine_datasets()
processed_data = preprocess_data(df)
print(f"✅ Total samples: {len(processed_data)}")


# Cell 7: Tạo Dataset Class
class MedicalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()
        }

# Cell 8: Load Model & Tokenizer
model_name = "microsoft/BioGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Cấu hình Tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

# Cell 9: Chia tập train/val
train_data, val_data = train_test_split(processed_data, test_size=0.1, random_state=42)
train_dataset = MedicalDataset(train_data, tokenizer)
val_dataset = MedicalDataset(val_data, tokenizer)
print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

# Cell 10: Tính ROUGE score với xử lý từng batch nhỏ
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    batch_size = 8  # Giảm batch size khi tính toán metrics
    
    # Xử lý từng batch nhỏ để tránh tràn bộ nhớ
    rouge_results = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for i in range(0, len(preds), batch_size):
        batch_preds = preds[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        batch_preds = np.where(batch_preds != -100, batch_preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(batch_preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
        
        batch_results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        for k in rouge_results:
            rouge_results[k].append(batch_results[k])
    
    # Tính trung bình kết quả
    return {k: round(np.mean(v), 4) for k, v in rouge_results.items()}

# Cell 11: Training Arguments tối ưu
training_args = TrainingArguments(
    output_dir="./medical_chatbot",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Có thể điều chỉnh tùy GPU
    per_device_eval_batch_size=8,   # Batch size validation nhỏ hơn train
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps", 
    save_steps=2000,
    fp16=True,
    gradient_checkpointing=True,  # Bật để tiết kiệm bộ nhớ
    eval_accumulation_steps=2,   # Tích lũy khi evaluation
    load_best_model_at_end=True,
    report_to="none",
    seed=42
)

# Cell 12: Khởi tạo Trainer với xử lý bộ nhớ
from transformers import TrainerCallback
import torch

class MemorySaverCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[MemorySaverCallback()]  # Tự động giải phóng bộ nhớ
)

# Chuẩn bị cho GPU
model, trainer.optimizer = accelerator.prepare(model, trainer.optimizer)


# Cell 13: Bắt đầu training
print("🚀 Training started...")
start_time = time.time()
trainer.train()

# Cell 14: Lưu model
accelerator.wait_for_everyone()
model.save_pretrained("./medical_chatbot_final")
tokenizer.save_pretrained("./medical_chatbot_final")

# Log thời gian
training_time = time.time() - start_time
print(f"✅ Training completed in {training_time/3600:.2f} hours")