import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import time
import random

# Thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tạo thư mục output
os.makedirs('./output2', exist_ok=True)

# Load dataset
df = pd.read_csv('data/ai-medical-chatbot.csv')

# Hàm lọc các câu trả lời vô nghĩa
def is_meaningful_answer(answer):
    patterns = [
        "consult a doctor",
        "consult a physician",
        "consult.*specialist",
        "online consultation",
        "sexologist online",
        "for more info consult"
    ]
    answer = answer.lower()
    return not any(p in answer for p in patterns)

# Tiền xử lý
def preprocess_data(df):
    questions = df['Patient'].dropna().tolist()
    answers = df['Doctor'].dropna().tolist()
    data = [
        {'text': f"Question: {q.strip()} Answer: {a.strip()}"}
        for q, a in zip(questions, answers)
        if is_meaningful_answer(a)
    ]
    random.shuffle(data)
    return data[:15000]  # Tăng lên 20k mẫu meaningful

processed_data = preprocess_data(df)

# Dataset class
class MedicalChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encodings = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

# Load lại model đã huấn luyện
model_path = "./trained_model2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Fix pad token nếu thiếu
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

# Train/Val split
train_data, eval_data = train_test_split(processed_data, test_size=0.1)
train_dataset = MedicalChatDataset(train_data, tokenizer)
eval_dataset = MedicalChatDataset(eval_data, tokenizer)

# Logging thời gian
start_time = time.time()
print("🚀 Training bắt đầu lúc:", time.strftime('%Y-%m-%d %H:%M:%S'))

# Huấn luyện
training_args = TrainingArguments(
    output_dir='./results2',
    num_train_epochs=2,  # Ít epoch vì chỉ fine-tune tiếp
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=200,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",  # Có thể lưu theo từng epoch
    report_to="none"
)

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

end_time = time.time()
print("✅ Training kết thúc lúc:", time.strftime('%Y-%m-%d %H:%M:%S'))
print(f"🕒 Tổng thời gian huấn luyện: {end_time - start_time:.2f} giây")

# Lưu lại model fine-tune tiếp
model.save_pretrained('./trained_model2')
tokenizer.save_pretrained('./trained_model2')

# Lưu log ra CSV
log_history = trainer.state.log_history
log_df = pd.DataFrame(log_history)
log_df.to_csv('./output2/training_log.csv', index=False)

# Vẽ loss
plt.figure(figsize=(12, 6))
if 'loss' in log_df.columns:
    train_loss_df = log_df[log_df['loss'].notnull()][['step', 'loss']]
    plt.plot(train_loss_df['step'], train_loss_df['loss'], label='Training Loss', marker='o')

if 'eval_loss' in log_df.columns:
    eval_loss_df = log_df[log_df['eval_loss'].notnull()][['step', 'eval_loss']]
    plt.plot(eval_loss_df['step'], eval_loss_df['eval_loss'], label='Validation Loss', marker='x')

plt.title('Loss theo Step')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('./output2/loss_plot.png')
plt.show()
