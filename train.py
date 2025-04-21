import time
import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def print_time(step_name, start_time):
    elapsed = time.time() - start_time
    print(f"[{step_name}] Hoàn thành sau: {pd.to_timedelta(elapsed, unit='s')}")

# [1] Load dữ liệu
start_time = time.time()
print("[1] Đang load dữ liệu...")

medquad_df = pd.read_csv("data/medquad.csv").dropna(subset=["question", "answer"])
chatbot_train_df = pd.read_csv("data/train_data_chatbot.csv").dropna(subset=["short_question", "short_answer"])
chatbot_val_df = pd.read_csv("data/validation_data_chatbot.csv").dropna(subset=["short_question", "short_answer"])

medquad_df = medquad_df.rename(columns={"question": "input", "answer": "output"})
chatbot_train_df = chatbot_train_df.rename(columns={"short_question": "input", "short_answer": "output"})
chatbot_val_df = chatbot_val_df.rename(columns={"short_question": "input", "short_answer": "output"})

train_df = pd.concat([medquad_df, chatbot_train_df], ignore_index=True)
val_df = chatbot_val_df.copy()

print_time("1] Load dữ liệu", start_time)

# [2] Chuẩn bị Dataset
start_time = time.time()
print("[2] Đang chuẩn bị Dataset...")

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

print_time("2] Chuẩn bị Dataset", start_time)

# [3] Tokenizing
start_time = time.time()
print("[3] Tokenizing...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 không có pad token

def tokenize(example):
    if example['input'] is None or example['output'] is None:
        return tokenizer("")
    prompt = f"Question: {example['input']}\nAnswer: {example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

train_tokenized = train_ds.map(tokenize, batched=False)
val_tokenized = val_ds.map(tokenize, batched=False)

print_time("3] Tokenization", start_time)

# [4] Fine-tune GPT-2
start_time = time.time()
print("[4] Huấn luyện mô hình GPT-2...")

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-medical-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

print_time("4] Huấn luyện GPT-2", start_time)

# [5] Lưu mô hình sau huấn luyện
model.save_pretrained("output_model/")
tokenizer.save_pretrained("output_model/")
print("[5] Đã lưu mô hình vào thư mục 'output_model/'")

# [6] Test model với một câu hỏi mẫu
print("[6] Test câu hỏi mẫu:")
sample_input = "What causes high blood pressure?"
prompt = f"Question: {sample_input}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    pad_token_id=tokenizer.eos_token_id
)
decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded)
