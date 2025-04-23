import os
import re
import json
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)

from accelerate import Accelerator
import evaluate
from evaluate import load as load_metric
from accelerate.state import AcceleratorState


# ========== CONFIGURATION ==========
class Config:
    MODEL_NAME = "microsoft/BioGPT"
    DATASETS = {
        "medquad": "data/medquad.csv",
        "medical_chatbot": "data/ai-medical-chatbot.csv" 
    }
    OUTPUT_DIR = "./medical_chatbot_optimized"
    MAX_LENGTH = 512
    TRAIN_BATCH_SIZE = 6  # Optimized for 16GB VRAM
    EVAL_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 4
    WARMUP_RATIO = 0.1
    SEED = 42

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ========== DATA PROCESSING ==========
def clean_medical_text(text):
    """Advanced cleaning for medical text"""
    text = str(text).strip()
    # Keep essential medical characters (/, -, .)
    text = re.sub(r'[^\w\s\/\-.,]', '', text)  
    # Normalize whitespace
    text = ' '.join(text.split())
    return text[:500]  # Truncate long texts

def load_and_preprocess_data():
    """Load and preprocess datasets with medical safety checks"""
    # Load datasets
    medquad = pd.read_csv(Config.DATASETS["medquad"])[["question", "answer"]].dropna()
    chatbot = pd.read_csv(Config.DATASETS["medical_chatbot"])[["Patient", "Doctor"]].dropna()
    chatbot.columns = ["question", "answer"]
    
    # Combine and shuffle
    df = pd.concat([medquad, chatbot]).sample(frac=1, random_state=Config.SEED)
    
    # Filter and clean
    processed_data = []
    disclaimer = "IMPORTANT: Consult a healthcare professional. "
    sensitive_keywords = ["diagnose", "prescribe", "treatment for"]
    
    for _, row in tqdm(df.iterrows(), desc="Processing"):
        q = clean_medical_text(row["question"])
        a = clean_medical_text(row["answer"])
        
        # Skip sensitive or low-quality entries
        if (len(a) < 15 or 
            any(keyword in q.lower() for keyword in sensitive_keywords)):
            continue
            
        processed_data.append({
            "text": f"""### Role: Medical AI Assistant
### Safety: Do not provide diagnoses or treatment plans.
### Question: {q}
### Answer: {disclaimer}{a}"""
        })
    
    return processed_data

# ========== DATASET CLASS ==========
class MedicalDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone()
        }

# ========== METRICS ==========
def setup_metrics():
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    def compute_medical_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Basic metrics
        rouge_results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Medical term accuracy (simplified)
        medical_terms = ["symptom", "treatment", "diagnosis"]
        med_term_acc = np.mean([
            any(term in pred.lower() for term in medical_terms)
            for pred in decoded_preds
        ])
        
        return {
            "rougeL": round(rouge_results["rougeL"], 4),
            "medical_term_acc": round(med_term_acc, 4),
            "length_ratio": round(np.mean([len(p)/len(l) for p,l in zip(decoded_preds, decoded_labels)]), 4)
        }
    
    return compute_medical_metrics

# ========== TRAINING SETUP ==========

def initialize_training():
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")
    print(f"Number of processes: {accelerator.num_processes}")
    
    # Load data
    all_data = load_and_preprocess_data()
    train_data, eval_data = train_test_split(
        all_data, test_size=0.1, random_state=Config.SEED
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Datasets
    train_dataset = MedicalDataset(train_data, tokenizer)
    eval_dataset = MedicalDataset(eval_data, tokenizer)
    
    # Prepare everything with Accelerator
    model, tokenizer, train_dataset, eval_dataset = accelerator.prepare(
        model, tokenizer, train_dataset, eval_dataset
    )
    
    return accelerator, model, tokenizer, train_dataset, eval_dataset


# ========== MAIN TRAINING ==========

def train():
    # Initialize
    accelerator, model, tokenizer, train_dataset, eval_dataset = initialize_training()
    compute_metrics = setup_metrics()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=Config.WARMUP_RATIO,
        logging_dir=f"{Config.OUTPUT_DIR}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        report_to="none",
        seed=Config.SEED,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=False
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Scheduler
    num_training_steps = len(train_dataset) * Config.NUM_EPOCHS // (Config.TRAIN_BATCH_SIZE * accelerator.num_processes)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(Config.WARMUP_RATIO * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )
    
    # Accelerate preparation
    model, optimizer, train_dataset = accelerator.prepare(
        model, optimizer, train_dataset
    )
    
    # Training
    print("🚀 Starting training...")
    start_time = time.time()
    trainer.train()
    
    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(f"{Config.OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{Config.OUTPUT_DIR}/final_model")
    
    # Save metadata
    metadata = {
        "training_time": (time.time() - start_time)/3600,
        "config": vars(Config),
        "best_metrics": trainer.state.best_metric
    }
    with open(f"{Config.OUTPUT_DIR}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Training completed in {metadata['training_time']:.2f} hours")


if __name__ == "__main__":
    train()