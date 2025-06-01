import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Đường dẫn file CSV
csv_path = "/data/ai-medical-chatbot.csv"

# Đọc file CSV
df = pd.read_csv(csv_path)

# Cấu hình hiển thị để không bị rút gọn chuỗi
pd.set_option('display.max_colwidth', None)

# In ra 2 dòng đầy đủ: dòng đầu tiên và dòng thứ 10 (index 9)
print(df.iloc[0])
print("\n---\n")
print(df.iloc[9])




# 1. Đọc dữ liệu CSV
csv_path = "/data/ai-medical-chatbot.csv"
df = pd.read_csv(csv_path)

# 2. Chuẩn bị system prompt (có thể tùy chỉnh nếu muốn)
system_prompt = (
    "You are a helpful medical assistant who answers patient questions professionally and clearly."
)

# 3. Hàm làm sạch text: bỏ xuống dòng, dấu thừa, tránh lỗi encoding
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    # Có thể bổ sung thêm bước xóa dấu cách thừa, ký tự lạ nếu cần
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text

# 4. Tạo file huấn luyện LLaMA 3.2 định dạng chuẩn
output_file = "llama_training_data.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        patient = clean_text(row.get('Patient', ''))
        doctor = clean_text(row.get('Doctor', ''))

        if not patient or not doctor:
            # Bỏ qua mẫu không đủ dữ liệu
            continue

        # Định dạng theo chuẩn token đặc biệt LLaMA 3.2 1B
        formatted_text = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>"
            f"{system_prompt}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>"
            f"{patient}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
            f"{doctor}"
            "<|eot_id|>\n"
        )
        f.write(formatted_text)

print(f"Hoàn tất tạo file dữ liệu huấn luyện: {output_file}")




csv_path = "/data/ai-medical-chatbot.csv"
df = pd.read_csv(csv_path)

system_prompt = (
    "You are a helpful medical assistant who answers patient questions professionally and clearly."
)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace('\n', ' ').replace('\r', ' ').strip()
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text

# In ra 3 đoạn dữ liệu mẫu đã format chuẩn
count = 0
for idx, row in df.iterrows():
    patient = clean_text(row.get('Patient', ''))
    doctor = clean_text(row.get('Doctor', ''))

    if not patient or not doctor:
        continue

    formatted_text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>"
        f"{system_prompt}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>"
        f"{patient}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
        f"{doctor}"
        "<|eot_id|>\n"
    )
    print(f"--- Đoạn dữ liệu thứ {count+1} ---")
    print(formatted_text)
    print("="*50)
    count += 1
    if count >= 3:
        break




# 1. Tạo file token IDs đúng cách
def create_correct_tokenized_dataset():
    input_file = "/data/llama_training_data.txt"
    output_file = "/data/llama_training_data_token_ids_30k.txt"
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Đọc từng dòng (mỗi dòng là 1 mẫu hoàn chỉnh)
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Chỉ lấy 30000 dòng đầu
    lines = lines[:10000]
    
    # Tạo file token IDs
    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            # Mã hóa toàn bộ mẫu
            tokens = tokenizer.encode(line, add_special_tokens=False)
            f_out.write(" ".join(map(str, tokens)) + "\n")
            
            if i % 1000 == 0:
                print(f"Processed {i} samples")
    
    print(f"Saved token ids of {len(lines)} samples to {output_file}")
    return output_file

# 2. Tạo dataset đúng cách
class CompleteSampleDataset(Dataset):
    def __init__(self, token_list):
        self.samples = []
        for tokens in token_list:
            # Chỉ thêm các mẫu có độ dài hợp lý
            if len(tokens) > 10:
                self.samples.append(torch.tensor(tokens, dtype=torch.long))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# 3. Hàm kiểm tra dữ liệu
def verify_data_samples(tokenizer, val_dataset, num_samples=3):
    print("\n" + "="*50)
    print("VERIFYING DATA SAMPLES")
    print("="*50 + "\n")
    
    for i in range(num_samples):
        sample_idx = i % len(val_dataset)
        token_ids = val_dataset[sample_idx].tolist()
        
        # Giải mã toàn bộ mẫu
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        # Tách thành các phần
        parts = full_text.split("<|eot_id|>")
        
        # Đảm bảo có đủ 3 phần: system, user, assistant
        if len(parts) >= 3:
            prompt = "<|eot_id|>".join(parts[:2]) + "<|eot_id|>"
            actual_completion = parts[2].replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
            
            print(f"--- Sample {i+1} ---")
            print(f"Full text: {full_text[:500]}...")
            print(f"\nPrompt: {prompt[:500]}...")
            print(f"\nActual Response: {actual_completion[:500]}...")
            print("="*100)
        else:
            print(f"--- Sample {i+1} INVALID ---")
            print(f"Only {len(parts)} parts found")
            print(f"Content: {full_text[:500]}...")
            print("="*100)

# 4. Tải dữ liệu đã token hóa
def load_tokenized_data(file_path):
    tokenized_sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            if tokens:  # Bỏ qua dòng trống
                tokenized_sequences.append(tokens)
    return tokenized_sequences

# 5. Chia dữ liệu thành tập train và validation
def split_data(tokenized_sequences, test_size=0.1):
    train_data, val_data = train_test_split(tokenized_sequences, test_size=test_size, random_state=42)
    print(f"Train sequences: {len(train_data)}, Validation sequences: {len(val_data)}")
    return train_data, val_data

# 6. Chạy toàn bộ quy trình
def main():
    # Tạo file token IDs đúng cách
    token_ids_file = create_correct_tokenized_dataset()
    
    # Tải dữ liệu đã token hóa
    tokenized_sequences = load_tokenized_data(token_ids_file)
    
    # Chia dữ liệu
    train_data, val_data = split_data(tokenized_sequences)
    
    # Tạo datasets
    train_dataset = CompleteSampleDataset(train_data)
    val_dataset = CompleteSampleDataset(val_data)
    
    # Tải tokenizer để kiểm tra
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Kiểm tra dữ liệu
    verify_data_samples(tokenizer, val_dataset)
    
    return train_dataset, val_dataset, tokenizer

# Chạy chương trình chính
if __name__ == "__main__":
    train_dataset, val_dataset, tokenizer = main()