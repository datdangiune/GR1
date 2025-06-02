# /app/model_loader.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def load_model_and_tokenizer(model_path: str):
    """
    Tải model và tokenizer đã được fine-tune từ một đường dẫn cụ thể.
    """
    print(f"Bắt đầu tải model từ: {model_path}...")
    start_time = time.time()

    # Xác định kiểu dữ liệu để tải model (tương tự như trong code huấn luyện)
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if bf16_supported else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",  # Tự động phân phối model trên các thiết bị có sẵn (GPU/CPU)
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Đảm bảo model ở chế độ đánh giá (không tính toán gradient)
    model.eval()

    end_time = time.time()
    print(f"✅ Model và tokenizer đã được tải thành công sau {end_time - start_time:.2f} giây.")
    
    return model, tokenizer