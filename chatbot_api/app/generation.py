# /app/generation.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    system_prompt: str, 
    user_input: str
) -> str:
    """
    Tạo ra phản hồi từ model dựa trên system prompt và input của người dùng.
    Hàm này tái sử dụng cấu trúc prompt chính xác như khi huấn luyện.
    """
    # 1. Định dạng prompt theo đúng template của Llama 3
    # Đây là bước cực kỳ quan trọng để đảm bảo model hoạt động đúng.
    # Cấu trúc: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|><|start_header_id|>user<|end_header_id|>...<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # Sử dụng apply_chat_template của tokenizer là cách an toàn và chuẩn xác nhất
    # Nó sẽ tự động thêm các token đặc biệt (BOS, EOS, EOT, etc.)
    # add_generation_prompt=True sẽ đảm bảo prompt kết thúc bằng `<|start_header_id|>assistant<|end_header_id|>`
    prompt_token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    # 2. Tạo token mới
    with torch.no_grad():
        output_tokens = model.generate(
            prompt_token_ids,
            max_new_tokens=512,  # Tăng giới hạn token cho câu trả lời dài hơn
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            # Llama 3 sử dụng nhiều token kết thúc, `<|eot_id|>` là token chính
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
        )

    # 3. Giải mã câu trả lời
    # output_tokens chứa cả prompt và phần được tạo ra, ta cần loại bỏ phần prompt
    num_prompt_tokens = prompt_token_ids.shape[1]
    generated_tokens = output_tokens[0][num_prompt_tokens:]
    
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response_text.strip()