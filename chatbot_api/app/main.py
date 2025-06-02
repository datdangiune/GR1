# /app/main.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch

from .model_loader import load_model_and_tokenizer
from .generation import generate_response

# --- Cấu hình ---
# Đường dẫn đến model đã được fine-tune của bạn
MODEL_PATH = "./model" 

# System prompt phải giống với prompt đã dùng khi huấn luyện
SYSTEM_PROMPT = "You are a helpful medical assistant who answers patient questions professionally and clearly."

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="Medical Chatbot API",
    description="API for a Llama 3-based medical chatbot, fine-tuned on Q&A data.",
    version="1.0.0"
)

# --- Định nghĩa model cho Request và Response ---
class ChatRequest(BaseModel):
    question: str
    
class ChatResponse(BaseModel):
    answer: str

# --- Tải model khi khởi động ứng dụng ---
# Sử dụng sự kiện "startup" của FastAPI để chỉ tải model một lần duy nhất.
# Điều này giúp tránh việc tải lại model với mỗi request, tiết kiệm thời gian và bộ nhớ.
@app.on_event("startup")
def startup_event():
    print("🚀 Server is starting up, loading AI model...")
    if not torch.cuda.is_available():
        print("⚠️ WARNING: No CUDA-compatible GPU found. Model will run on CPU, which will be very slow.")
    
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    # Lưu model và tokenizer vào state của ứng dụng để có thể truy cập từ các endpoint
    app.state.model = model
    app.state.tokenizer = tokenizer
    print("✅ AI Model loaded and ready.")

# --- Định nghĩa API Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    """Endpoint để kiểm tra xem API có đang hoạt động không."""
    return {"status": "ok", "message": "Medical Chatbot API is running."}

@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_endpoint(request: ChatRequest):
    """
    Nhận câu hỏi từ người dùng và trả về câu trả lời từ chatbot.
    """
    try:
        # Lấy model và tokenizer đã được tải từ state của ứng dụng
        model = app.state.model
        tokenizer = app.state.tokenizer

        print(f"Received question: {request.question}")

        # Gọi hàm tạo văn bản
        answer = generate_response(
            model=model,
            tokenizer=tokenizer,
            system_prompt=SYSTEM_PROMPT,
            user_input=request.question
        )
        
        print(f"Generated answer: {answer}")
        
        return ChatResponse(answer=answer)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))