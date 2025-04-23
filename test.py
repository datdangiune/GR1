# test.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

class MedicalChatbot:
    def __init__(self, model_path):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Load local model/tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model = self.accelerator.prepare(self.model)
        
    def generate_response(self, question, max_length=300):
        prompt = f"""### Role: Medical AI Assistant
### Safety: Do not provide diagnoses.
### Question: {question.strip()}
### Answer: IMPORTANT: Consult a healthcare professional."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=2.0,
            num_beams=3,
            early_stopping=True
        )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response.split("### Answer:")[-1].strip()

if __name__ == "__main__":
    chatbot = MedicalChatbot("./medical_chatbot_optimized/final_model")
    
    test_questions = [
        "What are common symptoms of COVID-19?",
        "How to relieve migraine pain?",
        "When should I worry about a headache?"
    ]
    
    print("🧪 Medical Chatbot Evaluation:")
    for question in test_questions:
        response = chatbot.generate_response(question)
        print(f"\n❓ {question}")
        print(f"🤖 {response}")
