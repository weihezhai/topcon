'''
./models/
├── base_model/          # Original Llama model cache
│   ├── config.json
│   ├── tokenizer.json
│   └── pytorch_model.bin
└── finetuned_model/     # Fine-tuned model with LoRA
    ├── adapter_config.json
    ├── adapter_model.bin
    └── tokenizer files
'''

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os

def load_finetuned_model(finetuned_model_path, base_model_path):
    """Load the fine-tuned model for inference"""
    print(f"Loading fine-tuned model from {finetuned_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=2,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, finetuned_model_path)
    model.eval()
    
    return model, tokenizer

def predict(text, model, tokenizer, max_length=1024):
    """Make prediction on a single text"""
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move inputs to the same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return {
        'prediction': 'accept' if predicted_class == 1 else 'reject',
        'confidence': confidence,
        'probabilities': {
            'reject': predictions[0][0].item(),
            'accept': predictions[0][1].item()
        }
    }

def batch_predict(texts, model, tokenizer, max_length=1024, batch_size=4):
    """Make predictions on multiple texts"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1)
            
            for j, pred_class in enumerate(predicted_classes):
                confidence = predictions[j][pred_class].item()
                results.append({
                    'prediction': 'accept' if pred_class.item() == 1 else 'reject',
                    'confidence': confidence,
                    'probabilities': {
                        'reject': predictions[j][0].item(),
                        'accept': predictions[j][1].item()
                    }
                })
    
    return results

def main():
    # Configuration
    FINETUNED_MODEL_PATH = "./models/finetuned_model"
    BASE_MODEL_PATH = "./models/base_model"
    
    # Check if fine-tuned model exists
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"Error: Fine-tuned model not found at {FINETUNED_MODEL_PATH}")
        print("Please run training first.")
        return
    
    # Load model
    model, tokenizer = load_finetuned_model(FINETUNED_MODEL_PATH, BASE_MODEL_PATH)
    print("Model loaded successfully!")
    
    # Example single prediction
    sample_text = """
    Your sample paper text here...
    This would be the content of a research paper.
    """
    
    result = predict(sample_text, model, tokenizer)
    print(f"Single prediction result: {result}")
    
    # Example batch prediction
    sample_texts = [
        "First paper text...",
        "Second paper text...",
        "Third paper text..."
    ]
    
    batch_results = batch_predict(sample_texts, model, tokenizer)
    print(f"Batch prediction results: {batch_results}")

if __name__ == "__main__":
    main()