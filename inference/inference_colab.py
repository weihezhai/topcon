"""
Inference script for paper acceptance prediction using fine-tuned model from Hugging Face.
Designed for Google Colab usage.

Usage in Colab:
1. Install dependencies:
   !pip install transformers torch accelerate

2. Run inference:
   from inference_colab import PaperAcceptancePredictor
   
   predictor = PaperAcceptancePredictor("your-huggingface-username/your-model-name")
   result = predictor.predict("Your paper abstract and introduction text here...")
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import List, Dict, Union, Optional
import numpy as np
from tqdm import tqdm

class PaperAcceptancePredictor:
    """
    A predictor for paper acceptance using a fine-tuned language model.
    """
    
    def __init__(
        self, 
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 10000,
        use_auth_token: Optional[str] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_name: Hugging Face model name (e.g., "username/model-name")
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for input
            use_auth_token: Hugging Face auth token for private models
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model '{model_name}' on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate dtype for memory efficiency
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_auth_token=use_auth_token,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Move to device if not using device_map
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Cache token IDs for yes/no
        self._cache_decision_tokens()
        
        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def _cache_decision_tokens(self):
        """Cache the token IDs for yes/no decisions."""
        yes_tokens = self.tokenizer(" yes", add_special_tokens=False)['input_ids']
        no_tokens = self.tokenizer(" no", add_special_tokens=False)['input_ids']
        
        if len(yes_tokens) > 0 and len(no_tokens) > 0:
            self.yes_token_id = yes_tokens[0]
            self.no_token_id = no_tokens[0]
        else:
            raise ValueError("Could not find token IDs for 'yes' and 'no'")
    
    def _create_prompt(self, text: str) -> str:
        """
        Create a prompt for the model.
        
        Args:
            text: Paper abstract and introduction text
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Paper content:\n{text}\n\nBased on this research paper's abstract and introduction, should this paper be accepted?\n\nDecision:"
        return prompt
    
    def predict_single(
        self, 
        text: str,
        return_probabilities: bool = False
    ) -> Dict[str, Union[str, float]]:
        """
        Predict acceptance for a single paper.
        
        Args:
            text: Paper abstract and introduction text
            return_probabilities: Whether to return probability scores
            
        Returns:
            Dictionary with prediction results
        """
        # Create prompt
        prompt = self._create_prompt(text)
        
        # Tokenize (truncate if necessary)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - 10,  # Reserve space for response
            return_attention_mask=True
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get logits for the next token position
            next_token_logits = outputs.logits[0, -1, :]
            
            # Get probabilities for yes/no tokens
            yes_logit = next_token_logits[self.yes_token_id].item()
            no_logit = next_token_logits[self.no_token_id].item()
            
            # Convert to probabilities using softmax
            yes_prob = torch.softmax(
                torch.tensor([yes_logit, no_logit]), 
                dim=0
            )[0].item()
            no_prob = 1 - yes_prob
            
            # Make decision
            decision = "accept" if yes_logit > no_logit else "reject"
            confidence = max(yes_prob, no_prob)
        
        result = {
            "decision": decision,
            "confidence": confidence
        }
        
        if return_probabilities:
            result["accept_probability"] = yes_prob
            result["reject_probability"] = no_prob
            result["yes_logit"] = yes_logit
            result["no_logit"] = no_logit
        
        return result
    
    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 4,
        return_probabilities: bool = False,
        show_progress: bool = True
    ) -> Union[Dict, List[Dict]]:
        """
        Predict acceptance for one or more papers.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing multiple texts
            return_probabilities: Whether to return probability scores
            show_progress: Whether to show progress bar
            
        Returns:
            Single result dict or list of result dicts
        """
        # Handle single text
        if isinstance(texts, str):
            return self.predict_single(texts, return_probabilities)
        
        # Handle batch of texts
        results = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing papers", total=len(texts)//batch_size + 1)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            # Process each text in batch
            for text in batch_texts:
                result = self.predict_single(text, return_probabilities)
                results.append(result)
            
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def predict_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        text_column: str = "text",
        batch_size: int = 4,
        return_probabilities: bool = True
    ) -> List[Dict]:
        """
        Predict from a JSON file containing papers.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to save results (optional)
            text_column: Name of the column containing text
            batch_size: Batch size for processing
            return_probabilities: Whether to return probability scores
            
        Returns:
            List of prediction results
        """
        import json
        
        # Load data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Extract texts
        if isinstance(data, list):
            texts = [item.get(text_column, item) if isinstance(item, dict) else item 
                    for item in data]
        elif isinstance(data, dict):
            texts = list(data.values())
        else:
            raise ValueError("Input file must contain a list or dictionary")
        
        print(f"Loaded {len(texts)} papers from {input_file}")
        
        # Make predictions
        results = self.predict(texts, batch_size, return_probabilities)
        
        # Add original data to results if available
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            for i, result in enumerate(results):
                result['original_data'] = data[i]
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results
    
    def interactive_demo(self):
        """
        Run an interactive demo for testing the model.
        """
        print("\n" + "="*60)
        print("Paper Acceptance Prediction - Interactive Demo")
        print("="*60)
        print("Enter paper abstract and introduction (press Enter twice to submit):")
        print("Type 'quit' to exit\n")
        
        while True:
            # Collect multi-line input
            lines = []
            while True:
                line = input()
                if line.lower() == 'quit':
                    print("Exiting demo...")
                    return
                if line == '' and len(lines) > 0 and lines[-1] == '':
                    break
                lines.append(line)
            
            text = '\n'.join(lines[:-1])  # Remove last empty line
            
            if len(text.strip()) == 0:
                print("Please enter some text or type 'quit' to exit.\n")
                continue
            
            # Make prediction
            print("\nAnalyzing paper...")
            result = self.predict_single(text, return_probabilities=True)
            
            # Display results
            print("\n" + "-"*40)
            print(f"Decision: {result['decision'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Accept Probability: {result['accept_probability']:.2%}")
            print(f"Reject Probability: {result['reject_probability']:.2%}")
            print("-"*40)
            print("\nEnter another paper or type 'quit' to exit:\n")


# Convenience functions for quick usage
def quick_predict(model_name: str, text: str, auth_token: Optional[str] = None) -> Dict:
    """
    Quick prediction function for single paper.
    
    Args:
        model_name: Hugging Face model name
        text: Paper text
        auth_token: Hugging Face auth token (optional)
        
    Returns:
        Prediction result dictionary
    """
    predictor = PaperAcceptancePredictor(model_name, use_auth_token=auth_token)
    return predictor.predict(text, return_probabilities=True)


# Example usage for Colab
def colab_example():
    """
    Example code for using in Google Colab.
    """
    example_code = '''
# Install dependencies (run once)
!pip install transformers torch accelerate tqdm

# Import the predictor
from inference_colab import PaperAcceptancePredictor

# Initialize predictor with your model
predictor = PaperAcceptancePredictor("your-username/your-model-name")

# Example 1: Single prediction
paper_text = """
Abstract: This paper presents a novel approach to...
Introduction: In recent years, deep learning has...
"""

result = predictor.predict(paper_text, return_probabilities=True)
print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")

# Example 2: Batch prediction
papers = [paper1_text, paper2_text, paper3_text]
results = predictor.predict(papers, batch_size=2)

# Example 3: Interactive demo
predictor.interactive_demo()

# Example 4: Predict from file
results = predictor.predict_from_file(
    "papers.json",
    output_file="predictions.json",
    text_column="abstract_and_intro"
)
'''
    print(example_code)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper acceptance prediction inference")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--text", type=str, help="Paper text for single prediction")
    parser.add_argument("--file", type=str, help="JSON file with papers for batch prediction")
    parser.add_argument("--output", type=str, help="Output file for predictions")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--auth_token", type=str, help="Hugging Face auth token")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PaperAcceptancePredictor(args.model, use_auth_token=args.auth_token)
    
    # Run appropriate mode
    if args.demo:
        predictor.interactive_demo()
    elif args.text:
        result = predictor.predict(args.text, return_probabilities=True)
        print(json.dumps(result, indent=2))
    elif args.file:
        results = predictor.predict_from_file(
            args.file,
            output_file=args.output,
            return_probabilities=True
        )
        print(f"Processed {len(results)} papers")
        
        # Print summary
        accepted = sum(1 for r in results if r['decision'] == 'accept')
        print(f"Accepted: {accepted}/{len(results)} ({accepted/len(results)*100:.1f}%)")
    else:
        print("Please specify --text, --file, or --demo")
        colab_example()
