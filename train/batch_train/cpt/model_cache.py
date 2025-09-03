import os
import json
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ModelCache:
    """Handles caching and loading of models."""
    
    def __init__(self, cache_dir: str = "/mnt/parscratch/users/acr24wz/etu/topcon/models"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_model_cache_path(self, model_name: str) -> str:
        """Generate cache path for a specific model."""
        # Extract model name from HuggingFace path (e.g., "Qwen/Qwen3-4B" -> "qwen3_4B")
        model_suffix = model_name.replace("/", "_").lower()
        cache_path = os.path.join(self.cache_dir, model_suffix)
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    
    def is_model_cached(self, cache_path: str) -> bool:
        """Check if model is already cached."""
        config_file = os.path.join(cache_path, "config.json")
        model_file = os.path.join(cache_path, "model.safetensors")
        pytorch_model = os.path.join(cache_path, "pytorch_model.bin")
        
        # Check if config exists and at least one model file exists
        return os.path.exists(config_file) and (
            os.path.exists(model_file) or 
            os.path.exists(pytorch_model) or
            os.path.exists(os.path.join(cache_path, "model.safetensors.index.json"))
        )
    
    def download_and_cache_model(self, 
                                model_name: str, 
                                cache_path: str,
                                torch_dtype=None,
                                attn_implementation: str = "eager",
                                trust_remote_code: bool = True) -> AutoModelForCausalLM:
        """Download and save model to cache."""
        print(f"Downloading model {model_name} to {cache_path}...")
        
        # Download and save model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_path,
        )
        
        # Save model to the cache path
        model.save_pretrained(cache_path)
        print(f"Model saved to {cache_path}")
        
        return model
    
    def load_cached_model(self,
                         cache_path: str,
                         torch_dtype=None,
                         attn_implementation: str = "eager",
                         trust_remote_code: bool = True,
                         device_map: Optional[str] = None,
                         max_memory: Optional[dict] = None,
                         offload_folder: Optional[str] = None) -> AutoModelForCausalLM:
        """Load model from cache."""
        print(f"Loading cached model from {cache_path}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
        )
        
        print(f"Model loaded from cache")
        return model
    
    def load_or_download_model(self,
                              model_name: str,
                              torch_dtype=None,
                              attn_implementation: str = "eager",
                              trust_remote_code: bool = True,
                              device_map: Optional[str] = None,
                              max_memory: Optional[dict] = None,
                              offload_folder: Optional[str] = None,
                              use_finetuned: bool = False,
                              finetuned_path: Optional[str] = None) -> Tuple[AutoModelForCausalLM, str]:
        """Load model from cache or download if not cached."""
        
        # If using fine-tuned model for evaluation
        if use_finetuned and finetuned_path:
            if not os.path.exists(finetuned_path):
                raise FileNotFoundError(f"Fine-tuned model not found at {finetuned_path}. Please run training first.")
            
            print(f"Loading fine-tuned model from {finetuned_path}")
            model = self.load_cached_model(
                finetuned_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder
            )
            return model, finetuned_path
        
        # For base model
        cache_path = self.get_model_cache_path(model_name)
        
        if self.is_model_cached(cache_path):
            print(f"Using cached model from {cache_path}")
            model = self.load_cached_model(
                cache_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder
            )
        else:
            print(f"Model not found in cache. Downloading...")
            # First download without device_map for caching
            model = self.download_and_cache_model(
                model_name,
                cache_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code
            )
            
            # If device_map is needed, reload with it
            if device_map:
                del model  # Free memory
                model = self.load_cached_model(
                    cache_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation,
                    trust_remote_code=trust_remote_code,
                    device_map=device_map,
                    max_memory=max_memory,
                    offload_folder=offload_folder
                )
        
        return model, cache_path
    
    def load_or_download_tokenizer(self, 
                                  model_name: str,
                                  use_fast: bool = True,
                                  trust_remote_code: bool = True,
                                  use_finetuned: bool = False,
                                  finetuned_path: Optional[str] = None) -> AutoTokenizer:
        """Load tokenizer from cache or download if not cached."""
        
        # If using fine-tuned model for evaluation
        if use_finetuned and finetuned_path:
            if not os.path.exists(finetuned_path):
                raise FileNotFoundError(f"Fine-tuned model not found at {finetuned_path}")
            
            print(f"Loading tokenizer from fine-tuned model at {finetuned_path}")
            return AutoTokenizer.from_pretrained(
                finetuned_path,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code
            )
        
        # For base model
        cache_path = self.get_model_cache_path(model_name)
        
        # Check if tokenizer is cached
        tokenizer_config = os.path.join(cache_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config):
            print(f"Loading cached tokenizer from {cache_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                cache_path,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code
            )
        else:
            print(f"Downloading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code
            )
            # Save tokenizer to cache
            tokenizer.save_pretrained(cache_path)
            print(f"Tokenizer saved to {cache_path}")
        
        return tokenizer
