from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical sentence-based training."""
    
    # Sentence chunking
    max_sentences: int = 128
    max_sentence_length: int = 64
    min_sentence_length: int = 3
    
    # Sentence selection
    max_selected_sentences: int = 32  # M
    k_soft_tokens_per_chunk: int = 4  # K
    
    # Prompts
    prompt_prefix: str = "Analyze the following scientific paper:\n"
    prompt_suffix: str = "\nBased on the content, should this paper be accepted? Answer yes or no.\nDecision:"
    
    # Training strategy
    freeze_backbone: bool = True
    num_unfrozen_layers: int = 2
    use_gradient_checkpointing: bool = True
    
    # Optimization
    learning_rate_scorer: float = 1e-3  # Higher LR for new heads
    learning_rate_backbone: float = 1e-5  # Lower LR for pretrained
    warmup_ratio: float = 0.1
    
    # Memory optimization
    gradient_accumulation_steps: int = 16
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    
    # Reproducibility
    seed: int = 42
    
    def total_soft_tokens(self) -> int:
        """Calculate total number of soft tokens."""
        return self.max_selected_sentences * self.k_soft_tokens_per_chunk
    
    def log_config(self):
        """Print configuration summary."""
        print("="*80)
        print("HIERARCHICAL TRAINING CONFIGURATION")
        print("="*80)
        print(f"Sentence Processing:")
        print(f"  Max sentences per doc: {self.max_sentences}")
        print(f"  Max tokens per sentence: {self.max_sentence_length}")
        print(f"  Min words per sentence: {self.min_sentence_length}")
        print(f"\nSelection & Projection:")
        print(f"  Max selected sentences (M): {self.max_selected_sentences}")
        print(f"  Soft tokens per sentence (K): {self.k_soft_tokens_per_chunk}")
        print(f"  Total soft tokens: {self.total_soft_tokens()}")
        print(f"\nTraining Strategy:")
        print(f"  Freeze backbone: {self.freeze_backbone}")
        print(f"  Unfrozen layers: {self.num_unfrozen_layers}")
        print(f"  Gradient checkpointing: {self.use_gradient_checkpointing}")
        print(f"\nOptimization:")
        print(f"  LR (scorer): {self.learning_rate_scorer}")
        print(f"  LR (backbone): {self.learning_rate_backbone}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.gradient_accumulation_steps * self.per_device_train_batch_size}")
        print("="*80)
