import torch
import torch.nn as nn
from typing import Optional

class SoftTokenProjector(nn.Module):
    """Projects sentence embeddings to soft tokens."""
    
    def __init__(
        self,
        hidden_size: int,
        k_soft_tokens: int = 4,
        projection_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.k_soft_tokens = k_soft_tokens
        self.projection_dim = projection_dim or hidden_size
        
        # Projection layers
        self.input_proj = nn.Linear(hidden_size, self.projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)
        
        # Generate K soft tokens per sentence
        self.soft_token_generator = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.projection_dim * 2, hidden_size * k_soft_tokens)
        )
        
    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project sentence embeddings to soft tokens.
        
        Args:
            sentence_embeddings: [batch_size, num_selected, hidden_size]
            
        Returns:
            soft_tokens: [batch_size, num_selected * k_soft_tokens, hidden_size]
        """
        batch_size, num_selected, _ = sentence_embeddings.shape
        
        # Project and normalize
        x = self.input_proj(sentence_embeddings)  # [B, N, proj_dim]
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Generate K soft tokens per sentence
        soft_tokens = self.soft_token_generator(x)  # [B, N, hidden * K]
        
        # Reshape to separate soft tokens
        soft_tokens = soft_tokens.view(
            batch_size, 
            num_selected, 
            self.k_soft_tokens, 
            self.hidden_size
        )
        
        # Flatten selected sentences and soft tokens
        soft_tokens = soft_tokens.view(
            batch_size,
            num_selected * self.k_soft_tokens,
            self.hidden_size
        )
        
        return soft_tokens
