import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from soft_token_projector import SoftTokenProjector

class HierarchicalPromptModel(nn.Module):
    """Hierarchical model with sentence selection and soft token generation."""
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_selected_sentences: int = 32,
        k_soft_tokens_per_chunk: int = 4,
        prompt_prefix: str = "Analyze the following scientific paper:\n",
        prompt_suffix: str = "\nBased on the content, should this paper be accepted? Answer yes or no.\nDecision:",
        freeze_backbone: bool = True,
        num_unfrozen_layers: int = 2
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.max_selected_sentences = max_selected_sentences
        self.k_soft_tokens_per_chunk = k_soft_tokens_per_chunk
        
        # Get model config
        self.config = base_model.config
        self.hidden_size = self.config.hidden_size
        
        # Tokenize prompts
        self.prompt_prefix_ids = tokenizer(
            prompt_prefix, 
            add_special_tokens=False, 
            return_tensors='pt'
        )['input_ids']
        
        self.prompt_suffix_ids = tokenizer(
            prompt_suffix,
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids']
        
        # Get yes/no token IDs
        self.yes_token_id = tokenizer(" yes", add_special_tokens=False)['input_ids'][0]
        self.no_token_id = tokenizer(" no", add_special_tokens=False)['input_ids'][0]
        
        # Sentence scoring head
        self.sentence_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Position embeddings for sentences
        self.sentence_position_embeddings = nn.Embedding(
            256,  # Max position
            self.hidden_size
        )
        
        # Soft token projector
        self.soft_token_projector = SoftTokenProjector(
            hidden_size=self.hidden_size,
            k_soft_tokens=k_soft_tokens_per_chunk,
            dropout=0.1
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone(num_unfrozen_layers)
    
    def _freeze_backbone(self, num_unfrozen_layers: int = 2):
        """Freeze the backbone model except for top layers."""
        # Freeze all parameters first
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze top transformer layers
        if hasattr(self.base_model, 'model'):  # For models like Qwen
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            return  # Can't identify layer structure
        
        # Unfreeze last N layers
        for layer in layers[-num_unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Always unfreeze the LM head
        if hasattr(self.base_model, 'lm_head'):
            for param in self.base_model.lm_head.parameters():
                param.requires_grad = True
    
    def encode_sentences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all sentences using the base model.
        
        Args:
            input_ids: [batch_size, num_sentences, sentence_length]
            attention_mask: [batch_size, num_sentences, sentence_length]
            chunk_mask: [batch_size, num_sentences]
            
        Returns:
            sentence_embeddings: [batch_size, num_sentences, hidden_size]
            sentence_mask: [batch_size, num_sentences]
        """
        batch_size, num_sentences, seq_len = input_ids.shape
        
        # Flatten for batch processing
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        
        # Encode with base model (get hidden states)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.base_model.model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get last hidden states
        hidden_states = outputs.last_hidden_state  # [batch*sentences, seq_len, hidden]
        
        # Reshape back
        hidden_states = hidden_states.view(batch_size, num_sentences, seq_len, self.hidden_size)
        
        # Pool over tokens (mean pooling with mask)
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * attention_mask_expanded).sum(dim=2)
        sum_mask = attention_mask_expanded.sum(dim=2).clamp(min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask  # [batch, sentences, hidden]
        
        return sentence_embeddings, chunk_mask
    
    def select_top_sentences(
        self,
        sentence_embeddings: torch.Tensor,
        sentence_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score and select top-M sentences.
        
        Args:
            sentence_embeddings: [batch_size, num_sentences, hidden_size]
            sentence_mask: [batch_size, num_sentences]
            
        Returns:
            selected_embeddings: [batch_size, max_selected, hidden_size]
            selected_indices: [batch_size, max_selected]
        """
        batch_size, num_sentences, _ = sentence_embeddings.shape
        
        # Add position embeddings
        positions = torch.arange(num_sentences, device=sentence_embeddings.device)
        position_embeds = self.sentence_position_embeddings(positions)
        sentence_embeddings = sentence_embeddings + position_embeds.unsqueeze(0)
        
        # Score sentences
        scores = self.sentence_scorer(sentence_embeddings).squeeze(-1)  # [batch, sentences]
        
        # Mask invalid sentences
        scores = scores.masked_fill(~sentence_mask, float('-inf'))
        
        # Select top-M
        M = min(self.max_selected_sentences, num_sentences)
        top_scores, top_indices = torch.topk(scores, M, dim=1, sorted=True)
        
        # Gather selected embeddings
        selected_embeddings = torch.gather(
            sentence_embeddings,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        
        return selected_embeddings, top_indices
    
    def build_final_sequence(
        self,
        soft_tokens: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the final embedded sequence with prompts and target.
        
        Args:
            soft_tokens: [batch_size, num_soft_tokens, hidden_size]
            labels: [batch_size] (binary labels)
            batch_size: int
            
        Returns:
            embedded_sequence: [batch_size, total_length, hidden_size]
            attention_mask: [batch_size, total_length]
            lm_labels: [batch_size, total_length]
        """
        device = soft_tokens.device
        
        # Get embeddings for prefix and suffix
        prefix_ids = self.prompt_prefix_ids.to(device).expand(batch_size, -1)
        suffix_ids = self.prompt_suffix_ids.to(device).expand(batch_size, -1)
        
        # Get embedding matrix
        embed_matrix = self.base_model.get_input_embeddings()
        
        prefix_embeds = embed_matrix(prefix_ids)
        suffix_embeds = embed_matrix(suffix_ids)
        
        # Get target token embeddings based on labels
        target_token_ids = torch.where(
            labels == 1,
            torch.tensor(self.yes_token_id, device=device),
            torch.tensor(self.no_token_id, device=device)
        )
        target_embeds = embed_matrix(target_token_ids).unsqueeze(1)  # [batch, 1, hidden]
        
        # Concatenate all parts
        embedded_sequence = torch.cat([
            prefix_embeds,
            soft_tokens,
            suffix_embeds,
            target_embeds
        ], dim=1)
        
        # Create attention mask (all 1s)
        total_length = embedded_sequence.shape[1]
        attention_mask = torch.ones(batch_size, total_length, device=device)
        
        # Create labels for LM loss
        lm_labels = torch.full(
            (batch_size, total_length),
            -100,
            dtype=torch.long,
            device=device
        )
        # Only supervise the last token
        lm_labels[:, -1] = target_token_ids
        
        return embedded_sequence, attention_mask, lm_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the hierarchical model.
        
        Args:
            input_ids: [batch_size, num_sentences, sentence_length]
            attention_mask: [batch_size, num_sentences, sentence_length]
            chunk_mask: [batch_size, num_sentences]
            labels: [batch_size] (binary labels)
            
        Returns:
            CausalLMOutputWithPast with loss and logits
        """
        batch_size = input_ids.shape[0]
        
        # Step 1: Encode all sentences
        sentence_embeddings, sentence_mask = self.encode_sentences(
            input_ids, attention_mask, chunk_mask
        )
        
        # Step 2: Select top-M sentences
        selected_embeddings, selected_indices = self.select_top_sentences(
            sentence_embeddings, sentence_mask
        )
        
        # Step 3: Project to soft tokens
        soft_tokens = self.soft_token_projector(selected_embeddings)
        
        # Step 4: Build final sequence
        embedded_sequence, final_attention_mask, lm_labels = self.build_final_sequence(
            soft_tokens, labels, batch_size
        )
        
        # Step 5: Forward through LM head
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Pass embeddings directly through the model
            outputs = self.base_model(
                inputs_embeds=embedded_sequence,
                attention_mask=final_attention_mask,
                labels=lm_labels,
                return_dict=True
            )
        
        # Extract decision logits (at position before target)
        decision_logits = outputs.logits[:, -2, :]  # [batch, vocab_size]
        
        # Get yes/no logits only
        yes_no_logits = torch.stack([
            decision_logits[:, self.no_token_id],
            decision_logits[:, self.yes_token_id]
        ], dim=1)  # [batch, 2]
        
        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=yes_no_logits,  # Return simplified logits for metrics
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )
