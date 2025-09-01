import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

@dataclass
class HierarchicalDataCollator:
    """Collate sentence chunks into 3D tensors for batch processing."""
    
    tokenizer: PreTrainedTokenizer
    max_sentences: int = 128
    max_sentence_length: int = 64
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate features into 3D tensors.
        
        Returns:
            - input_ids: [batch_size, num_sentences, sentence_length]
            - attention_mask: [batch_size, num_sentences, sentence_length]
            - chunk_mask: [batch_size, num_sentences] (1 for real sentences, 0 for padding)
            - labels: [batch_size] (binary labels)
        """
        batch_size = len(features)
        
        # Find max dimensions in this batch
        max_num_sentences = max(f['num_sentences'] for f in features)
        max_num_sentences = min(max_num_sentences, self.max_sentences)
        
        # Find max sentence length in batch
        max_sent_len = 0
        for f in features:
            for chunk in f['chunks']:
                max_sent_len = max(max_sent_len, len(chunk))
        max_sent_len = min(max_sent_len, self.max_sentence_length)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_sent_len = (
                (max_sent_len + self.pad_to_multiple_of - 1) 
                // self.pad_to_multiple_of 
                * self.pad_to_multiple_of
            )
        
        # Initialize tensors
        input_ids = torch.full(
            (batch_size, max_num_sentences, max_sent_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_num_sentences, max_sent_len),
            dtype=torch.long
        )
        chunk_mask = torch.zeros(
            (batch_size, max_num_sentences),
            dtype=torch.bool
        )
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        
        # Fill tensors
        for b, feature in enumerate(features):
            num_sents = min(feature['num_sentences'], max_num_sentences)
            
            for s in range(num_sents):
                if s < len(feature['chunks']):
                    chunk = feature['chunks'][s]
                    chunk_mask_seq = feature['chunk_masks'][s]
                    
                    sent_len = min(len(chunk), max_sent_len)
                    
                    # Fill input_ids and attention_mask
                    input_ids[b, s, :sent_len] = torch.tensor(chunk[:sent_len])
                    attention_mask[b, s, :sent_len] = torch.tensor(chunk_mask_seq[:sent_len])
                    
                    # Mark this sentence as valid
                    chunk_mask[b, s] = True
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'chunk_mask': chunk_mask,
            'labels': labels
        }
