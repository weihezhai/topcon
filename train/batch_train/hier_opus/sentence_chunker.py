import re
import torch
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizer

class SentenceChunker:
    """Splits documents into sentences and tokenizes them."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_sentences: int = 128,
        max_sentence_length: int = 64,
        min_sentence_length: int = 3
    ):
        self.tokenizer = tokenizer
        self.max_sentences = max_sentences
        self.max_sentence_length = max_sentence_length
        self.min_sentence_length = min_sentence_length
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling various edge cases."""
        # Remove multiple newlines and normalize whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split on sentence boundaries (. ! ? followed by space and capital letter)
        # Also handle numbered lists and bullet points
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=[-•*\d])'
        
        # Initial split
        sentences = re.split(sentence_pattern, text)
        
        # Further split very long sentences on semicolons or commas if needed
        final_sentences = []
        for sent in sentences:
            if len(sent.split()) > 100:  # If sentence is very long
                # Try splitting on semicolons first
                sub_sents = re.split(r';\s+', sent)
                if len(sub_sents) > 1:
                    final_sentences.extend(sub_sents)
                else:
                    # If still too long, split on commas at natural boundaries
                    parts = re.split(r',\s+(?=\w+\s+\w+)', sent)
                    if len(parts) > 2:
                        # Group parts to maintain reasonable sentence length
                        current = []
                        for part in parts:
                            current.append(part)
                            if len(' '.join(current).split()) > 50:
                                final_sentences.append(' '.join(current))
                                current = []
                        if current:
                            final_sentences.append(' '.join(current))
                    else:
                        final_sentences.append(sent)
            else:
                final_sentences.append(sent)
        
        # Filter out empty or very short sentences
        sentences = [s.strip() for s in final_sentences if s.strip()]
        sentences = [s for s in sentences if len(s.split()) >= self.min_sentence_length]
        
        # Limit to max_sentences
        if len(sentences) > self.max_sentences:
            sentences = sentences[:self.max_sentences]
            
        return sentences
    
    def tokenize_sentences(
        self, 
        sentences: List[str]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Tokenize each sentence independently."""
        chunks = []
        chunk_masks = []
        
        for sentence in sentences:
            # Tokenize with truncation
            encoding = self.tokenizer(
                sentence,
                truncation=True,
                max_length=self.max_sentence_length,
                padding=False,
                return_attention_mask=True,
                add_special_tokens=False  # No special tokens per sentence
            )
            
            chunks.append(encoding['input_ids'])
            chunk_masks.append(encoding['attention_mask'])
            
        return chunks, chunk_masks
    
    def process_document(
        self, 
        text: str,
        label: int
    ) -> Dict[str, any]:
        """Process a single document into sentence chunks."""
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        # Handle empty document
        if not sentences:
            sentences = ["[Empty document]"]
        
        # Tokenize sentences
        chunks, chunk_masks = self.tokenize_sentences(sentences)
        
        return {
            'chunks': chunks,  # List of token ID lists
            'chunk_masks': chunk_masks,  # List of attention masks
            'labels': label,
            'num_sentences': len(chunks)
        }
