# dataset/chunker.py
from typing import Dict, List
import re

# --- Sliding-window chunking (token-based) ---
def chunk_and_tokenize_sliding(text: str, tokenizer, chunk_len=1536, stride=256, add_bos=True) -> Dict[str, List[List[int]]]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if add_bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    step = max(1, chunk_len - stride)
    chunks, masks = [], []
    for start in range(0, len(ids), step):
        piece = ids[start:start+chunk_len]
        if not piece: break
        chunks.append(piece)
        masks.append([1]*len(piece))
    return {"chunks": chunks, "chunk_masks": masks}

# --- Sentence chunking (flexible) ---
_SENT_SPLIT = re.compile(r'(?<!\b[A-Z])[.!?](?=\s+[A-Z])|(?<=\.)\s+(?=[A-Z][a-z])')

def split_sentences(text: str):
    text = text.strip()
    parts = []
    for block in re.split(r'\n{2,}', text):
        block = block.strip()
        if not block: continue
        if len(block) < 80:
            parts.append(block)
        else:
            parts.extend([s.strip() for s in re.split(_SENT_SPLIT, block) if s.strip()])
    return parts or [text]

def chunk_and_tokenize_sentence(text: str, tokenizer, max_sentences=128, add_bos=True) -> Dict[str, List[List[int]]]:
    sents = split_sentences(text)[:max_sentences]
    chunks, masks = [], []
    bos = tokenizer.bos_token_id if add_bos else None
    for s in sents:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if bos is not None: ids = [bos] + ids
        if not ids: ids = [tokenizer.pad_token_id]
        chunks.append(ids)
        masks.append([1]*len(ids))
    return {"chunks": chunks, "chunk_masks": masks}
