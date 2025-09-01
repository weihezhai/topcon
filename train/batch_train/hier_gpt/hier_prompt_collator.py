# dataloaders/hier_prompt_collator.py
import torch

class HierPromptCollator:
    """
    Pads ragged chunks to [B, C, S] and *also* builds the final label sequence [B, L]:
      L = len(prefix_ids) + (C*K) + len(suffix_ids) + 1
    Only labels[:, -1] holds the yes/no token id; the rest are -100.
    """
    def __init__(self, tokenizer, prompt_prefix_ids, prompt_suffix_ids, k_soft_tokens_per_chunk=4, max_seq_len=2048):
        self.tok = tokenizer
        self.P_ids = prompt_prefix_ids
        self.S_ids = prompt_suffix_ids
        self.K = int(k_soft_tokens_per_chunk)
        self.max_seq_len = max_seq_len

        # yes/no single-token ids
        yes_ids = tokenizer(" yes", add_special_tokens=False)["input_ids"]
        no_ids  = tokenizer(" no",  add_special_tokens=False)["input_ids"]
        assert len(yes_ids)==1 and len(no_ids)==1, "Ensure ' yes' and ' no' are single tokens."
        self.YES_ID, self.NO_ID = yes_ids[0], no_ids[0]

    def __call__(self, features):
        # Determine batch sizes
        max_chunks = max(len(f["chunks"]) for f in features)
        max_len = min(
            self.max_seq_len,
            max(max(len(c) for c in f["chunks"]) for f in features)
        )

        B = len(features)
        # chunk tensors
        input_ids = torch.full((B, max_chunks, max_len), self.tok.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_chunks, max_len), dtype=torch.long)
        chunk_mask = torch.zeros((B, max_chunks), dtype=torch.bool)

        # Fill chunk tensors
        for i, f in enumerate(features):
            C = len(f["chunks"])
            chunk_mask[i, :C] = True
            for j, (ids, msk) in enumerate(zip(f["chunks"], f["chunk_masks"])):
                ids = ids[:max_len]; msk = msk[:max_len]
                Lc = len(ids)
                input_ids[i, j, :Lc] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, j, :Lc] = torch.tensor(msk, dtype=torch.long)

        # Build final sequence labels (only last position is supervised)
        P = len(self.P_ids); S = len(self.S_ids); CK = max_chunks * self.K
        L = P + CK + S + 1
        labels = torch.full((B, L), -100, dtype=torch.long)
        # last position holds the target token id based on doc label 0/1
        for i, f in enumerate(features):
            doc_label = int(f["labels"])  # 0/1 in your dataset
            labels[i, -1] = self.YES_ID if doc_label == 1 else self.NO_ID

        return {
            "input_ids": input_ids,          # [B, C, S]
            "attention_mask": attention_mask,# [B, C, S]
            "chunk_mask": chunk_mask,        # [B, C]
            "labels": labels,                # [B, L]
        }
