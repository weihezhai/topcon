# models/prompt_hier_yesno.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class PromptHierYesNo(nn.Module):
    """
    Build a sequence in *embedding space*:
      [PROMPT_PREFIX ids] + [K soft tokens per chunk] + [PROMPT_SUFFIX ids + "Decision:"] + [TARGET yes/no token]
    We pass 'labels' with only the last position != -100 to preserve your LM loss.
    For your unchanged compute_metrics (which reads logits at the *label position*), we return
    a 1-step *shifted* copy of logits so predictions[t] correspond to next-token logits.

    Inputs from collator:
      input_ids      [B, C, S]  (token ids per chunk)
      attention_mask [B, C, S]
      chunk_mask     [B, C]     (True for real chunk)
      labels         [B, L]     (sequence labels; only last pos != -100, set to yes/no id)
    Returns:
      {"loss": loss_from_HF, "logits": shifted_logits_for_metrics [B, L, V]}

    Notes on device placement with device_map="auto":
      • Do NOT manually .to(device) the backbone inputs (input_ids, attention_mask). Accelerate will shard/move them.
      • Keep/create all custom tensors and layers (soft tokens, attn, prefix/suffix embeds, idx_emb, chunk_proj, labels)
        on the *embedding device* (the device of the word embedding matrix).
      • If you unfreeze top layers later, grads are enabled dynamically.
    """
    def __init__(
        self,
        base_model_path: str,
        tokenizer,
        prompt_prefix: str,
        prompt_suffix: str,
        k_soft_tokens_per_chunk: int = 4,
        freeze_lm_for_chunks: bool = True,
        device_map: str = "auto",
        torch_dtype = torch.bfloat16,
    ):
        super().__init__()
        self.tok = tokenizer
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        # access internals
        self.backbone = self.lm.model if hasattr(self.lm, "model") else self.lm.transformer
        self.word_emb = self.lm.get_input_embeddings()
        self.lm_head  = self.lm.lm_head
        self.H = self.lm.config.hidden_size
        self.lm.config.use_cache = False  # safer for training

        # Anchor device/dtype = where embeddings live
        self.emb_device = self.word_emb.weight.device
        self.emb_dtype  = self.word_emb.weight.dtype

        # tokenize prompts once and keep on buffer (left on CPU until first .to in _embed_ids)
        self.register_buffer(
            "prefix_ids",
            torch.tensor(self.tok(prompt_prefix, add_special_tokens=False)["input_ids"], dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "suffix_ids",
            torch.tensor(self.tok(prompt_suffix, add_special_tokens=False)["input_ids"], dtype=torch.long),
            persistent=False
        )

        # yes/no (single-token) ids
        yes_ids = self.tok(" yes", add_special_tokens=False)["input_ids"]
        no_ids  = self.tok(" no",  add_special_tokens=False)["input_ids"]
        assert len(yes_ids)==1 and len(no_ids)==1, "Use leading space so ' yes'/' no' are single tokens."
        self.yes_id, self.no_id = yes_ids[0], no_ids[0]

        # chunk→soft tokens head
        self.K = k_soft_tokens_per_chunk
        self.chunk_proj = nn.Sequential(
            nn.Linear(self.H, self.H*self.K),
            nn.GELU()
        )

        # (optional) chunk index embedding (keeps order info)
        self.max_chunks = 1024
        self.idx_emb = nn.Embedding(self.max_chunks, self.H)
        nn.init.normal_(self.idx_emb.weight, std=0.02)

        # Place custom params on the embedding device/dtype (they are NOT in device_map)
        self.chunk_proj.to(device=self.emb_device, dtype=self.emb_dtype)
        self.idx_emb.to(device=self.emb_device, dtype=self.emb_dtype)

        # mean pooling over chunk tokens
        self.token_pool = lambda hs, m: (hs * m.unsqueeze(-1)).sum(1) / m.sum(1).clamp(min=1).unsqueeze(-1)

        # Stage-1: freeze the LM (encoder path); later, Stage-2 unfreezes top layers
        self.freeze_lm_for_chunks = freeze_lm_for_chunks
        if freeze_lm_for_chunks:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # utilities
    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.word_emb(ids.to(self.emb_device))

    @torch.no_grad()
    def prompt_lengths(self):
        # lengths on current device
        return int(self.prefix_ids.numel()), int(self.suffix_ids.numel())

    def forward(self, input_ids, attention_mask, chunk_mask, labels=None):
        """
        input_ids:      [B, C, S]
        attention_mask: [B, C, S]
        chunk_mask:     [B, C]
        labels:         [B, L] final sequence labels (from collator); only last position != -100
        """
        B, C, S = input_ids.shape
        dev = self.emb_device
        H, K = self.H, self.K

        # ---- 1) encode chunks -> mean pooled embeddings [B,C,H]
        # DO NOT move x/m to any device here; let Accelerate handle per-shard movement.
        x = input_ids.view(B*C, S)
        m = attention_mask.view(B*C, S)

        # Enable grads only if any backbone params are trainable (after potential unfreezing)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            enc = self.backbone(input_ids=x, attention_mask=m, use_cache=False)
            last_h = enc.last_hidden_state  # [B*C, S, H] (on the shard/last layer device)
            # Align mask dtype/device to last_h for safe pooling
            m_dev = m.to(last_h.device, dtype=last_h.dtype)
            chunk_emb = self.token_pool(last_h, m_dev).view(B, C, H)  # [B,C,H] (still on last_h.device)
            chunk_emb = chunk_emb * chunk_mask.to(last_h.device).unsqueeze(-1)

        # Move pooled chunks to the embedding device/dtype for the rest of the path
        chunk_emb = chunk_emb.to(dev, dtype=self.emb_dtype)

        # + index embedding (on embedding device)
        idx = torch.arange(C, device=dev).unsqueeze(0).expand(B, C).clamp(max=self.max_chunks-1)
        chunk_emb = chunk_emb + self.idx_emb(idx)

        # ---- 2) project each chunk -> K soft tokens
        proj = self.chunk_proj(chunk_emb).view(B, C, K, H)   # [B,C,K,H] (on emb_device)
        proj = proj * chunk_mask.unsqueeze(-1).unsqueeze(-1).to(proj.dtype)
        soft_tokens = proj.view(B, C*K, H)                   # [B, C*K, H]

        # ---- 3) compose full inputs_embeds
        pref = self._embed_ids(self.prefix_ids)              # [P,H]
        suff = self._embed_ids(self.suffix_ids)              # [S,H]
        P, Suf = pref.size(0), suff.size(0)
        pref_b, suff_b = pref.unsqueeze(0).expand(B, P, H), suff.unsqueeze(0).expand(B, Suf, H)

        # The batch is padded to C (max chunks in batch) by the collator.
        CK = C * K
        # Compose: [prefix | soft-chunks | suffix | target yes/no token]
        L = P + CK + Suf + 1
        inputs_embeds = torch.zeros(B, L, H, dtype=pref_b.dtype, device=dev)
        inputs_embeds[:, :P, :] = pref_b
        inputs_embeds[:, P:P+CK, :] = soft_tokens
        inputs_embeds[:, P+CK:P+CK+Suf, :] = suff_b

        # last token = target token embedding (teacher-forcing)
        # We derive target ids back from 'labels' last element per sample (already set by collator).
        if labels is not None:
            labels_on = labels.to(dev)
            tgt_ids = labels_on[:, -1]
        else:
            labels_on = None
            tgt_ids = torch.full((B,), self.no_id, dtype=torch.long, device=dev)
        inputs_embeds[:, -1, :] = self.word_emb(tgt_ids)

        # ---- 4) attention mask for the composed sequence
        attn = torch.zeros(B, L, dtype=torch.long, device=dev)
        attn[:, :P] = 1
        cm = chunk_mask.to(dev).unsqueeze(-1).expand(B, C, K).reshape(B, CK)
        attn[:, P:P+CK] = cm.long()
        attn[:, P+CK:P+CK+Suf] = 1
        attn[:, -1] = 1

        # ---- 5) forward LM with labels (standard HF CausalLM loss)
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels_on, use_cache=False)

        # ---- 6) return 1-step shifted logits for your unchanged compute_metrics
        lm_logits = out.logits                              # [B, L, V]
        shifted = torch.zeros_like(lm_logits)
        shifted[:, 1:, :] = lm_logits[:, :-1, :]
        return {"loss": out.loss, "logits": shifted}

    # -------- stage-2 helpers --------
    def unfreeze_top_layers(self, n: int):
        """
        Unfreeze top-n decoder blocks of the backbone.
        Works for LLaMA/Qwen-style models with .model.layers / .transformer.layers
        """
        layers = getattr(self.backbone, "layers", None) or getattr(self.backbone, "h", None)
        if layers is None:
            # fallback: unfreeze everything if we cannot locate layers list
            for p in self.backbone.parameters():
                p.requires_grad = True
            return
        if n <= 0:
            return
        total = len(layers)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in layers[max(0, total - n):]:
            for p in blk.parameters():
                p.requires_grad = True
        # also unfreeze final norm/ln + lm_head to give capacity
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
        for p in self.lm_head.parameters():
            p.requires_grad = True