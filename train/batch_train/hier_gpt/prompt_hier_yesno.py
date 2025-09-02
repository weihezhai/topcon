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
        # self.backbone = self.lm.model if hasattr(self.lm, "model") else self.lm.transformer
        self.H = self.lm.config.hidden_size
        self.lm.config.use_cache = False  # safer for training

        # tokenize prompts once and keep on buffer
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
        self.chunk_proj = nn.Sequential(nn.Linear(self.H, self.H*self.K), nn.GELU())

        # (optional) chunk index embedding (keeps order info)
        self.max_chunks = 1024
        self.idx_emb = nn.Embedding(self.max_chunks, self.H)
        nn.init.normal_(self.idx_emb.weight, std=0.02)

        # mean pooling over chunk tokens
        self.token_pool = lambda hs, m: (
            (hs * m.to(device=hs.device, dtype=hs.dtype).unsqueeze(-1)).sum(1) /
            m.to(device=hs.device, dtype=hs.dtype).sum(1).clamp(min=1).unsqueeze(-1)
        )
        # Stage-1: freeze the LM (encoder path); later, Stage-2 unfreezes top layers
        self.freeze_lm_for_chunks = freeze_lm_for_chunks
        if freeze_lm_for_chunks:
            for p in self.backbone.parameters():
                p.requires_grad = False
    @property
    def backbone(self):
        return self.lm.model if hasattr(self.lm, "model") else self.lm.transformer
        # self.word_emb = self.lm.get_input_embeddings()
    @property
    def word_emb(self):
        return self.lm.get_input_embeddings()
    # self.lm_head  = self.lm.lm_head
    @property
    def lm_head(self):
        return self.lm.lm_head

    # utilities
    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.word_emb(ids.to(self.word_emb.weight.device))

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
        dev = self.word_emb.weight.device
        H, K = self.H, self.K

        # ---- 1) encode chunks -> mean pooled embeddings [B,C,H]
        x = input_ids.view(B*C, S)
        m = attention_mask.view(B*C, S)

        with torch.set_grad_enabled(not self.freeze_lm_for_chunks):
            enc = self.backbone(input_ids=x, attention_mask=m, use_cache=False)
            last_h = enc.last_hidden_state               # [B*C, S, H]
            m_pool = m.to(device=last_h.device, dtype=last_h.dtype)
            chunk_emb = self.token_pool(last_h, m_pool).view(B, C, H).to(dev)
            # chunk_emb = self.token_pool(last_h, m).view(B, C, H)  # [B,C,H]
            # chunk_emb = chunk_emb * chunk_mask.to(dev).unsqueeze(-1)
            chunk_emb = chunk_emb * chunk_mask.to(device=dev, dtype=chunk_emb.dtype).unsqueeze(-1)

        # + index embedding
        idx_dev = self.idx_emb.weight.device
        idx = torch.arange(C, device=idx_dev).unsqueeze(0).expand(B, C).clamp(max=self.max_chunks-1)
        chunk_emb = chunk_emb.to(idx_dev) + self.idx_emb(idx)

        # ---- 2) project each chunk -> K soft tokens
        proj = self.chunk_proj(chunk_emb).view(B, C, K, H)   # [B,C,K,H]
        m = chunk_mask.to(device=proj.device, dtype=proj.dtype)
        proj = proj * m.unsqueeze(-1).unsqueeze(-1)
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
            # labels: [B, L], only labels[:, -1] != -100 (value = yes/no id)
            tgt_ids = labels[:, -1].to(dev)
        else:
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
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels, use_cache=False)

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
            for p in self.backbone.parameters(): p.requires_grad = True
            return
        if n <= 0: return
        total = len(layers)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in layers[max(0, total - n):]:
            for p in blk.parameters():
                p.requires_grad = True
        # also unfreeze final norm/ln + lm_head to give capacity
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters(): p.requires_grad = True
        for p in self.lm_head.parameters(): p.requires_grad = True
