import torch
import torch.nn as nn
import numpy as np
import tiktoken
from transformers import GPT2LMHeadModel



GPT_CONFIG_774M = {
    "vocab_size": 50257,
    "context_length": 512,   
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": True
}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, t, _ = x.shape

        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        att = q @ k.transpose(2, 3)

        # FP16 SAFE MASK (ONLY CHANGE HERE)
        att.masked_fill_(self.mask[:t, :t].bool(), -1e4)

        att = torch.softmax(att / (self.head_dim ** 0.5), dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous()
        out = out.view(b, t, self.d_out)
        return self.out_proj(out)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * x ** 3)
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            cfg["emb_dim"], cfg["emb_dim"],
            cfg["context_length"], cfg["drop_rate"],
            cfg["n_heads"], cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        x = self.drop(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)



def load_hf_weights(custom_gpt):
    hf = GPT2LMHeadModel.from_pretrained("gpt2-large")
    hf = hf.half()
    sd = hf.state_dict()

    custom_gpt.tok_emb.weight.data.copy_(sd["transformer.wte.weight"])
    custom_gpt.pos_emb.weight.data.copy_(
    sd["transformer.wpe.weight"][:custom_gpt.pos_emb.weight.size(0)]
)

    for i, block in enumerate(custom_gpt.trf_blocks):
        p = f"transformer.h.{i}."

        qkv_w = sd[p + "attn.c_attn.weight"]
        qkv_b = sd[p + "attn.c_attn.bias"]

        q, k, v = qkv_w.split(1280, dim=1)
        qb, kb, vb = qkv_b.split(1280, dim=0)

        block.att.W_query.weight.data.copy_(q.T)
        block.att.W_key.weight.data.copy_(k.T)
        block.att.W_value.weight.data.copy_(v.T)

        block.att.W_query.bias.data.copy_(qb)
        block.att.W_key.bias.data.copy_(kb)
        block.att.W_value.bias.data.copy_(vb)

        block.att.out_proj.weight.data.copy_(sd[p + "attn.c_proj.weight"].T)
        block.att.out_proj.bias.data.copy_(sd[p + "attn.c_proj.bias"])

        block.ff.layers[0].weight.data.copy_(sd[p + "mlp.c_fc.weight"].T)
        block.ff.layers[0].bias.data.copy_(sd[p + "mlp.c_fc.bias"])

        block.ff.layers[2].weight.data.copy_(sd[p + "mlp.c_proj.weight"].T)
        block.ff.layers[2].bias.data.copy_(sd[p + "mlp.c_proj.bias"])

        block.norm1.scale.data.copy_(sd[p + "ln_1.weight"])
        block.norm1.shift.data.copy_(sd[p + "ln_1.bias"])
        block.norm2.scale.data.copy_(sd[p + "ln_2.weight"])
        block.norm2.shift.data.copy_(sd[p + "ln_2.bias"])

    custom_gpt.final_norm.scale.data.copy_(sd["transformer.ln_f.weight"])
    custom_gpt.final_norm.shift.data.copy_(sd["transformer.ln_f.bias"])
    custom_gpt.out_head.weight.data.copy_(sd["transformer.wte.weight"])



def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50):
    for _ in range(max_new_tokens):
        logits = model(idx)[:, -1, :]
        logits /= temperature

        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -1e4

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat([idx, idx_next], dim=1)

    return idx

