import json
import torch
from functools import partial
from torch.utils.data import DataLoader, Dataset
import tiktoken
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel


def load_local_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

tokenizer = tiktoken.get_encoding("gpt2")   

file_path = "instruction-data.json"

data = load_local_json(file_path)
print("Number of entries:", len(data))




def format_input(entry):                                                               #convert in alpaca format
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text





train_portion = int(len(data) * 0.85)  
test_portion = int(len(data) * 0.1)                                                      #85% train, 10% test, 5% val
val_portion = len(data) - train_portion - test_portion  

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]







def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,   # ignored by cross entropy function    
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        
        new_item += [pad_token_id]                #pad one token at the end
        
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  #truncate the last token for inputs
        targets = torch.tensor(padded[1:])  #shift +1 to the right for targets

        
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index             #ignore all pad tokens except the first one

       
        if allowed_max_length is not None:                    #truncate if exceeds allowed max length
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

   
    inputs_tensor = torch.stack(inputs_lst).to(device)         #stack into a single tensor
    targets_tensor = torch.stack(targets_lst).to(device)             

    return inputs_tensor, targets_tensor





                                                                 #dataset and dataloader will use this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)




num_workers = 0
batch_size = 8

torch.manual_seed(123)



class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
    

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

                                                         #######data lodeders are ready 

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
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
        att.masked_fill_(self.mask[:t, :t].bool(), -torch.inf)
        att = torch.softmax(att / (self.head_dim ** 0.5), dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous()
        out = out.view(b, t, self.d_out)
        return self.out_proj(out)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) *
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
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

def load_hf_weights(custom_gpt):
    hf = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    sd = hf.state_dict()

    custom_gpt.tok_emb.weight.data.copy_(sd["transformer.wte.weight"])
    custom_gpt.pos_emb.weight.data.copy_(sd["transformer.wpe.weight"])

    for i, block in enumerate(custom_gpt.trf_blocks):
        p = f"transformer.h.{i}."

        qkv_w = sd[p + "attn.c_attn.weight"]
        qkv_b = sd[p + "attn.c_attn.bias"]

        q, k, v = qkv_w.split(custom_gpt.tok_emb.embedding_dim, dim=1)
        qb, kb, vb = qkv_b.split(custom_gpt.tok_emb.embedding_dim)

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

# ===============================
# TEXT GENERATION
# ===============================
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50):
    for _ in range(max_new_tokens):
        logits = model(idx)[:, -1, :]
        logits /= temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -torch.inf

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat([idx, idx_next], dim=1)

    return idx

# ===============================
# RUN
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = GPTModel(BASE_CONFIG).to(device)
load_hf_weights(model)
model.eval()                                                     ###########model loaded with gpt2-medium weights

tokenizer = tiktoken.get_encoding("gpt2")        




def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
      
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

                                         #main training loop
    for epoch in range(num_epochs):
        model.train()  
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel() 
            global_step += 1

                                                                  #evaluation step
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                                #print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen




def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_text_simple(model, idx, max_new_tokens, context_size):
    
    for _ in range(max_new_tokens):
        
        
        idx_cond = idx[:, -context_size:]
        
      
        with torch.no_grad():
            logits = model(idx_cond) 
       
        logits = logits[:, -1, :]  

       
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx





def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())





def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()




import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 3

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

torch.save(model.state_dict(), "MOVI_LLM.pth")       

