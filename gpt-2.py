import torch
import torch.nn as nn
import tiktoken
import math
from dataclasses import dataclass
from torch.nn import functional as F

torch.manual_seed(1337)
torch.mps.manual_seed(1337)

# Check that MPS is available
if not torch.backends.mps.is_available():
    print('MPS not available')
else:
    mps_device = torch.device("mps")

class DataLoaderLite():
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)}")
        self.current_position = 0

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position: self.current_position + (B*T+1)]
        buf = buf.to(mps_device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B*T
        print(self.current_position)
        if(self.current_position + (B*T+1) > len(self.tokens)):
            self.current_position = 0
        return x, y

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # q,k,v in a single tensor
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x) # B, T, 3C
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head,  C // self.n_head).transpose(1, 2) # B, nh, T, hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, hs
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, hs
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) # B, nh, T, T
        att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # B, nh, T, hs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme between the 1st and the last layer
        self.transformer.wte.weight = self.lm_head.weight

        #init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance (module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # T, C
        tok_emb = self.transformer.wte(idx) # B, T, C
        x = pos_emb + tok_emb # B, T, C
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # B, T, C
        logits = self.lm_head(x) # B, T, vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        '''Loads pretrained GPT-2 model weights from hugging face'''
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('loading weights from pretrained GPT: %s', model_type)

        # n_embd, n_head and n_layer is determined by the model type
        config_args = {
            'gpt2' : dict(n_embd=768, n_head=12, n_layer=12),
            'gpt2-medium' : dict(n_embd=1024, n_head=16, n_layer=24),
            'gpt2-large' : dict(n_embd=1280, n_head=16, n_layer=36),
            'gpt2-xl' : dict(n_embd=1600, n_head=25, n_layer=48),
        }[model_type]

        # vocab_size and block_size is same for all model types
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        # get hugging face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']# Print keys in custom model not in Hugging Face model

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch in {k}: {sd_hf[k].shape} (HF) vs {sd[k].shape} (custom)"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


num_return_sequences = 5
max_length = 50


train_loader = DataLoaderLite(B=4, T=32)


# model = GPT.from_pretrained('gpt2-xl')
model = GPT(GPTConfig())
model.to(mps_device)

# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    x, y = train_loader.next_batch()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

import sys; sys.exit(0)

model.eval()
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Here's an haikoo")
tokens = torch.tensor(tokens, dtype=torch.long) # 8
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(mps_device)



while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

# print generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()