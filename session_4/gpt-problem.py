import torch
import torch.nn as nn
from torch.nn import functional as F

# final results with this code
# step=4800: train loss = 1.0883, val loss = 1.4908
# LARTIUS:
# Now Mars, they love me.

# LARTIUS:
# Yeven altood my opprovoty.

# MARIANA:
# Traitor, I will hang

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_heads = 6
n_layer = 6
dropout = .2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

# Copy your Head, MultiHeadAttention, FeedForward and Block classes here
    
class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = ???
        self.pos_embedding_table = ???
        # define blocks, a layer norm and a linear layer
        ???

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = ??? # (B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = ??? # sum the token embeddings and position embeddings
        ??? # apply blocks, layer norm and linear layer (leading to the logits variable)

        # do not modify the rest of the method (it computes the loss during the forward pass)
        if targets is None:
            loss = None
        else:
            # idx and targets are both (B,T) tensor of integers
            B, T, C = logits.shape
            logits = logits.view (B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block size tokens
                idx_cond = idx[:, -block_size:]
                # get the predictions
                logits, loss = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx
    
model = GPT()
m = model.to(device)
print(m.parameters())
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters): # increase number of steps for good results... 
    
    # evaluate once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print (f"step={iter}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx = context, max_new_tokens=100)[0].tolist()))