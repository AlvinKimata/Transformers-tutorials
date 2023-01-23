import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F

#Download the tiny shakespeare daaset.
#!wget 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

#Hyperparameters.
torch.manual_seed(1337)
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
n_embd = 384
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
#==============================


#Inspect the dataset.
with open('/content/input.txt', 'r', encoding = 'utf-8') as f:
  text = f.read()

#Get the nuber of unique characters that occur in the text.
chars = sorted(list(set(text)))
vocab_size = len(chars)


#Create a mapping from characters sto integers.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #Encode: Take a string and output a list of integers.
decode = lambda l: ''.join([itos[i] for i in l]) #Decoder: Take a list of integers and output a string.

#Encode the entire text dataset and store it in a torch.Tensor
data = torch.tensor(encode(text), dtype = torch.long)

#Split the data in train and test splits.
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

x = train_data[:block_size]
y = train_data[1: block_size + 1]

for t in range(block_size):
  context = x[:t + 1]
  target = y[t]

  # print(f'When input is {context} the target: {target}')

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


#Create the head attention layer.
class Head(nn.Module):
  '''one head of self-attention.'''
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer('tril', torch.tril(
      torch.ones(block_size, block_size)))
    self.dropout  = nn.Dropout(dropout)
  
  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)

    #Compute attention scores (affinity)
    wei = q @ k.transpose(-2, -1) * C ** -0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim = -1)
    wei = self.dropout(wei) #Prevent some of the nodes from communicating with each other.

    #Perform weighted aggregation of the values.
    v = self.value(x)
    out = wei @ v
    return out


#Create the multi-head attention layer.
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(num_heads * head_size, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.proj(out)
    return out


#Create a feed-forward layer.
class FeedFoward(nn.Module):
  '''A simple linear layer followed by non-linearity.'''
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout)
    )
  
  def forward(self, x):
    return self.net(x)

#Create the decoder block.
class Block(nn.Module):
  '''Transformer block: Communication followed by computation'''

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedFoward(n_embd)

    #Perform layer normalization.
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


#Create the final bigram model.
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()

    #Each token directly reads off the logits from the next token from a lookup table.
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #number of embedding dimensions.
    self.position_embedding_table = nn.Embedding(block_size, n_embd) #number of embedding dimensions.
    self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)


  def forward(self, idx, targets = None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #(batch, time, channel)
    pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(time, channel)
    x = tok_emb + pos_emb #Broadcasting.(Batch, time, channel)
    x = self.blocks(x) 
    logits = self.lm_head(x) # (batch, time, vocab_size)

    if targets is None:
      loss = None
    else:
      #Reshape logits.
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B * T) #Reduce dimension by one.

      #Negative log-likelihood loss.
      loss = F.cross_entropy(logits, targets) #Measures quality of logits to the target. 

    return logits, loss


  def generate(self, idx, max_new_tokens):
    '''Generate function for the model.'''
    for _ in range(max_new_tokens):
      #Crop index to the last block_size tokens.
      idx_cond = idx[:, -block_size:]
      #Get the predictions.
      logits, loss = self(idx_cond)

      #Focus only on the last timestep.
      logits = logits[:, -1, :]

      #Apply a softmax to get the probabilities.
      probs = F.softmax(logits, dim = -1)

      #Sample from the distribution.
      idx_next = torch.multinomial(probs, num_samples = 1)

      #Append sampled index to the running sequence.
      idx = torch.cat((idx, idx_next), dim = 1)

    return idx

m = BigramLanguageModel()
m = m.to(device)

#Print the number of model parameters.
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

#Create an optimizer object.
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

#Training loop.
for steps in tqdm(range(1000)):
  if steps % eval_interval == 0:
    #Evaluate the model.
    losses = estimate_loss()
    print(f"Train loss: {losses['train']}, validation loss: {losses['val']} at step {steps}")

  #Sample a batch of data.
  xb, yb = get_batch('train')

  #Evaluate the loss.
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()

print(loss.item())

#Generate model outputs.
context = torch.zeros((1, 1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))

#Log outputs to output file.
open('outputs.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
