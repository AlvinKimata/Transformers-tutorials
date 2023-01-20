import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F

#Download the tiny shakespeare daaset.
#!wget 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

#Hyperparameters.
torch.manual_seed(1337)
batch_size = 8
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
#==============================


#Inspect the dataset.
with open('inputs/input.txt', 'r', encoding = 'utf-8') as f:
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

block_size = 8 #context size.

x = train_data[:block_size]
y = train_data[1: block_size + 1]

for t in range(block_size):
  context = x[:t + 1]
  target = y[t]

  # print(f'When input is {context} the target: {target}')

def get_batch(split):
  #Generate a small batch of data of inputs x and target y.
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i: i + block_size] for i in ix])
  y = torch.stack([data[i+1: i + block_size + 1] for i in ix])
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

xb, yb = get_batch('train')
print(f'Inputs: \n {xb.shape}')
print(f'Target: \n {yb.shape}')

for b in range(batch_size):

  for t in range(batch_size):
    context = xb[b, :t + 1]
    target = yb[b, t]

    # print(f'When input is {context.tolist()} the target: {target}')


#Create a model.
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()

    #Each token directly reads off the logits from the next token from a lookup table.
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #number of embedding dimensions.
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets = None):
    tok_emb = self.token_embedding_table(idx) #(batch, time, channel)
    logits = self.lm_head(tok_emb) # (batch, time, vocab_size)

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
      #Get the predictions.
      logits, loss = self(idx)

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
logits, loss = m(xb, yb)


idx = torch.zeros((1, 1), dtype = torch.long)
# print(decode(m.generate(idx, max_new_tokens = 100)[0].tolist()))



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

print(decode(m.generate(idx, max_new_tokens = 500)[0].tolist()))

torch.ones((3, 1)) @ torch.ones((1, 3))

