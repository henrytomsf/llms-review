import logging

import torch
import torch.nn as nn 
from torch.nn import functional as F


# Add logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('llms.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8  # context size also called block size
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 200
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seed
torch.manual_seed(42)

def read_input_data() -> str:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    return text

# Unique chars in the set
text = read_input_data()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create tokenizer mappings (using the most basic tokenizer, can use other ones that tokenize at the part of word level)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

# Create encoder and decoder to transform back and forth between results
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Split the data into train and test
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data loader into training of the model
def get_batch(split: str):
    # Generate batch of dataw of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+1+BLOCK_SIZE] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# Loss function to train the bigram model
# no_grad since we want to disable gradient calcuations to save on memory/speed esp for evaluation or inference
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Define the bigram model that only predicts words given the previous word P(w_t | w_t-1)
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup embedding table that's initialized randomly (that needs to be trained)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B,T,C) (batch, time or sequence_length, channel or number of classes or vocab size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # print("DEBUG logits shape: ", logits.shape)
            # print("DEBUG logits: ", logits)
            logits = logits.view(B*T, C)
            # print("DEBUG AFTER logits shape: ", logits.shape)
            # print("DEBUG AFTER logits: ", logits)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictrions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # beocmes (B, C)
            # Apply softmax to get probs
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution (probably want to get the max)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for iter in range(MAX_ITERS):
        # Evaluate at intervals
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Sample batch of data to train
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)  # Need to do this to avoid accumulating gradients
        loss.backward()
        optimizer.step()





if __name__ == '__main__':
    xb, yb = get_batch(split='train')
    model = BigramLanguageModel(vocab_size=vocab_size)
    model = model.to(DEVICE)
    logits, loss = model(xb, yb)
    print(loss)

    logger.info(f'Model training for {MAX_ITERS} steps...')
    train(model)

    logger.info(f'Generating some text...')
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    # print(model.generate(idx, max_new_tokens=100)[0].tolist())
    print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
