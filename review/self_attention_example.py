import torch
import torch.nn as nn 
from torch.nn import functional as F


torch.manual_seed(42)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)
print(type(x))

# Single head of self attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B, T, 16)
q = query(x)  # (B, T, 16)
v = value(x)
wei = q @ k.transpose(-2, -1)  # only last 2 dimensions and not include the batch dimension, (B, T, 16) @ (B, 16, T) -> (B, T, T)

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))  # This will be data dependent during training
wei = wei.masked_fill(tril==0, float('-inf'))  # Deleting this line will make this script an encoding attention block, otherwise it's decoding
wei = F.softmax(wei, dim=-1)
out = wei @ v  # x is private information to this token, value is what is being communicated to the other tokens

out * head_size**-0.5  # Essentially get back to unit-variance at initialization

print(out.shape)
print(out[0])

# Self attention has no notion of space, which is why we add positional encoding
# self attention -> means source is all the same for Q, K, V
# if there are different sources for Q, K, V, then this will be cross attention
