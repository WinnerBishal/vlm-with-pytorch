import torch
from torch import nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
  '''
    Multi-Head Self Attention Module
    Input: (batch_size, seq_len, embed_dim)
    - Attention is computed over the seq_len dimension
    - Multiple heads are computed in parallel
    - Each head has its own set of Q, K, V linear projections
    - Final linear layer projects the outputs of all heads to embed_dim

    Output: (batch_size, seq_len, embed_dim)
  '''

  def __init__(self, config):
    super().__init__()

    self.config = config

    self.embed_dim = self.config['embed_dim']
    self.n_heads = self.config['n_heads']

    self.qk_dim = self.config['qk_dim']
    self.v_dim = self.config['v_dim']

    self.head_dim = self.embed_dim // self.n_heads        # Integer division

    self.wQ = nn.Linear(self.embed_dim, self.qk_dim * self.n_heads)
    self.wK = nn.Linear(self.embed_dim, self.qk_dim * self.n_heads)
    self.wV = nn.Linear(self.embed_dim, self.v_dim * self.n_heads)
    self.wO = nn.Linear(self.v_dim * self.n_heads, self.embed_dim)

  def forward(self, X):

    self.batch_size = X.shape[0]
    self.seq_len = X.shape[1]

    # print(f'MultiHeadAttention: || Recieved : {X.shape} ||, config: {self.config}')

    q = self.wQ(X)
    q = q.view([self.batch_size, self.seq_len, self.n_heads, self.qk_dim])
    q = q.transpose(1, 2)

    k = self.wK(X)
    k = k.view([self.batch_size, self.seq_len, self.n_heads, self.qk_dim])
    k = k.transpose(1, 2)

    v = self.wV(X)
    v = v.view([self.batch_size, self.seq_len, self.n_heads, self.v_dim])
    v = v.transpose(1, 2)

    # Raw scores
    k_transposed = torch.transpose(k, -2, -1)       # -2, -1 means secon last dim and last dim, since we could have 4D tensors
    raw_scores = q @ k_transposed
    raw_scaled_scores = raw_scores / math.sqrt(self.qk_dim)

    # Compute softmax
    prob_scores = F.softmax(raw_scaled_scores, -1)

    # Compute attention
    output = prob_scores @ v

    output = output.transpose(1, 2)
    output = output.contiguous().view(self.batch_size, self.seq_len, self.v_dim * self.n_heads)

    output = self.wO(output)

    # print(f'MultiHeadAttention: || Processed into : {output.shape} ||')

    return output

class FFNN(nn.Module):
  '''
    Feed Forward Neural Network Module
    Input: (batch_size, seq_len, embed_dim)
    - Two linear layers with GELU activation in between
    - Hidden layer has dimension 4 * embed_dim
    Output: (batch_size, seq_len, embed_dim)
  '''

  def __init__(self, config):
    super().__init__()

    self.embed_dim = config['embed_dim']
    self.n_heads = config['n_heads']

    self.qk_dim = config['qk_dim']
    self.v_dim = config['v_dim']

    self.hidden_dim = 4*self.embed_dim

    self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim)
    self.gelu = nn.GELU()
    self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim)

  def forward(self, x):
    output = self.fc1(x)
    output = self.gelu(output)
    output = self.fc2(output)

    return output

class TransformerBlock(nn.Module):
  '''
   Combines Multi-Head Self Attention and Feed Forward Neural Network with Layer Normalization and Residual Connections
   Input: (batch_size, seq_len, embed_dim)
   Output: (batch_size, seq_len, embed_dim)
  '''

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.embed_dim = config["embed_dim"]

    self.multi_head = MultiHeadSelfAttention(self.config)
    self.ffn = FFNN(self.config)

    self.lnorm_in = nn.LayerNorm(self.embed_dim)
    self.lnorm_out = nn.LayerNorm(self.embed_dim)

  def forward(self, x):

    # Attention Part
    normalized_input = self.lnorm_in(x)                        # Normalization
    attn_scores = self.multi_head(normalized_input)            # Attention
    residual_output = attn_scores + x                          # Residual

    # Feedforward Part
    normalized_output = self.lnorm_out(residual_output)        # Normalization
    mlp_output = self.ffn(normalized_output)                   # Feed Forward
    final_output = mlp_output + residual_output                                       # Residual

    return final_output


