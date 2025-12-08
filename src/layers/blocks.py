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

  def forward(self, X, mask = None):

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

    # Mask to turn off attention for PAD tokens
    raw_scaled_scores = raw_scaled_scores.mask(mask == 0, -1e9)
    
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

  def forward(self, x, mask=None):

    # Attention Part
    normalized_input = self.lnorm_in(x)                        # Normalization
    attn_scores = self.multi_head(normalized_input, mask)            # Attention
    residual_output = attn_scores + x                          # Residual

    # Feedforward Part
    normalized_output = self.lnorm_out(residual_output)        # Normalization
    mlp_output = self.ffn(normalized_output)                   # Feed Forward
    final_output = mlp_output + residual_output                                       # Residual

    return final_output

class PatchEmbedding(nn.Module):
  '''
  Converts input images into patch embeddings
  Input : (batch_size, in_channels, in_height, in_width)
  Output: (batch_size, n_patches + 1, embed_dim)

  pos_embed: Positional embeddings to retain spatial information.
  cls_token: Classification token to aggregate information for classification.

  It is useful to note that, positional embeddings are ADDED to the patch embeddings.
  But, cls_token is CONCATENATED to the patch embeddings.

  Note the difference between adding and concatenating.

  It is similar to tokenization but for images.
  '''
  def __init__(self, config):

    super().__init__()

    self.patch_size = config['patch_size']
    self.in_width = config['img_size']
    self.in_height = config['img_size']
    self.in_channels = config['in_channels']
    self.embed_dim = config['embed_dim']

    self.n_patch_along_width = self.in_width // self.patch_size
    self.n_patch_along_height = self.in_height // self.patch_size

    self.n_patches = self.n_patch_along_height * self.n_patch_along_width

    self.projection = nn.Conv2d(in_channels = self.in_channels,
                                out_channels= self.embed_dim,
                                kernel_size = self.patch_size,
                                stride = self.patch_size,
                                )

    # Classification token which will be learned to match the language associations
    self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    # Positional embeddings
    self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, self.embed_dim))

  def forward(self, x):

    # print(f'ImageToEmbedding : || Image received: {x.shape} ||')

    # x shape: (Batch, 3, 224, 224)
    x = self.projection(x)
    # x shape: (Batch, 768, 14, 14)

    # We need to convert it to (Batch, 14x14 = 196, 768)
    x = x.flatten(2)
    x = x.transpose(-1, -2)

    # Expand and integrate cls token
    B = x.shape[0]
    cls_token = self.cls_token.expand(B, -1, -1)

    # print(f'Cls_token: {cls_token.shape}, x: {x.shape}')
    x = torch.cat((cls_token, x), dim = 1)

    # Expand and integrate positional embeddings
    pos_embeddings = self.pos_embed.expand(B, -1, -1)
    # print(f'pos_embeddingg: {pos_embeddings.shape}, X : {x.shape}')
    x = x + pos_embeddings

    # print(f'ImageToEmbedding: ||Image processed into : {x.shape} ||')

    return x

class InputImageEncoder(nn.Module):
  '''
  self.forward :
  Input Image ---> ImageEmbedding ---> TransformerBlock ---> LayerNorm ---> Learnt [CLS] Embedding
                                                                      |
                                                                      ---> Image Class Probabilities
  Given a batch of images, it generates image embeddings and processes the
    patches (image tokens) through transformer blocks
      to finally generate encoded representations.

  It handles following nuances:

  1. Sequential pass-through on transfomer blocks
  2. layer normalization on block output
  3. It has a head which can convert [CLS] token to class probabilities
  4. When the global representation of whole image is required, values learnt at [CLS] token is considered as output

  DEPENDENCIES

  1. ImageToEmbedding
  2. TransformerBlock

  # Hence, called ImageEncoder.

  '''

  def __init__(self, venc_config, patch_config, trans_config):

    super().__init__()

    self.venc_config = venc_config
    self.patch_config = patch_config
    self.trans_config = trans_config

    self.depth = self.venc_config['depth']
    self.image_to_embedding = PatchEmbedding(self.patch_config)

    # self.trans_config['seq_len'] = self.image_to_embedding.n_patches + 1
    # print(f'seq_len : {self.trans_config['seq_len']}')

    self.transformer_blocks = [TransformerBlock(self.trans_config) for _ in range(self.depth)]

    self.blocks = nn.Sequential(*[self.image_to_embedding, *self.transformer_blocks])

    self.layer_norm = nn.LayerNorm(self.venc_config['embed_dim'])
    self.head = nn.Linear(self.venc_config['embed_dim'], self.venc_config['n_classes'])


  def forward(self, x):
    # x is image of shape b, c, w, h
    x = self.blocks(x)

    # x is of shape b, n_patches + 1, embed_dim (one patch is classification token)
    x = self.layer_norm(x)

    # If we only want to output the learnt [CLS] embeddings itself
    output = x[:, 0, :]

    # If we want to classify image based on [CLS] token
    # output_cls_vector = x[:, 0, :]
    # output = self.head(output_cls_vector)

    return output

class SimpleTokenizer(nn.Module):

  def __init__(self, tokenizer_config):
    super().__init__()
    
    self.config = tokenizer_config

    # Vocabulary building
    self.pad_token = self.config['pad_token']
    self.eos_token = self.config['eos_token']
    self.unknown_token = self.config['unknown_token']
    self.word_to_id = None     
    
    self.vocab_size_learned = None
  
  def fit_tokenizer(self, data):
    """
    Docstring for fit_tokenizer
    
    :param self: Description
    :param data: list of sentences
    Stores each token with token id mappings in self.word_to_id
    """
    words = []

    for sentence in data:
      word_list = sentence.split()
      for word in word_list:
        words.append(word)
    
    unique_words = set(words)
    sorted_words = sorted(unique_words)

    vocab = [self.pad_token, self.eos_token, self.unknown_token] + sorted_words
    self.vocab_size_learned = len(vocab)
    
    token_with_ids = dict()
    for id, token in enumerate(vocab):
      token_with_ids[token] = id
    
    self.word_to_id = token_with_ids
    
  def tokenize(self, sentence):
    '''
    Docstring for tokenize
    
    :param self: Description
    :param sentence: gets the sentence string to be tokenized
    :return: tensor of token ids of shape (max_len,), n_words: no. of words before eos token
    '''

    # Split a sentence into words and add all words' token id to token_sequence
    words = sentence.split()
    token_sequence = []
    length = len(words)
    for word in words:
      try:
        token_sequence.append(self.word_to_id[word])
      except:
        token_sequence.append(self.word_to_id[self.unknown_token])
        # print(f"Word ['{word}'] not found in vocabulary.")
    
    # Add token id of EOS token at the end
    token_sequence.append(self.word_to_id[self.eos_token])
    
    # Handle padding and truncation to match max sequence length
    max_len = self.config['max_len']
    pad_id = self.word_to_id[self.pad_token]
    
    if len(token_sequence) < max_len:
      n_pads = max_len - len(token_sequence)  
      token_sequence = token_sequence + [pad_id]*n_pads
    
    else:
      # truncate and add EOS such that len(token_sequence) == max_len
      token_sequence = token_sequence[:max_len - 1]
      token_sequence.append(self.word_to_id[self.eos_token])
      length = len(token_sequence) - 1
    
    return torch.tensor(token_sequence), length

class InputTextEncoder(nn.Module):
  
  def __init__(self, tokenizer_config, text_encoder_config, transformer_config):
    super().__init__()
    self.enc_config = text_encoder_config
    self.trans_config = transformer_config
    self.tkn_config = tokenizer_config
    
    # Embeddings
    self.embedding = nn.Embedding(self.enc_config['vocab_size'], self.trans_config['embed_dim'])
    self.pos_embed = nn.Parameter(torch.randn(1, self.tkn_config['max_len'], self.trans_config['embed_dim']))
    
    # Transformer
    self.blocks = nn.ModuleList([TransformerBlock(self.trans_config) for _ in range(self.enc_config['depth'])])
    self.norm = nn.LayerNorm(self.trans_config['embed_dim'])
    
  def forward(self, x, lengths):
    '''
    Docstring for forward
    
    :param x: Batch of Tensor of token ids (batch_size, Max_len)
    :param lengths: tensor of lengths (batch_size, )
    :return cls_embeddings: Batch of Tensor of embeddings
    '''
    
    # Get Embeddings
    x = self.embedding(x)
    
    # Add positional embedding
    x = x + self.pos_embed[:, :x.shape[1], :]               # Slicing in case input doesn't have expected max_len
    
    # Create mask
    mask_indices = torch.arange(x.shape[1], device = x.device)
    mask = mask_indices.unsqueeze(0) < lengths.unsqueeze(1)
    
    # Transform
    # Allows self-attention to learn better embeddings
    for block in enumerate(self.blocks):
      x = block(x, mask)
    x = self.norm(x)
    
    # Extraction of EOS token
    batch_indices = torch.arange(x.shape[0], device=x.device)
    cls_embeddings = x[batch_indices, lengths]
    
    return cls_embeddings, mask