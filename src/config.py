from dataclasses import dataclass

@dataclass
class TransformerConfig:
    embed_dim: int = 768
    qk_dim: int = 100
    v_dim: int = 48
    n_heads: int = 8

@dataclass
class PatcherConfig:
    patch_size: int = 16
    in_channels: int = 3
    img_size: int = 224
    embed_dim: int = 768

@dataclass
class VEncoderConfig:
    depth: int = 12
    n_classes: int = 1000

@dataclass
class TokenizerConfig:
    max_len: int = 100          # maximum tokens per sentence to process
    pad_token: str = '[PAD]'
    eos_token: str = '[EOS]'
    unknown_token: str = '[UNK]'

@dataclass
class TextEncoderConfig:
    embed_dim: int = 768        # embedding vector length
    depth: int = 12             # no. of transformer blokcs to stack
    vocab_size: int = 309       # total no. of tokens to learn from the training data, should update after running tokenizer for once
    
    







