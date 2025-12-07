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
class TextEncoderConfig:
    max_len: int = 100
    embed_dim: int = 768
    vocab_size: int = 20
    
    







