import sys
import os

sys.path.insert(0, os.path.abspath('../src/layers'))
sys.path.insert(0, os.path.abspath('../src'))
from config import TextEncoderConfig, TransformerConfig, TokenizerConfig

from blocks import SimpleTokenizer, InputTextEncoder
from dataclasses import asdict

enc_config = TextEncoderConfig()
enc_config = asdict(enc_config)

trans_config = TransformerConfig()
trans_config = asdict(trans_config)

tokenizer_config = TokenizerConfig()
tokenizer_config = asdict(tokenizer_config)

tokenizer = SimpleTokenizer(tokenizer_config)
text_encoder = InputTextEncoder(tokenizer_config, enc_config, trans_config)

input_txt_file = 'sentences.txt'

input_sentences = []

with open(input_txt_file, 'r') as f:
    for line in f:
        line = line.strip()
        # Extract sentences based on end punctuation
        if line.split() and line[-1] in {'.', '!', '?'}:
            sentences = line.split('.')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
            input_sentences.extend(sentences)

print(input_sentences)

tokenizer.fit_tokenizer(input_sentences)
print("Vocabulary Size Learned:", tokenizer.vocab_size_learned)

sequences = []
lengths = []

for sentence in input_sentences:
    tkn_sequence, length = tokenizer.tokenize(sentence)
    sequences.append(tkn_sequence)
    sequences.append(length)
