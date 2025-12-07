from config import TextEncoderConfig

from blocks import InputTextEncoder
from dataclasses import asdict

config = TextEncoderConfig()
config = asdict(config)
text_encoder = InputTextEncoder(config)

input_txt_file = 'sentences.txt'

input_sentences = []

with open(input_txt_file, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Extract sentences based on end punctuation
        if line.split() and line[-1] in {'.', '!', '?'}:
            sentences = line.split('. ')
            input_sentences.extend(sentences)

print(input_sentences)

text_encoder.fit_tokenizer(input_sentences)
token_sequence = text_encoder.tokenize('We can experience in the right time.')
print(token_sequence)