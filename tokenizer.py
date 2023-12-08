import numpy as np
import torch

class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.idx_to_word = {idx: word for word, idx in vocab.items()}
        self.vocab_size = len(self.vocab)
        self.start_token_idx = self.vocab.get('<start>')
        self.end_token_idx = self.vocab.get('<end>')

    def encode(self, text):
        if isinstance(text, tuple):
            max_length = 0
            encoded_list = []
            for cap in text:
                tokens = cap.split()
                max_length = max(max_length, len(tokens))
                encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
                encoded_list.append(encoded)
            for encoded in encoded_list:
                if len(encoded) < max_length:
                    encoded.extend([-1] * (max_length - len(encoded)))
            return np.stack(encoded_list)
        elif isinstance(text, str):
            tokens = text.split()
            encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
            return encoded

    def decode(self, encoded):
        if isinstance(encoded, str):
            tokens = [self.idx_to_word.get(idx, '<unk>') for idx in encoded]
            text = ' '.join(tokens)
            return text
        elif isinstance(encoded, torch.Tensor):
            texts = []
            for i in range(encoded.shape[0]):
                tokens = []
                for j in encoded[i]:
                    j = j.item()
                    # print(f'debug: j: {type(j)}')
                    if j != self.end_token_idx:
                        tokens.append(self.idx_to_word.get(j))
                    else:
                        tokens.append('<end>')
                        break
                # print(f'debug: tokens:{tokens}')
                texts.append(' '.join(tokens))
            return texts

    def not_end(self, tokens):
        # print(f'debug: not end: {tokens.shape}, {type(tokens)}')
        # if len(tokens.shape) == 2:
        #     tokens = torch.stack(tokens)
        # else:
        #     tokens = np.array(tokens)[None, :]
        not_end_idx = tokens != self.end_token_idx
        return not_end_idx