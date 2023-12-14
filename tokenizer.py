import numpy as np
import torch

# class Tokenizer:
#     def __init__(self, vocab):
#         self.vocab = vocab
#         self.idx_to_word = {idx: word for word, idx in vocab.items()}
#         self.vocab_size = len(self.vocab)
#         self.start_token_idx = self.vocab.get('<start>')
#         self.end_token_idx = self.vocab.get('<end>')

#     def encode(self, text):
#         if isinstance(text, tuple):
#             max_length = 0
#             encoded_list = []
#             for cap in text:
#                 tokens = cap.split()
#                 max_length = max(max_length, len(tokens))
#                 encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
#                 encoded_list.append(encoded)
#             for encoded in encoded_list:
#                 if len(encoded) < max_length:
#                     encoded.extend([-1] * (max_length - len(encoded)))
#             return np.stack(encoded_list)
#         elif isinstance(text, str):
#             tokens = text.split()
#             encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
#             return encoded

#     def decode(self, encoded):
#         if isinstance(encoded, str):
#             tokens = [self.idx_to_word.get(idx, '<unk>') for idx in encoded]
#             text = ' '.join(tokens)
#             return text
#         elif isinstance(encoded, torch.Tensor):
#             texts = []
#             for i in range(encoded.shape[0]):
#                 tokens = []
#                 for j in encoded[i]:
#                     j = j.item()
#                     # print(f'debug: j: {type(j)}')
#                     if j != self.end_token_idx:
#                         tokens.append(self.idx_to_word.get(j))
#                     else:
#                         tokens.append('<end>')
#                         break
#                 # print(f'debug: tokens:{tokens}')
#                 texts.append(' '.join(tokens))
#             return texts

#     def not_end(self, tokens):
#         # print(f'debug: not end: {tokens.shape}, {type(tokens)}')
#         # if len(tokens.shape) == 2:
#         #     tokens = torch.stack(tokens)
#         # else:
#         #     tokens = np.array(tokens)[None, :]
#         not_end_idx = tokens != self.end_token_idx
#         return not_end_idx
    
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.key_to_index = self.vocab.wv.key_to_index
        self.idx_to_word = self.vocab.wv.index_to_key
        self.word_vectors = self.vocab.wv.vectors
        self.vocab_size = len(vocab.wv.key_to_index)
        self.start_token_idx = self.key_to_index['<start>']
        self.end_token_idx = self.key_to_index['<end>']

    # 转成100维的向量
    def encode(self, text):
        if isinstance(text, tuple):
            max_length = 0
            encoded_list = []
            target_idxs = []
            for cap in text:
                tokens = cap.split()
                max_length = max(max_length, len(tokens))
                encoded = [self.vocab.wv[token] for token in tokens]
                idxs = [self.key_to_index[token] for token in tokens]
                encoded_list.append(np.stack(encoded))
                target_idxs.append(idxs)
            for i, encoded in enumerate(encoded_list):
                cur_len = encoded.shape[0]
                if cur_len < max_length:
                    encoded_list[i] = np.concatenate((encoded, np.ones((max_length - cur_len, encoded.shape[1])) * -1), 0)
                    target_idxs[i].extend([-1] * (max_length - len(encoded)))
                    # print(f'debug: ')
            return np.stack(encoded_list), np.stack(target_idxs)
        elif isinstance(text, str):
            tokens = text.split()
            encoded = [self.vocab.wv[token] for token in tokens]
            return np.stack(encoded)

    # 输入是idx，输出是文本
    def decode(self, encoded):
        if isinstance(encoded, str):
            tokens = [self.idx_to_word[idx] for idx in encoded]
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
                        tokens.append(self.idx_to_word[j])
                    else:
                        tokens.append('<end>')
                        break
                # print(f'debug: tokens:{tokens}')
                texts.append(' '.join(tokens))
            return texts

    # def get_vector(self, idxs):


    def not_end(self, tokens):
        # print(f'debug: not end: {tokens.shape}, {type(tokens)}')
        # if len(tokens.shape) == 2:
        #     tokens = torch.stack(tokens)
        # else:
        #     tokens = np.array(tokens)[None, :]
        not_end_idx = tokens != self.end_token_idx
        return not_end_idx