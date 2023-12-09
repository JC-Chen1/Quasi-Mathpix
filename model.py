import torch
import torch.nn as nn
import math
import torchvision

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # 讲分类层去掉，只需要前面的feature map
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-2]))

    def forward(self, image):
        encoded_out = None
        with torch.no_grad():
            encoded_out = self.resnet(image)
            encoded_out = encoded_out.view(encoded_out.size(0), encoded_out.size(1), -1)
            encoded_out = encoded_out.permute(0, 2, 1)
            encoded_out = encoded_out.detach()
        return encoded_out

# todo: 加入位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x, pos=None):
        if pos is None:
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len]
            return x
        else:
            x = x + self.pe[:, pos]
            return x
    

class Decoder_transformer(nn.Module):
    def __init__(self, tokenizer, max_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.output_size = tokenizer.vocab_size

        self.input_size = 2048
        self.embed_size = 256
        self.num_layer = 1
        self.num_head = 4
        self.max_length = max_length
        self.embedding_net = nn.Linear(1, self.embed_size)
        self.pe = PositionalEncoding(d_model=self.embed_size, max_len=max_length)

        self.input_to_encoded = nn.Linear(self.input_size, self.embed_size)

        decoder_layers = nn.TransformerDecoderLayer(d_model=self.embed_size,nhead=self.num_head,batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=self.num_layer)
        
        # output_net
        self.output_net = nn.Linear(self.embed_size, self.output_size)

    # no mask
    def forward(self, encoder_out, encoded_captions = None, caption_lengths = None):
        bs = encoder_out.shape[0]
        device = encoder_out.device
        # print(f'debug: device in train {device}')

        # sort by the length for the ease of paralleling
        # print(f'debug: captions length:{caption_lengths}')
        caption_lengths, sort_idx = torch.sort(caption_lengths, dim=-1, descending=True)
        # print(f'debug: after captions length:{caption_lengths}')

        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx].unsqueeze(-1)

        decoder_memory = self.input_to_encoded(encoder_out) # bs, 49, embed_size

        # ! 在encoder处加入了位置编码
        decoder_memory = self.pe(decoder_memory)
        
        # print(f'debug: decoder_memory:{decoder_memory.shape}')

        # print(f'debug: h:{h.shape}, c:{c.shape}')
        assert not torch.any(torch.isnan(encoded_captions)), "Nan happen in en_cap!!!"
        embeded_captions = self.embedding_net(encoded_captions[:, :-1] / math.sqrt(self.output_size))
        assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in em_cap1!!!"
        embeded_captions = self.pe(embeded_captions)
        assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in em_cap2!!!"

        decoder_length = (caption_lengths - 1)
        # print(f'debug: decoder length:{decoder_length}')
        
        predicts = (torch.ones((bs, max(decoder_length), self.output_size)) * -1).to(device)

        for t in range(max(decoder_length)):
            working_index = sum([l > t for l in decoder_length])
            decoder_memory = decoder_memory[:working_index]
            decoder_tgt = embeded_captions[:working_index, :t+1]
            # print(f'debug: decoder_tgt:{decoder_tgt.shape}')
            out_decoder = self.decoder(decoder_tgt, decoder_memory) # bs, t+1, embed_size

            assert not torch.any(torch.isnan(decoder_memory)), "Nan happen in memory!!!"
            assert not torch.any(torch.isnan(decoder_tgt)), "Nan happen in tgt!!!"
            assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder!!!"

            out_decoder = self.output_net(out_decoder[:, -1])
            assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder1!!!"
            # print(f'debug: out_lstm:{out_lstm.shape}, predict:{predicts[:working_index, t].shape}')
            predicts[:working_index, t] = out_decoder

        # Generate a mask for parallel decoding
        # mask = nn.Transformer().generate_square_subsequent_mask(max(decoder_length)).to(device)
        # print(f'debug: mask:{mask.shape}')

        # out_decoder = self.decoder(embeded_captions, decoder_memory, tgt_mask=mask)  # bs, seq_len, embed_size
        # out_decoder = self.output_net(out_decoder)
        # print(f'debug: length:{max(decoder_length)}, out_decoder:{out_decoder.shape}')
        # predicts[:, :out_decoder.shape[1]] = out_decoder
        # assert not torch.any(torch.isnan(decoder_memory)), "Nan happen in memory!!!"
        # assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in tgt!!!"
        # assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder!!!"

        return predicts, encoded_captions, decoder_length, sort_idx


    # mask
    # def forward(self, encoder_out, encoded_captions = None, caption_lengths = None):
    #     bs = encoder_out.shape[0]
    #     device = encoder_out.device
    #     # print(f'debug: device in train {device}')

    #     # sort by the length for the ease of paralleling
    #     # print(f'debug: captions length:{caption_lengths}')
    #     caption_lengths, sort_idx = torch.sort(caption_lengths, dim=-1, descending=True)
    #     # print(f'debug: after captions length:{caption_lengths}')

    #     encoder_out = encoder_out[sort_idx]
    #     encoded_captions = encoded_captions[sort_idx].unsqueeze(-1)

    #     decoder_memory = self.input_to_encoded(encoder_out) # bs, 49, embed_size
        
    #     # print(f'debug: decoder_memory:{decoder_memory.shape}')

    #     # print(f'debug: h:{h.shape}, c:{c.shape}')
    #     assert not torch.any(torch.isnan(encoded_captions)), "Nan happen in en_cap!!!"
    #     embeded_captions = self.embedding_net(encoded_captions[:, :-1] / math.sqrt(self.output_size))
    #     assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in em_cap1!!!"
    #     embeded_captions = self.pe(embeded_captions)
    #     assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in em_cap2!!!"

    #     decoder_length = (caption_lengths - 1)
    #     # print(f'debug: decoder length:{decoder_length}')
        
    #     predicts = (torch.ones((bs, max(decoder_length), self.output_size)) * -1).to(device)

    #     # for t in range(max(decoder_length)):
    #     #     working_index = sum([l > t for l in decoder_length])
    #     #     decoder_memory = decoder_memory[:working_index]
    #     #     decoder_tgt = embeded_captions[:working_index, :t+1]
    #     #     # print(f'debug: decoder_tgt:{decoder_tgt.shape}')
    #     #     out_decoder = self.decoder(decoder_tgt, decoder_memory) # bs, t+1, embed_size

    #     #     assert not torch.any(torch.isnan(decoder_memory)), "Nan happen in memory!!!"
    #     #     assert not torch.any(torch.isnan(decoder_tgt)), "Nan happen in tgt!!!"
    #     #     assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder!!!"

    #     #     out_decoder = self.output_net(out_decoder[:, -1])
    #     #     assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder1!!!"
    #     #     # print(f'debug: out_lstm:{out_lstm.shape}, predict:{predicts[:working_index, t].shape}')
    #     #     predicts[:working_index, t] = out_decoder

    #     # Generate a mask for parallel decoding
    #     mask = nn.Transformer().generate_square_subsequent_mask(max(decoder_length)).to(device)
    #     print(f'debug: mask:{mask.shape}')

    #     out_decoder = self.decoder(embeded_captions, decoder_memory, tgt_mask=mask)  # bs, seq_len, embed_size
    #     out_decoder = self.output_net(out_decoder)
    #     print(f'debug: length:{max(decoder_length)}, out_decoder:{out_decoder.shape}')
    #     predicts[:, :out_decoder.shape[1]] = out_decoder
    #     assert not torch.any(torch.isnan(decoder_memory)), "Nan happen in memory!!!"
    #     assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in tgt!!!"
    #     assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder!!!"

    #     return predicts, encoded_captions, decoder_length, sort_idx

    # interface for test (generation)
    # todo: 在encode处除了个scale
    def generate(self, encoder_out, temperature = 0, top_p = 1):
        bs = encoder_out.shape[0]
        device = encoder_out.device

        scale = math.sqrt(self.output_size)

        working_index = torch.arange(bs).to(device)

        encoded_input = self.input_to_encoded(encoder_out)

        # ! 在encoder处加入了位置编码
        encoded_input = self.pe(encoded_input)

        # init selected tokens
        max_length = self.max_length
        selected_tokens = torch.zeros((bs, max_length, 1)).to(device)
        selected_tokens[:,0] = self.tokenizer.start_token_idx

        selected_embeded_tokens = torch.zeros((bs, max_length, self.embed_size)).to(device)
        selected_embeded_tokens[:, 0] = self.pe(self.embedding_net(selected_tokens[:, 0] / scale),pos=0)

        cur_len = 1
        while working_index.shape[0] > 0:
            decoder_memory = encoded_input[working_index]
            decoder_tgt = selected_embeded_tokens[working_index, :cur_len]
            # print(f'debug: memo:{decoder_memory.shape}, tgt:{decoder_tgt.shape}')
            out_decoder = self.decoder(decoder_tgt, decoder_memory)

            out_decoder = self.output_net(out_decoder[:, -1])
            if temperature > 0:
                probs = torch.softmax(out_decoder / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
                # print(f'debug: next_token:{next_token.shape}')
            else:
                next_token = torch.argmax(out_decoder, dim=-1).unsqueeze(-1)
            selected_tokens[working_index, cur_len, :] = next_token.float()
            selected_embeded_tokens[working_index, cur_len, :] = self.pe(self.embedding_net(selected_tokens[working_index, cur_len, :] / scale), pos=cur_len)
            # print(f'debug: se:{selected_tokens.shape}, seem:{selected_embeded_tokens.shape}')

            not_end_index = self.tokenizer.not_end(next_token).squeeze(-1)
            # print(f'debug: index:{not_end_index}')
            working_index = working_index[not_end_index]
            # print(f'debug: widx:{working_index}')
            cur_len += 1 
            if cur_len == max_length - 1:
                selected_tokens[working_index, cur_len] = self.tokenizer.end_token_idx
                break
        return selected_tokens.cpu().long().squeeze()


# todo: 加入top-p采样
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token