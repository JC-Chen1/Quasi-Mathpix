import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
from utils import sample_top_p

class Encoder4lstm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 256)
        self.relu = nn.LeakyReLU()

    def forward(self, image):
        encoded_out = None
        with torch.no_grad():
            encoded_out = self.resnet(image)
        # only tune the fc network
        out = self.fc(encoded_out)
        out = self.relu(out)
        return out
    
class Decoder_lstm(nn.Module):
    def __init__(self, tokenizer, max_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.output_size = tokenizer.vocab_size
        self.max_length = max_length

        self.input_size =256
        self.hidden_size = 512
        self.num_layer = 1
        self.lstm = nn.LSTM(
            input_size = self.input_size, 
            hidden_size = self.hidden_size, 
            num_layers = self.num_layer, 
            batch_first = True)
        self.output_net = nn.Linear(self.hidden_size, self.output_size)
        self.embedding_net = nn.Linear(1, self.input_size)

        self.init_h_net = nn.Linear(self.input_size, self.hidden_size)
        self.init_c_net = nn.Linear(self.input_size, self.hidden_size)
        
    
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


        # h_0 = torch.zeros(1, bs, self.hidden_size).to(device)
        # c_0 = torch.zeros_like(h_0)

        h = self.init_h_net(encoder_out).unsqueeze(0)
        c = self.init_c_net(encoder_out).unsqueeze(0)

        # print(f'debug: h:{h.shape}, c:{c.shape}')
        embeded_captions = self.embedding_net((2 * encoded_captions - self.output_size) / self.output_size)

        decoder_length = (caption_lengths - 1)
        # print(f'debug: decoder length:{decoder_length}')
        
        predicts = (torch.ones((bs, max(decoder_length), self.output_size)) * -1).to(device)

        for t in range(max(decoder_length)):
            working_index = sum([l > t for l in decoder_length])
            x_in = embeded_captions[:working_index, t].unsqueeze(1)
            h = h[:, :working_index]
            c = c[:, :working_index]
            # print(f'debug: x_in:{x_in.shape}')
            out_lstm, (h, c) = self.lstm(x_in, (h, c))
            out_lstm = self.output_net(out_lstm).squeeze(1)
            # no need to softmax
            # out_lstm = torch.softmax(out_lstm, -1)
            # print(f'debug: out_lstm:{out_lstm.shape}, predict:{predicts[:working_index, t].shape}')
            predicts[:working_index, t] = out_lstm
        
        return predicts, encoded_captions, decoder_length, sort_idx
        

    def generate(self, encoder_out, temperature = 0, top_p = 1):
        bs = encoder_out.shape[0]
        device = encoder_out.device

        cur_len = 0

        h = self.init_h_net(encoder_out).unsqueeze(0)
        c = self.init_c_net(encoder_out).unsqueeze(0)
        
        working_index = torch.arange(bs).to(device)

        # init selected tokens
        max_length = self.max_length
        selected_tokens = torch.zeros((bs, max_length, 1)).to(device)
        selected_tokens[:,0] = self.tokenizer.start_token_idx


        while working_index.shape[0] > 0:
            x_in = (2 * selected_tokens[working_index, cur_len] - self.output_size) / self.output_size
            # print(f'debug: x_in:{x_in.shape}')
            x_in = self.embedding_net(x_in).unsqueeze(1)
            # print(f'debug: x_in:{x_in.shape}')

            out_lstm, (h, c) = self.lstm(x_in, (h, c))
            out_lstm = self.output_net(out_lstm).squeeze(1)

            if temperature > 0:
                probs = torch.softmax(out_lstm / temperature, -1)
                select_tok = sample_top_p(probs, top_p)
            else:
                select_tok = torch.argmax(out_lstm, -1).unsqueeze(-1)
            # print(f'debug: select_tok{select_tok}')
            selected_tokens[working_index, cur_len+1, :] = select_tok.float()


            not_end_index = self.tokenizer.not_end(select_tok).squeeze(-1)
            # print(f'debug: index:{not_end_index}')
            working_index = working_index[not_end_index]
            # print(f'debug: working_idx:{working_index}')
            h = h[:, not_end_index]
            c = c[:, not_end_index]

            cur_len += 1

            if cur_len == max_length - 2:
                selected_tokens[working_index, cur_len+1] = self.tokenizer.end_token_idx
                break
        
        return selected_tokens.cpu().long().squeeze()


    