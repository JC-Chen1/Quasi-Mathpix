from torch.nn.utils.rnn import pack_padded_sequence
from model.transformer import Encoder4transformer, Decoder_transformer
from model.lstm import Encoder4lstm, Decoder_lstm

import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from utils import clip_grad_norms
import time
from score import *

Get_encoder = {
    'lstm': Encoder4lstm,
    'transformer': Encoder4transformer
}

Get_decoder = {
    'lstm': Decoder_lstm,
    'transformer': Decoder_transformer
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Runner(object):
    def __init__(self, config, tokenizer) -> None:
        self.config = config
        self.encoder = Get_encoder.get(config.selected_model)(config.train_mode)
        # self.decoder = Decoder(tokenizer)

        self.decoder = Get_decoder.get(config.selected_model)(tokenizer, config)

        self.tokenizer = tokenizer

        if config.selected_model == 'transformer':
            self.optimizer = torch.optim.Adam(
                [{'params':self.encoder.parameters(), 'lr': self.config.lr_decoder}] +
                [{'params':self.decoder.parameters(), 'lr': self.config.lr_decoder}]
            )
        elif config.selected_model == 'lstm':
            self.optimizer = torch.optim.Adam(
                [{'params':self.encoder.parameters(), 'lr': self.config.lr_decoder}] +
                [{'params':self.decoder.parameters(), 'lr': self.config.lr_decoder}]
            )

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay, last_epoch=-1,)

        # debug
        # print(f'debug: 参数量:{count_parameters(self.decoder)}')
        # torch.rand()
        # move to device
        device = config.device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        self.criterion = nn.CrossEntropyLoss().to(device)


    def train(self, dataloader, eval_dataloader, test_dataloader, tb_logger = None, only_eval = False):
        max_epoch = self.config.max_epoch
        start_epoch = self.config.start_epoch

        self.loss_list = []
        self.learning_step = 0

        # only eval
        if only_eval:
            self.eval(eval_dataloader,-1,tb_logger)
        else:
            for epoch in range(start_epoch, max_epoch+1):
                print(f'Training epoch [{epoch}/{max_epoch}]...')
                epoch_loss_list = []
                for bat_image, bat_caption, caption_length in tqdm(dataloader,desc=f"run_name:{self.config.run_name}-epoch[{epoch}/{self.config.max_epoch}], lr={round(self.optimizer.param_groups[0]['lr'], 5)}"):
                    bat_loss = self.train_batch(bat_image, bat_caption, caption_length)
                    self.learning_step+=1
                    self.loss_list.append(bat_loss)
                    epoch_loss_list.append(bat_loss)
                    if (tb_logger is not None) and (self.learning_step % self.config.log_period == 0):
                        tb_logger.add_scalar("loss", bat_loss, self.learning_step)
                # self.save_checkpoint(f"saved_model/transformer/{self.config.run_name}", epoch)
                
                print(f'Loss of epoch [{epoch}/{max_epoch}] is {np.mean(epoch_loss_list)}, current learning steps:{self.learning_step}')
                
                if epoch % 5 == 0:
                    self.save_checkpoint(f"saved_model/transformer/{self.config.run_name}", epoch)
                    self.eval(eval_dataloader,epoch,tb_logger)
                    if epoch >= 50:
                        self.test(test_dataloader, self.config.test_out_path, epoch)

                self.lr_scheduler.step()
            loss_path = f"saved_model/transformer/{self.config.run_name}"
            with open(f"{loss_path}/loss-{epoch}.pt", 'wb') as f:
                torch.save(self.loss_list, f)
        tb_logger.close()
        # save model 
        # self.save_checkpoint("saved_model", max_epoch)
        # self.load("saved_model/epoch-20.pt")

    def train_batch(self, image, caption, length):
        self.encoder.train()
        self.decoder.train()


        device = self.config.device
        image = image.to(device)
        encoded_caption, target_idxs = self.tokenizer.encode(caption)
        # print(f'debug: {encoded_caption.shape}')
        encoded_caption = torch.FloatTensor(encoded_caption).to(device)
        target_idxs = torch.tensor(target_idxs, dtype=torch.long).to(device)
        length = length.to(device)
        # print(f'debug: length:{length}')

        encoded_img = self.encoder(image)
        # print(f'debug: encoded_img:{encoded_img.shape}')

        scores, encoded_caption, decode_length, sort_idx = self.decoder(encoded_img, encoded_caption, length)
        # print(f'debug: scores:{scores.shape}')
        # scores = scores.permute(0, 2, 1)
        targets = target_idxs[sort_idx][:, 1:]
        
        scores = pack_padded_sequence(scores, decode_length.cpu(), batch_first=True).data
        targets = pack_padded_sequence(targets, decode_length.cpu(), batch_first=True).data.squeeze()
        # print(f'debug: scores:{scores.shape}, targets:{targets.shape}')
        # print(f'debug: scores:{scores.shape}, targets:{targets.shape}')
        assert not torch.any(torch.isnan(scores)), "Nan happen in scores!!!"
        assert not torch.any(torch.isnan(targets)), "Nan happen in targets!!!"

        loss = self.criterion(scores, targets.long())

        assert not torch.isnan(loss), "Nan happen in loss!!!"
        # print(f'debug: loss:{loss}')
        
        self.optimizer.zero_grad()
        loss.backward()

        clip_grad_norms(self.optimizer.param_groups, max_norm=self.config.max_norm)

        self.optimizer.step()

        return loss.detach().cpu()
    

    def save_checkpoint(self, saving_path, epoch):
        print('Saving model...')
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        if self.config.selected_model == 'transformer':
            torch.save(
                {
                    'encoder': self.encoder.state_dict() if self.config.train_mode == 'full' else None,
                    'decoder': self.decoder.state_dict(),
                    'decoder_optimizer': self.optimizer.state_dict(),
                },
                os.path.join(saving_path, f'epoch-{epoch}.pt')
            )
        elif self.config.selected_model == 'lstm':
            torch.save(
                {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'decoder_optimizer': self.optimizer.state_dict(),
                },
                os.path.join(saving_path, f'epoch-{epoch}.pt')
            )

        print('Successfully save model...')

    def load(self, path):
        save_dict = torch.load(path)
        if self.config.selected_model == 'transformer':
            if self.config.train_mode == 'full':
                self.encoder.load_state_dict(save_dict['encoder'])
            self.decoder.load_state_dict(save_dict['decoder'])
            self.optimizer.load_state_dict(save_dict['decoder_optimizer'])
        elif self.config.selected_model == 'lstm':
            self.encoder.fc.load_state_dict(save_dict['encoder'])
            self.decoder.load_state_dict(save_dict['decoder'])
            self.optimizer.load_state_dict(save_dict['decoder_optimizer'])
        print(' [*] Loading data from {}'.format(path))


    # only for evalutional
    # return score
    def eval(self, dataloader, epoch, tb_logger=None):
        self.encoder.eval()
        self.decoder.eval()

        # 是否要分词？
        refers = []
        hypos = []
        with torch.no_grad():
            device = self.config.device
            for images, refer, length in tqdm(dataloader, desc="Evaluating..."):
                images = images.to(device)
                encoded_img = self.encoder(images)

                # todo: add temperature, no top_p
                predicts = self.decoder.generate(encoder_out=encoded_img, temperature=0, top_p=0.25)

                predicts_string = self.tokenizer.decode(predicts)
                refers.extend(refer)
                hypos.extend(predicts_string)
        
        score1, score2, score3, t_score = total_score(refers, hypos)
        print(f'eval scores: BLEU Score:{score1}, Edit Distance Score:{score2}, Exact Match Score:{score3}')
        if tb_logger is not None:
            tb_logger.add_scalar("eval/score1", score1, epoch)
            tb_logger.add_scalar("eval/score2", score2, epoch)
            tb_logger.add_scalar("eval/score3", score3, epoch)
            tb_logger.add_scalar("eval/t_score", t_score, epoch)
        

    def test(self, test_dataloader, output_path, epoch):
        self.encoder.eval()
        self.decoder.eval()
        
        device = self.config.device
        save_list = []
        save_ids = []
        with torch.no_grad():
            for  ids, images in tqdm(test_dataloader,desc="Testing..."):
                images = images.to(device)
                encoded_img = self.encoder(images)

                # todo: add temperature
                predicts = self.decoder.generate(encoder_out=encoded_img)
                predicts_string = self.tokenizer.decode(predicts)
                save_list.extend(predicts_string)
                save_ids.extend(ids)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(output_path + f"test_out-{epoch}.txt", 'w') as f:
            for id, str in zip(save_ids, save_list):
                f.write(id + ": " + str + '\n')
