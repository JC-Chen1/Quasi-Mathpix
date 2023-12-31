
import torch
from runner import Runner
from dataset import CocoDataset, CocoDataset_test
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from tensorboardX import SummaryWriter
import time
import os
from PIL import Image
import pickle
import json
import numpy as np


class Config(object):
    def __init__(self) -> None:
        self.start_epoch = 0
        self.max_epoch = 100
        self.lr_decoder = 1e-3
        self.lr_min = 1e-4
        self.lr_decay=pow((self.lr_min/self.lr_decoder),1/(self.max_epoch))
        
        self.device = torch.device('cuda:2')
        self.validate_period = 5
        self.batch_size = 32
        self.save_period = 5
        self.log_period = 30
        self.max_token_length = 50
        self.max_norm = 1

        self.selected_model = 'transformer'
        self.task_num = 1
        self.train_mode = 'full'
        self.image_folder = f'/home/chenjiacheng/neural_network/big2/dataset{self.task_num}/train/images'
        self.annotation_file = f'load_data/task{self.task_num}.json'
        self.run_name = f'full_resnet_task{self.task_num}'
        self.eval_image_folder = f'/home/chenjiacheng/neural_network/big2/dataset{self.task_num}/dev/images'
        self.eval_annotation_file = f'load_data/task{self.task_num}_eval.json'
        

        
        # checkpoint load path
        self.load_path = None

        # for test
        self.id_path = f'/home/chenjiacheng/neural_network/big2/dataset{self.task_num}/test_ids.txt'
        self.test_image_folder = f'/home/chenjiacheng/neural_network/big2/dataset{self.task_num}/test/images'

        if self.task_num == 1:
            self.max_length = 200
        else:
            self.max_length = 250

if __name__ == '__main__':
    config = Config()
    
    # set seed
    # torch.manual_seed(42)
    # np.random.seed(42)
    training = True
    only_eval = False
    if (not only_eval) and training:
        config.run_name = "{}_{}".format(config.run_name, time.strftime("%Y%m%dT%H%M%S"))
    else:
        # to configure
        config.run_name = 'full_resnet_task1_20231214T165031'

    config.test_out_path = f'saved_data/{config.run_name}/'
    config.tokenizer_path = f'load_data/{config.run_name}-saved_tokenizer{config.task_num}.pkl'
    # # process dataloader
    assert config.selected_model in ['lstm','transformer'], 'The seleted model is not supported yet!!'
    assert config.train_mode in ['full', 'partial'], 'Unsupported training mode!!'
    # # training process
    # trainer = Trainer(config=config)

    # trainer.train()
    
    image_folder = config.image_folder
    annotation_file = config.annotation_file
    # Define transformations to be applied to the images
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize image to a fixed size
    #     transforms.ToTensor(),  # Convert image to tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    # ])

    transform = transforms.Compose([
        transforms.Resize((40, 240)),  # Resize image to a fixed size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    
    if training:
        tb_logger = SummaryWriter(os.path.join("logs", config.run_name))

    
    if training:
        # Create an instance of the CocoDataset
        dataset = CocoDataset(image_folder, annotation_file, transform)
        if not only_eval:
            tokenizer = Tokenizer(dataset.vocab)
            # save pre_trained tokenizer
            with open(config.tokenizer_path,'wb') as f:
                pickle.dump(tokenizer,f,-1)
                print("Successfully saved tokenizer!")
        else:
            with open(config.tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
        # with open('2.pkl','wb') as f:
        #     pickle.dump(dataset.vocab, f, -1)
        # torch.rand()
        # if training:
        #     json_data = json.dumps(tokenizer.vocab)
        #     with open('saved_vocab.json','w') as f:
        #         f.write(json_data)
        
        # print(tokenizer.vocab)
        print(f'debug: max length: {dataset.max_length}')

        
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        eval_dataset = CocoDataset(config.eval_image_folder, config.eval_annotation_file,transform)
        eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size,shuffle=False)
        print(f'debug: evalset max length: {eval_dataset.max_length}')

        test_dataset = CocoDataset_test(config.test_image_folder, config.id_path, transform)
        test_dataloader = DataLoader(test_dataset,batch_size=config.batch_size, shuffle=False)

        # training process
        trainer = Runner(config=config, tokenizer=tokenizer)
        

        # resume training
        if config.load_path is not None:
            trainer.load(config.load_path)

        trainer.train(dataloader, eval_dataloader, test_dataloader,tb_logger=tb_logger, only_eval=only_eval)
    else:
        # load tokenizer
        with open(config.tokenizer_path,'rb') as f:
            tokenizer = pickle.load(f)
        # training process
        trainer = Runner(config=config, tokenizer=tokenizer)
        # load model
        trainer.load(config.load_path)

        test_dataset = CocoDataset_test(config.test_image_folder, config.id_path, transform)
        dataloader = DataLoader(test_dataset,batch_size=config.batch_size, shuffle=False)
        trainer.test(dataloader, './', 1)