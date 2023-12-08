
import torch
from runner import Runner
from dataset import CocoDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from tensorboardX import SummaryWriter
import time
import os
from PIL import Image
import pickle
import json

# todo: encode是否要scale，训练模式

class Config(object):
    def __init__(self) -> None:
        self.max_epoch = 50
        self.lr_decoder = 1e-3
        
        self.device = torch.device('cuda:2')
        self.validate_period = 5
        self.batch_size = 32
        self.save_period = 5
        self.log_period = 30
        self.max_token_length = 50
        self.max_norm = 1

        self.task_num = 2
        self.image_folder = f'/home/chenjiacheng/neural_network/big2/dataset{self.task_num}/train/images'
        self.annotation_file = f'load_data/task{self.task_num}.json'
        self.run_name = f'load_data/task{self.task_num}'
        self.eval_image_folder = f'/home/chenjiacheng/neural_network/big2/dataset{self.task_num}/dev/images'
        self.eval_annotation_file = f'load_data/task{self.task_num}_eval.json'
        self.tokenizer_path = f'load_data/saved_tokenizer{self.task_num}.pkl'

        # checkpoint load path
        self.load_path = ''

        if self.task_num == 1:
            self.max_length = 200
        else:
            self.max_length = 250

if __name__ == '__main__':
    config = Config()
    config.run_name = "{}_{}".format(config.run_name, time.strftime("%Y%m%dT%H%M%S"))
    # # process dataloader


    # # training process
    # trainer = Trainer(config=config)

    # trainer.train()
    training = True
    image_folder = config.image_folder
    annotation_file = config.annotation_file
    # Define transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to a fixed size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])

    
    
    tb_logger = SummaryWriter(os.path.join("logs", config.run_name))

    
    if training:
        # Create an instance of the CocoDataset
        dataset = CocoDataset(image_folder, annotation_file, transform)

        tokenizer = Tokenizer(dataset.vocab)
        if training:
            json_data = json.dumps(tokenizer.vocab)
            with open('saved_vocab.json','w') as f:
                f.write(json_data)
        # print(tokenizer.vocab)
        print(f'debug: max length: {dataset.max_length}')

        # save pre_trained tokenizer
        with open(config.tokenizer_path,'wb') as f:
            pickle.dump(tokenizer,f,-1)
            print("Successfully saved tokenizer!")
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        eval_dataset = CocoDataset(config.eval_image_folder, config.eval_annotation_file,transform)
        eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size,shuffle=False)
        print(f'debug: evalset max length: {eval_dataset.max_length}')
        # training process
        trainer = Runner(config=config, tokenizer=tokenizer)

        trainer.train(dataloader, eval_dataloader,tb_logger=tb_logger)
    else:
        # load tokenizer
        with open(config.tokenizer_path,'rb') as f:
            tokenizer = pickle.load(f)
        # training process
        trainer = Runner(config=config, tokenizer=tokenizer)
        # load model
        trainer.load(config.load_path)

        # todo
        folder_path = "test2017"
        images = []
        image_ids = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):  # 可以根据需要添加其他图像格式
                image_ids.append(filename)
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path).convert("RGB")
                image = transform(image)
                images.append(image)
        
        images = torch.stack(images, 0)
        print(f'debug: {len(images)}, {images.shape}')

        tokens = trainer.rollout(images)
        print(f'debug: out_tokens:{tokens}')
        
        decoded_tokens = tokenizer.decode(tokens)

        for id, text in zip(image_ids, decoded_tokens):
            print(f'id:{id}, caption:{text}')

        # print(tokenizer.idx_to_word.get(6))


