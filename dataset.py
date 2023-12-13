import json
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from collections import Counter
from tokenizer import Tokenizer
from gensim.models import Word2Vec

class CocoDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None):
        self.image_folder = image_folder
        self.annotation_file = annotation_file
        self.transform = transform

        # Load annotations
        self.annotations = self._load_annotations()

        # Build vocabulary
        self.vocab = self._build_vocab()
        # self.tokenizer = Tokenizer(self.vocab)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Load image
        image_id = self.annotations[index]['id']
        image_path = os.path.join(self.image_folder, f'{image_id}.png')
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        # Load caption and its length
        caption = self.annotations[index]['annotation']
        
        caption = '<start>' + ' ' + caption + ' ' + '<end>'
        caption_length = len(caption.split())

        # encoded_cap = self.tokenizer.encode(caption)
        # print(f'from dataset: {caption} encoded:{encoded_cap}')

        return image, caption, caption_length

    def _load_annotations(self):
        # Load annotations from JSON file
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations

    # 改成了word2vec
    def _build_vocab(self):
        # Extract captions
        captions = ['<start> ' + annotation['annotation'] + ' <end>' for annotation in self.annotations]

        # Tokenize captions
        tokenized_captions = [caption.split() for caption in captions]
        self.max_length = max([len(cap) for cap in tokenized_captions])

        vocab = Word2Vec(tokenized_captions, vector_size=100, window=5, min_count=1, workers=4)
        # # Count the frequency of each word
        # word_counts = Counter([word for caption in tokenized_captions for word in caption])

        # # Sort words by frequency in descending order
        # sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # # Create vocabulary
        # vocab = {
        #     '<pad>': 0,  # Padding token
        #     '<start>': 1,  # Start of sequence token
        #     '<end>': 2,  # End of sequence token
        #     '<unk>': 3  # Unknown token
        # }

        # # Assign unique integer identifiers to the words
        # for word, _ in sorted_words:
        #     if word not in vocab:
        #         vocab[word] = len(vocab)

        return vocab

# only image
class CocoDataset_test(Dataset):
    def __init__(self, image_folder, id_path, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.ids = self._load_ids(id_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Load image
        image_id = self.ids[index]
        image_path = os.path.join(self.image_folder, f'{image_id}.png')
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        return image_id, image

    def _load_ids(self, path):
        with open(path, 'r') as f:
            ids = f.read().split()
        return ids
        


if __name__ == '__main__':
    # Set the paths to the image folder and annotation file
    image_folder = 'train2017'
    annotation_file = 'annotations/captions_train2017.json'

    # Define transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to a fixed size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])

    # Create an instance of the CocoDataset
    dataset = CocoDataset(image_folder, annotation_file, transform)

    # Accessing the elements of the dataset
    image, caption, caption_length = dataset[0]
    print(f'image:{type(image)}, captions:{type(caption)}, len:{type(caption_length)}')