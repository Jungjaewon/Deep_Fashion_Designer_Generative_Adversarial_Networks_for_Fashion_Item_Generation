import torch
import os
import os.path as osp
import random
import glob
import pickle
import json
import copy

from torch.utils import data
from torchvision import transforms as T
from PIL import Image

class OutfitDataset(data.Dataset):
    """Dataset class for the Polyevore dataset."""

    def __init__(self, config, transform, mode='train'):
        """Initialize and preprocess the Polyevore dataset."""

        data_mode = mode.split('_')[0] if '_' in mode else mode

        self.image_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], data_mode)
        self.train_dir = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.transform, self.mode = transform, mode
        self.seed = config['TRAINING_CONFIG']['SEED']
        self.data_num = config['TRAINING_CONFIG']['DATA_NUM']
        self.neg_rate = float(config['TRAINING_CONFIG']['NEG_RATE'])
        self.neg_mode = config['TRAINING_CONFIG']['NEG_MODE']
        with open(config['TRAINING_CONFIG']['TAGGED_PLK'], 'rb') as fp:
            self.tagged_dict = pickle.load(fp)[data_mode]

        assert self.neg_mode in ['None', 'BLACK', 'FIRST']
        print(f'mode : {self.mode}, data_num : {self.data_num}, neg_rate : {self.neg_rate}')

        assert self.data_num in list(range(1, 6))

        self.dataset = list()

        self.outfit_list = glob.glob(osp.join(self.image_dir, '*'))
        random.shuffle(self.outfit_list)

        for outfit_path in self.outfit_list:
            t_idx_list = random.sample(list(range(1, 6)), self.data_num)
            for t_idx in t_idx_list:
                base_list = list(range(1, 6))
                base_list.remove(t_idx)
                assert len(base_list) == 4
                self.dataset.append([outfit_path, t_idx, base_list])

        print(f'the number of data : {len(self.dataset)}')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        outfit_path, t_idx, base_list = self.dataset[index]
        source_img_list, outfit_id = list(), osp.basename(outfit_path)
        t_cat = self.tagged_dict[f'{outfit_id}_{str(t_idx)}']['cate_idx'] + 1

        if self.mode == 'train' and random.uniform(0, 1) < self.neg_rate:
            target_path, neg_flag, label, t_cat = 'black.jpg', True, 0, 0
        else:
            target_path, neg_flag, label = osp.join(outfit_path, f'{t_idx}.jpg'), False, 1

        target_image = Image.open(target_path)
        target_image = self.transform(target_image.convert('RGB'))

        if self.mode == 'train' and neg_flag:
            random_sel = random.sample(self.outfit_list, len(base_list))
            idx_list = random.sample([f'{i}' for i in range(1, 6)], len(base_list))

            for idx, data in enumerate(zip(random_sel, idx_list)):
                outfit_path, s_idx = data

                temp_path = osp.join(outfit_path, f'{s_idx}.jpg')
                temp_image = Image.open(temp_path)
                temp_image = self.transform(temp_image.convert('RGB'))

                if idx == 0 and self.neg_mode == 'FIRST':
                    target_image = temp_image

                source_img_list.append(temp_image)
        else:
            for s_idx in base_list:
                temp_path = osp.join(outfit_path, f'{s_idx}.jpg')
                temp_image = Image.open(temp_path)
                temp_image = self.transform(temp_image.convert('RGB'))
                source_img_list.append(temp_image)

        source_img_list = torch.stack(source_img_list)
        outfit_id = torch.LongTensor([int(outfit_id)])
        t_idx = torch.LongTensor([int(t_idx)])
        t_cat = torch.LongTensor([int(t_cat)])
        label = torch.LongTensor([label])

        return outfit_id, t_cat, t_idx, label, target_image, source_img_list

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)

def get_loader(config, mode='train'):
    """Build and return a data loader."""
    transform = list()
    transform.append(T.Resize((config['TRAINING_CONFIG']['IMG_SIZE'], config['TRAINING_CONFIG']['IMG_SIZE'])))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform = T.Compose(transform)
    
    dataset = OutfitDataset(config, transform, mode)

    if mode == 'train':
        batch_size = config['TRAINING_CONFIG']['BATCH_SIZE']
    else:
        batch_size = 1

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True, pin_memory=True)
    return data_loader