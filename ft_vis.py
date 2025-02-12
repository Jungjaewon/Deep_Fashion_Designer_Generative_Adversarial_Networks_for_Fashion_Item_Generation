import os
import torch
import torch.nn as nn
import os.path as osp
import torch.backends.cudnn as cudnn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

from model import Generator as G
from model import NewEncoder as New_E
from model import OldEmbedBlock
from data_loader import get_loader
from sklearn.manifold import TSNE
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# http://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html

class FTVIS(object):

    def __init__(self, config):
        """Initialize configurations."""

        assert torch.cuda.is_available()

        train_config = config['TRAINING_CONFIG']

        self.train_loader = get_loader(config, 'train_tsne')
        self.test_loader = get_loader(config, 'test_tsne')
        self.img_size    = train_config['IMG_SIZE']

        self.class_list = sorted(['tops', 'bags', 'bottoms', 'dresses', 'earrings', 'shoes', 'eyeglasses'])
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        self.markers = ['.', ',', 'o', 'v', '^', '<', '>']
        self.train_maker = 'o'
        self.test_maker = ','
        self.batch_size = train_config['BATCH_SIZE']

        # architecture configuration
        self.num_item = train_config['NUM_ITEM']
        self.num_target = train_config['NUM_TARGET']
        self.num_source = self.num_item - self.num_target

        self.latent_size = train_config['LATENT_SIZE']
        self.latent_v    = train_config['LATENT_V']
        
        self.encoder_last_ch = 128

        self.concat_mode = train_config['CONCAT_MODE']
        self.num_cls = 8

        self.f_add = train_config['F_ADD'] == 'True' if 'F_ADD' in train_config else False
        self.s_add = train_config['S_ADD'] == 'True' if 'S_ADD' in train_config else False

        self.gpu = train_config['GPU']
        self.gpu = torch.device(f'cuda:{self.gpu}')

        # Directory
        self.result_dir = osp.join('feature_result')
        os.makedirs(self.result_dir, exist_ok=True)
        self.build_model()

    def build_model(self):
        g_num_cls = self.num_cls if self.s_add else 0
        encoder_last_ch = self.encoder_last_ch
        self.G = G(base_channel=encoder_last_ch, n_cls=g_num_cls).to(self.gpu).eval()

        e_num_cls = self.num_cls if self.f_add else 0
        emd_ch = self.encoder_last_ch * self.num_source
        self.Emd = OldEmbedBlock(emd_ch, num_cls=e_num_cls)
        self.Emd = self.Emd.to(self.gpu).eval()
        
        self.E = New_E(target_channel=self.encoder_last_ch).to(self.gpu)
        self.E.eval()

        self.inception = models.inception_v3(aux_logits=False)
        n_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(n_features, 7)
        self.inception = self.inception.eval().to(self.gpu)

    def load_model(self):

        last_ckpt = osp.join('1000-model.ckpt')
        assert osp.exists(last_ckpt)

        ckpt_dict = torch.load(last_ckpt)
        print(f'ckpt_dict key : {ckpt_dict.keys()}')
        self.G.load_state_dict(ckpt_dict['G'])
        print('G is load')

        self.Emd.load_state_dict(ckpt_dict['Emd'])
        print('Emd is load')

        self.E.load_state_dict(ckpt_dict['E'])
        print('E is load')

        print(f'All model is laod from {last_ckpt}')

        ckpt = 'inception_014-98-model.ckpt'
        self.inception.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage))
        print('inception weight is load')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print("The number of parameters: {}".format(num_params))

    def get_latent_code(self, source_images):
        
        if self.latent_v == 3:
            return self.get_latent_code_old(source_images)
        else:
            raise NotImplemented

    def get_latent_code_old(self, source_image_list):

        latent_code_list = list()

        for b in range(source_image_list.size(0)):
            latent_code = list()
            for i in range(len(source_image_list[b])):
                #https://discuss.pytorch.org/t/what-is-the-difference-between-view-and-unsqueeze/1155
                #https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
                z = self.E(source_image_list[b][i].unsqueeze(0))
                z = torch.squeeze(z, 0)
                latent_code.append(z)
            latent_code_list.append(latent_code)

        if self.concat_mode == 0:
            for b in range(len(latent_code_list)):
                latent_code_list[b] = torch.cat(latent_code_list[b])
        else:
            raise Exception('Unspecified concat mode!')

        latent_code_list = torch.stack(latent_code_list)
        return latent_code_list

    def add_conditional(self, latent_code, c):
        if len(latent_code.size()) == 4:
            b, ch, _, _ = latent_code.size()
            latent_code = latent_code.contiguous().view(b, ch)
        c = F.one_hot(c, num_classes=self.num_cls).squeeze(1).to(self.gpu)
        latent_code = torch.cat((latent_code, c), dim=1)
        latent_code = latent_code.unsqueeze(len(latent_code.size()))
        latent_code = latent_code.unsqueeze(len(latent_code.size()))
        return latent_code

    def processing_latent_code(self, source_images, t_cat):

        latent_code = self.get_latent_code(source_images)
        if self.f_add:
            latent_code = self.add_conditional(latent_code, t_cat)
        
        latent_code = self.Emd(latent_code)
        
        if self.s_add:
            latent_code = self.add_conditional(latent_code, t_cat)

        return latent_code

    def run(self):

        self.load_model()

        train_feature_list = list()
        train_target_list = list()

        test_feature_list = list()
        test_target_list = list()

        total_feature_list = list()
        total_target_list = list()

        with torch.no_grad():
            for item in tqdm([['train', self.train_loader], ['test', self.test_loader]]):
                mode, data_loader = item
                for data in data_loader:
                    outfit_id, t_cat, t_idx, _, target_images, source_images = data
                    
                    target_images = target_images.to(self.gpu)
                    source_images = source_images.to(self.gpu)
                    t_cat = t_cat.to(self.gpu)

                    latent_code = self.processing_latent_code(source_images, t_cat)

                    prediction = self.inception(target_images)
                    _, pred_idx = torch.max(prediction, 1)
                    pred_idx = pred_idx.item()

                    if mode == 'train':
                        train_feature_list.append(latent_code.cpu().squeeze().detach().numpy())
                        train_target_list.append(pred_idx)
                        #train_target_list.append(t_cat.cpu().squeeze().detach().numpy())
                    elif mode == 'test':
                        test_feature_list.append(latent_code.cpu().squeeze().detach().numpy())
                        test_target_list.append(pred_idx)
                        #test_target_list.append(t_cat.cpu().squeeze().detach().numpy())

        total_feature_list.extend(train_feature_list)
        total_feature_list.extend(test_feature_list)
        total_target_list.extend(train_target_list)
        total_target_list.extend(test_target_list)

        train_feature_list = np.array(train_feature_list)
        train_target_list = np.array(train_target_list)

        test_feature_list = np.array(test_feature_list)
        test_target_list = np.array(test_target_list)

        num_train = len(train_feature_list)

        print(f'train_feature_list : {np.shape(train_feature_list)}')
        print(f'train_target_list : {np.shape(train_target_list)}')
        print(f'test_feature_list : {np.shape(test_feature_list)}')
        print(f'test_target_list : {np.shape(test_target_list)}')

        for mode in ['total', 'train', 'test']:

            if mode == 'train':
                feature_list = train_feature_list
                target_list = train_target_list
            elif mode == 'test':
                feature_list = test_feature_list
                target_list = test_target_list
            elif mode == 'total':
                feature_list = np.array(total_feature_list)
                target_list = np.array(total_target_list)
                #marker = 'o'
            else:
                raise NotImplemented

            for p in [10]:
                tsne = TSNE(n_components=2, random_state=0, verbose=1, perplexity=p, n_iter=500)
                feature_2d = tsne.fit_transform(feature_list)

                if mode == 'total':
                    train_2d = feature_2d[:num_train]
                    test_2d = feature_2d[num_train:]

                    for i, c, label in zip(range(7), self.colors, self.class_list):
                        plt.scatter(train_2d[train_target_list == i, 0], train_2d[train_target_list == i, 1],
                                    marker=self.train_maker, c=c, label=label)
                        plt.scatter(test_2d[test_target_list == i, 0], test_2d[test_target_list == i, 1],
                                    marker=self.test_maker, c=c, label=label)
                else:
                    for i, c, label in zip(range(7), self.colors, self.class_list):
                        plt.scatter(feature_2d[target_list == i, 0], feature_2d[target_list == i, 1], c=c, label=label)
                plt.legend()
                plt.savefig(osp.join(self.result_dir, f'{mode}_p{p}_tsne.png'))
                plt.close()