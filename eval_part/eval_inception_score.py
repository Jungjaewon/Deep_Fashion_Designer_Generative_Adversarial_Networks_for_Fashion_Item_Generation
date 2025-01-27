import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from PIL import Image
import torch.utils.data
import torchvision.transforms as transforms
import os
import numpy as np

from torchvision import models as models
from scipy.stats import entropy


def inception_score(dataset, cuda=True, gpu='0', batch_size=32, splits=10, network='inception_v3', img_size=256):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(dataset)
    assert network in ['inception_v3']
    assert batch_size > 0 and N > batch_size
    assert img_size in [256, 299]

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=0,
                                             shuffle=False,)
    
    num_classes = 7
    model = models.inception_v3(pretrained=False, aux_logits=False)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, num_classes)

    if img_size == 256:
        ckpt = 'inception_014-98-model.ckpt'
    else:
        raise NotImplemented

    assert os.path.exists(ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage))

    model.eval()
    if torch.cuda.is_available():
        #model.cuda()
        model.to(f'cuda:{gpu}')

    def get_pred(x):
        #if resize:
        #    x = up(x)
        x = model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, num_classes))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype).to(f'cuda:{gpu}')
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    cnt = len(dataloader)

    return np.mean(split_scores), np.std(split_scores), cnt * batch_size


class Polyvore(data.Dataset):

    def __init__(self, folder_name, network='inception_v3', img_size=256):
        self.folder_name = folder_name

        assert network in ['inception_v3']
        assert img_size in [256, 299]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dataset = list()
        self.processing()

    def processing(self):

        file_list = os.listdir(self.folder_name)

        for file in file_list:
            self.dataset.append(os.path.join(self.folder_name,file))

        return

    def __getitem__(self, index):
        target_image = Image.open(os.path.join(self.dataset[index]))
        target_image = target_image.convert('RGB')

        return self.transform(target_image)

    def __len__(self):
        return len(self.dataset)