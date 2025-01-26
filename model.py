import torch.nn as nn
import torch

from CBAM import CBAM

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, LR=0.2, spec=False):
        super(ConvBlock, self).__init__()

        self.down = in_channels != out_channels
        stride = 2 if self.down else 1

        self.main = list()
        self.main.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.main.append(nn.InstanceNorm2d(out_channels, affine=True))
        self.main.append(nn.LeakyReLU(LR, inplace=True))

        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, LR=0.2):
        super(ResBlock, self).__init__()

        self.down = in_channels != out_channels
        stride = 2 if self.down else 1

        self.residual, self.shor_cut = list(), list()
        self.residual.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.residual.append(nn.InstanceNorm2d(out_channels, affine=True))
        self.residual.append(nn.LeakyReLU(LR, inplace=True))
        self.residual = nn.Sequential(*self.residual)
        
        self.shor_cut.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.shor_cut.append(nn.InstanceNorm2d(out_channels, affine=True))
        self.shor_cut.append(nn.LeakyReLU(LR, inplace=True))
        self.shor_cut = nn.Sequential(*self.shor_cut)

    def forward(self, x):
        return self.residual(x) + self.shor_cut(x)

class OldEmbedBlock(nn.Module):

    def __init__(self, input_channel, nLayer=6, LR=0.01, num_cls=0):
        super(OldEmbedBlock,self).__init__()
        self.num_layer = nLayer
        self.layers = list()

        self.layers.append(nn.Linear(input_channel + num_cls, input_channel))
        self.layers.append(nn.LeakyReLU(LR, inplace=True))

        for _ in range(nLayer - 1):
            self.layers.append(nn.Linear(input_channel, input_channel))
            self.layers.append(nn.LeakyReLU(LR, inplace=True))
        self.main = nn.Sequential(*self.layers)

    def forward(self, x):
        b, ch, _, _ = x.size()
        x = x.contiguous().view(b, ch)
        output = self.main(x)
        output = output.unsqueeze(len(output.size())).unsqueeze(len(output.size()))
        return output
    
class NewEncoder(nn.Module):
    def __init__(self, img_ch=3, start_channel=64, target_channel=128, nlayers=7, LR=0.01, att=True):
        super(NewEncoder, self).__init__()
        layers = list()
        print(f'E start_ch : {start_channel}, target_ch : {target_channel}, nLayer : {nlayers}, LR : {LR}, att : {att}')
        self.channel = start_channel
        
        # 256 -> 128
        layers.append(ResBlock(img_ch, self.channel, LR=LR))

        for _ in range(nlayers - 3): # 64 32 16 8
            layers.append(ConvBlock(self.channel, self.channel * 2, LR=LR))
            self.channel *= 2

        layers.append(CBAM(self.channel, 4))
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))

        # 4 -> 1
        layers.append(nn.Conv2d(self.channel, target_channel, kernel_size=4, stride=1, bias=False))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class Generator(nn.Module):
    def __init__(self, base_channel=128, nlayers=7, n_item=4, LR=0.01, res=False, n_cls=0):
        super(Generator, self).__init__()

        layers = list()
        self.n_item, self.res = n_item, res
        self.mchannel = base_channel * self.n_item * 4
        self.startChannel = base_channel * self.n_item

        layers.append(nn.ConvTranspose2d(self.startChannel + n_cls, self.mchannel, kernel_size=4, bias=False))
        layers.append(nn.InstanceNorm2d(self.mchannel, affine=True))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        for _ in range(nlayers - 2):
            layers.append(nn.ConvTranspose2d(self.mchannel, self.mchannel // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(self.mchannel // 2, affine=True))
            layers.append(nn.LeakyReLU(LR, inplace=True))
            self.mchannel = self.mchannel // 2

        layers.append(nn.ConvTranspose2d(self.mchannel, 3, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        return self.main(z)