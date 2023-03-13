import torch
import torch.nn as nn

class Downnet(nn.Module):
    def __init__(self, img_channel):
        super().__init__()
        ###img_latent(,4,64,64),text_latent(1, 77, 1024)
        self.down_conv = nn.Conv2d(img_channel, 3, (4,4), padding=0, stride=3)
        self.down_batchnorm = nn.BatchNorm2d(3)
        
        
        
    def forward(self, img_latent):
        down_latent = self.down_batchnorm( self.down_conv(img_latent) )
        return down_latent    

class I2T_Prompt(nn.Module):
    def __init__(self, down_net, text_emb_size, linear_dim):
        super().__init__()
        ###img_latent(,4,64,64),text_latent(1, 77, 1024)
        self.text_emb_size = text_emb_size
        self.linear_dim = linear_dim
        self.down_net = down_net
        self.direct_transfrom = nn.Linear(linear_dim, text_emb_size)
        self.fuse = nn.Sequential(
                    nn.Linear(2*text_emb_size, text_emb_size),
                    nn.ReLU(),
                    nn.Linear(text_emb_size, text_emb_size),
                    nn.ReLU(),
                    nn.Linear(text_emb_size, text_emb_size))
        
        
        
    def forward(self, img_latent, text_latent):
        down_latent = self.down_net(img_latent)
        img_feature = down_latent.view(-1, self.linear_dim)
        direct_i2t = self.direct_transfrom(img_feature)
        text_mean = torch.mean(text_latent, dim=1)
        fuse_i2t = self.fuse(torch.cat([direct_i2t, text_mean], dim=1))
        i2t_prompt = direct_i2t + fuse_i2t
        f_i2t_prompt = i2t_prompt.view(-1, 1, self.text_emb_size)        
        return f_i2t_prompt 