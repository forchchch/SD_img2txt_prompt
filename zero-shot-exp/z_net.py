import torch
import torch.nn as nn

class Downnet(nn.Module):
    def __init__(self, img_channel):
        super().__init__()
        ###img_latent(,4,64,64),text_latent(1, 77, 1024)
        self.down_conv1 = nn.Conv2d(img_channel, 8, (3,3), padding=0, stride=2)
        self.down_batchnorm1 = nn.BatchNorm2d(8)
        self.down_conv2 = nn.Conv2d(8, 16, (3,3), padding=0, stride=2)
        self.down_batchnorm2 = nn.BatchNorm2d(16)        

    def forward(self, img_latent):
        down_latent = self.down_batchnorm1( self.down_conv1(img_latent) )
        down_latent = self.down_batchnorm2( self.down_conv2(down_latent) )
        return down_latent    

class I2T_Prompt(nn.Module):
    def __init__(self, down_net, text_emb_size, linear_dim):
        super().__init__()
        ###img_latent(,4,64,64),text_latent(1, 77, 1024)
        self.text_emb_size = text_emb_size
        self.linear_dim = linear_dim
        self.down_net = down_net
        self.direct_transfrom = nn.Sequential(
                                    nn.Linear(linear_dim, text_emb_size),
                                    nn.ReLU(),
                                    nn.Linear(text_emb_size, text_emb_size)
                                    )
        
        
        
    def forward(self, img_latent):
        down_latent = self.down_net(img_latent)
        img_feature = down_latent.view(-1, self.linear_dim)
        i2t_prompt = self.direct_transfrom(img_feature)
        f_i2t_prompt = i2t_prompt.view(-1, 1, self.text_emb_size)        
        return f_i2t_prompt 

class Image_adapter(nn.Module):
    def __init__(self, emb_size=1024):
        super().__init__()
        ###img_latent(,4,64,64),text_latent(1, 77, 1024)
        self.transform = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )
        self.adapter_weight = 1.0
        
        
        
    def forward(self, img_feature):
        out = img_feature + self.adapter_weight*self.transform(img_feature)
        return out

class Multimodal_adapter(nn.Module):
    def __init__(self, emb_size=1024):
        super().__init__()
        self.emb_size = emb_size
        self.K = nn.Linear(emb_size, emb_size)
        self.Q = nn.Linear(emb_size, emb_size)
        self.V = nn.Linear(emb_size, emb_size)
        self.nonlinear = nn.Sequential(
            nn.Linear(emb_size, emb_size//2),
            nn.ReLU(),
            nn.Linear(emb_size//2, emb_size)
        )
        self.img_part = Image_adapter()
        
        
    def forward(self, img_feature, text_emb):
        # img_feature: (bs, 1 ,1024), text_emb: (bs, 77, 1024)
        img_query = self.Q( img_feature )
        text_key = self.K( text_emb )
        weight = (img_query*text_key).sum(2, keepdim=True)/self.emb_size**0.5 #bs,77, 1
        img_value = self.V( img_feature ).expand(-1, text_emb.size()[1], -1) #bs, 77, 1024
        text_img1 = text_emb + weight*img_value
        out_feature = text_img1 + self.nonlinear(text_img1)
        
        out_feature = out_feature + self.img_part( img_feature ) 
        return out_feature     