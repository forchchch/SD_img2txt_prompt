import torch
import torch.nn as nn

class Uncertainty_weighter(nn.Module):
    def __init__(self, hidden_size=(4,64,64)):
        super().__init__()
        self.weights = nn.Parameter( -0.5*torch.ones(hidden_size) )

        
        
    def forward(self, losses):
        final_loss = ( 1 / (2 * torch.exp(self.weights)) * losses + self.weights / 2 ).mean()

        return final_loss      

class Image_adapter(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mask = nn.Parameter(torch.zeros(hidden_size))
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, feature):
        out_feature = self.adapter(feature) + self.sigmoid(self.mask)*feature

        return out_feature     