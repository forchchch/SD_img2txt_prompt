import torch
import torch.nn as nn

class Uncertainty_weighter(nn.Module):
    def __init__(self, hidden_size=(4,64,64)):
        super().__init__()
        self.weights = nn.Parameter( -0.5*torch.ones(hidden_size) )

        
        
    def forward(self, losses):
        final_loss = ( 1 / (2 * torch.exp(self.weights)) * losses + self.weights / 2 ).mean()

        return final_loss      