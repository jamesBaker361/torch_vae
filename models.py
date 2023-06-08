import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

def calculate_same_padding(input_size, kernel_size, stride=1, dilation=1):
    output_size = (input_size + stride - 1) // stride
    padding = max(0, (output_size - 1) * stride + dilation * (kernel_size - 1) + 1 - input_size)
    padding_before = padding // 2
    padding_after = padding - padding_before
    return padding_before, padding_after

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
class Encoder(nn.Module):
    def __init__(self, shared, unshared):
        super(Encoder, self).__init__()
        self.shared=shared
        self.unshared=unshared

    def forward(self,x):
        x=self.unshared(x)
        mu, logvar= self.shared(x)
        return mu, logvar
    

class UnsharedEncoder(nn.Module):
    def __init__(self,out_channels,layers=2):
        super(UnsharedEncoder, self).__init__()
        start_channels=out_channels*(2**(layers-1))
        model_layers=[
            nn.Conv2d(3,start_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ]
        while start_channels>out_channels:
            model_layers+=[
                nn.Conv2d(start_channels,start_channels//2, kernel_size=3, stride=2, padding=1),
                nn.ReLU()]
            start_channels=start_channels//2
        self.model=nn.Sequential(
            *model_layers
        )

    def forward(self,x):
        return self.model(x)
    
class SharedEncoder(nn.Module):
    def __init__(self,in_channels,image_dim):
        super(SharedEncoder, self).__init__()
        input_nodes=in_channels * image_dim * image_dim
        self.fc_mu = nn.Linear(input_nodes, 10)
        self.fc_logvar = nn.Linear(input_nodes, 10)

    def forward(self,x):
        return self.fc_mu(x),self.fc_logvar(x)