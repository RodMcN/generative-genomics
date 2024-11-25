import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, data_dim=2500, latent_dim=512, dropout=0.25, num_classes=126, class_emb_dim=64):
        super().__init__()

        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_classes, class_emb_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(data_dim + class_emb_dim, 2048), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(2048),
            nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(1024),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(256),
            nn.Linear(256, latent_dim * 2) # outputs mu and logvar
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + class_emb_dim, 256), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(256),
            nn.Linear(256, 512), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(512),
            nn.Linear(512, 1024), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(1024),
            nn.Linear(1024, 2048), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(2048),
            # nn.Linear(2048, data_dim*2)
            nn.Linear(2048, data_dim)
        )



    # def forward(self, x, classes):
    #     class_emb = self.embedding(classes)

    #     x = torch.cat((x, class_emb), dim=-1)

    #     encoded = self.encoder(x)
    #     mu, logvar = torch.chunk(encoded, 2, dim=-1)

    #     std = torch.exp(0.5 * logvar)
    #     q = Normal(mu, std)
    #     z = q.rsample()

    #     z = torch.cat((z, class_emb), dim=-1)

    #     reconstructed = self.decoder(z)
    #     recon_mu, recon_logvar = torch.chunk(reconstructed, 2, dim=-1)

    #     return recon_mu, recon_logvar, q
    
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, classes):
        class_emb = self.embedding(classes)

        x = torch.cat((x, class_emb), -1)
        
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        
        z = torch.cat((z, class_emb), -1)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

    
    def generate(self, classes, device="cuda", seed=None):
        training = self.training
        self.eval()

        gen = None
        if seed is not None:
            if device.startswith('cuda'):
                gen = torch.Generator(device='cuda')  # Move generator to CUDA
            else:
                gen = torch.Generator()  # Default to CPU generator
            gen.manual_seed(seed)
        
        try:
            with torch.no_grad():
                z = torch.randn(len(classes), self.latent_dim, device=device, generator=gen).to(device)
                if not isinstance(classes, torch.Tensor):
                    classes = torch.LongTensor(classes)
                classes = classes.to(device)
                classes = self.embedding(classes)
                inpt = torch.cat((z, classes), -1)
                out = self.decoder(inpt)
                # out, _ = torch.chunk(out, 2, dim=-1)
                
        finally:
            if training:
                self.train()

        return out

    @classmethod
    def load(cls, state_dict=None, device=None):
        model = cls()
        model.load_state_dict(torch.load(state_dict))
        if device is not None:
            model.to(device)
        return model

# def vae_loss(recon_mu, recon_logvar, x, q, beta=1.0):
#     # Reconstruction loss as negative log-likelihood
#     recon_std = torch.exp(0.5 * recon_logvar)
#     recon_distribution = Normal(loc=recon_mu, scale=recon_std)
#     recon_loss = -recon_distribution.log_prob(x)
#     recon_loss = recon_loss.sum() / x.size(0)

#     # KL divergence
#     p = Normal(torch.zeros_like(q.loc), torch.ones_like(q.scale))
#     kl_loss = kl_divergence(q, p).sum() / x.size(0)
#     kl_loss = beta * kl_loss

#     # Total loss
#     loss = recon_loss + kl_loss
#     return loss, recon_loss.item(), kl_loss.item()

def vae_loss(reconstructed, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(reconstructed, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    kl_loss = beta * kl_loss
    loss = recon_loss + kl_loss
    return loss, recon_loss.item(), kl_loss.item()