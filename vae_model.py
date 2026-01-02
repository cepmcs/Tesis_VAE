import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularVAE(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, latent_size=128):
        super(MolecularVAE, self).__init__()
        
        # --- 1. ENCODER ---
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        
        # --- 2. DECODER ---
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # x shape: [batch, seq_len]
        
        # --- ENCODING ---
        embed = self.embedding(x)
        _, h = self.encoder_rnn(embed)
        h = h.squeeze(0) 
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # --- DECODING ---
        
        # 1. Preparar estado oculto inicial con z
        h_decoder = self.decoder_input(z).unsqueeze(0)
        
        # 2. Preparar entradas del decoder (shifted inputs)
        # Creamos una columna de ceros (Token de inicio simulado)
        batch_size, seq_len = x.size()
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
        
        # Concatenamos el cero al principio y quitamos la última letra de x
        # Así mantenemos el mismo largo (120)
        decoder_inputs = torch.cat([start_token, x[:, :-1]], dim=1)
        
        # embeddeamos la entrada desplazada
        embed_decoder = self.embedding(decoder_inputs)
        
        # Pasamos por la RNN
        out, _ = self.decoder_rnn(embed_decoder, h_decoder)
        prediction = self.fc_out(out) 
        
        return prediction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, kl_weight):
    vocab_size = recon_x.size(-1)
    
    # Cross Entropy
    recon_loss = F.cross_entropy(
        recon_x.reshape(-1, vocab_size), 
        x.reshape(-1), 
        ignore_index=0, 
        reduction='sum'
    )
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + (kl_weight * kld), recon_loss, kld