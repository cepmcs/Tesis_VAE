import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularVAE(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, latent_size=128):
        super(MolecularVAE, self).__init__()
        
        # --- 1. ENCODER ---
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)       # Media μ
        self.fc_logvar = nn.Linear(hidden_size, latent_size)   # Log-varianza σ²
        
        # --- 2. DECODER ---
        self.decoder_input = nn.Linear(latent_size, hidden_size)  # z → h₀ del decoder
        self.decoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)          # Logits sobre vocabulario
        
    def reparameterize(self, mu, logvar):
        """Truco de reparametrización: z = μ + σ * ε, con ε ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # x: [batch, seq_len] con formato [SOS, t1, t2, ..., EOS, PAD, PAD, ...]
        
        # --- ENCODING ---
        embed = self.embedding(x)
        _, h = self.encoder_rnn(embed)
        h = h.squeeze(0) 
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # --- DECODING (Teacher Forcing) ---
        # Input del decoder: x[:, :-1] = [SOS, t1, ..., tn-1]
        # Target esperado:   x[:, 1:]  = [t1, t2, ..., EOS, PAD...]
        h_decoder = self.decoder_input(z).unsqueeze(0)    # Estado inicial h₀
        embed_decoder = self.embedding(x[:, :-1])          # Embeddings shifted
        out, _ = self.decoder_rnn(embed_decoder, h_decoder)
        prediction = self.fc_out(out)  # [batch, seq_len-1, vocab_size]
        
        return prediction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, kl_weight, pad_idx=0):
    """
    Loss del VAE: L = Recon + β·KL
    
    Args:
        recon_x: Logits del decoder [batch, seq_len-1, vocab_size]
        x: Secuencia original [batch, seq_len] → [SOS, t1, ..., EOS, PAD...]
        mu, logvar: Parámetros de la distribución latente q(z|x)
        kl_weight: Factor β para KL annealing
        pad_idx: Índice del token de padding (ignorado en cross-entropy)
    
    Returns:
        (loss_total, recon_loss, kl_loss)
    """
    vocab_size = recon_x.size(-1)
    target = x[:, 1:]  # Target sin SOS: [t1, t2, ..., EOS, PAD...]
    
    # Reconstrucción: Cross-Entropy sumada sobre todos los tokens no-padding
    recon_loss = F.cross_entropy(
        recon_x.reshape(-1, vocab_size), 
        target.reshape(-1), 
        ignore_index=pad_idx, 
        reduction='sum'
    )
    
    # KL Divergence: D_KL(q(z|x) || p(z)), con p(z) = N(0, I)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + (kl_weight * kld), recon_loss, kld