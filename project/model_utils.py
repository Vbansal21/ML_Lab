import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Precompute frequencies cis = 1 / (theta ^ (2i / dim))
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("freqs", freqs)

        # Precompute embeddings for all positions up to max_seq_len
        t = torch.arange(self.max_seq_len, device=device)
        freqs_cis = torch.outer(t, self.freqs)  # Shape (max_seq_len, dim / 2)
        # Convert to complex numbers: cos(t*f) + i*sin(t*f)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs_cis), freqs_cis))  # Shape (max_seq_len, dim / 2)

    def _apply_rotary_emb(self, x, freqs_cis):
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_reshaped)

        if x_complex.ndim == 4:  # (batch, seq_len, heads, dim/2)
            freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim/2)
        else:  # (batch, seq_len, dim/2)
            freqs_cis = freqs_cis.unsqueeze(0)  # (1, seq_len, dim/2)

        freqs_cis = freqs_cis.to(x_complex.device)
        x_rotated_complex = x_complex * freqs_cis
        x_rotated = torch.view_as_real(x_rotated_complex)
        x_out = x_rotated.flatten(start_dim=-2)
        return x_out.type_as(x)

class XPosEmbedding(nn.Module):
    """XPos Embedding: Applies RoPE rotation followed by element-wise exponential decay."""
    def __init__(self, dim, max_seq_len=2048, theta=10000.0, gamma=0.996, device=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.gamma = gamma  # Decay factor lambda < 1
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate RoPE for the rotation part
        self.rope = RotaryEmbedding(dim=dim, max_seq_len=max_seq_len, theta=theta, device=device)

        # Precompute decay factors: scale_m = gamma ** m applied uniformly across the dimension
        t = torch.arange(max_seq_len, device=device).float()
        scale = self.gamma ** t  # Shape: (max_seq_len,)
        # Reshape for broadcasting: (1, max_seq_len, 1)
        self.register_buffer("scale", scale.view(1, -1, 1))

    def forward(self, x, seq_dim=1):
        # x shape: (batch, seq_len, dim) or (batch, seq_len, heads, dim_per_head)
        seq_len = x.shape[seq_dim]

        if seq_len > self.max_seq_len:
            logging.warning(f"Sequence length {seq_len} exceeds XPosEmbedding max_seq_len {self.max_seq_len}. Clamping.")
            seq_len = self.max_seq_len
            raise ValueError(f"Sequence length {x.shape[seq_dim]} exceeds XPosEmbedding max_seq_len {self.max_seq_len}. Adjust max_seq_len parameter.")

        # 1. Apply RoPE rotation
        freqs_cis_seq = self.rope.freqs_cis[:seq_len]
        x_rotated = self.rope._apply_rotary_emb(x, freqs_cis_seq)

        # 2. Apply Exponential Decay Scaling
        scale_seq = self.scale[:, :seq_len, :].to(x_rotated.device)

        if x_rotated.ndim == 4:  # If applied after head splitting (B, S, H, D/H)
            scale_seq = scale_seq.unsqueeze(-1)  # (1, S, 1, 1)

        x_scaled = x_rotated * scale_seq

        return x_scaled

def apply_xpos_emb(q, k, xpos_emb):
    """Apply XPos to query and key"""
    q_xpos = xpos_emb(q, seq_dim=1)
    k_xpos = xpos_emb(k, seq_dim=1)
    return q_xpos, k_xpos

class PreNorm(nn.Module):
    """Applies Layer Normalization before the function/module."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """Standard FeedForward block: Linear -> Activation -> Linear."""
    def __init__(self, dim, hidden_dim, dropout=0., activation_name='ReLU'):
        super().__init__()
        # Select activation function based on name
        if activation_name == 'GELU':
            activation = nn.GELU
        elif activation_name == 'SiLU':
            activation = nn.SiLU
        else:  # Default to ReLU
            activation = nn.ReLU
            
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class AttentionWrapper(nn.Module):
    """Wrapper for MultiheadAttention with optional XPos and causal masking."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., xpos_emb=None):
        super().__init__()
        inner_dim = dim_head * heads  # Standard MHA expects embed_dim = dim

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # Check divisibility for PyTorch MHA
        if dim % heads != 0:
            logging.warning(f"AttentionWrapper Warning: dim ({dim}) must be divisible by heads ({heads}). Adjusting heads.")
            # Find largest divisor
            for i in range(heads // 2, 0, -1):
                if dim % i == 0:
                    heads = i
                    break
            if dim % heads != 0: 
                heads = 1  # Default to 1 head
            logging.warning(f"AttentionWrapper: Using heads={heads}")
            self.heads = heads

        self.xpos_emb = xpos_emb  # Store the XPos embedding instance

        # Use batch_first=True for (batch, seq, feature) format
        self.attend = nn.MultiheadAttention(embed_dim=dim, num_heads=self.heads, dropout=dropout, batch_first=True)

    def forward(self, x, attn_mask=None, use_xpos=False):
        # x shape: (batch_size, seq_len, dim)
        q, k, v = x, x, x

        # Apply XPos *before* passing to MultiheadAttention
        if use_xpos and self.xpos_emb is not None:
            q, k = apply_xpos_emb(q, k, self.xpos_emb)
            logging.debug("XPos applied to attention")

        attn_output, _ = self.attend(q, k, v, attn_mask=attn_mask, need_weights=False)
        return attn_output

class TransformerBlock(nn.Module):
    """ Core Transformer block: PreNorm(Attention) + PreNorm(FeedForward). """
    def __init__(self, dim, heads, ff_mult=4, dropout=0., xpos_emb=None, activation_name='ReLU'):
        super().__init__()
        # Pass the XPos embedding instance down to the attention wrapper
        self.attn = PreNorm(dim, AttentionWrapper(dim, heads=heads, dropout=dropout, xpos_emb=xpos_emb))
        self.ff = PreNorm(dim, FeedForward(dim, dim * ff_mult, dropout=dropout, activation_name=activation_name))

    def forward(self, x, attn_mask=None, use_xpos=False):
        # x shape: (batch_size, seq_len, dim)
        x = x + self.attn(x, attn_mask=attn_mask, use_xpos=use_xpos)  # Residual connection for attention
        x = x + self.ff(x)   # Residual connection for feed-forward
        return x

class UnifiedTransformerAE(nn.Module):
    """Unified Transformer Autoencoder for multi-stock prediction with shared weight blocks."""
    
    def __init__(self, config, num_input_features, num_tickers):
        """
        Initialize the model using configuration dictionary.
        
        Args:
            config: Model configuration dictionary
            num_input_features: Number of input features (excluding Ticker_ID)
            num_tickers: Number of unique stock tickers
        """
        super().__init__()
        
        # Store dimensions
        self.num_input_features = num_input_features
        self.num_target_features = len(config['data']['target_features'])
        self.num_tickers = num_tickers
        self.ticker_embedding_dim = config['model']['ticker_embedding_dim']
        self.window_size = config['data']['window_size']
        self.block_dim = config['model']['block_dim']
        self.latent_dim = config['model']['latent_dim']
        self.block_weight_indices = config['model']['block_weight_indices']
        self.num_layers = len(self.block_weight_indices)
        self.bottleneck_layer_index = config['model']['bottleneck_layer_index']
        self.num_heads = config['model']['num_heads']
        self.ff_mult = config['model']['ff_mult']
        self.dropout = config['model']['dropout']
        self.activation_name = config['model']['activation']
        self.sparsity_reg = config['training']['l1_sparsity_reg']
        
        # Positional embedding
        pos_emb_config = config['model']['positional_embedding']
        if pos_emb_config['type'] == 'XPos':
            max_seq_len = self.window_size * pos_emb_config['max_seq_len_multiplier']
            self.xpos_emb = XPosEmbedding(
                dim=self.block_dim,
                max_seq_len=max_seq_len,
                theta=pos_emb_config['rope_theta'],
                gamma=pos_emb_config['xpos_gamma']
            )
            self.xpos_start_layer = pos_emb_config['xpos_start_layer']
            self.xpos_apply_alternating = pos_emb_config.get('xpos_apply_alternating', True)
        else:
            self.xpos_emb = None
            self.xpos_start_layer = float('inf')  # Never apply
            self.xpos_apply_alternating = False
            
        # Validate configurations
        if len(self.block_weight_indices) == 0:
            raise ValueError("block_weight_indices must be non-empty")
            
        # --- Input Embedding ---
        self.ticker_embedding = nn.Embedding(num_tickers, self.ticker_embedding_dim)
        
        # Input projection
        effective_input_dim = num_input_features + self.ticker_embedding_dim
        self.input_proj = nn.Linear(effective_input_dim, self.block_dim)
        self.input_dropout = nn.Dropout(self.dropout)
        
        # --- Transformer Blocks (Weight Shared) ---
        num_unique_blocks = max(self.block_weight_indices) + 1
        logging.info(f"Creating {num_unique_blocks} unique transformer blocks for {self.num_layers} layers with pattern: {self.block_weight_indices}")
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.block_dim, 
                heads=self.num_heads, 
                ff_mult=self.ff_mult,
                dropout=self.dropout, 
                xpos_emb=self.xpos_emb,
                activation_name=self.activation_name
            )
            for _ in range(num_unique_blocks)
        ])
        
        # --- Bottleneck Layer ---
        self.bottleneck_norm = nn.LayerNorm(self.block_dim)
        self.bottleneck_proj_down = nn.Linear(self.block_dim, self.latent_dim)
        
        # Select bottleneck activation
        if self.activation_name == 'GELU':
            self.bottleneck_activation = nn.GELU()
        elif self.activation_name == 'SiLU':
            self.bottleneck_activation = nn.SiLU()
        else:
            self.bottleneck_activation = nn.ReLU()
            
        self.bottleneck_proj_up = nn.Linear(self.latent_dim, self.block_dim)
        
        # --- Output Layer ---
        self.output_norm = nn.LayerNorm(self.block_dim)
        # Output projection maps block_dim back to the number of *input* features
        self.output_proj = nn.Linear(self.block_dim, num_input_features)
        
        logging.info(f"Model initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def _generate_causal_mask(self, sz, device):
        """Generate causal attention mask."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x_features, x_ids):
        # x_features shape: (batch_size, window_size, num_input_features)
        # x_ids shape: (batch_size,) - 1D tensor, one ID per sequence
        batch_size, seq_len, _ = x_features.shape
        device = x_features.device

        # 1. Get Ticker Embedding
        ticker_embs = self.ticker_embedding(x_ids)  # Shape: (batch_size, ticker_embedding_dim)
        
        # Expand embedding to match sequence length for concatenation
        ticker_embs = ticker_embs.unsqueeze(1).expand(-1, seq_len, -1)  # Shape: (batch_size, seq_len, ticker_embedding_dim)

        # 2. Concatenate Features and Ticker Embedding
        x = torch.cat((x_features, ticker_embs), dim=-1)  # Shape: (batch_size, seq_len, num_input_features + ticker_embedding_dim)

        # 3. Input Projection
        x = self.input_proj(x)  # Shape: (batch_size, seq_len, block_dim)
        x = self.input_dropout(x)

        # 4. Generate Causal Mask
        attn_mask = self._generate_causal_mask(seq_len, device)

        latent_representation = None  # To store bottleneck output

        # 5. Transformer Layers
        for i in range(self.num_layers):
            block_index = self.block_weight_indices[i]
            block = self.transformer_blocks[block_index]
            use_xpos = False
            if self.xpos_emb is not None and i >= self.xpos_start_layer:
                if self.xpos_apply_alternating:
                    if (i - self.xpos_start_layer) % 2 == 0: 
                        use_xpos = True
                else:
                    use_xpos = True
            x = block(x, attn_mask=attn_mask, use_xpos=use_xpos)
            
            if i == self.bottleneck_layer_index:
                bottleneck_in = self.bottleneck_norm(x)
                latent_representation = self.bottleneck_proj_down(bottleneck_in)
                latent_representation = self.bottleneck_activation(latent_representation)
                x = self.bottleneck_proj_up(latent_representation)

        # 6. Final Norm and Output Projection
        x = self.output_norm(x)
        predicted_full_features = self.output_proj(x)  # Shape: (batch_size, seq_len, num_input_features)

        # Handle case where bottleneck wasn't applied
        if latent_representation is None and self.bottleneck_layer_index < self.num_layers:
            logging.warning(f"bottleneck_layer_index ({self.bottleneck_layer_index}) >= num_layers ({self.num_layers}) logic seems incorrect. Applying bottleneck after last layer.")
            bottleneck_in = self.output_norm(x)  # Use the last layer's output norm state before projection
            latent_representation = self.bottleneck_proj_down(bottleneck_in)
            latent_representation = self.bottleneck_activation(latent_representation)

        return predicted_full_features, latent_representation  # Return predicted features and latent code

    def l1_activity_loss(self, latent_representation):
        """Calculate L1 regularization on the latent representation"""
        if latent_representation is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)  # Return zero loss if no latent
        return self.sparsity_reg * torch.abs(latent_representation).mean()

def create_model(config, num_input_features, num_tickers):
    """Factory function to create the appropriate model based on config."""
    model_type = config['model']['type']
    
    if model_type == 'UnifiedTransformerAE':
        return UnifiedTransformerAE(config, num_input_features, num_tickers)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 