"""
Transformer Feature Extractor Module
====================================

Advanced Optimization #3: Architecture SOTA pour comprendre les séries temporelles.
Remplace le MLP standard par un Transformer Encoder avec attention mécanisme.

L'attention permet au modèle de dire: "Le pattern d'il y a 15 jours est plus
important que celui d'hier pour ma décision d'aujourd'hui."

Cet extractor est compatible Stable-Baselines3.

Impact: Meilleure compréhension des patterns temporels.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding: Ajoute l'information de position dans la séquence.
    
    Sans PE, le modèle ne sait pas l'ordre des timesteps.
    Avec PE, il sait "ceci est le t-15eme timestep".
    
    Source: "Attention is All You Need" (Vaswani et al., 2017)
    """
    
    def __init__(self, d_model: int, max_len: int = 60):
        """
        Args:
            d_model: Dimension of embeddings (128)
            max_len: Maximum sequence length (lookback period)
        """
        super().__init__()
        
        # Create PE matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Frequency component (2i and 2i+1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Odd indices
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature Extractor based on Transformer Encoder.
    
    Architecture:
    1. Linear projection: Raw features (1293) -> d_model (128)
    2. Positional Encoding: Add sequence position information
    3. Transformer Encoder: Multi-head self-attention (4 heads, 2 layers)
    4. Final projection: d_model * seq_len -> features_dim (512)
    
    Input shape: (batch_size, lookback, 1293)
    Output shape: (batch_size, 512)
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 60,
    ):
        """
        Args:
            observation_space: Gymnasium observation space
            features_dim: Output feature dimension (for policy/value networks)
            d_model: Dimension of transformer embeddings
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward layer dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length (lookback period)
        """
        super().__init__(observation_space, features_dim)
        
        # Assume observation is either:
        # - Shape (batch, lookback, n_features) if already 3D
        # - Shape (batch, n_features) if flat (we reshape in forward)
        
        if len(observation_space.shape) == 1:
            # Flat observation: (n_features,)
            self.n_features = observation_space.shape[0]
            self.is_flat = True
            self.lookback = max_seq_len
        else:
            # 2D observation: (lookback, n_features)
            self.lookback = observation_space.shape[0]
            self.n_features = observation_space.shape[1] if len(observation_space.shape) > 1 else 1
            self.is_flat = False
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        logger.info(
            f"TransformerFeatureExtractor initialized: "
            f"input_features={self.n_features}, "
            f"lookback={self.lookback}, "
            f"d_model={d_model}, "
            f"n_heads={n_heads}, "
            f"output_dim={features_dim}"
        )
        
        # 1. Input projection: features -> d_model
        self.input_proj = nn.Linear(self.n_features, d_model)
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU is better than ReLU for transformers
            batch_first=True,   # Input format: (batch, seq, feature)
            norm_first=False,   # LayerNorm after attention (original)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # 4. Output projection: (d_model * lookback) -> features_dim
        self.final_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * self.lookback, features_dim * 2),
            nn.LayerNorm(features_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim * 2, features_dim),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            observations: Either:
                - Shape (batch, n_features) if flat input
                - Shape (batch, lookback, n_features) if already 3D
        
        Returns:
            Features of shape (batch, features_dim)
        """
        # Reshape if necessary
        if self.is_flat and len(observations.shape) == 2:
            batch_size = observations.shape[0]
            observations = observations.view(batch_size, self.lookback, -1)
        
        # 1. Project to d_model
        x = self.input_proj(observations)  # (batch, lookback, d_model)
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)  # (batch, lookback, d_model)
        
        # 3. Transformer encoder with self-attention
        x = self.transformer_encoder(x)  # (batch, lookback, d_model)
        
        # 4. Final projection and output
        features = self.final_proj(x)  # (batch, features_dim)
        
        return features


class ConvTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Hybrid: CNN + Transformer.
    
    CNN for local pattern extraction, Transformer for temporal dependencies.
    Souvent plus efficace que pure Transformer seul.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 60,
    ):
        """
        Args:
            observation_space: Gymnasium observation space
            features_dim: Output dimension
            d_model: Transformer embedding dimension
            n_heads: Attention heads
            n_layers: Transformer layers
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__(observation_space, features_dim)
        
        if len(observation_space.shape) == 1:
            self.n_features = observation_space.shape[0]
            self.lookback = max_seq_len
        else:
            self.lookback = observation_space.shape[0]
            self.n_features = observation_space.shape[1]
        
        logger.info(
            f"ConvTransformerFeatureExtractor initialized: "
            f"lookback={self.lookback}, n_features={self.n_features}"
        )
        
        # 1. CNN layers for local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # 4. Output projection
        self.final_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * self.lookback, features_dim),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            observations: Shape (batch, lookback, n_features)
        
        Returns:
            Shape (batch, features_dim)
        """
        # Reshape for CNN: (batch, n_features, lookback)
        x = observations.transpose(1, 2)
        
        # CNN
        x = self.cnn(x)  # (batch, d_model, lookback)
        
        # Back to (batch, lookback, d_model) for transformer
        x = x.transpose(1, 2)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Output
        features = self.final_proj(x)
        
        return features


# ============================================================================
# USAGE WITH STABLE-BASELINES3
# ============================================================================

if __name__ == "__main__":
    # Example: Use with PPO
    
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create a dummy environment
    env = gym.make("CartPole-v1")
    
    # Create PPO with TransformerFeatureExtractor
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=512,
            d_model=128,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
        ),
        net_arch=[256, 256],  # Actor-Critic network
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        verbose=1,
    )
    
    # Train
    model.learn(total_timesteps=10_000)
    
    print("✅ Model trained successfully!")
