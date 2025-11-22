"""
Enhanced Transformer ASR model with contextual embeddings and cross-modal attention.

This model extends the base Transformer ASR with:
- Contextual word embeddings
- Cross-modal attention between audio and text
- Multi-task learning (CTC + Word2Vec auxiliary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from models.asr_base import ASREncoder, ASRDecoder


class ContextualEmbedding(nn.Module):
    """
    Contextual word embedding layer.
    
    Uses Transformer encoder to create context-aware word embeddings
    that can be used for semantic similarity and contextual biasing.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, d_model: int,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize contextual embedding layer.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Word embedding dimension
            d_model: Transformer model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding for text sequence
        self.pos_encoding = nn.Parameter(
            torch.randn(1000, embedding_dim) * 0.1
        )
        
        # Transformer encoder for contextualization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Projection to d_model
        self.projection = nn.Linear(embedding_dim, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Create contextual embeddings from token IDs.
        
        Args:
            token_ids: Token IDs (batch, seq_len)
            
        Returns:
            embeddings: Contextual embeddings (batch, seq_len, d_model)
        """
        # Word embeddings
        embeddings = self.word_embedding(token_ids)
        
        # Add positional encoding
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Transformer encoder for contextualization
        contextual_embeddings = self.encoder(embeddings)
        
        # Project to d_model
        contextual_embeddings = self.projection(contextual_embeddings)
        
        return contextual_embeddings


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between audio and text.
    
    Allows the model to attend to relevant parts of the audio based on
    text context, and vice versa. Useful for contextual biasing.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize cross-modal attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor,
                audio_mask: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.
        
        Args:
            audio_features: Audio features (batch, audio_len, d_model)
            text_features: Text features (batch, text_len, d_model)
            audio_mask: Attention mask for audio
            text_mask: Attention mask for text
            
        Returns:
            attended_audio: Audio features with text attention
            attended_text: Text features with audio attention
        """
        batch_size = audio_features.size(0)
        
        # Audio attends to text
        Q_audio = self.W_q(audio_features)
        K_text = self.W_k(text_features)
        V_text = self.W_v(text_features)
        
        # Reshape for multi-head attention
        Q_audio = Q_audio.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K_text = K_text.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V_text = V_text.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q_audio, K_text.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply masks
        if text_mask is not None:
            scores = scores.masked_fill(text_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attended_audio = torch.matmul(attn_weights, V_text)
        attended_audio = attended_audio.transpose(1, 2).contiguous()
        attended_audio = attended_audio.view(batch_size, -1, self.d_model)
        attended_audio = self.W_o(attended_audio)
        
        # Residual connection and layer norm
        attended_audio = self.layer_norm(audio_features + attended_audio)
        
        # Text attends to audio (similar process)
        Q_text = self.W_q(text_features)
        K_audio = self.W_k(audio_features)
        V_audio = self.W_v(audio_features)
        
        Q_text = Q_text.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K_audio = K_audio.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V_audio = V_audio.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q_text, K_audio.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if audio_mask is not None:
            scores = scores.masked_fill(audio_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attended_text = torch.matmul(attn_weights, V_audio)
        attended_text = attended_text.transpose(1, 2).contiguous()
        attended_text = attended_text.view(batch_size, -1, self.d_model)
        attended_text = self.W_o(attended_text)
        
        attended_text = self.layer_norm(text_features + attended_text)
        
        return attended_audio, attended_text


class EnhancedASRModel(nn.Module):
    """
    Enhanced ASR model with contextual embeddings and cross-modal attention.
    
    Architecture:
        Audio → Encoder → Cross-Modal Attention ← Text (Contextual Embeddings)
                         ↓
                      CTC Decoder
                      Word2Vec Auxiliary (optional)
    """
    
    def __init__(self,
                 input_dim: int,
                 vocab_size: int,
                 d_model: int = 256,
                 num_encoder_layers: int = 6,
                 num_heads: int = 4,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 use_contextual_embeddings: bool = True,
                 use_cross_modal_attention: bool = True,
                 use_word2vec_auxiliary: bool = False,
                 word2vec_dim: int = 256):
        """
        Initialize enhanced ASR model.
        
        Args:
            input_dim: Input feature dimension (e.g., 80)
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_encoder_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            use_contextual_embeddings: Use contextual word embeddings
            use_cross_modal_attention: Use cross-modal attention
            use_word2vec_auxiliary: Add Word2Vec auxiliary task
            word2vec_dim: Word2Vec embedding dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_contextual_embeddings = use_contextual_embeddings
        self.use_cross_modal_attention = use_cross_modal_attention
        self.use_word2vec_auxiliary = use_word2vec_auxiliary
        
        # Audio encoder (same as base model)
        self.audio_encoder = ASREncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Contextual text embeddings (optional)
        if use_contextual_embeddings:
            self.contextual_embedding = ContextualEmbedding(
                vocab_size=vocab_size,
                embedding_dim=d_model // 2,
                d_model=d_model,
                num_layers=2,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.contextual_embedding = None
        
        # Cross-modal attention (optional)
        if use_cross_modal_attention:
            self.cross_modal_attention = CrossModalAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.cross_modal_attention = None
        
        # CTC decoder
        self.decoder = ASRDecoder(d_model, vocab_size)
        
        # Word2Vec auxiliary task (optional)
        if use_word2vec_auxiliary:
            self.word2vec_projection = nn.Linear(d_model, word2vec_dim)
        else:
            self.word2vec_projection = None
    
    def forward(self,
                audio_features: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                text_context: Optional[torch.Tensor] = None,
                text_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional contextual inputs.
        
        Args:
            audio_features: Audio features (batch, time, features)
            audio_lengths: Audio sequence lengths
            text_context: Optional text context for biasing (batch, text_len)
            text_lengths: Text sequence lengths
            
        Returns:
            Dictionary containing:
                - logits: CTC logits (batch, time, vocab_size)
                - output_lengths: Updated sequence lengths
                - word2vec_embeddings: Optional Word2Vec embeddings for auxiliary task
        """
        # Encode audio
        encoded_audio, output_lengths = self.audio_encoder(audio_features, audio_lengths)
        
        # Contextual embeddings and cross-modal attention (if enabled)
        if text_context is not None and self.use_contextual_embeddings:
            # Get contextual text embeddings
            text_embeddings = self.contextual_embedding(text_context)
            
            # Apply cross-modal attention if enabled
            if self.use_cross_modal_attention:
                encoded_audio, _ = self.cross_modal_attention(
                    encoded_audio, text_embeddings,
                    audio_mask=None,  # Can add masks if needed
                    text_mask=None
                )
        
        # CTC decoding
        logits = self.decoder(encoded_audio)
        
        result = {
            'logits': logits,
            'output_lengths': output_lengths
        }
        
        # Word2Vec auxiliary task
        if self.use_word2vec_auxiliary:
            word2vec_embeddings = self.word2vec_projection(encoded_audio)
            result['word2vec_embeddings'] = word2vec_embeddings
        
        return result
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test enhanced model
    model = EnhancedASRModel(
        input_dim=80,
        vocab_size=100,
        d_model=256,
        use_contextual_embeddings=True,
        use_cross_modal_attention=True,
        use_word2vec_auxiliary=True
    )
    
    audio = torch.randn(2, 100, 80)
    text_context = torch.randint(0, 100, (2, 20))
    
    output = model(audio, text_context=text_context)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Parameters: {model.get_num_params():,}")

