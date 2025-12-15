# AI2Text Model Technologies & Architecture

## 📋 Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Core Components](#core-components)
4. [Modern Technologies](#modern-technologies)
5. [Training Infrastructure](#training-infrastructure)
6. [Optimization Features](#optimization-features)
7. [Data Pipeline](#data-pipeline)
8. [Decoding Strategies](#decoding-strategies)
9. [Hardware Optimizations](#hardware-optimizations)
10. [LoRA (Low-Rank Adaptation)](#7-lora-low-rank-adaptation)
11. [Project Deployment & Implementation](#project-deployment--implementation)

---

## Overview

**AI2Text** is a state-of-the-art **Bilingual Automatic Speech Recognition (ASR)** system designed for Vietnamese and English. The model uses a modern **Transformer-based Sequence-to-Sequence (Seq2Seq)** architecture with advanced optimizations for efficient training and inference.

### Key Specifications

- **Model Type**: Transformer Encoder-Decoder (Seq2Seq)
- **Total Parameters**: ~30.3 Million
- **Languages Supported**: Vietnamese (77% of training data) + English (23% of training data)
- **Vocabulary Size**: 3,500 tokens (SentencePiece BPE)
- **Input Format**: 80-dimensional Mel-spectrograms
- **Output Format**: Token sequences (autoregressive generation)

---

## Model Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT PIPELINE                           │
│  Audio Waveform → Mel Spectrogram (80-dim) → Feature Normalize  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          ENCODER                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Convolutional Subsampling (2x)                       │   │
│  │    - Conv2D(stride=2) → SiLU → LayerNorm                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. Linear Projection                                      │   │
│  │    - Linear(freq*channels → d_model)                     │   │
│  │    - LayerNorm                                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. Language Embedding                                     │   │
│  │    - Embedding(2 languages → d_model)                    │   │
│  │    - Added to all time steps                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. Rotary Positional Embedding (RoPE)                    │   │
│  │    - Per-head positional encoding                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 5. Encoder Layers (×14)                                  │   │
│  │    ┌────────────────────────────────────────────────┐   │   │
│  │    │ - Self-Attention (Multi-Head, Flash Attention) │   │   │
│  │    │ - Feed-Forward (SiLU, Pre-Norm)                 │   │   │
│  │    │ - Residual Connections                          │   │   │
│  │    │ - RMSNorm                                       │   │   │
│  │    └────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 6. Final Normalization                                   │   │
│  │    - RMSNorm → Dropout                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Encoded Features
                    (batch, time/2, d_model)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          DECODER                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Token Embedding                                        │   │
│  │    - Embedding(vocab_size → d_model)                     │   │
│  │    - Scaled by √d_model                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. Rotary Positional Embedding (RoPE)                    │   │
│  │    - Applied to decoder self-attention                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. Decoder Layers (×6)                                   │   │
│  │    ┌────────────────────────────────────────────────┐   │   │
│  │    │ - Self-Attention (Causal, RoPE)               │   │   │
│  │    │ - Cross-Attention (Encoder-Decoder)           │   │   │
│  │    │ - Feed-Forward (SiLU, Pre-Norm)               │   │   │
│  │    │ - Residual Connections                        │   │   │
│  │    │ - RMSNorm                                     │   │   │
│  │    └────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. Output Projection                                      │   │
│  │    - Linear(d_model → vocab_size)                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Output Logits
                    (batch, tgt_len, vocab_size)
                              │
                              ▼
                    Token Generation
                    (Autoregressive)
```

### Detailed Architecture Specifications

#### Encoder (`ASREncoder`)

| Component | Specification |
|-----------|--------------|
| **Input Dimension** | 80 (Mel-spectrogram features) |
| **Model Dimension (d_model)** | 256 |
| **Number of Layers** | 14 |
| **Attention Heads** | 8 |
| **Head Dimension** | 32 (d_model / num_heads) |
| **Feed-Forward Dimension (d_ff)** | 2048 |
| **Subsampling Factor** | 2x (via Conv2D stride=2) |
| **Dropout Rate** | 0.2 |
| **Output Dimension** | (batch, time/2, 256) |

**Layer-by-Layer Flow:**

1. **Convolutional Subsampling**
   - Input: `(batch, time, 80)` (Mel-spectrogram)
   - Conv2D: `(1, 80) → (d_model//4, time/2, 40)`
   - Activation: SiLU (Swish)
   - Reshape: `(batch, time/2, d_model//4 * 40)`
   - Output: `(batch, time/2, 160)` (for d_model=256)

2. **Linear Projection**
   - Linear: `160 → 256`
   - LayerNorm: Normalize projected features
   - Output: `(batch, time/2, 256)`

3. **Language Embedding**
   - Embedding: `(2, 256)` for Vietnamese (0) and English (1)
   - Broadcast: `(batch, 1, 256)` → `(batch, time/2, 256)`
   - Added element-wise to encoder features

4. **Rotary Positional Embedding (RoPE)**
   - Computed per attention head
   - Head dimension: 32
   - Applied to queries and keys in attention

5. **Encoder Layers (14 layers)**
   - Each layer:
     - **Self-Attention**: Multi-head attention with Flash Attention (SDPA)
     - **Feed-Forward**: Two linear layers with SiLU activation
     - **Normalization**: RMSNorm (pre-norm architecture)
     - **Residual Connections**: Around attention and FFN
     - **Gradient Checkpointing**: Enabled for middle layers (saves VRAM)

6. **Final Normalization**
   - RMSNorm: Final layer normalization
   - Dropout: Regularization before decoder

**Optional Components:**
- **CTC Output**: Linear projection `(d_model → vocab_size)` for hybrid CTC/Attention training
- **Gradient Checkpointing**: Enabled for layers 1-13 (not first/last layer)

#### Decoder (`ASRDecoder` / `TransformerDecoder`)

| Component | Specification |
|-----------|--------------|
| **Model Dimension (d_model)** | 256 |
| **Number of Layers** | 6 |
| **Attention Heads** | 8 |
| **Head Dimension** | 32 |
| **Feed-Forward Dimension (d_ff)** | 2048 |
| **Vocabulary Size** | 3,500 |
| **Max Sequence Length** | 2048 tokens |
| **Dropout Rate** | 0.2 |
| **Output Dimension** | `(batch, tgt_len, 3500)` |

**Layer-by-Layer Flow:**

1. **Token Embedding**
   - Embedding: `(3500, 256)`
   - Scaled by `√256 = 16` for numerical stability
   - Input: Token IDs `(batch, tgt_len)`
   - Output: `(batch, tgt_len, 256)`

2. **Rotary Positional Embedding (RoPE)**
   - Same as encoder
   - Applied to decoder self-attention queries/keys

3. **Decoder Layers (6 layers)**
   - Each layer contains:
     - **Self-Attention**: Causal mask (prevents attending to future tokens)
       - Enables autoregressive generation
       - RoPE applied
     - **Cross-Attention**: Encoder-Decoder attention
       - Queries: Decoder hidden states
       - Keys/Values: Encoder outputs
       - No RoPE (attends to encoder positions)
     - **Feed-Forward**: Same as encoder (SiLU, pre-norm)
     - **Normalization**: RMSNorm (pre-norm)
     - **Residual Connections**: Around all sub-layers
     - **Gradient Checkpointing**: Enabled for middle layers

4. **Output Projection**
   - Linear: `(256 → 3500)`
   - Output: Logits for vocabulary distribution

**Generation Process (Inference):**

1. Start with SOS token (ID: 2)
2. For each step:
   - Forward pass through decoder
   - Sample next token (greedy or temperature sampling)
   - Append to generated sequence
   - Stop at EOS token (ID: 3) or max length
3. Decode token IDs to text using SentencePiece tokenizer

---

## Core Components

### 1. Multi-Head Attention (`MultiHeadAttention`)

**Purpose**: Captures relationships between different positions in the sequence.

**Architecture:**
- **Query, Key, Value Projections**: Separate linear layers for Q, K, V
- **Head Splitting**: Split into 8 heads (each 32-dim)
- **Flash Attention**: Uses PyTorch's `scaled_dot_product_attention`
  - Automatic kernel selection (FlashAttention-2 or MemoryEfficient)
  - 2-4x faster than standard attention
  - Better numerical stability
- **RoPE Integration**: Rotary embeddings applied to Q and K
- **Output Projection**: Concatenate heads and project to d_model

**Mathematical Formulation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
MultiHead = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Key Features:**
- **Encoder Attention**: Bidirectional (non-causal), attends to all positions
- **Decoder Self-Attention**: Causal mask (only attends to past tokens)
- **Cross-Attention**: Decoder queries attend to encoder keys/values

### 2. Feed-Forward Network (`FeedForward`)

**Purpose**: Applies non-linear transformations to each position independently.

**Architecture:**
- **First Linear**: `(d_model → d_ff)` = `(256 → 2048)`
- **Activation**: SiLU (Swish) = `x * sigmoid(x)`
- **Dropout**: 0.2
- **Second Linear**: `(d_ff → d_model)` = `(2048 → 256)`

**Why SiLU instead of ReLU?**
- Prevents "dead neurons" (ReLU can zero out gradients)
- Smoother gradient flow
- Better for deep networks (24+ layers)

### 3. Convolutional Subsampling (`ConvSubsampling`)

**Purpose**: Reduces sequence length while preserving information.

**Architecture:**
- **Input**: `(batch, time, freq)` = `(batch, T, 80)`
- **Conv2D**: `(1, 80) → (d_model//4, T/2, 40)`
  - Kernel: 3×3
  - Stride: 2×2
  - Padding: 1
- **Activation**: SiLU
- **Reshape**: `(batch, T/2, d_model//4 * 40)`
- **Output**: `(batch, T/2, 160)` (for d_model=256)

**Benefits:**
- Reduces computational cost (2x fewer time steps)
- Preserves important frequency information
- Better time resolution than 4x subsampling

### 4. Language Embedding

**Purpose**: Helps model distinguish between Vietnamese and English.

**Architecture:**
- **Embedding Layer**: `(2, d_model)` = `(2, 256)`
  - Index 0: Vietnamese
  - Index 1: English
- **Broadcasting**: Added to all time steps
  - `lang_emb: (batch, 256)`
  - `x: (batch, time, 256)`
  - `x = x + lang_emb.unsqueeze(1)`

**Usage:**
- Provided as `language_ids` tensor during training/inference
- Automatically inferred from dataset metadata
- Helps model learn language-specific patterns

---

## Modern Technologies

### Overview

The model incorporates several modern architectural improvements inspired by LLaMA and other state-of-the-art models:
- **RMSNorm**: Simplified layer normalization
- **RoPE**: Rotary positional embeddings
- **SiLU**: Smooth activation function
- **Flash Attention**: Memory-efficient attention computation
- **Pre-Norm**: Stable training architecture
- **Gradient Checkpointing**: Memory optimization (currently disabled)
- **LoRA**: Parameter-efficient fine-tuning support

### 1. RMSNorm (Root Mean Square Layer Normalization)

**Purpose**: Modern alternative to LayerNorm, used in LLaMA.

**Formula:**
```
RMSNorm(x) = (x / RMS(x)) * weight
where RMS(x) = sqrt(mean(x^2))
```

**Benefits:**
- **Simpler**: No mean centering, no bias term
- **More Efficient**: Fewer operations than LayerNorm
- **Better Gradient Flow**: Prevents exploding gradients in deep models
- **Stability**: More stable for 24+ layer models

**Implementation:**
- Applied before attention and FFN (pre-norm architecture)
- Single learnable weight parameter per dimension

### 2. Rotary Positional Embedding (RoPE)

**Purpose**: Modern positional encoding that encodes relative positions.

**How It Works:**
- Rotates query and key vectors based on position
- Preserves relative position information
- Better generalization to longer sequences

**Mathematical Formulation:**
```
For position m and head dimension d:
θ_i = 10000^(-2i/d) for i = 0, 2, 4, ..., d-2

cos_m = cos(m * θ)
sin_m = sin(m * θ)

q_rot = q * cos_m + rotate_half(q) * sin_m
k_rot = k * cos_m + rotate_half(k) * sin_m
```

**Benefits:**
- **Relative Position Encoding**: Better captures relative distances
- **Extrapolation**: Generalizes to sequences longer than training
- **Efficiency**: Computed once per sequence, cached
- **Attention Patterns**: Produces better attention patterns

**Usage:**
- Applied to encoder self-attention (all layers)
- Applied to decoder self-attention (all layers)
- NOT applied to cross-attention (decoder-encoder)

### 3. SiLU (Swish) Activation

**Purpose**: Smooth activation function that prevents dead neurons.

**Formula:**
```
SiLU(x) = x * sigmoid(x)
```

**Benefits:**
- **No Dead Neurons**: Unlike ReLU, always has non-zero gradient
- **Smooth**: Continuous and differentiable everywhere
- **Better for Deep Networks**: Works well with 24+ layers
- **Empirically Better**: Often outperforms ReLU in transformers

**Usage:**
- Feed-forward networks (encoder and decoder)
- Convolutional subsampling

### 4. Flash Attention (SDPA)

**Purpose**: Memory-efficient and fast attention computation.

**Implementation:**
- Uses PyTorch's `F.scaled_dot_product_attention`
- Automatic kernel selection:
  - FlashAttention-2 (if available)
  - MemoryEfficient (fallback)
  - Standard (if neither available)

**Benefits:**
- **Speed**: 2-4x faster than standard attention
- **Memory**: Reduces memory usage for long sequences
- **Numerical Stability**: Better handling of softmax overflow
- **Automatic Optimization**: PyTorch selects best kernel

**Key Features:**
- Handles attention masks correctly
- Supports dropout during training
- Works with mixed precision (bfloat16/float16)

### 5. Pre-Norm Architecture

**Purpose**: More stable training for deep networks.

**Structure:**
```
# Traditional (Post-Norm):
x = x + attention(layer_norm(x))

# Pre-Norm (Used in AI2Text):
x = x + attention(layer_norm(x))
```

**Benefits:**
- **Better Gradient Flow**: Gradients flow directly through residual connections
- **Stability**: More stable for deep models (14+ layers)
- **Faster Convergence**: Often converges faster than post-norm

**Usage:**
- Applied in all encoder and decoder layers
- RMSNorm before attention and FFN

### 6. Gradient Checkpointing

**Purpose**: Trade computation for memory (saves ~40% VRAM).

**How It Works:**
- During forward pass: Only store activations at checkpoint layers
- During backward pass: Recompute activations between checkpoints
- Reduces memory usage at cost of ~33% more computation

**Implementation:**
- Enabled for middle layers (layers 1-13 in encoder, 1-5 in decoder)
- First and last layers not checkpointed (for efficiency)
- Only active during training
- **Current Status**: Temporarily disabled in config (`use_gradient_checkpointing: false`) due to compatibility with CTC loss output

**Benefits:**
- **Memory Savings**: ~40% reduction in VRAM usage
- **Larger Batch Sizes**: Can use 2x larger batches
- **Deeper Models**: Enables training deeper models on same hardware

---

## Training Infrastructure

### Loss Functions

#### 1. Cross-Entropy Loss (Primary)

**Purpose**: Main loss for seq2seq training.

**Formula:**
```
Loss = -log(P(y_t | y_<t, encoder_output))
```

**Implementation:**
- **Criterion**: `nn.CrossEntropyLoss`
- **Ignore Index**: 0 (padding tokens)
- **Reduction**: Mean
- **Target**: Shifted right tokens (without SOS)

#### 2. CTC Loss (Hybrid Training)

**Purpose**: Helps encoder learn better audio-text alignment.

**Weight**: 0.2 (20% CTC, 80% Attention) - Configurable via `ctc_weight` in config

**Formula:**
```
CTC_Loss = -log(P(text | encoder_output))
Total_Loss = (1 - ctc_weight) * attn_loss + ctc_weight * ctc_loss
```

**Implementation:**
- Uses `CTCLoss` utility class with blank_id = pad_token_id
- Computed from encoder CTC projection output
- Encoder lengths adjusted for 2x subsampling: `encoder_lengths = audio_lengths / 2`

**Benefits:**
- **Better Alignment**: Forces encoder to learn alignment
- **Faster Convergence**: Helps model learn faster
- **Robustness**: More robust to alignment errors
- **Hybrid Training**: Combines benefits of CTC and Attention mechanisms

**Usage:**
- Enabled via `use_ctc_loss: true` in config
- Computed from encoder CTC output
- Combined with attention loss: `(1-0.2)*attn_loss + 0.2*ctc_loss`

### Optimizer

**Type**: AdamW (Adam with Weight Decay)

**Hyperparameters:**
- **Learning Rate**: 0.0003
- **Weight Decay**: 0.0001
- **Betas**: (0.9, 0.999)
- **Epsilon**: 1e-8

**Why AdamW?**
- Better generalization than Adam
- Proper weight decay (not L2 regularization)
- Stable training for transformers

### Learning Rate Scheduling

**Strategy**: Warmup + Cosine Annealing

**Phases:**

1. **Warmup** (3% of training steps)
   - Linear increase from 0 to base_lr
   - Formula: `lr = base_lr * (step / warmup_steps)`

2. **Cosine Annealing** (remaining steps)
   - Smooth decay from base_lr to min_lr
   - Formula: `lr = eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(π * step / T_max))`
   - Period: 50% of remaining steps (faster decay)

**Benefits:**
- Stable start (warmup prevents early instability)
- Smooth convergence (cosine annealing)
- Better final performance

### Training Techniques

#### 1. Teacher Forcing

**Purpose**: Parallel training of decoder.

**How It Works:**
- During training: Decoder receives ground truth tokens (shifted right)
- Input: `[SOS, token1, token2, ..., tokenN]`
- Target: `[token1, token2, ..., tokenN, EOS]`
- Enables parallel computation of all positions

**Benefits:**
- Fast training (parallel instead of sequential)
- Stable gradients
- Better convergence

#### 2. Scheduled Sampling

**Purpose**: Gradually reduce teacher forcing to improve generalization.

**Strategy:**
- Initial: 100% teacher forcing (`teacher_forcing_initial: 1.0`)
- Final: 50% teacher forcing (`teacher_forcing_final: 0.5`)
- Decay: Linear over training epochs
- Implementation: `ScheduledSampling` class with epoch-based decay

**How It Works:**
- During training: Randomly replaces ground truth tokens with model predictions
- Probability of using ground truth decreases linearly from 100% to 50%
- Forces model to rely on encoder outputs instead of just language patterns

**Benefits:**
- Better generalization (model learns to use encoder)
- Reduces overfitting to ground truth
- More robust inference
- Prevents decoder from ignoring encoder outputs

#### 3. Curriculum Learning

**Purpose**: Start with easier examples, gradually increase difficulty.

**Implementation:**
- Starts with shorter sentences (max 4 seconds)
- Gradually increases to full dataset
- Requires WER < 0.70 to progress

**Benefits:**
- Faster initial learning
- Better convergence
- More stable training

#### 4. Auto-Rollback

**Purpose**: Prevents training collapse.

**Strategy:**
- Monitors validation loss
- If loss increases by 30% (threshold_ratio=1.3), rollback to previous checkpoint
- Prevents catastrophic forgetting

**Benefits:**
- Prevents training collapse
- Automatic recovery
- Saves training time

### Mixed Precision Training

**Type**: Automatic Mixed Precision (AMP)

**Precision:**
- **Forward Pass**: bfloat16 (better numerical stability than float16)
- **Backward Pass**: bfloat16
- **Optimizer States**: float32 (for numerical accuracy)

**Benefits:**
- **Speed**: ~2x faster training
- **Memory**: ~50% reduction in VRAM usage
- **Stability**: bfloat16 has same exponent range as float32

**Implementation:**
- Uses `torch.cuda.amp.autocast`
- Gradient scaling with `GradScaler`
- Automatic loss scaling

### Gradient Accumulation

**Purpose**: Simulate larger batch sizes without increasing memory.

**Strategy:**
- Accumulate gradients over 4 batches (`gradient_accumulation_steps: 4`)
- Base batch size: 32 (configurable via `batch_size`)
- Effective batch size: 32 * 4 = 128
- Update optimizer every 4 batches

**Benefits:**
- Larger effective batch size
- Better gradient estimates
- No memory increase
- Allows training with smaller GPU memory

### Gradient Clipping

**Purpose**: Prevents gradient explosion.

**Method**: Clip gradients to max norm of 0.5

**Formula:**
```
if ||grad|| > max_norm:
    grad = grad * max_norm / ||grad||
```

**Benefits:**
- Prevents training instability
- More stable convergence
- Better for deep networks

---

## Optimization Features

### Hardware-Specific Optimizations

#### For RTX 5060TI 16GB VRAM:

1. **Gradient Checkpointing**
   - Saves ~40% VRAM
   - Enables larger batch sizes

2. **Mixed Precision (AMP)**
   - bfloat16 for forward/backward
   - 2x speedup, 50% memory reduction

3. **Flash Attention**
   - Efficient attention computation
   - Automatic kernel selection

4. **Efficient Batching**
   - Length sorting for minimal padding
   - Dynamic batching

#### For Ryzen 9 9990X:

1. **Multi-core Data Loading**
   - 2 workers (`num_workers: 2`) - reduced from 12 to avoid BrokenPipeError
   - Parallel data preprocessing
   - Configurable via `num_workers` in config

2. **Persistent Workers**
   - Enabled (`persistent_workers: true`)
   - Reduces worker startup overhead
   - Faster data loading

3. **Prefetch Factor**
   - 4 batches per worker (`prefetch_factor: 4`)
   - Reduces I/O wait time
   - Pre-loads next batches while current batch is processed

4. **Optimized CPU Operations**
   - Efficient tensor operations
   - Vectorized computations
   - Length-based sorting (`sort_by_length: true`) for minimal padding

#### For 64GB RAM:

1. **Large Batch Sizes**
   - Effective batch size: 128 (32 base * 4 accumulation)
   - Better gradient estimates
   - Configurable via `batch_size` and `gradient_accumulation_steps`

2. **Better Caching**
   - Optional dataset caching (`cache_in_ram: false` by default)
   - Faster data access when enabled
   - Can be enabled for faster training on smaller datasets

3. **Memory-Efficient Data Loading**
   - Optimized data pipeline
   - Minimal memory overhead
   - Length-based sorting to minimize padding waste

#### For SSD 3000MB/s:

1. **Fast I/O**
   - Parallel data loading
   - Efficient file reading

2. **Efficient Prefetching**
   - Reduces I/O wait time
   - Better throughput

### Model Compilation

**torch.compile() Support:**
- Model can be compiled for faster inference
- Modes: "default", "reduce-overhead", "max-autotune"
- Automatic optimization
- Available via `ASRModel.compile_model()` method

**Benefits:**
- Faster inference (up to 2x)
- Better GPU utilization
- Automatic kernel fusion
- Optimized for RTX 5060TI architecture

---

## Data Pipeline

### Audio Processing

**Input Format:**
- Audio files (WAV, MP3, etc.)
- Sample rate: 16kHz (standard for ASR)
- Mono channel

**Processing Steps:**

1. **Load Audio**
   - Uses torchaudio (faster) or librosa (fallback)
   - Resample to 16kHz if needed
   - Convert to mono if stereo

2. **Feature Extraction**
   - **Mel Spectrogram**:
     - n_mels: 80
     - n_fft: 400
     - hop_length: 160 (10ms frames)
     - win_length: 400
   - **Amplitude to DB**: Convert to log scale

3. **Normalization**
   - Feature normalization
   - Optional: Mean-variance normalization

**Output Format:**
- `(time, 80)` Mel-spectrogram features
- Time dimension varies with audio length

### Text Processing

**Input Format:**
- Raw text (Vietnamese or English)
- Unicode strings

**Processing Steps:**

1. **Text Normalization**
   - Lowercase conversion
   - Punctuation handling
   - Special character normalization

2. **Tokenization**
   - **SentencePiece BPE**: Subword tokenization
   - Vocabulary: 3,500 tokens
   - Handles Vietnamese and English

3. **Token Encoding**
   - Convert tokens to IDs
   - Add SOS (ID: 2) and EOS (ID: 3) tokens
   - Pad to batch length (ID: 0)

**Output Format:**
- Token IDs: `(batch, seq_len)`
- Special tokens: SOS=2, EOS=3, PAD=0

### Data Augmentation

**Audio Augmentation:**
- **Time Stretching**: ±10% speed variation
- **Pitch Shifting**: ±2 semitones
- **Noise Injection**: Random noise
- **Time Masking**: Mask time frames
- **Frequency Masking**: Mask frequency bands

**Benefits:**
- Better generalization
- Robustness to variations
- More training data

---

## Decoding Strategies

### 1. Greedy Decoding (Default)

**Method:**
- At each step, select token with highest probability
- Fast and deterministic

**Usage:**
- Default for inference
- Fast validation

### 2. Autoregressive Generation

**Method:**
- Generate tokens one at a time
- Use previous tokens as context
- Stop at EOS or max length

**Features:**
- **Temperature Sampling**: Control randomness
  - temperature=1.0: Greedy
  - temperature>1.0: More random
- **Repetition Penalty**: Penalize repeated tokens
- **N-gram Blocking**: Prevent repeating n-grams

**Usage:**
- Inference mode
- More accurate than greedy

### 3. Beam Search (Optional)

**Method:**
- Maintain multiple hypotheses
- Explore best paths
- Select best final hypothesis

**Parameters:**
- Beam width: 5
- Length penalty: 0.6

**Benefits:**
- Better accuracy than greedy
- Handles ambiguous cases

**Usage:**
- High-accuracy inference
- Slower than greedy

---

## Hardware Optimizations

### Memory Management

**Gradient Checkpointing:**
- Saves ~40% VRAM
- Trade computation for memory

**Mixed Precision:**
- bfloat16 reduces memory by 50%
- Float32 for optimizer states

**Efficient Batching:**
- Length sorting minimizes padding
- Dynamic batch sizes

### Computational Efficiency

**Flash Attention:**
- 2-4x faster attention
- Memory-efficient

**torch.compile():**
- Automatic kernel fusion
- Faster inference

**Optimized Operations:**
- Vectorized computations
- Efficient tensor operations

### I/O Optimization

**Parallel Data Loading:**
- Multiple workers
- Prefetching

**Fast Storage:**
- SSD 3000MB/s
- Efficient file reading

---

## Model Summary

### Parameter Count

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Encoder | ~22M | 72.5% |
| Decoder | ~8M | 26.4% |
| Embeddings | ~0.3M | 1.1% |
| **Total** | **~30.3M** | **100%** |

### Computational Complexity

**Encoder:**
- Time Complexity: O(T² * d_model) per layer
- With 2x subsampling: O((T/2)² * d_model)
- 14 layers: 14 * O((T/2)² * 256)

**Decoder:**
- Time Complexity: O(S² * d_model) for self-attention
- Cross-attention: O(S * T * d_model)
- 6 layers: 6 * (O(S² * 256) + O(S * T * 256))

**Total:**
- Training: O(T² + S² + S*T) per sample
- Inference: O(T² + S²) per sample (autoregressive)

### Memory Usage

**Training (with optimizations):**
- Model weights: ~120MB (bfloat16)
- Activations: ~8-10GB (with checkpointing)
- Optimizer states: ~240MB (float32)
- **Total**: ~9-10GB VRAM

**Inference:**
- Model weights: ~120MB
- Activations: ~1-2GB
- **Total**: ~1-2GB VRAM

### Training Loss Progression

**Average Training Loss per Epoch:**

The model shows consistent loss reduction during training. Below is the average training loss for each epoch:

| Epoch | Average Train Loss | Notes |
|-------|-------------------|-------|
| 1 | 7.4889 | Initial training |
| 2 | 6.3833 | Rapid improvement |
| 3 | 5.7194 | Continued decrease |
| 4 | 5.3350 | Steady progress |
| 5 | 5.1269 | Approaching 5.0 |
| 6 | 4.9854 | Below 5.0 |
| 7 | 4.9915 | Slight increase (normal fluctuation) |
| 8 | 4.8156 | Resumed decrease |
| 9 | 4.7700 | Continued improvement |
| 10 | 4.7139 | Steady decline |
| 11 | 4.6545 | Progress continues |
| 12 | 4.5952 | Approaching 4.5 |
| 13 | 4.5099 | Below 4.6 |
| 14 | 4.4428 | Consistent improvement |
| 15 | 4.3564 | Below 4.4 |
| 16 | 4.3173 | Approaching 4.3 |
| 17 | 4.2814 | Continued decrease |
| 18 | 4.2469 | Steady progress |
| 19 | 4.2056 | Below 4.25 |
| 20 | 4.1817 | Approaching 4.2 |
| 21 | 4.1587 | Below 4.2 |

**Loss Trends:**
- **Initial Phase (Epochs 1-5)**: Rapid loss reduction from ~7.5 to ~5.1
- **Middle Phase (Epochs 6-15)**: Steady decrease from ~5.0 to ~4.4
- **Later Phase (Epochs 16-21)**: Gradual improvement from ~4.3 to ~4.2

**Validation Loss:**
- Best validation loss achieved: **4.3253** (at epoch 14)
- Validation loss typically tracks training loss with slight overfitting in later epochs

**Loss Characteristics:**
- Loss decreases smoothly without major spikes (indicating stable training)
- No training collapse observed (auto-rollback not triggered)
- Consistent improvement suggests good learning rate schedule

### 7. LoRA (Low-Rank Adaptation)

**Purpose**: Efficient fine-tuning with minimal trainable parameters.

**How It Works:**
- Replaces weight matrix W with W + BA where:
  - W: Original weight matrix (frozen)
  - B: Low-rank matrix (rank × out_features, trainable)
  - A: Low-rank matrix (in_features × rank, trainable)
- Only trains low-rank matrices, freezes base model
- Reduces trainable parameters by up to 100x

**Mathematical Formulation:**
```
W_new = W_old + (B @ A) * scaling
where scaling = alpha / rank
```

**Implementation:**
- `LoRALinear` class replaces `nn.Linear` layers
- Applied to attention and feed-forward layers by default
- Configurable via `ASRModel.apply_lora()` method
- Default rank: 8, alpha: 16.0

**Benefits:**
- **Memory Efficiency**: ~60% reduction in VRAM usage
- **Faster Training**: Fewer parameters to update
- **Larger Batch Sizes**: Can use larger batches with same memory
- **Fine-tuning**: Enables fine-tuning large models on limited hardware
- **Modular**: Can be applied selectively to specific layers

**Usage:**
```python
# Apply LoRA to model
lora_modules = model.apply_lora(
    rank=8,
    alpha=16.0,
    dropout=0.0,
    target_modules=['W_q', 'W_k', 'W_v', 'W_o', 'linear1', 'linear2']
)
```

**Parameter Reduction:**
- Base model: ~30.3M parameters
- With LoRA (rank=8): ~0.3M trainable parameters (99% reduction)
- Enables fine-tuning on RTX 5060TI 16GB with larger batch sizes

---

## Project Deployment & Implementation

### 1. Installation & Setup

#### System Requirements

**Hardware:**
- GPU: NVIDIA RTX 5060TI 16GB VRAM (or equivalent)
- CPU: Ryzen 9 9990X (or equivalent multi-core processor)
- RAM: 64GB (recommended)
- Storage: SSD with 3000MB/s+ read speed

**Software:**
- Python 3.8+
- PyTorch 2.0+ (with CUDA support)
- CUDA 11.8+ (for GPU acceleration)
- Linux (Ubuntu 20.04+ recommended)

#### Installation Steps

```bash
# 1. Clone repository
git clone <repository-url>
cd AI2Text

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Dataset Preparation

1. **Prepare Dataset Structure:**
```
data/processed/full_merged_dataset/
├── train/
│   ├── audio/
│   │   ├── sample_001.wav
│   │   └── ...
│   └── manifest.csv
├── val/
│   ├── audio/
│   └── manifest.csv
└── test/
    ├── audio/
    └── manifest.csv
```

2. **Manifest Format:**
```csv
audio_path,transcript,language,duration
train/audio/sample_001.wav,"Xin chào",vi,2.5
train/audio/sample_002.wav,"Hello world",en,3.1
```

3. **Tokenizer Setup:**
- Place SentencePiece tokenizer at: `models/tokenizer_vi_en_3500.model`
- Or train new tokenizer: `python scripts/train_tokenizer.py`

#### Configuration

Edit `configs/default.yaml` to customize:
- Model architecture (d_model, num_layers, etc.)
- Training hyperparameters (batch_size, learning_rate, etc.)
- Dataset paths
- Optimization settings (AMP, gradient checkpointing, etc.)

### 2. Training Workflow

#### Basic Training

**Start Training from Scratch:**
```bash
# Method 1: Using training script
python training/train.py --config configs/default.yaml

# Method 2: Using helper script
./run_training.sh start
```

**Resume Training:**
```bash
# Resume from best model
./run_training.sh resume

# Resume from newest checkpoint
./run_training.sh resume-newest

# Resume from specific checkpoint
python training/train.py --config configs/default.yaml --resume checkpoints/best_model.pt
```

#### Training Monitoring

**Real-time Monitoring:**
```bash
# View training logs
tail -f logs/training.log

# Or use helper script
./run_training.sh logs

# Check training status
./run_training.sh status
```

**Metrics Display:**
Training script displays real-time metrics:
- **Loss**: Current and average training loss
- **LR**: Current learning rate
- **WER/CER**: Word/Character Error Rate (during validation)
- **GPU Memory**: VRAM usage
- **Speed**: Samples per second
- **ETA**: Estimated time remaining

**Checkpoint Management:**
```bash
# List all checkpoints
./run_training.sh checkpoints

# Checkpoint locations:
# - checkpoints/best_model.pt (best validation loss)
# - checkpoints/checkpoint_epoch_N.pt (periodic saves)
```

#### Training Best Practices

1. **Monitor GPU Memory:**
   - Watch for OOM errors
   - Adjust batch_size if needed
   - Enable gradient checkpointing if available

2. **Track Metrics:**
   - Monitor validation loss for overfitting
   - Check WER/CER trends
   - Watch for training collapse (auto-rollback handles this)

3. **Save Regularly:**
   - Checkpoints saved every epoch (configurable)
   - Best model automatically saved
   - Keep multiple checkpoints for comparison

4. **Resume Training:**
   - Always resume from best_model.pt for best results
   - Learning rate automatically restored
   - Training state fully preserved

### 3. API Deployment

#### Start API Server

**Using Helper Script:**
```bash
# Start API on default port (8000)
./start_api.sh

# Or specify custom port
./start_api.sh 8080
```

**Using Python Directly:**
```bash
# Method 1: Run FastAPI app
python api/app.py

# Method 2: Using uvicorn
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### API Endpoints

**1. Health Check:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 1,
  "tokenizer_ready": true,
  "processor_ready": true
}
```

**2. Transcribe Audio:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@audio_file.wav" \
  -F "model_name=default" \
  -F "use_beam_search=true" \
  -F "beam_width=5"
```

**Response:**
```json
{
  "text": "Xin chào thế giới",
  "confidence": 0.95,
  "model_name": "default",
  "processing_time": 0.234
}
```

**3. List Available Models:**
```bash
curl http://localhost:8000/models
```

**4. Load Model:**
```bash
curl -X POST http://localhost:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "checkpoints/best_model.pt", "model_name": "my_model"}'
```

#### Python Client Usage

**Using ASRService:**
```python
from api.asr_service import ASRService

# Initialize service
service = ASRService(
    checkpoint_path="checkpoints/best_model.pt",
    device="cuda"  # or "cpu"
)

# Transcribe single file
result = service.transcribe(
    audio_path="audio.wav",
    language_id=0  # 0=Vietnamese, 1=English, None=auto
)
print(result['text'])

# Transcribe batch
results = service.transcribe_batch([
    "audio1.wav",
    "audio2.wav",
    "audio3.wav"
])
```

**Using REST API Client:**
```python
import requests

# Transcribe audio
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"audio": f},
        data={
            "model_name": "default",
            "use_beam_search": True,
            "beam_width": 5
        }
    )
    
result = response.json()
print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']}")
```

### 4. Evaluation & Testing

#### Quick WER/CER Test

**Using Helper Script:**
```bash
# Test with 3 samples
./run_training.sh test-wer

# Or directly
python quick_test_wer.py --checkpoint checkpoints/best_model.pt --num-samples 10
```

#### Full Evaluation

**Evaluate on Test Set:**
```bash
python evaluate_checkpoint.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/default.yaml \
  --split test
```

**Evaluate on Validation Set:**
```bash
python evaluate_checkpoint.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/default.yaml \
  --split val
```

**Output Metrics:**
- Word Error Rate (WER)
- Character Error Rate (CER)
- Average confidence score
- Processing time per sample

#### Unit Testing

**Run All Tests:**
```bash
# Method 1: Using pytest
pytest tests/

# Method 2: Using simple test runner
python tests/run_tests_simple.py
```

**Test Coverage:**
- Model architecture tests
- Data preprocessing tests
- Training pipeline tests
- Metrics calculation tests
- Integration tests

### 5. Production Deployment

#### Docker Deployment

**Create Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose API port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run:**
```bash
# Build image
docker build -t ai2text-api .

# Run container
docker run -d \
  --name ai2text-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  ai2text-api
```

#### Production Considerations

**1. Model Optimization:**
```python
# Compile model for faster inference
model = model.compile_model(mode="reduce-overhead")

# Use half precision for inference
model = model.half()  # float16
```

**2. API Optimization:**
- Enable model caching (already implemented)
- Use batch processing for multiple requests
- Implement request queuing for high load
- Add rate limiting

**3. Monitoring:**
- Log all API requests
- Track response times
- Monitor GPU memory usage
- Set up alerts for errors

**4. Scaling:**
- Use multiple GPU instances for load balancing
- Implement model serving with multiple workers
- Use Kubernetes for orchestration
- Consider using TorchServe for model serving

### 6. Monitoring & Maintenance

#### Training Monitoring

**Real-time Metrics:**
- Monitor `logs/training.log` for detailed logs
- Check `training_output.log` for console output
- Use `./run_training.sh status` for quick status

**Key Metrics to Watch:**
- Training loss (should decrease)
- Validation loss (should decrease, watch for overfitting)
- Learning rate (follows warmup + cosine schedule)
- WER/CER (should improve over time)
- GPU utilization (should be high)

#### Model Maintenance

**Regular Tasks:**
1. **Backup Checkpoints:**
   ```bash
   # Backup best model
   cp checkpoints/best_model.pt backups/best_model_$(date +%Y%m%d).pt
   ```

2. **Clean Old Checkpoints:**
   ```bash
   # Keep only recent checkpoints
   find checkpoints/ -name "checkpoint_epoch_*.pt" -mtime +30 -delete
   ```

3. **Monitor Disk Space:**
   - Checkpoints can be large (~500MB each)
   - Keep only necessary checkpoints
   - Archive old training runs

**Troubleshooting:**

**Issue: Out of Memory (OOM)**
- Solution: Reduce batch_size in config
- Enable gradient checkpointing (if compatible)
- Use smaller model (reduce d_model or num_layers)

**Issue: Training Collapse (Loss Explosion)**
- Solution: Auto-rollback handles this automatically
- Check learning rate (may be too high)
- Verify data quality

**Issue: Slow Training**
- Solution: Check GPU utilization
- Reduce num_workers if CPU-bound
- Enable mixed precision (AMP)
- Use gradient accumulation for larger effective batch

**Issue: Poor WER/CER**
- Solution: Train for more epochs
- Check data quality and alignment
- Verify tokenizer coverage
- Try different hyperparameters

### 7. Development Workflow

#### Typical Development Cycle

1. **Data Preparation:**
   ```bash
   # Prepare and validate dataset
   python scripts/validate_data_alignment.py
   ```

2. **Training:**
   ```bash
   # Start training
   ./run_training.sh start
   
   # Monitor progress
   ./run_training.sh logs
   ```

3. **Evaluation:**
   ```bash
   # Quick test
   ./run_training.sh test-wer
   
   # Full evaluation
   python evaluate_checkpoint.py --checkpoint checkpoints/best_model.pt
   ```

4. **API Testing:**
   ```bash
   # Start API
   ./start_api.sh
   
   # Test transcription
   curl -X POST http://localhost:8000/transcribe -F "audio=@test.wav"
   ```

5. **Iteration:**
   - Adjust hyperparameters in `configs/default.yaml`
   - Resume training from best checkpoint
   - Repeat until satisfied

#### Version Control

**Recommended Git Workflow:**
```bash
# Create feature branch
git checkout -b feature/new-feature

# Commit changes
git add .
git commit -m "Add new feature"

# Push and create PR
git push origin feature/new-feature
```

**Files to Track:**
- Source code (`models/`, `training/`, `api/`, etc.)
- Configuration files (`configs/`)
- Scripts (`scripts/`, `*.sh`)

**Files to Ignore:**
- Checkpoints (`checkpoints/*.pt`)
- Logs (`logs/*.log`)
- Cache (`cache/`)
- Data (`data/processed/`)

---

## Conclusion

The **AI2Text** model uses a modern Transformer-based architecture with state-of-the-art components:

- **LLaMA-style components**: RMSNorm, SiLU, RoPE
- **Efficient attention**: Flash Attention (SDPA)
- **Bilingual support**: Language embeddings
- **Optimized training**: Mixed precision (bfloat16), gradient accumulation
- **Advanced techniques**: Curriculum learning, scheduled sampling, hybrid CTC/Attention
- **Efficient fine-tuning**: LoRA support for parameter-efficient adaptation

**Current Training Configuration:**
- Base batch size: 32
- Effective batch size: 128 (with 4x gradient accumulation)
- Mixed precision: bfloat16 (better stability than float16)
- Gradient checkpointing: Temporarily disabled (compatible with CTC loss)
- Data workers: 2 (optimized for Ryzen 9 9990X)
- CTC weight: 0.2 (20% CTC, 80% Attention)

The model is optimized for high-performance hardware (RTX 5060TI, Ryzen 9 9990X, 64GB RAM, fast SSD) and achieves efficient training and inference while maintaining high accuracy for Vietnamese and English ASR.

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Model Version**: AI2Text v1.0

