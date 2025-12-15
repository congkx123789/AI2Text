# Công Nghệ và Triển Khai - AI-LLM System

## 📋 Tổng Quan

Tài liệu này mô tả chi tiết các công nghệ được sử dụng và cách triển khai hệ thống AI-LLM - một pipeline tích hợp Speech-to-Text (ASR) và Large Language Model (LLM) với khả năng RAG (Retrieval Augmented Generation).

---

## 🏗️ Kiến Trúc Tổng Thể

Hệ thống được xây dựng theo kiến trúc modular với các thành phần chính:

1. **API Layer** - FastAPI REST API
2. **Model Management Layer** - Unified Model Manager
3. **ASR Pipeline** - Whisper Speech-to-Text
4. **LLM Pipeline** - Qwen/Gemini Text Generation
5. **RAG Pipeline** - Hybrid Retrieval + Reranking

---

## 🔧 Công Nghệ Core

### 1. Speech-to-Text (ASR) - Whisper

#### Công Nghệ Sử Dụng

**Base Model:**
- **Model**: OpenAI Whisper Small (`openai/whisper-small`)
- **Format**: CTranslate2 (tối ưu hóa cho inference nhanh)
- **Framework**: `faster-whisper` (Python wrapper cho CTranslate2)
- **Fine-tuning**: HuggingFace Transformers

**Tính Năng:**
- Hỗ trợ đa ngôn ngữ (tiếng Anh + tiếng Việt)
- Auto-detect language
- VAD (Voice Activity Detection) filtering
- Segment timestamps
- Beam search decoding

#### Triển Khai Chi Tiết

**1. Model Loading (`src/tools/ai2text_bridge.py`)**

```python
# Hệ thống tự động detect format model:
# - CTranslate2 format: có file model.bin
# - HuggingFace format: có config.json + model files

def _get_model(size, device, compute):
    # Auto-detect device (CUDA/CPU)
    # Auto-select compute type (float16/int8)
    # Cache models để tránh load lại
```

**Các tính năng:**
- **Lazy Loading**: Model chỉ load khi cần
- **Caching**: Cache models theo key (size:device:compute)
- **Auto-detection**: Tự động detect CUDA/CPU và chọn compute type phù hợp
- **Format Support**: Hỗ trợ cả CTranslate2 và HuggingFace format

**2. Transcription Pipeline**

```python
def transcribe(path, size, lang, device, compute):
    # 1. Load model (cached)
    # 2. Process audio với faster-whisper
    # 3. Return text + segments + language
```

**Workflow:**
1. Load audio file (tự động resample về 16kHz nếu cần)
2. Extract log-Mel spectrogram features
3. Encode với Whisper encoder
4. Decode với beam search
5. Post-process và return kết quả

**3. Fine-tuning Implementation (`scripts/train_whisper.py`)**

**Công nghệ training:**
- **Framework**: HuggingFace Transformers
- **Trainer**: `Seq2SeqTrainer` (chuyên cho sequence-to-sequence)
- **Data Collator**: Custom `DataCollatorSpeechSeq2SeqWithPadding`
- **Metrics**: WER (Word Error Rate) evaluation
- **Optimization**: 
  - BF16 training (cho GPU Ampere+)
  - Gradient checkpointing
  - Mixed precision training

**Training Process:**
```python
# 1. Load dataset (JSONL format)
# 2. Prepare audio features (log-Mel spectrograms)
# 3. Tokenize transcripts
# 4. Train với Seq2SeqTrainer
# 5. Evaluate với WER metric
# 6. Save checkpoints
```

**Hyperparameters:**
- Batch size: 16-32 (tùy GPU)
- Learning rate: 1e-5
- Epochs: 3
- Warmup steps: 500
- Gradient accumulation: 1

**4. Model Conversion (`scripts/convert_whisper_to_ct2.py`)**

**Công nghệ:**
- **Tool**: `ct2-transformers-converter` (CTranslate2)
- **Format**: HuggingFace → CTranslate2
- **Optimization**: 
  - Quantization (int8, float16)
  - Model optimization cho inference

**Lợi ích:**
- **Tốc độ**: ~5x nhanh hơn HuggingFace
- **Memory**: Tiết kiệm VRAM
- **Latency**: Giảm độ trễ inference

---

### 2. Large Language Model (LLM) - Qwen

#### Công Nghệ Sử Dụng

**Base Model:**
- **Model**: Qwen2.5-0.5B-Instruct
- **Architecture**: Transformer-based decoder-only
- **Size**: 513MB (merged model)
- **Format**: Merged model (LoRA adapter đã merge vào base)

**Fine-tuning:**
- **Method**: QLoRA (Quantized LoRA)
- **Framework**: Unsloth (optimized training)
- **Adapter**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (GPTQ)

#### Triển Khai Chi Tiết

**1. Model Loading (`src/llm/load.py`)**

```python
def load_llm(name):
    # Auto-detect LoRA adapter
    # Load base model + adapter
    # Merge adapter vào base (cho inference nhanh)
    # Return tokenizer + model
```

**Tính năng:**
- **Auto-detection**: Tự động detect LoRA adapter
- **Merging**: Merge adapter vào base model (không cần PEFT runtime)
- **Device mapping**: Auto device placement (CUDA/CPU)
- **Quantization**: Hỗ trợ 4-bit/8-bit models

**2. Text Generation (`src/llm/infer.py`)**

**Multi-task Support:**
- `summarize`: Tóm tắt văn bản
- `answer`: Trả lời câu hỏi
- `translate`: Dịch sang tiếng Anh
- `analyze`: Phân tích nội dung
- `extract`: Trích xuất thông tin

**Generation Process:**
```python
def generate_text(text, task, question, max_tokens):
    # 1. Create prompt theo task
    # 2. Tokenize input
    # 3. Generate với model
    # 4. Decode và extract response
```

**3. Fine-tuning Implementation (`scripts/train_llm.py`)**

**Công nghệ training:**
- **Framework**: Unsloth (optimized) + HuggingFace Transformers
- **Trainer**: `SFTTrainer` (Supervised Fine-Tuning)
- **Method**: QLoRA (4-bit quantization + LoRA)
- **Optimizer**: AdamW 8-bit (tiết kiệm 75% VRAM)

**Training Process:**
```python
# 1. Load model với 4-bit quantization
# 2. Add LoRA adapters
# 3. Format dataset với Qwen template
# 4. Train với SFTTrainer
# 5. Save adapter
```

**LoRA Configuration:**
- **Rank (r)**: 16
- **Alpha**: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Dropout**: 0.05

**Hyperparameters:**
- Batch size: 2-4 (effective batch = batch_size * gradient_accumulation)
- Gradient accumulation: 16
- Learning rate: 2e-4
- Epochs: 1-3
- Max sequence length: 2048

**4. Model Merging (`scripts/merge_lora_adapter.py`)**

**Công nghệ:**
- **Tool**: PEFT `merge_and_unload()`
- **Process**: Merge LoRA weights vào base model
- **Output**: Standalone model (không cần adapter)

**Lợi ích:**
- **Inference nhanh hơn**: Không cần load adapter riêng
- **Đơn giản hóa**: Model độc lập, dễ deploy
- **Tương thích**: Hoạt động với mọi framework

---

### 3. Google Gemini API Integration

#### Công Nghệ Sử Dụng

**API:**
- **SDK**: `google-generativeai` (Python)
- **Models**: 
  - `gemini-2.5-flash` (nhanh, phù hợp real-time)
  - `gemini-2.5-pro` (chất lượng cao, phù hợp complex tasks)

**Tính Năng:**
- Text generation với multi-task support
- RAG với citations
- Streaming support (có thể mở rộng)

#### Triển Khai (`src/llm/gemini.py`)

**Initialization:**
```python
def _init_gemini():
    # Configure API key
    # Create GenerativeModel instance
    # Cache client
```

**Generation:**
```python
def generate_with_gemini(text, task, question, max_tokens):
    # 1. Create prompt theo task
    # 2. Call API với generation config
    # 3. Return response text
```

**Generation Config:**
- Temperature: 0.7
- Top-p: 0.95
- Top-k: 40
- Max output tokens: configurable

**Lợi ích:**
- **Tốc độ**: ~200-500 tokens/second
- **Không cần GPU**: Chạy trên cloud
- **Chất lượng cao**: Model lớn hơn, hiểu context tốt hơn
- **Dễ scale**: Không giới hạn bởi hardware local

---

### 4. Unified Model Manager

#### Công Nghệ Sử Dụng

**Design Pattern:**
- **Singleton Pattern**: Global instance management
- **Lazy Loading**: Models load khi cần
- **Provider Abstraction**: Hỗ trợ nhiều LLM providers

#### Triển Khai (`src/models/unified.py`)

**Class Structure:**
```python
class UnifiedModelManager:
    def __init__(self, asr_model, gen_model, llm_provider):
        # Initialize với config
        # Models sẽ load lazy
    
    def process_audio(self, audio_path, task, question):
        # 1. Transcribe với Whisper
        # 2. Generate với LLM (Qwen hoặc Gemini)
        # 3. Return kết quả
```

**Features:**
- **Unified Interface**: Một interface cho cả ASR và LLM
- **Provider Selection**: Chọn Qwen (local) hoặc Gemini (API)
- **Task Support**: Multi-task (summarize, answer, translate, etc.)
- **Error Handling**: Graceful error handling và fallback

**Usage:**
```python
manager = get_unified_manager(llm_provider="gemini")
result = manager.process_audio(
    audio_path="audio.wav",
    task="summarize"
)
```

---

### 5. RAG Pipeline (Retrieval Augmented Generation)

#### Công Nghệ Sử Dụng

**Components:**
1. **Hybrid Retriever**: BM25 (lexical) + FAISS (semantic)
2. **Reranker**: Cross-encoder model
3. **Vector Store**: FAISS index
4. **Embedding Model**: Sentence Transformers

**Libraries:**
- `sentence-transformers`: Embedding generation
- `faiss`: Vector similarity search
- `rank_bm25`: BM25 lexical search
- `cross-encoder`: Reranking model

#### Triển Khai Chi Tiết

**1. Index Building (`src/rag/indexer.py`)**

```python
class HybridIndex:
    def __init__(self, embed_model):
        # Initialize embedding model
        # Prepare BM25 và FAISS index
    
    def build(self):
        # 1. Tokenize texts cho BM25
        # 2. Generate embeddings cho FAISS
        # 3. Build indexes
```

**Process:**
1. Load transcripts từ JSONL
2. Generate embeddings với Sentence Transformers
3. Build BM25 index (lexical search)
4. Build FAISS index (semantic search)
5. Save indexes to disk

**2. Hybrid Retrieval (`src/rag/retriever.py`)**

```python
class HybridRetriever:
    def search(self, query, k):
        # 1. BM25 search (lexical)
        # 2. FAISS search (semantic)
        # 3. Combine scores (hybrid)
        # 4. Return top-k
```

**Hybrid Scoring:**
- BM25 scores: Lexical matching
- FAISS scores: Semantic similarity
- Combined: Sum of normalized scores

**3. Reranking (`src/rag/reranker.py`)**

**Model:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Cross-encoder architecture (query + document)
- Fine-tuned trên MS MARCO dataset

**Process:**
```python
def rerank(self, query, hits, top_k):
    # 1. Score query-document pairs
    # 2. Sort by scores
    # 3. Return top-k
```

**4. RAG Pipeline (`src/rag/pipeline.py`)**

```python
class RAGPipeline:
    def ask(self, query, k):
        # 1. Retrieve với HybridRetriever
        # 2. Rerank với Reranker
        # 3. Generate answer với citations
        # 4. Return answer + contexts
```

**Workflow:**
1. Query → Hybrid Retrieval (BM25 + FAISS)
2. Rerank results với cross-encoder
3. Generate answer với LLM (Qwen hoặc Gemini)
4. Include citations từ retrieved contexts

---

### 6. REST API Server

#### Công Nghệ Sử Dụng

**Framework:**
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

**Features:**
- RESTful API design
- Swagger UI documentation
- CORS support
- File upload handling
- Error handling

#### Triển Khai (`src/api/server.py`)

**Endpoints:**

1. **`/health`** - Health check
   - Kiểm tra models đã load
   - Kiểm tra vector store availability

2. **`/transcribe`** - Transcribe từ file path
   - Input: `audio_path` (JSON)
   - Output: `text`, `segments`, `language`

3. **`/transcribe/upload`** - Upload và transcribe
   - Input: `file` (multipart/form-data)
   - Parameters: `language`, `model_size`
   - Auto preprocessing audio

4. **`/audio-to-answer`** - Whisper + LLM pipeline
   - Input: `audio_path`, `task`, `question`, `llm_provider`
   - Output: `transcription`, `response`, `language`, `task`

5. **`/audio-to-answer/upload`** - Upload audio + process
   - Input: `file`, `task`, `question`, `llm_provider`
   - Auto preprocessing

6. **`/ask`** - RAG question answering
   - Input: `query`, `top_k`
   - Output: `answer`, `contexts` (citations)

**Middleware:**
- **CORS**: Cho phép cross-origin requests
- **Error Handling**: Graceful error responses

**Lazy Loading:**
- Models load khi cần (không load khi startup)
- Singleton pattern cho models
- Caching để tối ưu performance

---

## 📦 Dependencies và Libraries

### Core Dependencies

**ASR:**
- `faster-whisper`: CTranslate2 wrapper cho Whisper
- `transformers`: HuggingFace Transformers (cho fine-tuning)
- `librosa`: Audio processing
- `torch`: PyTorch (deep learning framework)

**LLM:**
- `transformers`: Model loading và inference
- `peft`: Parameter-Efficient Fine-Tuning (LoRA)
- `unsloth`: Optimized training framework
- `bitsandbytes`: Quantization (4-bit/8-bit)
- `google-generativeai`: Gemini API client

**RAG:**
- `sentence-transformers`: Embedding models
- `faiss-cpu` / `faiss-gpu`: Vector similarity search
- `rank-bm25`: BM25 lexical search
- `torch`: Cross-encoder reranking

**API:**
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `pydantic`: Data validation
- `python-multipart`: File upload support

**Utilities:**
- `python-dotenv`: Environment variables
- `numpy`: Numerical operations
- `datasets`: HuggingFace datasets
- `evaluate`: Metrics evaluation

---

## 🔄 Data Flow và Workflows

### Workflow 1: Audio → Text (Transcription)

```
Audio File
  ↓
[Preprocessing]
  - Resample to 16kHz
  - Mono channel
  - Normalize
  ↓
[Whisper ASR]
  - Feature extraction (log-Mel spectrogram)
  - Encoder (Transformer)
  - Decoder (Beam search)
  ↓
Text + Segments + Language
```

### Workflow 2: Audio → Answer (Whisper + LLM)

```
Audio File
  ↓
[Whisper ASR] → Text
  ↓
[LLM Processing]
  - Prompt creation (theo task)
  - Tokenization
  - Generation
  ↓
Transcription + Response
```

**Task Types:**
- `summarize`: Tóm tắt transcription
- `answer`: Trả lời câu hỏi từ transcription
- `translate`: Dịch transcription sang tiếng Anh
- `analyze`: Phân tích nội dung
- `extract`: Trích xuất thông tin quan trọng

### Workflow 3: Question → Answer (RAG)

```
Question
  ↓
[Hybrid Retrieval]
  - BM25 (lexical search)
  - FAISS (semantic search)
  - Combine scores
  ↓
[Top-k Candidates]
  ↓
[Reranking]
  - Cross-encoder scoring
  - Sort by relevance
  ↓
[Top-k Reranked]
  ↓
[LLM Generation]
  - Create prompt với contexts
  - Generate answer với citations
  ↓
Answer + Citations
```

---

## 🎯 Optimization Techniques

### 1. Model Optimization

**ASR:**
- **CTranslate2 format**: ~5x nhanh hơn HuggingFace
- **Quantization**: int8/float16 để giảm memory
- **Caching**: Cache models để tránh reload
- **Lazy loading**: Chỉ load khi cần

**LLM:**
- **Merged model**: Không cần PEFT runtime
- **4-bit quantization**: Giảm memory footprint
- **Device mapping**: Auto placement trên GPU/CPU
- **Batch processing**: Có thể mở rộng cho batch inference

### 2. Training Optimization

**Whisper:**
- **BF16 training**: Cho GPU Ampere+ (RTX 30xx+)
- **Gradient checkpointing**: Tiết kiệm VRAM
- **Mixed precision**: FP16/BF16 training
- **Data loading**: Optimized audio loading

**Qwen:**
- **Unsloth**: Optimized training framework
- **QLoRA**: 4-bit quantization + LoRA
- **8-bit optimizer**: AdamW 8-bit (tiết kiệm 75% VRAM)
- **Gradient accumulation**: Effective batch size lớn hơn

### 3. Inference Optimization

**Caching:**
- Model caching (theo key)
- Embedding caching (có thể mở rộng)
- Response caching (có thể mở rộng)

**Parallel Processing:**
- Batch inference (có thể mở rộng)
- Async API endpoints
- Multi-threading cho I/O

**Memory Management:**
- Lazy loading
- Model offloading (có thể mở rộng)
- Garbage collection

---

## 🔐 Configuration Management

### Environment Variables (`.env`)

**Model Paths:**
```env
ASR_FINETUNED_MODEL=./models/final/whisper-vi-en-ct2
GEN_FINETUNED_MODEL=./models/finetuned/qwen-mixed-merged
```

**Device Configuration:**
```env
ASR_DEVICE=auto          # auto, cuda, cpu
ASR_COMPUTE=float16      # float16, int8_float16, int8
```

**Generation Configuration:**
```env
GEN_MAX_TOKENS=512
LLM_PROVIDER=qwen        # qwen (local) or gemini (API)
```

**Gemini API:**
```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.5-flash
```

**RAG Configuration:**
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
VECTOR_DIR=./vectorstore
```

**Data Directories:**
```env
DATA_DIR=./data
MODELS_DIR=./models
```

---

## 📊 Performance Metrics

### ASR Performance

**Whisper Small (CTranslate2):**
- **GPU (RTX 5060 Ti)**: ~1.4 samples/second
- **CPU**: ~0.3 samples/second
- **Model size**: 463MB (float16)
- **Latency**: ~0.7s per sample (GPU)

### LLM Performance

**Qwen 0.5B (Local):**
- **GPU**: ~50-100 tokens/second
- **CPU**: ~10-20 tokens/second
- **Model size**: 513MB
- **Memory**: ~2-3GB VRAM (4-bit)

**Gemini API:**
- **Speed**: ~200-500 tokens/second
- **Latency**: ~100-500ms (network dependent)
- **No local resources**: Chạy trên cloud

### RAG Performance

**Retrieval:**
- **BM25**: ~1-5ms per query
- **FAISS**: ~5-20ms per query (tùy index size)
- **Reranking**: ~50-200ms per query (tùy số candidates)

**End-to-end:**
- **Total latency**: ~200-500ms (không tính LLM generation)

---

## 🚀 Deployment Considerations

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 5GB (cho models)

**Recommended:**
- GPU: RTX 3060 Ti / RTX 5060 Ti (16GB VRAM)
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 10GB+ (cho models + data)

### Software Requirements

- Python 3.10+
- CUDA 11.8+ (cho GPU)
- ffmpeg 4.4+ (cho audio processing)

### Scalability

**Horizontal Scaling:**
- API server có thể scale với load balancer
- Models có thể deploy trên multiple servers
- Vector store có thể shard

**Vertical Scaling:**
- GPU memory: Có thể dùng GPU lớn hơn
- CPU: Có thể tăng cores
- RAM: Có thể tăng memory

**Optimization:**
- Model quantization: Giảm memory
- Batch processing: Tăng throughput
- Caching: Giảm latency

---

## 📝 Best Practices

### Code Organization

- **Modular design**: Mỗi component độc lập
- **Separation of concerns**: API, models, utilities tách biệt
- **Configuration management**: Centralized config
- **Error handling**: Graceful error handling

### Model Management

- **Version control**: Track model versions
- **Model caching**: Cache để tránh reload
- **Lazy loading**: Chỉ load khi cần
- **Memory management**: Monitor và optimize memory

### API Design

- **RESTful**: Follow REST principles
- **Documentation**: Swagger UI
- **Error responses**: Consistent error format
- **Validation**: Pydantic validation

### Training

- **Data quality**: Ensure high-quality training data
- **Hyperparameter tuning**: Tune cho dataset cụ thể
- **Evaluation**: Regular evaluation với test set
- **Monitoring**: Monitor training metrics

---

## 🔮 Future Enhancements

### Potential Improvements

1. **Streaming Support**
   - Real-time transcription
   - Streaming LLM responses
   - WebSocket API

2. **Advanced RAG**
   - Multi-hop reasoning
   - Graph-based retrieval
   - Query expansion

3. **Multi-modal**
   - Image understanding
   - Video processing
   - Document understanding

4. **Optimization**
   - Model distillation
   - Pruning
   - Advanced quantization

5. **Monitoring**
   - Metrics collection
   - Logging
   - Performance monitoring

---

## 📚 References

### Papers và Documentation

- **Whisper**: [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- **Qwen**: [Qwen2.5 Documentation](https://github.com/QwenLM/Qwen2.5)
- **LoRA**: [LoRA Paper](https://arxiv.org/abs/2106.09685)
- **QLoRA**: [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- **RAG**: [RAG Paper](https://arxiv.org/abs/2005.11401)

### Libraries

- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [Unsloth](https://github.com/unslothai/unsloth)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 📄 License

[Your License]

---

**Tài liệu này được tạo tự động dựa trên codebase hiện tại. Cập nhật lần cuối: 2024**

