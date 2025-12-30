# Implementation Plan: Scalable Text-Only Foundational Model
**Version:** v1.0
**Date:** 2025-11-23
**Status:** Design Phase

---

## Executive Summary

This document provides the complete technical implementation plan for building a scalable text-only foundational model from scratch. The design incorporates architectural patterns from DeepSeek R1, Llama 3.1, Mistral, and other top-performing open-source models. All components are engineered for M1 prototype execution with zero-rewrite scalability to cluster-scale training.

---

## 1. Architecture Design

### 1.1 Transformer Core Architecture

**Base Configuration:**
- **Type:** Decoder-only transformer (GPT-style)
- **Normalization:** Pre-LayerNorm architecture with RMSNorm (following Llama 3.1, DeepSeek)
- **Activation:** SwiGLU activation in FFN (following Llama 3.1, DeepSeek)
- **Positional Encoding:** RoPE (Rotary Position Embeddings)

**Architectural Components:**

```python
# Configuration schema (scalable from 10M to multi-B parameters)
ModelConfig:
    vocab_size: int            # 8k (prototype), 32k-100k (production)
    n_layers: int              # 6-12 (prototype), 32-80 (production)
    d_model: int               # 512-768 (prototype), 2048-8192 (production)
    n_heads: int               # 8-12 (prototype), 32-128 (production)
    head_dim: int = 64         # Fixed at 64 (industry standard)
    ffn_mult: float            # 4.0 (standard), up to 8.0 (sparse MoE future)
    max_seq_len: int           # 256-512 (prototype), 4096-128k (production)
    dropout: float             # 0.0-0.1
    vocab_size: int
    tie_embeddings: bool = True  # Tie input/output embeddings
    rope_base: float = 10000.0   # 500k for long context (Llama 3.1)
    rope_scaling: dict = None    # For context extension
```

**Layer Structure:**
```
TransformerBlock:
    1. RMSNorm (pre-attention)
    2. Multi-Head Attention with RoPE
       - Optional: Grouped Query Attention (GQA) for efficiency
       - Optional: FlashAttention-2/3 kernel integration
    3. Residual connection
    4. RMSNorm (pre-FFN)
    5. SwiGLU FFN
    6. Residual connection
```

### 1.2 Attention Mechanism Options

**Standard Multi-Head Attention (MHA):**
- Prototype implementation
- n_heads = n_kv_heads
- Simpler implementation, easier debugging

**Grouped Query Attention (GQA):**
- Production optimization (Llama 3.1 pattern)
- n_kv_heads = n_heads // 4 or n_heads // 8
- Reduces KV cache by 4-8x
- Critical for long context inference

**Multi-Query Attention (MQA):**
- Alternative: n_kv_heads = 1
- Maximum KV cache reduction
- Used in some models (Falcon, StarCoder)

**Implementation Priority:**
1. MHA for prototype (weeks 1-4)
2. GQA for scalable version (week 5+)
3. Optional MQA variant for experimentation

### 1.3 Positional Encoding: RoPE

**Implementation Details:**
- Uses complex number rotations to encode position
- Applied to Q and K after projection, before attention
- Inherently supports relative position information
- No learned parameters required

**Configuration:**
```python
RoPEConfig:
    dim: int                    # head_dim
    base: float = 10000.0       # Standard: 10k, Long context: 500k
    max_seq_len: int            # Maximum supported sequence
    scaling_factor: float = 1.0 # For linear/dynamic scaling
```

**Advantages over learned embeddings:**
- Zero additional parameters
- Naturally extends to longer sequences
- Better length generalization
- Used by Llama, DeepSeek, Mistral, Qwen

### 1.4 Normalization: RMSNorm

**Why RMSNorm over LayerNorm:**
- 10-15% faster computation
- No bias parameters (simplicity)
- Equivalent performance to LayerNorm
- Standard in modern LLMs (Llama, DeepSeek, Mistral)

**Implementation:**
```python
RMSNorm:
    weight: Parameter[d_model]
    eps: float = 1e-6

    forward(x):
        rms = sqrt(mean(x^2) + eps)
        return weight * (x / rms)
```

### 1.5 Feed-Forward Network: SwiGLU

**Architecture:**
```
FFN(x):
    gate = Linear(d_model, ffn_dim, bias=False)
    up = Linear(d_model, ffn_dim, bias=False)
    down = Linear(ffn_dim, d_model, bias=False)

    return down(silu(gate(x)) * up(x))
```

**Parameters:**
- ffn_dim = int(ffn_mult * d_model)
- Standard ffn_mult = 4.0 (8/3 * 4 ≈ 10.67 effective for SwiGLU)
- No bias terms (following Llama 3.1)

**Advantages:**
- Better performance than ReLU/GELU
- Gating mechanism improves expressiveness
- Standard in modern architectures

---

## 2. Tokenizer Implementation

### 2.1 Tokenizer Choice: SentencePiece BPE

**Rationale:**
- Language-agnostic (no pre-tokenization required)
- Handles whitespace as tokens (critical for code, structured text)
- Reversible encoding/decoding
- Production-ready (used by Llama, T5, ALBERT)
- Fast training and inference (50k sentences/sec)

### 2.2 Training Configuration

**Prototype (M1):**
```python
SentencePieceConfig:
    model_type: "bpe"           # BPE algorithm
    vocab_size: 8000            # Small for fast iteration
    character_coverage: 1.0     # English-focused
    model_prefix: "tokenizer_v1"
    normalization_rule_name: "nfkc"
    remove_extra_whitespaces: False  # Preserve for code
    add_dummy_prefix: True      # Space normalization
    split_by_unicode_script: True
    split_by_whitespace: True
    byte_fallback: True         # Handle unknown bytes
    unk_id: 0
    bos_id: 1
    eos_id: 2
    pad_id: 3
```

**Production:**
```python
SentencePieceConfig:
    vocab_size: 32000           # Llama-style
    # OR
    vocab_size: 100000          # Llama 3.1 style (better multilingual)
```

### 2.3 Training Pipeline

**Input Data:**
- Deduplicated, cleaned text corpus
- Minimum 100MB for prototype (10GB+ for production)
- UTF-8 encoded plain text

**Training Steps:**
```bash
# 1. Prepare training corpus
cat corpus/*.txt > training_data.txt

# 2. Train SentencePiece model
spm_train \
    --input=training_data.txt \
    --model_prefix=tokenizer_v1 \
    --vocab_size=8000 \
    --model_type=bpe \
    --character_coverage=1.0 \
    --byte_fallback=true \
    --normalization_rule_name=nfkc

# 3. Generate artifacts
# Outputs: tokenizer_v1.model, tokenizer_v1.vocab
```

### 2.4 Versioning and Reproducibility

**Artifacts to Track:**
- `tokenizer_v1.model` (SentencePiece binary model)
- `tokenizer_v1.vocab` (Human-readable vocabulary)
- `tokenizer_config.json` (Training configuration)
- `special_tokens.json` (BOS, EOS, PAD, UNK mappings)

**Determinism Requirements:**
- Fixed input corpus (checksummed)
- Fixed random seed for SentencePiece training
- Versioned model files (Git LFS recommended)

### 2.5 Special Token Design

```python
SpecialTokens:
    unk_token: "<unk>"    # ID: 0
    bos_token: "<s>"      # ID: 1
    eos_token: "</s>"     # ID: 2
    pad_token: "<pad>"    # ID: 3
    # Reserve IDs 4-15 for future special tokens
```

---

## 3. Data Pipeline Architecture

### 3.1 Data Sources

**Prototype (M1):**
- Wikipedia (English): 20GB raw text
- BookCorpus or similar: 5GB
- CC-News (filtered): 5GB
- Synthetic validation set: 100MB

**Production:**
- Common Crawl (filtered): 500GB-5TB
- Wikipedia (multilingual): 100GB
- Books3/PG19: 100GB
- ArXiv papers: 50GB
- GitHub code (optional future): 100GB
- StackExchange: 20GB

### 3.2 Preprocessing Pipeline

**Stage 1: Raw Text Cleaning**
```python
CleaningPipeline:
    1. HTML/XML removal (Beautiful Soup or trafilatura)
    2. Unicode normalization (NFKC)
    3. Whitespace normalization (preserve structure)
    4. Line break standardization
    5. Remove control characters (except \n, \t)
    6. Low-entropy filtering:
       - Remove documents with >50% punctuation
       - Remove documents with <10 unique words
       - Remove docs with mean word length <2 or >15
    7. Language filtering (fastText language ID)
       - Target: English (en) with confidence >0.8
    8. Quality filtering:
       - Heuristic-based (Gopher rules)
       - Perplexity-based (optional: KenLM model)
```

**Stage 2: Deduplication**
```python
DeduplicationPipeline:
    Algorithm: MinHash + LSH

    Parameters:
        num_perm: int = 128           # MinHash permutations
        threshold: float = 0.8        # Jaccard similarity threshold
        ngram_size: int = 5          # For shingling

    Process:
        1. Generate shingles (5-grams) for each document
        2. Compute MinHash signatures (128 permutations)
        3. Use LSH for candidate pair generation
        4. Compute exact Jaccard for candidates
        5. Remove duplicates (keep first occurrence)

    Implementation:
        - datasketch library (prototype)
        - NeMo Curator (production, GPU-accelerated)
        - Custom CUDA kernels (advanced optimization)
```

**Stage 3: Document Filtering**
```python
FilteringRules:
    # Following Llama/Gopher/C4 practices
    min_doc_length: int = 100        # tokens
    max_doc_length: int = 100000     # tokens
    min_word_count: int = 50
    max_word_count: int = 100000

    # PII filtering (basic)
    email_pattern: regex
    phone_pattern: regex
    ssn_pattern: regex

    # Toxicity filtering (basic)
    blocklist: List[str]             # Bad words list
    toxicity_threshold: float = 0.5  # Perspective API score
```

### 3.3 Tokenization Stage

**Pre-tokenization Strategy:**
```python
TokenizationPipeline:
    Input: Cleaned, deduplicated documents (.jsonl)
    Output: Binary token arrays (.bin) + index

    Process:
        for document in documents:
            tokens = tokenizer.encode(document['text'])
            tokens = [BOS_ID] + tokens + [EOS_ID]
            write_binary(tokens, dtype=uint16)  # Support vocab up to 65k

    Index Format:
        document_id: int64
        byte_offset: int64
        token_count: int32
```

### 3.4 Sharding Strategy

**Shard Configuration:**
```python
ShardConfig:
    shard_size_mb: int = 500          # 500MB per shard
    format: str = "binary"            # or "webdataset", "arrow"
    compression: str = "none"         # or "lz4", "zstd"

    Structure:
        data/
            train/
                shard_0000.bin
                shard_0000.idx
                shard_0001.bin
                shard_0001.idx
                ...
                manifest.json
            val/
                shard_0000.bin
                shard_0000.idx
                manifest.json
```

**Manifest Format:**
```json
{
    "version": "1.0",
    "tokenizer_version": "v1",
    "total_shards": 100,
    "total_tokens": 5000000000,
    "total_documents": 10000000,
    "shards": [
        {
            "shard_id": 0,
            "path": "shard_0000.bin",
            "index_path": "shard_0000.idx",
            "token_count": 50000000,
            "document_count": 100000,
            "checksum": "sha256:..."
        }
    ]
}
```

### 3.5 Data Loading for Training

**DataLoader Requirements:**
- Deterministic shuffling (reproducibility)
- Efficient random access (memory-mapped files)
- Multi-worker prefetching
- Sequence packing for efficiency

**Implementation:**
```python
TokenDataset:
    def __init__(
        self,
        shard_dir: Path,
        seq_len: int,
        shuffle: bool = True,
        seed: int = 42
    ):
        self.shards = load_manifest(shard_dir)
        self.seq_len = seq_len
        self.rng = np.random.RandomState(seed)

    def __getitem__(self, idx):
        # Memory-mapped random access
        shard_idx, local_idx = self.map_global_to_local(idx)
        tokens = self.read_tokens(shard_idx, local_idx, self.seq_len + 1)

        # Causal LM format: input = tokens[:-1], target = tokens[1:]
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }
```

**Sequence Packing Strategy (Advanced):**
- Pack multiple documents into single sequence
- Prevents padding waste
- Requires attention masking to prevent cross-document attention
- Increases token throughput by 20-30%

---

## 4. Training System Design

### 4.1 Training Objective

**Loss Function:**
```python
def compute_loss(logits, labels):
    # Logits: [batch, seq_len, vocab_size]
    # Labels: [batch, seq_len]

    # Shift is already handled in data loading
    # logits corresponds to predicting labels

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=PAD_TOKEN_ID,
        reduction='mean'
    )

    return loss
```

**Optional Enhancements:**
- Label smoothing (epsilon=0.1) for regularization
- Z-loss (DeepSeek) for logit stability
- Auxiliary losses (future: MoE load balancing)

### 4.2 Optimizer Configuration

**AdamW Settings:**
```python
OptimizerConfig:
    optimizer: "AdamW"
    learning_rate: float
        # Prototype: 6e-4 (small models)
        # Production: 3e-4 (standard), 1.5e-4 (large models)

    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1

    # Decay applied to all params except:
    # - Biases (none in our architecture)
    # - LayerNorm/RMSNorm weights
    # - Embeddings (optional)
```

**Per-Parameter Learning Rates (Advanced):**
```python
param_groups = [
    {'params': embed_params, 'lr': lr * 0.1},      # Lower LR for embeddings
    {'params': attention_params, 'lr': lr},
    {'params': ffn_params, 'lr': lr},
    {'params': norm_params, 'lr': lr, 'weight_decay': 0.0}
]
```

### 4.3 Learning Rate Schedule

**Warmup + Cosine Decay:**
```python
SchedulerConfig:
    warmup_steps: int
        # Prototype: 500-2000 steps
        # Production: 2000-10000 steps
        # Rule of thumb: 2-5% of total steps

    max_steps: int
        # Total training steps

    min_lr_ratio: float = 0.1
        # Final LR = max_lr * min_lr_ratio

Schedule:
    if step < warmup_steps:
        lr = max_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

**Alternative: Warmup-Stable-Decay (WSD):**
```python
# Used in some recent models
# Warmup (5%) -> Stable (85%) -> Decay (10%)
WSDSchedule:
    warmup_ratio: float = 0.05
    stable_ratio: float = 0.85
    decay_ratio: float = 0.10
```

### 4.4 Gradient Management

**Gradient Clipping:**
```python
ClippingConfig:
    max_grad_norm: float = 1.0

# Applied before optimizer step
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=max_grad_norm,
    norm_type=2.0
)
```

**Gradient Accumulation:**
```python
AccumulationConfig:
    gradient_accumulation_steps: int
        # Prototype: 1-4 (depends on batch size fit)
        # Production: 4-32 (for large effective batch)

# Effective batch size = micro_batch * accum_steps * world_size
```

### 4.5 Mixed Precision Training

**Prototype (M1 GPU):**
```python
PrecisionConfig:
    dtype: torch.float16      # or torch.bfloat16 if supported

    # Using torch.cuda.amp
    scaler = GradScaler()

    with autocast(dtype=torch.float16):
        logits = model(input_ids)
        loss = compute_loss(logits, labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
```

**Production (Cluster):**
```python
PrecisionConfig:
    dtype: torch.bfloat16     # Preferred for stability

    # Optional: FP8 training (H100+)
    # Requires nvidia-transformer-engine
```

### 4.6 Batch Size Strategy

**Prototype:**
```python
BatchConfig:
    micro_batch_size: int = 4-16      # Fits in M1 memory
    seq_len: int = 256-512
    gradient_accumulation_steps: int = 4

    effective_batch_size = micro_batch_size * gradient_accumulation_steps
    effective_tokens_per_step = effective_batch_size * seq_len
```

**Production:**
```python
BatchConfig:
    micro_batch_size: int = 8-32
    seq_len: int = 4096
    gradient_accumulation_steps: int = 1
    world_size: int = 8-512

    effective_batch_size = micro_batch_size * world_size
    effective_tokens_per_step = effective_batch_size * seq_len

# Target: 2-4M tokens per step (GPT-3/Llama style)
```

### 4.7 Training Loop Structure

```python
def train_step(model, batch, optimizer, scheduler, scaler):
    """Single training step"""

    # Forward pass
    with autocast(dtype=torch.bfloat16):
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        loss = compute_loss(outputs.logits, batch['labels'])
        loss = loss / gradient_accumulation_steps

    # Backward pass
    scaler.scale(loss).backward()

    # Optimizer step (if accumulation complete)
    if (step + 1) % gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)

        # Gradient clipping
        grad_norm = clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        )

        # Optimizer update
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        return loss.item() * gradient_accumulation_steps, grad_norm

    return None, None
```

---

## 5. Distributed Training Architecture

### 5.1 Scaling Strategy

**Prototype → Production Migration Path:**

```
Stage 1: Single GPU (M1)
    - Model: 5-50M params
    - Batch: 4-16 samples
    - No distributed code required

Stage 2: Single Node Multi-GPU
    - Model: 50M-1B params
    - Strategy: DDP (Data Parallel)
    - 2-8 GPUs

Stage 3: Multi-Node (Small Cluster)
    - Model: 1B-7B params
    - Strategy: FSDP (Fully Sharded Data Parallel)
    - 8-64 GPUs (2-8 nodes)

Stage 4: Large Cluster
    - Model: 7B-70B+ params
    - Strategy: FSDP + optional tensor parallelism
    - 64-1024 GPUs (8-128 nodes)

Stage 5: Extreme Scale
    - Model: 100B-1T params
    - Strategy: 3D parallelism (FSDP + TP + PP)
    - Framework: Megatron-DeepSpeed or NeMo
```

### 5.2 Framework Selection

**FSDP (Fully Sharded Data Parallel):**
```python
# Native PyTorch, best for getting started
Advantages:
    - Native PyTorch integration
    - Simple API
    - Good for 1B-70B models
    - Auto-wrapping support
    - CPU offloading built-in

Configuration:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=transformer_auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )
```

**DeepSpeed ZeRO:**
```python
# Microsoft's framework, maximum flexibility
Advantages:
    - More configuration options than FSDP
    - ZeRO-Offload (CPU/NVMe)
    - ZeRO-Infinity for extreme scale
    - Better memory optimization
    - Pipeline parallelism support

Configuration (ZeRO-3):
    {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",

        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "initial_scale_power": 16
        },

        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            },
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
        }
    }
```

**Megatron-LM (NVIDIA):**
```python
# For tensor/pipeline parallelism at extreme scale
Advantages:
    - Best for 100B+ parameter models
    - Tensor parallelism (split layers across GPUs)
    - Pipeline parallelism (split model vertically)
    - Optimized CUDA kernels
    - Used by Megatron-DeepSpeed combo

Use Cases:
    - Models >70B parameters
    - Need tensor parallelism (model doesn't fit in GPU even with FSDP)
    - Maximum performance required
```

**Recommendation:**
- **Prototype (M1):** No distributed training
- **Production (up to 13B):** FSDP (simpler, native PyTorch)
- **Production (13B-70B):** FSDP or DeepSpeed ZeRO-3
- **Production (70B+):** DeepSpeed or Megatron-DeepSpeed

### 5.3 FSDP Implementation Details

**Wrapping Strategy:**
```python
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy
)

# Option 1: Transformer-aware wrapping
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        TransformerBlock,  # Our transformer layer class
    },
)

# Option 2: Size-based wrapping
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1e6  # Wrap modules with >1M params
)
```

**Mixed Precision Policy:**
```python
from torch.distributed.fsdp import MixedPrecision

mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
```

**Activation Checkpointing:**
```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Apply to transformer blocks to save memory
check_fn = lambda submodule: isinstance(submodule, TransformerBlock)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=check_fn,
)
```

### 5.4 Communication Optimization

**Gradient Compression (Optional):**
- Reduces communication bandwidth
- Trade accuracy for speed
- PowerSGD or 1-bit Adam

**Overlapping Computation and Communication:**
- FSDP does this automatically
- DeepSpeed has overlap_comm flag
- Critical for multi-node efficiency

**Network Requirements:**
- Prototype: N/A
- Single node: PCIe 4.0 / NVLink
- Multi-node: 100Gbps+ InfiniBand (recommended)
- Minimum: 25Gbps Ethernet (acceptable for <8 nodes)

---

## 6. Checkpoint System Design

### 6.1 Checkpoint Format

**Single-GPU Checkpoint:**
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # For AMP

    # Training state
    'step': int,
    'epoch': int,
    'tokens_processed': int,
    'best_val_loss': float,

    # Configuration
    'model_config': model_config.to_dict(),
    'train_config': train_config.to_dict(),

    # Reproducibility
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state(),
    'numpy_rng_state': np.random.get_state(),
    'python_rng_state': random.getstate(),

    # Metadata
    'timestamp': datetime.now().isoformat(),
    'git_commit': git_commit_hash,
}

# Save
torch.save(checkpoint, f'checkpoint_step_{step}.pt')
```

### 6.2 Distributed Checkpoint (FSDP)

**Sharded Checkpoint:**
```python
from torch.distributed.checkpoint import (
    save_state_dict,
    load_state_dict,
    FileSystemReader,
    FileSystemWriter,
)

# Save (each rank saves its shard)
state_dict = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}

save_state_dict(
    state_dict=state_dict,
    storage_writer=FileSystemWriter(f'checkpoint_{step}/'),
    planner=DefaultSavePlanner(),
)

# Save training state separately (rank 0 only)
if rank == 0:
    training_state = {
        'step': step,
        'tokens_processed': tokens_processed,
        'config': config.to_dict(),
    }
    torch.save(training_state, f'checkpoint_{step}/training_state.pt')
```

**Loading with Different World Size:**
```python
# Can load even if number of GPUs changed
state_dict = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}

load_state_dict(
    state_dict=state_dict,
    storage_reader=FileSystemReader(f'checkpoint_{step}/'),
    planner=DefaultLoadPlanner(),
)

model.load_state_dict(state_dict['model'])
optimizer.load_state_dict(state_dict['optimizer'])
```

### 6.3 Safetensors Format

**For Model Weights Only:**
```python
from safetensors.torch import save_file, load_file

# Save model weights (no optimizer state)
save_file(
    model.state_dict(),
    f'model_step_{step}.safetensors',
    metadata={
        'step': str(step),
        'model_config': json.dumps(model_config.to_dict()),
    }
)

# Load
state_dict = load_file(f'model_step_{step}.safetensors')
model.load_state_dict(state_dict)
```

**Advantages:**
- Fast loading (memory-mapped)
- Safe (no arbitrary code execution)
- Cross-framework compatible
- HuggingFace standard

**Limitations:**
- Cannot store optimizer state (use PyTorch .pt for full checkpoints)
- Best for inference-only or final model weights

### 6.4 Checkpoint Strategy

**Prototype:**
```python
CheckpointConfig:
    save_every_steps: int = 1000
    keep_last_n: int = 5
    save_best: bool = True
    save_format: str = "pytorch"
```

**Production:**
```python
CheckpointConfig:
    save_every_steps: int = 5000
    save_every_hours: int = 6        # Walltime-based backup
    keep_last_n: int = 3
    keep_best_n: int = 3
    save_format: str = "distributed"  # FSDP/DeepSpeed
    async_save: bool = True          # Non-blocking checkpoint
```

### 6.5 Checkpoint Organization

```
checkpoints/
    run_20251123_v1/
        checkpoint_step_5000/
            model/                  # FSDP sharded model
                __0_0.distcp
                __1_0.distcp
                ...
            optimizer/              # FSDP sharded optimizer
                __0_0.distcp
                __1_0.distcp
                ...
            training_state.pt       # Step, config, RNG states
            metadata.json
        checkpoint_step_10000/
            ...
        best_checkpoint/
            -> symlink to best checkpoint
        latest_checkpoint/
            -> symlink to latest checkpoint
```

---

## 7. Evaluation Framework

### 7.1 Training Metrics

**Per-Step Metrics:**
```python
Metrics:
    loss: float                      # Cross-entropy loss
    perplexity: float                # exp(loss)
    learning_rate: float
    grad_norm: float                 # After clipping
    param_norm: float                # L2 norm of parameters
    tokens_per_second: float
    samples_per_second: float
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
```

**Per-Epoch/Validation Metrics:**
```python
ValidationMetrics:
    val_loss: float
    val_perplexity: float

    # Optional: downstream tasks
    lambada_accuracy: float
    hellaswag_accuracy: float
```

### 7.2 Validation Strategy

**Prototype:**
```python
ValidationConfig:
    eval_every_steps: int = 500
    eval_steps: int = 100            # Num validation batches
    val_batch_size: int = 8
```

**Production:**
```python
ValidationConfig:
    eval_every_steps: int = 2000
    eval_steps: int = 500
    val_batch_size: int = 16

    # Multiple validation sets
    val_sets: List[str] = [
        "val_general",               # Main validation
        "val_books",                 # Domain-specific
        "val_code",
        "val_reasoning",
    ]
```

### 7.3 Benchmark Suite

**Prototype Benchmarks:**
```python
Benchmarks:
    1. WikiText-2 perplexity
    2. Synthetic cloze tasks (simple completion)
    3. Few-shot generation quality (manual inspection)
```

**Production Benchmarks:**
```python
# Using lm-evaluation-harness
Benchmarks:
    # Zero-shot
    - HellaSwag (commonsense reasoning)
    - PIQA (physical reasoning)
    - WinoGrande (coreference resolution)
    - ARC-Easy, ARC-Challenge (science QA)

    # Few-shot
    - LAMBADA (context understanding)
    - StoryCloze (narrative understanding)

    # Perplexity
    - WikiText-103
    - Penn Tree Bank (PTB)
    - C4 validation set

    # Generation
    - ROUGE/BLEU on summarization
    - Human evaluation on samples
```

### 7.4 Evaluation Implementation

**Integration:**
```python
# Use EleutherAI lm-evaluation-harness
from lm_eval import evaluator, tasks

results = evaluator.simple_evaluate(
    model=model_wrapper,
    tasks=["hellaswag", "piqa", "winogrande", "lambada"],
    num_fewshot=0,
    batch_size=8,
)

# Log results
for task, result in results['results'].items():
    print(f"{task}: {result}")
```

**Custom Task Support:**
```python
# For domain-specific evaluation
class CustomTask:
    def __init__(self, data_path):
        self.data = load_data(data_path)

    def evaluate(self, model, tokenizer):
        # Custom evaluation logic
        pass
```

### 7.5 Logging and Monitoring

**Logging Backends:**
```python
# Option 1: TensorBoard (built-in)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=f'runs/{run_name}')

# Option 2: Weights & Biases (recommended for production)
import wandb
wandb.init(project='llm-training', name=run_name, config=config)

# Option 3: MLflow
import mlflow
mlflow.start_run(run_name=run_name)
```

**Logged Information:**
```python
LoggingConfig:
    # Scalars
    - loss, perplexity (every step)
    - learning_rate, grad_norm (every step)
    - throughput metrics (every 10 steps)
    - validation metrics (every eval)

    # Distributions (every 1000 steps)
    - parameter distributions
    - gradient distributions
    - activation statistics

    # Text samples (every 5000 steps)
    - Model generations on fixed prompts
    - Attention visualizations (optional)

    # System metrics
    - GPU utilization
    - Memory usage
    - Network bandwidth (multi-node)
```

---

## 8. Governance and Safety

### 8.1 Data Governance

**Dataset Documentation:**
```markdown
# Required for each dataset
DataCard:
    - Name and version
    - Source URLs
    - License information
    - Collection methodology
    - Known biases and limitations
    - Intended use cases
    - PII handling procedures
    - Ethical considerations
```

**PII Filtering:**
```python
PIIFilter:
    patterns:
        - Email addresses (regex)
        - Phone numbers (regex + validation)
        - Social Security Numbers
        - Credit card numbers
        - IP addresses (optional)
        - Physical addresses (NER-based)

    action: "redact"  # or "remove_document"
```

### 8.2 Model Card

**Template:**
```markdown
# Model Card: [Model Name]

## Model Details
- Version:
- Date:
- Architecture: Decoder-only transformer
- Parameters:
- Training tokens:
- License:

## Intended Use
- Primary use: Research and education
- Out-of-scope: Production deployment without additional safety measures

## Training Data
- Sources: [List datasets]
- Size: [Token count]
- Languages: English
- Preprocessing: [Describe pipeline]

## Evaluation
- Benchmarks: [Results table]
- Limitations: [Known failure modes]

## Ethical Considerations
- Biases: [Documented biases]
- Risks: [Potential harms]
- Mitigations: [Safety measures]

## Carbon Footprint
- GPU hours:
- Estimated CO2: [Using ML CO2 calculator]
```

### 8.3 Safety Filtering

**Prototype (Rule-Based):**
```python
SafetyFilter:
    # Basic toxicity blocklist
    blocklist: List[str]  # Hate speech, slurs

    # Document-level filtering
    def filter_document(text: str) -> bool:
        text_lower = text.lower()
        for term in blocklist:
            if term in text_lower:
                return True  # Filter out
        return False
```

**Production (Model-Based):**
```python
SafetyFilter:
    # Perspective API or similar
    toxicity_model: ToxicityClassifier
    threshold: float = 0.7

    # Content classifiers
    pii_detector: PIIDetector
    hate_speech_detector: HateSpeechClassifier
```

### 8.4 License Compliance

**Code License:**
- MIT or Apache 2.0 (permissive)

**Data Licenses:**
- Track all source licenses
- Ensure compatibility with model license
- Document in dataset manifest

**Model License:**
- Clear terms of use
- Attribution requirements
- Derivative work permissions

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Objectives:**
- Set up repository structure
- Implement core tokenizer
- Build basic data pipeline

**Tasks:**
```
Week 1:
- [x] Repository structure and environment setup
- [ ] SentencePiece tokenizer implementation
- [ ] Tokenizer training on sample corpus
- [ ] Unit tests for tokenizer

Week 2:
- [ ] Data cleaning pipeline (Stage 1)
- [ ] MinHash deduplication implementation
- [ ] Binary sharding implementation
- [ ] Data loading + batching tests
```

**Deliverables:**
- Tokenizer with 8k vocab
- 1GB preprocessed training data
- Dataloader with unit tests

### Phase 2: Model Architecture (Weeks 3-4)

**Objectives:**
- Implement transformer architecture
- Verify correctness with small-scale tests
- Set up training loop

**Tasks:**
```
Week 3:
- [ ] RMSNorm implementation
- [ ] RoPE implementation
- [ ] Multi-head attention (no FlashAttention yet)
- [ ] SwiGLU FFN
- [ ] TransformerBlock integration
- [ ] Full model assembly

Week 4:
- [ ] Training loop implementation
- [ ] Loss computation and validation
- [ ] Optimizer + scheduler setup
- [ ] Mixed precision integration
- [ ] Gradient clipping
- [ ] Unit tests for all components
```

**Deliverables:**
- Complete transformer model (10M params)
- Training loop with validation
- Checkpoint save/load

### Phase 3: Prototype Training (Weeks 5-6)

**Objectives:**
- Train first prototype model
- Validate training stability
- Implement logging and monitoring

**Tasks:**
```
Week 5:
- [ ] Logging setup (TensorBoard)
- [ ] Training monitoring dashboard
- [ ] Run first training (5M params, 100M tokens)
- [ ] Debug convergence issues
- [ ] Hyperparameter tuning

Week 6:
- [ ] Train larger prototype (50M params, 500M tokens)
- [ ] Validation metrics implementation
- [ ] Basic evaluation suite
- [ ] Sample generation testing
- [ ] Document results
```

**Deliverables:**
- Trained 50M parameter model
- Training curves and metrics
- Sample generations
- Initial model card

### Phase 4: Scaling Infrastructure (Weeks 7-8)

**Objectives:**
- Implement distributed training support
- Optimize data pipeline for scale
- Prepare for cluster deployment

**Tasks:**
```
Week 7:
- [ ] FSDP integration
- [ ] Distributed data loading
- [ ] Multi-GPU testing (if available)
- [ ] Activation checkpointing
- [ ] Distributed checkpoint format

Week 8:
- [ ] FlashAttention integration
- [ ] GQA implementation
- [ ] Data pipeline optimization (sequence packing)
- [ ] Throughput benchmarking
- [ ] Memory optimization
```

**Deliverables:**
- FSDP-enabled training code
- Optimized data pipeline (2x throughput)
- Scalability tests up to available GPUs

### Phase 5: Evaluation and Documentation (Week 9)

**Objectives:**
- Comprehensive evaluation
- Documentation completion
- Production readiness

**Tasks:**
```
Week 9:
- [ ] lm-evaluation-harness integration
- [ ] Benchmark suite execution
- [ ] Model card completion
- [ ] Dataset documentation
- [ ] Code documentation (docstrings, README)
- [ ] Reproducibility testing
```

**Deliverables:**
- Benchmark results on standard tasks
- Complete model card
- Full documentation
- Reproducible training script

### Phase 6: Production Scaling (Week 10+)

**Objectives:**
- Scale to 1B+ parameters
- Multi-node training
- Advanced optimizations

**Tasks:**
```
Week 10+:
- [ ] Scale to 1B parameters
- [ ] Multi-node FSDP testing
- [ ] Long context (4k-8k) training
- [ ] Advanced evaluation
- [ ] Instruction tuning infrastructure (future)
```

**Deliverables:**
- 1B+ parameter model
- Multi-node training documentation
- Production-ready codebase

---

## 10. Technical Stack

### 10.1 Core Dependencies

**Python Environment:**
```
python >= 3.10
```

**Essential Libraries:**
```
torch >= 2.1.0              # PyTorch with FSDP
numpy >= 1.24.0
sentencepiece >= 0.1.99     # Tokenizer
datasets >= 2.14.0          # HuggingFace datasets (optional)
```

**Data Processing:**
```
datasketch >= 1.6.0         # MinHash deduplication
pyarrow >= 13.0.0           # Arrow format (optional)
beautifulsoup4 >= 4.12.0    # HTML cleaning
ftfy >= 6.1.0               # Text normalization
```

**Training & Optimization:**
```
transformers >= 4.35.0      # HF utilities (optional)
accelerate >= 0.24.0        # Multi-GPU training helpers
flash-attn >= 2.3.0         # FlashAttention (optional, requires compilation)
apex                        # NVIDIA optimizations (optional)
```

**Distributed Training:**
```
# Built-in PyTorch distributed
torch.distributed

# OR DeepSpeed
deepspeed >= 0.12.0
```

**Evaluation:**
```
lm-eval >= 0.4.0            # EleutherAI evaluation harness
```

**Logging:**
```
tensorboard >= 2.14.0       # Built-in option
wandb >= 0.15.0             # Weights & Biases (optional)
```

### 10.2 Optional Performance Libraries

**CUDA Optimizations:**
```
triton >= 2.1.0             # Custom CUDA kernels
xformers >= 0.0.22          # Memory-efficient attention
```

**Data Loading:**
```
webdataset >= 0.2.0         # Efficient data sharding
nvidia-dali                 # GPU data loading (advanced)
```

### 10.3 Development Tools

**Code Quality:**
```
pytest >= 7.4.0
black >= 23.0.0
isort >= 5.12.0
mypy >= 1.5.0
```

**Version Control:**
```
git-lfs                     # For model checkpoints
dvc                         # Data version control (optional)
```

---

## 11. Directory Structure

```
LLM/
├── README.md
├── IMPLEMENTATION_PLAN.md (this file)
├── PRD.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── model/
│   │   ├── 5M.yaml
│   │   ├── 50M.yaml
│   │   ├── 1B.yaml
│   │   └── 7B.yaml
│   ├── tokenizer/
│   │   ├── bpe_8k.yaml
│   │   └── bpe_32k.yaml
│   ├── training/
│   │   ├── prototype.yaml
│   │   └── production.yaml
│   └── distributed/
│       ├── fsdp.yaml
│       └── deepspeed_zero3.json
│
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── transformer.py
│   │   ├── attention.py
│   │   ├── ffn.py
│   │   ├── normalization.py
│   │   ├── rope.py
│   │   └── embeddings.py
│   │
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── tokenizer.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing/
│   │   │   ├── cleaner.py
│   │   │   ├── deduplication.py
│   │   │   ├── filters.py
│   │   │   └── normalizer.py
│   │   ├── sharding.py
│   │   ├── dataset.py
│   │   └── collator.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   └── distributed.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── benchmarks.py
│   │   └── generation.py
│   │
│   ├── checkpoint/
│   │   ├── __init__.py
│   │   ├── saver.py
│   │   └── loader.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── metrics.py
│       └── reproducibility.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_tokenizer.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
│
├── tests/
│   ├── test_model.py
│   ├── test_tokenizer.py
│   ├── test_data.py
│   └── test_training.py
│
├── data/
│   ├── raw/                # Raw downloaded data
│   ├── processed/          # Cleaned data
│   ├── shards/             # Binary shards
│   │   ├── train/
│   │   └── val/
│   └── tokenizer/          # Tokenizer artifacts
│
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
├── results/                # Evaluation results
└── docs/                   # Additional documentation
    ├── architecture.md
    ├── data_pipeline.md
    └── training_guide.md
```

---

## 12. Key Design Decisions Summary

### 12.1 Architectural Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Architecture** | Decoder-only transformer | Standard for generative LLMs, simpler than encoder-decoder |
| **Normalization** | RMSNorm (pre-norm) | 10-15% faster than LayerNorm, modern standard |
| **Activation** | SwiGLU | Better performance than GELU/ReLU in modern LLMs |
| **Position Encoding** | RoPE | Superior length generalization, no learned params |
| **Attention** | MHA → GQA | MHA for prototype, GQA for production KV cache efficiency |
| **Precision** | BF16/FP16 | 2x speedup, minimal quality loss |

### 12.2 Training Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Optimizer** | AdamW | Industry standard, stable convergence |
| **Schedule** | Warmup + Cosine Decay | Proven effective for LLM training |
| **Gradient Clipping** | Global norm = 1.0 | Stability without limiting learning |
| **Batch Size** | 2-4M tokens/step (production) | Following GPT-3/Llama best practices |
| **Sequence Length** | 512 (prototype), 4096 (production) | Balance compute and context |

### 12.3 Infrastructure Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Distributed Training** | FSDP (primary) | Native PyTorch, simpler than DeepSpeed for <70B |
| **Data Format** | Binary shards | Fastest I/O, simple implementation |
| **Checkpoint** | Distributed (sharded) | Load on different world sizes, faster save/load |
| **Logging** | TensorBoard/W&B | Standard tools, good ecosystem support |

### 12.4 Data Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Tokenizer** | SentencePiece BPE | Language-agnostic, production-ready, used by Llama |
| **Deduplication** | MinHash + LSH | Scalable, proven effective (Llama 3, FineWeb) |
| **Vocab Size** | 8k (proto), 32k-100k (prod) | 8k for fast iteration, 32k+ for efficiency |

---

## 13. Performance Targets

### 13.1 Prototype (M1 GPU)

```
Hardware: Single M1/A100 GPU
Model Size: 50M parameters
Sequence Length: 512 tokens
Batch Size: 16 samples

Targets:
    - Throughput: 10,000-50,000 tokens/sec
    - Memory: <12GB GPU memory
    - Training time: 1-2 days for 500M tokens
    - Validation perplexity: <30 on WikiText-2
```

### 13.2 Production (Cluster)

```
Hardware: 64x A100 (8 nodes)
Model Size: 7B parameters
Sequence Length: 4096 tokens
Batch Size: 512 samples (8 per GPU)

Targets:
    - Throughput: 5-10M tokens/sec (cluster aggregate)
    - MFU (Model FLOPs Utilization): 40-60%
    - Training time: 2-3 weeks for 2T tokens
    - Validation perplexity: <10 on WikiText-103
    - HellaSwag accuracy: >70%
```

---

## 14. Risk Mitigation

### 14.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Training instability** | High | Gradient clipping, warmup, small initial LR, FP32 master weights |
| **Memory OOM** | High | Gradient checkpointing, mixed precision, smaller micro-batches |
| **Slow data loading** | Medium | Memory-mapped files, multi-worker prefetch, pre-tokenization |
| **Distributed failures** | Medium | Automatic checkpoint recovery, fault-tolerant data loading |
| **Poor convergence** | Medium | Extensive validation, learning rate tuning, architecture verification |

### 14.2 Data Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Low-quality data** | High | Multi-stage filtering, quality scoring, manual inspection |
| **Insufficient deduplication** | Medium | MinHash with low threshold, URL-level dedup |
| **PII leakage** | High | Regex + NER-based filtering, manual audits |
| **License violations** | High | Strict license tracking, source documentation |
| **Bias amplification** | Medium | Diverse data sources, bias documentation, safety filtering |

### 14.3 Infrastructure Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Hardware failures** | High | Frequent checkpointing, multi-replica storage |
| **Network bottlenecks** | Medium | InfiniBand, gradient compression, overlap comm/compute |
| **Storage limits** | Medium | Data compression, cleanup old checkpoints, cloud storage |
| **Cost overruns** | Medium | Cost monitoring, spot instances, training efficiency optimization |

---

## 15. Success Criteria

### 15.1 Prototype Success

- [ ] Tokenizer trains and produces reversible encoding
- [ ] Model architecture verified with unit tests
- [ ] Training loop converges on toy dataset
- [ ] 50M model trains end-to-end on M1 GPU
- [ ] Validation perplexity decreases monotonically
- [ ] Checkpoint save/load works across restarts
- [ ] Generates coherent text (>5 words)
- [ ] Code runs in CPU-only mode

### 15.2 Production Readiness

- [ ] Distributed training verified on multi-GPU
- [ ] Data pipeline scales to 100GB+ corpus
- [ ] 1B model trains without OOM
- [ ] Throughput >1M tokens/sec on 8 GPUs
- [ ] Benchmark results within 10% of comparable open models
- [ ] Checkpoint compatible with different world sizes
- [ ] Complete documentation and model card
- [ ] Reproducible training from scratch

### 15.3 Research Viability

- [ ] Architecture modifications require no rewrites
- [ ] Easy to swap components (attention, FFN, norm)
- [ ] Clear logging and debugging tools
- [ ] Ablation studies feasible
- [ ] Supports experimentation (new optimizers, schedules)

---

## 16. References and Research Sources

### 16.1 Architecture Papers

1. **DeepSeek-R1** (2025): Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
   - Multi-head Latent Attention (MLA)
   - Mixture of Experts (MoE)
   - Group Relative Policy Optimization (GRPO)
   - [arXiv:2501.12948](https://arxiv.org/pdf/2501.12948)

2. **Llama 3.1** (2024): Meta's open-source models up to 405B parameters
   - Grouped Query Attention (GQA)
   - Extended RoPE (500k base frequency)
   - 128K context window
   - [Meta AI Blog](https://ai.meta.com/blog/meta-llama-3-1/)

3. **Mistral/Mixtral** (2024): Mixture of Experts architecture
   - Sparse MoE with 8 experts
   - Sliding Window Attention
   - [Mistral AI](https://mistral.ai/)

4. **RoFormer** (2021): Rotary Position Embeddings
   - [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

5. **FlashAttention-3** (2024): Fast and Accurate Attention
   - [Tri Dao's Blog](https://tridao.me/blog/2024/flash3/)

### 16.2 Training Infrastructure

6. **PyTorch FSDP**: Fully Sharded Data Parallel
   - [PyTorch Docs](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)

7. **DeepSpeed ZeRO**: Zero Redundancy Optimizer
   - [DeepSpeed Docs](https://www.deepspeed.ai/training/)

8. **Megatron-LM**: NVIDIA's large-scale training framework
   - [NVIDIA Megatron](https://github.com/NVIDIA/Megatron-LM)

### 16.3 Data Processing

9. **MinHash LSH for Deduplication**
   - Used in Llama 3, FineWeb
   - [Milvus Blog](https://milvus.io/blog/minhash-lsh-in-milvus-the-secret-weapon-for-fighting-duplicates-in-llm-training-data.md)

10. **NeMo Curator**: NVIDIA's data curation toolkit
    - [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/)

11. **FED**: Fast and Efficient Dataset Deduplication
    - [arXiv:2501.01046](https://arxiv.org/html/2501.01046v2)

### 16.4 Training Stability

12. **Methods of Improving LLM Training Stability** (2024)
    - [arXiv:2410.16682](https://arxiv.org/html/2410.16682v1)

13. **Stabilizing LLM Training: Techniques and Insights**
    - [Rohan Paul Blog](https://www.rohan-paul.com/p/stabilizing-llm-training-techniques)

### 16.5 Tokenization

14. **SentencePiece**: Google's tokenizer
    - [GitHub](https://github.com/google/sentencepiece)
    - [Guide](https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15/)

### 16.6 Checkpointing

15. **Safetensors**: Safe tensor serialization format
    - [HuggingFace + PyTorch](https://pytorch.org/blog/huggingface-safetensors-support-in-pytorch-distributed-checkpointing/)

---

## 17. Next Steps

### Immediate Actions (This Week)

1. **Set up development environment**
   - Create virtual environment
   - Install dependencies
   - Initialize Git repository
   - Set up pre-commit hooks

2. **Create project structure**
   - Generate directory tree
   - Create placeholder files
   - Write initial README

3. **Begin tokenizer implementation**
   - Install SentencePiece
   - Prepare sample corpus (100MB)
   - Train initial 8k vocab tokenizer
   - Write tokenizer wrapper class

4. **Start data pipeline**
   - Download Wikipedia sample (1GB)
   - Implement basic cleaning
   - Test binary serialization

### Short-term Goals (Next 2 Weeks)

1. Complete data preprocessing pipeline
2. Implement full transformer architecture
3. Set up training loop with validation
4. Train first 5M parameter model
5. Verify checkpoint save/load

### Medium-term Goals (Next 2 Months)

1. Train 50M parameter prototype to completion
2. Implement distributed training support
3. Scale to 1B parameters
4. Run comprehensive evaluation suite
5. Complete all documentation

---

## Appendix A: Configuration Examples

### A.1 Prototype Model Config (5M)

```yaml
model:
  vocab_size: 8000
  n_layers: 6
  d_model: 512
  n_heads: 8
  head_dim: 64
  ffn_mult: 4.0
  max_seq_len: 512
  dropout: 0.1
  tie_embeddings: true
  rope_base: 10000.0

training:
  micro_batch_size: 16
  gradient_accumulation_steps: 1
  seq_len: 512
  max_steps: 50000

  optimizer:
    type: adamw
    lr: 6.0e-4
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 0.1

  scheduler:
    warmup_steps: 500
    min_lr_ratio: 0.1

  precision: fp16
  grad_clip: 1.0

  logging:
    log_every: 10
    eval_every: 500
    save_every: 1000
```

### A.2 Production Model Config (7B)

```yaml
model:
  vocab_size: 32000
  n_layers: 32
  d_model: 4096
  n_heads: 32
  n_kv_heads: 8  # GQA
  head_dim: 128
  ffn_mult: 3.5  # ~14336 FFN dim
  max_seq_len: 4096
  dropout: 0.0
  tie_embeddings: false
  rope_base: 10000.0

training:
  micro_batch_size: 8
  gradient_accumulation_steps: 1
  seq_len: 4096
  max_steps: 500000  # ~2T tokens

  optimizer:
    type: adamw
    lr: 3.0e-4
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 0.1

  scheduler:
    warmup_steps: 2000
    min_lr_ratio: 0.1

  precision: bf16
  grad_clip: 1.0

  distributed:
    backend: fsdp
    sharding_strategy: full_shard
    activation_checkpointing: true

  logging:
    log_every: 1
    eval_every: 2000
    save_every: 5000
```

---

## Appendix B: Estimated Compute Requirements

### B.1 Training Compute (FLOPs)

```
Formula (per token, decoder-only):
    FLOPs ≈ 6 * N

    Where N = number of parameters

Examples:
    50M model: 300M FLOPs/token
    1B model: 6T FLOPs/token
    7B model: 42T FLOPs/token
```

### B.2 GPU Hours Estimate

```
Prototype (50M, 500M tokens):
    Total FLOPs: 1.5e17
    A100 (312 TFLOPS @ BF16): ~8 hours @ 50% MFU

Production (7B, 2T tokens):
    Total FLOPs: 8.4e22
    64x A100: ~700 GPU hours = ~11 hours walltime @ 50% MFU

More realistic (40% MFU): ~14 hours walltime
```

### B.3 Memory Estimates

```
Model Parameters:
    Formula: Memory ≈ N * (4 bytes for FP32, 2 bytes for FP16)

    50M model: ~100MB (FP16)
    7B model: ~14GB (FP16)

Optimizer State (AdamW):
    ~2x model size (momentum + variance)

    50M: ~200MB
    7B: ~28GB

Gradients:
    Same as model parameters

Activations (per sample):
    Formula (rough): 2 * L * d_model * seq_len * batch

    7B model, seq_len=4096, batch=8:
        ~2 * 32 * 4096 * 4096 * 8 * 2 bytes = ~34GB

Total (single GPU, no FSDP):
    7B: 14GB + 28GB + 14GB + 34GB = 90GB (doesn't fit in A100!)

With FSDP (64 GPUs):
    Per GPU: ~90GB / 64 + overhead ≈ 2-3GB (fits easily!)
```

---

**END OF IMPLEMENTATION PLAN**

This document will be updated as the implementation progresses and new insights are gained.
