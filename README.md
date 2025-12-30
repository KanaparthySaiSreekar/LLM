# Scalable Text-Only Foundational Model

A decoder-only transformer language model built from scratch with modern architectural choices, designed to scale from 10M to multi-billion parameters without code rewrites.

## Features

- **Modern Architecture**: RMSNorm, SwiGLU, RoPE, Grouped Query Attention (GQA)
- **Scalable Design**: Same code runs on single GPU (M1) and distributed clusters
- **Clean Implementation**: Modular, well-documented, production-ready code
- **Full Pipeline**: Data preprocessing, tokenization, training, evaluation

## Architecture

- **Type**: Decoder-only Transformer (GPT-style)
- **Normalization**: RMSNorm (pre-norm)
- **Activation**: SwiGLU in feed-forward layers
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Attention**: Multi-Head Attention with optional Grouped Query Attention
- **Precision**: FP16/BF16 mixed precision training

Based on architectural patterns from:
- Llama 3/3.1 (Meta)
- DeepSeek R1
- Mistral/Mixtral
- Other top open-source models

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Test Model Architecture

```bash
python scripts/test_model.py
```

This will:
- Initialize models of different sizes (5M, 50M, 125M)
- Run forward passes
- Test text generation
- Verify all components work correctly

### Train a Tokenizer

```bash
# Prepare training data (plain text files)
cat corpus/*.txt > data/raw/training_data.txt

# Train tokenizer
python scripts/train_tokenizer.py \
    --input data/raw/training_data.txt \
    --model-prefix data/tokenizer/tokenizer \
    --vocab-size 8000 \
    --model-type bpe
```

## Implementation Status

### ✅ Completed
- [x] Model architecture (Transformer, Attention, FFN, RoPE, RMSNorm)
- [x] Model configurations (5M to 7B)
- [x] Tokenizer wrapper (SentencePiece)
- [x] Tokenizer training script
- [x] Model testing script
- [x] Directory structure
- [x] Requirements and documentation

### 🚧 In Progress
- [ ] Data preprocessing pipeline
- [ ] Training loop
- [ ] Distributed training support (FSDP)
- [ ] Evaluation suite
- [ ] Checkpoint management

### 📋 TODO
- [ ] Data cleaning and deduplication (MinHash)
- [ ] Binary data sharding
- [ ] Training configuration files
- [ ] Logging and monitoring (TensorBoard/W&B)
- [ ] Unit tests
- [ ] Benchmarking suite (HellaSwag, LAMBADA, etc.)

## References

- Implementation Plan: `IMPLEMENTATION_PLAN.md`
- Llama 3.1: https://ai.meta.com/blog/meta-llama-3-1/
- DeepSeek R1: https://arxiv.org/pdf/2501.12948

## License

MIT License