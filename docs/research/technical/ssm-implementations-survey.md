---
question: What practical SSM implementations, production deployments, and development patterns exist that could inform building a small local SSM sidecar for Signet's predictive memory scorer?
date: 2026-03-20
informed_by:
  - Mamba paper (Gu & Dao, 2023)
  - S4 paper (Gu et al., 2021)
  - Mamba-3 ICLR 2026 paper
  - Annotated S4 (Sasha Rush)
  - Multiple GitHub repositories and production systems surveyed below
---

# State-Space Model (SSM) Implementations Survey

Comprehensive survey of practical SSM implementations, production
deployments, training frameworks, and deployment patterns — evaluated
for relevance to building a small (<10M parameter) local SSM sidecar
for agent memory relevance prediction.

---

## 1. Reference Implementations

### 1.1 Official Mamba (Python/CUDA)

- **Repo**: https://github.com/state-spaces/mamba
- **Language**: Python + custom CUDA kernels (Triton)
- **Models**: mamba-130m, mamba-370m, mamba-790m, mamba-1.4b, mamba-2.8b,
  plus Mamba-2 variants at same sizes
- **Key insight**: The reference implementation is heavily optimized with
  hardware-aware CUDA kernels (kernel fusion, parallel scan,
  recomputation). Training uses convolutional representation
  (parallelizable), inference uses recurrent representation (O(1)
  memory per step, linear time).
- **Linux only** for the official CUDA implementation.
- **Relevance**: Source of pretrained weights (safetensors format).
  The 130m model is a natural starting point for distillation or
  fine-tuning, though its language modeling weights would need
  task-specific adaptation.

### 1.2 Official S4 (JAX/PyTorch)

- **Repo**: https://github.com/state-spaces/s4
- **Language**: Python (JAX + PyTorch)
- **Key insight**: S4 is the foundational structured SSM. The repo
  includes S4, S4D (diagonal, simplified), HiPPO, LSSL, DSS, and
  S4ND. The `s4d.py` file is a minimal pedagogical implementation.
  The S4D model with 200k parameters reaches 88% accuracy on
  sequential CIFAR — demonstrating that tiny SSMs can be effective.
- **Annotated S4**: https://srush.github.io/annotated-s4/ — A JAX
  reimplementation in ~200 LOC that clearly shows the core algorithm.
  Uses vmap, scan, jit. The best educational resource for
  understanding SSM internals.
- **Relevance**: S4D's minimal architecture proves sub-1M parameter
  SSMs can learn meaningful sequence patterns. Good reference for
  our custom architecture.

### 1.3 Minimal/Educational Implementations

**mamba-minimal** (Python)
- **Repo**: https://github.com/johnma2006/mamba-minimal
- Single-file PyTorch implementation. Prioritizes readability.
  Numerically equivalent to official implementation for forward
  and backward passes. Can load pretrained `mamba-370m` weights.
- **Relevance**: Best starting point for understanding the selective
  SSM algorithm before implementing in Rust.

**mamba.py** (Python/MLX)
- **Repo**: https://github.com/alxndrTL/mamba.py
- Pure PyTorch + MLX. Includes RNN-formulation inference mode for
  fast token-by-token generation. Parallel scan in `pscan.py`,
  core layer in `mamba.py`, LM wrapper in `lm.py`. ~2x slower
  than official CUDA but clear code. Supports Jamba (Mamba +
  attention) and Vision Mamba.
- **Relevance**: The RNN inference path is exactly what we need for
  our streaming/online use case. The MLX backend shows SSM
  inference works well on Apple Silicon without CUDA.

**mamba3-minimal** (Python)
- **Repo**: https://github.com/VikramKarLex/mamba3-minimal
- ~800 lines of pure PyTorch implementing Mamba-3 (ICLR 2026).
  Dependencies: torch, einops only. Endorsed by Albert Gu (Mamba
  co-author) as "a great supplement to the official code."
- **Key Mamba-3 improvements over Mamba-2**:
  - Trapezoidal discretization (2nd-order accurate vs Euler)
  - Complex-valued SSM with RoPE (enables state tracking)
  - MIMO formulation (higher arithmetic intensity at decode)
  - No short convolution needed (trapezoidal rule replaces Conv1d)
  - Achieves same perplexity with half the state size
- **Relevance**: Mamba-3 is inference-first by design. Half state
  size = half memory for our sidecar. MIMO improves decode
  throughput. This is the architecture to target for new work.

**mamba2-minimal** (Python)
- **Repo**: https://github.com/tommyip/mamba2-minimal
- Single-file Mamba-2 implementation. The SSD (State Space Duality)
  algorithm is ~25 lines of code and mostly matrix multiplications.
- **Relevance**: Shows the mathematical simplicity of the core
  algorithm — feasible to port to Rust.

### 1.4 Rust SSM Implementations (Critical for Sidecar)

**mamba.rs** — Pure Rust Mamba
- **Repo**: https://github.com/LaurentMazare/mamba.rs
- Pure Rust, minimal dependencies. Memory-mapped weight loading.
  Parallel matmul via rayon. Supports 130m through 2.8b models.
- **Limitation**: Matrix multiplication is "not cache friendly"
  per author. TODO: SIMD, more parallelism, quantization.
- **License**: Apache 2.0 / MIT dual
- **Relevance**: HIGH. Proves pure Rust Mamba inference is viable.
  The simplicity (no framework dependency) makes it ideal for
  embedding in our Rust sidecar. Needs optimization but the
  architecture is right.

**mamba-ssm (Candle-based Rust)**
- **Repo**: https://github.com/flawedmatrix/mamba-ssm
- Built on Hugging Face Candle. Targets Apple Silicon (Accelerate).
  Loads safetensors format. CPU-first design.
- **Performance**: M3 Max, FP32: ~6.5 tok/s generation, ~24.68
  tok/s prompt processing, ~1.6s model load.
- **Features**: BF16 (CUDA only). Missing: FP16, quantization.
- **License**: MIT
- **Relevance**: HIGH. Shows Candle + Mamba works in Rust. The
  performance numbers on M3 Max are a useful baseline. For our
  sub-10M model, expect much faster inference.

**web-rwkv** — Pure Rust/WebGPU RWKV
- **Repo**: https://github.com/cryscan/web-rwkv
- Pure Rust, WebGPU backend. Supports RWKV v4 through v7.
  Int8 and Float4 quantization. Batched inference. Async runtime
  via tokio. Supports Vulkan, DirectX 12, OpenGL, and WASM.
- **Relevance**: MEDIUM-HIGH. RWKV-7 is a strong SSM alternative.
  The Rust/WebGPU architecture is production-quality. Quantization
  support is ahead of Rust Mamba implementations. However, RWKV
  has a different ecosystem than Mamba/S4.

**oxidizr (ml-rust ecosystem)**
- **Repo**: https://github.com/ml-rust (organization)
- Production-grade LLM training framework in Rust, built on Candle.
  Supports Mamba/Mamba2/Mamba3, MLA, MoE, hybrid architectures.
  CUDA acceleration, multi-GPU via NCCL. Companion inference
  server `blazr` with OpenAI-compatible API.
- **Status**: "Stable for Transformer, Mamba2, Mamba3, and hybrid."
- **Relevance**: HIGH. This is the most complete Rust SSM training
  + inference stack. If we train in Rust, this is the framework.
  blazr could serve our sidecar's inference endpoint.

### 1.5 C/C++ SSM Implementations

**llama.cpp Mamba support**
- **Repo**: https://github.com/ggml-org/llama.cpp (PR #5328, #9126)
- Full Mamba-1 and Mamba-2 support in llama.cpp via GGUF format.
  Custom `ggml_ssm_scan` and `ggml_ssm_conv` operators.
- **Performance**: 20-30% speedup from operation fusion. Constant
  memory regardless of context length. Each KV slot ~23.75 MiB
  for Mamba 3B.
- **Quantization**: SSM-specific weights kept at f32 for stability;
  linear projections use k-quants. Q4_K_M = 5.76 bits/weight for
  2.8B model.
- **Relevance**: HIGH. GGUF format is the most portable model
  format for local inference. If we use GGUF for model distribution,
  users get llama.cpp ecosystem compatibility. However, for our
  tiny custom model, building a dedicated Rust inference engine
  may be simpler than integrating ggml.

### 1.6 Portable Model Formats

**ONNX Export Status**
- ONNX export of Mamba is **not straightforward**. Key blockers:
  - MambaCache is not a known ONNX type
  - Triton dependency prevents CPU/ONNX conversion
  - If-else control flow in cached inference breaks export
  - Works when `use_cache=False` but changes computation path
- Active community effort (issues on pytorch, transformers, mamba
  repos) but no clean solution as of March 2026.
- **Relevance**: ONNX is NOT a viable path for Mamba model
  distribution right now. Safetensors + custom Rust inference
  or GGUF + llama.cpp are better options.

**ort (ONNX Runtime for Rust)**
- **Crate**: https://crates.io/crates/ort
- Rust wrapper around ONNX Runtime. 9x faster than naive setups.
  Supports CUDA, OpenVINO, QNN, CANN. Used by Google Magika and
  Bloop.
- **Relevance**: MEDIUM. If we could export our model to ONNX, ort
  would be the ideal Rust inference crate. But ONNX export
  limitations for SSMs make this path uncertain. Worth revisiting
  if ONNX SSM support improves.

---

## 2. Production SSM Deployments

### 2.1 Companies Using SSMs in Production

**AI21 Labs — Jamba**
- First production-grade hybrid SSM-Transformer at scale.
  Architecture: 1 attention layer per 8 total layers (7 Mamba + 1
  attention). 52B total params, 12B active (MoE). 256K context.
- Deployed on: Google Cloud Vertex AI, Azure, NVIDIA NIM.
  Jamba 1.5: 398B total, 94B active. 2.5x faster inference on
  long contexts vs similar-sized transformers.
- **Key insight**: Hybrid architectures dominate production because
  pure SSMs struggle with precise associative recall. The ~10:1
  SSM:attention ratio is a production-proven sweet spot.

**NVIDIA — Nemotron-H / Nemotron 3 Super**
- 120.6B hybrid Mamba-2/Transformer with LatentMoE. Interleaves
  Mamba-2 for sequence processing, attention for recall precision.
  4x improved memory/compute efficiency. 1M token context window.
- "Most capable single-GPU agentic AI deployment in open weights"
  as of March 2026. Optimized TensorRT LLM deployment.
- **Key insight**: NVIDIA is all-in on hybrid SSM-Transformer for
  agentic AI workloads.

**IBM — Bamba / Granite 4.0**
- Bamba-9B: hybrid SSM + attention. 2x throughput vs similar
  transformers, matching accuracy. Trained on 3T tokens, compressed
  via quantization from 18GB to 9GB. Matches LLaMA-3.1-8B with
  7x less data. Apache 2.0 license.
- Being incorporated into Granite 4.0 enterprise models.

**Cartesia AI — Sonic / Rene / Llamba**
- SSM-native company. Built custom SSM inference stack for Sonic
  (voice model, low-latency API). ~10:1 SSM:attention ratio in
  hybrid stacks.
- Rene-v0.1 (1.3B): hybrid Mamba-2 + MLP + sliding window
  attention. First SLM based on recurrent architecture matching
  similarly-sized SotA SLMs. Outperforms OpenELM 1.1B and
  RecurrentGemma 2B.
- Llamba family (1B, 3B, 8B): Mamba-2 layers distilled from
  Llama-3.1 via MOHAWK. Available in PyTorch and MLX with
  quantization. Competitive with transformers despite fewer
  training tokens.
- Open-sourced `edge` library: PyTorch, Metal, MLX backends.
  Custom Metal kernels for Mamba-2 (laptop + mobile).
- **Key insight**: Distillation from transformers to SSMs is a
  proven shortcut to competitive SSM models.

**TII — Falcon Mamba 7B**
- First open-source pure SSM language model at scale. No attention
  layers. Trained on ~5500GT. Outperforms LLaMA 3.1 8B and
  Mistral 7B on some benchmarks. Constant memory regardless of
  sequence length.
- **Key insight**: Pure SSMs ARE competitive at 7B scale. No
  attention needed for many tasks.

**BrainChip — TENN**
- 1B parameter model, 24 SSM layers. Runs on read-only flash
  memory, under 0.5W, <100ms inference. Designed for dashcams,
  medical devices, security cameras.
- **Key insight**: SSMs are uniquely suited to extreme edge
  deployment. Sub-watt inference is achievable.

**Together AI — Mamba-3**
- Co-developed Mamba-3 (ICLR 2026). Open-sourced kernels
  (Triton + TileLang + CuTe DSL). SISO variant 7x faster than
  Llama-3.2-1B at 16K sequence length. MIMO doubles inference
  throughput for same hardware.

### 2.2 SSMs in Recommendation Systems

**SS4Rec** (2025)
- Hybrid SSM for sequential recommendation. Time-aware SSM layer
  handles irregular interaction intervals; relation-aware SSM
  captures sequential patterns. Tested on 5 benchmark datasets.
- **Relevance**: DIRECT. This is SSMs predicting "what's relevant
  next" from a sequence of user interactions — very close to our
  memory relevance scoring use case.

**SSD4Rec** (2024)
- Bidirectional Structured State-Space Duality for user behavior
  modeling. 154% faster than attention-based methods, 66% faster
  than prior SSM methods.
- **Key insight**: SSMs naturally handle variable-length user
  behavior sequences with irregular timestamps.

**TiM4Rec** (2024)
- Uses time-difference in user interactions to capture interest
  shifts. Leverages SSM's continuous-time formulation.

### 2.3 SSMs in Time Series Forecasting

**Time-SSM** (2024)
- Unifying SSM framework for time series forecasting. Shows SSMs
  naturally model continuous systems sampled at specific
  frequencies.

**SpikySpace** (2026)
- Fully spiking SSM for edge deployment. Targets urban traffic,
  industrial monitoring, on-device sensing. Practical path to
  neuromorphic time series forecasting.

### 2.4 SSMs in Dynamic Graph / Knowledge Graph

**DyGMamba** (2024)
- Dual-SSM for continuous-time dynamic graphs. Node-level SSM
  encodes interaction histories; time-level SSM learns temporal
  patterns and dynamically weights neighbor importance.
  State-of-the-art for temporal link prediction on Wikipedia,
  Reddit, MOOC, LastFM, Enron, SocialEvo, UCI datasets.
- **Relevance**: DIRECT. Our knowledge graph has temporal dynamics
  (entity mentions over time, relationship evolution). DyGMamba's
  approach of using SSMs to weight historical interactions by
  temporal pattern is directly applicable to our retention/decay
  scoring.

---

## 3. Training Frameworks and Tools

### 3.1 Training SSMs from Scratch

**PyTorch (Official)**
- Use `MambaLMHeadModel` from state-spaces/mamba repo. Pretrained
  weights available on HuggingFace: mamba-130m through mamba-2.8b.
- **Critical**: SSMs are sensitive to numerical precision in
  recurrent dynamics. Use fp32 for parameters (AMP with fp32
  master weights). The state matrix A is particularly sensitive
  to quantization.
- Fine-tuning with PEFT (LoRA) is supported via HuggingFace
  transformers integration.

**oxidizr (Rust)**
- Full training framework in Rust on Candle. Supports Mamba2/3,
  hybrid architectures. Config-driven training loop with
  checkpointing. CPU training supported (no GPU required).
- **Relevance**: Could train our custom model entirely in Rust.

**JAX (S4/HiPPO)**
- The Annotated S4 shows S4 training in ~200 LOC of JAX. The S4
  repository uses PyTorch-Lightning + Hydra for experiments.
- S4D with 200k params reaches 88% on sequential CIFAR —
  proving sub-1M SSMs can learn.

### 3.2 Converting to Efficient Inference

**Safetensors** (recommended)
- The standard format for SSM weight distribution. Native Rust
  crate (`safetensors` on crates.io) for loading. Memory-mapped
  loading supported. HuggingFace auto-converts uploaded models.

**GGUF (via llama.cpp)**
- Mamba-1 and Mamba-2 supported. Includes quantized variants.
  Portable across platforms. However, adding custom architectures
  to llama.cpp requires C++ development.

**ONNX**
- NOT viable for SSMs currently due to cache/control-flow issues.

### 3.3 Training Pipeline for Our Use Case

Recommended approach for Signet's predictor:

1. **Architecture**: Custom small Mamba-3 or S4D model, <10M params
2. **Training**: PyTorch initially (mature ecosystem, easier
   debugging), then consider Rust training via oxidizr for
   production pipeline
3. **Weight format**: Safetensors for Rust loading
4. **Inference**: Custom Rust engine (mamba.rs-style) or Candle-based
5. **Quantization**: INT8 minimum for A matrix and state; INT4 for
   projection weights (heterogeneous quantization per the edge AI
   research)

---

## 4. SSM Alternatives and Hybrids

### 4.1 Linear Attention Models

**RWKV (v7 "Goose")**
- **Repo**: https://github.com/BlinkDL/RWKV-LM
- RNN with transformer-level performance. Linear time, constant
  space, no KV-cache. RWKV-7 introduces Dynamic State Evolution,
  surpassing both Transformers and RWKV-6.
- Rust inference: web-rwkv (pure WebGPU/Rust, v4-v7 support,
  INT8/Float4 quantization, async tokio runtime).
- **vs Mamba**: Similar theoretical properties (linear time,
  constant memory). RWKV has a stronger community for small
  model deployment (web-rwkv). Mamba has stronger academic
  backing and more architectural variants.

**RetNet**
- Transformer-successor RNN: trains in parallel, infers
  sequentially. Fixed exponential decay for forgetting.
  Similar to RWKV in spirit.

**Flash Linear Attention (fla)**
- **Repo**: https://github.com/fla-org/flash-linear-attention
- Triton kernels for GLA, DeltaNet, RetNet, RWKV, and more.
  Used by HuggingFace transformers for Qwen3.5. Shows linear
  RNN kernels are faster than Flash Attention.
- **Relevance**: The kernel library, not directly usable from
  Rust, but demonstrates the performance ceiling.

### 4.2 Hybrid SSM-Attention Architectures

**Griffin / RecurrentGemma** (Google DeepMind)
- **Repo**: https://github.com/google-deepmind/recurrentgemma
- Alternates gated linear recurrences with local sliding window
  attention. Fixed-size state for efficient long-sequence
  inference.
- **Critical finding**: Research shows "complete functional
  segregation" — retrieval depends EXCLUSIVELY on attention
  layers. SSM layers show NO compensatory retrieval mechanisms.
  This means for our use case (memory retrieval scoring), pure
  SSM should work because we're doing relevance PREDICTION, not
  exact recall.

**Jamba** (AI21) — see Section 2.1

**Nemotron-H** (NVIDIA) — see Section 2.1

**Bamba** (IBM) — see Section 2.1

### 4.3 Implications for Our Architecture

The hybrid research conclusively shows:
- **SSMs excel at**: pattern recognition, temporal modeling,
  sequence understanding, relevance prediction, behavior modeling
- **SSMs struggle with**: exact recall, precise associative
  memory lookup, needle-in-haystack retrieval
- **Our use case** (predicting which memories are relevant to
  inject) is a PREDICTION task, not a RETRIEVAL task. SSMs are
  well-suited. We don't need attention layers.

---

## 5. Specific SSM Variants for Our Use Case

### 5.1 User Behavior Sequence Modeling

SS4Rec, SSD4Rec, and TiM4Rec all demonstrate SSMs modeling user
interaction sequences with irregular timestamps to predict next
relevant items. This maps directly to our problem:
- User interaction = agent session/turn
- Item = memory
- Irregular timestamps = variable time between sessions
- Prediction target = which memories should be injected

### 5.2 Document/Memory Relevance Prediction

**RankMamba** shows Mamba achieving competitive document ranking
performance vs BERT/T5. Key findings:
- mamba-130m matched strong transformer baselines on TREC DL19/DL20
- mamba-370m matched roberta-large performance range
- SSMs CAN model query-document semantic relationships effectively
- Training throughput currently lower than flash attention
  (implementation issue, not architectural)

### 5.3 Temporal Knowledge Graph Reasoning

**DyGMamba** applies dual SSMs to dynamic graph link prediction.
Node-level SSM encodes interaction history, time-level SSM learns
temporal patterns. This directly maps to our knowledge graph's
entity mention patterns and relationship evolution over time.

### 5.4 Online/Streaming Inference

SSMs are INHERENTLY streaming-friendly:
- Recurrent mode: O(1) memory per step, constant time per token
- No KV cache needed (unlike transformers)
- State update is a fixed-size matrix operation
- fastiSSM accelerates general SSM inference via online model
  approximation
- Intel Loihi 2 demonstrates 1000x less energy, 75x lower
  latency vs GPU for token-by-token SSM inference

For our sidecar: each new turn/memory triggers a single state
update + prediction. No reprocessing of history needed.

### 5.5 Sub-10M Parameter Models

Proven approaches for tiny SSMs:
- **S4D**: 200k params, 88% on sequential CIFAR
- **Quantized S4D**: 3,850 to 630,794 params tested. 6x memory
  reduction via heterogeneous quantization with 96% accuracy
  retention (QAT required).
- **Elastic SSM**: Single training, arbitrary runtime truncation
  without retraining.
- **Key stability requirement**: A matrix and internal state need
  >=8-bit precision. Other components tolerate 4-bit.

---

## 6. Deployment Patterns

### 6.1 Inference Efficiency

**SSM vs Transformer latency**:
- Mamba (L4-d64): 0.380ms per sample vs Transformer (L4-H4-d64):
  4.12ms — **10.8x faster** at small scale
- Mamba-3 at 16K: 140.61s vs Llama-3.2-1B: 976.50s — **~7x faster**
- General: 5x higher throughput, constant memory vs growing KV cache

**Memory footprint**:
- Mamba 2.8B: ~5.16GB VRAM (fp16)
- Constant regardless of sequence length (no KV cache)
- For a sub-10M model: expect <50MB fp32, <15MB int8, <8MB int4

### 6.2 Quantization for Edge

- INT8: 4x smaller, widely deployable, preserves accuracy
- INT4: 8x smaller, needs QAT for stability
- Heterogeneous: 4-bit for projections, 8-bit for A/state = 6x
  compression with 96% accuracy
- Critical: A matrix eigenvalues must stay within unit circle.
  Clip state values to [-50, +50] per timestep to prevent
  divergence.

### 6.3 Serving Patterns

For our Rust sidecar:
1. **Embedded inference**: No server overhead. Load safetensors
   weights at startup, run forward pass inline.
2. **Memory-mapped weights**: mamba.rs pattern — mmap the weight
   file, zero-copy loading.
3. **Batched prediction**: When processing multiple memories for
   a query, batch the relevance scoring. web-rwkv shows batched
   Rust inference is viable.
4. **Quantized deployment**: INT8 for the sidecar to minimize
   memory. The 6x compression factor means our model stays
   well under 10MB.

### 6.4 Recommended Architecture for Signet Sidecar

Based on this survey:

```
Architecture: Custom Mamba-3 variant (MIMO, trapezoidal discretization)
Parameters: 2-5M (sweet spot for relevance prediction)
State size: 16-32 (Mamba-3 needs half vs Mamba-2)
Layers: 4-8 Mamba-3 blocks
d_model: 64-128
Quantization: Heterogeneous INT8/INT4
Weight format: Safetensors
Inference: Pure Rust (mamba.rs style) or Candle-based
Training: PyTorch → export safetensors → Rust inference
Expected latency: <1ms per prediction (based on 0.38ms benchmark
  for similar-sized 4-layer model)
Expected memory: <10MB quantized
```

**Training data**: Agent feedback signals (what helped per turn),
accumulated across sessions. Federated base model from community
(anonymized) + local fine-tuning per user.

**Inference mode**: Recurrent (token-by-token state update). Each
session turn updates the hidden state. At query time, the current
state + candidate memory features → relevance score.

---

## 7. Key Repositories Reference

| Project | URL | Language | Relevance |
|---------|-----|----------|-----------|
| state-spaces/mamba | https://github.com/state-spaces/mamba | Python/CUDA | Reference impl, pretrained weights |
| state-spaces/s4 | https://github.com/state-spaces/s4 | Python/JAX | S4D minimal variant, pedagogical |
| mamba.rs | https://github.com/LaurentMazare/mamba.rs | Rust | Pure Rust Mamba, minimal deps |
| mamba-ssm (Candle) | https://github.com/flawedmatrix/mamba-ssm | Rust | Candle-based, Apple Silicon focus |
| web-rwkv | https://github.com/cryscan/web-rwkv | Rust | RWKV v4-v7, WebGPU, quantization |
| oxidizr/blazr | https://github.com/ml-rust | Rust | Full training+inference stack |
| mamba-minimal | https://github.com/johnma2006/mamba-minimal | Python | Best educational implementation |
| mamba3-minimal | https://github.com/VikramKarLex/mamba3-minimal | Python | Mamba-3 reference, ~800 LOC |
| mamba.py | https://github.com/alxndrTL/mamba.py | Python/MLX | RNN inference mode, clear code |
| Annotated S4 | https://srush.github.io/annotated-s4/ | JAX | ~200 LOC, best tutorial |
| llama.cpp | https://github.com/ggml-org/llama.cpp | C/C++ | GGUF Mamba support, quantization |
| cartesia-ai/edge | https://github.com/cartesia-ai/edge | Python/C++/Metal | On-device SSM, Metal kernels |
| flash-linear-attention | https://github.com/fla-org/flash-linear-attention | Python/Triton | Kernel library, benchmarks |
| recurrentgemma | https://github.com/google-deepmind/recurrentgemma | Python | Hybrid SSM-attention reference |
| DyGMamba | https://github.com/ZifengDing/DyGMamba | Python | SSM for temporal graphs |
| ort | https://github.com/pykeio/ort | Rust | ONNX Runtime wrapper (if ONNX improves) |

---

## 8. Decision Matrix for Signet

| Criterion | Mamba-3 (Rust) | RWKV-7 (web-rwkv) | S4D (Custom Rust) | ONNX + ort |
|-----------|---------------|-------------------|-------------------|------------|
| Inference speed | Excellent | Excellent | Excellent | Good |
| Rust maturity | Early (mamba.rs) | Production (web-rwkv) | None (build from scratch) | Production (ort) |
| Training ecosystem | Strong (PyTorch) | Strong (PyTorch) | Moderate (JAX/PyTorch) | N/A (inference only) |
| Quantization | Community (llama.cpp) | Built-in (INT8/F4) | Research (paper above) | Built-in |
| Sub-10M viability | Proven (RankMamba 130m) | Untested at scale | Proven (200k S4D) | Depends on export |
| Model format | Safetensors | Custom → safetensors | Custom | ONNX (broken for SSMs) |
| Community momentum | Very high (ICLR 2026) | High (v7 active) | Low (mature, stable) | N/A for SSMs |
| Relevance to our task | High (selection mechanism) | High (gating) | Medium (fixed params) | Medium |

**Recommendation**: Start with Mamba-3 architecture, train in
PyTorch, export to safetensors, build custom Rust inference engine
based on mamba.rs patterns with Candle tensors. Fall back to
web-rwkv (RWKV-7) if Mamba Rust ecosystem proves too immature.
Keep S4D as the ultra-minimal fallback for proving the concept
with fewest parameters.

---

## Sources

### Official Implementations
- [state-spaces/mamba](https://github.com/state-spaces/mamba)
- [state-spaces/s4](https://github.com/state-spaces/s4)
- [Annotated S4](https://srush.github.io/annotated-s4/)

### Rust Implementations
- [mamba.rs](https://github.com/LaurentMazare/mamba.rs)
- [mamba-ssm (Candle)](https://github.com/flawedmatrix/mamba-ssm)
- [web-rwkv](https://github.com/cryscan/web-rwkv)
- [ml-rust / oxidizr](https://github.com/ml-rust)
- [ort ONNX Runtime](https://github.com/pykeio/ort)
- [Candle](https://github.com/huggingface/candle)
- [Burn](https://burn.dev/)

### Educational / Minimal Implementations
- [mamba-minimal](https://github.com/johnma2006/mamba-minimal)
- [mamba3-minimal](https://github.com/VikramKarLex/mamba3-minimal)
- [mamba2-minimal](https://github.com/tommyip/mamba2-minimal)
- [mamba.py](https://github.com/alxndrTL/mamba.py)

### Production Deployments
- [AI21 Jamba](https://www.ai21.com/blog/announcing-jamba/)
- [IBM Bamba](https://research.ibm.com/blog/bamba-ssm-transformer-model)
- [NVIDIA Nemotron 3 Super](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/)
- [Cartesia Edge](https://github.com/cartesia-ai/edge)
- [Cartesia On-Device](https://cartesia.ai/blog/on-device)
- [Falcon Mamba 7B](https://huggingface.co/blog/falconmamba)
- [BrainChip TENN](https://brainchip.com/temporal-event-based-neural-networks-a-new-approach-to-temporal-processing/)
- [Together AI Mamba-3](https://www.together.ai/blog/mamba-3)

### SSMs for Recommendation / Retrieval
- [SS4Rec](https://arxiv.org/abs/2502.08132)
- [SSD4Rec](https://arxiv.org/html/2409.01192v1)
- [RankMamba](https://arxiv.org/abs/2403.18276)
- [DyGMamba](https://arxiv.org/abs/2408.04713)

### Hybrid Architectures
- [RecurrentGemma / Griffin](https://github.com/google-deepmind/recurrentgemma)
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- [Llamba](https://arxiv.org/html/2502.14458v1)

### Quantization / Edge
- [Quantizing Small-Scale SSMs](https://arxiv.org/html/2506.12480)
- [llama.cpp Mamba PR](https://github.com/ggml-org/llama.cpp/pull/5328)
- [State Space Models for Edge AI (InfoQ)](https://www.infoq.com/news/2025/07/state-space-models-edge-compute/)

### Architecture Papers
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba-3 (ICLR 2026)](https://openreview.net/forum?id=HwCvaJOiCj)
- [Mamba-2 Blog](https://tridao.me/blog/2024/mamba2-part1-model/)
- [Griffin Paper](https://arxiv.org/pdf/2402.19427)
- [SSM Retrieval Segregation](https://arxiv.org/html/2510.19861)
