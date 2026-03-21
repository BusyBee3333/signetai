---
question: Can state-space models replace or augment the predictive memory scorer as a local-first, low-footprint neural architecture for agent memory relevance, retention modeling, and temporal pattern recognition?
date: 2026-03-20
---

# State-Space Models for Agent Memory Systems: Comprehensive Literature Review

This document surveys the state-space model (SSM) landscape with specific
focus on applicability to Signet's local-first agent memory system. The
review covers foundational architectures, small/efficient variants,
memory-specific applications, comparative analysis with transformers,
training on personal data, and 2025-2026 developments.

---

## 1. Foundational SSM Papers

### 1.1 HiPPO: Recurrent Memory with Optimal Polynomial Projections

- **Authors**: Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Re
- **Date**: August 2020 (NeurIPS 2020)
- **URL**: https://arxiv.org/abs/2008.07669
- **Key Insight**: Introduces a general framework for online compression of
  continuous signals by projection onto polynomial bases. HiPPO produces
  operators that project arbitrary functions onto the space of polynomials,
  enabling optimal online memorization. The key matrices (HiPPO-LegS,
  HiPPO-LagT) form the mathematical backbone of all subsequent SSMs.
- **Signet Relevance**: HiPPO formalizes exactly what Signet's memory
  system needs -- a principled way to compress a continuous stream of
  observations (agent interactions) into a fixed-size representation that
  optimally preserves information about the past. The polynomial projection
  framework could inform how memory embeddings decay and which memories
  get retained.

### 1.2 S4: Efficiently Modeling Long Sequences with Structured State Spaces

- **Authors**: Albert Gu, Karan Goel, Christopher Re
- **Date**: November 2021 (ICLR 2022 Oral)
- **URL**: https://arxiv.org/abs/2111.00396
- **Code**: https://github.com/state-spaces/s4
- **Key Insight**: Proposes the Structured State Space sequence model based
  on a new parameterization for the SSM. S4 combines continuous-time
  modeling (handles irregular sampling), recurrent processing (unbounded
  context with constant state), and convolutional computation (efficient
  parallel training). It was the first model to solve the Path-X task on
  Long Range Arena (sequences of length 16,384), setting substantial SotA
  on every LRA task.
- **Architecture**: Continuous-time SSM discretized via bilinear method.
  State matrix A initialized with HiPPO. Diagonal-plus-low-rank
  parameterization enables O(N log N) computation via Cauchy kernel.
- **Signet Relevance**: S4's ability to handle very long sequences with
  constant memory is directly applicable to processing long agent session
  histories. Its continuous-time nature means it naturally handles
  irregularly-spaced events (agent interactions don't happen at fixed
  intervals).

### 1.3 H3: Hungry Hungry Hippos -- Towards Language Modeling with State Space Models

- **Authors**: Daniel Y. Fu, Tri Dao, Khaled K. Saab, Arman W. Thomas,
  Atri Rudra, Christopher Re
- **Date**: December 2022 (ICLR 2023)
- **URL**: https://arxiv.org/abs/2212.14052
- **Code**: https://github.com/HazyResearch/H3
- **Key Insight**: Identified two critical capabilities SSMs lacked for
  language modeling: (1) recalling earlier tokens in the sequence, and
  (2) comparing tokens across the sequence. H3 addresses these by
  combining an SSM layer with a multiplicative interaction (similar to
  attention's QKV pattern). This was the first SSM to approach transformer
  quality on language modeling and scale to billions of parameters.
- **Architecture**: Two SSM layers (shift SSM for copying, diagonal SSM
  for comparison) with multiplicative gating. Critical component: hardware
  efficiency through FlashConv, which is essential for scaling SSMs to
  billion-parameter models.
- **Signet Relevance**: H3's insight about recall vs. comparison maps
  directly to memory retrieval needs. An agent memory system needs both
  capabilities: recalling specific past interactions (the shift/copy
  mechanism) and comparing current context against stored memories (the
  comparison mechanism).

### 1.4 Hyena Hierarchy: Towards Larger Convolutional Language Models

- **Authors**: Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu,
  Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Re
- **Date**: February 2023 (ICML 2023)
- **URL**: https://arxiv.org/abs/2302.10866
- **Key Insight**: Proposes a subquadratic drop-in replacement for
  attention constructed by interleaving implicitly parametrized long
  convolutions and data-controlled gating. Hyena achieves attention-quality
  results with subquadratic cost. At sequences of hundreds of thousands of
  tokens, Hyena operators are over 100x faster than optimized attention.
- **Architecture**: Order-N Hyena operator: N rounds of element-wise
  gating interleaved with long convolutions. Implicit parameterization of
  filters via a small neural network. Data-controlled gating provides the
  selectivity that purely linear models lack.
- **Signet Relevance**: Hyena's implicit filter parameterization is elegant
  for memory systems -- the filters themselves are learned and can adapt,
  rather than being fixed. The data-controlled gating is similar to what
  Signet needs: selectively attending to relevant memories based on
  current context.

### 1.5 Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- **Authors**: Albert Gu, Tri Dao
- **Date**: December 2023
- **URL**: https://arxiv.org/abs/2312.00752
- **Code**: https://github.com/state-spaces/mamba
- **Key Insight**: The breakthrough paper. Mamba introduces *selective*
  SSMs where the (A, B, C, delta) parameters are functions of the input,
  allowing the model to selectively propagate or forget information along
  the sequence depending on the current token. Combined with a
  hardware-aware parallel algorithm and a simplified architecture (no
  attention, no MLP blocks), Mamba achieves 5x throughput vs transformers,
  linear scaling in sequence length, and Mamba-3B matches transformers
  twice its size.
- **Three key ingredients** (per Albert Gu's July 2025 analysis):
  1. **State size**: Hidden state h_t is N-dimensional (N >> 1), storing
     N times more information than classical RNNs like LSTMs.
  2. **State expressivity**: Selective (data-dependent) state transitions
     A_t, B_t, C_t allow the model to choose what to remember vs forget
     -- impossible with time-invariant SSMs.
  3. **Training efficiency**: Hardware-aware parallel scan algorithm
     enables efficient GPU training despite the recurrent formulation.
- **Signet Relevance**: CRITICAL. Mamba's selective mechanism is
  essentially what Signet's memory scorer needs -- a learned function that
  decides what information to retain and what to forget, operating in
  constant memory with linear-time complexity. The 5x throughput advantage
  matters for a sidecar process that must not impact the host system.

### 1.6 Mamba-2: Transformers are SSMs (State Space Duality)

- **Authors**: Albert Gu, Tri Dao
- **Date**: May 2024
- **URL**: https://arxiv.org/abs/2405.21060
- **Key Insight**: Proves a deep mathematical connection: the SSD
  (Structured State Space Dual) model is simultaneously an SSM and a form
  of linear attention. The duality shows that a scalar-identity structure
  on A_t makes the SSM equivalent to causal linear attention with an
  input-dependent positional mask. This enables much larger state
  dimensions (N=64 to N=256 vs N=16 in Mamba-1) while being much faster
  to train via matrix multiplications.
- **Key tradeoff**: Mamba-2 restricts the diagonal A to scalar-times-
  identity (less expressive per-dimension dynamics) but compensates with
  much larger state size and faster training. Pure inference may still
  favor Mamba-1's more expressive diagonal A at smaller state sizes.
- **Performance**: At the Pile benchmark, slightly better scaling laws
  than Mamba-1. Substantially better on hard associative recall tasks
  (MQAR) due to 16x larger state sizes.
- **Signet Relevance**: The SSD algorithm's "chunkwise" processing
  (splitting sequences into segments, computing attention-like operations
  within segments, passing state between segments) maps well to how agent
  sessions work -- you process a session chunk, carry state forward.

### 1.7 Mamba-3: Improved Sequence Modeling using State Space Principles

- **Authors**: Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang,
  Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu
- **Date**: March 16, 2026 (3 days ago)
- **URL**: https://arxiv.org/abs/2603.15569
- **Affiliations**: CMU, Princeton, Together AI, Cartesia AI
- **Key Insight**: Three methodological improvements driven by an
  "inference-first" perspective:
  1. **Exponential-Trapezoidal Discretization**: More expressive
     recurrence derived from proper SSM discretization theory. Reveals
     an implicit convolution in the SSM input, replacing the short causal
     convolution previously thought essential for recurrent models.
  2. **Complex-valued State**: Enables richer state tracking (solves
     parity/arithmetic tasks that Mamba-2 cannot). Equivalent to a
     data-dependent rotary embedding, efficiently computable.
  3. **Multi-Input Multi-Output (MIMO)**: Switches from outer-product to
     matrix-multiplication state updates. Increases decoding FLOPs by 4x
     at fixed state size while maintaining similar wall-clock latency,
     improving both performance and hardware utilization.
- **Results at 1.5B**: +2.2 over Transformers, +1.9 over Mamba-2, +1.8
  over Gated DeltaNet on downstream accuracy. Achieves Mamba-2 perplexity
  with half the state size.
- **Signet Relevance**: The MIMO formulation is directly relevant -- it
  lets you pack more computation into the memory-bound decode step without
  increasing state size or latency. For a sidecar process doing inference,
  this means better predictions at the same memory cost. The complex-
  valued state tracking is interesting for modeling cyclical patterns
  (daily/weekly usage patterns).

---

## 2. The Broader Modern Recurrent Model Family

### 2.1 Related Architectures

All of the following models can be cast into the same SSM equation
h_t = A_t * h_{t-1} + B_t * x_t, differing mainly in the structure of
A_t and the training algorithm:

| Model | Origin | Key Contribution | Date |
|-------|--------|-----------------|------|
| **RWKV** | Bo Peng et al. | RNN-centric; "Receptance Weighted Key Value" | 2023 |
| **RetNet** | Microsoft | Retentive network with multi-scale retention | 2023 |
| **Griffin/Hawk** | Google DeepMind | Mixing gated linear recurrences with local attention | 2024 |
| **xLSTM** | Sepp Hochreiter et al. | Extended LSTM with exponential gating, scalar (sLSTM) and matrix (mLSTM) memory | 2024 (NeurIPS 2024) |
| **GLA** (Gated Linear Attention) | Yang et al. | Gating mechanism similar to Mamba in linear attention framework | 2024 |
| **Gated DeltaNet** | Yang et al. | Delta rule for associative memory; linear attention with online learning updates | 2024-2025 |
| **DeltaProduct** | 2025 | Increases DeltaNet expressivity through products | 2025 |
| **S5** | Smith, Warrington, Linderman | Simplified SSM using parallel scan on diagonalized linear SSM | 2023 (ICLR 2023) |
| **Based** | Arora, Eyuboglu et al. (Cartesia) | Simple linear attention that balances recall-throughput tradeoff | 2024 |

**Critical observation from Albert Gu (July 2025)**: "All of these models
are much more similar to each other than they are to quadratic attention."
The real divide is between constant-memory compressed-state models
(SSMs/linear RNNs) and growing-cache models (transformers with KV cache).

### 2.2 Test-Time Training (TTT) Layers

- **Paper**: "Learning to (Learn at Test Time): RNNs with Expressive
  Hidden States"
- **Authors**: Yu Sun, Xinhao Li, Karan Dalal, et al.
- **Date**: July 2024
- **URL**: https://arxiv.org/abs/2407.04620
- **Key Insight**: Makes the hidden state of a recurrent model a machine
  learning model itself, and the update rule a step of self-supervised
  learning. TTT-Linear (hidden state = linear model) and TTT-MLP (hidden
  state = 2-layer MLP). Unlike Mamba which plateaus after 16k context,
  TTT keeps reducing perplexity with more tokens, like transformers.
- **Signet Relevance**: HIGH. TTT directly formalizes what Signet's
  predictor does conceptually -- learning from the test sequence itself.
  The hidden state as an "associative memory updated by online
  optimization" is exactly the paradigm of learning user patterns at
  inference time.

### 2.3 Hybrid SSM-Transformer Models

The consensus from 2024-2026 research is that the optimal architecture
combines SSM layers with a small proportion of attention layers:

| Model | Organization | Architecture | Scale | Date |
|-------|-------------|--------------|-------|------|
| **Jamba** | AI21 Labs | First production-grade Transformer-Mamba-MoE hybrid | 52B (12B active) | March 2024 |
| **Samba** | Microsoft | Mamba + sliding window attention | Various | June 2024 |
| **Zamba** | Zyphra | Mamba + shared attention | 7B | May 2024 |
| **Bamba** | IBM + CMU + Princeton | Mamba2 + attention, open-source | 9B | April 2025 |
| **Nemotron-H** | NVIDIA | Mamba2 + attention hybrid, SotA at scale | Up to 56B (560B MoE total) | March 2025 |
| **Tencent T1/TurboS** | Tencent | SSM-Transformer hybrid | Large-scale | March 2025 |
| **Qwen3 (linear variant)** | Alibaba | Gated DeltaNet layers in Qwen | Various | 2025 |
| **Kimi Linear** | Moonshot AI | Gated DeltaNet-based linear model | Various | 2025 |

**Optimal ratio**: 3:1 to 10:1 SSM:attention layers, independently
verified by dozens of research groups. This is not just about efficiency
-- it actually improves modeling power. "Taking a pure Transformer and
replacing most layers with SSM layers would both improve efficiency *and*
performance." (Gu, July 2025)

---

## 3. Small/Efficient SSM Architectures

### 3.1 IBM FlowState (9.1M Parameters)

- **Organization**: IBM Research
- **Date**: September 2025 (NeurIPS 2025 Workshop)
- **URL**: https://research.ibm.com/blog/SSM-time-series-model
- **Model**: https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1
- **Key Insight**: At just 9.1M parameters, FlowState is the smallest
  model in the GIFT-Eval top 10 for zero-shot time-series forecasting,
  outperforming rivals over 20x its size. Uses an S5-based encoder paired
  with a basis-function decoder that converts timescale-invariant hidden
  states into predictions at arbitrary resolution.
- **Architecture**: S5 SSM encoder + basis-function decoder. The encoder
  converts time series into timescale-invariant representations; the
  decoder interprets hidden state elements as coefficients of basis
  functions.
- **Signet Relevance**: CRITICAL. This proves that SSMs at sub-10M
  parameters can outperform much larger models on temporal prediction
  tasks. This is exactly the parameter range for a Rust sidecar binary.
  The timescale-invariant encoding is relevant for modeling agent
  interactions that happen at varying frequencies.

### 3.2 Quantizing Small-Scale SSMs for Edge AI

- **Authors**: Leo Zhao, Tristan Torchet, Melika Payvand, Laura Kriener,
  Filippo Moro
- **Date**: June 2025
- **URL**: https://arxiv.org/abs/2506.12480
- **Key Insight**: Analyzes quantization effects on small-scale SSMs
  (S4D architecture). Post-training quantization (PTQ) at 8-bit drops
  performance to 40% on sMNIST, but quantization-aware training (QAT)
  recovers it to 96%. State matrix A and internal state x are particularly
  sensitive to quantization. Proposes heterogeneous quantization strategy
  (different precision for different components) achieving 6x memory
  reduction without performance loss.
- **Key finding**: The A matrix and hidden state require higher precision
  than weights, suggesting mixed-precision is essential for edge SSMs.
- **Signet Relevance**: DIRECTLY APPLICABLE. For the Rust sidecar, this
  paper provides the recipe: use QAT (not PTQ), keep A matrix and hidden
  state at higher precision (fp16 or bf16), quantize weights to int8 or
  lower. 6x memory reduction means a 10M-parameter SSM could fit in ~7MB
  of memory.

### 3.3 Efficient Unstructured Pruning of Mamba for Edge

- **Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma
- **Date**: November 2025 (EMNLP 2025)
- **URL**: https://aclanthology.org/2025.emnlp-main.562/
- **Key Insight**: Achieves up to 70% parameter reduction on Mamba with
  only 3-9% performance drop. Uses pruning based on both weight and
  gradient importance (preserving Mamba's unique recurrent dynamics),
  gradual pruning schedule, and global allocation strategy. Results:
  1.77x faster inference, 46% memory reduction.
- **Key finding**: Mamba is remarkably robust to pruning, more so than
  transformers. This is likely because the recurrent state naturally
  distributes information across parameters rather than concentrating it
  in specific attention heads.
- **Signet Relevance**: DIRECTLY APPLICABLE. A 130M Mamba model pruned
  70% becomes ~39M parameters. Combined with int8 quantization, this
  could yield a sub-20MB binary -- well within sidecar constraints.

### 3.4 Distilling Low-Rank Mamba for Edge

- **Paper**: "Distilling Low-Rank Mamba for Edge Multispectral Fusion
  Object Detection"
- **Date**: March 2026
- **URL**: https://arxiv.org/abs/2603.06920
- **Key Insight**: Applies knowledge distillation and low-rank
  factorization to create lightweight Mamba variants for edge deployment.
  Demonstrates that Mamba's state structure is amenable to rank reduction
  without catastrophic quality loss.

### 3.5 Parameter Budget Analysis for Signet

Given the research, here is a realistic parameter budget for a Signet
memory predictor SSM sidecar:

| Component | Parameters | Memory (int8) | Memory (fp16) |
|-----------|-----------|---------------|---------------|
| 4-layer Mamba, d_model=128, N=16 | ~2M | ~2MB | ~4MB |
| 8-layer Mamba, d_model=256, N=32 | ~8M | ~8MB | ~16MB |
| 12-layer Mamba, d_model=384, N=64 | ~25M | ~25MB | ~50MB |
| FlowState-equivalent S5 | ~9M | ~9MB | ~18MB |

The sweet spot for a Rust sidecar appears to be 2-10M parameters,
yielding a binary in the 5-20MB range. IBM's FlowState proves this
range is viable for strong temporal prediction.

---

## 4. SSMs for Memory and Retrieval

### 4.1 Mathematical Formalism for Memory Compression in Selective SSMs

- **Author**: Siddhanth Bhat
- **Date**: October 2024
- **URL**: https://arxiv.org/abs/2410.03158
- **Key Insight**: Develops rigorous information-theoretic framework for
  understanding memory compression in selective SSMs. Uses mutual
  information and rate-distortion theory to formalize the tradeoff between
  memory efficiency and information retention. Proves stability and
  convergence of hidden state in selective SSMs, ensuring reliable
  long-term memory retention. Provides theoretical bounds on compressible
  information without sacrificing performance.
- **Key results**: Selective gating mechanism (Mamba's selectivity)
  dynamically filters based on input relevance, achieving significant
  compression. The rate-distortion analysis shows SSMs operate near the
  theoretical optimum for sequence compression.
- **Signet Relevance**: HIGH. This paper provides the theoretical
  foundation for using SSMs as memory compressors. The information-
  theoretic bounds tell us exactly how much agent interaction history
  can be compressed into a fixed-size state without losing critical
  information for prediction.

### 4.2 SleepGate: Sleep-Inspired Memory Consolidation for LLMs

- **Author**: Ying Xie (Kennesaw State University)
- **Date**: March 15, 2026 (5 days ago)
- **URL**: https://arxiv.org/abs/2603.14517
- **Key Insight**: Proposes biologically-inspired framework for active
  memory management in transformer KV caches, directly inspired by
  sleep-dependent memory consolidation. Three mechanisms:
  1. **Conflict-aware temporal tagger**: Detects when new entries
     supersede old ones using semantic signatures.
  2. **Forgetting gate**: Lightweight network trained to selectively
     evict or compress stale cache entries.
  3. **Consolidation module**: Merges related surviving entries into
     compact summary representations.
  These operate in "sleep micro-cycles" triggered by attention entropy.
- **Results**: 99.5% retrieval accuracy at interference depth 5, 97% at
  depth 10, while all baselines remain below 18%.
- **Connection to SSMs (from paper)**: "State-space models implicitly
  implement forms of forgetting through their inductive biases. However,
  these are fixed architectural choices rather than learned, content-
  dependent forgetting policies."
- **Signet Relevance**: DIRECTLY RELEVANT TO SIGNET'S RETENTION DECAY.
  SleepGate's three modules map exactly to Signet's memory pipeline
  needs: (1) detecting when new memories supersede old ones (entity
  dedup), (2) selectively forgetting stale memories (retention decay),
  (3) consolidating related memories into summaries (session summaries).
  The "sleep micro-cycle" concept could be a periodic maintenance pass.

### 4.3 The Brain-Database Analogy (Gu, July 2025)

Albert Gu's influential blog post "On the Tradeoffs of SSMs and
Transformers" (https://goombalab.github.io/blog/2025/tradeoffs/)
provides the most useful conceptual framework for understanding SSMs
in a memory context:

- **Transformers are like databases**: Every observation is filed away
  for future reference (KV cache grows linearly). Good for exact
  retrieval of specific items.
- **SSMs are like brains**: Finite-sized memories that are always on,
  processing new inputs in real-time. Good for maintaining a long,
  persistent summary of context without needing to recall every detail.

"Maintaining a continual conversation with an assistant is much more
like human conversations and relationships: what matters is a long,
persistent *summary* of the context, remembering the *shape and flow*
of the interactions without needing to recall every specific detail.
No one needs a scratchpad to have a continual relationship with their
friend. This is exactly where the more brain-like nature of SSMs is
more suitable."

This is *exactly* Signet's use case. The agent memory system needs the
"brain-like" persistent compressed state for ongoing context, augmented
by database-like retrieval (the SQLite + FTS5 + vector search that
already exists) for specific recall.

### 4.4 Infini-attention: Compressive Memory for Transformers

- **Authors**: Google (Tsendsuren Munkhdalai et al.)
- **Date**: April 2024
- **URL**: https://arxiv.org/abs/2404.07143
- **Key Insight**: Incorporates a compressive memory into vanilla
  attention, theoretically enabling infinite context length with bounded
  memory. Maintains a single buffer for all memory of earlier segments.
- **Practical issues**: Follow-up analysis by Hugging Face showed the
  approach struggles in practice, with the compressive memory failing to
  adequately represent earlier context.
- **Signet Relevance**: CAUTIONARY. Shows that naive compression of
  attention-based memory doesn't work well. SSMs' inherent compression
  (via the recurrent state) may be more principled than bolting
  compression onto attention.

### 4.5 Memory Augmented State Space Models for Time Series

- **Source**: IJCAI 2022
- **URL**: https://www.ijcai.org/proceedings/2022/0479.pdf
- **Key Insight**: Unlike fixed-order Markovian SSMs, this model features
  an external memory system that stores informative latent state
  experiences. The external memory augments the SSM's fixed-size state
  with retrievable historical states.
- **Signet Relevance**: The hybrid of SSM compressed state + external
  memory retrieval is exactly the architecture Signet should consider:
  SSM for temporal pattern compression, SQLite/vector store for
  retrievable specific memories.

### 4.6 SSMs for Sequential Recommendation

Multiple recent papers apply SSMs to sequential recommendation, which
is closely analogous to predicting memory relevance:

- **SIGMA** (AAAI 2025): "Selective Gated Mamba for Sequential
  Recommendation" -- Uses Mamba's selective mechanism to model user
  preference evolution over item interaction sequences.
- **M2Rec** (May 2025): "Multi-scale Mamba for Efficient Sequential
  Recommendation" -- Multi-scale SSM captures both short-term and
  long-term user preference patterns.
- **Hierarchical Mamba for Recommendation** (2025): Models long and
  short-term preference with hierarchical Mamba architecture.

These are directly analogous to Signet's task: given a sequence of
agent interactions, predict which stored memories will be relevant to
the next interaction.

---

## 5. SSMs vs Transformers: Where SSMs Clearly Win

### 5.1 Summary from Comprehensive Benchmarking (July 2025)

From "Characterizing State Space Model (SSM) and SSM-Transformer"
(https://arxiv.org/abs/2507.12442):

| Dimension | SSM Advantage | Transformer Advantage |
|-----------|--------------|----------------------|
| **Long sequences** | Linear time/memory | Quadratic bottleneck |
| **Streaming/online** | Constant state, instant | Growing KV cache |
| **Inference latency** | O(N^2) per step (state) | O(TN) per step (cache scan) |
| **Memory at inference** | Constant (state size) | Linear in context (KV cache) |
| **Byte/character-level** | Strong advantage | Needs tokenization |
| **Time series** | Native continuous-time | Requires patching |
| **Training throughput** | Comparable (Mamba-2+) | Highly optimized |
| **Exact recall** | Weak (fuzzy memory) | Strong (full cache) |
| **Algorithmic reasoning** | Improving (Mamba-3) | Strong |

### 5.2 Key Quantitative Findings

From Albert Gu's analysis (July 2025) and NVIDIA's benchmarks:

1. **Throughput**: Mamba achieves 5x higher throughput than transformers
   of equal size at inference time.
2. **Memory scaling**: Transformer KV cache grows linearly with context;
   SSM state is constant regardless of context length.
3. **Byte-level modeling**: When comparing on raw bytes (no tokenization),
   SSMs strongly outperform transformers even when transformers use 2x
   more FLOPs. "Keeping the same models and same data, but simply
   untokenizing the inputs, simultaneously lets the Transformer use much
   more compute but also decreases its performance relative to the SSM."
4. **DNA modeling**: SSMs scale substantially better than transformers
   on DNA (4-token vocabulary, high-resolution input).
5. **Hybrid advantage**: Replacing 70-90% of transformer layers with SSM
   layers improves both efficiency AND quality.

### 5.3 Where SSMs Struggle

1. **Exact associative recall**: "SSMs can't memorize a phonebook in one
   pass and then recite it back." But Mamba-3's complex-valued state and
   MIMO formulation significantly improve recall capabilities.
2. **State tracking**: Pure SSMs historically failed at parity detection
   of bit sequences. Mamba-3's complex-valued state solves this.
3. **In-context learning**: SSMs show weaker in-context learning than
   transformers in some settings, though TTT layers address this.

### 5.4 The Tokenization Insight

Gu's analysis reveals a deeper principle: "The inductive bias of soft
attention is hard attention." Transformers work best on *pre-compressed,
semantically meaningful tokens*. When data is high-resolution and
individual elements aren't meaningful (characters, raw sensor data,
time-series points), SSMs have a clear modeling advantage because they
naturally compress data into meaningful abstractions.

For Signet: agent interaction data (timestamps, memory accesses,
feedback signals) is high-resolution, irregularly-sampled, and
individual data points are not independently meaningful. This is
exactly the regime where SSMs excel.

---

## 6. Training SSMs on Small/Personal Data

### 6.1 Few-Shot and Low-Data Properties

SSMs have inherent advantages for low-data regimes:

1. **Structural priors**: The SSM parameterization (A matrix from HiPPO,
   continuous-time formulation) provides strong inductive biases that
   reduce sample complexity. As noted in the Deep State Space Models for
   Time Series paper (NeurIPS): "When there is little data to learn from,
   the structure imposed by the SSM can alleviate overfitting."

2. **FlowState proof point**: IBM's 9.1M-parameter FlowState achieves
   top-10 zero-shot performance on GIFT-Eval, outperforming 200M+
   parameter models. This demonstrates that SSMs can achieve strong
   generalization with far fewer parameters, implying less training
   data is needed.

3. **Transfer from pre-training**: The HiPPO initialization provides a
   mathematically optimal starting point for the state dynamics. This
   is a form of inductive bias that doesn't require pre-training data.

### 6.2 Online/Continual Learning

Several works directly address continual learning with SSMs:

1. **TTT layers** (Sun et al., 2024): The hidden state literally learns
   from the test sequence via self-supervised gradient steps. This is
   the most direct form of online learning -- the model adapts to each
   new user/context at inference time.

2. **TTT4Rec** (September 2024, https://arxiv.org/abs/2409.19142):
   Applies test-time training specifically to recommendation, using
   self-supervised learning during inference to dynamically update model
   parameters. Directly applicable to Signet's memory relevance
   prediction.

3. **FedSSM** (IEEE TMC, 2026): Uses state-space models to mitigate
   catastrophic forgetting in personalized federated learning. By
   capturing temporal evolution of model parameters through hidden
   states, SSMs enhance retention of critical knowledge across training
   rounds. This directly validates SSMs for personalization without
   forgetting.

4. **STAD** (October 2024, https://arxiv.org/abs/2407.12492): Temporal
   Test-Time Adaptation with State-Space Models. Proposes a probabilistic
   SSM that adapts deployed models to temporal distribution shifts by
   learning the dynamics of distribution change.

### 6.3 Personalization Strategy for Signet

Based on the research, the optimal strategy for Signet would be:

1. **Pre-train a base SSM** on anonymized community training signals
   (the federated learning approach described in VISION.md).
2. **Ship the base model** with every Signet install (9-25M parameters,
   <20MB binary).
3. **Fine-tune locally** using the user's own interaction data via:
   - Low-rank adaptation (LoRA-like) to keep fine-tuning efficient
   - Online learning with the user's own memory feedback signals
   - Periodic "sleep" consolidation passes (SleepGate-inspired)
4. **The SSM hidden state itself** serves as a persistent representation
   of the user's interaction patterns, updated with each session.

This maps 1:1 to VISION.md: "A neural network unique to each user,
trained on their own interaction patterns, that gets sharper the longer
you use it. No shared personal weights. Your weights never leave your
machine."

---

## 7. Recent Developments (2025-2026)

### 7.1 Production Deployments

SSMs have moved from research to production:

- **Cartesia AI** (Albert Gu's company): Production SSM-based models for
  real-time audio (Sonic TTS) and language. Claims models efficient
  enough to "run pretty much anywhere." Uses SSMs for streaming voice
  synthesis in 40+ languages.
- **IBM Granite 4.0**: Incorporating Bamba's hybrid SSM-transformer
  architecture into production enterprise models.
- **NVIDIA Nemotron-H**: Production hybrid at 560B total parameters with
  state-of-the-art performance.
- **Qwen3 / Kimi Linear**: Production LLMs using Gated DeltaNet (SSM
  family) layers.
- **Together AI**: Hosting Mamba-3 models for inference, actively
  developing SSM infrastructure.

### 7.2 Rust/Native Inference Ecosystem

The Rust ML ecosystem now supports SSM inference:

- **Candle** (Hugging Face): Minimalist ML framework for Rust with
  Mamba inference support. "Serverless (on CPU), small and fast
  deployments." Supports quantization.
  GitHub: https://github.com/huggingface/candle
- **Burn**: Rust ML framework with ONNX import support.
- **ONNX Runtime**: Mature Rust bindings, supports custom ops needed
  for SSM selective scan.

Candle's existing Mamba implementation is particularly relevant --
it demonstrates that SSM inference in Rust on CPU is already viable
today. This could serve as the foundation for Signet's predictor
sidecar.

### 7.3 Timeline of Key Developments

| Date | Development |
|------|------------|
| Aug 2020 | HiPPO: Mathematical foundation for SSM memory |
| Nov 2021 | S4: First efficient deep SSM (ICLR 2022) |
| Dec 2022 | H3: SSMs approach transformer quality on language |
| Feb 2023 | Hyena: Subquadratic attention replacement |
| Aug 2023 | S5: Simplified SSM with parallel scan |
| Dec 2023 | **Mamba: Selective SSMs break through** |
| Mar 2024 | Jamba: First production hybrid SSM-Transformer |
| Apr 2024 | Infini-attention: Compressive memory for transformers |
| May 2024 | **Mamba-2: State Space Duality** |
| Jul 2024 | TTT: Test-time training layers |
| Oct 2024 | Memory compression formalism for selective SSMs |
| Nov 2024 | xLSTM published (NeurIPS 2024) |
| Nov 2025 | Mamba pruning for edge (EMNLP 2025) |
| Mar 2025 | **Nemotron-H: SSM-Transformer hybrid at scale** |
| Apr 2025 | Bamba: IBM's open hybrid model |
| Jun 2025 | Quantized small-scale SSMs for edge AI |
| Jul 2025 | Albert Gu's "Tradeoffs" analysis |
| Sep 2025 | FlowState: 9.1M-param SSM beats 200M+ models |
| Mar 2026 | **Mamba-3: Inference-first design** |
| Mar 2026 | SleepGate: Sleep-inspired memory consolidation |
| Mar 2026 | FedSSM: SSMs for personalized federated learning |

---

## 8. Synthesis: Architecture Recommendation for Signet

### 8.1 Why SSMs Are the Right Choice for Signet's Predictor

1. **Constant memory**: The predictor processes arbitrarily long
   interaction histories without growing memory usage.
2. **Linear time**: Processing cost scales linearly with session length,
   not quadratically.
3. **Streaming inference**: SSMs naturally process data as a stream,
   matching how agent interactions arrive in real-time.
4. **Compression is the point**: The SSM hidden state IS a compressed
   representation of interaction history. This is not a limitation --
   it is the desired behavior for a memory relevance predictor.
5. **Small footprint**: Proven at 9.1M parameters (FlowState). Amenable
   to quantization (6x reduction) and pruning (70% sparsity).
6. **Rust-ready**: Candle already implements Mamba inference in Rust.
7. **Online learning**: TTT and related work show SSMs can adapt at
   inference time, enabling the personalization described in VISION.md.

### 8.2 Recommended Architecture

```
Signet Memory Predictor SSM
=============================
Architecture: Mamba-2 or Mamba-3 based (selective SSM)
Parameters:   5-10M (4-8 layers, d_model=192-256, N=32-64)
Precision:    Heterogeneous (A matrix fp16, weights int8)
Binary size:  ~10-15MB (Rust, quantized)
Inference:    CPU-only, <5ms per interaction event
Training:     Online fine-tuning via LoRA + periodic consolidation
State:        Persistent hidden state saved between sessions

Input features per interaction event:
- Memory access patterns (which memories were retrieved)
- Agent feedback signals (helpful/not helpful)
- Temporal features (time-of-day, day-of-week, session duration)
- Content similarity scores (current context vs. retrieved memories)
- Memory metadata (age, access count, decay score)

Output:
- Memory relevance scores (which memories to inject)
- Retention predictions (which memories to preserve vs. decay)
- Temporal pattern indicators (cyclical access patterns)
```

### 8.3 Open Questions for Further Research

1. **Complex-valued state for cyclical patterns**: Mamba-3's complex
   state could naturally model daily/weekly usage patterns. Worth
   prototyping.
2. **MIMO for multi-signal prediction**: Can MIMO SSMs simultaneously
   predict relevance, retention, and temporal patterns from a single
   forward pass?
3. **SleepGate for memory consolidation**: Can the sleep micro-cycle
   concept replace or augment Signet's current retention decay?
4. **Federated pre-training**: What's the minimum community data needed
   to pre-train a useful base model? FlowState's success at 9.1M params
   suggests the data requirements may be modest.
5. **SSM state as identity embedding**: Could the persistent SSM state
   serve as a compact representation of agent identity/personality,
   complementing SOUL.md?

---

## References

Complete bibliography of papers cited in this review:

1. Gu, A., Dao, T., Ermon, S., Rudra, A., Re, C. (2020). HiPPO: Recurrent Memory with Optimal Polynomial Projections. NeurIPS 2020. arXiv:2008.07669
2. Gu, A., Goel, K., Re, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022. arXiv:2111.00396
3. Fu, D.Y., Dao, T., Saab, K.K., Thomas, A.W., Rudra, A., Re, C. (2023). Hungry Hungry Hippos: Towards Language Modeling with State Space Models. ICLR 2023. arXiv:2212.14052
4. Poli, M. et al. (2023). Hyena Hierarchy: Towards Larger Convolutional Language Models. ICML 2023. arXiv:2302.10866
5. Gu, A., Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752
6. Gu, A., Dao, T. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. arXiv:2405.21060
7. Lahoti, A. et al. (2026). Mamba-3: Improved Sequence Modeling using State Space Principles. arXiv:2603.15569
8. Smith, J.T., Warrington, A., Linderman, S.W. (2023). Simplified State Space Layers for Sequence Modeling. ICLR 2023. arXiv:2208.04933
9. Sun, Y. et al. (2024). Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv:2407.04620
10. Bhat, S. (2024). Mathematical Formalism for Memory Compression in Selective State Space Models. arXiv:2410.03158
11. Xie, Y. (2026). Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models. arXiv:2603.14517
12. Zhao, L. et al. (2025). Quantizing Small-Scale State-Space Models for Edge AI. arXiv:2506.12480
13. Shihab, I.F., Akter, S., Sharma, A. (2025). Efficient Unstructured Pruning of Mamba State-Space Models for Resource-Constrained Environments. EMNLP 2025.
14. AI21 Labs (2024). Jamba: A Hybrid Transformer-Mamba Language Model. arXiv:2403.19887
15. NVIDIA (2025). Nemotron-H: A Family of Accurate, Efficient Hybrid Mamba-Transformer Models. arXiv:2504.03624
16. IBM Research (2025). Bamba: An Open Hybrid SSM-Transformer Model. Hugging Face.
17. IBM Research (2025). FlowState: SSM-Based Time-Series Foundation Model. NeurIPS 2025 Workshop.
18. Zhang, W. et al. (2026). Mitigating Catastrophic Forgetting in Personalized Federated Learning for Edge Devices Using State-Space Models. IEEE TMC.
19. Gu, A. (2025). On the Tradeoffs of SSMs and Transformers. Goomba Lab Blog.
20. Munkhdalai, T. et al. (2024). Efficient Infinite Context Transformers with Infini-attention. arXiv:2404.07143
21. Beck, M. et al. (2024). xLSTM: Extended Long Short-Term Memory. NeurIPS 2024. arXiv:2405.04517
22. Yang, S. et al. (2024). Gated Linear Attention / DeltaNet. arXiv:2406.06484
23. SIGMA (2025). Selective Gated Mamba for Sequential Recommendation. AAAI 2025.
24. M2Rec (2025). Multi-scale Mamba for Efficient Sequential Recommendation. arXiv:2505.04445
25. Cartesia AI (2024). Based: Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.
26. Graf, L. et al. (2025). FlowState-R1: A State-Space Model for Time-Series Forecasting. IBM Research.
