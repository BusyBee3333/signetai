---
question: How can state-space models be adapted for continual, online, and per-user learning in an agent memory system without catastrophic forgetting?
date: 2026-03-20
informed_by:
  - docs/research/technical/SSM-LITERATURE-REVIEW.md
  - docs/research/technical/RESEARCH-SSM-INTEGRATION.md
---

# SSM Continual Learning Deep Dive: Test-Time Training, Online Adaptation, and Personalization Without Forgetting

This research document surveys the state of the art in making SSMs
learn continuously from streaming user data while maintaining a
shared base model. It covers seven intersecting research areas and
synthesizes practical implications for Signet's predictive memory
scorer architecture.

---

## 1. Test-Time Training (TTT) with SSMs

### 1.1 TTT Layers (Sun et al., 2024)

**Paper**: "Learning to (Learn at Test Time): RNNs with Expressive
Hidden States" (ICML 2024)
**Source**: https://arxiv.org/abs/2407.04620
**Code**: https://github.com/test-time-training/ttt-lm-pytorch

**Core Mechanism**: The hidden state is itself a machine learning
model (weight matrix W_t). Instead of a fixed-size vector updated
by a learned transition function, TTT makes W_t a parameterized
model f and updates it via gradient descent on a self-supervised
reconstruction loss at every token:

```
W_t = W_{t-1} - eta * grad_loss(W_{t-1}; x_t)
loss(W; x_t) = ||f(theta_K * x_t; W) - theta_V * x_t||^2
```

theta_K corrupts the input (training view) and theta_V provides the
reconstruction target (label view). The model must discover
correlations between dimensions to reconstruct, analogous to
denoising autoencoders.

**Two Instantiations**:
- **TTT-Linear**: f(x) = x + LN(Wx), hidden state is a square matrix.
  Matches Mamba wall-clock time, faster than Transformers at 8k context.
- **TTT-MLP**: f(x) = x + LN(MLP(x)), hidden dim 4x input size, GELU.
  Higher FLOPs but better perplexity on long contexts.

**Mini-Batch TTT**: Instead of online GD (one gradient per token),
batch b=16 tokens together. Gradient computation parallelizes within
the batch via cumulative sum. Mini-batch TTT contributed the largest
single improvement: -1.70 perplexity over linear attention baseline.

**Quantitative Results** (1.3B params, Pile):
- 2k context: TTT-Linear ~11.09 ppl, matches Transformer/Mamba
- 8k context: TTT-Linear faster than Transformer, significantly
  better than Mamba
- 16k+ context: Mamba stops improving; TTT-Linear continues reducing
  perplexity
- At 760M and 1.3B, TTT-Linear outperforms Mamba using fewer FLOPs

**Computational Cost**: O(1) per token like RNNs, but the hidden
state stores d*d matrices vs d-dimensional vectors. The backward
pass computes Hessian-vector products (gradients of gradients).
Prior implementations achieved below 5% peak FLOPS on A100s due to
small mini-batch sizes.

**Signet Relevance**: TTT is the most direct mechanism for "learning
at inference time." A per-user predictor could use TTT layers where
the hidden state literally trains on the user's interaction patterns
as it processes them. The self-supervised loss provides automatic
adaptation without labeled data.

### 1.2 Test-Time Training Done Right (LaCT, 2025)

**Paper**: "Test-Time Training Done Right"
**Source**: https://arxiv.org/abs/2505.23884

**Key Breakthrough**: Uses extremely large chunks (2048 to 1M tokens)
as the basic unit for updating fast weights, solving TTT's GPU
utilization problem.

**Results**:
- GPU utilization: 5% -> 70% on A100 (14x improvement)
- State-to-parameter ratio >= 40% (vs 0.1-5% in prior work)
- Muon optimizer variant consistently outperforms gradient descent
  and momentum
- Models at 14B parameters with sequences up to 56k tokens

**Compute intensity scales with chunk size**: r <= min(h/2, b) where
h is state dimension and b is chunk size. Larger chunks improve the
compute-to-memory ratio dramatically.

**Signet Relevance**: The chunk-based approach maps naturally to
Signet's session model. A user's session is a natural chunk boundary.
Rather than updating per-token, update the predictor's fast weights
once per session using the full interaction as a chunk.

### 1.3 End-to-End TTT for Long Context (Sun et al., 2025)

**Paper**: "End-to-End Test-Time Training for Long Context"
**Source**: https://arxiv.org/abs/2512.23675
**Code**: https://github.com/test-time-training/e2e

**Architecture**: Standard Transformer with sliding-window attention.
TTT updates only the final 1/4 of MLP layers during inference using
next-token prediction as the self-supervised objective.

**Meta-Learning Outer Loop**: Prepares initialization via MAML-style
optimization. Each training sequence is treated as if it were a test
sequence; TTT is performed in the inner loop; the loss after TTT is
averaged and optimized through gradients of gradients.

**Inner Loop**: Mini-batch GD with b=1K tokens per batch. For 128K
context, this means 128 inner-loop gradient steps during prefill.

**Forgetting Mitigation**:
- Dual MLPs per block: one static (preserves pre-trained knowledge),
  one updated during TTT
- Selective layer updates: only 1/4 of blocks are updated
- Mini-batch GD (b=1K) vs online updates (b=1) for stability

**Results** (3B models, 164B training tokens):
- TTT-E2E scales with context length identically to full attention;
  Mamba-2 and Gated DeltaNet do not
- 2.7x faster than full attention at 128K context on H100
- 35x faster at 2M context
- Constant inference latency regardless of context length

**Training Cost**: 3.4x slower than full attention at 8K due to
gradient-of-gradient computation; 1.2x faster at 128K.

**Signet Relevance**: The dual-MLP architecture (static + trainable)
is directly applicable. A per-user Signet predictor can have frozen
base weights plus per-user trainable weights, updated via TTT's
inner loop during each session. The meta-learning outer loop can be
trained on the federated signal from all users.

### 1.4 SR-TTT: Surprisal-Aware Residual TTT (2026)

**Paper**: "SR-TTT: Surprisal-Aware Residual Test-Time Training"
**Source**: https://arxiv.org/abs/2603.06642
**Code**: https://github.com/swamynathanvp/Surprisal-Aware-Residual-Test-Time-Training

**Problem**: Pure TTT catastrophically fails on exact-recall tasks
because fast weights aggressively compress context into an
information bottleneck. Highly surprising tokens are overwritten.

**Solution**: Augments TTT with a loss-gated sparse memory:
- **Surprisal Filter**: Flags tokens when per-token reconstruction
  loss exceeds EMA-smoothed 95th percentile threshold AND chunk
  mean loss > 0.8 * threshold. Uses locality-sensitive hashing
  for O(1) amortized cost.
- **Residual Cache**: Fixed-capacity KV cache with priority-based
  eviction for flagged tokens.
- **Fusion Gate**: Output = TTT(x) + alpha * CacheAttention(x),
  with clamped alpha parameterization.

**Two-Stage Curriculum**: Stage 1 (steps 1-7K): base TTT training,
cache disabled. Stage 2 (steps 7K-10K): backbone frozen, cache
enabled.

**Results** (15.8M params, 2048 context):
- Needle-in-haystack exact match: +23% at depth 0.50, +20% at 0.75
- Small scale proof-of-concept; scaling behavior unknown

**Signet Relevance**: The surprisal filter concept maps to memory
importance scoring. Memories that cause high reconstruction loss
(surprising, novel information) should be preserved exactly rather
than compressed into the hidden state.

### 1.5 TTT4Rec: TTT for Sequential Recommendation

**Paper**: "TTT4Rec: A Test-Time Training Approach for Rapid
Adaption in Sequential Recommendation"
**Source**: https://arxiv.org/abs/2409.19142

**Architecture**: Embedding layer with RoPE + stacked TTT residual
blocks + prediction layer. Inner-loop reconstruction loss for
continuous weight updates during inference.

**Results**:
- Matches/exceeds SASRec on Gowalla (HR@10: 0.0934)
- Outperforms SASRec on NDCG@10 by 5.65%
- 3-6% improvement over BERT4Rec and Mamba4Rec
- Particularly benefits limited training scenarios (3:2:5 split
  shows 4-6% improvement vs 1-4% with more data)

**Data Requirements**: Works with average sequences as short as 9.54
items (Amazon-video-game dataset, minimum 5).

**Signet Relevance**: Directly validates TTT for personalized
sequential prediction. Agent memory retrieval is fundamentally a
sequential recommendation problem: given the sequence of past
interactions, predict which memories are most relevant now.

---

## 2. Catastrophic Forgetting in SSMs

### 2.1 Do SSMs Suffer Less Than Transformers?

**The nuanced answer**: SSMs have fundamentally different memory
characteristics but do NOT inherently suffer less from catastrophic
forgetting. The tradeoffs:

**SSM Memory Model** (Goomba Lab, 2025):
- SSMs compress all history into a fixed-size state. This creates a
  natural "fuzzy summary" that retains the shape and flow of past
  interactions but cannot preserve exact token-level details.
- Mamba catastrophically forgets distant content once context exceeds
  memory capacity.
- SSMs exhibit strong recency bias, intrinsic to the architecture
  regardless of selectivity mechanisms.
- SSMs naturally form abstractions (forced by compression), which
  may be "fundamental to intelligence" but loses fine-grained detail.

**Transformer Memory Model**:
- Full KV cache preserves exact token-level records but grows linearly.
- Better at exact recall, fine-grained retrieval, and associative tasks.
- Less compression means less forced abstraction.

**Key Insight**: For agent memory, SSMs are well-suited to maintaining
a persistent "sense" of the user (preferences, patterns, style) while
exact factual recall should use retrieval-augmented approaches. This
maps to Signet's hybrid search (vector + keyword) architecture.

**Source**: https://goombalab.github.io/blog/2025/tradeoffs/

### 2.2 SSM Hidden State as Gradient Descent

**Paper**: "State-space models can learn in-context by gradient
descent" (ICLR 2025)
**Source**: https://arxiv.org/abs/2410.11687

**Key Finding**: A single SSM layer with multiplicative gating can
reproduce gradient descent. The hidden state accumulates gradients:

```
z_t = I * z_{t-1} + (w_0^T * x_t - y_t) * x_t
```

This directly mirrors iterative gradient computation for an implicit
least-squares regression problem. The diagonal linear recurrence
acts as a gradient accumulator.

**Implications**: SSMs are not just sequence processors; they can
literally perform optimization algorithms in their forward pass.
This theoretical grounding connects SSM dynamics to learning theory
and suggests SSMs may be uniquely suited for online adaptation tasks.

### 2.3 Mamba-CL: Null-Space Projection for Continual Learning

**Paper**: "Mamba-CL: Optimizing Selective State Space Model in Null
Space for Continual Learning"
**Source**: https://arxiv.org/abs/2411.15469

**Mechanism**: Projects gradient updates onto subspaces orthogonal to
feature representations of previous tasks. Four sufficient conditions
maintain output consistency:

1. delta_t * Delta_A = 0 (A updates orthogonal to discretization)
2. X_t * Delta_W^delta = 0 (delta updates orthogonal to input features)
3. delta_t * X_t * Delta_W^B = 0 (B updates orthogonal to scaled inputs)
4. X_t * Delta_W^C = 0 (C updates orthogonal to input features)

Three projection matrices (H^1, H^2, H^3) project gradients during
training. Validated on up to 20 sequential tasks (20-split ImageNet-R).

**Signet Relevance**: When training the predictor on new user patterns,
null-space projection can ensure new learning doesn't destroy existing
per-user knowledge. The constraint is mathematical rather than heuristic.

### 2.4 MambaCL: Meta-Learning Mamba for Continual Learning

**Paper**: "Learning Mamba as a Continual Learner"
**Source**: https://arxiv.org/abs/2412.00776

**Architecture**: Mamba model processes streaming data as sequence
prediction: (x_1, y_1, ..., x_t, y_t, x_k^test) -> y_k^test.

**Selectivity Regularization**: Creates a binary ground-truth
association pattern indicating which preceding training samples match
a query's label. Regularizes Mamba's implicit selectivity patterns
via KL divergence during training only (no inference overhead).

**Results**:
- CUB-200: Mamba 83.0% vs Transformer 81.4%
- Aircraft: Mamba 55.3% vs Transformer 53.9%
- Generalizes beyond trained configurations (20 tasks -> 5-100 tasks)
- Requires significantly fewer parameters than Transformers

**Signet Relevance**: Validates Mamba as fundamentally suitable for
continual learning with strong generalization.

### 2.5 Inf-SSM: Exemplar-Free Continual Learning via Grassmannian Geometry

**Paper**: "Exemplar-Free Continual Learning for State Space Models"
**Source**: https://arxiv.org/abs/2505.18604

**Core Innovation**: Regularizes the infinite-horizon evolution of
SSMs using geometry of the infinite-dimensional Grassmannian manifold.
Instead of constraining weights directly, constrains the extended
observability subspace:

```
O_inf(A, C) = [C, CA, CA^2, CA^3, ...]^T
```

This subspace encodes full behavioral trajectory and remains invariant
under P-equivalence transformations (different parameterizations can
produce identical behavior).

**Regularization**: Chordal distance on the infinite Grassmannian:
L_ISM = E{d^2_chord(S_inf(A_{T-1}, C_{T-1}), S_inf(A_T, C_T))}

**Computational Efficiency**: Reduces Sylvester equation solving from
O(n^3) to O(n^2). For Vim (n=16), 100x FLOP reduction. Applies to
Mamba/S6 via averaging state representations across dimensions.

**Results** (ImageNet-R, 10 tasks):
- AIA: 62.82% vs 58.43% (best baseline, LwF)
- Forgetting measure reduced by 33.11%
- Average accuracy improved by 29.19%

**Signet Relevance**: The most mathematically principled approach to
SSM continual learning. The observability subspace constraint ensures
the SSM's behavioral trajectory is preserved, not just its weights.
This is ideal for a predictor that must add new user patterns without
disrupting existing predictions.

### 2.6 EWC Applied to Deep State-Space Models

**Paper**: "Regularization-Based Efficient Continual Learning in
Deep State-Space Models"
**Source**: https://arxiv.org/abs/2403.10123

**Methods Benchmarked**: EWC, MAS, SI, LwF applied to DSSMs.

**EWC for SSMs**: Quadratic regularization anchored to previously
learned parameters, weighted by Fisher information:
L = L_task + (lambda/2) * sum(M_j * (theta - theta_j*)^2)

**Key Finding**: LwF (functional regularization via output matching)
outperforms parameter-based methods (EWC, MAS, SI) for SSMs.
Computational costs independent of number of tasks and training data
size.

**Results** (Power Consumption, 4 sequential tasks):
- Baseline DSSM: MSE 26.15 -> 463.36 (catastrophic forgetting)
- CLDSSM-LwF: MSE 26.39 -> 69.65 (dramatic improvement)
- EWC/MAS: O(d_z^2) memory and compute
- LwF: O(d_z * T) memory

**Signet Relevance**: EWC provides the simplest integration path.
The Fisher information matrix identifies which predictor weights are
most important for existing predictions and penalizes changes to
them during new learning.

---

## 3. Online/Streaming Learning with Sequential Models

### 3.1 S6MOD: Plug-and-Play SSM for Online Continual Learning

**Paper**: "Enhancing Online Continual Learning with Plug-and-Play
State Space Model and Class-Conditional Mixture of Discretization"
(CVPR 2025)
**Source**: https://arxiv.org/abs/2412.18177
**Code**: https://github.com/MyToumaKazusa/S6MOD

**Architecture**: Extra branch after backbone with mixture of
discretization candidates. Each candidate Delta_i produced by
separate projection layers. Class-conditional routing selects more
patterns for uncertain classes, fewer for confident ones.

**Class-Conditional Routing**: Maintains feature prototypes per class
via moving average. Class uncertainty sigma_k computed from average
margin. Number of patterns N_k = ceil(N * sigma_k).

**Contrastive Discretization Loss**: Penalizes dissimilarity between
same-class samples while encouraging dissimilarity between different
classes.

**Results**: Designed for OCL where data arrives sequentially and
allows only one epoch. Integrates with 7 baselines showing consistent
1-4% improvements.

**Signet Relevance**: The class-conditional routing maps to
per-pattern routing in memory prediction. Uncertain predictions
(ambiguous user intent) get more model capacity; confident
predictions use minimal resources.

### 3.2 State Memory Replay (SMR)

**Paper**: "SMR: State Memory Replay for Long Sequence Modeling"
(ACL Findings 2024)
**Source**: https://arxiv.org/abs/2405.17534

**Problem**: Event-Triggered Control theory reveals the Non-Stable
State problem in SSMs -- deviations at sampling points cause error
transmission and accumulation, diverging the hidden state.

**Solution**: Learnable memories adjust the current state with
multi-step information, generalizing to sampling points different
from training data. Plug-and-play mechanism compatible with any SSM.

**Signet Relevance**: Session boundaries create natural sampling
discontinuities. SMR's approach of adjusting state with stored
memories at these boundaries maps to Signet's session summary and
memory consolidation workflow.

### 3.3 Concept Drift Detection

**Key Methods** for detecting when user behavior changes:

- **ADWIN (Adaptive Windowing)**: Adaptively partitions sliding
  windows, computes statistical differences, detects drift when
  difference exceeds threshold. Requires no prior knowledge.
- **Page-Hinkley Test**: Measures current vs mean accuracy over a
  window, comparing cumulative differences to a threshold.
- **Trinity-Controller ADWIN** (2025): Fuses three signals:
  Volatility Controller (statistically grounded), Adaptive Rate
  Controller (dynamic sensitivity), Performance-Based Controller
  (EMA of online accuracy).
- **Neural approaches** (2024): Autoencoders detect drift via
  reconstruction error changes; diffusion learning when drift
  detected, adversarial learning when stable.

**Signet Relevance**: The predictor must detect concept drift (user
changes projects, roles, or preferences). The Trinity-Controller's
multi-signal approach is closest to what Signet needs: combine
prediction accuracy EMA, session-level volatility, and retrieval
relevance feedback into a unified drift detector that triggers
predictor adaptation.

### 3.4 Online Learning Rate Schedules for SSMs

**Key Finding** (Mamba-2 / TTT connection): The SSM recurrent update
can be viewed as online optimization with the state as associative
memory. Mamba-2 parallelize via mini-batch gradient descent with the
state viewed as an associative memory.

**Practical Guidance**:
- Learning rate and batch size should be jointly scheduled
- Increasing batch size + learning rate decay ensures convergence
- Adaptive scheduling based on observed gradient norm decay
- For SSMs: compute intensity r <= min(h/2, b); larger chunks
  allow higher effective learning rates

---

## 4. Personalization Without Forgetting

### 4.1 Parameter-Efficient Fine-Tuning of SSMs

**Paper**: "Parameter-Efficient Fine-Tuning of State Space Models"
(NeurIPS 2024)
**Source**: https://arxiv.org/abs/2410.09016

**Critical Finding**: LoRA on SSM-specific parameters (A, B, C,
Delta) is **ineffective**. Performance drops to 76.9 GLUE when
applied to SSM modules alone. However, LoRA on **linear projection
matrices** achieves 87.0 GLUE -- near full fine-tuning (89.4).

**Why**: Tuning linear projections can match the expressive power of
SSM parameters (W^B, W^C, W^Delta). The SSM parameters' roles are
largely captured by their surrounding projections.

**SDT (Sparse Dimension Tuning)**: Specialized method for SSM
modules. Classifies state dimensions as zero (pruned), frozen
(aligned), or trainable (needs update). Combined SDT + LoRA on
projections achieves state-of-the-art with <1% parameters.

**Per-User Implications**: For a per-user adapter on a shared SSM:
- Target linear projections with LoRA (not SSM matrices)
- Use rank r=8-16 for balance
- Total adapter size: ~1% of model parameters
- Training time: ~0.67x full fine-tuning

### 4.2 MambaPEFT: Comprehensive PEFT for Mamba

**Paper**: "MambaPEFT: Exploring Parameter-Efficient Fine-Tuning
for Mamba"
**Source**: https://arxiv.org/abs/2411.03855

**7 methods, 20 variations evaluated**:

| Method | Params (K) | Best For |
|--------|-----------|----------|
| LoRAp(X) on in_proj/out_proj | 1,483 | Limited data |
| Conv1d-tuning | 156 | Memory-constrained |
| Additional-scan | 672 | Larger datasets |
| Affix-tuning (w/o proj) | 230 | General efficiency |
| ParallelAdapter | 663 | Medium |

**Key Findings**:
- LoRAp(X) achieves 71.52% on VTAB-1k vs ViT best 69.87%
- Simply combining high-performing methods is NOT sufficient (119M
  params: 67.68%; optimized 1M: 71.80%)
- Affix-tuning can merge during inference (zero cost)
- LoRA rank can exceed full-rank for small Mamba dimensions without
  degradation (unlike Transformers)

**Per-User Deployment**:
- Low memory: Conv1d-tuning (156K params per user)
- Best quality: LoRAp(X) (1.5M params per user)
- Zero inference cost: Affix-tuning with merge

### 4.3 Mamba's Lyapunov Stability for Fine-Tuning

**Paper**: "Mamba State-Space Models Can Be Strong Downstream
Learners" (TMLR 2025)
**Source**: https://arxiv.org/abs/2406.00209

**Theoretical Guarantee**: Mamba's recurrent dynamics have non-positive
maximal Lyapunov exponents. Small perturbations don't amplify
exponentially. This makes Mamba fundamentally more stable than
traditional RNNs during fine-tuning.

**Mixed-Precision Robustness**: Mamba shows average divergence of
0.1 in FP16 fine-tuning (vs Pythia 0.14, OpenELM 0.54).

**LoRA on Memory Buffer W**: Targeting Mamba's large memory buffer
for LoRA creates implicit weight-tying across temporal parameters
(Delta_t, B_t, C_t).

**Scale Results**: MPFT+PEFT enables fine-tuning 2.8B Mamba models on
24GB GPUs. 2.15x faster training throughput, 62.7% lower memory.

**Instruction tuning on OpenHermes (242K samples)**: Instruction-tuned
Mamba achieves 81.5-133% of few-shot improvements relative to Pythia.

**Signet Relevance**: Lyapunov stability guarantees that per-user
fine-tuning won't destabilize the model. The stability bounds hold
across all tested configurations, giving confidence that continuous
user-specific updates are safe.

### 4.4 Few-Shot Personalization

**FSPO** (ICLR 2025): Reframes reward modeling as meta-learning.
Generates 1M+ synthetic personalized preferences. Achieves 87%
Alpaca Eval winrate for synthetic users, 72% for real humans.

**Key Constraint**: "It is not practical to gather enough data or
store separate copies of the model or low-rank adapter weights for
every user." Solutions must share structure.

**TTT4Rec Evidence**: Meaningful adaptation from as few as 5-10
interactions per user. Limited training data (3:2:5 split) actually
shows larger TTT benefits (4-6%) than abundant data (1-4%).

**Signet Relevance**: The predictor can start producing meaningful
per-user predictions within the first few sessions. TTT's inner-loop
gradient updates provide immediate adaptation; longer-term patterns
accumulate through the adapter weights.

---

## 5. Federated Learning with SSMs

### 5.1 FedSSM: State-Space Models for Federated Learning

**Paper**: "Mitigating Catastrophic Forgetting in Personalized
Federated Learning for Edge Devices Using State-Space Models"
**Source**: https://ieeexplore.ieee.org/document/11195747
**Code**: https://github.com/wei-d-zhang/FedSSM

**Problem**: Bidirectional catastrophic forgetting in federated
learning -- local personalization overwritten by global updates,
and global representations degraded by heterogeneous local data.

**Solution**: SSMs capture temporal evolution of local model
parameters through hidden states, enhancing retention of critical
knowledge across training rounds.

**Datasets**: Fashion-MNIST, CIFAR-10, CIFAR-100 (ResNet-18 backbone)

**Baselines**: Outperforms Ditto, FedACG, FedALA, MOON, FedDecorr,
FedCross -- particularly with high data heterogeneity.

**Signet Relevance**: Directly validates the VISION.md architecture.
FedSSM demonstrates that SSMs can track parameter evolution in
federated settings, exactly what's needed for aggregating anonymous
training signals across users into a shared base model.

### 5.2 Privacy-Preserving Aggregation

**Current State**: No specific work on differential privacy applied
to SSM gradients was found. However, general FL privacy mechanisms
apply:

- Each client adds Gaussian/Laplace noise to gradients before
  sharing, scaled by privacy parameter epsilon
- FedSA-LoRA-DP (2025): Selective low-rank adaptation with
  differential privacy -- applicable to SSM LoRA adapters
- Communication efficiency: Only transmit LoRA delta weights
  (<1% of parameters) rather than full model

**For Signet**: The federated signal consists of anonymous adapter
weight deltas. With LoRA rank 8-16, the per-user communication
payload is ~150-1500KB per update. Differential privacy noise can
be added at the client level before transmission.

---

## 6. Sleep/Consolidation-Based Learning

### 6.1 SleepGate (Xie, 2026)

**Paper**: "Learning to Forget: Sleep-Inspired Memory Consolidation
for Resolving Proactive Interference in Large Language Models"
**Source**: https://arxiv.org/abs/2603.14517

**Three Consolidation Modules**:

1. **Conflict-Aware Temporal Tagger**: Augments KV cache entries with
   semantic signature vectors (d_s=64) and binary superseded flags.
   Uses cosine similarity > delta=0.85 to detect when newer entries
   override older ones. LSH for O(1) amortized cost.

2. **Forgetting Gate Network**: 2-layer MLP assigning retention scores
   (0-1). Feature vector = key/value content + relative position +
   semantic signatures + cumulative attention. Thresholds:
   >= 0.7 keep, 0.3-0.7 compress, < 0.3 evict. Soft attention
   biasing: b_i = beta * log(max(r_i, epsilon)), where beta=5
   gives ~-23 bias for r_i=0.01 (effectively zero attention weight).

3. **Consolidation Module**: Clusters compress-marked entries by
   semantic similarity. Produces consolidated KV pairs via
   recency-biased attention with learned query vector.
   Compression ratio: |S_m|:1 per cluster.

**Sleep Triggers**:
- Attention entropy > running mean + 1.5 std deviations
- Conflict density > 40% of cache entries superseded
- Fallback: periodic every 128 tokens

**Results**:
| PI Depth | SleepGate | Best Baseline | Factor |
|----------|-----------|--------------|--------|
| n=2      | 99.0%     | 18.0%        | 5.5x   |
| n=5      | 99.5%     | 10.0%        | 10x    |
| n=10     | 97.0%     | 6.0%         | 16x    |
| n=15     | 73.5%     | --           | --     |
| n=30     | 16.5%     | 5.5%         | 3x     |

**Memory Efficiency**: Steady-state bounded by
|C'| <= N / (f_e + (1 - f_e)(1 - 1/c)). With eviction rate f_e=0.3
and compression c=4: 1.67x reduction vs unbounded growth.

**Theoretical**: Reduces effective PI horizon from O(n) to O(log n)
under >= 90% gate accuracy.

**Signet Relevance**: The three-module architecture maps directly to
Signet's memory lifecycle:
- Temporal tagger = memory deduplication/supersession detection
- Forgetting gate = retention decay scoring
- Consolidation = session summary compression

The attention entropy trigger is analogous to Signet detecting when
retrieved memories are becoming uniformly unhelpful (agent "doesn't
know where to look").

### 6.2 Wake-Sleep Consolidated Learning (WSCL, 2024)

**Paper**: "Wake-Sleep Consolidated Learning"
**Source**: https://arxiv.org/abs/2401.08623

**Three Phases**:
1. **Wake**: Process sensory input, dynamic parameter freezing for
   stability, store episodic memories in short-term temporary memory
2. **NREM Sleep**: Synaptic consolidation via replay from both
   short-term and long-term memory. Plasticity mechanism strengthens
   important connections, weakens unimportant ones.
3. **REM Sleep**: Encounter previously-unseen realistic sensory
   experiences. "Dreaming activation" explores potential feature
   space, preparing synapses for future knowledge.

**Results**: Superior performance on CIFAR-10, Tiny-ImageNet,
FG-ImageNet for continual visual classification.

**Signet Relevance**: The three-phase model maps to Signet's daemon
lifecycle:
- Wake = active session processing (memory extraction, prediction)
- NREM = idle period consolidation (memory graph maintenance,
  deduplication, summary generation)
- REM = potential future feature: synthetic interaction generation
  to explore prediction space

### 6.3 Wake-Sleep Energy Based Models (CVPR 2024)

**Source**: CVPR 2024 Workshop on Continual Learning in Computer Vision

**Architecture**: Short wake phases followed by long sleep phases.
Wake phase minimizes free energy of correct solutions. Sleep phase
minimizes free energy of entire system contrastively, pushing
incorrect solutions further away.

**Key Insight**: Orthogonality between sequential task vectors and
flatness of optimized energy surfaces guide continual learning.

### 6.4 Expansion Span: Combining Fading Memory and Retrieval

**Paper**: "Expansion Span: Combining Fading Memory and Retrieval
in Hybrid State Space Models"
**Source**: https://arxiv.org/abs/2412.13328

**Problem**: SSM memory fades exponentially. Attention is eidetic
but finite span. Hybrids still can't recall the distant past.

**Solution**: Reserve a fraction of the Attention context for tokens
retrieved from arbitrarily distant past ("expansion span").
Span-Expanded Attention (SE-Attn) allocates based on relevancy
rather than recency. HyLoRA extends LoRA for efficient adaptation
of hybrid models on long spans.

**Signet Relevance**: Directly analogous to Signet's hybrid search.
The SSM provides compressed "sense" of the user; the retrieval system
(vector + keyword search) provides exact recall of distant memories
when needed.

---

## 7. Warm-Starting and Transfer Learning for SSMs

### 7.1 HiPPO Initialization vs Alternatives

**HiPPO**: Projects functions onto polynomial bases. Stores Legendre
coefficients of input history. Designed for long-range dependencies.

**Autocorrelation-Based Initialization** (ICLR 2025):
- Data-dependent: set timescale Delta proportional to 1/sqrt(L * lambda_max)
  where lambda_max is the max eigenvalue of input autocorrelation
- Real part of eigenvalues: zero initialization prevents memory
  degradation (vs HiPPO's fixed negative values)
- Imaginary part: well-separated values produce better-conditioned
  optimization (separation > 2.3 sufficient)
- **Source**: https://arxiv.org/abs/2411.19455

**PTD (Perturb-Then-Diagonalize)**: Pseudospectral theory for
approximate diagonalization. Stronger convergence to HiPPO than
standard S4D/S5 initializations.

**For Signet's Predictor**: Initialize with autocorrelation-aware
scheme matched to actual memory access patterns. Different users
will have different temporal autocorrelation profiles.

### 7.2 Cross-Architecture Distillation: Transformer to SSM

**MOHAWK Framework** (2024):
- Views both Transformers and SSMs as applying mixing matrices
- Progressive distillation: match mixing matrices -> hidden units ->
  end-to-end predictions
- Phi-Mamba distilled from Phi-1.5 using only 3B tokens (<1% of
  typical training data)
- **Source**: https://arxiv.org/abs/2408.10189

**Mamba in the Llama** (NeurIPS 2024):
- Replaces Transformer attention heads with fine-tuned linear RNN
  layers, keeping MLP layers frozen
- Multi-stage distillation: progressive replacement (every 2 layers,
  then every 4)
- 8x80G A100 GPUs, 3-4 days to reproduce
- Hybrid (1/4 attention) matches original on chat benchmarks
- 300+ tokens/second throughput for 7B Mamba model
- Natural length extrapolation: perfect needle-in-haystack at 20x
  distillation length
- **Source**: https://arxiv.org/abs/2408.15237

**Llamba** (2025): MOHAWK-based distillation achieving results with
<0.1% of typical training data. Replaces Llama self-attention with
Mamba-2 layers.

### 7.3 Mamba-3: Latest SSM Architecture (ICLR 2026)

**Paper**: "Mamba-3: Improved Sequence Modeling using State Space
Principles"
**Source**: https://arxiv.org/abs/2603.15569

**Three Innovations**:
1. **Exponential-Trapezoidal Discretization**: Second-order accurate
   (vs first-order Euler). Removes the causal convolution entirely.
2. **Complex-Valued SSMs**: Data-dependent RoPE for state tracking.
   Parity task: Mamba-2 ~0.90% -> Mamba-3 100%.
3. **MIMO Formulation**: Input projected to matrix, state via matrix
   product. Increases compute-per-byte, pushing to compute-bound.

**Results** (1.5B scale):
- +0.6pp over Gated DeltaNet; MIMO adds +1.2pp (total +1.8pp)
- Comparable perplexity to Mamba-2 with half the state size
- 4% better than Transformer baseline, 7x faster at long sequences

**Signet Relevance**: Mamba-3's halved state size with maintained
quality means smaller per-user storage. Complex-valued states enable
state tracking tasks impossible with real-valued SSMs.

---

## 8. Synthesis: Architecture for Signet's Adaptive Predictor

Based on this research, the optimal architecture for a per-user
memory predictor that learns continuously would combine:

### 8.1 Base Architecture

**Shared base model**: Small SSM (Mamba-3 architecture, ~1-10M
parameters) pre-trained via federated learning on anonymized
training signals. Initialize with autocorrelation-aware scheme
matched to typical memory access patterns.

**Per-user adapter**: LoRA on linear projections (not SSM matrices)
with rank 8-16. Total per-user storage: ~150KB-1.5MB. Use
Conv1d-tuning (156K params) for memory-constrained devices.

### 8.2 Online Learning Strategy

**Session-level TTT** (inspired by LaCT): Treat each session as a
chunk. At session end, perform one gradient step on the per-user
adapter using the session's interaction sequence as input and memory
usefulness feedback as signal. This is the "test-time training"
inner loop.

**Forgetting prevention** (inspired by TTT-E2E + Inf-SSM):
- Dual-path architecture: frozen base SSM + trainable per-user
  adapter
- Apply Inf-SSM's observability subspace regularization during
  adapter updates to prevent behavioral drift
- Fallback: EWC on the per-user adapter weights with session-level
  Fisher information updates

### 8.3 Concept Drift Detection

**Trinity-Controller approach**: Combine three signals:
1. Prediction accuracy EMA (are retrieved memories actually useful?)
2. Session volatility (is the user's behavior pattern changing?)
3. Adapter gradient magnitude (are updates large, suggesting drift?)

When drift detected, temporarily increase the learning rate for
the per-user adapter and expand the number of active discretization
patterns (S6MOD-inspired).

### 8.4 Sleep/Consolidation Cycle

**Idle-period processing** (SleepGate + WSCL inspired):
1. **Consolidation**: Merge redundant memories, update knowledge graph
2. **Adapter refinement**: Replay recent session summaries through
   the predictor with EWC regularization
3. **Drift assessment**: Compare adapter weights to base model weights;
   if divergence exceeds threshold, trigger federated contribution

### 8.5 Federated Aggregation

**Communication**: Per-user LoRA deltas (~150KB-1.5MB), transmitted
with differential privacy noise. Frequency: weekly or on significant
drift events.

**Aggregation**: FedSSM-style temporal tracking of how adapters
evolve across rounds. New installs receive the current shared base
model, achieving warm-start (no cold-start problem).

### 8.6 Hybrid Memory: Compressed State + Exact Retrieval

**Expansion Span principle**: The SSM predictor maintains a compressed
"sense" of the user (fuzzy, abstracted patterns). Signet's existing
hybrid search (vector + keyword) provides exact factual recall.
The predictor scores whether a given memory is likely relevant,
while retrieval handles the actual content.

---

## 9. Open Questions and Risks

1. **TTT computational cost on edge devices**: TTT gradient steps
   add overhead. Session-level chunking (1 step per session) may
   be sufficient but needs benchmarking on consumer hardware.

2. **Adapter interference in multi-task scenarios**: User switches
   between very different projects. S6MOD's class-conditional routing
   suggests per-context adapters, but storage cost grows linearly.

3. **Federated aggregation heterogeneity**: Users have wildly
   different interaction patterns. FedSSM handles heterogeneous data
   but has not been tested at Signet's scale of behavioral diversity.

4. **State size for meaningful personalization**: Mamba-3 achieves
   Mamba-2 quality at half state size. How small can the per-user
   state go before personalization degrades measurably?

5. **Evaluation metrics**: MemoryArena (2026) shows models that
   score near-perfectly on passive recall drop to 40-60% on
   decision-relevant memory use. Signet's predictor must be
   evaluated on downstream task improvement, not just retrieval
   accuracy.

---

## Sources

### Test-Time Training
- [TTT Layers (Sun et al., 2024)](https://arxiv.org/abs/2407.04620)
- [TTT Done Right / LaCT (2025)](https://arxiv.org/abs/2505.23884)
- [End-to-End TTT for Long Context (2025)](https://arxiv.org/abs/2512.23675)
- [SR-TTT: Surprisal-Aware Residual TTT (2026)](https://arxiv.org/abs/2603.06642)
- [TTT4Rec (2024)](https://arxiv.org/abs/2409.19142)
- [TTT Layers in Recommendation (AAAI 2025)](https://arxiv.org/abs/2411.15186)
- [TTT-LM PyTorch](https://github.com/test-time-training/ttt-lm-pytorch)
- [NVIDIA Blog: Reimagining LLM Memory](https://developer.nvidia.com/blog/reimagining-llm-memory-using-context-as-training-data-unlocks-models-that-learn-at-test-time)

### Catastrophic Forgetting in SSMs
- [Mamba-CL: Null-Space Projection](https://arxiv.org/abs/2411.15469)
- [MambaCL: Meta-Learning for Continual Learning](https://arxiv.org/abs/2412.00776)
- [Inf-SSM: Exemplar-Free CL](https://arxiv.org/abs/2505.18604)
- [Regularization-Based CL in DSSMs](https://arxiv.org/abs/2403.10123)
- [SSMs Learn In-Context by Gradient Descent](https://arxiv.org/abs/2410.11687)
- [SSM vs Transformer Tradeoffs (Goomba Lab)](https://goombalab.github.io/blog/2025/tradeoffs/)
- [SpectralGuard: Memory Collapse Attacks](https://arxiv.org/abs/2603.12414)

### Online/Streaming Learning
- [S6MOD: Plug-and-Play SSM for OCL (CVPR 2025)](https://arxiv.org/abs/2412.18177)
- [State Memory Replay (ACL 2024)](https://arxiv.org/abs/2405.17534)
- [Online Deep Learning Survey (2025)](https://link.springer.com/article/10.1007/s10115-025-02351-3)
- [Concept Drift Detection Survey](https://www.mdpi.com/2078-2489/15/12/786)

### Personalization / PEFT
- [PEFT of State Space Models (NeurIPS 2024)](https://arxiv.org/abs/2410.09016)
- [MambaPEFT (2024)](https://arxiv.org/abs/2411.03855)
- [Mamba as Strong Downstream Learner (TMLR 2025)](https://arxiv.org/abs/2406.00209)
- [ProDiaL: Projector-targeted Mamba Tuning (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Ham_Parameter_Efficient_Mamba_Tuning_via_Projector-targeted_Diagonal-centric_Linear_Transformation_CVPR_2025_paper.pdf)
- [FSPO: Few-Shot Personalization (ICLR 2025)](https://arxiv.org/abs/2502.19312)
- [LLM Personalization Survey](https://arxiv.org/abs/2411.00027)

### Federated Learning
- [FedSSM](https://ieeexplore.ieee.org/document/11195747)
- [FedSSM Code](https://github.com/wei-d-zhang/FedSSM)
- [FedSA-LoRA-DP](https://www.mdpi.com/2076-3417/15/24/13102)

### Sleep/Consolidation
- [SleepGate (2026)](https://arxiv.org/abs/2603.14517)
- [Wake-Sleep Consolidated Learning (2024)](https://arxiv.org/abs/2401.08623)
- [Wake-Sleep EBMs (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/html/Singh_Wake-Sleep_Energy_Based_Models_for_Continual_Learning_CVPRW_2024_paper.html)
- [Sleep-like Unsupervised Replay (Nature Comms 2022)](https://www.nature.com/articles/s41467-022-34938-7)
- [Expansion Span: Fading Memory + Retrieval](https://arxiv.org/abs/2412.13328)

### Transfer Learning / Warm-Starting
- [MOHAWK: Transformer to SSM Distillation](https://arxiv.org/abs/2408.10189)
- [Mamba in the Llama (NeurIPS 2024)](https://arxiv.org/abs/2408.15237)
- [Llamba: Scaling Distilled Recurrent Models](https://arxiv.org/abs/2502.14458)
- [Mamba-3 (ICLR 2026)](https://arxiv.org/abs/2603.15569)
- [Autocorrelation-Based SSM Init (ICLR 2025)](https://arxiv.org/abs/2411.19455)
- [How to Train Your HiPPO](https://arxiv.org/abs/2206.12037)
- [Robustifying SSMs (PTD)](https://www.stat.berkeley.edu/~mmahoney/pubs/4131_robustifying_state_space_model.pdf)

### Agent Memory
- [Memory for Autonomous LLM Agents Survey (2026)](https://arxiv.org/abs/2603.07670)
- [Personalized Long-term LLM Interactions](https://arxiv.org/abs/2510.07925)
- [MemAgents Workshop (ICLR 2026)](https://openreview.net/pdf?id=U51WxL382H)
