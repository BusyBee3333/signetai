---
question: What novel, underexplored applications of state-space models exist beyond standard language modeling and recommendation, and how could they inform the design of an agent memory system?
date: 2026-03-20
informed_by:
  - SSM-LITERATURE-REVIEW.md
  - RESEARCH-SSM-INTEGRATION.md
  - ssm-implementations-survey.md
  - VISION.md
---

# Novel SSM Applications for Agent Memory: Deep Research

This document surveys creative, unexpected, and underexplored applications
of state-space models (SSMs) across diverse fields, with specific
attention to how each finding could be applied to Signet's agent memory
system. The standard applications (language modeling, recommendation,
image classification) are covered in the companion literature review.
This document focuses on the weird, the lateral, and the genuinely novel.

Companion documents:
- [Literature Review](SSM-LITERATURE-REVIEW.md) -- foundational papers
- [Implementation Survey](ssm-implementations-survey.md) -- production
  deployments, Rust ecosystem
- [Integration Synthesis](RESEARCH-SSM-INTEGRATION.md) -- how SSMs
  map to Signet's pipeline

---

## 1. Anomaly Detection: SSMs as Behavioral Drift Sensors

### 1.1 KambaAD: Kolmogorov-Arnold + Mamba for Time Series Anomalies

- **Source**: [KambaAD](https://openreview.net/forum?id=XmnTfSX5Az)
  (OpenReview 2025)
- **Key Insight**: Fuses Kolmogorov-Arnold Networks (KAN) with Mamba's
  selective SSM. KAN enforces data consistency for rapid anomaly
  detection; Mamba captures anomalies caused by local variations; the
  internal selection mechanism handles distribution shifts without
  retraining.
- **Signet Application**: The significance gate today uses fixed
  thresholds (minTurns=4, noveltyThreshold=0.4). KambaAD's architecture
  suggests a learned gate that adapts per-user: the SSM hidden state
  encodes the user's normal interaction pattern, and a KAN-style
  consistency check flags sessions that deviate from this norm. When a
  developer suddenly starts asking about DevOps instead of frontend
  work, the anomaly score spikes, triggering more aggressive extraction.
  The selection mechanism handles the gradual drift case too -- when a
  user's focus shifts over weeks, the SSM adapts without triggering
  false alarms.

### 1.2 MambaITD: Insider Threat Detection from Behavioral Sequences

- **Source**: [MambaITD](https://arxiv.org/html/2508.05695) (2025)
- **Key Insight**: Encodes heterogeneous enterprise logs into behavioral
  sequences using a Mapping ID combining action type, device context,
  and temporal segment. A Mamba encoder captures long-range dependencies
  in behavioral + interval sequences. Cross-modal fusion dynamically
  weights behavioral vs temporal information. Adaptive threshold
  optimization per-user by analyzing probability distributions. Achieved
  91.31 F1 on CERT dataset, 1.36x faster than Transformer, 13% lower
  GPU memory.
- **Signet Application**: The behavioral Mapping ID concept maps
  directly to hook events. Each agent interaction can be encoded as
  (action_type={prompt|tool_call|file_edit|search}, context={project|
  session_type}, time_segment={morning|afternoon|sprint|idle}). The
  SSM then learns normal patterns per user. When the pattern deviates --
  a new project starts, the user's rhythm changes, or a session type
  never seen before appears -- the system can trigger proactive memory
  operations (broader search, new arc creation, identity refresh).
  The interval sequence modeling is particularly valuable: Signet could
  detect that a user who normally interacts every few minutes has gone
  silent for hours, suggesting a context switch.

### 1.3 MAAT: Mamba Adaptive Anomaly Transformer

- **Source**: [MAAT](https://arxiv.org/abs/2502.07858) (2025)
- **Key Insight**: Combines sparse attention for long-range dependency
  capture with a Mamba block for local variation sensitivity. An
  adaptive gating mechanism fuses both pathways. A skip connection and
  Gated Attention adaptively weight features from attention and SSM
  branches, improving anomaly localization.
- **Signet Application**: Contradiction detection. The current system
  uses 34 hardcoded antonym pairs. MAAT's dual-pathway architecture
  suggests a contradiction detector where the attention pathway captures
  global semantic relationships (the same entity discussed in different
  ways across sessions) while the SSM pathway captures local temporal
  shifts (this week's statement contradicts last week's). The adaptive
  gating learns which pathway matters for which entity domain.

---

## 2. Code Understanding: SSMs as Session Transcript Parsers

### 2.1 CodeSSM: What State Space Models Learn About Code

- **Source**: [CodeSSM](https://arxiv.org/html/2602.06774) (2025)
- **Key Insight**: SSMs outperform comparable Transformers at capturing
  code properties during pretraining -- superior syntactic capture and
  semantic understanding in deeper layers. Introduces SSM-Interpret, a
  frequency-domain analysis showing SSM convolution kernels function as
  spectral filters (low-pass for long-range, high-pass for short-range).
  Critical finding: fine-tuning causes CodeSSM to lose syntactic
  information through a "spectral shift" emphasizing short-range
  dependencies. Complementary kernel behavior (forward high-pass +
  backward low-pass) captures the most code properties.
- **Signet Application**: Agent session transcripts are saturated with
  code. Today's extraction pipeline treats code blocks as opaque text.
  An SSM pre-trained on code could: (1) identify which code entities in
  a transcript are structurally significant (function signatures vs
  debug print statements), (2) extract semantic relationships between
  code entities (dependency graphs, call hierarchies) that inform the
  knowledge graph, (3) detect when code discussion reveals architecture
  decisions worth remembering vs routine implementation details. The
  spectral filter insight is particularly relevant: different SSM layers
  naturally separate high-frequency detail (variable names, syntax) from
  low-frequency structure (architectural patterns, design decisions) --
  exactly the distinction the extraction pipeline needs.
- **Warning**: The "spectral shift" during fine-tuning means care is
  needed when adapting a code-pretrained SSM to memory-specific tasks.
  The model may lose its long-range code understanding if fine-tuned
  aggressively on short memory classification tasks.

---

## 3. Planning and Decision-Making: SSMs as Anticipatory Engines

### 3.1 Decision Mamba: Multi-Grained SSM for Offline RL

- **Source**: [Decision Mamba](https://arxiv.org/html/2406.05427v2)
  (NeurIPS 2024)
- **Key Insight**: Dual-branch architecture with coarse-grained SSM
  (InterS3M) for global sequential dependencies and fine-grained SSM
  (IntraS3M) for local state-action-return triplets. Progressive
  Self-Evolution Regularization (PSER) blends prior predictions with
  ground truth: a_refined = (1-beta)*a_true + beta*a_prior. Achieves
  83.2% normalized score on D4RL Gym-MuJoCo, surpassing Decision
  Transformer (75.8%).
- **Signet Application**: The dual-granularity architecture maps to
  session-level vs turn-level memory prediction. The coarse SSM models
  the arc of an entire session (what broad topic is being explored),
  while the fine SSM models individual turns (what specific memory will
  be needed for the next prompt). PSER is directly applicable to
  Signet's prediction refinement -- blend the predictor's previous
  guess with ground truth feedback to progressively improve without
  overfitting to noisy training signals. The key insight: offline RL
  trajectories have hierarchical structure. So do agent sessions.

### 3.2 Drama: SSMs as World Models

- **Source**: [Drama](https://arxiv.org/html/2410.08893v4) (ICLR 2025)
- **Key Insight**: First Mamba-based world model for model-based RL.
  Only 7M parameters vs DreamerV3's 200M (28x reduction). Discrete
  VAE encodes observations into latent embeddings; Mamba-2 processes
  concatenated latent states and actions to predict futures. Dynamic
  Frequency-based Sampling (DFS) favors transitions the world model has
  learned sufficiently. Trains on standard laptop hardware. Mamba's
  input-dependent design automatically resets hidden states at episode
  boundaries.
- **Signet Application**: Drama proves SSMs can serve as world models
  with extreme parameter efficiency. For Signet, a 7M-parameter "session
  world model" could predict what the user will do next -- not just what
  memory to retrieve, but what the entire next session fragment will look
  like. The DFS insight is directly applicable: prioritize training on
  session transitions the model has already learned, avoiding wasted
  compute on rare edge cases. The automatic episode boundary detection
  maps to session start/end detection -- no manual reset logic needed
  in the session tracker.

### 3.3 Hierarchical Decision Mamba (HDM)

- **Source**: [HDM](https://www.emergentmind.com/topics/decision-mamba)
  (2025)
- **Key Insight**: Stacked hierarchy with meta-Mamba (high-level
  sub-goal planning) and control-Mamba (low-level action execution).
  The meta level sets objectives; the control level executes within
  those constraints.
- **Signet Application**: Maps directly to Signet's session-arc-epoch
  hierarchy. A meta-SSM operates at the arc level (what broad objective
  is the user pursuing over multiple sessions?) while a control-SSM
  operates at the session level (what specific memories does this
  session need?). This replaces the hardcoded arc threshold (8 sessions)
  and epoch threshold (4 arcs) with learned boundaries.

---

## 4. Multi-Agent Coordination: SSMs for Shared Identity

### 4.1 Multi-Agent Memory as Computer Architecture

- **Source**: [Multi-Agent Memory](https://arxiv.org/html/2603.10062)
  (2025)
- **Key Insight**: Proposes three-layer memory hierarchy mirroring
  classical computer architecture: I/O layer (ingestion), Cache layer
  (fast limited-capacity for immediate reasoning -- KV caches,
  embeddings), Memory layer (persistent retrieval -- vector DB, graph
  DB). Identifies two critical unsolved problems: (1) no principled
  protocol for sharing cached artifacts across agents (analogous to
  multiprocessor cache coherence), (2) under-specified access control
  for multi-agent memory (scope: documents, chunks, records, or trace
  segments?). The most pressing challenge: memory consistency models
  specifying which updates become visible to which agents and in what
  order.
- **Signet Application**: This is Signet's multi-agent future described
  in architectural terms. The paper validates Signet's existing
  hierarchy (working memory cache + SQLite persistent store + graph DB)
  but identifies the exact gap Signet will face: when multiple agents
  share an identity, how do you handle concurrent memory writes? The
  SSM angle: each agent maintains a local SSM hidden state representing
  its session trajectory. Synchronization happens by merging hidden
  states (averaging, weighted combination, or concatenation + reduction)
  rather than merging raw memories. This is dramatically more efficient
  than reconciling individual memory records.

### 4.2 Communicating Plans via World Models

- **Source**: [Plan Communication](https://arxiv.org/html/2508.02912v4)
  (2025)
- **Key Insight**: Instead of communicating raw observations between
  agents, use a compact world model to simulate future states. A Message
  Generation Network compresses plans into compact messages. Engineered
  world model approach shows superior performance, sample efficiency,
  and scalability vs emergent communication.
- **Signet Application**: When multiple Signet agents share identity,
  they don't need to share entire session transcripts or memory deltas.
  Instead, each agent's SSM world model generates a compact "session
  plan summary" -- what was accomplished, what changed, what the user's
  trajectory looked like -- and sends that compressed state vector to
  other agents. This is orders of magnitude more bandwidth-efficient
  than the current approach of syncing raw memories.

---

## 5. NLU Tasks: SSMs for Significance Gating and Classification

### 5.1 Mamba for Long Text Classification

- **Source**: [Mamba Long Text](https://ieeexplore.ieee.org/document/
  10819509/) (2025)
- **Key Insight**: Mamba-130m achieves strong balance between efficiency
  and performance for long text classification, with input-dependent
  parameters enabling content-aware information propagation. Linear
  complexity makes it suitable for classifying very long documents.
- **Signet Application**: Session transcripts are long documents
  (frequently 20k+ characters). Classification tasks include:
  significance assessment (worth extracting?), topic detection (which
  arc does this session belong to?), sensitivity classification (does
  this contain secrets?). A small Mamba classifier could replace the
  current regex-based significance gate with a learned gate that
  considers the full session content, not just turn count and entity
  overlap.

### 5.2 RankMamba and SSMs as Text Rerankers

- **Source**: [RankMamba](https://arxiv.org/html/2403.18276v3) (2024);
  [SSM Rerankers](https://arxiv.org/abs/2412.14354) (2024)
- **Key Insight**: Mamba achieves competitive document ranking with
  transformers at similar model sizes. The Mamba Retriever excels in
  inference speed with linear time scaling, particularly suited for
  long-text retrieval. Mamba-2 outperforms Mamba-1 in both performance
  and efficiency. Current practical limitation: training throughput
  lags behind flash attention-optimized transformers.
- **Signet Application**: Memory retrieval reranking. The current
  pipeline does alpha-blended BM25 + vector (0.7/0.3) with a top-5
  cutoff. A Mamba reranker could take the top-20 candidate memories and
  rerank them using the full session context as a query, with linear
  time complexity enabling real-time reranking even during streaming.
  The reranker maintains a hidden state across the session, so later
  reranking queries benefit from the accumulated session context --
  something the current stateless per-turn approach cannot do.

### 5.3 ss-Mamba: Semantic-Spline Selective SSM

- **Source**: [ss-Mamba](https://arxiv.org/abs/2506.14802) (2025)
- **Key Insight**: Integrates semantic-aware embeddings within a
  selective state-space framework. Enhances forecasting by making the
  SSM aware of semantic context, not just numerical patterns.
- **Signet Application**: Memory embeddings today are static Nomic
  vectors computed once at storage time. An ss-Mamba-style approach
  could produce context-dependent embeddings that change meaning based
  on the current session's semantic context. "Python" in the context of
  a snake discussion vs a programming discussion would produce different
  effective embeddings, improving retrieval precision without
  re-embedding the entire memory store.

---

## 6. Event Stream Processing: SSMs for Hook Event Pipelines

### 6.1 SSMs for Complex Event Detection in CPS-IoT

- **Source**: [CPS-IoT Foundation](https://arxiv.org/html/2503.12282)
  (2025)
- **Key Insight**: Mamba outperforms LSTM, TCN, causal Transformers,
  and neurosymbolic FSMs for online complex event detection from sensor
  traces. A 12-block Mamba (1.8M params, hidden=128, state=64) achieves
  F1~0.92 on standard data and 0.75 on 30-minute out-of-distribution
  sequences (trained on 5-minute sequences only). Key: Mamba generalizes
  to sequences 6x longer than training data.
- **Signet Application**: The daemon's hook pipeline is a complex event
  stream: session_start, user_prompt, tool_call, file_edit,
  session_end, interleaved across multiple concurrent sessions. Today
  each hook is processed independently. An SSM event detector could
  recognize complex event patterns spanning multiple hooks: "the user
  started a session, asked about X, edited file Y, then asked about Z"
  is a pattern that should trigger specific memory operations. The 6x
  generalization is critical -- the model can learn from short sessions
  and generalize to marathon sessions.

### 6.2 Compute-in-Memory SSM Implementation

- **Source**: [CiM SSM](https://www.nature.com/articles/s41467-025-
  68227-w) (Nature Communications 2025)
- **Key Insight**: Implements SSMs in energy-efficient compute-in-memory
  hardware for event-driven processing. Event-SSM requires 1.68 GFLOPs
  for DVS128 Gesture, vs ResNet-18's 104.28 GFLOPs (62x reduction).
  Processing is truly event-driven: computation only occurs when
  state-changing events arrive, with idle periods consuming near-zero
  power.
- **Signet Application**: Validates the thesis that SSMs can be
  event-driven rather than clock-driven. Signet's hook pipeline is
  inherently event-driven -- hooks fire when the user does something,
  not on a fixed schedule. An SSM that only computes when hooks arrive
  (rather than continuously sampling) would be maximally efficient.
  This architecture means the SSM sidecar could sit at near-zero CPU
  when the user is idle, waking only when hooks fire.

---

## 7. Hierarchical Reasoning: Multi-Scale Temporal Memory

### 7.1 MS-SSM: Multi-Scale State Space Model

- **Source**: [MS-SSM](https://openreview.net/forum?id=gfm4W1g7NH)
  (ICLR 2025)
- **Key Insight**: A multi-resolution framework that represents sequence
  dynamics across multiple levels of detail simultaneously, capturing
  both fine-grained high-frequency patterns and coarse low-frequency
  trends.
- **Signet Application**: Directly maps to Signet's temporal hierarchy:
  turn-level (what to retrieve now), session-level (arc tracking),
  arc-level (project tracking), epoch-level (identity evolution). Today
  these are handled by separate hardcoded systems (session tracker, arc
  summarizer, epoch condensation). A multi-scale SSM processes all four
  resolutions simultaneously in a single model, with learned
  interactions between scales. When a turn-level anomaly (unusual
  query) propagates up to trigger an arc-level state change (new
  project detected), this happens through learned dynamics, not
  hardcoded thresholds.

### 7.2 Hierarchical Spatio-Temporal SSM for fMRI

- **Source**: [HSTSSM](https://link.springer.com/chapter/10.1007/978-3-
  031-90252-9_6) (2025)
- **Key Insight**: Processes spatial and temporal information separately
  via hierarchical Mamba-based encoders, then fuses them. Discovers
  neurological biomarkers by decomposing spatio-temporal dynamics.
- **Signet Application**: The spatial/temporal decomposition applies
  to memory in a non-obvious way. "Spatial" in Signet's context is the
  knowledge graph topology (which entities are connected, how). "Temporal"
  is the sequence of interactions. An SSM that processes graph structure
  and temporal dynamics separately, then fuses them, could answer
  questions the current system cannot: "Given the entity graph
  neighborhood of X AND the recent temporal trajectory of the session,
  what memory is most relevant?" -- combining structural and temporal
  reasoning in a principled way rather than ad-hoc alpha blending.

---

## 8. Causal Inference: SSMs for Understanding "Why"

### 8.1 CausalMamba: Causal Discovery from Temporal Sequences

- **Source**: [CausalMamba](https://arxiv.org/html/2511.16191) (2025)
- **Key Insight**: Unifies Mamba sequence encoding, GCN graph encoding,
  and differentiable causal discovery (NOTEARS). Mamba encoder (2
  layers, 128-dim hidden) captures sequential content dependencies; GCN
  encodes structural relationships; NOTEARS-based learner discovers
  weighted adjacency matrices representing causal relationships. Feature
  fusion: H = H_seq + 0.3*H_graph. L1 sparsity + acyclicity constraint
  ensures interpretable DAGs. Intervention simulation validates causal
  structure.
- **Signet Application**: The most creative application in this
  document. CausalMamba's architecture could discover WHY certain
  memories become relevant. Instead of just tracking "memory X was
  retrieved when user asked Y," a CausalMamba-style model discovers
  causal chains: "because the user changed to project Z, memory X about
  framework W became relevant, which caused retrieval of memory V about
  configuration." This transforms the memory system from reactive
  (retrieve what matches) to explanatory (understand why things match).
  The causal DAG could also inform memory retention: memories that are
  causal ancestors of frequently-retrieved memories should decay slower,
  even if they themselves are rarely retrieved directly.

### 8.2 State-Aware Causal Inference

- **Source**: [Observational Causality](https://www.nature.com/articles/
  s42005-025-02447-w) (Communications Physics 2025)
- **Key Insight**: Quantifies causality as information gain about future
  states. Can characterize causal influence as a function of system
  state and distinguish between redundant and synergistic interactions.
- **Signet Application**: Information-gain-based causality maps to
  memory importance. A memory's importance is not just its similarity
  to the current query -- it is the information gain it provides about
  the user's future trajectory. A memory that, when retrieved, changes
  the prediction of what the user will do next is causally important.
  A memory that, when retrieved, doesn't change the prediction is
  redundant and can decay faster.

---

## 9. SSMs as Embedding Engines: Beyond Static Vectors

### 9.1 The Hybrid Attention-SSM Complementarity Finding

- **Source**: [Hybrid Architecture Survey](https://www.askaibrain.com/en/
  posts/end-of-transformers-hybrids-attention-state-space-2025) (2025);
  [Jamba](https://proceedings.iclr.cc/paper_files/paper/2025/file/
  a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf) (ICLR 2025)
- **Key Insight**: A 2025 ablation study on Jamba found that removing
  attention layers causes retrieval accuracy to drop to 0%. SSM layers
  alone contribute nothing to associative recall. Pure SSMs fail on
  copy and recall tasks despite excelling at long sequences. Hybrids
  use Mamba for bulk sequence processing (7 of 8 layers) and attention
  for precision recall (1 of 8 layers).
- **Signet Application**: This is a critical design constraint. SSMs
  alone cannot replace the vector similarity search that Signet relies
  on for memory retrieval. The architecture MUST be hybrid: SSM for
  temporal state tracking, sequence compression, and pattern detection;
  attention (or vector similarity) for precise associative recall. The
  implication: the SSM does not replace Nomic embeddings. It augments
  them with temporal context. The SSM hidden state conditions the
  retrieval query, but the retrieval itself uses attention-like
  mechanisms (cosine similarity, cross-attention).

### 9.2 GrassNet: SSMs for Graph Spectral Filtering

- **Source**: [GrassNet](https://arxiv.org/abs/2408.08583) (2025,
  Pattern Recognition 2026)
- **Key Insight**: First use of SSMs to design GNN spectral filters.
  SSMs model correlations of graph signals at different frequencies,
  deriving unique rectifications for each frequency in the graph
  spectrum. Overcomes polynomial filter limitations (low-order
  truncation, identical treatment of numerically close frequencies).
  Superior performance on 9 benchmarks.
- **Signet Application**: The knowledge graph is a graph. GrassNet
  suggests using an SSM to learn spectral filters over the entity graph,
  producing entity embeddings that capture graph-structural information
  at multiple frequency scales. Low-frequency filters capture global
  community structure (which entities cluster together), while high-
  frequency filters capture local neighborhood variation (which entity
  is anomalous in its neighborhood). This replaces the current static
  graph traversal with learned graph reasoning that adapts to the
  graph's spectral properties.

### 9.3 DyGMamba: Dynamic Graph Embeddings

- **Source**: [DyGMamba](https://arxiv.org/abs/2408.04713) (2025);
  [DyG-Mamba](https://arxiv.org/abs/2408.06966) (NeurIPS 2025)
- **Key Insight**: Two-level SSM for continuous-time dynamic graphs.
  Node-level SSM encodes historical interactions per entity; time-level
  SSM exploits temporal patterns to dynamically select critical
  information from interaction history. DyG-Mamba treats irregular
  timespans between events as control signals, allowing the model to
  dynamically adjust forgetting based on inter-event intervals.
- **Signet Application**: The knowledge graph is not static -- entities
  gain and lose relevance over time, relationships form and dissolve.
  DyGMamba's architecture maps to: (1) per-entity SSM that tracks each
  entity's interaction history (when was it mentioned, in what context,
  how has its usage evolved), (2) a global temporal SSM that identifies
  which entities are currently "hot" based on cross-entity temporal
  patterns. The inter-event interval as control signal is particularly
  elegant: a long gap between entity mentions should increase forgetting
  (the entity is fading from relevance), while a burst of mentions
  should strengthen retention. This replaces the hardcoded 0.95^ageDays
  with learned, entity-specific decay dynamics.

---

## 10. Edge Hardware: SSMs on User Machines Without GPUs

### 10.1 BrainChip TENN: Temporal Event-Based Neural Networks

- **Source**: [BrainChip TENN](https://brainchip.com/temporal-event-
  based-neural-networks-a-new-approach-to-temporal-processing/) (2025);
  [Akida Cloud](https://www.businesswire.com/news/home/20250805783156/en/)
  (2025)
- **Key Insight**: TENNs build on SSM architecture with event-driven
  processing -- computation only happens when state-changing events
  arrive, skipping periods of no change. Combines spatial and temporal
  convolutions for fewer parameters and MACs per inference than
  traditional networks. Operates in temporal convolution mode (training)
  or recurrent mode (inference). Akida Pico FPGA Cloud launched Feb
  2026, AKD2500 custom silicon targeting Q3 2026 (TSMC 12nm). Power
  consumption in tens of milliwatts for video object detection.
- **Signet Application**: TENN's event-driven processing is the ideal
  architecture for Signet's sidecar. The SSM should not run continuously
  -- it should activate only when hooks fire (agent interaction events).
  Between events, power consumption is near-zero. The TENN approach of
  dual-mode operation (temporal convolution for training parallelism,
  recurrent for inference efficiency) maps to Signet's needs: train the
  model offline using accumulated session data (parallel), then deploy
  as a recurrent model that processes one hook at a time (sequential).
  The Akida FPGA Cloud could enable Signet to train models remotely
  on neuromorphic hardware and download the trained model for local
  recurrent inference on CPU.

### 10.2 Mamba-X: End-to-End SSM Accelerator for Edge

- **Source**: [Mamba-X](https://arxiv.org/html/2508.02977v1) (2025)
- **Key Insight**: Hybrid H2 quantization tailored to SSM data
  distributions, quantizing weights and activations to INT8. Addresses
  memory constraints of edge devices specifically.
- **Signet Application**: INT8 quantization of the SSM sidecar would
  reduce memory footprint from ~40MB (FP32, 10M params) to ~10MB
  (INT8). Combined with Mamba's already-small parameter counts (Drama
  achieves world modeling at 7M params), a quantized Signet SSM could
  fit in 7-10MB of RAM and run on any modern CPU without GPU.

### 10.3 FastMamba: Hadamard-Quantized SSMs

- **Source**: [FastMamba](https://openreview.net/pdf/bd4cfd9e6528985d
  fbf397995a38e896464faa3b.pdf) (2025)
- **Key Insight**: 8-bit quantization via Hadamard transforms for outlier
  mitigation. 68.8x speedup over CPU, 8.9x over GPU for Mamba2-130M.
  6x higher energy efficiency than RTX 3090. Power-of-two scaling for
  SSM and convolution blocks.
- **Signet Application**: 68.8x CPU speedup means a Mamba2-class
  model that takes 1 second per inference on baseline CPU would take
  14.5ms with FastMamba quantization. For Signet's per-hook processing,
  14.5ms is well within the budget (hooks currently allow up to 500ms).
  The energy efficiency matters for laptop users -- the SSM sidecar
  should not drain battery.

### 10.4 Fully Quantized Mamba in 1.58 Bits

- **Source**: [1.58-bit Mamba](https://aclanthology.org/2025.coling-
  main.316.pdf) (COLING 2025)
- **Key Insight**: Extreme quantization to ternary weights (-1, 0, 1)
  while maintaining usable performance. Replaces all multiplications
  with additions/subtractions.
- **Signet Application**: A 1.58-bit SSM sidecar at 10M parameters
  would occupy ~2MB. This is small enough to ship as a bundled binary
  inside the signetai npm package. No separate download, no model
  registry, no GPU detection. The model literally fits inside the CLI
  bundle. Performance would degrade from full precision, but for
  tasks like significance gating and importance decay (where precision
  matters less than speed), ternary weights may be sufficient.

---

## 11. Continual Learning: SSMs That Never Forget

### 11.1 Mamba-CL: Null-Space Projection for Continual Learning

- **Source**: [Mamba-CL](https://arxiv.org/abs/2411.15469) (2024)
- **Key Insight**: Updates SSM parameters only in directions orthogonal
  to the feature subspace of previous tasks. Theoretically guarantees
  that new learning never interferes with old learning. Derives
  constraints on four critical time-invariant parameters within Mamba's
  architecture. Superior results on 4 class-incremental benchmarks.
- **Signet Application**: The SSM sidecar will learn from the user's
  interaction patterns. But users change over time -- new projects, new
  tools, new coding styles. Mamba-CL's orthogonal projection ensures
  that learning new patterns (user adopts Rust after years of TypeScript)
  doesn't destroy the model's knowledge of old patterns (TypeScript
  knowledge stays intact for when the user revisits those projects).
  This is exactly the continual learning guarantee Signet needs:
  the predictor gets better over time without catastrophic forgetting.

### 11.2 MemMamba: Solving SSM Long-Range Forgetting

- **Source**: [MemMamba](https://arxiv.org/abs/2510.03279) (2025)
- **Key Insight**: Proves mathematically that SSM long-range memory
  decays exponentially. Introduces state summarization (inspired by
  human document summarization) + dual attention (cross-layer and
  cross-token) to combat decay while preserving linear complexity.
  48% speedup in inference. Breaks the complexity-memory trade-off.
- **Signet Application**: The exponential decay proof is both a warning
  and an opportunity. Warning: a naive SSM WILL forget early session
  content by the end of a long session. Opportunity: MemMamba's state
  summarization mechanism mirrors Signet's existing session summary
  system. The SSM could produce compressed "memory checkpoints" at
  intervals during long sessions, preventing information loss. The
  cross-layer attention enables memories at different abstraction levels
  (facts vs entities vs arcs) to interact, which the current pipeline
  handles through explicit joins.

### 11.3 MambaCL: Meta-Learning for Continual Learning

- **Source**: [MambaCL](https://arxiv.org/html/2412.00776) (2024)
- **Key Insight**: Treats continual learning as next-token prediction.
  Introduces selectivity regularization bridging SSM selective
  parameters with Transformer attention weights via the duality between
  SSMs and linear attention. During training, ground-truth associations
  guide Mamba's selective behavior via KL divergence loss. Key finding:
  Mamba matches or exceeds Transformer performance with fewer
  parameters, with particular strength on fine-grained tasks and length
  generalization (5x longer sequences than training).
- **Signet Application**: The selectivity regularization concept is
  directly transferable. During training, Signet knows which memories
  were actually useful (via the feedback signal). This ground-truth
  association can guide the SSM's selective gating -- when the SSM sees
  a session that resembles one where memory X was useful, it should
  increase its gate for features related to X. The 5x length
  generalization means a model trained on 10-turn sessions can work
  on 50-turn sessions.

### 11.4 Mamba-FSCIL: Few-Shot Class-Incremental Learning

- **Source**: [Mamba-FSCIL](https://arxiv.org/html/2407.06136v1) (2024)
- **Key Insight**: Dual selective SSM projector with frozen base branch
  + trainable incremental branch. Class-sensitive selective scan with
  suppression loss (keep old patterns quiet) and separation loss
  (make new patterns distinct). Reduces performance drop on base
  classes while learning new classes from minimal examples.
- **Signet Application**: When a user starts a new project, the SSM
  needs to learn new patterns from very few sessions (few-shot). Mamba-
  FSCIL's dual-branch approach keeps the base model frozen (general
  interaction patterns) while training a lightweight branch on the new
  project's patterns. The suppression/separation losses ensure that
  recognizing "user is working on Rust project" doesn't interfere with
  the model's existing ability to recognize "user is working on
  TypeScript project."

---

## 12. Information-Theoretic Foundations: The Theory of SSM Memory

### 12.1 Mathematical Formalism for SSM Memory Compression

- **Source**: [Memory Compression Formalism](https://arxiv.org/html/
  2410.03158v1) (2024)
- **Key Insight**: Applies rate-distortion theory and information
  bottleneck to SSM gating. Key theorem: if gating ensures mutual
  information between compressed state and input exceeds I_min, then
  distortion is bounded by D_max(I_min). Selective gating achieves
  30-60% reduction in hidden state utilization for long sequences
  without proportional performance loss. Fano's inequality provides
  lower bounds on prediction error from compressed states. The gating
  function G(x_t) acts as a dynamic information filter: values near 0
  retain previous state (persistence), values near 1 incorporate new
  input (flow).
- **Signet Application**: This formalizes what Signet's memory system
  does intuitively. Every memory decision is an information compression
  decision: what to keep, what to discard, how much detail to retain.
  The rate-distortion framework provides principled bounds: given a
  target memory budget (say, 500 memories), what is the minimum
  information loss achievable? The gating function's persistence/flow
  duality maps exactly to memory retention: high persistence = long-term
  memory (rarely updated facts), high flow = working memory (frequently
  revised state).

### 12.2 MPS-SSM: Minimal Predictive Sufficiency

- **Source**: [MPS-SSM](https://arxiv.org/html/2508.03158) (2025)
- **Key Insight**: Seeks maximal compression of the past under the
  strict constraint that no predictive capability is lost. Unlike
  information bottleneck (which trades off compression vs prediction),
  MPS achieves BOTH maximal compression AND full predictive power by
  filtering exclusively non-predictive information. Provably converges
  to minimal sufficient statistics. Invariant to non-causal
  perturbations (noise robustness). Sweet spot: clean data needs modest
  regularization; noisy/complex data needs stronger regularization.
- **Signet Application**: This is the theoretical foundation for the
  entire memory system. Signet should not store everything and should
  not discard randomly. It should store the minimal set of memories
  that preserves full predictive power over the user's future needs.
  MPS-SSM provides the mathematical framework: for each memory, ask
  "does this memory improve prediction of future retrieval queries?"
  If not, it can be discarded without loss. If yes, it must be retained.
  The regularization sweet spot insight means Signet should be more
  aggressive about compression in clean, predictable interaction
  patterns, and more conservative in novel/complex scenarios.

---

## 13. Bonus: Forgetting Curves and Bioinspired Memory Dynamics

### 13.1 Human-Like Forgetting Curves in Neural Networks

- **Source**: [Forgetting Curves in DNNs](https://arxiv.org/html/
  2506.12034v2) (2025)
- **Key Insight**: MLPs exhibit Ebbinghaus-like memory decay: sharp
  initial loss followed by stabilization. Recall probability decays
  as power law. When recall drops to 80% of baseline, "review sessions"
  restore it -- with progressively lengthening intervals (4, 10, 31
  epochs), mirroring biological spaced repetition. Repeated
  reinforcement increases peak recall beyond pre-review levels.
- **Signet Application**: Validates that the SSM sidecar's internal
  memory decay will follow biological patterns, not catastrophic
  forgetting. The spaced repetition finding is directly applicable to
  memory maintenance: instead of the current fixed 30-day retention
  window, schedule memory "review" based on the SSM's internal recall
  probability. Memories whose SSM representation is fading get
  re-presented during training at progressively longer intervals. This
  is the neural implementation of VISION.md's "learns what to remember."

### 13.2 FOREVER: Forgetting Curve-Inspired Memory Replay

- **Source**: [FOREVER](https://arxiv.org/html/2601.03938v1) (2026)
- **Key Insight**: Measures model evolution through parameter-space
  distance (L2 norm of parameter changes) rather than step counts.
  Calibrates "model days" from early training dynamics. Schedules
  replay at Ebbinghaus-curve intervals (1, 2, 4, 7, 15 model-days).
  Intensity-aware regularization adjusts replay strength based on
  current learning rate instability. Consistent 1.2% overall
  improvement, 0.9% backward transfer improvement across 0.6B-13B
  parameter scales.
- **Signet Application**: FOREVER's "model-centric time" concept
  solves a real problem. The SSM sidecar's training schedule should not
  be clock-based (retrain every N hours) but model-evolution-based
  (retrain when the model has drifted sufficiently from its last
  checkpoint). During active sessions with many new memories, the model
  evolves rapidly and needs frequent replay. During quiet periods, the
  model barely changes and replay is wasteful. The intensity-aware
  regularization prevents the model from over-adapting to a single
  intense session while remaining plastic for gradual evolution.

---

## 14. Bonus: Multimodal Fusion and Cross-Modal Memory

### 14.1 Mixture-of-Mamba: Modality-Aware Sparsity

- **Source**: [Mixture-of-Mamba](https://openreview.net/forum?id=
  Valt8gMdfl) (2025)
- **Key Insight**: Modality-specific parameterization of the Mamba block
  enables efficient multi-modal processing. Matches image loss at 35%
  of training FLOPs; matches speech loss at 25% of FLOPs. Each modality
  gets its own sparse pathway through the SSM.
- **Signet Application**: Agent interactions are multimodal: text
  conversations, code blocks, file paths, URLs, tool outputs, error
  messages, screenshots (via browser extensions). Mixture-of-Mamba
  suggests that the SSM should have modality-specific pathways for
  different memory types. A code memory should activate different SSM
  parameters than a conversational memory. This reduces interference
  between modalities and improves efficiency: code-heavy sessions route
  through code-specific parameters, while planning sessions route
  through planning-specific parameters.

### 14.2 AlignMamba: Cross-Modal Alignment in SSMs

- **Source**: [AlignMamba](https://openaccess.thecvf.com/content/
  CVPR2025/papers/) (CVPR 2025)
- **Key Insight**: Injects cross-modal alignment signals -- both
  token-level (local) and distributional (global) -- before the Mamba
  backbone processes the fused representation. Maintains linear
  computational complexity.
- **Signet Application**: When a memory contains both text and code
  (as most extracted memories do), AlignMamba's approach suggests
  aligning the text and code representations before feeding them to
  the SSM. This ensures the SSM sees a coherent multimodal
  representation rather than a concatenated mess. The distributional
  alignment is particularly relevant for detecting when text
  descriptions of code diverge from the actual code behavior (a form
  of contradiction detection).

---

## 15. Bonus: Brain Activity Encoding and Cognitive State

### 15.1 Brain-Mamba: Neural Signal Encoding via SSMs

- **Source**: [Brain-Mamba](https://proceedings.mlr.press/v248/
  behrouz24a.html) (2024)
- **Key Insight**: Attention-free, scalable framework using SSMs to
  encode brain activity across multiple neuroimaging modalities. Two
  modules: MLP for multi-channel integration + S4 for temporal encoding;
  GNN for learning brain region interdependencies. Outperforms all
  baselines across 7 datasets and 3 modalities.
- **Signet Application**: The "brain region interdependency" concept
  maps to entity interdependency in the knowledge graph. Brain-Mamba's
  architecture of encoding temporal dynamics per-channel (per-entity)
  and then learning cross-channel structure (entity relationships) is
  exactly what a graph-aware SSM for Signet's knowledge graph should
  look like. Each entity gets its own temporal encoding, and a GNN
  layer learns which entities are interdependent -- replacing the
  current static graph traversal with dynamic, learned entity
  relationship reasoning.

### 15.2 EEGMamba: Foundation Models for Neural Signals

- **Source**: [EEGMamba](https://www.sciencedirect.com/science/article/
  abs/pii/S0893608025006963) (Neural Networks 2025)
- **Key Insight**: Foundation model for EEG using Mamba encoder,
  achieving SOTA across 6 downstream tasks. Demonstrates that SSMs
  can serve as general-purpose encoders for high-dimensional temporal
  signals, transferring representations across vastly different tasks.
- **Signet Application**: If EEGMamba can encode brain signals across
  6 different tasks with a single pretrained model, a similar SSM
  foundation model could encode agent interaction patterns across
  different pipeline tasks (significance gating, extraction quality
  prediction, retention scoring, contradiction detection) with shared
  representations. Pre-train once on session data, fine-tune for each
  pipeline stage.

---

## 16. Bonus: Test-Time Adaptation and Online Learning

### 16.1 Test-Time Training for Continuous Adaptation

- **Source**: [TTT](https://test-time-training.github.io/e2e.pdf) (2025)
- **Key Insight**: Equips sequence models with hidden states that
  function as lightweight parametric models updated online during
  inference. Contextual information compressed into state matrix S;
  the product Sq generates output. TTT-E2E achieves constant inference
  latency regardless of context length (2.7x faster than full attention
  at 128K context).
- **Signet Application**: The SSM sidecar could update its parameters
  during inference -- not just during training. As each hook fires,
  the model not only produces predictions but also learns from the
  result. If the model predicted that memory X would be relevant but
  the user ignored it, the model adjusts in real-time. This is the
  ultimate expression of "gets sharper the longer you use it" from
  VISION.md -- the model improves within a single session, not just
  between training runs.

---

## Synthesis: The Creative Map

The 10 research areas above, plus 6 bonus areas, converge on a
coherent architecture. Here is how they compose:

### What SSMs Replace (learned dynamics replacing heuristics)

| Current Heuristic | SSM Replacement | Source |
|---|---|---|
| 0.95^ageDays decay | Per-entity learned decay via DyGMamba | Sec 9.3 |
| minTurns=4, novelty=0.4 | Learned significance gate via KambaAD | Sec 1.1 |
| 34 antonym pairs | Dual-pathway contradiction detector via MAAT | Sec 1.3 |
| BM25+vector 0.7/0.3 | Session-context reranker via RankMamba | Sec 5.2 |
| Fixed 30-day retention | Forgetting-curve retention via FOREVER | Sec 13.2 |
| 8-session arc threshold | Learned hierarchy via MS-SSM | Sec 7.1 |
| Static graph traversal | Learned spectral filtering via GrassNet | Sec 9.2 |
| Stateless per-turn search | Session-state search via MemMamba | Sec 11.2 |

### What SSMs Add (capabilities that don't exist today)

| New Capability | Source |
|---|---|
| Behavioral change detection (new project, rhythm change) | Sec 1.2 |
| Code-aware extraction (architecture vs syntax) | Sec 2.1 |
| Next-session prediction (anticipatory retrieval) | Sec 3.2 |
| Causal memory chains (why X became relevant) | Sec 8.1 |
| Multi-agent state synchronization via hidden states | Sec 4.1 |
| Modality-specific memory pathways | Sec 14.1 |
| Test-time adaptation (learn within single session) | Sec 16.1 |
| Minimal predictive sufficient memory (information-theoretic) | Sec 12.2 |

### Design Constraints Discovered

1. **SSMs cannot do retrieval alone** (Sec 9.1). The architecture MUST
   be hybrid: SSM for temporal state + attention/similarity for recall.
2. **SSMs forget exponentially** (Sec 11.2). Long sessions need state
   summarization checkpoints.
3. **Fine-tuning causes spectral shift** (Sec 2.1). Code understanding
   degrades if fine-tuned aggressively on non-code tasks.
4. **Training throughput lags flash attention** (Sec 5.2). SSMs are
   faster at inference but slower to train than optimized transformers.
5. **Quantization to INT8 is practical** (Sec 10.2-10.3). 1.58-bit is
   achievable for coarse tasks.

### The Minimal Viable SSM Stack for Signet

Based on the research above, the smallest useful SSM integration would be:

1. **Event-driven session state SSM** (~2M params): Processes hook
   events, maintains session trajectory, provides temporal context for
   retrieval queries. Based on Drama's architecture (Sec 3.2) with
   event-driven activation (Sec 6.2).

2. **Significance gate SSM** (~1M params): Replaces regex-based
   significance assessment with a learned gate trained on extraction
   outcomes. Based on KambaAD (Sec 1.1) for anomaly-aware gating.

3. **Importance decay SSM** (~2M params): Per-entity learned decay
   replacing 0.95^ageDays. Based on DyGMamba (Sec 9.3) for dynamic
   graph-temporal modeling.

Total: ~5M parameters, ~10MB at INT8, ~2MB at 1.58-bit.
Inference: <15ms per hook on CPU (Sec 10.3).
Architecture: Rust sidecar, same as current predictor.

---

## Research Gaps

The following questions remain unanswered by the literature:

1. **No SSM work on memory contradiction detection.** CausalMamba
   detects causal structure but not semantic contradictions between
   memories. This is novel territory.

2. **No SSM work on belief revision.** The belief revision literature
   (Sec 8) has not been connected to SSM architectures. An SSM that
   maintains a "belief state" and revises it when contradicting evidence
   arrives would be genuinely novel.

3. **No SSM work on agent identity coherence.** Multiple SSM hidden
   states representing the same agent across sessions -- how to merge
   them into a single coherent identity -- is unexplored.

4. **No SSM work on privacy-preserving memory.** SSM hidden states
   compress user data. What information leaks from the hidden state?
   Can differential privacy be applied to SSM state updates?

5. **No SSM work on federated memory learning.** VISION.md describes
   federated learning for the base model. Can SSM weights be aggregated
   across users without exposing individual patterns?

These gaps represent opportunities for Signet to publish novel research.
