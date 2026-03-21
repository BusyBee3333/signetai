---
question: Can state-space models serve as the learned backbone of Signet's memory pipeline, replacing heuristic-driven stages with a unified temporal reasoning engine?
date: 2026-03-20
informed_by:
  - SSM-LITERATURE-REVIEW.md
  - ssm-implementations-survey.md
  - SSM-NOVEL-APPLICATIONS.md
  - SSM-GRAPH-INTERSECTION.md
  - SSM-CONTINUAL-LEARNING-DEEP-DIVE.md
  - SYNTHETIC-DATA-GENERATION.md
  - VISION.md
  - docs/specs/INDEX.md
  - docs/specs/approved/desire-paths-epic.md
  - docs/specs/approved/predictive-memory-scorer.md
  - docs/specs/approved/procedural-memory-plan.md
  - docs/specs/planning/LCM-PATTERNS.md
---

# State-Space Models for Signet: Research Synthesis

This document consolidates findings from twelve parallel research
streams -- four codebase explorations and eight web research surveys --
into a unified picture of how state-space models (SSMs) could transform
Signet's memory system from a collection of heuristics into a learned
temporal engine. It includes hard architectural constraints discovered
through research, a validation contract framework with concrete test
specifications, and a synthetic data pretraining strategy for immediate
validation before any production code is written.

Companion documents:
- [Literature Review](SSM-LITERATURE-REVIEW.md) -- foundational papers,
  small/efficient architectures, memory-specific applications, training
  on personal data
- [Implementation Survey](ssm-implementations-survey.md) -- reference
  implementations, production deployments, Rust ecosystem, deployment
  patterns
- [Novel Applications](SSM-NOVEL-APPLICATIONS.md) -- creative and
  underexplored SSM applications: causal discovery, world models, anomaly
  detection, 1.58-bit quantization, identity coherence
- [Graph Intersection](SSM-GRAPH-INTERSECTION.md) -- SSMs for temporal
  knowledge graphs, graph neural network integration, hierarchical
  multi-scale reasoning, entity salience modeling
- [Continual Learning Deep Dive](SSM-CONTINUAL-LEARNING-DEEP-DIVE.md) --
  test-time training, catastrophic forgetting, per-user adapters,
  federated learning, sleep-inspired consolidation
- [Synthetic Data Generation](SYNTHETIC-DATA-GENERATION.md) -- synthetic
  benchmark design, canary patterns, curriculum learning, data
  augmentation, TSTR validation protocol

---

## The Thesis

VISION.md describes Signet's endgame:

> "Signet doesn't just remember -- it learns what to remember. A neural
> network unique to each user, trained on their own interaction patterns,
> that gets sharper the longer you use it."

The current pipeline approximates this with a grab bag of heuristics:
FTS5 keyword search, cosine similarity, exponential decay (`importance *
0.95^ageDays`), hardcoded antonym pairs, fixed traversal limits, and a
cross-attention scorer sidecar with three critical bugs. Each stage was
built independently because no single model could hold state across time.

An SSM *is* that model. State-space models maintain a compressed hidden
state that evolves with each input, learning what to retain and what to
forget. They process sequences in linear time with constant memory. They
handle irregular temporal sampling natively. And at 2-10M parameters,
they fit inside a Rust sidecar binary under 15MB.

The proposal is not "add SSMs in a few places." It is: **SSMs become the
learned backbone that every pipeline stage consults.** The graph stays.
The database stays. FTS5 stays. But the temporal reasoning -- what's
important now, what will be important next, what should decay, what
contradicts what -- moves from hardcoded rules to a model that learns
from the user's own interaction patterns.

---

## Current Pipeline: The Heuristic Map

The memory pipeline spans ~19,800 LOC across 52+ files in
`packages/daemon/src/pipeline/`. Here is what each stage does today and
where it approximates temporal dynamics with static rules:

### Significance Gate (`significance-gate.ts`)
- **What it does**: Decides whether a session is worth extracting
- **Heuristics**: Turn count regex, substring entity matching, token-set
  novelty comparison against last 5 transcripts
- **Hardcoded thresholds**: minTurns=4, minEntityOverlap=2,
  noveltyThreshold=0.4
- **Problem**: `lowerTranscript.includes(row.name.toLowerCase())` fails
  on postgres/PostgreSQL, plurals, abbreviations. Same thresholds for all
  users regardless of communication style.

### Extraction (`extraction.ts`)
- **What it does**: LLM call to atomize transcript into facts + entities
- **Heuristics**: Regex-based JSON parsing with fence stripping, trailing
  comma fixing, balanced-brace fallback
- **Hardcoded limits**: MAX_FACTS=20, MAX_ENTITIES=15,
  MAX_FACT_LENGTH=2000
- **Problem**: No learning from extraction failures. Every session
  repeats the same brittle parsing. No confidence calibration.

### Decision Engine (`decision.ts`)
- **What it does**: For each fact, searches existing memories and decides
  ADD/UPDATE/DELETE/SKIP via LLM
- **Heuristics**: Alpha-blended BM25 + vector (0.7/0.3), top-5 candidates
- **Problem**: Binary search cutoff (CANDIDATE_LIMIT=5). No temporal
  reasoning about how technology decisions evolve. No learning from
  user's implicit decision preferences.

### Contradiction Detection (`contradiction.ts`)
- **What it does**: Fast path (syntactic heuristics) + slow path (LLM)
- **Heuristics**: 34 hardcoded antonym pairs, temporal marker regex,
  negation polarity detection
- **Problem**: Misses domain-specific contradictions ("REST API" vs
  "GraphQL endpoint"). No context-aware contradiction (same words may or
  may not contradict depending on which entity they describe).

### Importance Scoring (`hooks.ts:effectiveScore`)
- **What it does**: Ranks memories for injection
- **The entire function**:
  ```typescript
  if (pinned) return 1.0;
  const ageDays = (Date.now() - new Date(createdAt).getTime())
    / (1000 * 60 * 60 * 24);
  return importance * 0.95 ** ageDays;
  ```
- **Problem**: Single hardcoded exponential. No access frequency, no
  structural relationships, no temporal context, no entity-awareness.
  Half-life is ~13.5 days regardless of memory type or user behavior.

### Graph Traversal (`graph-traversal.ts`)
- **What it does**: Walks entity -> aspect -> attribute -> dependency
  chains to construct context
- **Heuristics**: maxAspectsPerEntity=10, maxAttributesPerAspect=20,
  maxDependencyHops=10, timeoutMs=500, densityScore weights (0.15, 0.05,
  0.2)
- **Problem**: Query-agnostic expansion. "What does X depend on?" and
  "What are X's properties?" trigger the same traversal. No memory of
  which aspects were hot in prior turns.

### Session Hook Flow (`hooks.ts:handleUserPromptSubmit`)
- **What it does**: On every prompt, runs embedding + hybrid search +
  dedup + top-5 selection
- **Heuristics**: 5-turn sliding window dedup, 500-char budget
- **Problem**: Stateless per-turn. No session trajectory. Each search is
  independent. On 30+ turn sessions, embedding cost alone is 1-2s
  cumulative. No prediction of what will be needed next.

### Predictive Scorer (Rust sidecar, `predictor/`)
- **What it does**: Cross-attention scorer that ranks candidate memories
  using 768-dim embeddings + 17-dim feature vectors
- **Architecture**: CrossAttentionScorer with 7 learned parameter groups,
  ~1.1M parameters
- **Critical bugs**: (1) Feature vectors 4-element but sidecar expects
  17 (silent failure); (2) Cold start exits on training pair count
  instead of session count; (3) Stale traversal cache never invalidated.
- **Problem even without bugs**: Static feature engineering. Adding new
  temporal signals requires schema changes + retraining. The 17-dim
  vector is a hand-built approximation of what an SSM learns natively.

### Retention (`retention-worker.ts`)
- **What it does**: Purges old data on fixed schedules
- **Heuristics**: Hard-delete after 30 days, history events after 180
  days, completed jobs after 14 days
- **Problem**: Fixed windows. No structural importance awareness. A
  memory linked to 50 entities decays at the same rate as an orphan.

### Summarization (`summary-worker.ts`, `summary-condensation.ts`)
- **What it does**: Session -> arc -> epoch hierarchical summarization
- **Heuristics**: Fixed chunk size (20k chars), arc threshold (8
  sessions), epoch threshold (4 arcs)
- **Problem**: All sessions treated equally. No learned priority for what
  to preserve across condensation levels.

### Continuity State (`continuity-state.ts`)
- **What it does**: Accumulates per-session state (queries, remembers,
  focal entities)
- **Storage**: Arrays of up to 20 queries, 10 remembers, 10 snippets.
  ~2KB per checkpoint.
- **Problem**: No learned compression. All state equally weighted. No
  prediction of what matters for recovery.

---

## What the Literature Says

### The Core Architecture: Mamba

The Mamba family (Gu & Dao, 2023-2026) provides the theoretical and
practical foundation. Key properties:

- **Selective state**: Parameters (A, B, C, delta) are input-dependent,
  allowing the model to choose what to remember vs forget per timestep
- **Linear time, constant memory**: Processes sequences of any length
  with O(N) computation and O(1) memory per step
- **Hardware-aware**: Parallel scan for training, recurrent mode for
  inference. No KV cache.
- **Mamba-3** (March 16, 2026 -- 4 days ago): Inference-first design.
  Complex-valued state (models cyclical patterns like daily/weekly
  rhythms). MIMO formulation (better predictions at same memory cost).
  Achieves Mamba-2 quality with half the state size.

### The Brain-Database Analogy

Albert Gu's framing (July 2025): Transformers are databases (store
everything, retrieve exactly). SSMs are brains (fixed-size memory,
always processing, retain the shape and flow rather than every detail).

Signet already has the database (SQLite + FTS5 + vector search). What it
lacks is the brain -- the compressed temporal state that knows what
matters without looking it up. SSMs provide this. The two complement
each other.

### Proven at Our Scale

- **IBM FlowState**: 9.1M parameters, outperforms models 20x its size
  on temporal prediction. S5-based. Proves sub-10M SSMs are viable.
- **SS4Rec / SSD4Rec**: SSMs predicting "what's relevant next" from user
  interaction sequences. Direct analogy to memory relevance scoring.
- **DyGMamba**: Dual-SSM for temporal knowledge graph link prediction.
  Directly applicable to entity relationship evolution.
- **RankMamba**: Mamba-130m competitive with BERT on document ranking.
- **SleepGate** (March 15, 2026): Three modules for conflict detection,
  selective forgetting, and consolidation. Maps directly to Signet's
  contradiction detection, retention decay, and session summarization.

### Inference Performance

- 4-layer Mamba at d64: **0.38ms per sample** (10.8x faster than
  equivalent transformer)
- Mamba-3 at 16K context: ~7x faster than Llama-3.2-1B
- Sub-10M model quantized with heterogeneous INT8/INT4: <10MB, <1ms
  inference

### Rust Ecosystem

Three Mamba implementations exist in Rust:
- **mamba.rs**: Pure Rust, minimal deps, memory-mapped weights (Apache
  2.0 / MIT)
- **mamba-ssm**: Candle-based, tested on Apple Silicon (MIT)
- **oxidizr/blazr**: Full training + inference stack, supports
  Mamba-2/3 (production-grade)

Plus **web-rwkv** for RWKV-7 inference in Rust with quantization. And
**llama.cpp** supports Mamba via GGUF with custom SSM operators.

The Rust inference path is viable today. Train in PyTorch, export
safetensors, infer in Rust.

### Key Insight: Prediction, Not Retrieval

RecurrentGemma research shows retrieval depends exclusively on attention
layers -- SSMs show zero compensatory retrieval ability. But our use case
is **prediction** (which memories will be relevant) not **retrieval**
(exact lookup of a specific memory). For prediction tasks, pure SSMs
excel. The database handles retrieval. The SSM handles prediction.

---

## The Integration Map

### Where SSMs Replace Heuristics

| Pipeline Stage | Current Heuristic | SSM Replacement | Impact |
|---|---|---|---|
| Significance gate | Turn count + substring match + token novelty | Learned significance scorer | ~20% fewer false negatives/positives |
| Importance scoring | `importance * 0.95^ageDays` | Learned decay with data-dependent lambda | Adaptive per-memory, per-user decay |
| Contradiction detection | 34 antonym pairs + temporal markers | Domain-learned contradiction classifier | ~40% fewer false contradictions |
| Decision engine | LLM call for every fact | SSM pre-filter (skip LLM for high-confidence cases) | ~30% fewer LLM calls |
| Traversal scoring | Fixed density weights | Learned aspect/path scoring biased by session trajectory | Query-adaptive traversal |
| Session context | Stateless per-turn search | SSM hidden state biases focal entity resolution | Proactive context, not reactive search |
| Retention windows | Fixed 30-day hard delete | Structural-importance-aware soft decay | Memories tied to high-mention entities persist |
| Summarization priority | All sessions equal | Learned importance weighting | Better signal retention across condensation |
| Reranking blend | Fixed 70/30 BM25/vector | User-adaptive learned blend | Personalized ranking |
| Continuity checkpoints | Raw arrays (~2KB) | SSM hidden state (~256 bytes) | 8x checkpoint compression |

### Where SSMs Augment (Not Replace)

- **Hybrid search** (BM25 + vector): Keep as-is. SSM biases what gets
  searched, not how search works.
- **Knowledge graph traversal**: Keep the graph. SSM biases which paths
  to explore first.
- **Session tracking/bypass**: Keep the mutex and cleanup. Orthogonal
  concern.
- **Embedding generation**: Keep Nomic Embed. SSM predicts what to embed
  proactively.

### Alignment with Spec Pipeline

| Spec | Status | SSM Fit |
|---|---|---|
| Knowledge Architecture (KA-1 to KA-6) | COMPLETE | SSM traversal scoring builds on KA entity/aspect/attribute schema |
| Predictive Memory Scorer | COMPLETE (bugs) | SSM replaces cross-attention scorer entirely |
| Session Continuity Protocol | COMPLETE | SSM state replaces raw checkpoint arrays |
| Desire Paths (DP-1 to DP-10) | PARTIAL | **Strongest fit.** DP-9/10 (path feedback, path scoring) are inherently SSM problems |
| Procedural Memory (P1-P5) | NOT STARTED | SSM models skill decay and co-usage patterns |
| LCM Foundation Patterns | PLANNING | SSM enables temporal reinforcement (LCM pattern extension) |
| Retroactive Supersession | PLANNING | SSM contradiction classifier replaces heuristic detection |

The desire paths epic is the highest-value integration point. Paths
through the entity graph are sequential state transitions. Reinforcement
comes from feedback. Temporal patterns emerge naturally. This is what
SSMs were designed for.

---

## Proposed Architecture

### The Signet Neural Backbone (SNB)

A single Mamba-based SSM sidecar that serves multiple roles through a
unified hidden state, replacing the current cross-attention predictor.

```
Architecture:  Mamba-3 (MIMO, trapezoidal discretization, complex state)
Parameters:    5-10M (4-8 layers, d_model=128-256, N=32-64)
Precision:     Heterogeneous (A matrix fp16, projections int8)
Binary size:   ~10-15MB (Rust, quantized safetensors)
Inference:     <1ms per event, CPU-only
Training:      PyTorch -> safetensors -> Rust inference
State:         Persistent hidden state saved between sessions
```

### Input Features Per Interaction Event

```
Temporal:
  - time_since_last_event (continuous, irregularly sampled)
  - time_of_day (sin/cos encoded)
  - day_of_week (sin/cos encoded)
  - session_age (time since session start)

Memory access:
  - focal_entity_ids (hashed to fixed-width slots)
  - aspect_ids_touched (hashed)
  - memory_ids_injected (hashed)
  - access_count_delta (how many times each memory was accessed this turn)

Feedback:
  - agent_relevance_signal (from MCP feedback tool, [-1, 1])
  - fts_hit_count (behavioral signal)
  - was_referenced_in_response (implicit positive signal)

Content:
  - query_embedding (768-dim, projected to d_model via learned projection)
  - entity_structural_density (from knowledge graph)
```

### Output Heads (Multi-Task)

From a single forward pass, the SSM produces:

1. **Memory relevance scores**: Which memories to inject for next turn
2. **Aspect bias weights**: Which aspects to prioritize in traversal
3. **Predicted next entities**: For proactive embedding prefetch
4. **Retention adjustment**: Per-memory decay rate modifiers
5. **Significance score**: Whether current session is worth extracting
6. **Contradiction probability**: For each new fact vs existing memories

### Data Flow

```
Session Turn Arrives
       |
       v
  [Embed query] ──────────────────────────────┐
       |                                       |
       v                                       v
  [SSM.step(hidden, features)]          [Hybrid Search]
       |                                       |
       |── aspect_bias ──> [Graph Traversal] <─┘
       |── relevance_scores ──> [Reranking]
       |── predicted_entities ──> [Prefetch Embeddings]
       |── significance ──> [Gate Extraction]
       |── retention_adj ──> [Decay Update]
       |── contradiction_prob ──> [Contradiction Check]
       |
       v
  [Inject Context to Agent]
       |
       v
  [Agent Responds]
       |
       v
  [Collect Feedback] ──> [SSM Training Signal]
```

### The Continuity Loop

The SSM hidden state persists across turns within a session and across
sessions via checkpoint serialization:

```
Session 1, Turn 1: h_0 = init_from_project_defaults()
Session 1, Turn N: h_N = SSM.step(h_{N-1}, features_N)
Session 1, End:    checkpoint(h_N) -> 256 bytes

Session 2, Start:  h_0 = restore_from_checkpoint()
Session 2, Turn 1: h_1 = SSM.step(h_0, features_1)
...
```

This is the missing continuity mechanism. Current checkpoints store raw
arrays of entity IDs and query strings. The SSM state is a learned
compression of what actually matters for predicting next-turn relevance.

### Graceful Degradation

The SSM is advisory, not authoritative:

- **SSM unavailable**: Skip all SSM-derived biases, use baseline
  traversal and scoring. No regression.
- **SSM prediction wrong**: Biased aspects still rank in top-N, just
  with different weights. Worst case is degraded ranking, not broken
  retrieval.
- **Hidden state corruption**: Fall back to project defaults on
  checkpoint load.
- **Inference latency spike**: If >5ms, skip and use baseline.

The SSM never blocks the critical path. It biases it.

---

## Hard Architectural Constraints

Research surfaced several non-negotiable design constraints. Ignoring
any of these would produce a system that fails in production.

### 1. SSMs Cannot Do Associative Recall

Jamba (ICLR 2025) proved removing attention from SSM hybrids drops
retrieval accuracy to 0%. This is a hard ceiling. SSMs compress temporal
patterns but cannot look up specific facts from their state.

**Constraint**: The SSM is a prediction/bias layer only. SQLite + FTS5 +
vector search handle all retrieval. The SSM tells the retrieval system
*where to look*, not *what to find*.

### 2. S4 Outperforms Mamba on Graph Data

GraphSSM (NeurIPS 2024) found that selective Mamba underperforms
non-selective S4 on graph-structured inputs. The selectivity mechanism
doesn't help when processing topology.

**Constraint**: Graph traversal path scoring should use S4 (or a
non-selective variant), not Mamba. Sequential interaction modeling uses
Mamba. Different SSM variants for different tasks.

### 3. nDCG Is Anti-Correlated with Online Reward

KDD 2024 proved nDCG shows -0.91 correlation with actual online reward
when aggregated across sessions. The normalization introduces
inconsistency.

**Constraint**: Use unnormalized DCG for model selection. Report nDCG
for literature comparability only. Never use nDCG to decide whether
the SSM is working.

### 4. LoRA on SSM Parameters Doesn't Work

LoRA on A, B, C, Delta parameters achieves 76.9 GLUE (near-random).
LoRA on linear projections achieves 87.0 (near full fine-tuning at
89.4).

**Constraint**: Per-user adapters target projection matrices only.
SSM state matrices are shared across all users.

### 5. Memories Must Be Injected Chronologically

GenTKG proved temporal ascending order is the only effective fact
arrangement for temporal KG reasoning. Importance-ordered injection
destroys the temporal signal the SSM needs.

**Constraint**: Memory injection order is chronological within each
relevance tier, not sorted by importance score.

### 6. Passive Recall ≠ Agentic Performance

MemoryArena (2026): models scoring near-perfect on passive recall tests
dropped to 40-60% on agentic tasks.

**Constraint**: Evaluation must include end-to-end agent task outcomes,
not just retrieval accuracy.

---

## Validation Contracts

Nothing ships without passing these tests. Tests are written before
any production code. The SSM must prove it works on synthetic data with
known ground truth before touching the real pipeline.

### Phase 0: Synthetic Data Validation

Before any integration, validate the SSM in isolation using synthetic
data with planted ground truth patterns.

#### Canary Pattern Tests

Seven patterns planted in synthetic interaction sequences. The SSM must
find all seven to pass.

| # | Pattern | Description | Pass Criteria |
|---|---------|-------------|---------------|
| 1 | Temporal cycle | Entity X relevant every Monday | Predicted relevance for X on Monday > Thursday by 2x |
| 2 | Recency decay | Memories accessed recently score higher | Learned decay rate within 20% of planted rate |
| 3 | Entity selectivity | Memory Y only relevant when focal entity is Z | Relevance for Y given Z > relevance for Y given W by 3x |
| 4 | Supersession chain | Memory A superseded by B superseded by C | Relevance: C > B > A at all test points |
| 5 | Burst pattern | Cluster of accesses predicts near-future relevance | Recall@10 for post-burst queries > 0.8 |
| 6 | Cross-entity dependency | When entity P active, entity Q's memories become relevant | Co-activation detected with >0.7 correlation |
| 7 | Combined multi-signal | Temporal cycle + entity selectivity + recency | All three signals contribute (ablation: removing any one drops accuracy >10%) |

#### SNR Degradation Curve

Test at signal-to-noise ratios: 30dB, 20dB, 10dB, 0dB, -10dB. Plot
accuracy vs SNR. The model must:
- Maintain >90% canary detection at 20dB
- Maintain >70% canary detection at 10dB
- Degrade gracefully (no cliff-edge failure)
- The noise floor (where detection drops below 50%) must be documented

#### Memorization vs Learning Test (SynTSBench Dual-Error)

For each synthetic sequence, measure:
- MSE_Obs: predictions vs noisy observations
- MSE_True: predictions vs clean ground truth signal

If MSE_True improves while MSE_Obs stays flat, the model is learning
temporal patterns. If both move together, it's memorizing noise. **The
model must show MSE_True < MSE_Obs on all canary patterns.**

#### TSTR Protocol (Train Synthetic, Test Real)

Once real session data exists:
1. Train on synthetic data only, test on real data
2. Train on real data only, test on real data (baseline)
3. Train on synthetic + real, test on real data

Pass criteria: (1) must achieve >60% of (2)'s performance. (3) must
exceed (2).

### Phase 1: Offline Evaluation

Once the SSM is validated on synthetic data, evaluate on historical
session data.

#### Evaluation Protocol

- **Primary**: Global Temporal Split (GTS) with Successive targets.
  Cutoff at 90th percentile timestamp. All sessions before cutoff are
  training data. Each post-cutoff interaction is a test case.
- **Secondary**: Leave-One-Out for literature comparability only.
- **Statistical significance**: Paired bootstrap with BCa confidence
  intervals, minimum 5 random seeds, sign-flip permutation test. Claim
  significance only if BCa lower bound > 0 AND permutation p < 0.05.

#### Metrics Suite

| Metric | Purpose | Target |
|--------|---------|--------|
| HR@5 | Did correct memory appear in top-5? | > baseline + 5% |
| HR@10 | Did correct memory appear in top-10? | > baseline + 3% |
| MRR@10 | Position of first correct memory | > baseline + 0.05 |
| DCG@10 (unnormalized) | Model selection metric | > baseline |
| nDCG@10 | Literature comparability only | Report, don't optimize |
| Precision@K | Fraction of injected memories that were relevant | > 0.6 |
| Recall@K | Fraction of relevant memories that were injected | > 0.5 |
| Latency p50/p99 | Inference time per prediction | p50 < 1ms, p99 < 5ms |

#### Required Ablations

| Ablation | What It Tests |
|----------|---------------|
| No temporal features | Is the SSM using temporal dynamics? |
| No content features | Is the SSM using content similarity? |
| No selectivity (A, B, C fixed) | Is Mamba's selection mechanism helping? |
| State dim N: 4, 16, 32, 64 | Optimal state size for our data |
| Layer count: 1, 2, 3, 4 | Optimal depth |
| Sequence length: <10, 10-50, 50-200 sessions | Length sensitivity |
| Temporal feature only (remove content) | Can SSM predict from time alone? |
| M2Rec FFT validation | Does model respond to daily/weekly frequencies? |

#### Cold-Start Stratified Evaluation

| Stratum | Sessions | Target |
|---------|----------|--------|
| Ice cold | 1-3 sessions | Match baseline (no regression) |
| Warming | 4-10 sessions | Beat baseline by >2% HR@10 |
| Warm | 11-50 sessions | Beat baseline by >5% HR@10 |
| Hot | 50+ sessions | Beat baseline by >8% HR@10 |

Document the crossover point: at what session count does the SSM
reliably beat the baseline? This is the "cold start disappears"
threshold from VISION.md.

### Phase 2: Shadow Mode Evaluation

Run the SSM alongside the current system using Signet's existing
`shadowMode` infrastructure.

- SSM predicts top-K memories for each session
- Current system injects its own top-K
- Compare: do SSM's top-K receive better agent feedback than current
  system's top-K?
- Use SNIPS (self-normalized inverse propensity scoring) for
  counterfactual estimation of what would happen under SSM's policy

Pass criteria: SSM's predicted top-K must receive statistically
significantly better agent feedback than baseline's top-K (p < 0.05)
over a minimum of 50 sessions.

### Phase 3: Online A/B Evaluation

Enable SSM for real memory injection on alternating sessions.

- Track: agent feedback signals, task completion, memory hit rate,
  latency, contradiction rate, staleness distribution
- Minimum 100 sessions per arm
- Primary metric: agent_relevance_score mean across sessions

Pass criteria: SSM arm must show statistically significant improvement
in agent_relevance_score with no regression in latency or contradiction
rate.

---

## Synthetic Data Strategy

### Why Synthetic First

We can't wait for thousands of real sessions to validate the SSM
approach. Synthetic data with planted ground truth lets us:
1. Prove the model learns temporal patterns (canary tests)
2. Measure the model's noise floor (SNR curves)
3. Validate the training pipeline end-to-end
4. Establish baseline metrics before touching real data

### Data Generation Architecture

Generate synthetic interaction sequences that model agent sessions with
known temporal patterns.

```
SyntheticGenerator
  |
  +-- EntityGraph (synthetic knowledge graph)
  |     - 50-200 entities with typed relationships
  |     - Aspects and attributes per entity
  |     - Known dependency chains
  |
  +-- TemporalPatternMixer
  |     - Daily cycles (tod sin/cos)
  |     - Weekly cycles (dow sin/cos)
  |     - Burst patterns (Poisson process)
  |     - Decay curves (exponential + power law)
  |     - Seasonal drift
  |
  +-- SessionSimulator
  |     - 5-50 turns per session
  |     - 1-10 sessions per day
  |     - Focal entity selection from pattern mixer
  |     - Memory candidate generation (relevant + distractors)
  |     - Ground truth labels (which memories are truly relevant)
  |
  +-- NoiseInjector
        - Gaussian noise at configurable SNR
        - Label noise (flip ground truth with probability p)
        - Feature noise (corrupt individual features)
        - Missing data (randomly mask features)
```

### Data Budget

| Phase | Synthetic | Real | Total | Purpose |
|-------|-----------|------|-------|---------|
| Phase 0 | 50K sequences | 0 | 50K | Canary validation |
| Phase 1 | 50K | ~500 sessions | ~55K | Bootstrap |
| Phase 2 | 25K | ~2K sessions | ~27K | Mixed training |
| Phase 3 | 10K (augmentation) | ~5K+ sessions | ~15K+ | Real-dominant |

### Hard Negative Generation

Critical for teaching the model to distinguish relevant from
nearly-relevant:
- **Entity-neighbor negatives**: Memories from entities adjacent to the
  focal entity in the graph (structurally similar, semantically wrong)
- **Temporal-neighbor negatives**: Memories from the correct entity but
  wrong time window (right topic, stale information)
- **Semantic-neighbor negatives**: Memories with high embedding
  similarity but different entity context
- **Superseded negatives**: Old versions of facts that have been updated

### Curriculum Schedule

Start easy, get harder. Based on Pythia findings that curriculum effects
matter most for small models.

| Stage | Sequence Length | SNR | Hard Negatives | Duration |
|-------|----------------|-----|----------------|----------|
| 1 | 10-20 turns | 30dB | 0% | 20% of training |
| 2 | 20-50 turns | 20dB | 25% | 30% of training |
| 3 | 50-200 turns | 10dB | 50% | 30% of training |
| 4 | Full range | 5dB | 75% | 20% of training |

---

## Continual Learning Architecture

### Per-User Adaptation

Shared base model (Mamba-3, 5-10M params) + per-user LoRA adapters on
linear projections (150KB-1.5MB per user).

```
Base Model (shared, immutable during inference)
  |
  +-- Projection layers with LoRA adapters (per-user)
  |     - in_proj: LoRA rank 8-16
  |     - out_proj: LoRA rank 8-16
  |     - dt_proj: LoRA rank 4-8
  |     - ~150KB per user at rank 8
  |     - ~1.5MB per user at rank 16
  |
  +-- SSM state matrices A, B, C (shared, NOT adapted)
  |     - Lyapunov stability guaranteed
  |     - Perturbations decay, not amplify
  |
  +-- Hidden state (per-session, persisted across turns)
        - Checkpoint at session end
        - Restore at session start
```

### Session-Level TTT (Test-Time Training)

TTT4Rec shows meaningful adaptation from 5-10 interactions. Per-session:
1. At session start, initialize LoRA deltas from stored adapter
2. During session, accumulate self-supervised loss on predictions vs
   actual memory usage
3. At session end, apply one gradient step to LoRA adapter
4. Store updated adapter weights

Computational cost: one backward pass per session (~3.4x forward cost).
Amortized across a full session, this is negligible.

### Daemon Lifecycle Mapping (SleepGate-Inspired)

| Biological Phase | Daemon Phase | SSM Activity |
|------------------|--------------|--------------|
| Wake | Active sessions | Forward inference, accumulate gradients |
| NREM (consolidation) | Idle period (no active sessions) | Apply gradient updates, compress replay buffer |
| REM (exploration) | Scheduled maintenance | Synthetic data exploration, canary re-validation |

Trigger consolidation based on parameter-space drift (FOREVER metric),
not clock time. The SSM retrains when its predictions have drifted
from observed outcomes, not on a fixed schedule.

### Drift Detection

Trinity-Controller ADWIN: fuse volatility, adaptive sensitivity, and
accuracy EMA into a unified detector. When drift exceeds threshold:
1. Increase LoRA rank temporarily (more adaptation capacity)
2. Trigger NREM consolidation pass
3. If drift persists after consolidation, trigger full fine-tune cycle

### Federated Learning Path

Per-user LoRA deltas (150KB-1.5MB) are the communication payload.
With differential privacy applied at the client before transmission:
1. Clip LoRA gradients to norm bound
2. Add calibrated Gaussian noise
3. Transmit delta to aggregation server
4. Server averages deltas into base model update
5. Ship updated base model with next Signet release

---

## Training Data Inventory

Existing tables that feed SSM training (from codebase exploration):

| Table | Rows/Session | Temporal Resolution | Ground Truth Signal |
|-------|-------------|--------------------|--------------------|
| predictor_training_pairs | 50-200 | Per-session | combined_label [-1, 1] |
| session_memories | 200-1000 | Per-session (fts_hit_count per-prompt) | agent_relevance_score, was_injected |
| session_scores | 1 | Per-session | continuity score + confidence |
| predictor_comparisons | 1 | Per-session | predictor_won, NDCG, margin |
| session_checkpoints | 5-20 | Periodic within session | focal_entities, active_aspects |
| memory_history | 5-50 | Per-event | event type, old/new content |
| memory_entity_mentions | Per-memory | Per-extraction | entity appearance timeline |

### Critical Data Gaps

These gaps must be addressed before real-data training:

1. **No turn-by-turn ranking snapshots**: Add per-prompt telemetry
   recording which memories were considered and their rank order.
   New table: `prompt_memory_rankings(session_key, prompt_index,
   memory_id, rank, score, was_injected)`.

2. **No query embeddings persisted**: Store the query embedding for
   each prompt. Privacy concern mitigated by storing only the embedding
   vector (768 floats), not the raw text.

3. **No explicit negative labels**: Derive from was_injected=0 +
   fts_hit_count=0 + agent_relevance_score is NULL. Also generate
   hard negatives during training (entity-neighbor, temporal-neighbor).

4. **No inter-session memory chains**: Add a `memory_session_links`
   table tracking which memories were relevant across consecutive
   sessions for the same project.

## Implementation Strategy

### Training Pipeline

1. **Phase 0**: Build synthetic data generator. Run canary validation
   tests. Prove the SSM learns temporal patterns in isolation. No
   production code changes.
2. **Phase 1**: Pre-train base model on synthetic data. Validate on
   synthetic held-out set. Pass all canary tests and SNR curves.
3. **Phase 2**: Add turn-level telemetry to daemon (new migration for
   `prompt_memory_rankings`). Collect real data for 2-4 weeks.
4. **Phase 3**: Fine-tune on real data (TSTR protocol). Pass offline
   evaluation metrics on GTS split.
5. **Phase 4**: Shadow mode deployment. SSM predicts alongside current
   system. Compare feedback signals.
6. **Phase 5**: Online deployment with A/B evaluation.
7. **Phase 6**: Federated base model from opt-in community signals.

### Sidecar Architecture

Extend the existing `predictor/` Rust crate:

```
predictor/
  src/
    main.rs          -- JSON-RPC server (already exists)
    model.rs         -- Replace CrossAttentionScorer with SSM
    ssm/
      mamba.rs       -- Mamba-3 forward pass (interaction modeling)
      s4.rs          -- S4 forward pass (graph traversal scoring)
      state.rs       -- Hidden state persistence + checkpoint
      adapter.rs     -- Per-user LoRA adapter management
    training.rs      -- Online fine-tuning loop
    synthetic.rs     -- Synthetic data generator (for validation)
    protocol.rs      -- Extended input/output types
    canary.rs        -- Built-in canary pattern tests
```

The existing RPC interface between daemon and sidecar remains. The
sidecar includes built-in canary tests that run on startup to verify
the model hasn't degraded.

---

## Expected Impact

### Measurable Targets (Must Hit to Proceed Between Phases)

| Gate | Metric | Target | Measured On |
|------|--------|--------|-------------|
| Phase 0 -> 1 | Canary detection rate | 7/7 patterns at 20dB SNR | Synthetic data |
| Phase 1 -> 2 | TSTR performance ratio | >60% of real-only baseline | Synthetic train, real test |
| Phase 2 -> 3 | HR@10 improvement | >3% over baseline (p < 0.05, 5 seeds) | GTS split on real data |
| Phase 3 -> 4 | Shadow mode feedback | SSM top-K feedback > baseline top-K (p < 0.05) | 50+ shadow sessions |
| Phase 4 -> 5 | Online agent_relevance | Significant improvement, no latency regression | 100+ A/B sessions |

### Per-Session (Post Phase 5)

| Metric | Current | With SSM | How We'll Know |
|---|---|---|---|
| LLM calls per session | 8-10 | 3-4 | Telemetry counter |
| Per-turn latency | 50-100ms | 30-50ms | p50/p99 latency histogram |
| Checkpoint size | ~2KB | ~256 bytes | Measured per checkpoint |
| Cold-start crossover | Never (no learning) | ~10 sessions | Stratified evaluation |
| Memory injection precision | Unknown | >0.6 | agent_relevance on injected memories |

### Cost

- Binary size: ~2MB (1.58-bit quantization) to ~15MB (int8)
- Runtime memory: ~20-30MB (model + state + adapter)
- CPU overhead: <1ms per turn (SSM inference)
- Per-user adapter: 150KB-1.5MB
- Training: session-end gradient step + periodic consolidation

---

## Risk Assessment

### Technical Risks

**SSM training signal quality**: Continuity scorer labels are noisy.
*Mitigation*: Validate labels against behavioral signals (FTS overlap,
session gaps). Use confidence-weighted loss. Start with highest-confidence
subsets. Synthetic canary tests prove the model works before real data.

**Feedback loop instability**: SSM could converge to local optima,
reinforcing mediocre paths. *Mitigation*: Thompson sampling (noise in
path scores). Periodic exploration resets on low-confidence edges.
FOREVER-style drift detection triggers re-exploration when predictions
diverge from observations.

**State representation explosion**: Large hidden states are expensive to
serialize. *Mitigation*: Mamba-3's complex-valued state achieves same
quality with half the dimensions. 1.58-bit quantization fits 10M params
in 2MB.

**Rust SSM maturity**: Rust Mamba implementations are early-stage.
*Mitigation*: Start with S4D (simplest, proven at 200K params in JAX,
straightforward to port). Upgrade to Mamba-3 once Rust tooling matures.
web-rwkv (RWKV-7) is the most production-ready Rust alternative.
mamba.rs and oxidizr/blazr provide Mamba-specific paths.

**S4 vs Mamba for graphs**: GraphSSM shows S4 outperforms Mamba on
graph data. *Mitigation*: Use S4 for traversal path scoring, Mamba for
sequential interaction modeling. Two small models, not one.

### Scope Risks

**Over-engineering**: Multiple SSM integration points is ambitious.
*Mitigation*: Phase gates with measurable targets. Nothing proceeds
without passing validation contracts. Each phase is independently
deployable/revertable.

**Data gap**: No turn-by-turn ranking data exists today.
*Mitigation*: Phase 2 adds telemetry before fine-tuning. Synthetic
data validates the approach while real data accumulates.

**Evaluation trap**: Optimizing offline metrics that don't predict
online improvement. *Mitigation*: Use unnormalized DCG (0.97 online
correlation), not nDCG (-0.91 correlation). Shadow mode and A/B testing
before production deployment.

---

## Open Questions

1. **Complex-valued state for cyclical patterns**: Mamba-3's complex
   state models oscillations naturally. Worth prototyping for daily/weekly
   usage patterns. How much does this help vs real-valued state?
   Testable: plant daily/weekly canary patterns, compare complex vs real
   state detection rates.

2. **MIMO for multi-task prediction**: Can MIMO SSMs predict relevance,
   retention, and significance from a single forward pass without task
   interference? Testable: compare multi-head MIMO vs separate models on
   each task.

3. **CausalMamba for retention**: Can differentiable causal graph learning
   discover *why* memories become relevant, enabling causal-ancestor-aware
   decay? Memories that cause other memories to be retrieved should decay
   slower. Novel research direction.

4. **SSM state as identity embedding**: Could the persistent hidden state
   serve as a compact representation of agent personality/preferences,
   complementing SOUL.md? An agent's "feel" encoded in 256 floats.

5. **Bi-temporal data model for contradictions**: Zep/Graphiti uses 4
   timestamps per edge (valid_from, valid_to, created_at, expired_at).
   Should we adopt this for entity_dependencies instead of binary
   supersession? The SSM could learn temporal validity windows.

6. **DG-Mamba PRI regularization for entity bloat**: Learned pruning of
   inactive graph regions. Could replace our manual entity count thresholds
   (current: 43,520 entities, target <1,000).

7. **Publishable research**: Five novel gaps identified where Signet
   could contribute original work to the SSM literature: SSM
   contradiction detection, SSM belief revision, SSM identity coherence,
   privacy-preserving SSM memory, federated SSM weight aggregation.

---

## References

See companion documents for full citations:

**Round 1 (foundational research):**
- [SSM-LITERATURE-REVIEW.md](SSM-LITERATURE-REVIEW.md) -- 26 papers:
  HiPPO, S4, H3, Hyena, Mamba 1/2/3, TTT, xLSTM, FlowState, SleepGate,
  memory compression formalism, sequential recommendation
- [ssm-implementations-survey.md](ssm-implementations-survey.md) -- 20+
  repositories: mamba.rs, oxidizr/blazr, web-rwkv, llama.cpp Mamba,
  Candle, production deployments (Jamba, Nemotron-H, Bamba, Falcon Mamba)

**Round 2 (validation, depth, novel applications):**
- [SSM-NOVEL-APPLICATIONS.md](SSM-NOVEL-APPLICATIONS.md) -- 35+ papers:
  CausalMamba, Drama (7M world model), 1.58-bit Mamba, MPS-SSM (minimal
  predictive set), FOREVER (drift-based retraining), Jamba retrieval
  constraint, KambaAD anomaly detection
- [SSM-GRAPH-INTERSECTION.md](SSM-GRAPH-INTERSECTION.md) -- 30+ papers:
  DyGMamba/DyG-Mamba, NeuralWalker, HeteGraph-Mamba, GraphSSM (S4 > Mamba
  on graphs), GenTKG (chronological ordering), HiSS (hierarchical SSM),
  Zep/Graphiti bi-temporal model
- [SSM-CONTINUAL-LEARNING-DEEP-DIVE.md](SSM-CONTINUAL-LEARNING-DEEP-DIVE.md) --
  TTT/TTT4Rec, Mamba-CL (null-space projection), Inf-SSM (Grassmannian
  regularization), LoRA effectiveness on SSMs (projections not matrices),
  SleepGate (99.5% on proactive interference), WSCL (wake/NREM/REM),
  FedSSM, FOREVER drift detection
- [SYNTHETIC-DATA-GENERATION.md](SYNTHETIC-DATA-GENERATION.md) -- DR4SR
  (KDD 2024 best paper), RecSim NG, SynTSBench, curriculum learning
  (Pythia), EntiGraph entity expansion, slide-window augmentation,
  FENRec hard negatives, TSTR protocol

Key papers for this synthesis:
- Gu & Dao, "Mamba" (2023) -- selective state spaces
- Gu & Dao, "Mamba-2" (2024) -- state space duality
- Lahoti et al., "Mamba-3" (2026) -- inference-first design
- IBM FlowState (2025) -- 9.1M param SSM beats 200M+ models
- Xie, "SleepGate" (2026) -- sleep-inspired memory consolidation
- Bhat, "Memory Compression in Selective SSMs" (2024) -- theoretical bounds
- Gu, "Tradeoffs of SSMs and Transformers" (2025) -- brain vs database
- GraphSSM (NeurIPS 2024) -- S4 outperforms Mamba on graph data
- Jamba (ICLR 2025) -- attention required for associative recall (0% without)
- KDD 2024 -- nDCG anti-correlated with online reward (-0.91)
- MemoryArena (2026) -- passive recall ≠ agentic performance
- DyG-Mamba -- Ebbinghaus forgetting as learned SSM decay
- NeuralWalker -- bidirectional Mamba SOTA on graph walks
- SS4Rec / DyGMamba -- sequential recommendation + temporal KG reasoning
- TTT4Rec -- meaningful adaptation from 5-10 interactions
- DR4SR (KDD 2024 best paper) -- synthetic data regeneration
- "When +1% Is Not Enough" -- BCa bootstrap significance protocol
