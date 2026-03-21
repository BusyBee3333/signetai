---
question: How should we generate synthetic data for pre-training and validating the Signet predictive memory scorer (SSM-based), and what minimum real data is needed after synthetic bootstrapping?
date: 2026-03-20
informed_by:
  - docs/research/technical/SSM-LITERATURE-REVIEW.md
  - docs/research/technical/RESEARCH-SSM-INTEGRATION.md
tags: [synthetic-data, pre-training, SSM, predictor, validation, curriculum-learning]
---

# Synthetic Data Generation for SSM-Based Memory Relevance Scoring

## Executive Summary

This document surveys the state of the art in synthetic data generation
for training and validating small state-space models on sequential
prediction tasks. The target application is the Signet predictive memory
scorer: a cross-attention + gated-feature model that takes a 17-dimensional
behavioral feature vector per memory candidate and learns to predict which
memories are relevant for a given session context.

The research covers six areas: (1) synthetic interaction sequence
generation, (2) synthetic benchmarks for temporal models, (3) curriculum
learning for small SSMs, (4) data augmentation for interaction sequences,
(5) generating realistic agent interaction patterns, and (6) validation
through synthetic data with known ground truth.

Key finding: synthetic pre-training can reduce real-data requirements by
3-10x, but the generator must produce sequences with the exact feature
structure the model consumes. For our 17-dimensional feature vector,
this means generating controlled temporal cycles (tod/dow/moy), entity
relationship patterns, and access frequency distributions with planted
ground truth.

---

## 1. Synthetic Data for Sequential Recommendation

### 1.1 Dataset Regeneration (DR4SR) -- KDD 2024 Best Student Paper

**Source**: [Dataset Regeneration for Sequential Recommendation](https://arxiv.org/html/2405.17795v3) (Yin et al., KDD 2024)

DR4SR regenerates training datasets by extracting frequent item
transition patterns, then training a Transformer-based regenerator to
produce diverse new patterns from original sequences.

**Algorithm**:
1. Sliding window of size alpha extracts frequent item transitions
   from user sequences. Patterns exceeding threshold beta form
   pre-training targets.
2. A diversity-promoted regenerator (encoder + K memory spaces +
   decoder) learns one-to-many mappings from sequences to patterns.
3. Hybrid inference: probability gamma for generative mode
   (unrestricted items), 1-gamma for restrictive mode (items from
   input sequence only).
4. Generates K diverse patterns per sequence via separate memory
   space initializations.

**Training details**: Adam optimizer, lr=1e-3, batch size 256,
embedding dim 64, max 1000 epochs with early stopping (20 patience
on NDCG@20). Diversity factor K in [1,3,5,7,9].

**Results**: 5-43% improvement across metrics, consistent across 5
diverse model architectures (GRU4Rec, SASRec, FMLP, GNN, CL4SRec).
Average pattern length: 3.7-4.2 items vs 8.3-8.9 for originals.

**Relevance to Signet**: The regeneration concept maps directly to
generating new session-memory interaction patterns from observed ones.
Instead of item transitions, we regenerate memory-access patterns
with diverse temporal feature configurations.

### 1.2 RecSim NG -- Google's Simulation Platform

**Source**: [RecSim NG](https://github.com/google-research/recsim_ng) (Google Research)

A probabilistic platform for multi-agent recommender systems
simulation. Key architecture:

- **User model**: Samples from distribution over latent features
  (interests, satisfaction), observable features (demographics), and
  behavioral features (visit frequency, time budget).
- **Document model**: Samples items from prior distribution over
  latent (quality) and observable (length, popularity) features.
- **State transitions**: After item consumption, user state
  transitions through configurable model (satisfaction/interest drift).
- **Story API**: Arbitrary actors interact, modeling entire
  recommender ecosystems.

**Relevance**: RecSim's user model architecture directly parallels
our session simulation needs. We can model "agent sessions" as users,
"memories" as items, and use configurable transition models to inject
temporal patterns.

### 1.3 Time-Varying Markov Chain Session Generation

**Source**: [Generating and understanding human daily activity sequences using Time-Varying Markov Chain models](https://www.sciencedirect.com/science/article/pii/S2214367X2300162X)

Daily activity patterns generated using time-varying Markov Chains
with just 6 parameters per area. Predictive power comparable to
neural network models. Key insight: the transition matrix varies by
time of day and day of week, naturally encoding the cyclical patterns
our feature vector captures (tod_sin/cos, dow_sin/cos).

**Implementation approach for Signet**:
- Define a time-varying transition matrix T(hour, dow) over memory
  categories (entity types, topic clusters).
- At each simulated timestep, sample which memory categories are
  "active" based on T.
- This naturally generates the cyclical access patterns our features
  [3]-[8] (tod/dow/moy sin/cos) are designed to detect.

---

## 2. Synthetic Benchmarks for Temporal Models

### 2.1 Mamba Selective Copying Task

**Source**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)

**Task specification**:
- **Vocabulary**: Alphabet size A. Tokens to copy: 1 to A-2.
  Marker token: A-1. Noise: 0.
- **Input**: [M tokens scattered among L noise positions | M marker tokens]
- **Output**: The M tokens in original (or reversed) order.
- **Key parameter**: `variable=True` randomizes token positions,
  creating the "selective" variant that requires content-aware reasoning.

**Implementation** ([selective-copying-mamba](https://github.com/MinhZou/selective-copying-mamba)):
```
Input:  [0, 6, 0, 0, 7, 0, 5, 0, 0, 9, 9, 9, 9, 9]
Output: [6, 7, 5]  (copy marked tokens, ignore noise)
```

**Why it matters for SSMs**: LTI (linear time-invariant) models
cannot solve this because they treat all positions identically.
Selective SSMs (Mamba) solve it by making state transitions
input-dependent -- exactly the property needed for memory scoring
where some memories are "marked" as relevant by temporal context.

**Adaptation for Signet**: Create "selective memory retrieval" task:
scatter relevant memory features among irrelevant ones, require the
model to identify which memories match the temporal context signal.
This tests whether the SSM learns to filter based on feature values
rather than position.

### 2.2 Induction Heads Task

**Source**: Same Mamba paper.

**Task specification**: Model observes pattern A->B in sequence,
then when A appears again, must predict B. Tests associative recall
across arbitrary distances.

**Adaptation for Signet**: Plant memory-entity associations (e.g.,
entity_slot X always co-occurs with high relevance during Monday
sessions), then test whether the model predicts this pattern when
encountering the entity on a new Monday.

### 2.3 State Tracking Tasks (Monoid Word Problems)

**Source**: [The Illusion of State in State-Space Models](https://arxiv.org/abs/2404.08819) (Merrill et al., 2024)

Tests SSM ability to track algebraic state through sequences:
- Z_60 (abelian group, 60 elements) -- easy
- A4 x Z5 (solvable non-abelian, 60 elements) -- medium
- A5 (non-solvable alternating group, 60 elements) -- hard

**Key finding**: Diagonal SSMs (S4, Mamba) require depth
monotonically increasing with sequence length. RNNs and IDS4 solve
arbitrary-length sequences with a single layer. This means our SSM
may struggle with complex state tracking -- we should design tasks
that stay within the capability envelope.

**Practical implication**: For memory scoring, the "state" is the
relevance context accumulated over a session. The feature vector
dimensions (17) are small enough that the SSM should handle this
without needing deep stacks. But we should test explicitly with
planted state-tracking canaries.

### 2.4 Parity Detection Limitations

**Source**: [The Expressive Limits of Diagonal SSMs for State-Tracking](https://arxiv.org/html/2603.01959)

Input-dependent non-negative diagonal SSMs (Mamba) cannot solve
parity in finite precision for arbitrary sequence lengths. This
is a fundamental limitation to be aware of -- our model should not
be expected to learn parity-like features. Fortunately, our feature
vector uses continuous values (log transforms, sin/cos encodings)
rather than discrete state, which sidesteps this limitation.

### 2.5 IBM PD-SSM for State Tracking

**Source**: [Efficient Transition Matrices to Enable State Tracking in State-Space Models](https://github.com/IBM/expressive-sparse-state-space-model) (NeurIPS 2025)

PD-SSM parametrizes the transition matrix as the product of a
column one-hot matrix (P) and a complex-valued diagonal matrix (D).
Achieves 98.5% average on state tracking benchmarks. If our model
hits state-tracking walls, PD-SSM's parametrization is a potential
upgrade path from diagonal Mamba-style SSMs.

### 2.6 SynTSBench -- Programmable Temporal Pattern Benchmark

**Source**: [SynTSBench](https://github.com/TanQitai/SynTSBench) (NeurIPS 2025 Datasets Track)

Systematic assessment framework with programmable feature
configuration:

**Temporal pattern types generated**:
- **Trends**: 11 functions (linear, exponential, logarithmic,
  logistic, power-law, Gompertz, Gaussian, quadratic,
  piecewise-linear, negative-exponential, step)
- **Periodicity**: 10 types based on Fourier series (single sine,
  double-sin through ten-sin, triangle, sawtooth, square waves,
  exponentially-modulated sine)
- **Dependencies**: ARMA(1,1), ARMA(2,2), ARMA-Long (lag-50),
  random walks, white noise
- **Multivariate**: Lagged features (5/10/24/48 steps), sine-noise,
  conditional relationships, nonlinear, 5-variable feedback

**Noise injection protocol**:
- SNR levels: clean, 30dB, 20dB, 10dB, 0dB, -10dB
- Formula: y_t = x_t + epsilon_t, epsilon ~ N(0, sigma^2_noise),
  SNR = 10 * log10(Var(x_t) / sigma^2_noise)
- Distributions: Gaussian, uniform, Laplace, t-distribution,
  heavy-tailed Levy stable
- Two MSE variants: MSE_Obs (vs noisy) and MSE_True (vs clean signal)

**Key results**:
- MLP architectures dominate trend forecasting
- DLinear excels at pure sinusoidal signals (7/10 functions)
- All models fail on square-wave (discontinuous) patterns
- Transformers show "most significant performance degradation as
  noise intensifies"
- At SNR -10dB, most models converge to MSE_Obs ~1.0

**Relevance**: Directly applicable to testing our SSM's ability to
detect planted temporal patterns in memory features. We should
generate features with known periodicity (our tod/dow/moy features
are literally sin/cos encodings) and verify the model detects them
at various noise levels.

---

## 3. Curriculum Learning for Small SSMs

### 3.1 Sequence Length Curriculum (DeepSpeed)

**Source**: [DeepSpeed Curriculum Learning Tutorial](https://www.deepspeed.ai/tutorials/curriculum-learning/)

Start training with shorter sequences, progressively increase to
full length. Three schedule types:

**Fixed Linear** (recommended): Linearly increases difficulty from
min to max over total_curriculum_step steps.
```json
{
  "schedule_type": "fixed_linear",
  "schedule_config": {
    "total_curriculum_step": 15000,
    "difficulty_step": 8
  }
}
```

**Fixed Root**: Uses root function for faster early ramp.

**Fixed Discrete**: Explicit difficulty levels at step boundaries.

**Results**: 3.3x faster GPT-2 pre-training, enables 8x larger
batch size / 4x larger learning rate without divergence.

**Tuning strategy**: Binary search for largest total_curriculum_step
without validation perplexity fluctuations. Start min_difficulty at
8 for million-scale models, 64 for billion-scale.

**Adaptation for Signet predictor**: Start with short session
sequences (5-10 memory candidates), increase to full candidate sets
(50+). Start with clear signal (high SNR synthetic data), progress
to noisier/ambiguous sequences.

### 3.2 Curriculum Learning Dynamics (ICLR 2025)

**Source**: [Curriculum Learning for LLM Pretraining](https://arxiv.org/abs/2601.21698)

Trained Pythia models (14M-410M params) under three curricula.
Key finding: **curriculum effects matter most for smaller models**,
where capacity constraints make data ordering particularly
consequential. Training follows shared latent phases regardless
of curriculum; curricula mainly change within-phase data exposure.

**Implication**: Our ~100K parameter predictor model is exactly
where curriculum learning matters most. The training order of
synthetic data will significantly impact final performance.

### 3.3 Synthetic Continued Pre-training with EntiGraph (ICLR 2025)

**Source**: [Synthetic Continued Pretraining](https://arxiv.org/html/2409.07431v1) (Zelikman et al., ICLR 2025)

EntiGraph extracts entities from source documents, generates
diverse synthetic text by drawing connections between entity pairs
and triplets.

**Scaling**: 1.3M real tokens -> 455M synthetic tokens (~350x
expansion). Performance follows mixture-of-exponential curve:
linear growth -> log-linear growth -> plateau.

**Key results**: 56.42% accuracy vs 38.15% for raw continued
pre-training (18.3pp improvement). Paraphrasing baseline plateaus
at ~2M tokens; EntiGraph scales to 600M.

**Code**: [github.com/ZitongYang/Synthetic_Continued_Pretraining.git](https://github.com/ZitongYang/Synthetic_Continued_Pretraining.git)

**Adaptation for Signet**: The entity-relationship expansion concept
maps to our knowledge graph. Given a small set of real session data,
generate diverse synthetic sessions that exercise the same entity
relationships from different temporal angles (different times of day,
different session gaps, different access counts).

### 3.4 Practical Ratios: Synthetic vs Real Data

**WRAP (Apple/CMU, ICLR 2024)**: Maintained 1:1 ratio of real and
synthetic data. 3x pre-training speedup with this balanced approach.
[Source](https://machinelearning.apple.com/research/recipe-for-compute)

**Phi-1 (Microsoft)**: 1.3B parameter model trained on 6B filtered
web tokens + 1B synthetic textbook tokens + 180M synthetic exercises.
Synthetic = ~15% of total training data. Achieved 50.6% pass@1 on
HumanEval. [Source](https://arxiv.org/abs/2306.11644)

**Self-Instruct**: 52K instructions from just 175 seed tasks. Only
54% of synthetic samples were fully valid, yet 33% improvement over
vanilla GPT-3.

**SPIN**: Matched its 50K-sample performance using just 1.8K
well-curated examples.

**Practical guidance for Signet predictor**:
- **Phase 1** (cold start): 100% synthetic data, ~50K sequences
- **Phase 2** (bootstrap): 1:1 mix once ~500 real sessions exist
- **Phase 3** (maturity): Shift to majority real data once ~2K+
  real sessions with feedback signals accumulate
- The predictor's small parameter count (~100K params) means even
  500 diverse real sessions may be sufficient for fine-tuning after
  synthetic pre-training.

---

## 4. Data Augmentation for Interaction Sequences

### 4.1 Sequence-Level Augmentation Strategies

**Source**: [Is Contrastive Learning Necessary?](https://arxiv.org/html/2403.11136v1) (ACM Web Conference 2024)

Eight strategies tested for sequential recommendation:

| Strategy | Description | Performance |
|----------|-------------|-------------|
| Slide-window | Fixed-length window sliding over sequence | Best (96.2% of optimal Recall@20) |
| Crop | Extract continuous subsequence | Second best (85.1%) |
| Subset-split | Probabilistic dropout (theta=0.25) | Good |
| Reorder | Shuffle items in sub-sequence | Moderate |
| Insert | Add random item at random position | Moderate |
| Delete | Remove random item | Moderate |
| Replace | Substitute item from pool | Poor on short sequences |
| Mask | Replace with mask token | Poor on short sequences |

**Key finding**: Slide-window alone outperformed all three
contrastive learning baselines (CL4SRec, CoSeRec, ICLRec) on most
datasets, at 1/5 to 1/10 the training time and 1/4 GPU memory.

**Best combination**: Slide-window + subset-split = +6.0% Recall@20
and +6.8% NDCG@20 over slide-window alone.

**Adaptation for Signet**: Apply slide-window augmentation over
session sequences. Each session has a sequence of memory candidates;
sliding windows create multiple training samples from one session.
Subset-split creates variations where some candidates are dropped.

### 4.2 Time Warping

**Source**: [Time Series Data Augmentation for Neural Networks by Time Warping](https://arxiv.org/pdf/2004.08780)

Two methods:
1. **Smooth warping path**: Cubic spline S(u) with knots creates
   smooth transitions between stretches and contractions.
2. **Window warping**: Select random window, stretch by 2x or
   compress by 0.5x. Rest of signal unchanged.

**Relevance**: Apply window warping to the temporal features in our
17-dim vector. Stretching the session_gap_days feature or warping
the time-of-day encoding simulates sessions happening at slightly
different times -- a natural form of temporal augmentation.

### 4.3 Hard Negative Generation (FENRec, AAAI 2025)

**Source**: [FENRec: Future Sight and Tough Fights](https://arxiv.org/html/2412.11589)

Generates hard negatives by mixing anchor representations with
in-batch negatives:
```
h_neg = lambda * h_anchor_norm + (1-lambda) * n_norm
        / ||lambda * h_anchor_norm + (1-lambda) * n_norm||
        * ||n||
```

Lambda in {0.1, 0.2, 0.3, 0.4, 0.5}. Stop-gradient prevents
incorrect backpropagation.

**Key insight**: Mixed negatives maintain higher similarity to
anchor than originals (proven in appendix), remaining "consistently
challenging" throughout training. Original random negatives become
easier over training epochs.

**Warm-up**: 20-epoch warm-up before introducing hard negatives.

**Results**: 6.34% improvement in HIT, 5.99% in NDCG.

**Adaptation for Signet**: Generate hard negative memories that
look relevant (high entity_slot match, similar temporal features)
but are actually irrelevant (wrong topic, superseded content). This
teaches the model to use the full feature vector rather than
shortcuts.

### 4.4 Soft Contrastive Learning for Time Series (ICLR 2024)

**Source**: [SoftCLT](https://github.com/seunghan96/softclt) (ICLR 2024)

Instead of binary positive/negative labels, assigns soft labels
based on temporal distance. Positive pairs get weights based on
Gaussian distribution over timestamp difference.

**Relevance**: For memory sequences, two sessions close in time
are "soft positives" -- they likely share relevant memories but
not identically. This is more realistic than binary labeling for
our training signal.

---

## 5. Generating Realistic Agent Interaction Patterns

### 5.1 MemoryAgentBench -- Multi-Competency Evaluation

**Source**: [MemoryAgentBench](https://arxiv.org/html/2507.05257v3)

Evaluates four memory competencies:
1. **Accurate Retrieval**: Extract correct information via single
   queries
2. **Test-Time Learning**: Acquire new behaviors during deployment
3. **Long-Range Understanding**: Integrate information across
   100K+ tokens
4. **Selective Forgetting**: Revise/remove outdated information

**Data generation pipeline**:
- Segment source material into 512-token (complex) or 4096-token
  (narrative) chunks
- Feed sequentially with memorization instructions
- Ground truth from original annotations
- Counterfactual edit pairs for testing conflict resolution

**Adaptation for Signet**: Our predictor needs all four competencies.
Synthetic data should include sequences where: (a) a specific memory
is clearly the right answer (accurate retrieval), (b) relevance
patterns shift mid-session (test-time learning), (c) relevant
information appeared many steps ago (long-range), (d) a memory was
superseded by a newer one (selective forgetting -- our is_superseded
feature [11]).

### 5.2 Generative Agents Memory Retrieval

**Source**: [Generative Agents: Interactive Simulacra of Human Behavior](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763) (Park et al., 2023)

Memory retrieval score = alpha_recency * recency + alpha_importance
* importance + alpha_relevance * relevance. All weights set to 1.

Our predictor's 17-dim feature vector is a superset of this:
- [0] log(age_days) = recency
- [1] importance = importance
- [2] log(access_count+1) = frequency (proxy for relevance signal)
- [3-8] temporal cyclical features = not in Generative Agents
- [9] session_gap_days = not in Generative Agents
- [10-16] structural features = not in Generative Agents

**Implication**: Synthetic data should exercise all 17 dimensions,
not just the recency/importance/relevance triad. The temporal and
structural features are what differentiate our model.

### 5.3 Cross-Attention Memory Scoring (ACAN)

**Source**: [Enhancing memory retrieval in generative agents through LLM-trained cross attention networks](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1591618/full)

**Training data generation**: Simulated 8 agents over 3 consecutive
days with ChatGPT-guided interactions. LLM evaluates retrieved
memory quality to generate training labels.

**Our advantage**: We have real feedback signals (session scores,
injection logs, FTS hit counts) rather than needing LLM-as-judge
labels. But for synthetic pre-training, we can generate labels
deterministically from planted patterns.

### 5.4 Procedural Memory Retrieval Benchmark

**Source**: [A Benchmark for Procedural Memory Retrieval in Language Agents](https://arxiv.org/html/2511.21730v1)

**Query stratification by difficulty**:
- EASY (n=15): Single-object tasks, mean 14.4 relevant trajectories
- MEDIUM (n=14): Multi-step with state changes, mean 12.5
- HARD (n=11): Multi-object coordination, mean 14.0

Classification considers: object cardinality, presence of state
transformations, and multi-step dependencies.

**Key finding**: Embedding-based methods (like our cross-attention
scorer) perform strongly on familiar contexts but degrade on novel
ones. LLM-generated procedural abstractions show reliable
cross-context transfer.

**Implication for synthetic data**: Include difficulty-stratified
sequences. Easy sequences have clear single-feature relevance
signals; hard sequences require combining multiple features
(temporal + entity + structural).

### 5.5 Temporal Knowledge Graph Dynamics

**Source**: [A Temporal Knowledge Graph Generation Dataset](https://www.nature.com/articles/s41597-025-05062-0)

Temporal knowledge graphs model entity-relationship evolution with
timestamps. Over 80% of all events throughout 24 years of ICEWS
data have already appeared during the previous time period.

**Implication**: Most memory access patterns are repetitive. Our
synthetic generator should produce sequences where ~80% of memory
accesses follow established patterns (testable with the model) and
~20% are novel (testing generalization).

---

## 6. Validation Through Synthetic Data

### 6.1 Contract Testing with Known Ground Truth

The core validation principle: generate sequences where the correct
answer is known by construction, then verify the model finds it.

**Approach from SynTSBench**: For each temporal pattern type,
calculate the theoretical optimum. Compare model prediction against
this optimum. Gaps indicate which patterns the model cannot detect.

**For Signet predictor, the contract tests are**:

1. **Temporal cycle detection**: Plant a pattern where entity X is
   relevant during morning sessions (tod_sin > 0.5) and irrelevant
   during evening sessions. Verify the model scores entity X higher
   for morning contexts.

2. **Recency decay verification**: Generate sequences where older
   memories (high age_days) should score lower. Verify monotonic
   decrease in score with age.

3. **Frequency amplification**: Generate sequences where frequently
   accessed memories (high access_count) are more relevant. Verify
   positive correlation between access_count and score.

4. **Entity slot selectivity**: Generate sequences where entity_slot
   matches between query and candidate predict relevance. Verify
   the model uses this feature.

5. **Superseded memory suppression**: Generate sequences where
   is_superseded=1 memories should always score below their
   replacement. Verify ordering.

6. **Session gap sensitivity**: Generate sequences where memories
   accessed in recent sessions (low session_gap_days) are more
   relevant. Verify the model captures this.

7. **Combined pattern detection**: Generate sequences where
   relevance depends on the interaction of multiple features (e.g.,
   entity X is relevant on Mondays AND in the morning AND when
   session gap < 3 days). This is the hardest test.

### 6.2 Canary Pattern Methodology

Plant specific, identifiable patterns in synthetic data that can
only be found if the model has learned temporal dynamics:

**Source**: [How Patterns Dictate Learnability](https://arxiv.org/html/2510.10744)

The minimal achievable risk estimator R_inf(Q*) provides a bound
on what any model can learn from the data. When empirical risk
approaches this bound, the model has captured available patterns.
Gaps indicate unexploited temporal dependencies.

**Implementation for Signet**:
- Generate AR(p) processes for each feature with known parameters
  (rho, p). The model should learn to predict next-step values.
- Measure mutual information between past and future feature
  segments. Compare model's predictive accuracy against this bound.
- Use Ising spin sequences with piecewise-stationary blocks to test
  whether the model detects non-stationarity (e.g., when a user
  changes their workflow).

### 6.3 Signal-to-Noise Ratio Experiments

**From SynTSBench protocol**:

1. Generate clean synthetic sequences with known relevance signals.
2. Inject noise at SNR levels: 30dB, 20dB, 10dB, 0dB, -10dB.
3. Measure model accuracy at each level.
4. Find the SNR threshold where accuracy drops below acceptable
   level (e.g., below 80% of clean accuracy).

**Expected thresholds for Signet predictor**:
- 30dB-20dB: Near-perfect pattern detection
- 10dB: Temporal cycle detection starts degrading
- 0dB: Only strong features (recency, importance) still work
- -10dB: Model effectively random

These thresholds tell us how robust the model is to noisy real-world
data and help set expectations for production performance.

### 6.4 Train on Synthetic, Test on Real (TSTR)

The standard validation protocol: train model entirely on synthetic
data, evaluate on held-out real data. If TSTR performance approaches
Train-on-Real performance, the synthetic data is capturing the right
distributions.

**For Signet**: Once we have ~100 real sessions with feedback, split
50/50 into synthetic-evaluation and real-baseline sets. Compare
TSTR model against real-only baseline on the evaluation set.

---

## 7. Proposed Synthetic Data Generator Architecture

Based on the research above, here is the proposed generator design
for the Signet predictor.

### 7.1 Generator Parameters

```
SessionGenerator {
  // Temporal parameters
  sessions_per_day: Distribution(mean=5, std=2)
  session_hour_distribution: MixtureGaussian([9,14,21], [2,2,1.5])
  days_of_week_activity: [1.0, 1.0, 1.0, 1.0, 0.9, 0.5, 0.3]
  session_gap_distribution: LogNormal(mean=0.2, std=0.5)

  // Memory pool parameters
  num_memories: Range(50, 500)
  entity_types: Range(5, 30)
  aspect_types: Range(3, 15)
  importance_distribution: Beta(2, 5)  // right-skewed
  access_count_distribution: Zipf(alpha=1.5)

  // Pattern injection parameters
  temporal_cycle_strength: Range(0.0, 1.0)  // SNR control
  entity_preference_strength: Range(0.0, 1.0)
  recency_decay_rate: Range(0.01, 0.5)
  supersede_rate: Range(0.0, 0.2)

  // Noise parameters
  snr_db: Range(-10, 30)
  noise_distribution: OneOf(Gaussian, Uniform, Laplace)
}
```

### 7.2 Feature Vector Generation

For each candidate memory in a synthetic session, generate the
17-dimensional feature vector:

```
[0]  log(age_days + 1)          <- from simulated creation time
[1]  importance                 <- from Beta(2,5) distribution
[2]  log(access_count + 1)     <- from Zipf, with temporal bias
[3]  sin(2*pi*hour/24)         <- from session timestamp
[4]  cos(2*pi*hour/24)         <- from session timestamp
[5]  sin(2*pi*dow/7)           <- from session timestamp
[6]  cos(2*pi*dow/7)           <- from session timestamp
[7]  sin(2*pi*month/12)        <- from session timestamp
[8]  cos(2*pi*month/12)        <- from session timestamp
[9]  log(session_gap_days + 1) <- from gap to last access
[10] is_embedded               <- Bernoulli(0.8)
[11] is_superseded             <- from supersede_rate
[12] entity_slot (0-1)         <- from entity type assignment
[13] aspect_slot (0-1)         <- from aspect type assignment
[14] is_constraint             <- Bernoulli(0.1)
[15] log(structural_density+1) <- from entity graph density
[16] is_ka_traversal           <- Bernoulli(0.05)
```

### 7.3 Label Generation (Known Ground Truth)

Relevance label for each candidate in a session is computed
deterministically from planted patterns:

```
base_score = 0.0

// Recency signal
base_score += recency_weight * exp(-age_days * decay_rate)

// Entity match signal
if entity_matches_session_topic:
    base_score += entity_weight * preference_strength

// Temporal cycle signal
if hour_matches_memory_peak_time:
    base_score += temporal_weight * cycle_strength

// Access frequency signal
base_score += frequency_weight * log(access_count + 1) / max_log_count

// Superseded penalty
if is_superseded:
    base_score -= 0.3

// Structural bonus
if is_ka_traversal and session_is_graph_traversal:
    base_score += 0.15

// Clip and add noise
label = clip(base_score + noise(snr), 0.0, 1.0)
```

### 7.4 Curriculum Schedule

Following DeepSpeed curriculum learning findings:

**Phase 1 -- Trivial patterns** (steps 0-5K):
- 5 candidates per session
- Single-feature relevance (recency only)
- SNR 30dB
- Short sequences (10 sessions)

**Phase 2 -- Two-feature patterns** (steps 5K-15K):
- 10 candidates per session
- Recency + entity match
- SNR 20dB
- Medium sequences (30 sessions)

**Phase 3 -- Multi-feature patterns** (steps 15K-30K):
- 20 candidates per session
- All features active
- SNR 10dB
- Long sequences (100 sessions)

**Phase 4 -- Realistic complexity** (steps 30K-50K):
- 50 candidates per session
- All features + noise + supersession dynamics
- SNR 0dB-10dB (mixed)
- Full-length sequences (200+ sessions)

### 7.5 Validation Test Suite

After synthetic pre-training, run the following contract tests:

1. **Selective copying canary**: 10 memories in a session, 3 are
   relevant (based on entity match). Model must rank them top-3.
   Pass threshold: 90% top-3 accuracy.

2. **Temporal induction canary**: Entity X was relevant in sessions
   at hour=9 historically. New session at hour=9 with entity X.
   Model must score entity X above median. Pass threshold: 85%.

3. **Recency ordering canary**: 5 identical memories differing only
   in age. Model must rank newest highest. Pass threshold: 95%
   correct ordering.

4. **Supersession canary**: Memory A is superseded by Memory B.
   Model must score B > A. Pass threshold: 90%.

5. **Noise robustness canary**: Same test suite at SNR 10dB, 0dB,
   -10dB. Track degradation curve.

6. **Combined pattern canary**: Relevance depends on interaction of
   3+ features. Pass threshold: 70% (this is the hard test).

---

## 8. Implementation Priority

1. **Build feature vector generator** (7.2) -- this is the
   foundation. Must exactly match the 17-dim format in
   `packages/predictor/src/protocol.rs`.

2. **Build label generator** (7.3) -- deterministic from planted
   patterns, giving us known ground truth.

3. **Build curriculum scheduler** (7.4) -- start with Phase 1 and
   validate the model learns trivial patterns before progressing.

4. **Build validation suite** (7.5) -- the contract tests that must
   pass before deploying to real data.

5. **Build augmentation pipeline** (section 4) -- slide-window +
   subset-split for expanding real sessions once available.

6. **Build hard negative generator** (4.3) -- critical for teaching
   the model to distinguish semantically similar but irrelevant
   memories.

---

## Sources

### Synthetic Data for Sequential Recommendation
- [DR4SR: Dataset Regeneration for Sequential Recommendation](https://arxiv.org/html/2405.17795v3) -- KDD 2024 Best Student Paper
- [RecSim NG: Flexible Recommender Systems Simulation](https://github.com/google-research/recsim_ng) -- Google Research
- [Time-Varying Markov Chain Activity Sequences](https://www.sciencedirect.com/science/article/pii/S2214367X2300162X)
- [Privacy-Preserving Synthetic Data for Recommendation](https://dl.acm.org/doi/abs/10.1145/3477495.3532044)

### Synthetic Benchmarks for Temporal Models
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Selective Copying Task Implementation](https://github.com/MinhZou/selective-copying-mamba)
- [The Illusion of State in State-Space Models](https://arxiv.org/abs/2404.08819)
- [The Expressive Limits of Diagonal SSMs](https://arxiv.org/html/2603.01959)
- [IBM PD-SSM: Efficient Transition Matrices](https://github.com/IBM/expressive-sparse-state-space-model) -- NeurIPS 2025
- [SynTSBench: Temporal Pattern Benchmark](https://github.com/TanQitai/SynTSBench) -- NeurIPS 2025
- [How Patterns Dictate Learnability](https://arxiv.org/html/2510.10744)

### Curriculum Learning
- [DeepSpeed Curriculum Learning](https://www.deepspeed.ai/tutorials/curriculum-learning/)
- [Curriculum Learning for LLM Pretraining](https://arxiv.org/abs/2601.21698) -- ICLR 2025
- [Synthetic Continued Pretraining (EntiGraph)](https://arxiv.org/html/2409.07431v1) -- ICLR 2025
- [Curriculum Learning for Small Code Models](https://arxiv.org/html/2407.10194v1)

### Data Augmentation
- [Is Contrastive Learning Necessary? (Augmentation Study)](https://arxiv.org/html/2403.11136v1) -- ACM Web 2024
- [FENRec: Hard Negatives for Sequential Recommendation](https://arxiv.org/html/2412.11589) -- AAAI 2025
- [Soft Contrastive Learning for Time Series](https://github.com/seunghan96/softclt) -- ICLR 2024
- [Time Warping Augmentation](https://arxiv.org/pdf/2004.08780)
- [Contrastive Learning with Hard Negatives](https://arxiv.org/abs/2010.04592)

### Agent Memory Patterns
- [MemoryAgentBench: LLM Memory Benchmark](https://arxiv.org/html/2507.05257v3)
- [Generative Agents: Smallville](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763)
- [Cross-Attention Memory Retrieval (ACAN)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1591618/full)
- [Procedural Memory Retrieval Benchmark](https://arxiv.org/html/2511.21730v1)
- [Temporal Knowledge Graph Generation](https://www.nature.com/articles/s41597-025-05062-0)
- [Reflective Memory Management](https://aclanthology.org/2025.acl-long.413.pdf)

### Synthetic Pre-training Economics
- [WRAP: Rephrasing the Web](https://machinelearning.apple.com/research/recipe-for-compute) -- Apple/CMU, ICLR 2024
- [Phi-1: Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) -- Microsoft
- [How to Generate and Use Synthetic Data](https://eugeneyan.com/writing/synthetic/) -- Eugene Yan
- [Fine-tuning with Limited Data (Survey)](https://arxiv.org/html/2411.09539v2)
- [Synthetic Eggs in Many Baskets](https://arxiv.org/html/2511.01490v1)

### Validation and Testing
- [SynTSBench SNR Testing Protocol](https://arxiv.org/html/2510.20273v1)
- [TSGM: Synthetic Time Series Framework](https://github.com/AlexanderVNikitin/tsgm) -- NeurIPS 2024
- [Synthetic Data for RAG Evaluation](https://developers.redhat.com/articles/2026/02/23/synthetic-data-rag-evaluation-why-your-rag-system-needs-better-testing)
- [Mamba Empirical Study](https://arxiv.org/html/2406.07887v1)

### SSM Architecture Reference
- [An Empirical Study of Mamba-based Language Models](https://arxiv.org/html/2406.07887v1) -- Training recipes
- [From S4 to Mamba: Comprehensive Survey](https://arxiv.org/pdf/2503.18970)
- [Mamba-3: Improved Sequence Modeling](https://arxiv.org/pdf/2603.15569)
