---
question: How can state-space models be combined with knowledge graphs, temporal graphs, and graph neural networks to model graph traversal dynamics, entity evolution, structural reasoning, contradiction detection, and hierarchical temporal patterns?
date: 2026-03-20
informed_by:
  - SSM-LITERATURE-REVIEW.md
  - RESEARCH-SSM-INTEGRATION.md
  - ssm-implementations-survey.md
  - VISION.md
  - docs/specs/INDEX.md
---

# SSMs at the Graph Intersection: Temporal KGs, GNNs, Entity Evolution, and Hierarchical Reasoning

This document surveys the intersection of state-space models with knowledge
graphs, temporal graphs, and graph neural networks. The goal is to
understand how SSMs can model graph traversal dynamics, entity evolution,
structural reasoning, contradiction detection, and multi-scale temporal
patterns -- all directly applicable to Signet's knowledge graph with
entities, aspects, attributes, and dependencies that evolve over time.

Companion documents:
- [Literature Review](SSM-LITERATURE-REVIEW.md) -- foundational SSM papers,
  small/efficient architectures, training on personal data
- [Implementation Survey](ssm-implementations-survey.md) -- reference
  implementations, production deployments, Rust ecosystem
- [Integration Synthesis](RESEARCH-SSM-INTEGRATION.md) -- how SSMs could
  transform Signet's memory pipeline

---

## Table of Contents

1. [SSMs for Temporal Knowledge Graphs](#1-ssms-for-temporal-knowledge-graphs)
2. [SSMs Combined with GNNs](#2-ssms-combined-with-gnns)
3. [SSMs for Dynamic/Evolving Graphs](#3-ssms-for-dynamicevolving-graphs)
4. [Entity Salience and Importance Modeling](#4-entity-salience-and-importance-modeling)
5. [Contradiction and Supersession in Knowledge Bases](#5-contradiction-and-supersession-in-knowledge-bases)
6. [Hierarchical Multi-Scale Temporal Reasoning](#6-hierarchical-multi-scale-temporal-reasoning)
7. [Synthesis: Implications for Signet](#7-synthesis-implications-for-signet)

---

## 1. SSMs for Temporal Knowledge Graphs

### 1.1 DyGMamba

- **Paper**: "DyGMamba: Efficiently Modeling Long-Term Temporal Dependency
  on Continuous-Time Dynamic Graphs with State Space Models"
- **URL**: https://arxiv.org/abs/2408.04713
- **Venue**: 2024

**Architecture**: DyGMamba is the first model to apply Mamba SSMs to
continuous-time dynamic graph (CTDG) representation learning. It uses a
two-level SSM architecture:

1. **Node-Level SSM**: Processes one-hop temporal neighbors of query nodes
   separately. Neighbor features (node embeddings, edge features, temporal
   encodings, co-occurrence frequency features) are concatenated, patched
   (grouped by factor p to reduce sequence length), and fed through Mamba
   layers with residual connections and layer normalization. The SSM's
   input-dependent parameters (B, C, Delta) vary across positions, enabling
   dynamic selection of relevant historical interactions.

2. **Time-Level SSM**: Extracts the k most recent interactions between a
   specific node pair, computes time differences between consecutive
   interactions, encodes them, and processes through a second SSM. Output
   is mean-pooled into a compressed temporal pattern representation.

3. **Dynamic Information Selection**: The time-level output serves as a
   query to weight encoded neighbors via softmax attention, aggregating
   neighbor representations into final node embeddings.

**Temporal Encoding**: Sinusoidal encoding from TGAT using trainable
frequencies and phases.

**Training**: Dynamic link prediction -- predicting whether an interaction
exists between two nodes at a future timestamp given all prior history.

**Metrics**: MRR, Hits@10, Hits@20, AUC-ROC.

**Key Results**: State-of-the-art on most benchmarks (Wikipedia, Reddit,
LastFM, MOOC) versus DyRep, JODIE, TGN, TGAT, GraphMixer, DyGFormer.
Significantly lower GPU memory and training time than Transformer-based
DyGFormer. Linear complexity versus quadratic for attention.

**Signet Relevance**: The two-level architecture maps naturally to Signet's
needs. The node-level SSM is analogous to encoding an entity's interaction
history (which other entities was it mentioned alongside, with what aspects).
The time-level SSM captures the temporal pattern of entity access -- exactly
the "when will this entity be queried next?" prediction we need. The
dynamic information selection mechanism (using temporal patterns to weight
historical neighbors) is directly analogous to how Signet should weight
entity importance based on temporal access patterns.

---

### 1.2 DyG-Mamba

- **Paper**: "DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs"
- **URL**: https://arxiv.org/abs/2408.06966
- **Venue**: 2024

**Architecture**: Distinct from DyGMamba. DyG-Mamba replaces Mamba's
data-dependent step-size parameter with time-span-dependent control signals.
The key insight is that vanilla Mamba lacks sufficient utilization of time
information with irregular intervals.

**Time-Span Control Signals**: Instead of using input features to derive
step sizes, the model computes normalized time-spans between consecutive
timestamps:

    Delta_t_i = SiLU(Linear(cos(omega * (t_{i+1} - t_i) / (tau - t_1))))

This draws from the Ebbinghaus Forgetting Curve: memory of past events is
strongly correlated with time intervals rather than specific details.

**Forgetting Mechanism**: Small time-spans preserve historical hidden states
(h_{i,k} ~ h_{i,k-1}), while larger time-spans enable forgetting through
exponential decay (h_{i,k} ~ -lambda_i^{-1} * u_k), analogous to LSTM
forget gates. This is proven formally in the paper (Theorem 2).

**Parameter Roles**: Delta (step-size) is time-dependent for irregular
sampling; B and C (projection matrices) remain data-dependent for selective
copying of relevant inputs (acting as a causal attention mechanism, Theorem 3).

**Key Results**: Average rank 1.08 across 12 datasets for inductive dynamic
link prediction (vs DyGFormer 2.50). Linear time/memory complexity O(bLd).

**Signet Relevance**: The Ebbinghaus-inspired forgetting mechanism is
exactly what Signet's retention decay needs. Currently Signet uses
`importance * 0.95^ageDays` -- a fixed exponential decay. DyG-Mamba learns
a forgetting curve from data, where the decay rate is determined by the
actual time gaps between entity interactions. This is a direct replacement
for hardcoded decay parameters. The formal theorem proving that small
time-spans preserve state while large gaps enable forgetting is the
mathematical foundation for learned retention decay.

---

### 1.3 TLogic

- **Paper**: "TLogic: Temporal Logical Rules for Explainable Link Forecasting
  on Temporal Knowledge Graphs"
- **URL**: https://arxiv.org/abs/2112.08025
- **Venue**: AAAI 2022

**Architecture**: Extracts temporal logical rules via temporal random walks
on the knowledge graph. Rules take the form of cyclic temporal paths with
time ordering constraints. Link forecasting uses these rules for prediction,
providing explainability that embedding-based methods lack.

**Key Results**: MRR 57.76 on ICEWS14 (vs RE-GCN 52.91, TANGO 50.71).
Works well in inductive settings where learned rules transfer to related
datasets with shared vocabularies.

**Metrics**: MRR, Hits@1/3/10 on ICEWS14, ICEWS18, WIKI, YAGO.

**Signet Relevance**: TLogic's temporal random walks are directly analogous
to how Signet's traversal-primary retrieval (DP-6) walks the knowledge
graph. The rule extraction mechanism could inform how Signet learns which
entity relationship paths are predictive of future queries. The
explainability property matters -- Signet needs to explain why it retrieved
certain memories, and temporal rule chains provide that trace.

---

### 1.4 RE-GCN

- **Paper**: "Temporal Knowledge Graph Reasoning Based on Evolutional
  Representation Learning"
- **URL**: https://arxiv.org/abs/2104.10353
- **Venue**: SIGIR 2021

**Architecture**: Uses a relation-aware GCN to capture structural
dependencies within the KG at each timestamp. The historical KG sequence
is modeled auto-regressively by GRU gate recurrent components. Static
entity properties (types) are incorporated via a static graph constraint.

**Key Results**: Up to 11.46% MRR improvement with 82x speedup over
prior state-of-the-art. Evaluated on six benchmark datasets.

**Signet Relevance**: The GRU-based recurrent modeling of entity evolution
is conceptually similar to what an SSM would do, but with quadratic
complexity for long histories. An SSM could replace the GRU component
while maintaining the relation-aware GCN for structural capture at each
timestamp -- a hybrid architecture worth considering.

---

### 1.5 TANGO

- **Paper**: "Temporal Knowledge Graph Forecasting with Neural ODE"
- **URL**: https://arxiv.org/abs/2101.05151
- **Venue**: 2021

**Architecture**: First to apply neural ODEs to multi-relational dynamic
graphs. The core equation decomposes into two parallel modules:

    dH(t)/dt = f_MGCN(H(t), G(t), t) + w * f_trans(H(t), T(t), G(t), t)

The MGCN component captures structural information through entity-relation
composition. The graph transition layer explicitly models edge
formation/dissolution via a transition tensor T(t) = A(t) - A(t-Delta_t)
where entries are {-1, 0, 1} for dissolution, stability, or formation.

**Continuous-Time Evolution**: Representations integrate over time via
fourth-order Runge-Kutta ODE solvers:

    H(t+Delta_t) - H(t) = integral[t, t+Delta_t] f_TANGO(H(tau)) dtau

**Key Results**: ICEWS05-15 MRR 42.86%, Hits@10 62.34%. Outperforms RE-Net
while requiring ~10x less training time. Novel evaluation on inductive link
prediction (unseen entities) and long horizontal forecasting.

**Signet Relevance**: TANGO's explicit modeling of edge formation and
dissolution (the transition tensor) maps directly to Signet's need to
track when entity relationships form, change, or dissolve. The
continuous-time ODE formulation avoids the discrete snapshot assumption,
which is important because Signet's knowledge graph updates continuously
during sessions, not at fixed intervals.

---

### 1.6 GenTKG

- **Paper**: "GenTKG: Generative Forecasting on Temporal Knowledge Graph
  with Large Language Models"
- **URL**: https://arxiv.org/abs/2310.07793
- **Venue**: NeurIPS 2023 / NAACL 2024 Findings

**Architecture**: Retrieval-augmented generation for temporal KG
forecasting. Uses temporal logical rule-based retrieval (TLR) with
exponentially weighted sampling prioritizing temporally closer edges.
Mines cyclic temporal logical rules, ranks by confidence, retrieves
top-k rule bodies within time windows. Retrieved facts are sequentialized
in temporal ascending order and formatted as natural language for LLM
processing. Uses LoRA for few-shot parameter-efficient instruction tuning.

**Key Results**: Outperforms baselines on Hits@1/3 across ICEWS14, ICEWS18,
GDELT, YAGO. Competitive with only 16 training samples (0.004% of full
data). Cross-domain generalization without retraining.

**Signet Relevance**: The retrieval-augmented approach validates Signet's
hybrid architecture where structured retrieval (knowledge graph traversal)
feeds into LLM processing. The extreme few-shot capability is encouraging
for local-first deployment where training data is limited. The temporal
ordering finding -- that ascending temporal order is the only effective
arrangement -- informs how Signet should present retrieved memories to the
LLM context.

---

### 1.7 Know-Evolve

- **Paper**: "Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge
  Graphs"
- **URL**: https://arxiv.org/abs/1705.05742
- **Venue**: ICML 2017

**Architecture**: Pioneering work on continuous-time dynamic KGs. Models
fact occurrence as a multivariate point process whose intensity function is
modulated by entity embedding scores. Entity representations evolve
non-linearly over time. Predicts both occurrence and recurrence time of
facts.

**Signet Relevance**: The temporal point process formulation -- predicting
*when* a fact will be relevant, not just *whether* -- is exactly what
Signet's predictive scorer needs. The recurrence time prediction is
analogous to predicting when an entity will be queried again.

---

### 1.8 EvoKG

- **Paper**: "EvoKG: Jointly Modeling Event Time and Network Structure for
  Reasoning over Temporal Knowledge Graphs"
- **URL**: https://keg.cs.tsinghua.edu.cn/yuxiao/papers/WSDM22-park-evokg.pdf
- **Venue**: WSDM 2022

**Architecture**: Jointly models event timing and network structure via
recurrent event modeling with temporal neighborhood aggregation. Uses
GNN-based neighborhood aggregation combined with RNN-based recurrent
modeling to capture structural dynamics over time. Temporal embeddings
evolve as events are processed.

**Signet Relevance**: The joint modeling of "when" and "what structure" is
the core challenge for Signet's knowledge graph. EvoKG validates that
these two aspects must be learned together, not separately.

---

### Temporal KG Evaluation Landscape

Standard metrics across all temporal KG work:
- **MRR (Mean Reciprocal Rank)**: Average inverse rank of correct
  predictions. Primary metric.
- **Hits@k**: Proportion of correct answers in top-k (k=1,3,10).
- **Time-aware filtering**: Only removes corrupted triples valid at query
  timestamp (fairer than time-unaware filtering).

Standard datasets: ICEWS14, ICEWS18, ICEWS05-15 (event-based), GDELT
(global events), Wikidata, YAGO (collaborative KBs), WIKI.

**Key takeaway for Signet**: MRR is the metric we should adopt for
evaluating whether Signet's retrieval returns the right memories. We can
define our own "temporal knowledge graph" from the entity/aspect/attribute
structure and evaluate retrieval quality with these established metrics.

---

## 2. SSMs Combined with GNNs

### 2.1 Graph Mamba Networks (GMN)

- **Paper**: "Graph Mamba: Towards Learning on Graphs with State Space
  Models"
- **URL**: https://arxiv.org/abs/2402.08678
- **Venue**: KDD 2024

**Architecture**: A 5-step framework for building GNNs from selective SSMs:

1. **Neighborhood Tokenization**: Each token represents a subgraph sampled
   from a k-hop neighborhood of a target node. This bridges node-level and
   subgraph-level approaches.

2. **Token Ordering**: The i-th token is sampled from the i-hop
   neighborhood. The paper suggests reversing order (largest to smallest
   hop) so inner subgraphs access information from outer, more global
   subgraphs. This implicit ordering from graph topology avoids the
   fundamental challenge of forcing arbitrary graph structure into sequences.

3. **Bidirectional Selective SSM Encoder**: Two Mamba-like SSMs process
   the token sequence in forward and reverse directions. Outputs are
   combined.

4. **Local Encoding**: Incorporates local graph structure.

5. **Positional/Structural Encodings**: Optional -- the paper shows these
   are dispensable when using the above components.

**Key Results**: Outstanding performance on long-range, small-scale,
large-scale, and heterophilic benchmarks. Best accuracy on OGBN-Arxiv with
substantially lower memory than Graph Transformers.

**Signet Relevance**: The neighborhood tokenization strategy directly
applies to how Signet could encode its knowledge graph entities for SSM
processing. Each entity has a multi-hop neighborhood of related entities,
aspects, and attributes. The k-hop tokenization converts this subgraph into
a sequence the SSM can process. The bidirectional processing ensures
information flows both from local context outward and from global context
inward -- matching Signet's need to consider both immediate entity context
and broader graph structure.

---

### 2.2 NeuralWalker

- **Paper**: "Revisiting Random Walks for Learning on Graphs"
- **URL**: https://arxiv.org/abs/2407.01214
- **Venue**: 2024

**Architecture**: Treats graphs as collections of random walk sequences.
Three main components:

1. **Walk Embedder**: Combines node embeddings, projected edge features,
   and positional encodings (identity encoding for node repetitions,
   adjacency encoding for edge connectivity).

2. **Sequence Layer**: Processes walk embeddings. Bidirectional Mamba
   emerged as the top-performing model, consistently outperforming S4,
   1D CNNs, and even Transformers.

3. **Walk Aggregator**: Pools walk features back to nodes via averaging
   across all walks containing each node.

Optional local and global message passing can complement the walk-based
features.

**Theoretical Properties**: With sufficiently long walks (>= 4n^3), the
model can distinguish non-isomorphic graphs. Strictly more expressive than
Weisfeiler-Lehman test and ordinary MPNNs.

**Key Results**: State-of-the-art across 19 datasets. 13% improvement on
PascalVOC-SP and COCO-SP. Bidirectional Mamba consistently the best
sequence model for processing walks.

**Signet Relevance**: This is the most directly relevant architecture for
Signet's graph traversal. Signet's DP-6 traversal-primary retrieval already
walks the knowledge graph. NeuralWalker shows that processing these walks
through a bidirectional Mamba SSM produces state-of-the-art graph
representations. The walk aggregation mechanism (pooling back to nodes)
is how Signet could maintain entity-level representations that incorporate
graph traversal information.

---

### 2.3 Learn to Jump (Adaptive Random Walks)

- **Paper**: "Learn to Jump: Adaptive Random Walks for Long-Range
  Propagation through Graph Hierarchies"
- **URL**: https://arxiv.org/abs/2509.01381
- **Venue**: 2025

**Architecture**: Constructs a multi-level hierarchy using METIS
partitioning. Virtual nodes represent coarsened abstractions. A learnable
MLP-based transition function determines movement probabilities via
Gumbel-Softmax. Walks can "jump" between distant regions through the
hierarchy rather than traversing original edges. Walk sequences are
processed by Mamba SSM.

**Signet Relevance**: Signet's knowledge graph naturally has hierarchical
structure (entities -> aspects -> attributes, and entities form communities).
This architecture shows how to learn traversal policies that can jump
between hierarchy levels -- exactly what Signet needs when a query relates
to a high-level concept that spans multiple entity clusters.

---

### 2.4 HeteGraph-Mamba

- **Paper**: "HeteGraph-Mamba: Heterogeneous Graph Learning via Selective
  State Space Model"
- **URL**: https://arxiv.org/abs/2405.13915
- **Venue**: 2024

**Architecture**: First exploration of selective SSMs for heterogeneous
graphs (graphs with multiple node and edge types). Two-level tokenization:

1. **Instance Level**: Each token is a subgraph containing a target node
   and its meta-path instances. An instance encoder processes node sequences
   within meta-paths using attention-weighted aggregation.

2. **Meta-Path Level**: Different meta-paths are aggregated using learned
   importance weights.

**Ordering Strategies**:
- Inner ordering: Group nodes by type, rank by meta-path instance count
  (reflects importance within type).
- Outer ordering: Extends across types using node degree.

**Multi-Type Handling**: Type-specific linear projections map heterogeneous
features into a unified representation space. Mamba's selective mechanism
filters relevant from irrelevant contexts via data-dependent B/C matrices.

**Key Results**: Outperformed 19 baselines on ogbn-mag (0.5763 accuracy),
DBLP (0.9602 F1), ACM (0.9484 F1).

**Signet Relevance**: Signet's knowledge graph IS a heterogeneous graph.
Entities, aspects, attributes, and dependencies are different node types
with different relation types. HeteGraph-Mamba demonstrates that SSMs can
handle this heterogeneity through meta-path-based tokenization and
type-specific projections. The inner ordering by meta-path instance count
is analogous to ordering entities by their knowledge graph connectivity
(more connected = more important).

---

### 2.5 Graph Topology -> SSM Input: The Encoding Challenge

The fundamental challenge in feeding graph topology to an SSM is that SSMs
are defined on ordered sequences with linear causal structure, while graphs
have complex topology with no canonical node ordering. Several approaches:

**Adjacency-based parameterization**: GSSMs achieve permutation equivariance
by parameterizing the SSM's transition function with the adjacency matrix,
Laplacian, or dynamically learned connectivity. This injects graph structure
into the SSM's dynamics rather than its input sequence.

**Random walks as sequences**: NeuralWalker, Learn-to-Jump, and related
work convert graph topology into sequences via random walks. This naturally
captures long-range dependencies, subgraph structures, and directional
information through sequential node ordering.

**k-hop neighborhood tokenization**: GMN converts multi-hop neighborhoods
into ordered token sequences, using hop distance as the ordering principle.

**Graph-conditioned SSM (GC-SSM)**: Graph-Enhanced Mamba injects
neighborhood context into channel-wise modulation of Mamba's selection
matrices. The graph structure conditions how the SSM processes features
rather than being encoded as input.

**Message-Passing SSMs (MP-SSM)**: Recent work enables permutation-
equivariant, long-range information propagation while preserving message
passing compatibility. The SSM operates within the message-passing
framework rather than replacing it.

**GG-SSMs (Graph-Generating SSMs)**: Dynamically construct graphs from
feature similarities using MST-based sparse construction, then propagate
hidden states along graph paths. Edges represent feature relationships
rather than predetermined topology.

**Key takeaway for Signet**: The most natural approach for Signet is a
combination of (a) random walks for traversal-primary retrieval, processed
by bidirectional Mamba, and (b) adjacency-conditioned SSM dynamics for
entity representation evolution. The graph structure should condition the
SSM's transition matrices (how entities evolve) rather than being forced
into a flat sequence.

---

## 3. SSMs for Dynamic/Evolving Graphs

### 3.1 GraphSSM (NeurIPS 2024)

- **Paper**: "State Space Models on Temporal Graphs: A First-Principles
  Study"
- **URL**: https://arxiv.org/abs/2406.00943
- **Venue**: NeurIPS 2024
- **Code**: https://github.com/EdisonLeeeee/GraphSSM

**Architecture**: The most principled treatment of SSMs on temporal graphs.
Introduces GHIPPO -- a Laplacian-regularized online function approximation
framework that extends HiPPO theory to graphs.

**Core Objective**:

    L_t(Z; G, X, mu) = integral[0,t] ||X(s) - Z(s)||^2 dmu_t(s)
                      + alpha * integral[0,t] Z(s)^T L(s) Z(s) dmu_t(s)

This simultaneously minimizes reconstruction error between node features
and memory states while encouraging smoothness with respect to adjacent
nodes via the Laplacian term L(s). The result is a "diffuse-then-update"
structure where features undergo Laplacian-weighted diffusion before memory
updates.

**Continuous-Time ODE**:

    dU(t)/dt = U(t) * A^T + (I + alpha*L(t))^{-1} * X(t) * B^T

Where A and B derive from HiPPO theory. The system is LTI over each
interval where graph structure remains fixed, becoming piecewise-linear
overall.

**Handling Graph Mutations**: Edge additions/deletions create discontinuous
jumps in L(t) at unobserved times. Between snapshots, structural mutations
produce intermediate graph states whose timing remains hidden. GraphSSM
addresses this via a mixed discretization strategy:

- **Ordinary ZOH**: Apply GNN to final snapshot directly
- **Feature Mixing**: Blend consecutive snapshot features before diffusion
- **Representation Mixing**: Mix GNN outputs from consecutive snapshots

**Three Variants**:
- GraphSSM-S4 (SISO): Single-input, single-output. Best performer.
- GraphSSM-S5 (MIMO): Multi-input, multi-output extension.
- GraphSSM-S6 (Selective): Input-controlled time intervals and state
  matrices. Interestingly, the selective mechanism may not be optimal for
  graph data specifically.

**Key Results**: 14% average gains in Micro-F1 and 2% in Macro-F1 on
Reddit/DBLP-10. Successfully scales to arXiv and Tmall where competing
models hit memory constraints. HIPPO initialization (S4D-Real) yields
optimal results.

**Signet Relevance**: GraphSSM is the theoretical foundation for everything
we want to build. The Laplacian-regularized objective formalizes exactly how
a knowledge graph's topology should influence memory compression. The
"diffuse-then-update" mechanism means entity representations are smoothed
by their graph neighborhood before being compressed into the SSM state.
The mixed discretization strategy handles the exact problem Signet faces:
the knowledge graph mutates continuously during sessions, but we only
observe snapshots. The finding that S4 outperforms S6 (selective Mamba) on
graph data is important -- it suggests that for graph-structured temporal
data, the HiPPO initialization and structured state matrix may matter more
than data-dependent selectivity.

---

### 3.2 DG-Mamba (AAAI 2025)

- **Paper**: "DG-Mamba: Robust and Efficient Dynamic Graph Structure
  Learning with Selective State Space Models"
- **URL**: https://arxiv.org/abs/2412.08160
- **Venue**: AAAI 2025
- **Code**: https://github.com/RingBDStack/DG-Mamba

**Architecture**: Addresses dynamic graphs suffering from structural
incompleteness, noise, and redundancy. Three components:

1. **Kernelized Message-Passing Operator**: Reduces quadratic attention
   complexity to linear via Positive Random Features kernel approximation.
   Decomposes learned weights into intra-graph (within snapshot) and
   inter-graph (temporal transitions) adjacency matrices.

2. **Cross-Snapshot SSM Discretization**: Models the dynamic graph as a
   self-contained continuous-time system. Discretizes using inter-graph
   adjacency:

       Delta_bar <- unsqueeze_N(Delta) * W * A_hat_inter^{1:T}

   This allows selective scan to attend to graph sequences while respecting
   temporal dependencies.

3. **Self-Supervised PRI (Principle of Relevant Information)**: Balances
   information preservation and noise reduction. Spatial regularization
   (intra-graph entropy + edge fidelity). Temporal regularization (KL
   divergence between SSM and message-passing outputs).

**Adversarial Robustness**: Under 20% edge removal: 68-79% AUC retained.
Under targeted poisoning (3 perturbations): 30-35% AUC drop vs 40-42% for
baselines.

**Signet Relevance**: The PRI regularization is directly relevant to
Signet's knowledge graph quality. Signet's entity bloat problem (43,520
entities, target <1,000) is exactly structural noise and redundancy. The
intra-graph entropy term encourages sparse, interpretable structures --
which is what entity pruning achieves heuristically. A learned version
via DG-Mamba's PRI would be more principled. The adversarial robustness is
also relevant: Signet's knowledge graph receives noisy LLM-extracted
entities that may contain errors, and the system needs to be robust to this
noise.

---

### 3.3 GG-SSMs (Graph-Generating State Space Models)

- **Paper**: "GG-SSMs: Graph-Generating State Space Models"
- **URL**: https://arxiv.org/abs/2412.12423
- **Venue**: 2024

**Architecture**: Rather than feeding a fixed graph into an SSM, GG-SSMs
dynamically construct graphs from feature similarities. Edge weights are
computed via cosine dissimilarity, then sparsified via Chazelle's MST
algorithm (near-linear complexity). Hidden states propagate along MST paths,
aggregating contributions from all nodes through cumulative state
transitions.

**Key Results**: ImageNet-1K 84.9% top-1 accuracy. State-of-the-art on
KITTI-15 optical flow (2.77% error). Superior on six time series
forecasting datasets.

**Signet Relevance**: The core insight -- that data-driven graph
construction outperforms predetermined scanning -- suggests Signet should
dynamically construct entity relationship graphs based on embedding
similarity rather than relying solely on explicitly extracted relationships.
The MST sparsification naturally produces a tree structure that mirrors the
hierarchical nature of Signet's entity/aspect/attribute graph.

---

### 3.4 Zep/Graphiti: Temporal Knowledge Graph for Agent Memory

- **Paper**: "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"
- **URL**: https://arxiv.org/abs/2501.13956
- **Venue**: January 2025

**Architecture**: Most directly comparable system to Signet. Three-tier
hierarchical knowledge graph:

1. **Episode Subgraph**: Raw input data as episodic nodes (messages, text,
   JSON). Non-lossy store from which entities are extracted.

2. **Semantic Entity Subgraph**: Extracted and resolved entities with
   relationship edges. Mirrors human semantic memory.

3. **Community Subgraph**: Clusters of strongly connected entities with
   high-level summaries.

**Bi-Temporal Data Model**: Each edge stores four timestamps:
- t'_created, t'_expired (system timestamps)
- t_valid, t_invalid (world-state timestamps)

This enables accurate handling of retroactive data, corrections, and
supersession of facts.

**Contradiction/Supersession Handling**: When new information contradicts
existing facts with overlapping timeframes, the system invalidates affected
edges by setting t_invalid to the t_valid of the invalidating edge. New
information takes precedence, preserving historical records.

**Entity Resolution**: Multi-stage: embedding-based cosine similarity,
full-text search on names/summaries, LLM evaluation against episode context.

**Key Results**: 94.8% accuracy on DMR (vs MemGPT 93.4%). 71.2% on
LongMemEval (vs 60.2% baseline). 90% latency reduction. 184% improvement
on single-session preference questions.

**Signet Relevance**: Zep is the closest architectural peer. The bi-temporal
data model is something Signet should adopt -- tracking both when facts
enter the system and when they were true in the world. The contradiction
handling via edge invalidation with preserved history is the exact mechanism
Signet needs for its supersession problem. However, Zep uses LLM-based
processing throughout, while Signet's thesis is that an SSM can handle
temporal reasoning locally. The question is whether an SSM can learn the
contradiction detection that Zep does via LLM calls.

---

## 4. Entity Salience and Importance Modeling

### 4.1 Temporal Evolution of Entity Importance

The temporal KG literature reveals several approaches to modeling entity
importance evolution:

**Statistical Estimation with Time Decay**: Weight historical entities
through statistical frequency combined with exponential decay. More recent
interactions contribute more to entity importance scores.

**Relation-Aware Attention**: Dynamically adapt importance weights of
relations between entities. Different relations have different importance
at different times.

**Evolutional Representation**: Entities' representations evolve at each
timestamp through recursive modeling. The "importance" of an entity is
implicit in its representation magnitude and connectivity.

**Key Models**:
- RE-GCN: GRU-based recurrent evolution of entity embeddings
- CEN: Extends RE-GCN with static attribute integration
- TEA-GNN: Temporal entity-aware attention

### 4.2 Personalized Entity Importance

**User-Specific Graph Weighting** (from recommendation systems): Compute
user-specific item embeddings by applying a trainable function that
identifies important knowledge graph relationships for a given user. The
KG is transformed into a user-specific weighted graph before GNN
processing.

**Personalized Entity Repositories** (Google's approach): Entities ranked
by user location, search history, content viewing patterns, time of day,
and cross-device signals.

**Factors Affecting Entity Salience**: Position/prominence in context,
frequency and distribution, contextual relevance to surrounding entities,
semantic relationships with other high-salience entities.

**Signet Relevance**: Signet's memory system is inherently personalized --
each user has their own knowledge graph. The entity importance should be
user-specific: a developer cares about different entities than a writer.
The approach of transforming the knowledge graph into a user-weighted
version (where weights are learned from interaction patterns) maps directly
to the predictive scorer's task. The SSM can learn these user-specific
importance weights from the temporal pattern of entity access.

### 4.3 SSMs for Recommendation Systems (Analogous to Memory Retrieval)

**Mamba4Rec**: First SSM-based sequential recommendation system. Linear
complexity for processing long behavior sequences while capturing complex
dependencies.

**SIGMA (Selective Gated Mamba, AAAI)**: Three components:
- Partially Flipped Mamba (PF-Mamba): Bidirectional context by partially
  reversing sequences -- adaptively reverses first n items while preserving
  last r items for recent preferences.
- Dense Selective Gate: Input-dependent gating combining both directions.
- Feature Extract GRU: Captures short-term dependencies where Mamba
  struggles with limited data.

Results: 0.76%-8.82% improvement over all baselines. Linear O(N) complexity.

**SS4Rec**: Continuous-time sequential recommendation. Two parallel SSM
layers:
- Time-aware SSM (S5-based): Variable discretization where actual
  interaction timestamps scale learnable timescale parameters.
- Relation-aware SSM (Mamba-based): Input-dependent selection for
  sequential dependencies.

Novel contribution: Predicts optimal interaction *timing*, not just next
item.

**TiM4Rec**: Time-aware enhancement integrated into State Space Duality
(Mamba-2) framework for sequential recommendation.

**Signet Relevance**: Sequential recommendation IS memory retrieval. The
user's "interaction sequence" is their session history. The "items" are
memories/entities. The "next item prediction" is "which memory should be
injected next." SIGMA's bidirectional approach (preserving recent
preferences while accessing historical context) maps to Signet's need to
balance recent session context with long-term memory. SS4Rec's time-aware
SSM with variable discretization based on actual timestamps is exactly
the mechanism Signet needs for its irregular interaction intervals.

### 4.4 Memory Decay and Forgetting Curves

**Ebbinghaus Forgetting Curve**: R = e^{-t/S} where R is retention, t is
time since learning, S is memory strength. Foundation for temporal decay
in both memory and recommendation systems.

**FMRES Model**: Integrates cognitive-inspired Ebbinghaus curve with item
attributes to model personalized forgetting patterns. Users have different
forgetting tendencies based on interaction frequency and engagement. A
memory replay mechanism revives forgotten-but-valuable items at appropriate
moments.

**Deep Knowledge Tracing**: Models memory retention with behavioral
variables (stress, sleep, learning type, information complexity) extending
Ebbinghaus's curve.

**Signet Relevance**: Signet currently uses a fixed `0.95^ageDays` decay.
The literature strongly suggests this should be:
1. Personalized per-user (different users forget different things at
   different rates)
2. Entity-specific (some entities are more memorable than others based
   on their attributes)
3. Interaction-dependent (entities that are accessed more frequently have
   higher "memory strength" S)
4. Include a replay mechanism (proactively resurface forgotten-but-important
   entities)

An SSM naturally learns all four properties from the temporal sequence
of entity accesses, without needing explicit decay parameters.

---

## 5. Contradiction and Supersession in Knowledge Bases

### 5.1 Temporal Consistency in Knowledge Graphs

**PaTeCon**: Pattern-based Temporal Constraint mining. Automatically
generates temporal constraints from graph patterns and statistical
information without human experts. Detects when facts violate temporal
constraints. Benchmarked on annotated Wikidata and Freebase.

**ETC Framework**: Uses maximum weight clique to detect temporal conflicts
in uncertain temporal knowledge graphs and eliminate them to achieve the
most probable consistent graph.

**TeCre (Temporal Conflict Resolution)**: Embedding-based approach where
temporal knowledge graph embeddings capture temporal validity patterns.
Detects and resolves conflicts by comparing fact embeddings with their
temporal context.

**Explainable Temporal Fact Validation**: Discovers simple and complex
temporal constraints that capture temporal consistency of facts within an
entity's timeline.

### 5.2 Bi-Temporal Data Models

The Zep architecture (Section 3.4) provides the most practical approach:
four timestamps per edge (created/expired system time, valid/invalid world
time). New information invalidates old edges by setting their t_invalid.
Historical records are preserved, not overwritten.

### 5.3 Knowledge Conflicts in LLMs (EMNLP 2024 Survey)

**Taxonomy**:
1. **Context-Memory Conflict**: External knowledge contradicts model's
   parametric knowledge. Root cause: temporal misalignment (training data
   is from the past, world has changed).
2. **Inter-Context Conflict**: Multiple external sources contradict each
   other.
3. **Intra-Memory Conflict**: Model generates inconsistent responses to
   semantically equivalent queries.

**Temporal Misalignment**: Identified as a fundamental driver. Models have
fixed parametric knowledge while external information evolves. Three
mitigation approaches, each with limitations:
- Knowledge editing: Can introduce new conflicts
- RAG: Maintains unresolved conflicts
- Continual learning: Catastrophic forgetting

**Detection Challenges**: LLMs struggle to identify specific conflicting
segments. Exhibit confirmation bias favoring parametric memory. Good at
detecting conflicts exist, poor at precise localization.

**Resolution Strategies**: Faithful-to-context (fine-tuning, prompting,
contrastive decoding), discriminating misinformation (query augmentation),
disentangling sources (separate context vs parametric answers), improving
factuality (combine both sources).

### 5.4 Belief Revision Frameworks

**AGM Framework** (Alchourron-Gardenfors-Makinson): Formal framework for
updating beliefs. Core principle: minimal change -- preserve as much
information as possible. Three operations:
- Expansion: Add new belief
- Contraction: Remove belief
- Revision: Add new belief, ensuring consistency

**Signet Relevance**: The contradiction detection problem for Signet breaks
into three layers:

1. **Structural Detection**: Use temporal constraints (PaTeCon-style) to
   automatically detect when entities have contradictory attributes at the
   same time. This is rule-based and cheap.

2. **Embedding-based Detection**: Use TeCre-style embedding comparison to
   detect when new facts are semantically inconsistent with existing facts,
   even without explicit temporal overlap.

3. **SSM-based Anomaly Detection**: Treat the knowledge graph update stream
   as a time series. Contradictions are anomalies -- deviations from the
   learned normal pattern of fact evolution. The SSM learns what "normal"
   entity evolution looks like and flags deviations. This is exactly what
   the Mamba-based time series anomaly detection literature demonstrates
   (Section 5.5).

The bi-temporal data model (Zep's approach) should be adopted regardless:
never delete facts, only invalidate them with timestamps. The SSM can then
learn supersession patterns from the invalidation history.

### 5.5 SSMs for Anomaly Detection (Contradictions as Anomalies)

**Mamba-TSAD**: Joint selective state space model and detrending for robust
time series anomaly detection. Uses DMamba blocks with Hodrick-Prescott
trend filtering and adaptive moving average. Detects anomalies as
reconstruction error exceeding learned thresholds.

**Key insight**: S6 (Mamba's selective mechanism) makes parameters
time-dependent, removing LTI constraints. This enables adaptive context
selection for anomaly detection. Achieved 75.37% F1-AF with 6.2% relative
improvement over baselines.

**MambaAD**: NeurIPS 2024. Multi-class unsupervised anomaly detection using
Mamba. Pioneering application to detecting anomalies across multiple
categories simultaneously.

**Signet Relevance**: If we model the knowledge graph update stream as a
time series of (entity, attribute, value, timestamp) tuples, contradictions
appear as anomalies in this stream. An SSM learns the normal pattern of how
entities evolve and what attribute values are expected. When a new fact
contradicts this learned pattern, the reconstruction error spikes, flagging
it for review. This transforms contradiction detection from a rule-based
problem into a learned, adaptive one.

---

## 6. Hierarchical Multi-Scale Temporal Reasoning

### 6.1 HiSS (Hierarchical State Space Models)

- **Paper**: "Hierarchical State Space Models for Continuous
  Sequence-to-Sequence Modeling"
- **URL**: https://arxiv.org/abs/2402.10211
- **Venue**: ICML 2024

**Architecture**: Stacks two SSMs operating at different temporal
resolutions. Input sequence is divided into equal-sized chunks of size k:

1. **Low-level SSM**: Processes each chunk independently, extracting local
   temporal patterns.
2. **High-level SSM**: Operates on condensed sequence of chunk-level
   features (final timestep of each chunk), capturing global dynamics.

The low-level SSM learns effective temporally-local representations while
the high-level SSM focuses on longer-term patterns. This explicitly
addresses multi-frequency phenomena.

**Key Results**: ~23% median MSE improvement over best flat SSMs. ~10% over
Transformers, ~14% over LSTMs. ~9.8% over other hierarchical variations.
Superior data efficiency (outperforms on 30-50% dataset subsets).

**Signet Relevance**: This is the exact architecture for Signet's session
-> arc -> epoch hierarchy:
- Low-level SSM = session-level (processes individual session interactions)
- High-level SSM = arc-level (operates on session summaries)
- A third level could be added for epoch-level patterns

The chunking mechanism maps directly to how sessions are natural temporal
boundaries. The "rarified chunk feature sequence" (extracting the final
state of each session) is exactly what session summaries provide.

---

### 6.2 MS-SSM (Multi-Scale State Space Model)

- **Paper**: "MS-SSM: A Multi-Scale State Space Model for Efficient
  Sequence Modeling"
- **URL**: https://arxiv.org/abs/2512.23824
- **Venue**: 2025

**Architecture**: Decomposes input sequences into multiple scales using
Stationary Wavelet Transform (SWT). Each scale gets its own independent
SSM with specialized dynamics:

- **Array of (S+2) independent SSMs** operating in parallel
- **Scale-dependent initialization**: Lower resolutions (higher s) have
  eigenvalues closer to 1 for long-range dependencies. Higher resolutions
  have smaller eigenvalues for local dynamics.
- **Scale Mixer**: Weighted summation with input-dependent weights combines
  outputs: z_t = E_t * y_t where E_t = Linear(x_t).

**Key Results**: 2x accuracy improvement on ListOps (hierarchical reasoning)
over Mamba (63.04% vs 38.02%). 2.4x higher mean mixing distance indicating
superior long-range dependency capture. Competitive on ECG classification.

**Signet Relevance**: The multi-scale wavelet decomposition maps to
Signet's temporal hierarchy: fine-grained (within-session interaction
patterns), medium-grained (session-to-session patterns), coarse-grained
(long-term behavioral trends). The scale-dependent initialization is
critical -- session-level SSMs need fast dynamics (small eigenvalues) to
track rapid context changes, while epoch-level SSMs need slow dynamics
(eigenvalues near 1) to maintain long-term state. The 2x improvement on
hierarchical reasoning validates that multi-scale SSMs are necessary for
nested temporal structures.

---

### 6.3 SST (Multi-Scale Hybrid Mamba-Transformer)

- **Paper**: "SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time
  Series Forecasting"
- **URL**: https://arxiv.org/abs/2404.14757
- **Venue**: 2024

**Architecture**: Assigns specialized roles to different model types:

- **Patterns Expert (Mamba)**: Processes coarse-grained long-range data to
  extract persistent trends. Linear O(L) complexity.
- **Variations Expert (Local Window Transformer)**: Handles fine-grained
  short-range data for local fluctuations. O(w*S) complexity.
- **Long-Short Router**: Learns adaptive weighting via softmax over two
  weights (p_L, p_S) indicating relative importance of long-range patterns
  vs short-range variations.

**Key Results**: 13.75% MSE improvement on ETTm1. Handles 6,000 timesteps
vs 3,240 for PatchTST and 336 for vanilla Transformer.

**Signet Relevance**: The expert routing concept applies to Signet's
retrieval: use Mamba for long-range pattern detection (which entities are
persistently important) and a local attention mechanism for short-range
context (what was just discussed in this session). The router learns when
to prioritize each -- sometimes the current session context matters most,
sometimes the long-term pattern dominates.

---

### 6.4 ms-Mamba and M-Mamba

**ms-Mamba**: Incorporates multiple temporal scales by using multiple Mamba
blocks with different sampling rates. Outperforms state-of-the-art
Transformer and Mamba models.

**M-Mamba**: Multi-resolution Mamba for long-term forecasting. High
resolutions capture fine-grained local patterns, low resolutions capture
broader trends. Adaptive fusion weights emphasize informative resolutions.

---

### 6.5 Mamba-3 (ICLR 2026)

- **Paper**: "Mamba-3: Improved Sequence Modeling using State Space
  Principles"
- **URL**: https://arxiv.org/abs/2603.15569
- **Venue**: ICLR 2026

**Three Improvements**:

1. **Trapezoidal Discretization**: Replaces first-order Exponential-Euler
   with second-order exponential-trapezoidal. Creates a dependency on both
   current and previous input (implicit convolution of size 2). Removes
   the short causal convolution that was a fixture since Mamba-1.

2. **Complex-Valued State Updates**: Discretized complex SSM is
   mathematically equivalent to a real-valued SSM with data-dependent
   Rotary Positional Embeddings (RoPE). Achieves 100% on parity task
   (Mamba-2: ~0.90%). Retains speed of real arithmetic with expressiveness
   of complex dynamics.

3. **MIMO Formulation**: Multi-Input Multi-Output. Switches from
   outer-product to matrix-multiplication state updates. Up to 4x more
   parallel operations per step, exploiting idle GPU arithmetic units.
   Increases compute per memory load, pushing into compute-bound regime.

**Key Results**: At 1.5B scale, +0.6pp over Gated DeltaNet, +1.8pp with
MIMO. Comparable perplexity to Mamba-2 with half the state size.

**Signet Relevance**: Mamba-3 is the SSM architecture we should target.
The complex-valued state updates enable tracking periodic patterns (entity
relevance that cycles) which real-valued SSMs cannot capture. The MIMO
formulation means we can process multiple entity streams simultaneously.
The trapezoidal discretization provides better numerical accuracy for the
continuous-time entity evolution we need to model. The half-state-size
result is critical for Signet's local-first constraint -- smaller states
mean smaller sidecar models.

---

## 7. Synthesis: Implications for Signet

### 7.1 Architecture Proposal

Based on this research, the optimal SSM architecture for Signet's knowledge
graph combines elements from multiple papers:

**Layer 1: Graph-Conditioned SSM for Entity Evolution** (from GraphSSM)
- Use GHIPPO's Laplacian-regularized objective to compress entity histories
- The knowledge graph's adjacency matrix conditions the SSM's transition
  dynamics
- Mixed discretization handles unobserved graph mutations between sessions
- Target: Mamba-3 with complex-valued states for periodic pattern capture

**Layer 2: Walk-Based SSM for Traversal** (from NeuralWalker + Learn-to-Jump)
- Generate random walks on the knowledge graph for each retrieval query
- Process walks through bidirectional Mamba to produce entity relevance
  scores
- Learnable jump mechanism for hierarchical traversal (entity -> aspect ->
  attribute)
- Walk aggregation pools back to entity-level representations

**Layer 3: Hierarchical Multi-Scale SSM** (from HiSS + MS-SSM)
- Session-level SSM: Fast dynamics, tracks within-session context
- Arc-level SSM: Medium dynamics, captures session-to-session patterns
- Epoch-level SSM: Slow dynamics, maintains long-term behavioral state
- Scale-dependent initialization following MS-SSM's eigenvalue strategy

**Layer 4: Anomaly Detection for Contradictions** (from Mamba-TSAD)
- Model KG update stream as time series
- SSM learns normal entity evolution patterns
- Reconstruction error flags contradictions for review
- Bi-temporal timestamps (from Zep) preserve history

### 7.2 Training Data

The training signal comes from Signet's own feedback loop:
- **Positive signal**: Memory was injected and agent feedback was positive
- **Negative signal**: Memory was injected and agent feedback was negative
  or memory was available but not retrieved
- **Temporal signal**: The sequence of entity accesses across sessions
- **Structural signal**: Knowledge graph topology changes over time

This matches the vision: "the agent itself telling us what helped, on every
prompt, accumulated across thousands of sessions."

### 7.3 Evaluation Metrics

Adopt from temporal KG literature:
- **MRR**: Mean Reciprocal Rank of correct memory retrieval
- **Hits@k**: Proportion of correct memories in top-k retrieved
- **Temporal precision**: Whether retrieved memories are temporally
  appropriate (not superseded, not contradicted)
- **Anomaly F1**: Precision/recall of contradiction detection

### 7.4 Key Insights for Implementation

1. **DyG-Mamba's Ebbinghaus forgetting** is the learned replacement for
   `0.95^ageDays`. Time gaps between entity accesses directly control
   forgetting, not fixed exponential decay.

2. **GraphSSM's S4 > S6 finding** suggests HiPPO-initialized structured
   matrices may outperform selective Mamba for graph-structured data.
   Test both.

3. **Zep's bi-temporal model** should be adopted immediately, independent
   of SSM work. Four timestamps per edge (system created/expired, world
   valid/invalid) enables proper contradiction handling.

4. **HeteGraph-Mamba's meta-path tokenization** handles Signet's
   heterogeneous entity types. Different node types (entity, aspect,
   attribute) need type-specific projections into a unified space.

5. **NeuralWalker's bidirectional Mamba** consistently outperformed all
   alternatives for processing graph walks. This validates our DP-6
   traversal-primary approach and suggests the walk sequences should be
   processed by Mamba, not just used for retrieval ordering.

6. **GenTKG's temporal ascending order** is the only effective arrangement
   for presenting temporal facts. Signet's memory injection should always
   present memories in chronological order.

7. **SIGMA's partial flip** for bidirectional context without full
   reversal is a practical trick: reverse older memories while preserving
   recent context in original order.

8. **MS-SSM's 2x improvement on hierarchical reasoning** validates that
   multi-scale is not optional -- a flat SSM cannot capture Signet's
   nested session/arc/epoch structure.

### 7.5 Open Questions

1. Can a 2-10M parameter SSM learn contradiction detection that currently
   requires LLM calls (Zep's approach)?

2. Does GraphSSM's Laplacian regularization compose with Mamba-3's complex
   states?

3. What is the minimum number of sessions needed before the SSM's learned
   forgetting curve outperforms the hardcoded `0.95^ageDays`?

4. Can the hierarchical SSM (HiSS-style) be trained incrementally as new
   sessions arrive, or does it require batch retraining?

5. How does the MST-based graph construction (GG-SSMs) compare to Signet's
   explicit entity extraction for building the knowledge graph?

---

## Sources

### SSMs for Temporal Knowledge Graphs
- [DyGMamba](https://arxiv.org/abs/2408.04713) -- Node-level + time-level SSM for CTDGs
- [DyG-Mamba](https://arxiv.org/abs/2408.06966) -- Time-span control signals, Ebbinghaus forgetting
- [TLogic](https://arxiv.org/abs/2112.08025) -- Temporal logical rules via random walks (AAAI 2022)
- [RE-GCN](https://arxiv.org/abs/2104.10353) -- Recurrent evolution GCN (SIGIR 2021)
- [TANGO](https://arxiv.org/abs/2101.05151) -- Neural ODEs for temporal KG forecasting
- [GenTKG](https://arxiv.org/abs/2310.07793) -- RAG for temporal KG (NeurIPS 2023 / NAACL 2024)
- [Know-Evolve](https://arxiv.org/abs/1705.05742) -- Deep temporal reasoning (ICML 2017)
- [EvoKG](https://keg.cs.tsinghua.edu.cn/yuxiao/papers/WSDM22-park-evokg.pdf) -- Joint event time + structure (WSDM 2022)
- [Temporal KG Survey](https://arxiv.org/abs/2403.04782) -- Comprehensive survey of 10 method categories

### SSMs Combined with GNNs
- [Graph Mamba Networks](https://arxiv.org/abs/2402.08678) -- 5-step GMN framework (KDD 2024)
- [NeuralWalker](https://arxiv.org/abs/2407.01214) -- Random walks + bidirectional Mamba
- [Learn to Jump](https://arxiv.org/abs/2509.01381) -- Adaptive walks with hierarchical jumps + Mamba
- [HeteGraph-Mamba](https://arxiv.org/abs/2405.13915) -- Heterogeneous graph SSM
- [GG-SSMs](https://arxiv.org/abs/2412.12423) -- Graph-generating state space models
- [MP-SSM](https://arxiv.org/abs/2505.18728) -- Message-passing state space models
- [Graph-Aware SSM survey](https://www.emergentmind.com/topics/graph-aware-state-space-model)
- [Graph SSM survey](https://www.emergentmind.com/topics/graph-state-space-models)

### SSMs for Dynamic/Evolving Graphs
- [GraphSSM](https://arxiv.org/abs/2406.00943) -- GHIPPO, Laplacian-regularized SSM (NeurIPS 2024)
- [DG-Mamba](https://arxiv.org/abs/2412.08160) -- Robust dynamic graph structure learning (AAAI 2025)
- [Zep/Graphiti](https://arxiv.org/abs/2501.13956) -- Temporal KG for agent memory (January 2025)

### Entity Salience and Importance
- [SIGMA](https://arxiv.org/abs/2408.11451) -- Selective Gated Mamba for recommendation (AAAI)
- [SS4Rec](https://arxiv.org/abs/2502.08132) -- Continuous-time SSM for sequential recommendation
- [Mamba4Rec](https://arxiv.org/abs/2403.03900) -- First SSM-based sequential recommender
- [TiM4Rec](https://arxiv.org/abs/2409.16182) -- Time-aware Mamba-2 for recommendation
- [FMRES](https://www.sciencedirect.com/science/article/abs/pii/S0306457325000123) -- Ebbinghaus curve + memory replay
- [Personalized Entity Resolution (Amazon)](https://arxiv.org/abs/2104.02667)

### Contradiction and Supersession
- [PaTeCon](https://arxiv.org/abs/2312.11053) -- Pattern-based temporal constraint mining
- [TeCre](https://www.mdpi.com/2078-2489/14/3/155) -- Temporal conflict resolution via embeddings
- [Knowledge Conflicts Survey](https://arxiv.org/abs/2403.08319) -- EMNLP 2024 comprehensive survey
- [ETC Framework](https://link.springer.com/chapter/10.1007/978-3-030-02922-7_23) -- Maximum weight clique conflict detection
- [Temporal Fact Validation](https://link.springer.com/chapter/10.1007/978-3-031-94575-5_13) -- Constraint discovery
- [Uncertainty in KG Construction](https://drops.dagstuhl.de/storage/08tgdk/tgdk-vol003/tgdk-vol003-issue001/TGDK.3.1.3/TGDK.3.1.3.pdf) -- Survey on uncertainty management
- [Belief Revision](https://en.wikipedia.org/wiki/Belief_revision) -- AGM framework

### Hierarchical Multi-Scale Temporal Reasoning
- [HiSS](https://arxiv.org/abs/2402.10211) -- Hierarchical SSMs for continuous prediction (ICML 2024)
- [MS-SSM](https://arxiv.org/abs/2512.23824) -- Multi-scale wavelet SSM
- [SST](https://arxiv.org/abs/2404.14757) -- Hybrid Mamba-Transformer experts
- [Mamba-3](https://arxiv.org/abs/2603.15569) -- Complex states, trapezoidal, MIMO (ICLR 2026)
- [ms-Mamba](https://arxiv.org/abs/2504.07654) -- Multi-scale Mamba for time series
- [M-Mamba](https://link.springer.com/article/10.1007/s11227-026-08224-y) -- Multi-resolution Mamba
- [Mamba-TSAD](https://arxiv.org/abs/2405.19823) -- SSM for time series anomaly detection

### SSMs for Time Series Anomaly Detection
- [Mamba-TSAD](https://arxiv.org/abs/2405.19823) -- Selective SSM + detrending (SPL 2024)
- [MambaAD](https://github.com/lewandofskee/MambaAD) -- Multi-class unsupervised anomaly detection (NeurIPS 2024)
- [Reconstruction-based SSM](https://arxiv.org/abs/2303.03324) -- SSM anomaly detection via reconstruction
