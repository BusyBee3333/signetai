#!/usr/bin/env python3
"""
SSM Proof of Concept for Signet Memory Prediction

Standalone benchmark comparing a selective SSM (Mamba-style) against a
baseline MLP on synthetic memory interaction sequences. Feature vectors
match the exact 17-dim layout from packages/predictor/src/protocol.rs.

No changes to the daemon or production codebase. Pure validation.

Usage:
    pip install torch numpy
    python ssm_proof_of_concept.py                          # hand-crafted synthetic
    python ssm_proof_of_concept.py --data scenarios.jsonl   # LLM-generated data
    python ssm_proof_of_concept.py --compare scenarios.jsonl # side-by-side comparison
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Feature dimensions (must match protocol.rs FEATURE_DIM = 17)
    feature_dim: int = 17
    embed_dim: int = 64
    ssm_state_dim: int = 32
    conv_kernel: int = 4
    num_heads: int = 5  # relevance, significance, retention, traversal, contradiction

    # Synthetic data
    num_sequences: int = 2000
    seq_len: int = 50
    num_candidates: int = 20
    canary_ratio: float = 0.3

    # Training
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Evaluation
    top_k: int = 5
    snr_levels: list = field(default_factory=lambda: [30, 20, 10, 5, 0, -5, -10])

    # Phase 0 gates
    gate_hrk_min: float = 0.60
    gate_dcg_improvement: float = 0.10
    gate_latency_max_ms: float = 5.0
    gate_canary_min_pass: int = 5


# ---------------------------------------------------------------------------
# Feature vector layout (mirrors protocol.rs)
# ---------------------------------------------------------------------------
# [0]  log(age_days)
# [1]  importance
# [2]  log(access_count + 1)
# [3]  tod_sin
# [4]  tod_cos
# [5]  dow_sin
# [6]  dow_cos
# [7]  moy_sin
# [8]  moy_cos
# [9]  log(max(0, session_gap_days) + 1)
# [10] is_embedded
# [11] is_superseded
# [12] entity_slot (normalized 0-1)
# [13] aspect_slot (normalized 0-1)
# [14] is_constraint
# [15] log(structural_density + 1)
# [16] is_ka_traversal

FEATURE_NAMES = [
    "log_age", "importance", "log_access", "tod_sin", "tod_cos",
    "dow_sin", "dow_cos", "moy_sin", "moy_cos", "log_session_gap",
    "is_embedded", "is_superseded", "entity_slot", "aspect_slot",
    "is_constraint", "log_structural_density", "is_ka_traversal",
]


# ---------------------------------------------------------------------------
# Synthetic data generator with planted canary patterns
# ---------------------------------------------------------------------------

class SyntheticGenerator:
    """Generates interaction sequences with 7 planted canary patterns."""

    def __init__(self, cfg: Config, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)

    def _base_features(self, n: int) -> np.ndarray:
        """Generate n random feature vectors in protocol.rs layout."""
        feats = np.zeros((n, self.cfg.feature_dim), dtype=np.float32)
        feats[:, 0] = np.log1p(self.rng.exponential(30, n))      # log_age
        feats[:, 1] = self.rng.beta(2, 5, n)                      # importance
        feats[:, 2] = np.log1p(self.rng.poisson(3, n))            # log_access
        hour = self.rng.uniform(0, 24, n)
        feats[:, 3] = np.sin(2 * np.pi * hour / 24)               # tod_sin
        feats[:, 4] = np.cos(2 * np.pi * hour / 24)               # tod_cos
        dow = self.rng.uniform(0, 7, n)
        feats[:, 5] = np.sin(2 * np.pi * dow / 7)                 # dow_sin
        feats[:, 6] = np.cos(2 * np.pi * dow / 7)                 # dow_cos
        moy = self.rng.uniform(0, 12, n)
        feats[:, 7] = np.sin(2 * np.pi * moy / 12)                # moy_sin
        feats[:, 8] = np.cos(2 * np.pi * moy / 12)                # moy_cos
        feats[:, 9] = np.log1p(self.rng.exponential(7, n))        # log_session_gap
        feats[:, 10] = (self.rng.random(n) > 0.3).astype(float)   # is_embedded
        feats[:, 11] = (self.rng.random(n) > 0.9).astype(float)   # is_superseded
        feats[:, 12] = self.rng.random(n)                          # entity_slot
        feats[:, 13] = self.rng.random(n)                          # aspect_slot
        feats[:, 14] = (self.rng.random(n) > 0.8).astype(float)   # is_constraint
        feats[:, 15] = np.log1p(self.rng.exponential(2, n))       # log_structural_density
        feats[:, 16] = (self.rng.random(n) > 0.85).astype(float)  # is_ka_traversal
        return feats

    def _apply_canary(self, feats: np.ndarray, labels: np.ndarray,
                      pattern: int) -> tuple:
        """Plant a canary pattern into the sequence. Returns (feats, labels, mask)."""
        mask = np.zeros(len(feats), dtype=bool)
        n = len(feats)

        if pattern == 0:
            # Recency bias: most recent items (low age) should rank higher
            idx = np.argsort(feats[:, 0])[:max(1, n // 4)]
            labels[idx] = 1.0
            mask[idx] = True

        elif pattern == 1:
            # Importance threshold: high-importance memories are relevant
            idx = np.where(feats[:, 1] > 0.7)[0]
            if len(idx) == 0:
                idx = np.array([np.argmax(feats[:, 1])])
            labels[idx] = 1.0
            mask[idx] = True

        elif pattern == 2:
            # Access frequency: frequently accessed memories stay relevant
            idx = np.where(feats[:, 2] > np.percentile(feats[:, 2], 75))[0]
            if len(idx) == 0:
                idx = np.array([np.argmax(feats[:, 2])])
            labels[idx] = 1.0
            mask[idx] = True

        elif pattern == 3:
            # Temporal coherence: same time-of-day memories cluster
            # pick a reference tod, find neighbors
            ref_sin = feats[0, 3]
            ref_cos = feats[0, 4]
            dist = np.sqrt((feats[:, 3] - ref_sin)**2 + (feats[:, 4] - ref_cos)**2)
            idx = np.where(dist < 0.5)[0]
            if len(idx) == 0:
                idx = np.array([np.argmin(dist)])
            labels[idx] = 1.0
            mask[idx] = True

        elif pattern == 4:
            # Entity clustering: memories sharing entity_slot proximity are co-relevant
            ref = feats[0, 12]
            idx = np.where(np.abs(feats[:, 12] - ref) < 0.15)[0]
            if len(idx) == 0:
                idx = np.array([0])
            labels[idx] = 1.0
            mask[idx] = True

        elif pattern == 5:
            # Supersession: superseded memories should NOT be retrieved
            sup_idx = np.where(feats[:, 11] > 0.5)[0]
            labels[sup_idx] = 0.0
            # non-superseded high-importance ones are relevant
            nonsup = np.where((feats[:, 11] < 0.5) & (feats[:, 1] > 0.5))[0]
            if len(nonsup) == 0:
                nonsup = np.where(feats[:, 11] < 0.5)[0][:1]
            labels[nonsup] = 1.0
            mask[sup_idx] = True
            mask[nonsup] = True

        elif pattern == 6:
            # Graph traversal priority: ka_traversal + high density = high relevance
            trav = feats[:, 16] > 0.5
            dense = feats[:, 15] > np.median(feats[:, 15])
            idx = np.where(trav & dense)[0]
            if len(idx) == 0:
                idx = np.where(trav)[0]
            if len(idx) == 0:
                # force one traversal item
                pick = self.rng.randint(0, n)
                feats[pick, 16] = 1.0
                feats[pick, 15] = 2.0
                idx = np.array([pick])
            labels[idx] = 1.0
            mask[idx] = True

        return feats, labels, mask

    def generate(self, snr_db: float = 30.0) -> dict:
        """Generate full dataset with canary patterns and optional noise."""
        sequences = []
        all_labels = []
        canary_masks = []
        pattern_ids = []

        noise_std = 10 ** (-snr_db / 20) if snr_db < 100 else 0.0

        for i in range(self.cfg.num_sequences):
            feats = self._base_features(self.cfg.num_candidates)
            labels = np.zeros(self.cfg.num_candidates, dtype=np.float32)
            mask = np.zeros(self.cfg.num_candidates, dtype=bool)

            # Plant canary in subset of sequences
            pattern = -1
            if self.rng.random() < self.cfg.canary_ratio:
                pattern = i % 7
                feats, labels, mask = self._apply_canary(feats, labels, pattern)
            else:
                # Random relevance for non-canary sequences
                labels = self.rng.beta(1, 3, self.cfg.num_candidates).astype(np.float32)

            # Add noise
            if noise_std > 0:
                feats += self.rng.randn(*feats.shape).astype(np.float32) * noise_std

            sequences.append(feats)
            all_labels.append(labels)
            canary_masks.append(mask)
            pattern_ids.append(pattern)

        return {
            "sequences": np.array(sequences),
            "labels": np.array(all_labels),
            "canary_masks": np.array(canary_masks),
            "pattern_ids": np.array(pattern_ids),
        }


# ---------------------------------------------------------------------------
# Selective SSM block (Mamba-style)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Simplified selective SSM with input-dependent discretization."""

    def __init__(self, d: int, state: int, conv: int):
        super().__init__()
        self.d = d
        self.state = state

        # Conv1d for local context
        self.conv = nn.Conv1d(d, d, conv, padding=conv - 1, groups=d)

        # Input-dependent projections for A, B, C, delta
        self.proj_delta = nn.Linear(d, d, bias=True)
        self.proj_b = nn.Linear(d, state, bias=False)
        self.proj_c = nn.Linear(d, state, bias=False)

        # Learnable A (log-space for stability)
        self.log_a = nn.Parameter(torch.log(torch.linspace(1, state, state).repeat(d, 1)))

        # Gate
        self.proj_gate = nn.Linear(d, d, bias=True)

        # Output projection
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d) -> (batch, seq, d)"""
        b, l, _ = x.shape

        # Conv
        xc = self.conv(x.transpose(1, 2))[:, :, :l].transpose(1, 2)
        xc = F.silu(xc)

        # Input-dependent parameters
        delta = F.softplus(self.proj_delta(xc))  # (b, l, d)
        B = self.proj_b(xc)                       # (b, l, state)
        C = self.proj_c(xc)                        # (b, l, state)

        # Discretize A
        A = -torch.exp(self.log_a)  # (d, state)

        # Scan (sequential for correctness — parallel scan is an optimization)
        h = torch.zeros(b, self.d, self.state, device=x.device)
        ys = []
        for t in range(l):
            dt = delta[:, t, :]                    # (b, d)
            bt = B[:, t, :]                        # (b, state)
            ct = C[:, t, :]                        # (b, state)
            xt = xc[:, t, :]                       # (b, d)

            # State update: h = exp(A * dt) * h + dt * (x outer B)
            decay = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))  # (b, d, state)
            inp = dt.unsqueeze(-1) * (xt.unsqueeze(-1) * bt.unsqueeze(1))  # (b, d, state)
            h = decay * h + inp

            # Output: y = h @ C
            y = (h * ct.unsqueeze(1)).sum(-1)      # (b, d)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (b, l, d)

        # Gate and project
        gate = torch.sigmoid(self.proj_gate(x))
        return self.out(y * gate)


# ---------------------------------------------------------------------------
# Memory predictor model with SSM backbone + multi-head readout
# ---------------------------------------------------------------------------

class MemoryPredictorSSM(nn.Module):
    """SSM-based memory predictor with multiple readout heads."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Input projection
        self.embed = nn.Linear(cfg.feature_dim, cfg.embed_dim)
        self.norm_in = nn.LayerNorm(cfg.embed_dim)

        # Two SSM layers
        self.ssm1 = SelectiveSSM(cfg.embed_dim, cfg.ssm_state_dim, cfg.conv_kernel)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.ssm2 = SelectiveSSM(cfg.embed_dim, cfg.ssm_state_dim, cfg.conv_kernel)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

        # Multi-head readout
        # Head 0: relevance (primary), 1: significance, 2: retention,
        # 3: traversal, 4: contradiction
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
                nn.GELU(),
                nn.Linear(cfg.embed_dim // 2, 1),
            )
            for _ in range(cfg.num_heads)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (batch, candidates, feature_dim)
        Returns dict with 'relevance' and other head scores.
        """
        h = self.norm_in(self.embed(x))
        h = h + self.ssm1(self.norm1(h))
        h = h + self.ssm2(self.norm2(h))

        results = {}
        head_names = ["relevance", "significance", "retention", "traversal", "contradiction"]
        for i, (head, name) in enumerate(zip(self.heads, head_names)):
            results[name] = head(h).squeeze(-1)  # (batch, candidates)

        return results

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Baseline MLP (mimics current heuristic approach)
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    """Simple MLP baseline — roughly what the current effectiveScore heuristic does."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        return {"relevance": self.net(x).squeeze(-1)}

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# JSONL loader for LLM-generated scenarios
# ---------------------------------------------------------------------------

def load_scenarios(path: str, max_candidates: int = 20) -> dict:
    """Load LLM-generated scenarios from JSONL into training format."""
    sequences = []
    labels = []
    names = []

    with open(path) as f:
        for line in f:
            pair = json.loads(line)
            feats = np.array(pair["features"], dtype=np.float32)
            labs = np.array(pair["labels"], dtype=np.float32)

            # Pad or truncate to fixed candidate count
            n = len(feats)
            if n < 5:
                continue
            if n > max_candidates:
                feats = feats[:max_candidates]
                labs = labs[:max_candidates]
            elif n < max_candidates:
                pad_f = np.zeros((max_candidates - n, 17), dtype=np.float32)
                pad_l = np.zeros(max_candidates - n, dtype=np.float32)
                feats = np.concatenate([feats, pad_f])
                labs = np.concatenate([labs, pad_l])

            sequences.append(feats)
            labels.append(labs)
            names.append(pair.get("name", "unknown"))

    return {
        "sequences": np.array(sequences),
        "labels": np.array(labels),
        "pattern_ids": np.array([-1] * len(sequences)),  # no canary IDs for LLM data
        "canary_masks": np.zeros((len(sequences), max_candidates), dtype=bool),
        "source": "llm",
        "pattern_names": names,
    }


# ---------------------------------------------------------------------------
# Heuristic baseline (effectiveScore = importance * 0.95^age_days)
# ---------------------------------------------------------------------------

def heuristic_score(feats: np.ndarray) -> np.ndarray:
    """Current production heuristic: importance * 0.95^age_days."""
    age_days = np.expm1(feats[:, :, 0])  # undo log1p
    importance = feats[:, :, 1]
    return importance * (0.95 ** np.clip(age_days, 0, 365))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def hit_rate_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of sequences where at least one relevant item is in top-k."""
    hits = 0
    for i in range(len(scores)):
        topk = np.argsort(scores[i])[::-1][:k]
        if np.any(labels[i, topk] > 0.5):
            hits += 1
    return hits / max(len(scores), 1)


def mrr_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Mean reciprocal rank of the first relevant item in top-k."""
    total = 0.0
    for i in range(len(scores)):
        topk = np.argsort(scores[i])[::-1][:k]
        for rank, idx in enumerate(topk, 1):
            if labels[i, idx] > 0.5:
                total += 1.0 / rank
                break
    return total / max(len(scores), 1)


def dcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Unnormalized DCG@K (per research: nDCG anti-correlated with online reward)."""
    total = 0.0
    for i in range(len(scores)):
        topk = np.argsort(scores[i])[::-1][:k]
        for rank, idx in enumerate(topk, 1):
            total += labels[i, idx] / math.log2(rank + 1)
    return total / max(len(scores), 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model: nn.Module, data: dict, cfg: Config,
                device: torch.device) -> list:
    """Train and return loss history."""
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    seqs = torch.tensor(data["sequences"], dtype=torch.float32, device=device)
    labs = torch.tensor(data["labels"], dtype=torch.float32, device=device)

    n = len(seqs)
    losses = []

    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        steps = 0

        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            xb = seqs[idx]
            yb = labs[idx]

            out = model(xb)
            pred = out["relevance"]

            # Binary cross-entropy with logits
            loss = F.binary_cross_entropy_with_logits(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            steps += 1

        scheduler.step()
        avg = epoch_loss / max(steps, 1)
        losses.append(avg)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{cfg.epochs}  loss={avg:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: nn.Module, data: dict, cfg: Config,
                   device: torch.device, name: str) -> dict:
    """Evaluate model on all metrics."""
    model.to(device)

    with torch.no_grad():
        seqs = torch.tensor(data["sequences"], dtype=torch.float32, device=device)
        out = model(seqs)
        scores = out["relevance"].cpu().numpy()

    labels = data["labels"]

    metrics = {
        "name": name,
        "hr_k": hit_rate_at_k(scores, labels, cfg.top_k),
        "mrr_k": mrr_at_k(scores, labels, cfg.top_k),
        "dcg_k": dcg_at_k(scores, labels, cfg.top_k),
    }

    # Per-head scores if available
    if "significance" in out:
        for head in ["significance", "retention", "traversal", "contradiction"]:
            if head in out:
                metrics[f"{head}_std"] = out[head].cpu().numpy().std()

    return metrics


def evaluate_heuristic(data: dict, cfg: Config) -> dict:
    """Evaluate production heuristic."""
    scores = heuristic_score(data["sequences"])
    labels = data["labels"]
    return {
        "name": "heuristic (importance * 0.95^age)",
        "hr_k": hit_rate_at_k(scores, labels, cfg.top_k),
        "mrr_k": mrr_at_k(scores, labels, cfg.top_k),
        "dcg_k": dcg_at_k(scores, labels, cfg.top_k),
    }


# ---------------------------------------------------------------------------
# Canary pattern tests
# ---------------------------------------------------------------------------

CANARY_NAMES = [
    "recency_bias",
    "importance_threshold",
    "access_frequency",
    "temporal_coherence",
    "entity_clustering",
    "supersession_filter",
    "graph_traversal_priority",
]


def run_canary_suite(model: nn.Module, cfg: Config,
                     device: torch.device) -> dict:
    """Test each canary pattern in isolation."""
    gen = SyntheticGenerator(cfg, seed=999)
    results = {}

    for pattern_id, name in enumerate(CANARY_NAMES):
        # Generate sequences that all have this canary
        seqs = []
        labs = []
        for _ in range(100):
            feats = gen._base_features(cfg.num_candidates)
            labels = np.zeros(cfg.num_candidates, dtype=np.float32)
            feats, labels, _ = gen._apply_canary(feats, labels, pattern_id)
            seqs.append(feats)
            labs.append(labels)

        seqs_arr = np.array(seqs)
        labs_arr = np.array(labs)

        with torch.no_grad():
            t = torch.tensor(seqs_arr, dtype=torch.float32, device=device)
            scores = model(t)["relevance"].cpu().numpy()

        hr = hit_rate_at_k(scores, labs_arr, cfg.top_k)
        mrr = mrr_at_k(scores, labs_arr, cfg.top_k)

        passed = hr >= 0.5
        results[name] = {"hr_k": hr, "mrr_k": mrr, "passed": passed}

    return results


# ---------------------------------------------------------------------------
# SNR degradation test
# ---------------------------------------------------------------------------

def snr_degradation_test(model: nn.Module, cfg: Config,
                         device: torch.device) -> list:
    """Measure model performance as noise increases."""
    results = []
    for snr in cfg.snr_levels:
        gen = SyntheticGenerator(cfg, seed=77)
        data = gen.generate(snr_db=snr)

        with torch.no_grad():
            t = torch.tensor(data["sequences"], dtype=torch.float32, device=device)
            scores = model(t)["relevance"].cpu().numpy()

        hr = hit_rate_at_k(scores, data["labels"], cfg.top_k)
        results.append({"snr_db": snr, "hr_k": hr})

    return results


# ---------------------------------------------------------------------------
# Dual-error memorization test (SynTSBench-inspired)
# ---------------------------------------------------------------------------

def dual_error_test(model: nn.Module, cfg: Config, device: torch.device) -> dict:
    """
    Check for memorization vs generalization.
    Train set error should be low. Held-out set with same patterns
    should also be low (not memorizing specific instances).
    """
    gen_train = SyntheticGenerator(cfg, seed=42)
    gen_test = SyntheticGenerator(cfg, seed=1337)

    train_data = gen_train.generate()
    test_data = gen_test.generate()

    with torch.no_grad():
        # Train set
        t = torch.tensor(train_data["sequences"], dtype=torch.float32, device=device)
        train_scores = model(t)["relevance"].cpu().numpy()
        train_hr = hit_rate_at_k(train_scores, train_data["labels"], cfg.top_k)

        # Test set
        t = torch.tensor(test_data["sequences"], dtype=torch.float32, device=device)
        test_scores = model(t)["relevance"].cpu().numpy()
        test_hr = hit_rate_at_k(test_scores, test_data["labels"], cfg.top_k)

    gap = train_hr - test_hr
    return {
        "train_hr": train_hr,
        "test_hr": test_hr,
        "gap": gap,
        "memorizing": gap > 0.15,
    }


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def latency_benchmark(model: nn.Module, cfg: Config,
                      device: torch.device, warmup: int = 50,
                      trials: int = 200) -> dict:
    """Measure single-batch inference latency."""
    model.to(device)

    x = torch.randn(1, cfg.num_candidates, cfg.feature_dim, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(trials):
            start = time.perf_counter_ns()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter_ns()
            times.append((end - start) / 1e6)  # ms

    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


# ---------------------------------------------------------------------------
# Comparison mode: hand-crafted vs LLM-generated training data
# ---------------------------------------------------------------------------

def run_comparison(cfg: Config, device: torch.device, llm_path: str):
    """Train SSM on both data sources and compare generalization."""
    print("=" * 60)
    print("COMPARISON: hand-crafted vs LLM-generated training data")
    print("=" * 60)
    print()

    # Prepare hand-crafted data
    gen = SyntheticGenerator(cfg, seed=42)
    hand_data = gen.generate(snr_db=30.0)

    # Load LLM data
    llm_data = load_scenarios(llm_path, cfg.num_candidates)

    # Held-out test set (different seed, hand-crafted — neutral ground)
    gen_test = SyntheticGenerator(cfg, seed=7777)
    test_data = gen_test.generate(snr_db=30.0)

    print(f"hand-crafted train: {hand_data['sequences'].shape[0]} sequences")
    print(f"LLM-generated train: {llm_data['sequences'].shape[0]} sequences")
    print(f"held-out test:       {test_data['sequences'].shape[0]} sequences")
    print()

    results = {}

    for label, train_data in [("hand-crafted", hand_data), ("LLM-generated", llm_data)]:
        print(f"--- training SSM on {label} data ---")
        model = MemoryPredictorSSM(cfg)
        train_model(model, train_data, cfg, device)

        # Evaluate on held-out test
        m = evaluate_model(model, test_data, cfg, device, label)
        results[label] = m

        # Canary suite (always tests on hand-crafted canaries)
        canary = run_canary_suite(model, cfg, device)
        passed = sum(1 for r in canary.values() if r["passed"])
        results[label]["canary_pass"] = passed

        # Dual error
        dual = dual_error_test(model, cfg, device)
        results[label]["generalization_gap"] = dual["gap"]

        print(f"  held-out HR@{cfg.top_k}: {m['hr_k']:.4f}  "
              f"MRR@{cfg.top_k}: {m['mrr_k']:.4f}  "
              f"DCG@{cfg.top_k}: {m['dcg_k']:.4f}")
        print(f"  canary pass: {passed}/7  gen gap: {dual['gap']:.4f}")
        print()

    # Combined training (both sources)
    print("--- training SSM on COMBINED data ---")
    combined = {
        "sequences": np.concatenate([hand_data["sequences"], llm_data["sequences"]]),
        "labels": np.concatenate([hand_data["labels"], llm_data["labels"]]),
    }
    model_combined = MemoryPredictorSSM(cfg)
    train_model(model_combined, combined, cfg, device)
    m = evaluate_model(model_combined, test_data, cfg, device, "combined")
    canary = run_canary_suite(model_combined, cfg, device)
    passed = sum(1 for r in canary.values() if r["passed"])
    dual = dual_error_test(model_combined, cfg, device)
    results["combined"] = m
    results["combined"]["canary_pass"] = passed
    results["combined"]["generalization_gap"] = dual["gap"]
    print(f"  held-out HR@{cfg.top_k}: {m['hr_k']:.4f}  "
          f"MRR@{cfg.top_k}: {m['mrr_k']:.4f}  "
          f"DCG@{cfg.top_k}: {m['dcg_k']:.4f}")
    print(f"  canary pass: {passed}/7  gen gap: {dual['gap']:.4f}")
    print()

    # Summary table
    print("=" * 60)
    print("COMPARISON SUMMARY (held-out test set)")
    print("=" * 60)
    print(f"{'source':20s} {'HR@K':>8s} {'MRR@K':>8s} {'DCG@K':>8s} {'canary':>8s} {'gap':>8s}")
    print("-" * 60)
    for label in ["hand-crafted", "LLM-generated", "combined"]:
        r = results[label]
        print(f"{label:20s} {r['hr_k']:8.4f} {r['mrr_k']:8.4f} {r['dcg_k']:8.4f} "
              f"{r['canary_pass']:>5d}/7  {r['generalization_gap']:7.4f}")

    # Heuristic for reference
    heur = evaluate_heuristic(test_data, cfg)
    print(f"{'heuristic':20s} {heur['hr_k']:8.4f} {heur['mrr_k']:8.4f} {heur['dcg_k']:8.4f} "
          f"{'---':>8s} {'---':>8s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SSM proof of concept benchmark")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to LLM-generated JSONL scenarios")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to JSONL for side-by-side comparison with hand-crafted")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--candidates", type=int, default=None,
                        help="Override candidates per sequence")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.candidates:
        cfg.num_candidates = args.candidates

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"feature_dim: {cfg.feature_dim} (matches protocol.rs FEATURE_DIM)")
    print()

    # --- Comparison mode ---
    if args.compare:
        run_comparison(cfg, device, args.compare)
        return

    # Load or generate data
    if args.data:
        print(f"loading LLM-generated scenarios from {args.data}...")
        data = load_scenarios(args.data, cfg.num_candidates)
        print(f"  sequences: {data['sequences'].shape}")
        pnames = set(data.get("pattern_names", []))
        print(f"  patterns: {len(pnames)} ({', '.join(sorted(pnames)[:8])}{'...' if len(pnames) > 8 else ''})")
    else:
        print("generating synthetic data (7 canary patterns)...")
        gen = SyntheticGenerator(cfg, seed=42)
        data = gen.generate(snr_db=30.0)
        print(f"  sequences: {data['sequences'].shape}")
        print(f"  canary sequences: {(data['pattern_ids'] >= 0).sum()}")
    print()

    # --- Heuristic baseline ---
    print("=" * 60)
    print("HEURISTIC BASELINE (effectiveScore = importance * 0.95^age)")
    print("=" * 60)
    heur_metrics = evaluate_heuristic(data, cfg)
    print(f"  HR@{cfg.top_k}:  {heur_metrics['hr_k']:.4f}")
    print(f"  MRR@{cfg.top_k}: {heur_metrics['mrr_k']:.4f}")
    print(f"  DCG@{cfg.top_k}: {heur_metrics['dcg_k']:.4f}")
    print()

    # --- MLP baseline ---
    print("=" * 60)
    print("MLP BASELINE")
    print("=" * 60)
    mlp = BaselineMLP(cfg)
    print(f"  params: {mlp.count_params():,}")
    print("  training...")
    train_model(mlp, data, cfg, device)
    mlp_metrics = evaluate_model(mlp, data, cfg, device, "mlp")
    print(f"  HR@{cfg.top_k}:  {mlp_metrics['hr_k']:.4f}")
    print(f"  MRR@{cfg.top_k}: {mlp_metrics['mrr_k']:.4f}")
    print(f"  DCG@{cfg.top_k}: {mlp_metrics['dcg_k']:.4f}")
    print()

    # --- SSM model ---
    print("=" * 60)
    print("SELECTIVE SSM (Mamba-style)")
    print("=" * 60)
    ssm = MemoryPredictorSSM(cfg)
    print(f"  params: {ssm.count_params():,}")
    print(f"  state dim: {cfg.ssm_state_dim}")
    print(f"  embed dim: {cfg.embed_dim}")
    print(f"  heads: {cfg.num_heads} (relevance, significance, retention, traversal, contradiction)")
    print("  training...")
    train_model(ssm, data, cfg, device)
    ssm_metrics = evaluate_model(ssm, data, cfg, device, "ssm")
    print(f"  HR@{cfg.top_k}:  {ssm_metrics['hr_k']:.4f}")
    print(f"  MRR@{cfg.top_k}: {ssm_metrics['mrr_k']:.4f}")
    print(f"  DCG@{cfg.top_k}: {ssm_metrics['dcg_k']:.4f}")
    print()

    # --- Canary suite ---
    print("=" * 60)
    print("CANARY PATTERN SUITE (SSM)")
    print("=" * 60)
    canary = run_canary_suite(ssm, cfg, device)
    passed = 0
    for name, result in canary.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {name:30s}  HR@{cfg.top_k}={result['hr_k']:.3f}  MRR@{cfg.top_k}={result['mrr_k']:.3f}")
        if result["passed"]:
            passed += 1
    print(f"  passed: {passed}/7")
    print()

    # --- SNR degradation ---
    print("=" * 60)
    print("SNR DEGRADATION CURVE (SSM)")
    print("=" * 60)
    snr_results = snr_degradation_test(ssm, cfg, device)
    for r in snr_results:
        bar = "#" * int(r["hr_k"] * 40)
        print(f"  SNR {r['snr_db']:4d} dB  HR@{cfg.top_k}={r['hr_k']:.3f}  {bar}")
    print()

    # --- Dual-error memorization ---
    print("=" * 60)
    print("DUAL-ERROR MEMORIZATION TEST (SSM)")
    print("=" * 60)
    dual = dual_error_test(ssm, cfg, device)
    print(f"  train HR@{cfg.top_k}: {dual['train_hr']:.4f}")
    print(f"  test  HR@{cfg.top_k}: {dual['test_hr']:.4f}")
    print(f"  gap:          {dual['gap']:.4f}")
    print(f"  memorizing:   {'YES (bad)' if dual['memorizing'] else 'NO (good)'}")
    print()

    # --- Latency ---
    print("=" * 60)
    print("INFERENCE LATENCY")
    print("=" * 60)
    for name, model in [("SSM", ssm), ("MLP", mlp)]:
        lat = latency_benchmark(model, cfg, device)
        print(f"  {name:4s}  mean={lat['mean_ms']:.2f}ms  p50={lat['p50_ms']:.2f}ms  "
              f"p95={lat['p95_ms']:.2f}ms  p99={lat['p99_ms']:.2f}ms")
    print()

    # --- Phase 0 gate determination ---
    print("=" * 60)
    print("PHASE 0 GATE DETERMINATION")
    print("=" * 60)

    ssm_lat = latency_benchmark(ssm, cfg, device)

    dcg_improvement = (ssm_metrics["dcg_k"] - heur_metrics["dcg_k"]) / max(heur_metrics["dcg_k"], 0.001)

    gate_hrk = ssm_metrics["hr_k"] >= cfg.gate_hrk_min
    gate_dcg = dcg_improvement >= cfg.gate_dcg_improvement
    gate_lat = ssm_lat["p95_ms"] <= cfg.gate_latency_max_ms
    gate_canary = passed >= cfg.gate_canary_min_pass

    checks = [
        ("HR@K >= 0.60", gate_hrk, f"{ssm_metrics['hr_k']:.3f}"),
        ("DCG improvement >= 10%", gate_dcg, f"{dcg_improvement*100:.1f}%"),
        ("P95 latency <= 5ms", gate_lat, f"{ssm_lat['p95_ms']:.2f}ms"),
        ("Canary pass >= 5/7", gate_canary, f"{passed}/7"),
    ]

    all_pass = True
    for desc, ok, val in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc:30s}  actual: {val}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  >>> PHASE 0 GATE: PASSED -- SSM is viable for Signet memory prediction <<<")
    else:
        print("  >>> PHASE 0 GATE: NOT YET PASSED -- needs tuning or architecture changes <<<")

    print()
    print(f"total SSM params: {ssm.count_params():,}")
    print(f"total MLP params: {mlp.count_params():,}")
    print("done.")


if __name__ == "__main__":
    main()
