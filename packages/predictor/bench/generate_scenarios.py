#!/usr/bin/env python3
"""
Synthetic scenario generator for SSM training data.

Uses a local LLM (via Ollama) to generate diverse memory interaction
scenarios that map to the 17-dim feature vector layout from protocol.rs.
The LLM generates behavioral narratives; we convert to feature vectors
deterministically.

Usage:
    pip install requests
    python generate_scenarios.py                    # 500 scenarios, gpt-oss:20b
    python generate_scenarios.py --count 2000       # more scenarios
    python generate_scenarios.py --model qwen3:8b   # different model
    python generate_scenarios.py --output data.jsonl # custom output path
"""

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GenConfig:
    model: str = "gpt-oss:20b"
    ollama_url: str = "http://localhost:11434"
    batch_size: int = 2          # scenarios per LLM call (keep small to avoid truncation)
    candidates_per_scenario: int = 20
    temperature: float = 0.85
    max_retries: int = 3
    timeout: int = 180
    num_predict: int = 16384     # token budget per call


# ---------------------------------------------------------------------------
# Pattern catalog — what we ask the LLM to generate
# ---------------------------------------------------------------------------

PATTERNS = [
    {
        "name": "recency_dominance",
        "desc": "Recent memories are most relevant. Older ones fade.",
        "difficulty": "easy",
    },
    {
        "name": "importance_spike",
        "desc": "A few high-importance memories matter regardless of age.",
        "difficulty": "easy",
    },
    {
        "name": "frequency_signal",
        "desc": "Frequently accessed memories are relevant (habitual knowledge).",
        "difficulty": "easy",
    },
    {
        "name": "temporal_coherence",
        "desc": "Memories from the same time-of-day or day-of-week cluster together. "
                "A user working at 2am retrieves other late-night memories.",
        "difficulty": "medium",
    },
    {
        "name": "entity_cluster",
        "desc": "Memories sharing the same entity (person, project, concept) are "
                "co-relevant when the user is working on that entity.",
        "difficulty": "medium",
    },
    {
        "name": "supersession_filter",
        "desc": "Updated memories should replace their predecessors. Superseded "
                "memories are irrelevant even if they match by entity.",
        "difficulty": "medium",
    },
    {
        "name": "graph_traversal",
        "desc": "Memories reached via knowledge graph traversal (high structural "
                "density, KA traversal flag) are relevant because they connect "
                "otherwise isolated knowledge.",
        "difficulty": "medium",
    },
    {
        "name": "cross_session_chain",
        "desc": "A memory from weeks ago becomes relevant because the user just "
                "started working on something that connects to it. Old age, low "
                "access, but high entity overlap with current context.",
        "difficulty": "hard",
    },
    {
        "name": "hard_negative_entity",
        "desc": "Memories that share the same entity as the query but are about a "
                "completely different aspect. They LOOK relevant but aren't.",
        "difficulty": "hard",
    },
    {
        "name": "hard_negative_recency",
        "desc": "Very recent memories that are NOT relevant — user just switched "
                "context. Recency alone is not enough.",
        "difficulty": "hard",
    },
    {
        "name": "dormant_reactivation",
        "desc": "A memory that hasn't been accessed in months suddenly becomes "
                "critical. Low access count, high age, but now entity-relevant.",
        "difficulty": "hard",
    },
    {
        "name": "project_context_switch",
        "desc": "User switches between two projects. Memories from project A are "
                "irrelevant while working on project B, even if recent.",
        "difficulty": "hard",
    },
    {
        "name": "constraint_priority",
        "desc": "Constraint memories (rules, policies, guardrails) should surface "
                "even when not directly entity-matched, if they apply to the "
                "current action type.",
        "difficulty": "hard",
    },
    {
        "name": "contradiction_pair",
        "desc": "Two memories contain contradictory information. The newer one is "
                "relevant; the older superseded one should be deprioritized.",
        "difficulty": "hard",
    },
]


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a synthetic data generator for a memory retrieval system.

The system has memories with these properties:
- age_days: how old the memory is (0 = just created, 365 = a year old)
- importance: 0.0 to 1.0 (how important the original content was)
- access_count: how many times this memory has been retrieved (0+)
- hour_of_day: 0-23 when the memory was created
- day_of_week: 0-6 (0=Monday)
- month_of_year: 1-12
- session_gap_days: days since this memory was last accessed in a session
- is_embedded: whether the memory has a vector embedding
- is_superseded: whether a newer version of this memory exists
- entity_slot: 0.0-1.0 normalized entity cluster ID (similar values = same entity group)
- aspect_slot: 0.0-1.0 normalized aspect within entity (similar = same facet)
- is_constraint: whether this memory represents a rule/policy/guardrail
- structural_density: 0.0+ how connected this memory is in the knowledge graph
- is_ka_traversal: whether this memory was found via graph traversal

You generate JSON arrays of memory candidates for training a neural ranker.
Each candidate has the properties above plus:
- relevant: true/false (is this memory relevant to the current query?)
- reason: brief explanation of why relevant or not

RULES:
- Generate exactly the requested number of candidates
- Ensure a realistic mix (not all relevant, not all irrelevant)
- relevant:true should be 15-40% of candidates (realistic retrieval)
- Make the pattern subtle — no trivially separable features
- Values must be realistic (age_days 0-500, access_count 0-50, etc.)
- Output ONLY valid JSON, no markdown fences, no commentary"""


def build_batch_prompt(patterns: list, n_candidates: int) -> str:
    """Build a prompt requesting multiple scenarios."""
    parts = ["Generate the following memory retrieval scenarios as a JSON array.\n"]
    parts.append(f"Each scenario has exactly {n_candidates} candidate memories.\n\n")

    for i, p in enumerate(patterns):
        parts.append(f"Scenario {i+1}: \"{p['name']}\" (difficulty: {p['difficulty']})")
        parts.append(f"  Description: {p['desc']}")
        parts.append(f"  Context: Invent a specific, realistic user situation.\n")

    parts.append("Output format — a JSON array of scenario objects:")
    parts.append("""[
  {
    "name": "pattern_name",
    "context": "what the user is doing right now",
    "candidates": [
      {
        "age_days": 5,
        "importance": 0.7,
        "access_count": 3,
        "hour_of_day": 14,
        "day_of_week": 2,
        "month_of_year": 3,
        "session_gap_days": 2,
        "is_embedded": true,
        "is_superseded": false,
        "entity_slot": 0.35,
        "aspect_slot": 0.5,
        "is_constraint": false,
        "structural_density": 1.5,
        "is_ka_traversal": false,
        "relevant": true,
        "reason": "recent config memory for current debug task"
      }
    ]
  }
]""")
    parts.append("\nOutput ONLY the JSON array. No other text.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Feature vector conversion (deterministic)
# ---------------------------------------------------------------------------

def candidate_to_features(c: dict) -> tuple:
    """Convert a candidate dict to (feature_vector_17d, label)."""
    age = max(0, c.get("age_days", 0))
    importance = np.clip(c.get("importance", 0.5), 0, 1)
    access = max(0, c.get("access_count", 0))
    hour = c.get("hour_of_day", 12) % 24
    dow = c.get("day_of_week", 0) % 7
    moy = c.get("month_of_year", 6)
    moy = max(1, min(12, moy))
    gap = max(0, c.get("session_gap_days", 0))
    embedded = 1.0 if c.get("is_embedded", True) else 0.0
    superseded = 1.0 if c.get("is_superseded", False) else 0.0
    entity = np.clip(c.get("entity_slot", 0.5), 0, 1)
    aspect = np.clip(c.get("aspect_slot", 0.5), 0, 1)
    constraint = 1.0 if c.get("is_constraint", False) else 0.0
    density = max(0, c.get("structural_density", 0))
    traversal = 1.0 if c.get("is_ka_traversal", False) else 0.0

    feats = np.array([
        math.log1p(age),                          # [0]  log(age_days)
        importance,                                 # [1]  importance
        math.log1p(access),                        # [2]  log(access_count + 1)
        math.sin(2 * math.pi * hour / 24),        # [3]  tod_sin
        math.cos(2 * math.pi * hour / 24),        # [4]  tod_cos
        math.sin(2 * math.pi * dow / 7),          # [5]  dow_sin
        math.cos(2 * math.pi * dow / 7),          # [6]  dow_cos
        math.sin(2 * math.pi * (moy - 1) / 12),  # [7]  moy_sin
        math.cos(2 * math.pi * (moy - 1) / 12),  # [8]  moy_cos
        math.log1p(gap),                           # [9]  log(session_gap + 1)
        embedded,                                   # [10] is_embedded
        superseded,                                 # [11] is_superseded
        entity,                                     # [12] entity_slot
        aspect,                                     # [13] aspect_slot
        constraint,                                 # [14] is_constraint
        math.log1p(density),                       # [15] log(structural_density + 1)
        traversal,                                  # [16] is_ka_traversal
    ], dtype=np.float32)

    label = 1.0 if c.get("relevant", False) else 0.0
    return feats, label


def scenario_to_training_pair(scenario: dict) -> dict | None:
    """Convert a parsed scenario to a training pair. Returns None on failure."""
    candidates = scenario.get("candidates", [])
    if len(candidates) < 5:
        return None

    features = []
    labels = []
    for c in candidates:
        try:
            f, l = candidate_to_features(c)
            features.append(f.tolist())
            labels.append(l)
        except (KeyError, TypeError, ValueError):
            continue

    if len(features) < 5:
        return None

    # Validate: need at least one positive and one negative
    if sum(labels) < 1 or sum(labels) >= len(labels):
        return None

    # Handle key variants the LLM might use
    name = (scenario.get("name")
            or scenario.get("scenario_name")
            or scenario.get("scenario")
            or scenario.get("pattern")
            or "unknown")

    return {
        "name": name,
        "context": scenario.get("context", ""),
        "features": features,
        "labels": labels,
        "n_candidates": len(features),
        "n_relevant": int(sum(labels)),
    }


# ---------------------------------------------------------------------------
# JSON extraction — robust parsing from LLM output
# ---------------------------------------------------------------------------

def extract_json(text: str) -> list | None:
    """Extract JSON array from LLM output, handling common issues."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass  # Fall through to heuristic extraction below

    # Try to find array boundaries
    start = text.find("[")
    if start == -1:
        return None

    # Find matching bracket
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break

    # Last resort: try fixing truncated JSON by adding closing brackets
    fragment = text[start:]
    for suffix in ["]", "]}]", '"}]}]', '"}]']:
        try:
            result = json.loads(fragment + suffix)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def generate_batch(patterns: list, cfg: GenConfig) -> list:
    """Call Ollama and parse the response into scenarios."""
    prompt = build_batch_prompt(patterns, cfg.candidates_per_scenario)

    payload = {
        "model": cfg.model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.num_predict,
        },
    }

    for attempt in range(cfg.max_retries):
        try:
            resp = requests.post(
                f"{cfg.ollama_url}/api/generate",
                json=payload,
                timeout=cfg.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            text = body.get("response", "")

            scenarios = extract_json(text)
            if scenarios is None:
                print(f"    retry {attempt+1}: failed to parse JSON")
                continue

            return scenarios

        except requests.RequestException as e:
            print(f"    retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    return []


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(count: int, cfg: GenConfig, output: Path) -> dict:
    """Generate `count` scenarios and write to JSONL."""
    rng = np.random.RandomState(42)

    total = 0
    failed = 0
    written = 0
    pattern_counts = {}

    # Calculate batches needed
    batches = math.ceil(count / cfg.batch_size)

    print(f"generating {count} scenarios in {batches} batches of {cfg.batch_size}")
    print(f"model: {cfg.model}")
    print(f"output: {output}")
    print()

    start = time.time()

    with open(output, "w") as f:
        for batch_idx in range(batches):
            remaining = count - written
            if remaining <= 0:
                break

            n = min(cfg.batch_size, remaining)

            # Sample patterns with curriculum weighting
            # Easy patterns more likely early, hard patterns increase over time
            progress = batch_idx / max(batches - 1, 1)
            weights = []
            for p in PATTERNS:
                if p["difficulty"] == "easy":
                    w = 1.0 - 0.5 * progress
                elif p["difficulty"] == "medium":
                    w = 1.0
                else:
                    w = 0.5 + 0.5 * progress
                weights.append(w)
            weights = np.array(weights) / sum(weights)
            chosen_idx = rng.choice(len(PATTERNS), size=n, replace=True, p=weights)
            chosen = [PATTERNS[i] for i in chosen_idx]

            pnames = ", ".join(p["name"] for p in chosen)
            print(f"batch {batch_idx+1}/{batches} [{pnames}]")

            scenarios = generate_batch(chosen, cfg)
            total += n

            if not scenarios:
                print(f"  empty response, skipping")
                failed += n
                continue

            for scenario in scenarios:
                pair = scenario_to_training_pair(scenario)
                if pair is None:
                    failed += 1
                    continue

                f.write(json.dumps(pair) + "\n")
                written += 1

                name = pair["name"]
                pattern_counts[name] = pattern_counts.get(name, 0) + 1

            elapsed = time.time() - start
            rate = written / max(elapsed, 0.01)
            print(f"  parsed {len(scenarios)} scenarios, "
                  f"total written: {written}/{count} ({rate:.1f}/s)")

    elapsed = time.time() - start

    stats = {
        "total_requested": count,
        "total_attempted": total,
        "written": written,
        "failed": failed,
        "success_rate": written / max(total, 1),
        "elapsed_s": round(elapsed, 1),
        "rate_per_s": round(written / max(elapsed, 0.01), 2),
        "pattern_distribution": pattern_counts,
    }

    return stats


# ---------------------------------------------------------------------------
# Validation pass — check generated data quality
# ---------------------------------------------------------------------------

def validate_dataset(path: Path) -> dict:
    """Run basic quality checks on generated JSONL."""
    issues = []
    n = 0
    total_relevant = 0
    total_candidates = 0
    patterns = set()

    with open(path) as f:
        for line in f:
            n += 1
            pair = json.loads(line)

            nc = pair["n_candidates"]
            nr = pair["n_relevant"]
            total_candidates += nc
            total_relevant += nr
            patterns.add(pair["name"])

            # Check feature dimensions
            for feat in pair["features"]:
                if len(feat) != 17:
                    issues.append(f"line {n}: feature dim {len(feat)} != 17")

            # Check relevance ratio
            ratio = nr / nc
            if ratio > 0.5:
                issues.append(f"line {n}: high relevance ratio {ratio:.2f} ({pair['name']})")
            if ratio == 0:
                issues.append(f"line {n}: zero relevant candidates ({pair['name']})")

    avg_ratio = total_relevant / max(total_candidates, 1)

    return {
        "scenarios": n,
        "patterns": len(patterns),
        "pattern_names": sorted(patterns),
        "avg_relevance_ratio": round(avg_ratio, 3),
        "total_candidates": total_candidates,
        "issues": issues[:20],  # cap at 20
        "valid": len(issues) == 0,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SSM training data")
    parser.add_argument("--count", type=int, default=500,
                        help="Number of scenarios to generate")
    parser.add_argument("--model", type=str, default="gpt-oss:20b",
                        help="Ollama model name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Scenarios per LLM call")
    parser.add_argument("--temperature", type=float, default=0.85,
                        help="LLM sampling temperature")
    parser.add_argument("--validate-only", type=str, default=None,
                        help="Only validate an existing JSONL file")
    parser.add_argument("--url", type=str, default="http://localhost:11434",
                        help="Ollama API URL")
    args = parser.parse_args()

    # Validate-only mode
    if args.validate_only:
        path = Path(args.validate_only)
        if not path.exists():
            print(f"file not found: {path}")
            sys.exit(1)
        print(f"validating {path}...")
        result = validate_dataset(path)
        print(f"  scenarios:  {result['scenarios']}")
        print(f"  patterns:   {result['patterns']} ({', '.join(result['pattern_names'])})")
        print(f"  avg relevance ratio: {result['avg_relevance_ratio']}")
        print(f"  total candidates:    {result['total_candidates']}")
        if result["issues"]:
            print(f"  issues ({len(result['issues'])}):")
            for issue in result["issues"]:
                print(f"    - {issue}")
        else:
            print("  no issues found")
        sys.exit(0 if result["valid"] else 1)

    # Generate mode
    output = Path(args.output) if args.output else Path(__file__).parent / "scenarios.jsonl"

    cfg = GenConfig(
        model=args.model,
        ollama_url=args.url,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )

    # Quick connectivity check
    try:
        r = requests.get(f"{cfg.ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(cfg.model in m for m in models):
            print(f"warning: model '{cfg.model}' not found in ollama")
            print(f"  available: {', '.join(models[:10])}")
    except requests.RequestException as e:
        print(f"cannot reach ollama at {cfg.ollama_url}: {e}")
        sys.exit(1)

    stats = generate_dataset(args.count, cfg, output)

    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  written:      {stats['written']}/{stats['total_requested']}")
    print(f"  failed:       {stats['failed']}")
    print(f"  success rate: {stats['success_rate']:.1%}")
    print(f"  elapsed:      {stats['elapsed_s']}s")
    print(f"  rate:         {stats['rate_per_s']}/s")
    print(f"  patterns:     {len(stats['pattern_distribution'])}")
    for name, count in sorted(stats["pattern_distribution"].items()):
        print(f"    {name:30s} {count}")

    # Auto-validate
    print()
    print("running validation...")
    result = validate_dataset(output)
    print(f"  avg relevance ratio: {result['avg_relevance_ratio']}")
    if result["issues"]:
        print(f"  {len(result['issues'])} issues found (first 5):")
        for issue in result["issues"][:5]:
            print(f"    - {issue}")
    else:
        print("  all clean")


if __name__ == "__main__":
    main()
