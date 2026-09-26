"""Shared, deterministic helpers for MERC/TRIX relation probing.

This module deliberately does not alter either model.  It only chooses target
triples, creates a one-candidate view of a query, and records an auditable
sampling manifest.
"""

import hashlib
import json
import os

import torch


def target_triplets(data):
    """Return target triples as ``(head, tail, relation)`` rows on CPU."""
    return torch.cat(
        [data.target_edge_index, data.target_edge_type.unsqueeze(0)], dim=0
    ).t().cpu()


def stratified_relation_sample(triplets, max_samples_per_relation=None, seed=1024):
    """Deterministically cap the number of queries for every direct relation.

    The returned rows retain their original dataset order.  Consequently MERC
    and TRIX receive byte-identical query triples when they use the same split,
    cap, and seed.
    """
    if triplets.ndim != 2 or triplets.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) triplet tensor, got {tuple(triplets.shape)}")
    if max_samples_per_relation is not None and max_samples_per_relation <= 0:
        raise ValueError("max_samples_per_relation must be positive or null")

    relation_ids = triplets[:, 2]
    selected = []
    rows = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    for relation_id in sorted(torch.unique(relation_ids).tolist()):
        indices = (relation_ids == relation_id).nonzero(as_tuple=False).flatten()
        available = int(indices.numel())
        if max_samples_per_relation is not None and available > max_samples_per_relation:
            order = torch.randperm(available, generator=generator)[:max_samples_per_relation]
            indices = indices[order]
        selected.append(indices)
        rows.append({
            "relation_id": int(relation_id),
            "available_samples": available,
            "selected_samples": int(indices.numel()),
        })

    selected_indices = torch.cat(selected).sort().values if selected else torch.empty(0, dtype=torch.long)
    sampled = triplets[selected_indices]
    digest = hashlib.sha256(sampled.contiguous().numpy().tobytes()).hexdigest()
    metadata = {
        "sampling_seed": int(seed),
        "max_samples_per_relation": max_samples_per_relation,
        "num_available_queries": int(triplets.shape[0]),
        "num_selected_queries": int(sampled.shape[0]),
        "num_relations": len(rows),
        "sampled_triplets_sha256": digest,
        "relations": rows,
    }
    return sampled, selected_indices, metadata


def single_candidate_batch(batch):
    """Create the model's expected ``(batch, candidates, 3)`` input unchanged."""
    if batch.ndim != 2 or batch.shape[1] != 3:
        raise ValueError(f"Expected a (B, 3) batch, got {tuple(batch.shape)}")
    return batch.unsqueeze(1)


def save_sampling_manifest(directory, metadata, selected_indices):
    os.makedirs(directory, exist_ok=True)
    payload = dict(metadata)
    payload["selected_target_indices"] = [int(index) for index in selected_indices.tolist()]
    with open(os.path.join(directory, "sampling_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def analysis_option(cfg, name, default=None):
    analysis = cfg.get("analysis", {})
    return analysis.get(name, default)

