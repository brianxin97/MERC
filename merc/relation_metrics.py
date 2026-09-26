"""Quantitative summaries for relation-similarity visualizations.

The functions in this module are deliberately independent of plotting.  They
summarize the exact embeddings/cosine matrices used by the existing heatmaps so
that adding numerical evidence cannot silently change figure generation.
"""

import csv
import json
import math
import os

import torch
from torch.nn import functional as F


METRIC_DEFINITIONS = {
    "offdiag_cosine_mean": (
        "Mean cosine similarity over unique off-diagonal vector pairs; lower means stronger separation."
    ),
    "offdiag_cosine_std": "Population standard deviation over unique off-diagonal cosine similarities.",
    "cosine_distance_mean": "Mean pairwise cosine distance (1 - cosine); higher means stronger separation.",
    "nearest_neighbor_cosine_mean": (
        "Mean, over vectors, of the most similar other vector; lower means fewer near-duplicate relations."
    ),
    "within_between_margin_macro": (
        "Macro mean of per-relation within-context cosine minus mean between-relation centroid cosine; higher is better."
    ),
    "within_between_margin_micro": (
        "Pair-weighted within-context cosine minus mean between-relation centroid cosine; higher is better but can be dominated by frequent relations."
    ),
}


def cosine_matrix(embeddings):
    """Return the cosine matrix for a two-dimensional embedding tensor."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape {tuple(embeddings.shape)}")
    normalized = F.normalize(embeddings.float(), p=2, dim=-1)
    return normalized @ normalized.t()


def unique_offdiag_values(matrix):
    """Return each off-diagonal symmetric pair once (strict upper triangle)."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square similarity matrix")
    row, col = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    return matrix[row, col]


def summarize_cosine_matrix(matrix):
    """Summarize a cosine matrix with separation-oriented scalar metrics."""
    values = unique_offdiag_values(matrix)
    if values.numel() == 0:
        raise ValueError("At least two vectors are required for pairwise metrics")

    without_diagonal = matrix.clone()
    without_diagonal.fill_diagonal_(float("-inf"))
    nearest = without_diagonal.max(dim=1).values

    return {
        "num_vectors": int(matrix.shape[0]),
        "num_pairs": int(values.numel()),
        "offdiag_cosine_mean": float(values.mean()),
        "offdiag_cosine_std": float(values.std(unbiased=False)),
        "offdiag_cosine_median": float(values.median()),
        "offdiag_cosine_min": float(values.min()),
        "offdiag_cosine_max": float(values.max()),
        "cosine_distance_mean": float(1.0 - values.mean()),
        "nearest_neighbor_cosine_mean": float(nearest.mean()),
        "nearest_neighbor_cosine_std": float(nearest.std(unbiased=False)),
    }


def summarize_embeddings(embeddings):
    return summarize_cosine_matrix(cosine_matrix(embeddings))


def relation_pair_rows(embeddings, relation_ids, relation_labels, **common_fields):
    """Produce an auditable row for every unique relation pair."""
    matrix = cosine_matrix(embeddings)
    rows = []
    for i in range(len(relation_ids)):
        for j in range(i + 1, len(relation_ids)):
            similarity = float(matrix[i, j])
            rows.append({
                **common_fields,
                "relation_a_id": relation_ids[i],
                "relation_a": relation_labels[i],
                "relation_b_id": relation_ids[j],
                "relation_b": relation_labels[j],
                "cosine_similarity": similarity,
                "cosine_distance": 1.0 - similarity,
            })
    return rows


def context_metrics(rel_to_embeddings, id2rel, **common_fields):
    """Compute per-relation context consistency and dataset-level separation.

    ``rel_to_embeddings`` maps relation IDs to ``(num_samples, dim)`` tensors.
    The within/between margin compares all within-relation context pairs with
    pairs of relation centroids.  It is a descriptive cluster-separation score,
    not a claim of formal representation disentanglement.
    """
    rows = []
    within_values = []
    centroid_ids = []
    centroid_labels = []
    centroids = []

    for relation_id in sorted(rel_to_embeddings, key=lambda rid: str(id2rel.get(rid, rid))):
        embeddings = rel_to_embeddings[relation_id]
        if embeddings.shape[0] < 2:
            continue
        matrix = cosine_matrix(embeddings)
        pair_values = unique_offdiag_values(matrix)
        metrics = summarize_cosine_matrix(matrix)
        rows.append({
            **common_fields,
            "relation_id": relation_id,
            "relation": id2rel.get(relation_id, f"Rel_{relation_id}"),
            "num_samples": int(embeddings.shape[0]),
            **metrics,
        })
        within_values.append(pair_values)
        centroid_ids.append(relation_id)
        centroid_labels.append(id2rel.get(relation_id, f"Rel_{relation_id}"))
        centroids.append(embeddings.float().mean(dim=0))

    if len(centroids) < 2:
        return rows, None, []

    centroid_matrix = torch.stack(centroids)
    centroid_cosine = cosine_matrix(centroid_matrix)
    between_metrics = summarize_embeddings(centroid_matrix)
    all_within = torch.cat(within_values)
    macro_within = torch.tensor([row["offdiag_cosine_mean"] for row in rows])
    within_mean = float(all_within.mean())

    summary = {
        **common_fields,
        "num_relations": len(rows),
        "num_context_pairs": int(all_within.numel()),
        "within_cosine_micro_mean": within_mean,
        "within_cosine_micro_std": float(all_within.std(unbiased=False)),
        "within_cosine_macro_mean": float(macro_within.mean()),
        "within_cosine_macro_std": float(macro_within.std(unbiased=False)),
        "between_centroid_cosine_mean": between_metrics["offdiag_cosine_mean"],
        "between_centroid_cosine_std": between_metrics["offdiag_cosine_std"],
        "between_centroid_nearest_neighbor_cosine_mean": between_metrics["nearest_neighbor_cosine_mean"],
        "within_between_margin_macro": float(macro_within.mean()) - between_metrics["offdiag_cosine_mean"],
        "within_between_margin_micro": within_mean - between_metrics["offdiag_cosine_mean"],
    }

    for index, row in enumerate(rows):
        other_centroids = torch.cat([
            centroid_cosine[index, :index], centroid_cosine[index, index + 1:]
        ])
        between_for_relation = float(other_centroids.mean())
        row["between_to_other_centroids_mean"] = between_for_relation
        row["within_between_margin"] = row["offdiag_cosine_mean"] - between_for_relation

    pair_rows = relation_pair_rows(
        centroid_matrix, centroid_ids, centroid_labels, **common_fields, scope="relation_centroids"
    )
    return rows, summary, pair_rows


def bootstrap_mean_ci(values, num_bootstrap=10000, seed=1024, confidence=0.95):
    """Percentile bootstrap CI for a macro mean over relation-level values."""
    values = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("At least two relation-level values are required for bootstrap")
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        values.numel(), (int(num_bootstrap), values.numel()), generator=generator
    )
    means = values[indices].mean(dim=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(torch.quantile(means, alpha)),
        "ci_high": float(torch.quantile(means, 1.0 - alpha)),
        "confidence": float(confidence),
        "num_bootstrap": int(num_bootstrap),
        "bootstrap_unit": "relation",
    }


def write_csv(path, rows):
    if not rows:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)


def finite_metrics(row):
    """Raise early rather than publishing NaN/Inf evidence tables."""
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite metric {key}={value}")
    return row
