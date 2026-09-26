"""NELL995 relation-context heatmaps and quantitative evidence for MERC.

The model is not modified.  Relation states are observed at the input of the
existing fusion module, and all statistics are computed offline.
"""

import collections
import os
import pprint
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "merc-matplotlib"))
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from merc import tasks, util
from merc.models import Merc
from merc.relation_metrics import (
    METRIC_DEFINITIONS,
    bootstrap_mean_ci,
    context_metrics,
    finite_metrics,
    write_csv,
    write_json,
)
from merc.relation_probe import (
    analysis_option,
    save_sampling_manifest,
    single_candidate_batch,
    stratified_relation_sample,
    target_triplets,
)


separator = ">" * 30


def build_id2rel_vocab(dataset):
    raw_dir = dataset.raw_dir
    if getattr(dataset, "name", "") == "nell995":
        files_to_read = [os.path.join(raw_dir, "facts.txt"), os.path.join(raw_dir, "train.txt")]
    else:
        files_to_read = [os.path.join(raw_dir, "train.txt")]

    inv_rel_vocab = {}
    for file_path in files_to_read:
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as stream:
            for line in stream:
                parts = line.strip().split(dataset.delimiter if getattr(dataset, "delimiter", None) else None)
                if len(parts) >= 3 and parts[1] not in inv_rel_vocab:
                    inv_rel_vocab[parts[1]] = len(inv_rel_vocab)
    return {relation_id: relation for relation, relation_id in inv_rel_vocab.items()}


def analysis_identity(cfg, split):
    dataset = cfg.dataset["class"]
    if cfg.dataset.get("version") is not None:
        dataset = f"{dataset}:{cfg.dataset.version}"
    return {"dataset": dataset, "split": split, "model": "MERC"}


def capture_relation_states(model, graph, candidate_batch):
    captured = []

    def fusion_pre_hook(module, args):
        captured.append(args[0].detach())

    handle = model.fusion.register_forward_pre_hook(fusion_pre_hook)
    try:
        model(graph, candidate_batch)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one fusion input, captured {len(captured)}")
    return captured[0]


def verify_single_candidate_equivalence(model, graph, query_batch):
    """Require bitwise-identical relation states before using the fast probe."""
    query = query_batch[:1]
    compact = capture_relation_states(model, graph, single_candidate_batch(query))
    all_tail_candidates, _ = tasks.all_negative(graph, query)
    original = capture_relation_states(model, graph, all_tail_candidates)
    if not torch.equal(compact, original):
        max_difference = float((compact - original).abs().max())
        raise RuntimeError(
            "Single-candidate probing changed MERC relation states "
            f"(max absolute difference={max_difference:.3e}); refusing to continue"
        )


def save_quantitative_evidence(rel_to_embeddings, id2rel, cfg, split, metrics_dir, seed, logger):
    identity = analysis_identity(cfg, split)
    per_relation, summary, centroid_pairs = context_metrics(rel_to_embeddings, id2rel, **identity)
    per_relation = [finite_metrics(row) for row in per_relation]
    if summary is None:
        raise RuntimeError("At least two relations with two samples each are required")
    summary = finite_metrics(summary)

    bootstrap = bootstrap_mean_ci(
        [row["within_between_margin"] for row in per_relation],
        num_bootstrap=int(analysis_option(cfg, "bootstrap_samples", 10000)),
        seed=seed,
    )
    summary.update({
        "relation_margin_bootstrap_mean": bootstrap["mean"],
        "relation_margin_ci_low": bootstrap["ci_low"],
        "relation_margin_ci_high": bootstrap["ci_high"],
    })

    write_csv(os.path.join(metrics_dir, "context_similarity_by_relation.csv"), per_relation)
    write_csv(os.path.join(metrics_dir, "relation_centroid_pairs.csv"), centroid_pairs)
    write_csv(os.path.join(metrics_dir, "relation_separation_summary.csv"), [summary])
    write_json(os.path.join(metrics_dir, "relation_separation_summary.json"), {
        "metrics": summary,
        "relation_level_margin_bootstrap": bootstrap,
        "definitions": METRIC_DEFINITIONS,
        "representation": "updated query-relation state immediately before MERC fusion",
        "interpretation": (
            "Higher within-between margin and lower between-centroid cosine indicate clearer descriptive "
            "separation. These values do not prove formal disentanglement."
        ),
    })
    logger.warning(
        "-> MERC: within=%.4f, between=%.4f, margin=%.4f, relation-bootstrap 95%% CI=[%.4f, %.4f]",
        summary["within_cosine_macro_mean"], summary["between_centroid_cosine_mean"],
        summary["within_between_margin_macro"], bootstrap["ci_low"], bootstrap["ci_high"],
    )


@torch.no_grad()
def cos_similarity(cfg, model, eval_data, device, logger, working_dir, id2rel, split="test"):
    if util.get_world_size() != 1:
        raise RuntimeError("Relation evidence scripts require single-process execution for an auditable sample")

    all_triplets = target_triplets(eval_data)
    max_per_relation = analysis_option(cfg, "max_samples_per_relation", 32)
    sampled, selected_indices, sampling = stratified_relation_sample(
        all_triplets, max_samples_per_relation=max_per_relation, seed=int(analysis_option(cfg, "sampling_seed", 1024))
    )
    metrics_dir = os.path.join(working_dir, "quantitative_relation_metrics")
    save_sampling_manifest(metrics_dir, sampling, selected_indices)

    loader = torch_data.DataLoader(
        sampled, batch_size=int(analysis_option(cfg, "probe_batch_size", cfg.train.batch_size)), shuffle=False
    )
    model.eval()
    rel_to_embeddings = collections.defaultdict(list)
    equivalence_checked = False

    logger.warning(
        "-> MERC %s: sampled %d/%d relation-stratified queries (cap=%s, sha256=%s)",
        split, len(sampled), len(all_triplets), max_per_relation, sampling["sampled_triplets_sha256"][:12],
    )
    for cpu_batch in loader:
        if cpu_batch.numel() == 0:
            continue
        batch = cpu_batch.to(device)
        if not equivalence_checked and analysis_option(cfg, "verify_single_candidate", True):
            verify_single_candidate_equivalence(model, eval_data, batch)
            logger.warning(
                "-> Passed the bitwise single-candidate/all-negative relation-state check"
            )
            equivalence_checked = True

        relation_features = capture_relation_states(model, eval_data, single_candidate_batch(batch)).cpu()
        for index, relation_id in enumerate(cpu_batch[:, 2].tolist()):
            rel_to_embeddings[int(relation_id)].append(relation_features[index])

    rel_to_embeddings = {
        relation_id: torch.stack(features) for relation_id, features in rel_to_embeddings.items()
    }
    os.makedirs(working_dir, exist_ok=True)
    torch.save(rel_to_embeddings, os.path.join(metrics_dir, "relation_context_embeddings.pt"))
    save_quantitative_evidence(
        rel_to_embeddings, id2rel, cfg, split, metrics_dir,
        seed=int(analysis_option(cfg, "bootstrap_seed", 1024)), logger=logger,
    )

    # Original plotting calculation: pairwise cosine among contexts of one relation.
    drawn_count = 0
    max_relations = int(analysis_option(cfg, "max_plotted_relations", 200))
    for relation_id, embeddings in rel_to_embeddings.items():
        if drawn_count >= max_relations or embeddings.shape[0] < 2:
            continue
        rel_text = id2rel.get(relation_id, f"Rel_{relation_id}")
        safe_rel_text = "".join(c if c.isalnum() else "_" for c in rel_text)
        normalized = F.normalize(embeddings, p=2, dim=-1)
        cosine = torch.mm(normalized, normalized.t()).numpy()

        plt.figure(figsize=(10, 8))
        plt.imshow(cosine, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
        colorbar = plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout(pad=1.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Sample Index", fontsize=16)
        plt.ylabel("Sample Index", fontsize=16)
        colorbar.ax.tick_params(labelsize=14)
        plt.title(f"{rel_text}\nContext Similarity ({embeddings.shape[0]} samples)", fontsize=14)
        plt.savefig(
            os.path.join(working_dir, f"cos_{relation_id}_{safe_rel_text}.pdf"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        drawn_count += 1
    logger.warning(
        "-> Generated %d heatmaps; quantitative outputs are in %s",
        drawn_count,
        metrics_dir,
    )


if __name__ == "__main__":
    args, variables = util.parse_args()
    cfg = util.load_config(args.config, context=variables)
    working_dir = util.create_working_directory(cfg)
    torch.manual_seed(args.seed + util.get_rank())
    logger = util.get_root_logger()
    logger.warning("Random seed: %d", args.seed)
    logger.warning("Config file: %s", args.config)
    logger.warning("Working Directory: %s", working_dir)
    logger.warning(pprint.pformat(cfg))

    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    split = str(analysis_option(cfg, "split", "test"))
    split_index = {"train": 0, "valid": 1, "test": 2}[split]
    eval_data = dataset[split_index].to(device)
    id2rel = build_id2rel_vocab(dataset)

    model = Merc(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
        num_relation=dataset[0].num_relations,
        entropy_reg_weight=cfg.model.entity_model.get("entropy_reg_weight", 0.003),
        attn_dropout=cfg.model.entity_model.get("attn_dropout", 0.1),
    )
    if not cfg.get("checkpoint"):
        raise ValueError("A trained checkpoint is required for quantitative evidence")
    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model = model.to(device)

    logger.warning(separator)
    cos_similarity(cfg, model, eval_data, device, logger, working_dir, id2rel, split=split)
