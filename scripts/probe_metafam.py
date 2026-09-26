"""MetaFam relation-space evidence for MERC."""

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from merc import tasks, util
from merc.models import Merc
from merc.relation_metrics import (
    METRIC_DEFINITIONS,
    finite_metrics,
    relation_pair_rows,
    summarize_embeddings,
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
from probe_nell995 import save_quantitative_evidence


separator = ">" * 30


def build_id2rel_vocab(dataset, split_name="valid"):
    file_to_read = dataset.raw_paths[0]
    if split_name == "test" or getattr(dataset, "valid_on_inf", True):
        if len(dataset.raw_paths) > 1:
            file_to_read = dataset.raw_paths[1]
    if not os.path.exists(file_to_read):
        return {}

    inv_rel_vocab = {}
    with open(file_to_read, "r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split(dataset.delimiter if getattr(dataset, "delimiter", None) else None)
            if len(parts) >= 3 and parts[1] not in inv_rel_vocab:
                inv_rel_vocab[parts[1]] = len(inv_rel_vocab)
    return {relation_id: relation for relation, relation_id in inv_rel_vocab.items()}


def display_branch_name(message_function):
    name = message_function.lower()
    if name == "rotate":
        return "Complex"
    if name == "split":
        return "Split-complex"
    return name.capitalize()


def analysis_identity(cfg, split):
    dataset = cfg.dataset["class"]
    if cfg.dataset.get("version") is not None:
        dataset = f"{dataset}:{cfg.dataset.version}"
    return {"dataset": dataset, "split": split, "model": "MERC"}


def capture_probe_states(model, graph, candidate_batch):
    relation_states = []
    branch_outputs = {}
    handles = []

    def fusion_pre_hook(module, args):
        relation_states.append(args[0].detach())

    def make_branch_hook(branch_name):
        def hook(module, inputs, output):
            branch_outputs[branch_name] = output.detach()
        return hook

    handles.append(model.fusion.register_forward_pre_hook(fusion_pre_hook))
    for branch in model.model_list:
        handles.append(model.entity_branches[branch][-1].register_forward_hook(make_branch_hook(branch)))
    try:
        model(graph, candidate_batch)
    finally:
        for handle in handles:
            handle.remove()

    if len(relation_states) != 1 or set(branch_outputs) != set(model.model_list):
        raise RuntimeError("Failed to capture the MERC relation/branch probe states")
    return relation_states[0], branch_outputs


def verify_single_candidate_equivalence(model, graph, query_batch):
    query = query_batch[:1]
    compact_relation, compact_branches = capture_probe_states(
        model, graph, single_candidate_batch(query)
    )
    all_tail_candidates, _ = tasks.all_negative(graph, query)
    original_relation, original_branches = capture_probe_states(model, graph, all_tail_candidates)
    tensors = [("relation", compact_relation, original_relation)] + [
        (branch, compact_branches[branch], original_branches[branch]) for branch in model.model_list
    ]
    for name, compact, original in tensors:
        if not torch.equal(compact, original):
            difference = float((compact - original).abs().max())
            raise RuntimeError(
                f"Single-candidate probing changed MERC {name} states "
                f"(max absolute difference={difference:.3e}); refusing to continue"
            )


def save_branch_evidence(branch_matrices, relation_ids, relation_labels, cfg, split, metrics_dir, logger):
    identity = analysis_identity(cfg, split)
    summary_rows = []
    pair_rows = []
    for branch, embeddings in branch_matrices.items():
        metrics = summarize_embeddings(embeddings)
        summary_rows.append(finite_metrics({
            **identity,
            "scope": "MERC entity-operator branch (within-model diagnostic only)",
            "branch": branch,
            "branch_display_name": display_branch_name(branch),
            "num_relations": len(relation_ids),
            **metrics,
        }))
        pair_rows.extend(relation_pair_rows(
            embeddings, relation_ids, relation_labels, **identity,
            branch=branch, branch_display_name=display_branch_name(branch),
        ))

    write_csv(os.path.join(metrics_dir, "branch_relation_separation_metrics.csv"), summary_rows)
    write_csv(os.path.join(metrics_dir, "branch_relation_pair_cosines.csv"), pair_rows)
    write_json(os.path.join(metrics_dir, "branch_relation_separation_metrics.json"), {
        "metrics": summary_rows,
        "definitions": METRIC_DEFINITIONS,
        "interpretation": (
            "These branch features reproduce the legacy MERC visualization and are only a within-MERC "
            "diagnostic. The matched MERC-vs-TRIX comparison uses the query-relation metrics instead."
        ),
    })
    logger.warning(
        "-> Saved the legacy branch diagnostic; it is not used as a direct MERC/TRIX comparison"
    )


def plot_centroid_heatmap(embeddings, labels, title, path):
    normalized = F.normalize(embeddings, p=2, dim=-1)
    cosine = torch.mm(normalized, normalized.t()).numpy()
    figure, axis = plt.subplots(1, 1, figsize=(8, 7))
    image = axis.imshow(cosine, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    axis.set_yticklabels(labels, fontsize=12)
    axis.set_title(title, fontsize=16, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=11)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def cos_similarity_across_branches(
        cfg, model, eval_data, device, logger, working_dir, id2rel, split="valid", min_samples=3):
    if util.get_world_size() != 1:
        raise RuntimeError("Relation evidence scripts require single-process execution for an auditable sample")

    all_triplets = target_triplets(eval_data)
    # MetaFam is small enough to use the complete validation split by default.
    max_per_relation = analysis_option(cfg, "max_samples_per_relation", None)
    sampled, selected_indices, sampling = stratified_relation_sample(
        all_triplets, max_samples_per_relation=max_per_relation,
        seed=int(analysis_option(cfg, "sampling_seed", 1024)),
    )
    metrics_dir = os.path.join(working_dir, "quantitative_relation_metrics")
    save_sampling_manifest(metrics_dir, sampling, selected_indices)
    loader = torch_data.DataLoader(
        sampled, batch_size=int(analysis_option(cfg, "probe_batch_size", cfg.train.batch_size)), shuffle=False
    )

    model.eval()
    relation_contexts = collections.defaultdict(list)
    branch_contexts = {
        branch: collections.defaultdict(list) for branch in model.model_list
    }
    equivalence_checked = False
    logger.warning(
        "-> MERC %s: sampled %d/%d queries (cap=%s, sha256=%s)",
        split, len(sampled), len(all_triplets), max_per_relation, sampling["sampled_triplets_sha256"][:12],
    )

    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        if not equivalence_checked and analysis_option(cfg, "verify_single_candidate", True):
            verify_single_candidate_equivalence(model, eval_data, batch)
            logger.warning(
                "-> Passed the bitwise single-candidate/all-negative representation check"
            )
            equivalence_checked = True

        relation_states, branch_outputs = capture_probe_states(
            model, eval_data, single_candidate_batch(batch)
        )
        heads = cpu_batch[:, 0]
        relations = cpu_batch[:, 2].tolist()
        relation_states = relation_states.cpu()
        for index, relation_id in enumerate(relations):
            relation_contexts[int(relation_id)].append(relation_states[index])
        for branch in model.model_list:
            output = branch_outputs[branch].cpu()
            query_features = output[torch.arange(len(heads)), heads]
            for index, relation_id in enumerate(relations):
                branch_contexts[branch][int(relation_id)].append(query_features[index])

    relation_contexts = {
        relation_id: torch.stack(values) for relation_id, values in relation_contexts.items()
    }
    branch_contexts = {
        branch: {relation_id: torch.stack(values) for relation_id, values in relation_map.items()}
        for branch, relation_map in branch_contexts.items()
    }
    os.makedirs(metrics_dir, exist_ok=True)
    torch.save(relation_contexts, os.path.join(metrics_dir, "relation_context_embeddings.pt"))
    save_quantitative_evidence(
        relation_contexts, id2rel, cfg, split, metrics_dir,
        seed=int(analysis_option(cfg, "bootstrap_seed", 1024)), logger=logger,
    )

    first_branch = model.model_list[0]
    relation_ids = sorted(
        [relation_id for relation_id, values in branch_contexts[first_branch].items()
         if len(values) >= min_samples and relation_id in id2rel],
        key=lambda relation_id: id2rel[relation_id],
    )
    if len(relation_ids) < 2:
        raise RuntimeError("At least two named relations are required for MetaFam heatmaps")
    relation_labels = [id2rel[relation_id] for relation_id in relation_ids]

    relation_centroids = torch.stack([
        relation_contexts[relation_id].mean(dim=0) for relation_id in relation_ids
    ])
    plot_centroid_heatmap(
        relation_centroids, relation_labels, "MERC Query-Relation Space",
        os.path.join(working_dir, "merc_relation_space.pdf"),
    )

    branch_matrices = {
        branch: torch.stack([
            branch_contexts[branch][relation_id].mean(dim=0) for relation_id in relation_ids
        ]) for branch in model.model_list
    }
    save_branch_evidence(
        branch_matrices, relation_ids, relation_labels, cfg, split, metrics_dir, logger
    )

    # Legacy plotting logic: one relation-centroid cosine heatmap per algebraic branch.
    figure, axes = plt.subplots(1, len(model.model_list), figsize=(8 * len(model.model_list), 7))
    if len(model.model_list) == 1:
        axes = [axes]
    for axis_index, branch in enumerate(model.model_list):
        axis = axes[axis_index]
        normalized = F.normalize(branch_matrices[branch], p=2, dim=-1)
        cosine = torch.mm(normalized, normalized.t()).numpy()
        image = axis.imshow(cosine, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
        axis.set_xticks(range(len(relation_ids)))
        axis.set_yticks(range(len(relation_ids)))
        axis.set_xticklabels(relation_labels, rotation=45, ha="right", fontsize=12)
        axis.set_yticklabels(relation_labels if axis_index == 0 else [], fontsize=12)
        axis.set_title(f"{display_branch_name(branch)} Space", fontsize=16, fontweight="bold")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=11)
    plt.suptitle(
        "Inter-Relational Cosine Similarity Across Algebraic Subspaces",
        fontsize=20, y=1.05, fontweight="bold",
    )
    plt.savefig(os.path.join(working_dir, "merc_algebraic_decoupling.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.warning(
        "-> Saved MetaFam heatmaps, aligned relation-space plots, and metrics to %s",
        working_dir,
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
    split = str(analysis_option(cfg, "split", "valid"))
    split_index = {"train": 0, "valid": 1, "test": 2}[split]
    eval_data = dataset[split_index].to(device)
    id2rel = build_id2rel_vocab(dataset, split_name=split)

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
    cos_similarity_across_branches(
        cfg, model, eval_data, device, logger, working_dir, id2rel,
        split=split, min_samples=int(analysis_option(cfg, "min_samples", 3)),
    )
