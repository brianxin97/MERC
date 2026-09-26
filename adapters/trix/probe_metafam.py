"""MetaFam matched relation-space heatmap and quantitative evidence for Trix.

Run it with the upstream Trix ``src/`` directory and the MERC artifact root on
``PYTHONPATH``. Keep ``probe_nell995.py`` in the same directory. Copying the
two files into Trix is optional.
"""

import collections
import os
import pprint
import shutil
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "merc-matplotlib"))
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

TRIX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(TRIX_ROOT)
for path in (PROJECT_ROOT, os.path.join(TRIX_ROOT, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from merc.relation_probe import (
    analysis_option,
    save_sampling_manifest,
    single_candidate_batch,
    stratified_relation_sample,
    target_triplets,
)
from trix import util
from trix.models_entity import TRIX
from probe_nell995 import (
    extract_query_relation_states,
    save_quantitative_evidence,
    verify_single_candidate_equivalence,
)


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


def plot_centroid_heatmap(embeddings, labels, path):
    normalized = F.normalize(embeddings, p=2, dim=-1)
    cosine = torch.mm(normalized, normalized.t()).numpy()
    figure, axis = plt.subplots(1, 1, figsize=(8, 7))
    image = axis.imshow(cosine, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    axis.set_yticklabels(labels, fontsize=12)
    axis.set_title("TRIX Query-Relation Space", fontsize=16, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=11)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def cos_similarity_trix(
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
    equivalence_checked = False
    logger.warning(
        "-> TRIX %s: sampled %d/%d queries (cap=%s, sha256=%s)",
        split, len(sampled), len(all_triplets), max_per_relation, sampling["sampled_triplets_sha256"][:12],
    )
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        if not equivalence_checked and analysis_option(cfg, "verify_single_candidate", True):
            verify_single_candidate_equivalence(model, eval_data, batch)
            logger.warning("-> Passed the bitwise single-candidate/all-negative relation-state check")
            equivalence_checked = True
        query_states = extract_query_relation_states(
            model, eval_data, single_candidate_batch(batch)
        ).cpu()
        for index, relation_id in enumerate(cpu_batch[:, 2].tolist()):
            relation_contexts[int(relation_id)].append(query_states[index])

    relation_contexts = {
        relation_id: torch.stack(values) for relation_id, values in relation_contexts.items()
    }
    os.makedirs(metrics_dir, exist_ok=True)
    torch.save(relation_contexts, os.path.join(metrics_dir, "relation_context_embeddings.pt"))
    save_quantitative_evidence(
        relation_contexts, id2rel, cfg, split, metrics_dir,
        seed=int(analysis_option(cfg, "bootstrap_seed", 1024)), logger=logger,
    )

    relation_ids = sorted(
        [relation_id for relation_id, values in relation_contexts.items()
         if len(values) >= min_samples and relation_id in id2rel],
        key=lambda relation_id: id2rel[relation_id],
    )
    if len(relation_ids) < 2:
        raise RuntimeError("At least two named relations are required for MetaFam heatmaps")
    labels = [id2rel[relation_id] for relation_id in relation_ids]
    centroids = torch.stack([
        relation_contexts[relation_id].mean(dim=0) for relation_id in relation_ids
    ])

    # Original TRIX centroid-cosine plotting calculation, now fed the same
    # deterministic query set and same semantic representation as MERC.
    legacy_path = os.path.join(working_dir, "trix_algebraic_decoupling.pdf")
    plot_centroid_heatmap(centroids, labels, legacy_path)
    # Explicit fair-comparison filename with byte-identical plot content.
    shutil.copyfile(legacy_path, os.path.join(working_dir, "trix_relation_space.pdf"))
    logger.warning("-> Saved the MetaFam relation-space plot and metrics to %s", working_dir)


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

    model = TRIX(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_1_cfg=cfg.model.entity_model_1,
        entity_model_2_cfg=cfg.model.entity_model_2,
    )
    if not cfg.get("checkpoint"):
        raise ValueError("A trained checkpoint is required for quantitative evidence")
    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model = model.to(device)

    logger.warning(separator)
    cos_similarity_trix(
        cfg, model, eval_data, device, logger, working_dir, id2rel,
        split=split, min_samples=int(analysis_option(cfg, "min_samples", 3)),
    )
