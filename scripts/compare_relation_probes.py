"""Create the paired MERC-vs-TRIX quantitative comparison table."""

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from merc.relation_metrics import bootstrap_mean_ci, write_csv, write_json


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def relation_rows(metrics_dir):
    rows = read_csv(os.path.join(metrics_dir, "context_similarity_by_relation.csv"))
    return {row["relation"]: row for row in rows}


def summary_row(metrics_dir):
    rows = read_csv(os.path.join(metrics_dir, "relation_separation_summary.csv"))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {metrics_dir}, got {len(rows)}")
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merc", required=True, help="MERC quantitative_relation_metrics directory")
    parser.add_argument("--trix", required=True, help="TRIX quantitative_relation_metrics directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    merc_manifest = read_json(os.path.join(args.merc, "sampling_manifest.json"))
    trix_manifest = read_json(os.path.join(args.trix, "sampling_manifest.json"))
    if merc_manifest["sampled_triplets_sha256"] != trix_manifest["sampled_triplets_sha256"]:
        raise ValueError("MERC and TRIX sampling manifests differ; paired comparison is invalid")

    merc = relation_rows(args.merc)
    trix = relation_rows(args.trix)
    if set(merc) != set(trix):
        raise ValueError("MERC and TRIX relation sets differ; paired comparison is invalid")

    paired_rows = []
    for relation in sorted(merc):
        merc_row, trix_row = merc[relation], trix[relation]
        row = {
            "dataset": merc_row["dataset"],
            "split": merc_row["split"],
            "relation": relation,
            "num_samples": int(merc_row["num_samples"]),
        }
        for metric in (
            "offdiag_cosine_mean",
            "between_to_other_centroids_mean",
            "within_between_margin",
        ):
            merc_value = float(merc_row[metric])
            trix_value = float(trix_row[metric])
            row[f"merc_{metric}"] = merc_value
            row[f"trix_{metric}"] = trix_value
            row[f"delta_merc_minus_trix_{metric}"] = merc_value - trix_value
        paired_rows.append(row)

    comparisons = {}
    for metric, higher_is_better in (
        ("offdiag_cosine_mean", True),
        ("between_to_other_centroids_mean", False),
        ("within_between_margin", True),
    ):
        deltas = [row[f"delta_merc_minus_trix_{metric}"] for row in paired_rows]
        ci = bootstrap_mean_ci(
            deltas, num_bootstrap=args.bootstrap_samples, seed=args.seed
        )
        comparisons[metric] = {
            **ci,
            "delta_definition": "MERC minus TRIX",
            "higher_is_better": higher_is_better,
        }

    merc_summary = summary_row(args.merc)
    trix_summary = summary_row(args.trix)
    margin_ci = comparisons["within_between_margin"]
    paper_summary = [{
        "dataset": paired_rows[0]["dataset"],
        "split": paired_rows[0]["split"],
        "num_relations": len(paired_rows),
        "num_queries": merc_manifest["num_selected_queries"],
        "merc_within_cosine_macro": float(merc_summary["within_cosine_macro_mean"]),
        "trix_within_cosine_macro": float(trix_summary["within_cosine_macro_mean"]),
        "merc_between_centroid_cosine": float(merc_summary["between_centroid_cosine_mean"]),
        "trix_between_centroid_cosine": float(trix_summary["between_centroid_cosine_mean"]),
        "merc_within_between_margin": float(merc_summary["within_between_margin_macro"]),
        "trix_within_between_margin": float(trix_summary["within_between_margin_macro"]),
        "margin_delta_merc_minus_trix": margin_ci["mean"],
        "margin_delta_ci_low": margin_ci["ci_low"],
        "margin_delta_ci_high": margin_ci["ci_high"],
    }]

    os.makedirs(args.output, exist_ok=True)
    write_csv(os.path.join(args.output, "paired_relation_comparison.csv"), paired_rows)
    write_csv(os.path.join(args.output, "paper_summary_comparison.csv"), paper_summary)
    write_json(os.path.join(args.output, "paired_model_comparison.json"), {
        "dataset": paired_rows[0]["dataset"],
        "split": paired_rows[0]["split"],
        "num_relations": len(paired_rows),
        "sampled_triplets_sha256": merc_manifest["sampled_triplets_sha256"],
        "paired_relation_bootstrap": comparisons,
        "interpretation": (
            "For MERC-minus-TRIX deltas, positive within-relation cosine and margin are favorable; "
            "negative between-centroid cosine is favorable. The bootstrap unit is relation."
        ),
    })
    margin = comparisons["within_between_margin"]
    print(
        f"paired relations={len(paired_rows)}; margin delta MERC-TRIX={margin['mean']:.6f} "
        f"(95% CI [{margin['ci_low']:.6f}, {margin['ci_high']:.6f}])"
    )


if __name__ == "__main__":
    main()
