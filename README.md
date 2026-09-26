# MERC

MERC is a structural knowledge-graph foundation model for zero-shot link
prediction on unseen entities and relations. It combines layer-wise
entity-to-relation feedback with Complex, Split-Complex, and Dual message
branches. MERC Flash retains the feedback mechanism with a single Complex
branch for lower-latency inference.

This repository provides the model implementation, frozen MERC checkpoints,
training and evaluation configurations, and the scripts used for the
biomedical, relation-pattern, relation-state, and efficiency analyses in the
paper.

## Repository layout

```text
merc/                  model, data loaders, tasks, and sparse kernels
scripts/               evaluation, pre-training, and relation-probe entry points
configs/
  evaluation/          CUDA/CPU and macOS/MPS inference configurations
  pretraining/         MERC and MERC Flash pre-training configurations
  probes/              MetaFam and NELL-995 relation-state probes
  profiling/           batch-size-8 MERC profiling configurations
  biomedical/          biomedical dataset list and subsampling ratios
checkpoints/            frozen MERC and MERC Flash checkpoints
adapters/
  trix/                 Trix-side matched relation-probe runners
  profiling/            MERC, Ultra, Trix, and Flock profiling runners
tools/
  biomedical/          biomedical graph-construction pipeline
  relation_patterns/   relation-pattern mining and safe evaluation staging
  profiling/           fixed-window CUDA profiling utilities
manifests/              baseline provenance and biomedical split SHA-256 values
references/             biomedical-source BibTeX entries
```

## Installation

MERC requires Python 3.10 or later. Install the common Python dependencies
after installing the appropriate PyTorch and PyTorch Geometric builds for the
target platform.

### Linux with CUDA

The paper's training environment used PyTorch 2.1.0, PyTorch Geometric 2.4.0,
and CUDA 11.8. A matching pip installation is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
  torch-geometric==2.4.0 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install -r requirements.txt
```

The fused RSPMM extension is compiled on first use. A CUDA development toolkit
with `nvcc` is required; set `CUDA_HOME` when it is not detected automatically.

### macOS with Apple Metal

The macOS path uses PyTorch's MPS backend and PyG native message passing. It
does not compile or call the CUDA/CPU RSPMM extension during MPS execution.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 torch-geometric==2.8.0.post1
python -m pip install -r requirements.txt
```

Use the configurations ending in `_mps.yaml` on macOS. They use a batch size of
1 to reduce unified-memory pressure. MPS execution is intended for inference
and diagnostics; the reported CUDA efficiency measurements are not comparable
to macOS timings.

The relation-state plots require:

```bash
python -m pip install -r requirements-analysis.txt
```

CKG extraction additionally requires the Neo4j Python driver and `tqdm`:

```bash
python -m pip install -r requirements-biomedical.txt
```

CUDA profiling additionally requires:

```bash
python -m pip install -r requirements-profiling.txt
```

## Checkpoints

| Checkpoint | Architecture | SHA-256 |
|---|---|---|
| `checkpoints/merc.pth` | Complex + Split-Complex + Dual with co-evolution | `9243ef82c0df3b7aa1139c68970a03711792feaf775959e273b2eddb8541b933` |
| `checkpoints/merc_flash.pth` | Complex-only with co-evolution | `e040ec4c72c3d26b46a015126f8dabc7e49c2b2a1f7385b0c5e3e591f5de46c3` |

Both checkpoints use 32-dimensional hidden states. Their pre-training
configurations are in `configs/pretraining/`.

## Zero-shot inference

Run full MERC on a transductive graph with one CUDA GPU:

```bash
python scripts/run.py \
  -c configs/evaluation/transductive.yaml \
  --dataset CoDExSmall \
  --epochs 0 \
  --bpe null \
  --gpus '[0]' \
  --ckpt checkpoints/merc.pth
```

For MERC Flash, use `configs/evaluation/transductive_flash.yaml` and
`checkpoints/merc_flash.pth`. On Linux, `--gpus null` selects CPU execution,
although compiling the fused RSPMM extension still requires the CUDA
development toolkit.

On macOS/MPS, use the dedicated configuration without a `--gpus` argument:

```bash
python scripts/run.py \
  -c configs/evaluation/transductive_mps.yaml \
  --dataset CoDExSmall \
  --epochs 0 \
  --bpe null \
  --ckpt checkpoints/merc.pth
```

Use `transductive_flash_mps.yaml` for MERC Flash. Inductive evaluation uses
`inductive.yaml` or `inductive_flash.yaml` on CUDA/CPU and the corresponding
`*_mps.yaml` files on macOS. For example, CUDA inductive inference is:

```bash
python scripts/run.py \
  -c configs/evaluation/inductive.yaml \
  --dataset FB15k237Inductive \
  --version v1 \
  --epochs 0 \
  --bpe null \
  --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth
```

Each run writes `log.txt` below
`output/Merc/<DatasetClass>/<timestamp>/`. The log contains validation and test
metrics; evaluation does not modify the frozen checkpoint.

### The 54-graph suite

The held-out suite contains 13 transductive graphs, 18 entity-inductive graphs
whose inference entities are unseen but relations are shared, and 23
entity-and-relation-inductive graphs. The groups cover encyclopedic, lexical,
commonsense, biomedical, taxonomic, and synthetic domains:

| Setting | Families | Graphs |
|---|---|---:|
| Transductive | CoDEx-S/L, NELL-995, YAGO3-10, WD-singer, NELL23K, three sparse FB15k-237 variants, DBpedia100K, Hetionet, AristoV4, and ConceptNet100K | 13 |
| Entity-inductive | GraIL splits of WN18RR, FB15k-237, and NELL-995; ILPC-small/large; and the three Hamaguchi plus INDIGO graphs | 18 |
| Entity-and-relation-inductive | FB, Wikidata, and NELL InGram variants; eight WikiTopics domain transfers; MetaFam; and FBNELL | 23 |

Exact loader names and versions are recorded in:

```text
configs/evaluation/benchmark_54_transductive.txt   13 graphs
configs/evaluation/benchmark_54_inductive.txt      18 + 23 graphs
```

Run each group sequentially on one GPU:

```bash
python scripts/run_many.py \
  -c configs/evaluation/transductive.yaml \
  --dataset-file configs/evaluation/benchmark_54_transductive.txt \
  --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth

python scripts/run_many.py \
  -c configs/evaluation/inductive.yaml \
  --dataset-file configs/evaluation/benchmark_54_inductive.txt \
  --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth
```

`run_many.py` also accepts comma-separated entries through `--datasets`, for
example `FB15k237Inductive:v1,FB15k237Inductive:v2`. Besides the per-run logs,
it writes a combined `merc_results_<timestamp>.csv` under `output/`.

Most suite loaders download their raw data on first use. A failed or retired
upstream URL must be handled by placing the expected files in the loader's
`raw/` directory; the exception message identifies that directory.

### Custom datasets

Every triple file is headerless text with exactly one `head relation tail`
triple per non-empty line, separated by tabs or whitespace. Entity and relation
identifiers must not contain whitespace; comments and column headers are not
accepted. A local transductive graph uses:

```text
datasets/my_graph/raw/train.txt
datasets/my_graph/raw/valid.txt
datasets/my_graph/raw/test.txt
```

Evaluate it without editing `merc/datasets.py`:

```bash
python scripts/run.py \
  -c configs/evaluation/custom_transductive.yaml \
  --data_name my_graph --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth
```

On macOS, use `custom_transductive_mps.yaml` and omit `--gpus`; this selects
MPS and reduces the batch size to 1.

A local inductive graph uses separate training and inference fact graphs:

```text
datasets/my_inductive/v1/raw/transductive_train.txt
datasets/my_inductive/v1/raw/inference_graph.txt
datasets/my_inductive/v1/raw/inf_valid.txt
datasets/my_inductive/v1/raw/inf_test.txt
```

```bash
python scripts/run.py \
  -c configs/evaluation/custom_inductive.yaml \
  --data_name my_inductive --version v1 --valid_on_inf true \
  --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth
```

Set `valid_on_inf` to `false` only when `inf_valid.txt` is defined over the
training graph. On macOS, use `custom_inductive_mps.yaml` and omit `--gpus`.
Use a new dataset name or delete that dataset's `processed/` cache after
changing raw files.

## Pre-training

Start full MERC pre-training on the FB15k-237, WN18RR, and CoDEx-M mixture:

```bash
python scripts/pretrain.py \
  -c configs/pretraining/merc.yaml \
  --gpus '[0]'
```

Use `configs/pretraining/merc_flash.yaml` for MERC Flash. Multi-GPU execution
uses PyTorch distributed data parallel. For four GPUs, run:

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  scripts/pretrain.py \
  -c configs/pretraining/merc.yaml \
  --gpus '[0,1,2,3]'
```

The configured per-process batch size is 16, giving the reported global batch
size of 64 with four processes. Checkpoints and logs are written below
`output/Merc/JointDataset/<timestamp>/`. Pre-training is computationally
intensive; the supplied checkpoints are the recommended entry point for
zero-shot evaluation.

## Relation-state probes

Install `requirements-analysis.txt`, then generate the fixed-checkpoint MERC
diagnostics used for MetaFam and NELL-995:

```bash
python scripts/probe_metafam.py \
  -c configs/probes/metafam.yaml \
  --ckpt checkpoints/merc.pth

python scripts/probe_nell995.py \
  -c configs/probes/nell995.yaml \
  --ckpt checkpoints/merc.pth
```

Each run creates a `quantitative_relation_metrics/` directory containing the
sampling manifest, sampled relation-context embeddings, per-relation metrics,
centroid-pair rows, summary CSV/JSON files, and heatmaps.

The matching Trix-side code is included under `adapters/trix/`. It uses the
same sampling helper and output schema. With an official Trix checkout and its
checkpoint, run directly from the artifact without modifying the Trix source:

```bash
PYTHONPATH=/absolute/path/to/TRIX/src:/absolute/path/to/merc_artifact \
python adapters/trix/probe_metafam.py \
  -c configs/probes/trix_metafam.yaml \
  --data_root /absolute/path/to/TRIX/datasets \
  --output_root /absolute/path/to/probe_output/trix \
  --gpus '[0]' \
  --ckpt /absolute/path/to/TRIX/entity_prediction.pth

PYTHONPATH=/absolute/path/to/TRIX/src:/absolute/path/to/merc_artifact \
python adapters/trix/probe_nell995.py \
  -c configs/probes/trix_nell995.yaml \
  --data_root /absolute/path/to/TRIX/datasets \
  --output_root /absolute/path/to/probe_output/trix \
  --gpus '[0]' \
  --ckpt /absolute/path/to/TRIX/entity_prediction.pth
```

`data_root` must make the Trix `Metafam` and `NELL995` loaders read the same raw
splits as MERC. Compare the resulting matched
`quantitative_relation_metrics/` directories with:

```bash
python scripts/compare_relation_probes.py \
  --merc /path/to/merc/quantitative_relation_metrics \
  --trix /path/to/trix/quantitative_relation_metrics \
  --output /path/to/paired_comparison
```

The comparison first requires identical sampled-triple SHA-256 values, then
writes `paired_relation_comparison.csv`, `paper_summary_comparison.csv`, and
`paired_model_comparison.json`.

## Biomedical graph construction

The first stage queries a running Neo4j instance containing the integrated CKG.
The integrated build of CKG is derived from the ckg_latest_4.2.3.dump and data.zip. 
Both ckg_latest_4.2.3.dump and data.zip are obtained directly from the official [CKG Documentation Page](https://ckg.readthedocs.io/en/latest/ckg_builder/graphdb-builder.html).
It groups relationships by their non-empty `source` property and retains all
relationship properties in a four-column export. Set the connection credentials
without placing a password in the repository:

```bash
export NEO4J_URI='bolt://localhost:7687'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='your-password'

python tools/biomedical/extract_ckg_triples.py \
  work/00_exported
python tools/biomedical/refine_relation_labels.py \
  work/00_exported work/01_refined
python tools/biomedical/deduplicate_triples.py \
  work/01_refined work/02_deduplicated
python tools/biomedical/scrub_symmetric_edges.py \
  work/02_deduplicated work/03_scrubbed
python tools/biomedical/reshape_graphs.py \
  work/03_scrubbed work/04_reshaped \
  --seed 2026 --report work/reshape_report.csv
python tools/biomedical/sparsify_graphs.py \
  work/04_reshaped work/05_sparsified \
  --target-average-degree 15 --pareto-ratio 0.25 \
  --report work/sparsify_report.csv
python tools/biomedical/split_datasets.py \
  work/05_sparsified datasets \
  --valid-ratio 0.1 --test-ratio 0.1 --seed 2026 \
  --report work/split_report.csv
```

The export uses Neo4j internal relationship IDs for batching and checkpointing.
Keep the database unchanged during a run. Resume with the generated checkpoint
after an interruption; after a completed export, start any new run in an empty
output directory to avoid appending duplicate rows.

Source-specific preprocessing is mandatory before deduplication or any graph
operation. Every exported source was processed
by the single deterministic entry point
`refine_relation_labels.py`. It also canonicalizes three CKG
source labels to the benchmark names `GENOMICS_ENGLAND`, `I2D`, and
`Intact-MutationDs`. Unsupported CKG sources are skipped and cannot enter later
stages without a dedicated preprocessing rule. The output and all subsequent
stages are headerless, three-column TSV files.

The six oversized evaluation graphs use the fixed seed and ratios in
`configs/biomedical/evaluation_subsampling.yaml`. Apply all six entries after
the initial split; the command preserves the full validation/test files as
`validf.txt` and `testf.txt`, then writes the sampled `valid.txt` and `test.txt`:

```bash
python tools/biomedical/apply_evaluation_subsampling.py \
  datasets \
  --config configs/biomedical/evaluation_subsampling.yaml
```

After constructing the final files, verify them against the paper's 93-entry
manifest:

```bash
python scripts/generate_sha256_manifest.py \
  /path/to/biomedical/datasets generated_biomedical_splits_sha256.tsv \
  --dataset-list configs/biomedical/datasets.txt
```

The repository does not redistribute CKG exports or materialized biomedical
splits. Integrated CKG sources retain their own terms, and DrugBank data are
access-controlled. `references/biomedical_sources.bib` records the source
citations used by the paper.

The biomedical classes in `merc/datasets.py` do not download data. Before
evaluation, place each materialized split under the class's dataset name:

```text
datasets/bhf-ucl/raw/train.txt
datasets/bhf-ucl/raw/valid.txt
datasets/bhf-ucl/raw/test.txt
```

The same layout is required for all 31 names in
`configs/biomedical/datasets.txt` (93 split files in total).

## Reproduction scope and external inputs

With the included source, configurations, and frozen checkpoints, users can
run MERC and MERC Flash inference on supported datasets, execute the supplied
single- or multi-GPU pre-training procedure, generate MERC-side relation-state
probes, run the supplied matched Trix probe against an external Trix checkout,
mine and evaluate relation-pattern subsets, construct biomedical graphs from a
compatible CKG database, and run the included model-specific profilers.

The following inputs are intentionally not included:

- the CKG Neo4j database snapshot, its source exports, and the 93 materialized
  biomedical split files; without the original snapshot, the manifest can
  verify an existing copy but cannot make a current CKG export bit-identical;
- a Trix source tree and checkpoint when running the included Trix probe;
- Ultra, Trix, and Flock source trees, checkpoints, and datasets when running
  their included profiling adapters; and
- H100 profiling logs and the numerical baseline-comparison outputs. Those
  values are already reported in the paper and are not duplicated here.

Most non-biomedical datasets in `merc/datasets.py` are downloaded by their
upstream loaders on first use. If an upstream host is unavailable, users must
place the corresponding raw files in that loader's `raw/` directory manually.

## Relation-pattern analysis

The miner examines the source split files in each discovered `raw/` directory,
assigns relations at threshold 0.97, and writes test subsets next to the
original evaluation file. Generated subsets are excluded from later mining
runs. The filenames end in `_symmetry.txt`, `_anti_symmetry.txt` (the paper's
low-reciprocity subset), or `_composition.txt`:

```bash
python tools/relation_patterns/mine_relation_patterns.py \
  datasets --threshold 0.97
```

Pass `--overwrite` when rerunning the command if any target subset already
exists. A subset with no matching triples is not written. For ordinary datasets
the miner partitions one of `test.txt`, `inf_test.txt`, or `test_ind.txt`. Two
loader-specific cases need extra care:

- Grail's default final test set concatenates `valid_ind.txt` and
  `test_ind.txt`, so the miner writes pattern subsets for both files.
- Files below a `SparseKG` directory use `(head, tail, relation)` order. The
  miner interprets that order correctly but preserves it in the subset files.

Do not overwrite the original test split to evaluate a subset. Stage it under a
new custom-dataset name. For a standard `(head, relation, tail)` transductive
graph:

```bash
python tools/relation_patterns/stage_pattern_dataset.py transductive \
  --train /path/to/source/raw/train.txt \
  --valid /path/to/source/raw/valid.txt \
  --subset /path/to/source/raw/test_anti_symmetry.txt \
  --output-root datasets \
  --name source_low_reciprocity

python scripts/run.py \
  -c configs/evaluation/custom_transductive.yaml \
  --data_name source_low_reciprocity --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth
```

`CustomTransductive` expects `(head, relation, tail)`, whereas SparseKG source
and mined subset files remain `(head, tail, relation)`. Convert every staged
split first; for example:

```bash
mkdir -p work/sparsekg_fb15k237_10
awk 'NF == 3 { print $1 "\t" $3 "\t" $2 }' \
  datasets/SparseKG/FB15K-237-10/raw/train.txt \
  > work/sparsekg_fb15k237_10/train.txt
awk 'NF == 3 { print $1 "\t" $3 "\t" $2 }' \
  datasets/SparseKG/FB15K-237-10/raw/valid.txt \
  > work/sparsekg_fb15k237_10/valid.txt
awk 'NF == 3 { print $1 "\t" $3 "\t" $2 }' \
  datasets/SparseKG/FB15K-237-10/raw/test_anti_symmetry.txt \
  > work/sparsekg_fb15k237_10/test_anti_symmetry.txt

python tools/relation_patterns/stage_pattern_dataset.py transductive \
  --train work/sparsekg_fb15k237_10/train.txt \
  --valid work/sparsekg_fb15k237_10/valid.txt \
  --subset work/sparsekg_fb15k237_10/test_anti_symmetry.txt \
  --output-root datasets \
  --name sparsekg_fb15k237_10_low_reciprocity
```

For an inductive graph, stage the training fact graph, inference fact graph,
validation queries, and mined test subset explicitly:

```bash
python tools/relation_patterns/stage_pattern_dataset.py inductive \
  --train-graph /path/to/transductive_train.txt \
  --inference-graph /path/to/inference_graph.txt \
  --valid /path/to/inf_valid.txt \
  --subset /path/to/inf_test_composition.txt \
  --output-root datasets \
  --name source_composition --version v1

python scripts/run.py \
  -c configs/evaluation/custom_inductive.yaml \
  --data_name source_composition --version v1 --valid_on_inf true \
  --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth
```

Use `--valid_on_inf true` only when the validation queries belong to the
inference graph; use `false` when they belong to the training graph.

For Grail, reproduce the loader's default merged-test protocol by concatenating
the two mined subsets in loader order, `valid_ind` followed by `test_ind`, then
stage the transductive validation split with `--valid_on_inf false`:

```bash
mkdir -p work/grail_composition
cat /path/to/grail/raw/valid_ind_composition.txt \
  /path/to/grail/raw/test_ind_composition.txt \
  > work/grail_composition/inf_test.txt

python tools/relation_patterns/stage_pattern_dataset.py inductive \
  --train-graph /path/to/grail/raw/train.txt \
  --inference-graph /path/to/grail/raw/train_ind.txt \
  --valid /path/to/grail/raw/valid.txt \
  --subset work/grail_composition/inf_test.txt \
  --output-root datasets \
  --name grail_composition --version v1

python scripts/run.py \
  -c configs/evaluation/custom_inductive.yaml \
  --data_name grail_composition --version v1 --valid_on_inf false \
  --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth
```

If the miner did not write one of the two Grail subset files because it was
empty, concatenate only the file that exists. If neither file exists, that
pattern has no Grail test examples to evaluate.

The normal evaluator writes the subset MRR and Hits values to its `log.txt`.
The paper's aggregate tables and baseline outputs are intentionally not copied
into the artifact.

## Efficiency profiling

Complete MERC, Ultra, Trix, and Flock inference/pre-training profiler scripts
are under `adapters/profiling/`. The protocol uses batch size 8, five warm-up
batches, and 50 measured batches. See `tools/profiling/README.md` for exact
commands, configuration requirements, and output fields. Profiling requires
CUDA and is separate from the complete evaluation used to obtain MRR and Hits;
the numerical paper results are not duplicated in this package.

## License and third-party software

MERC is distributed under the MIT License. Baseline implementations,
checkpoints, and datasets remain governed by their respective licenses and
terms. Exact baseline revisions and checkpoint hashes are listed in
`manifests/baseline_checkpoints.tsv`; see `NOTICE.md` for attribution and data
redistribution boundaries.

This project is based on and modifies the [ULTRA repository](https://github.com/DeepGraphLearning/ULTRA).
