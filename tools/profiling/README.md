# Fixed-window CUDA profiling

The paper profiles one NVIDIA H100 80GB GPU with a global batch size of 8. Each
run executes five warm-up batches and then measures 50 batches. Accuracy comes
from a separate complete evaluation; the profiling scripts exit successfully
after the 55-batch window.

Install the additional logging dependencies in each model's runnable
environment:

```bash
python -m pip install -r requirements-profiling.txt
```

## Included code

The complete model-specific scripts used for the protocol are included here:

```text
adapters/profiling/merc/time.py       MERC inference
adapters/profiling/merc/timet.py      MERC pre-training
adapters/profiling/ultra/time.py      Ultra inference
adapters/profiling/ultra/timet.py     Ultra pre-training
adapters/profiling/trix/time.py       Trix inference
adapters/profiling/trix/timet.py      Trix pre-training
adapters/profiling/flock/time.py      Flock inference
adapters/profiling/flock/timet.py     Flock pre-training
```

`time.py` measures both tail- and head-prediction forwards with CUDA events and
reports time per forward pass. `timet.py` measures complete optimization
batches with synchronized wall-clock time. Both report peak allocated CUDA
memory. CPU RSS is printed only as a diagnostic and is not the paper's GPU
memory quantity.

`tools/profiling/fixed_window.py` contains the same measurement window as a
small reusable utility for readers who prefer to instrument another runner.

## MERC

Run transductive inference profiling from the artifact root:

```bash
python adapters/profiling/merc/time.py \
  -c configs/evaluation/transductive.yaml \
  --dataset YAGO310 \
  --epochs 0 --bpe null --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth
```

For an inductive graph, use the batch-size-8 profiling configuration:

```bash
python adapters/profiling/merc/time.py \
  -c configs/profiling/merc_inductive.yaml \
  --dataset HM --version 1k --gpus '[0]' \
  --ckpt /absolute/path/to/merc_artifact/checkpoints/merc.pth
```

Profile pre-training batches without running a complete pre-training job:

```bash
python adapters/profiling/merc/timet.py \
  -c configs/profiling/merc_pretraining.yaml \
  --gpus '[0]'
```

Logs are written below `output/` or `output/profiling/`, according to the
selected configuration.

## Ultra

Use an official Ultra checkout at the revision recorded in
`manifests/baseline_checkpoints.tsv`. Create the two editable profiling
configurations first:

```bash
cp /absolute/path/to/ULTRA/config/transductive/inference.yaml \
  /absolute/path/to/ULTRA/config/transductive/inference_profile.yaml
cp /absolute/path/to/ULTRA/config/transductive/pretrain_3g.yaml \
  /absolute/path/to/ULTRA/config/transductive/pretrain_3g_profile.yaml
```

Set `train.batch_size: 8` and correct only local dataset/output paths; retain
the architecture, sampling, and optimizer fields from the recorded upstream
revision. Then run:

```bash
PYTHONPATH=/absolute/path/to/ULTRA \
python /absolute/path/to/merc_artifact/adapters/profiling/ultra/time.py \
  -c /absolute/path/to/ULTRA/config/transductive/inference_profile.yaml \
  --dataset YAGO310 --epochs 0 --bpe null --gpus '[0]' \
  --ckpt /absolute/path/to/ULTRA/ckpts/ultra_3g.pth

PYTHONPATH=/absolute/path/to/ULTRA \
python /absolute/path/to/merc_artifact/adapters/profiling/ultra/timet.py \
  -c /absolute/path/to/ULTRA/config/transductive/pretrain_3g_profile.yaml \
  --gpus '[0]'
```

## Trix

Use the recorded Trix revision. Create editable copies, set
`train.batch_size: 8`, and update dataset/output paths:

```bash
cp /absolute/path/to/TRIX/config/run_entity_transductive.yaml \
  /absolute/path/to/TRIX/config/run_entity_transductive_profile.yaml
cp /absolute/path/to/TRIX/config/pretrain_entity.yaml \
  /absolute/path/to/TRIX/config/pretrain_entity_profile.yaml
```

```bash
PYTHONPATH=/absolute/path/to/TRIX/src \
python /absolute/path/to/merc_artifact/adapters/profiling/trix/time.py \
  -c /absolute/path/to/TRIX/config/run_entity_transductive_profile.yaml \
  --dataset YAGO310 --epochs 0 --bpe null --gpus '[0]' \
  --ckpt /absolute/path/to/TRIX/entity_prediction.pth

PYTHONPATH=/absolute/path/to/TRIX/src \
python /absolute/path/to/merc_artifact/adapters/profiling/trix/timet.py \
  -c /absolute/path/to/TRIX/config/pretrain_entity_profile.yaml \
  --gpus '[0]'
```

## Flock

Use the graph-appropriate official Flock evaluation YAML, preserving its walk
count, walk length, test-sample count, and seed. For example, create editable
copies as follows:

```bash
cp /absolute/path/to/FLOCK/src_entity/config/zeroshot_transductive/n128_ensemble16.yaml \
  /absolute/path/to/FLOCK/src_entity/config/zeroshot_transductive/profile.yaml
cp /absolute/path/to/FLOCK/src_entity/config/pretrain_3g.yaml \
  /absolute/path/to/FLOCK/src_entity/config/pretrain_3g_profile.yaml
```

Set `train.batch_size: 8` and any separate test-loader batch-size field to 8,
then run:

```bash
PYTHONPATH=/absolute/path/to/FLOCK/src_entity:/absolute/path/to/FLOCK/graph-walker \
python /absolute/path/to/merc_artifact/adapters/profiling/flock/time.py \
  -c /absolute/path/to/FLOCK/src_entity/config/zeroshot_transductive/profile.yaml \
  --dataset YAGO310 --epochs 0 --bpe null --gpus '[0]' \
  --ckpt /absolute/path/to/FLOCK/checkpoints/flock_entity.pth

PYTHONPATH=/absolute/path/to/FLOCK/src_entity:/absolute/path/to/FLOCK/graph-walker \
python /absolute/path/to/merc_artifact/adapters/profiling/flock/timet.py \
  -c /absolute/path/to/FLOCK/src_entity/config/pretrain_3g_profile.yaml \
  --gpus '[0]'
```

An OOM under Flock's official walk configuration is recorded as OOM. A
reduced-walk diagnostic is a different operating point and must be labeled
separately.

## Output contract

Every successful window prints these values to the runner's `log.txt`:

- inference mean and standard deviation in seconds per forward pass;
- pre-training mean seconds per optimization batch;
- peak allocated GPU memory in MB; and
- the five-warm-up/50-measured-batch completion message.

The selected evaluation split must contain at least 55 loader batches (440
queries at batch size 8); the selected pre-training mixture must likewise
provide at least 55 optimization batches. This is required for the fixed window
to complete.

Do not combine these early-exit logs with MRR or Hits. Obtain ranking metrics
from the normal full evaluator under the same checkpoint and dataset.
