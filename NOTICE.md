# Third-party notice

MERC is released under the MIT License. Portions of its structural
knowledge-graph infrastructure are derived from the MIT-licensed ULTRA codebase:

- ULTRA: https://github.com/DeepGraphLearning/ULTRA
- Copyright (c) 2023 MilaGraph

The evaluation also uses the authors' upstream implementations and checkpoints:

- Trix: https://github.com/yuchengz99/TRIX
- Flock: https://github.com/jw9730/flock (MIT; Copyright (c) 2025 Jinwoo Kim)

Ultra, Trix, and Flock source trees and checkpoints are not redistributed in
this package. Their exact source revisions and checkpoint digests are recorded for
provenance in `manifests/baseline_checkpoints.tsv`.

Datasets and CKG-derived triples are governed by their respective providers'
terms. The MERC software license does not grant rights to redistribute those
data. In particular, DrugBank is access-controlled, and CKG's software license
does not replace the licenses of databases integrated by CKG. This repository
therefore provides biomedical construction code, source citations,
configuration, and final-file hashes rather than source exports or materialized
biomedical splits.
