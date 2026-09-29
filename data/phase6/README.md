# Phase 6 research record

`phase6.ipynb` is the step-by-step notebook. The Python files here implement its
replay diagnostics, teacher controls, hidden-state capture, bridge assessment, and
two parameter-efficient healing pilots. JSON summaries, requests, checksums,
manifests are versioned with the source so the saved
findings can be reviewed without recomputing any model calls.

Execution logs remain local beside the tensor packets; validated summaries and
the notebook outputs contain the results needed for review.

The `.pt` files are deliberately outside Git: this phase currently has about
3.4 GiB of tensor packets, including individual checkpoints above GitHub's
100 MiB per-file limit. They remain at their original local paths. The
experiment reports record remote artifact locations and SHA-256 hashes where
remote copies were made. Do not replace a missing tensor with a new forward pass
without first checking the recorded remote copy and provenance.

The completed block-5 healing pilots selected update 0: validation velocity
error was 5.518% before healing, and no saved trained checkpoint improved it.
The pilot protocol passed, but full architecture healing and image evaluation
remain open.
