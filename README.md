# Nihonga

Compression research for Qwen Image 2.1: 32 transformer blocks, approximately
6.9 billion DiT parameters. Run notebooks from this repository root.

The original notebooks end during Phase 6. Later Modal experiment records and
embedded worker sources have now been recovered under `data/recovered/`.
`phase6-7-recovered-continuation.ipynb` is an explicitly reconstructed notebook;
the original later notebooks were not present in the archive.
`phase7-adaptive-pruning.ipynb` is the Phase 7 notebook: every step with its exact
worker code (as executed), results and interpretation. Re-running it reuses saved
results; it only launches Modal work for a missing result when `ALLOW_GPU = True`.
New experiments are added there as notebook steps.
The published 4 October record is preserved byte for byte in
`data/published/2026-10-04/results.json`; its source inventory is in
`reconciliation.json` beside it. These are published summaries, not recovered
training notebooks. The reconciliation inventory distinguishes exact recovered
source files from local-only review records still missing.

The Phase 6 selected student removes original block 5, using a rank-256 pretrained
bridge (5.518% pilot validation velocity error). Full-weight healing was tested
according to the published record, but not promoted: the blinded 129-prompt test
gave update 120 a 2.3 percentage-point gain, CI [0.0, 5.4], below its 5-point bar.
The 45-prompt text review tied at 41 clear passes per model.

Phase 7 candidate A removes original blocks 4 and 5 together, using one rank-512
region bridge and 60 full-weight healing updates. Its wider validation velocity
error is 7.410%, above the historical 6.52% gate. The 30-prompt unblinded image
screen does not establish promotion. Removing blocks 2–5 failed.

See `data/phase7/NEXT_EXPERIMENT.md` for the execution prerequisites and order.
No conditional ranking or measured speedup is available. Candidate A remains
provisional; the selected Phase 6 artifact remains the fallback.

To audit a local copy of the published evidence without GPU calls:

```powershell
python tools/reconcile_record.py --site D:\niwaki-page\nihonga
```

On 17 October 2026 the user supplied access to the old Modal account, exposing
`nihonga-phase3-artifacts` and `nihonga-hf-cache`. Small records and embedded source
code are preserved in Git; selected large tensors are downloaded into `.archive/`
with SHA-256 ledgers. The inventories cover the full archive without requiring
hundreds of GiB of superseded optimizer checkpoints. Do not regenerate missing
historical artifacts and label them recovered. See `data/recovered/README.md`.

Modal account policy: `MODAL_TOKEN_ID_OLD` / `MODAL_TOKEN_SECRET_OLD` are for
archive downloads only. Uploads and all CPU/GPU jobs use the new account's
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`. Compute launchers explicitly reject
`--old-account`. Initial jobs launched in the old account were stopped; their
partial records are retained under `attempts/old-account-aborted/`.
See `data/recovery/migration-status.json` before removing the old credentials.
