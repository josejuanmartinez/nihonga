# Nihonga

Compression research for Qwen Image 2.1: 32 transformer blocks, approximately
6.9 billion DiT parameters. Run notebooks from this repository root.

The experiment archive is incomplete in this checkout. The original notebooks
end during Phase 6. The published 4 October record is preserved byte for byte in
`data/published/2026-10-04/results.json`; its source inventory is in
`reconciliation.json` beside it. These are published summaries, not recovered
training notebooks or independently verified experiment outputs.

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

On 17 October 2026 the configured Modal credentials authenticated, but exposed
only environment `main`, with no volumes. The recorded results volume
`nihonga-phase3-artifacts` was not found. Restore access to that archive and
`nihonga-hf-cache`, including original requests, workers, replay packets and
checkpoint hashes, before scheduling the next experiment. Do not regenerate
missing historical artifacts and label them recovered.
