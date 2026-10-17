# Conditional pruning and performance protocol

Status (17 October 2026): remote requests, sources and available summaries recovered;
conditional scan and latency benchmark launched. See their hash-addressed request
and call records under `conditional_importance/` and `latency_benchmark/`.
The authoritative saved protocol is the `phase7_adaptive_pruning_plan` field in
`../published/2026-10-04/results.json`. This document specifies execution order;
it does not replace that record or claim newly chosen thresholds were historical.

1. Recover original Phase 6/7 notebooks, workers, requests, saved inputs and
   checkpoint sidecars. Verify their published SHA-256 hashes using the inventory.
   Recover `data/phase7/blocks45_healing_v3/checkpoint_step_060.pt.remote.json`,
   then fetch and verify candidate A's checkpoint. Preserve original records.
2. Verify teacher and candidate A replay with the pinned model/runtime and fresh
   cache. Freeze all weights. Use the existing 10- and 50-prompt development
   panels and their saved early/middle/late inputs, seeds and conditioning.
3. Restore A before each independent trial, bypass one original block from
   0–3 or 6–31, and rebuild the cache independently. Never accumulate bypasses.
   Save input, checkpoint, worker and contract hashes and individual predictions.
   Report L2(v_bypass-v_A)/L2(v_A), teacher-relative L2 for A and bypass,
   and 100*(bypass_teacher_error-A_teacher_error) in percentage points.
   Summarize equal-prompt/equal-stage means, per-stage and prompt-group means,
   p10/p50/p90, maxima and worst prompt/stage regressions. Reject nonfinite
   outputs and zero reference norms. Negative teacher deltas may be cancellation.
4. Benchmark teacher, Phase 6 selected student and A on the same GPU, precision,
   pinned scheduler, 40 steps, prompts/seeds, resolution, cache/offload policy and
   batch size. Record cold loading separately. Warm each variant, use balanced
   randomized paired order with at least 30 timed pairs, synchronize CUDA around
   timing, and reset peak memory stats before each run. Report end-to-end latency,
   denoiser latency, throughput, peak allocated/reserved VRAM, median/p90, paired
   speed ratios with bootstrap CIs, raw trials and environment identities.
   Measure each model's residency separately so teacher weights do not inflate A.
5. Before accepting A as the basis of further pruning, register a fresh larger
   blinded matched image panel, scoring rubric, noninferiority tolerances, sample
   size and decision rule. Include text, counting, spatial and multi-object groups;
   compare A to both teacher and Phase 6 selected student. Preserve blind labels
   until scoring is locked. Historical velocity gate remains 0.0652 and failed.
6. Use the conditional scan to propose a learned replacement. Before training,
   freeze a versioned request with numeric overall/per-stage/worst-case velocity
   tolerances, latency and blinded-image acceptance rules. Fit only on TRAIN
   inputs captured inside A; validate the complete student against A and teacher.
   Accept only if all preregistered gates pass. Rerank after every accepted cut.
7. Once architecture is stable and accepted, register the speed-distillation LoRA
   experiment (40 to approximately 8 steps): fixed sigma schedule, teacher
   trajectory endpoints, average-velocity targets, rank, optimizer, TRAIN-only
   sampling, validation and independent quality/latency gates (Phases 9–12).
   Do not start training or pick a block from historical unconditional scores.

The workers now use the recovered requests, candidate A checkpoint hash, fixed
input hashes and original cache/replay contract. The scan verifies all 90 teacher
controls and historical candidate A predictions before any bypass trial. The
saved historical protocol remains unchanged with its original
`planned_not_executed` status; new execution records are stored separately.
