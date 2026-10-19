# NIHONGA: Qwen Image 2.1 Compression and Acceleration Plan

## Status (19 October 2026): Nihonga v1.0 released

**Nihonga v1.0** is Qwen-Image-2.1 with two of its 32 transformer blocks replaced by learned bridges and a fused
Lightning LoRA, sampled in 10 steps instead of 40.

| | Qwen-Image-2.1 | Nihonga v1.0 |
|---|---|---|
| Transformer | 32 blocks, 7.115 B parameters | 30 blocks + 2 bridges (slots 5 and 20), 6.729 B |
| Steps | 40 | 10 (points 0, 1, 3, 5, 10, 15, 20, 25, 30, 35 of the 40-step schedule) |
| Time per 1024 × 1024 image, one H100 | 6.89 s | 1.75 s (3.9× faster) |
| Exact spelling, 16 hard items | 16 | 15 |
| Peak GPU memory | 39.6 GB | 39.3 GB (the 17.5 GB text encoder is unchanged) |

Weights: https://huggingface.co/jjmcarrascosa/Nihonga-v1.0 · release page: https://niwakiai.com/nihonga/ ·
research log: https://josejuanmartinez.github.io/research/nihonga/ · shipped code: `release/nihonga.py`.
Built with Qwen; Qwen Research License (non-commercial).

### What we did, by notebook

| Notebook | Outcome |
|---|---|
| `phase0-exploration` | 32-block single-stream DiT, hidden 4096, ~7.1 B parameters; no CFG at default |
| `phase1-20k-pilot-dataset` | 20k prompts (17,991 train / 1,002 validation / 1,007 internal test), text rendering weighted ×2 |
| `phase2-20k-pilot-baseline-established` | Teacher baseline on the internal test split and Qwen-Image-Bench |
| `phase3-identify-layers-to-remove` | Block importance scans; first removal candidates |
| `phase4-insert-bridge-modules` | Residual bottleneck bridges inserted and verified |
| `phase5-pretrain-bridge-candidates` | Bridges pretrained; block 5, rank 256 selected (5.52% velocity error) |
| `phase6-one-block-student` | One-block student **B** accepted; full-weight healing not promoted after blinded QA |
| `phase7-second-block-removal` | Candidate A (blocks 4-5) rejected. Block 13 looked cheapest by velocity but carries spelling (teacher without it: ~5/12 signs); no bridge or healing recovered it. A **text-aware scan** chose **block 20**. Attention-conditioned bridges (8 heads, rank-1024 MLP, own prompt cache, ~25 M weights) trained end to end for block 20 (step 600) and a rebuilt block 5 (step 200) |
| `phase8-heal-30-block-student` | LoRA healing improved velocity (8.09% → 7.84%) but not spelling, so **v1 stays unhealed and keeps two removed blocks** (decision: first release with 2 blocks, not 4) |
| `phase9-lightning-baseline` | Few steps without distillation are unusable; the published LightX2V 4-step LoRA spells 13/16 ("Acchor"). Benchmark: FP32 bridges made v1 slower than the teacher; **BF16 bridges** make it ~4% faster per step |
| `phase10-lightning-lora` | Our own **Lightning LoRA** (rank 32, attention of the 30 blocks) by trajectory distillation. Teacher targets plateaued (12/16, 5 of 25 structure failures); **OCR text-region weighting** (letters ×10) helped text; **training on v1's own trajectories** (500 text + 500 general prompts) fixed the Rusty Anchor; a **drift test** showed structure failures are made in the first jump; a **gentler start** (3 small steps) plus the fused LoRA gives 10 steps with 15/16 spelling and 2 of 25 structure differences. Release build verified (identical reload, two scheduler routes pixel-identical) |

### Key lessons

- Spelling breaks first, and average metrics (velocity, latent distance, PSNR) cannot see it. Every run needs a
  quantitative and a qualitative test, and every checkpoint is kept.
- Choose blocks to prune by text, not only by average error.
- Small learned modules must run in BF16 at inference, or they cost more than the blocks they replace.
- Distill towards targets the student can draw (its own trajectories), not the teacher's.
- Few-step failures are made at the start; a gentler first segment fixes them without retraining.

## Goal

We create a significantly smaller and faster Qwen Image variant while preserving as much quality, prompt adherence, and general capability as possible.

The full pipeline combines:

1. Structured architecture pruning
2. Lightweight bridge modules to replace removed layers
3. Architecture healing / distillation
4. Speed distillation to reduce denoising steps
5. Optional quantization
6. Systematic benchmarking against the original model

---

## Phase 0 - Exploratory
Let's explore how Qwen Image 2.1 model looks like in terms of architecture

## Phase 1 — Build the Prompt Dataset

We create a master dataset of approximately 100k synthetic prompts.

The dataset should follow the same capability structure as Qwen-Image-Bench, but we do not reuse the benchmark prompts themselves.

Main capability groups should include:

- Prompt alignment
  - counting
  - colors
  - shapes
  - materials
  - multiple objects
  - actions
  - spatial relations
  - composition and layout

- Visual quality
  - textures
  - transparent materials
  - reflective materials
  - skin
  - fur
  - foliage
  - low-light scenes
  - fine detail

- Aesthetics
  - composition
  - lighting
  - color harmony
  - portraits
  - landscapes
  - cinematic framing
  - illustration styles

- Real-world fidelity
  - architecture
  - interiors
  - vehicles
  - clothing
  - everyday objects
  - culturally specific scenes
  - physically plausible interactions

- Creative generation
  - concept art
  - character design
  - posters
  - comics
  - text rendering
  - surreal scenes
  - storytelling
  - unusual compositions

We store metadata for every prompt:

```json
{
  "prompt": "...",
  "dimension": "alignment",
  "subdimension": "spatial_relation",
  "facet": "inside_left_of",
  "difficulty": 3,
  "style": "photographic",
  "aspect_ratio": "1:1",
  "language": "en"
}
```

We use a mixture of:

- structured templates
- combinatorial generation
- LLM rewriting
- multiple difficulty levels
- multiple prompting styles

We start with a 20k pilot dataset, then we generate the full 100k.

We keep separate:

```text
TRAIN
VALIDATION
INTERNAL TEST
QWEN-IMAGE-BENCH
```

We use Qwen-Image-Bench only for evaluation.

---

## Phase 2 — Establish the Original Baseline

Before modifying the model, we benchmark the original Qwen Image.

We record:

- Qwen-Image-Bench overall score
- category-level scores
- prompt adherence
- text rendering
- spatial reasoning
- visual quality
- aesthetic quality
- inference latency
- peak VRAM
- throughput
- number of denoising steps
- model size

We also create a fixed internal evaluation set with fixed:

```text
prompt
seed
resolution
aspect ratio
scheduler configuration
```

We compare every future model against exactly the same conditions.

---

## Phase 3 — Identify Layers to Remove

We start conservatively.

Example:

```text
Original: 32 transformer blocks
Selected Phase 6 student: 31 blocks + rank-256 bridge replacing block 5
Provisional Phase 7 candidate A: 30 blocks + region bridge replacing blocks 4-5
Failed experiment: 28 blocks, blocks 2-5 removed
```

We evaluate layer importance before pruning.

Possible signals:

- activation similarity between consecutive blocks
- output change when bypassing individual blocks
- hidden-state cosine similarity
- teacher output degradation after temporary layer removal
- gradient / sensitivity measurements

We prefer removing groups of relatively redundant layers rather than blindly keeping every Nth layer.

---

## Phase 4 — Insert Bridge Modules

Instead of simply deleting a block or group of blocks, we replace them with a very cheap learned bridge.

Teacher:

```text
h[n-1]
  ↓
Block n
  ↓
h[n+1]
```

Student:

```text
h[n-1]
  ↓
Bridge
  ↓
h'[n+1]
```

The simplest bridge is:

$$
h' = h + Wh
$$

A better version is a residual bottleneck adapter:

$$
h' = h + W_{\text{up}} \left(\mathrm{GELU} \left(W_{\text{down}}h \right) \right)
$$

For example:

```text
hidden dimension: 3072
bottleneck: 256–512
```

If several consecutive blocks are removed:

```text
Teacher:
h19 → B20 → B21 → B22 → h23

Student:
h19 → bridge → h23
```

We use one bridge for the removed region instead of recreating one mini-module per removed block.

The bridge should remain much cheaper than the Transformer blocks it replaces.

---

## Phase 5 — Pretrain the Bridges

Before performing full healing, we freeze the whole student except the bridges.

For each training sample:

```text
prompt
noise
timestep
latent x_t
```

We run the teacher and capture the hidden state after the removed region:

$$
h_{\text{target}} = h^{\text{teacher}}_{\text{after removed blocks}}
$$

Then we run the student bridge:

$$
h_{\text{bridge}} = \text{Bridge}(h_{\text{before}})
$$

We train with:

$$
L_{\text{bridge}} = \|h_{\text{bridge}}-h_{\text{target}}\|^2
$$

We prefer a normalized hidden-state loss:

$$
L_{\text{bridge}} = \| \text{normalize}(h_{\text{bridge}}) - \text{normalize}(h_{\text{target}}) \|^2
$$

This stage should be relatively cheap because only the bridge parameters are trainable.

---

## Phase 6 — Architecture Healing

After bridge pretraining, we train the pruned student to recover the behavior of the original teacher.

Teacher and student receive exactly the same:

```text
prompt
x_t
timestep
conditioning
```

Teacher:

```text
32-block frozen model
→ v_teacher
```

Student:

```text
pruned model + bridges
→ v_student
```

Main output loss:

$$
L_{\text{output}} = \|v_{\text{student}}-v_{\text{teacher}}\|^2
$$

We also perform hidden-state distillation between selected teacher and student layers:

$$
L_{\text{hidden}} = \sum_i \| \text{normalize}(h^S_i) - \text{normalize}(h^T_{map(i)}) \|^2
$$

Example mapping:

```text
Use original block IDs to align surviving slots, rather than compacted indices.
For candidate A, supervise the region bridge against teacher block 5's output.
Capture additional surviving slots only under a verified replay/cache contract.
```

We add bridge supervision if useful:

$$
L = L_{\text{output}} + \lambda_h L_{\text{hidden}} + \lambda_b L_{\text{bridge}}
$$

Initial experimental weighting could be:

```text
L_output = 1.0
L_hidden = 0.1
L_bridge = 0.1
```

These weights should be treated as starting points, not fixed values.

---

## Phase 7 — Healing Dataset Strategy

We use the master prompt dataset, but we sample broadly.

Healing should prioritize general capability preservation.

Example sampling:

```text
70% general prompts
30% hard / specialized prompts
```

We randomize:

- timestep
- noise
- seed
- aspect ratio
- resolution

The teacher provides the supervision.

No real-image dataset is required for the basic distillation stage.

The training sample is effectively:

```text
prompt
+ noise
+ timestep
→ teacher target
```

---

## Phase 8 — Evaluate the Pruned Model

After healing, we compare:

```text
Original 32-block teacher
vs
Pruned student before healing
vs
Pruned student after healing
```

We measure:

- Qwen-Image-Bench overall
- each benchmark category
- internal validation scores
- latency
- VRAM
- throughput

We pay special attention to regressions in:

- text rendering
- spatial relations
- counting
- multi-object scenes
- anatomy
- rare concepts
- complex composition

We do not only inspect the mean score.

We also track:

```text
p10
p50
p90
worst-performing prompts
largest teacher-student regressions
```

The goal is to understand what capabilities pruning damages.

---

## Phase 9 — Speed Distillation

We begin denoising-step reduction only after the smaller architecture is stable.

For example:

```text
Teacher student architecture:
accepted student architecture (31 or 30 blocks)
40 steps

Target:
accepted student architecture (31 or 30 blocks)
8 steps
```

Later:

```text
8 steps
→
4 steps
```

We do not initially combine aggressive architecture reduction and aggressive timestep reduction in one experiment.

---

## Phase 10 — Build Teacher Trajectory Targets

Suppose the student should jump:

```text
sigma 1.00
→
sigma 0.75
```

We let the teacher perform several smaller steps:

```text
1.00
→ 0.95
→ 0.90
→ 0.85
→ 0.80
→ 0.75
```

This produces:

$$
x^{\text{teacher}}_{\text{end}}
$$

The student makes a single prediction from:

$$
x_{\text{start}}
$$

We then construct the target average velocity:

$$
v_{\text{target}} = \frac{ x^{\text{teacher}}_{\text{end}} - x_{\text{start}} }{ \sigma_{\text{end}} - \sigma_{\text{start}} }
$$

Then we train:

$$
L_{\text{velocity}} = \|v_{\text{student}}-v_{\text{target}}\|^2
$$

This teaches one student forward pass to approximate several teacher forwards.

---

## Phase 11 — Student Timestep Schedule

For an 8-step model:

```text
1.00
0.875
0.75
0.625
0.50
0.375
0.25
0.125
0.00
```

For a 4-step model:

```text
1.00
0.75
0.50
0.25
0.00
```

During training, we sample intervals rather than always training the full trajectory.

Example:

```text
We choose an interval:
0.75 → 0.50

Teacher:
multiple small steps

Student:
one large step
```

Optionally, we introduce timestep jitter so the model does not overfit to a few exact sigma values.

---

## Phase 12 — Speed Distillation Losses

We start simple:

$$
L = L_{\text{velocity}}
$$

Then we optionally add perceptual supervision.

We decode the teacher and student latents:

```text
x_teacher → VAE → image_teacher
x_student → VAE → image_student
```

We use a visual encoder:

$$
L_{\text{perceptual}} = \| \phi(I_s) - \phi(I_t) \|^2
$$

Possible encoders:

- DINO
- SigLIP
- CLIP image encoder
- VGG-like perceptual features

Then:

$$
L = L_{\text{velocity}} + \lambda_p L_{\text{perceptual}}
$$

Later, we add prompt alignment:

$$
L_{\text{prompt}} = 1 - \cos(E_{\text{image}}(I_s), E_{\text{text}}(\text{prompt}))
$$

Final possible objective:

$$
L = L_{\text{velocity}} + \lambda_p L_{\text{perceptual}} + \lambda_t L_{\text{prompt}}
$$

We only add adversarial training if the accelerated model becomes visibly soft or loses fine texture.

---

## Phase 13 — Optional Adversarial Refinement

We introduce a discriminator only if necessary.

The discriminator learns:

```text
Teacher outputs → real
Student outputs → fake
```

The student learns to fool the discriminator.

This may recover:

- sharpness
- textures
- micro-detail
- visual realism

However, this adds considerable training complexity and instability, so it should not be part of the initial implementation.

---

## Phase 14 — Different Sampling for Different Training Stages

We use the same master prompt corpus, but with different sampling distributions.

Architecture healing:

```text
70% general
30% difficult
```

Speed distillation:

```text
45% general
55% difficult
```

During speed distillation, we oversample:

- complex scenes
- text rendering
- multiple constraints
- spatial relationships
- fine detail
- multi-object composition

These are likely to degrade first when reducing denoising steps.

---

## Phase 15 — Dataset Size Strategy

The master pool can contain 100k prompts without every experiment consuming all 100k.

Suggested development stages:

```text
Smoke test:
10k prompts
2k–5k optimizer steps

Architecture experiment:
20k–50k prompt pool
10k–20k steps

Serious healing:
100k prompt pool
20k–40k sampled steps as needed
```

The important variable is optimizer steps and coverage, not number of epochs over the complete dataset.

---

## Phase 16 — Cache Expensive Teacher Computation

Teacher inference will dominate training cost.

We cache when practical:

- prompt embeddings
- text encoder outputs
- teacher hidden states
- teacher velocity targets
- selected trajectory endpoints
- metadata for sampled timesteps

For speed distillation, precomputing teacher trajectories for selected prompt/seed/timestep combinations may greatly reduce GPU cost.

---

## Phase 17 — Make Bridges Timestep-Aware

A fixed bridge:

$$
\text{Bridge}(h)
$$

may not approximate removed blocks equally well at every denoising stage.

A stronger version is timestep-conditioned:

$$
h' = h + g(t)P(h)
$$

or:

$$
h' = h + \text{Bridge}(h,e_t)
$$

where:

$$
e_t
$$

is the timestep embedding.

This allows the cheap replacement module to behave differently during:

```text
high-noise stages
mid denoising
low-noise detail refinement
```

We start with a standard bridge first and introduce timestep conditioning only if evaluation shows timestep-dependent degradation.

---

## Phase 18 — Progressive Architecture Reduction

We do not apply the final pruning ratio immediately.

Recommended sequence:

```text
32 teacher blocks
-> 31 blocks + pretrained bridge (selected)
-> 30 blocks + region bridge (candidate A, provisional)
-> independent conditional scan, learned replacement and quality/latency gates
-> rerank after each accepted removal
```

We stop when the quality/latency tradeoff becomes unattractive.

The objective is not maximum pruning.

The objective is the best Pareto point between:

$$
\text{quality} \quad \text{vs} \quad \text{latency}
$$

---

## Phase 19 — Progressive Speed Reduction

Likewise:

```text
40 steps
→ 16
→ 8
→ 4
```

We benchmark after every stage.

We do not assume 4 steps is automatically better than 8.

If:

```text
8 steps = 5× faster with negligible quality loss
4 steps = 8× faster with major degradation
```

then 8 steps may be the better production model.

---

## Phase 20 — Quantization

Only after architecture and timestep distillation are stable, we apply quantization.

Candidates:

```text
BF16
→ FP8
→ INT8
→ potentially INT4
```

We benchmark every version separately.

Quantization affects:

- VRAM
- bandwidth
- latency
- output quality

We do not mix quantization debugging with pruning debugging.

---

## Phase 21 — Final Evaluation Matrix

The final comparison should look like:

| Variant | Layers | Steps | Precision | Quality | Alignment | Text | Latency | VRAM |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Original Qwen | 32 | 40 | BF16 | baseline | baseline | baseline | baseline | baseline |
| Pruned | 31 | 40 | BF16 | | | | | |
| Candidate A (provisional) | 30 | 40 | BF16 | | | | | |
| Distilled (planned) | TBD | 8 | BF16 | | | | | |
| Distilled (planned) | TBD | 4 | BF16 | | | | | |
| Distilled + FP8 | TBD | 8 | FP8 | | | | | |
| Distilled + INT8 | TBD | 8 | INT8 | | | | | |

We also report:

$$
\text{Speedup} = \frac{\text{Latency}_{\text{original}}}{\text{Latency}_{\text{student}}}
$$

and:

$$
\Delta \text{Quality} = \text{Score}_{\text{student}} - \text{Score}_{\text{original}}
$$

The final target should be defined as a Pareto objective, for example:

> Maximum inference speed improvement while keeping Qwen-Image-Bench degradation below an acceptable threshold.

---

# Recommended Experimental Order

We tick each step as we complete it (`- [ ]` → `- [x]`).

- [x] 1. Build the 20k pilot dataset (Phase 1).
- [x] 2. Establish the teacher baseline (Phase 2); extended performance benchmarking remains open.
- [x] 3. Diagnose the 32-block architecture and select first removal candidates (Phase 3).
- [x] 4. Insert and verify residual bottleneck bridges (Phase 4).
- [x] 5. Pretrain bridges; select block 5, rank 256 (Phase 5, 5.518% validation velocity error).
- [x] 6. Close Phase 6: healing not promoted. Published full-weight continuation and blinded 129-prompt adherence QA retain pretrained; gain +2.3 points, 95% CI [0.0, 5.4], below +5 promotion bar. Original later records await recovery.
- [x] 7. Second removal (Phase 7 notebook): candidate A (blocks 4-5) rejected; block 13 rejected on spelling; block 20 chosen by a text-aware scan; attention bridges for blocks 5 and 20.
- [x] 8. Heal the 30-block student (Phase 8 notebook): velocity better, spelling not; v1 ships unhealed.
- [x] 9. Few-step baselines and speed/memory benchmark (Phase 9 notebook); BF16 bridges.
- [x] 10. Distill 40 → 8 steps with a Lightning LoRA (Phase 10 notebook): OCR-weighted letters, v1's own trajectories, gentler start → **10 steps**, fused.
- [x] 11. Benchmark the release: 1.75 s vs 6.89 s per image, 15/16 spelling, 2/25 structure differences.
- [x] 12. Release Nihonga v1.0 (Hugging Face, niwakiai.com/nihonga, research log on josejuanmartinez.github.io).
- [ ] 13. Next steps: see the TODO below.

# Final Architecture Concept

```text
Original Qwen Image
32 blocks
40–50 denoising steps
BF16
        │
        │ structured pruning
        ▼
Pruned Qwen
accepted 31- or 30-block architecture
+ cheap bridge modules
        │
        │ architecture healing
        ▼
Recovered Small Qwen
accepted 31- or 30-block architecture
40 steps
        │
        │ trajectory / speed distillation
        ▼
Fast Small Qwen
accepted 31- or 30-block architecture
8 or 4 steps
        │
        │ quantization
        ▼
Production Model
smaller
faster
lower VRAM
minimal quality loss
```

The core principle is:

**We do not simply remove computation. We replace expensive computation with cheap approximations, then we distill the original model's behavior back into the smaller architecture, and only then do we compress the denoising trajectory.**

## As built: Nihonga v1.0

```text
Qwen-Image-2.1 (32 blocks, 40 steps, BF16)
        │ text-aware block selection; blocks 5 and 20 replaced
        ▼
30 blocks + 2 attention bridges (~25 M weights each, BF16)
        │ Lightning LoRA (rank 32), trajectory distillation on the student's own 40-step paths,
        │ letter tokens ×10 (OCR); fused into the weights
        ▼
Nihonga v1.0: 6.73 B parameters, 10 steps (3 small + 7 large), no CFG, 1.75 s per 1024² image on an H100
```

---

# TODO: next steps

Each item keeps the rule: a quantitative and a qualitative test, every checkpoint saved, Modal jobs watched and
nothing left running.

**Housekeeping**
- [ ] Delete the duplicate Hugging Face repo `jjmcarrascosa/nihonga-v1` (keep `jjmcarrascosa/Nihonga-v1.0`) — needs explicit OK.
- [ ] Run `example.py` from a clean environment against the Hugging Face repo (end-to-end install test).
- [ ] Publish the LinkedIn post (`release/social/linkedin-nihonga.jpg`).

**Phase 11: quantization**
- [ ] Quantize the fused v1 transformer (FP8 first, then 4-bit), and the text encoder, which holds 17.5 GB and dominates memory.
- [ ] Compare quantized against unquantized v1: speed, peak memory, spelling (16 items) and the 25 general prompts.

**Quality of v1**
- [ ] Fix the 2 remaining structure differences (missing laptop, five friends): on-policy training (from the student's own states) for the start, or a longer small-step start.
- [ ] Larger blinded evaluation (e.g. the 159-prompt protocol) of v1 against the teacher, plus Qwen-Image-Bench.
- [ ] More text data (2k+ text trajectories), and stylized lettering that OCR missed (sand, 3D, large characters).

**Faster**
- [ ] 4 steps: progressive distillation from the 8/10-step LoRA, then distribution matching or an adversarial loss if still soft (our trajectory-only 4-step LoRA stayed grey; LightX2V is crisp but misspells).
- [ ] Retrain the LoRA on the uneven schedule itself, aiming at 8 steps with the gentle start built in.

**Smaller (version 2)**
- [ ] Removals three and four: a text-aware scan inside v1, non-adjacent blocks, bridges as for block 20, then re-distill.
- [ ] Revisit healing once four blocks are removed (more damage to repair).

**Beyond text to image**
- [ ] Test image editing, reference images and RGBA output, which v1 did not train or test.
- [ ] A ComfyUI node or workflow using `sigmas.json`; optionally a diffusers custom pipeline.
- [ ] Write the paper ("Lost Letters: how to prune a diffusion transformer without breaking what fails first").

