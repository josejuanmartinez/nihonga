# Qwen Image Compression and Acceleration Plan

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
Original:
60 Transformer blocks

First target:
50 blocks

Second target:
40 blocks
```

We do not immediately jump from 60 to 30.

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
60-layer frozen model
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
Teacher 0   → Student 0
Teacher 10  → Student 7
Teacher 20  → Student 13
Teacher 30  → Student 20
Teacher 40  → Student 26
Teacher 50  → Student 33
Teacher 59  → Student 39
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
Original 60L
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
40 layers
40 steps

Target:
40 layers
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
60 layers
→ 50
→ heal + benchmark

50
→ 40
→ heal + benchmark

40
→ possibly 32–36
→ heal + benchmark
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
| Original Qwen | 60 | 40 | BF16 | baseline | baseline | baseline | baseline | baseline |
| Pruned | 40 | 40 | BF16 | | | | | |
| Pruned + healed | 40 | 40 | BF16 | | | | | |
| Distilled | 40 | 8 | BF16 | | | | | |
| Distilled | 40 | 4 | BF16 | | | | | |
| Distilled + FP8 | 40 | 8 | FP8 | | | | | |
| Distilled + INT8 | 40 | 8 | INT8 | | | | | |

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

- [ ] 1. We build a 20k prompt dataset ([Phase 1](#phase-1--build-the-prompt-dataset))
- [ ] 2. Then we benchmark the original Qwen ([Phase 2](#phase-2--establish-the-original-baseline))
- [ ] 3. We prune 60 → 50 layers ([Phase 3](#phase-3--identify-layers-to-remove))
- [ ] 4. We insert residual bottleneck bridges ([Phase 4](#phase-4--insert-bridge-modules))
- [ ] 5. We train the bridges only ([Phase 5](#phase-5--pretrain-the-bridges))
- [ ] 6. We run full architecture healing ([Phase 6](#phase-6--architecture-healing), [Phase 7](#phase-7--healing-dataset-strategy))
- [ ] 7. We benchmark ([Phase 8](#phase-8--evaluate-the-pruned-model))
- [ ] 8. We expand the dataset toward 100k ([Phase 15](#phase-15--dataset-size-strategy))
- [ ] 9. We try 50 → 40 layers ([Phase 18](#phase-18--progressive-architecture-reduction))
- [ ] 10. We heal again
- [ ] 11. We benchmark
- [ ] 12. We distill 40 steps → 8 steps ([Phases 9–12](#phase-9--speed-distillation))
- [ ] 13. We benchmark
- [ ] 14. We distill 8 → 4 if worthwhile ([Phase 19](#phase-19--progressive-speed-reduction))
- [ ] 15. We benchmark
- [ ] 16. We quantize ([Phase 20](#phase-20--quantization))
- [ ] 17. We run the final quality / latency / VRAM evaluation ([Phase 21](#phase-21--final-evaluation-matrix))

# Final Architecture Concept

```text
Original Qwen Image
60 layers
40–50 denoising steps
BF16
        │
        │ structured pruning
        ▼
Pruned Qwen
40–50 layers
+ cheap bridge modules
        │
        │ architecture healing
        ▼
Recovered Small Qwen
40–50 layers
40 steps
        │
        │ trajectory / speed distillation
        ▼
Fast Small Qwen
40–50 layers
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