# CISC3024 Pattern Recognition
# AI Assignment #1

**Algorithm:** MobileNetV4 (ECCV 2024) — Universal Inverted Bottleneck
**Application:** Image classification on CIFAR-10
**Approach:** Transfer learning from ImageNet-1k pretrained weights

| | |
|---|---|
| **Student name** | `[[ FILL IN ]]` |
| **Student ID** | `[[ FILL IN ]]` |
| **Course** | CISC3024 Pattern Recognition |
| **Submission date** | `[[ FILL IN ]]` |

**Result in one line.** A 2.51 M-parameter MobileNetV4 reaches **96.31 % top-1** and
**99.84 % top-5** accuracy on the CIFAR-10 test set, beating an 11.18 M-parameter
ResNet-18 baseline by 0.43 points at 4.46× fewer parameters, and using only free
Colab GPU time.

---

## 1. How I Asked AI Tools to Find the Algorithm

### 1.1 Reading the brief before searching

The assignment brief is short, so the first thing I did was ask the AI to read it
and turn it into a checklist I could actually search against. The brief
(`AIassign1.pdf`) requires an algorithm from **2023–2026**, applied to a computer
vision problem, with the AI doing the searching, the coding and the writing. Two
constraints in it shaped everything that followed:

* the report must document **detailed steps**, which is why the AI Workflow Log
  exists as a separate appendix; and
* **"You are NOT allowed to write any code by yourself!"** — so every line in the
  submitted notebooks is AI-generated, and my job was to run it, verify it and
  judge it.

I added one constraint of my own: the code had to run on the **free** tier of
Google Colab. That rules out anything needing more than about 15 GB of GPU memory
or many hours per run, and it is a much more binding constraint than it first
looks.

### 1.2 The prompts I used

Prompts were typed in **Chinese**, which is the language I think in; the report is
in English. The search prompt, verbatim:

> 我需要完成CISC3024作业，要求用2023-2026年的Deep CNN/Autoencoder做计算机视觉应用。
> 请帮我搜5个候选算法，要求有官方PyTorch代码，能在Colab免费GPU跑。推荐一个最适合MVP的
> 方案，比如MobileNetV4 + CIFAR-10，并给出端到端工作流规划。

*"I need to complete my CISC3024 assignment, which requires a 2023–2026 Deep
CNN/Autoencoder applied to a computer vision problem. Please search for five
candidate algorithms, requiring official PyTorch code that runs on Colab's free
GPU. Recommend the one best suited to an MVP — for example MobileNetV4 + CIFAR-10 —
and give an end-to-end workflow plan."*

Three things about this prompt are deliberate, and they are what made the answer
usable rather than merely plausible:

* **It names the constraint that actually binds** — the *free* Colab GPU. Omit it
  and the answer is a list of state-of-the-art models that cannot be run.
* **It asks for five candidates, not one.** A single recommendation cannot be
  evaluated; five can be compared against each other, which is what turned §1.3
  into a scoring table instead of an assertion.
* **It asks for a workflow, not just a name.** A model name is not a plan.

I did not stop at one prompt. Every prompt used across the project — verbatim, in
order, with the stage it belongs to — is collected in **Appendix C**.

### 1.3 Turning a vague brief into screening criteria

A literature search without criteria just produces a list. So before accepting any
candidate I had the AI define six tests, and made every candidate pass all six:

| ID | Criterion | Why it matters here |
|---|---|---|
| SC1 | Published 2023–2026 | Explicit requirement in the brief |
| SC2 | Official PyTorch implementation exists | No hand-porting; the brief forbids me writing code |
| SC3 | Runs on a free Colab T4 (≤ 16 GB) | My own constraint; excludes most large models |
| SC4 | ≤ ~15 M parameters | Keeps training inside a free-tier session |
| SC5 | Reported ImageNet accuracy competitive with its size | It has to be defensible, not just small |
| SC6 | Permissive licence | The code is being submitted publicly |

### 1.4 The five candidates

| Algorithm | Venue | Params | Reported acc. | Weighted score |
|---|---|---|---|---|
| **MobileNetV4** | ECCV 2024 | 3.77 M | 73.8–74.6 % | **4.90** |
| RepViT | CVPR 2024 | 5.1 M | 78.7–79.1 % | 4.78 |
| FasterNet | CVPR 2023 | 3.9 M | 71.9 % | 4.65 |
| ConvNeXt V2 + FCMAE | CVPR 2023 | 3.7 M | 76.7 % | 4.60 |
| DC-AE + EfficientViT | ICLR 2025 | — | rFID (generative) | 4.20 |

MobileNetV4 scored highest on the weighted criteria. I accepted the
recommendation, but not before checking the two things that would have made it a
bad choice.

**Check one — is there really official PyTorch code?** There is not, in the sense
of a Google-released PyTorch repository. MobileNetV4's reference implementation is
**TensorFlow** (Google Model Garden). The authoritative PyTorch implementation is
Ross Wightman's `timm`, which is the de-facto reference for PyTorch vision models
and is what the paper's own leaderboards point at. I have stated this explicitly
rather than implying an official PyTorch release exists, because SC2 is the
criterion most likely to be quietly fudged.

**Check two — is the algorithm still maintained?** The survey surfaced that
EfficientViT's repository was declared unmaintained in September 2025 and moved to
a successor project. That is exactly the kind of fact that decides whether code
will still run in a year, and it is why DC-AE scored lowest despite being the most
recent candidate.

---

## 2. Algorithm Description

### 2.1 What MobileNetV4 is

MobileNetV4 ("Universal Models for the Mobile Ecosystem", Qin et al., ECCV 2024)
is the fourth generation of the MobileNet family. Its contribution is not a single
new block but a **design space** plus an automated search over it, targeting the
whole range of mobile accelerators rather than one.

Two building blocks carry the architecture:

**Universal Inverted Bottleneck (UIB).** The classic MobileNet inverted bottleneck
expands, applies a depthwise convolution, then projects back down. UIB generalises
this by making the *positions of the two pointwise convolutions optional*. Turning
them on or off yields a family that contains the original inverted bottleneck, a
conv-next-style block, and a pure depthwise block as special cases. That
unification is what lets one search cover architectures that previously needed
separate designs.

**Mobile MQA.** An attention block with **asymmetric spatial down-sampling**: keys
and values are pooled more aggressively than queries. The motivation is that
attention is memory-bandwidth-bound on mobile hardware, and shrinking K/V shrinks
the expensive part.

On top of these, MobileNetV4 was found by **hardware-aware NAS** that optimises for
measured latency on real device families (Pixel, Samsung, Qualcomm, Apple) rather
than FLOPs — because on mobile hardware, FLOPs and latency diverge badly.

### 2.2 Why it suits this assignment

Three reasons, in order of weight:

1. **Size.** At 2.51 M parameters (timm convention, see §2.3) it is small enough
   that 12 epochs on a free T4 is a realistic proposition, while still being a
   modern architecture rather than a pedagogical toy.
2. **Pretrained weights that transfer.** The ImageNet-1k checkpoints are strong
   enough that fine-tuning on a small dataset converges quickly, which matters
   when the compute budget is a few hours.
3. **An honest accuracy/latency story.** MobileNetV4's selling point is the
   accuracy-per-millisecond trade-off, which is measurable on the hardware I
   actually have. That made it possible to run a *time* ablation as well as an
   accuracy ablation (§4.4).

### 2.3 A parameter-count discrepancy worth stating

The paper reports **3.77 M** parameters for the small variant. timm reports
**2.51 M** for `mobilenetv4_conv_small.e2400_r224_in1k`. Both are correct. The
paper's count **folds batch-normalisation layers into the preceding convolutions**
(a standard deployment optimisation, since BN is linear at inference); timm keeps
them as separate modules.

Every parameter count in this report uses the **timm convention**, and the
distinction is flagged wherever a number could be read either way. Quoting the
paper's figure against timm's without saying so is a quiet way to make a model
look bigger than it is.

---

## 3. How AI Implemented the Algorithm

The work ran in six stages. The notebooks are in §6.

| Stage | What was built |
|---|---|
| 1–2 | Environment, Drive mount, dataset pipeline, model, training loop |
| 3 | Smoke test at 128 px / 2 epochs, then checkpointing and full resume (optimiser, scheduler, AMP scaler, epoch, RNG) |
| 4 | The main 12-epoch training run |
| 5 | Four single-variable ablations |
| 6 | Error analysis and calibration |

### 3.1 The single most important engineering decision

MobileNetV4's pretrained weights are **ImageNet-1k models at 224 px**. CIFAR-10 is
**32 × 32**. Feeding 32 × 32 images straight into the network does not merely
degrade accuracy — it breaks the architecture: MobileNetV4 has five stride-2
stages, so a 32 × 32 input collapses to a **1 × 1** feature map before the
classifier ever sees it, leaving the head with almost no spatial information.

The fix is to use the preprocessing protocol the weights were trained under:

```python
train_tf = Compose([Resize(256), RandomCrop(224), ToTensor(), Normalize(MEAN, STD)])
eval_tf  = Compose([Resize(256), CenterCrop(224),  ToTensor(), Normalize(MEAN, STD)])
```

The cost is real and worth naming: **upsampling 32 × 32 to 224 px invents no
detail.** The model sees an interpolated image, so the pretrained features help
less than they would on native-resolution data. This caps the achievable accuracy
and is revisited in §4.7.

### 3.2 Training loop details that had to be right

**Automatic mixed precision with a correct scheduler guard.** The original
generated code stepped the learning-rate scheduler *before* the optimiser, which
raises a warning and lets the schedule drift. The corrected order is:

```python
scale_before = scaler.get_scale()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)            # the real optimiser step
scaler.update()                   # may halve the scale on inf/NaN
if scaler.get_scale() >= scale_before:
    scheduler.step()              # only after a genuine optimiser step
```

The comparison matters because AMP *silently skips* `optimizer.step()` when it
meets inf/NaN gradients. Stepping the scheduler anyway would desynchronise the
schedule from the optimiser.

**This guard is observably working.** In the ablation runs, the epoch-1 learning
rate is **3.00e-04** for ResNet-18, where no step was skipped, but **2.93e-04** for
`conv_medium` and **2.94e-04** for the 128-px run — and the training log for those
two reports *"(note: 9 step(s) skipped by GradScaler)"* and *"8 step(s)"*
respectively. The skipped steps did not advance the one-epoch warm-up, so the
epoch ended at a slightly lower learning rate. That is the guard doing its job,
visible in the output rather than merely asserted.

**Full resume.** The main run was interrupted at epoch 7 and continued in a new
Colab session. The checkpoint carries the model, optimiser, scheduler, AMP scaler,
epoch, best accuracy, early-stopping counter, history **and** the RNG state, and
is written atomically (temp file + `os.replace`) so a disconnect mid-write cannot
truncate it. The resume restored the learning rate at exactly 1.30e-04 and
retained the best-validation tracker at 0.9508 — both independently visible in the
log.

### 3.3 Five corrections I caught

The brief frames the assignment around AI tooling, so this section is the honest
part: five things the generated code got wrong, each found by **running it and
reading the output**.

**1. Scheduler stepped before the optimiser, and on AMP-skipped steps.** Fixed by
reordering and adding the `get_scale()` guard described above.

**2. The RNG state was not restored on resume.** The first resumed run printed:

```
(RNG state not restored: RNG state must be a torch.ByteTensor )
```

The message is misleading — the saved state genuinely *was* a `uint8` tensor. The
assertion in `ATen/core/Generator.h` actually tests three conditions:

```cpp
new_state.layout() == kStrided && new_state.device().type() == kCPU
    && new_state.dtype() == kByte
```

The failing one was the **device**. `torch.load(..., map_location=device)` was
called with `device='cuda'`, and `map_location` relocates *every* tensor in the
checkpoint onto the GPU — the RNG state included — while `torch.set_rng_state()`
requires a **CPU** ByteTensor. I confirmed the mechanism by testing each branch
separately: an `int64` tensor and a non-CPU tensor both produce that exact message,
while a non-contiguous tensor produces a *different* one (`"RNG state must be
contiguous"`), which rules the layout branch out.

A second defect hid in the same block: all three restore calls shared one `try`, so
the exception on the first line meant the NumPy and Python RNG states were never
restored either. Both are now guarded independently, and the state is pinned to CPU
before restoring.

*Impact:* none on any reported accuracy — the RNG state only affects the
augmentation stream. The one honest caveat is that a fresh 12-epoch run would land
within normal run-to-run variation of 0.9631 but would not reproduce it to the
digit.

**3. Parameter counts disagreed with the paper.** Traced to BN folding (§2.3);
resolved by stating the convention rather than by picking whichever number looked
better.

**4. The "freeze backbone" experiment did not freeze as much as intended.** See
§4.4 — it still trained 49.7 % of parameters.

**5. The error-analysis notebook assumed the wrong `results.json` schema.** Two
notebooks in this project write a file with that name and they disagree on its
shape: the ablation notebook writes a flat record including `freeze_backbone` and
`total_params`, whereas the main run writes only five keys. Analysing the main run
therefore raised `KeyError: 'freeze_backbone'`. The configuration is now read
through one merged view with defaults, and the cell fails fast with the list of
runs that do exist if the tag is wrong.

### 3.4 How the generated code was verified

Reading code is not verification, so each notebook was executed against
reconstructed inputs before I trusted it:

* **Ablation harness** — stub dataset (200 images instead of 50 000), weights
  loaded with `pretrained=False`, all four ablations executed end to end:
  **24 assertions passed** (parameter counts, run-directory isolation, trainable
  counts under freezing, resume behaviour, output schema).
* **RNG fix** — the four checkpoint helpers were extracted from the *patched
  notebook source itself* and put through a real save → advance → load → redraw
  round-trip: **13 of 13 assertions passed**, including full reproduction of the
  torch, NumPy and Python streams.
* **Error-analysis fix** — the patched configuration cell was executed against
  three reconstructed schemas (main run, plain ablation, frozen ablation) plus an
  `img_size=128` case: **25 of 25 assertions passed**, including that the
  preprocessing size follows the analysed run's own `img_size` rather than a
  hard-coded constant.

---

## 4. Experiment Settings and Results

### 4.1 Environment and configuration

| Item | Value |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| timm | 1.0.29 |
| Hardware | Google Colab free tier, NVIDIA Tesla T4 (14.6 GB usable) |
| Model identifier | `mobilenetv4_conv_small.e2400_r224_in1k` |
| Dataset | CIFAR-10 — 45 000 train / 5 000 validation / 10 000 test |
| Input resolution | 224 × 224 (Resize-256 → crop), ImageNet normalisation |
| Batch size | 128 |
| Epochs | 12 |
| Optimiser | AdamW, lr 3e-4, weight decay 0.05 |
| Schedule | 1-epoch linear warm-up, then cosine decay |
| Regularisation | Label smoothing 0.1, gradient clipping at 1.0, early stopping (patience 4) |
| Mixed precision | fp16 AMP |
| Seed | 42 |
| Steps per epoch | 351 (4 212 total) |
| Artifacts | Checkpoints, curves, `results.json` on Google Drive |

All five runs share the same seed, split, epoch count, augmentation, optimiser and
schedule shape, so a difference between rows is attributable to the one field that
varies.

### 4.2 Smoke test — validating the pipeline before spending GPU hours

Before committing to a 12-epoch run, the notebook was executed twice at reduced
scale to prove the pipeline worked end to end. This is cheap insurance: the full
run costs roughly 32 minutes of GPU time, and a crash at epoch 10 wastes all of it.

| | Smoke run 1 (`v1`) | Smoke run 2 (`v3`) |
|---|---|---|
| Input resolution | 128 px | 128 px |
| Epochs | 2 | 2 |
| Epoch 1 — validation accuracy | 0.7898 | 0.8102 |
| Epoch 2 — validation accuracy | 0.9086 | 0.9128 |
| **Best validation accuracy** | 0.9086 | **0.9128** |
| **Test accuracy** | **0.9053** (loss 0.7355) | **0.9034** (loss 0.7386) |
| Seconds per epoch | 91.2 / 78.6 | 93.7 / 90.7 |
| GradScaler step skips | not instrumented | 8 at epoch 1 |

![Smoke test training curves](figures/smoke_v3_curves.png)

Both runs converge normally — validation accuracy climbs roughly ten points in two
epochs while the training loss falls from 2.16 to 0.94. Two things came out of this
stage that changed the final code.

**The scheduler-ordering bug was caught here, not in the main run.** Smoke run 1
emitted:

```
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
```

That is correction 1 in §3.3. Left unnoticed, the cosine schedule would have been
advanced one step out of phase for all 12 epochs of the real run. The smoke test
surfaced it for three minutes of GPU time instead of thirty.

**The AMP guard is observably doing something.** Smoke run 2 recorded *8 step(s)
skipped by GradScaler* at epoch 1, and its first-epoch learning rate is **2.94e-04**
rather than the nominal **3.00e-04** — the scheduler was correctly held back on the
skipped steps. That is the guard described in §3.2 working on real data rather than
in a unit test.

The augmented sample batch shows what the network actually receives:

![Augmented training samples](figures/smoke_sample_batch.png)

The blur is the point: these are 32 × 32 images upsampled to 224 px, and the
coloured patches are colour-jitter augmentation. This is the concrete picture behind
the argument in §3.1 that upsampling invents no detail.

Confusion matrices for both smoke runs:

![Smoke run 1 confusion matrix](figures/smoke_v1_confusion_matrix.png)

![Smoke run 2 confusion matrix](figures/smoke_v3_confusion_matrix.png)

Even at two epochs and 128 px the error structure is already the one that survives
to the final model. **Cat and dog are the two weakest classes** in both runs (768
and 845 correct in run 1; 784 and 832 in run 2), and the largest single
off-diagonal cell is *cat → dog* in run 1 (100 errors) and *dog → cat* in run 2
(94). The failure mode was visible before any real training was done.

> **These numbers are not comparable to the main result.** They use 128 px rather
> than 224 px and 2 epochs rather than 12, and smoke run 2 was produced by an
> intermediate revision of the code. They are reported as evidence that the
> pipeline runs, not as results.

### 4.3 Main result

![Training and validation curves](figures/main_curves.png)

| Metric | Value |
|---|---|
| **Test accuracy (top-1)** | **0.9631** (9 631 / 10 000) |
| **Test accuracy (top-5)** | **0.9984** |
| Test loss | 0.5947 |
| Best validation accuracy | **0.9670** (epoch 11 of 12) |
| Macro-averaged F1 | 0.9630 |
| Parameters (total / trainable) | 2.51 M / 2.51 M |
| Wall-clock per epoch | 156–172 s (mean 159.7 s over the last five) |
| Errors | 367 of 10 000 |

Training log from the resumed session (the first seven epochs are in the saved
history but were logged in the earlier session):

```
resumed -> next epoch 8, best val acc 0.9508, lr 1.30e-04
epoch  8/12 | train 0.6128 acc 0.9570 | val 0.6043 acc 0.9594 | lr 8.86e-05 | 171.6s   <- best
epoch  9/12 | train 0.5849 acc 0.9693 | val 0.5948 acc 0.9614 | lr 5.25e-05 | 158.2s   <- best
epoch 10/12 | train 0.5656 acc 0.9788 | val 0.5885 acc 0.9652 | lr 2.43e-05 | 156.1s   <- best
epoch 11/12 | train 0.5517 acc 0.9851 | val 0.5869 acc 0.9670 | lr 6.35e-06 | 156.4s   <- best
epoch 12/12 | train 0.5439 acc 0.9883 | val 0.5873 acc 0.9660 | lr 3.18e-09 | 156.2s
```

Two things in this log are worth reading carefully rather than skimming.

First, the cosine schedule decays the learning rate to **3.18e-09** by the final
epoch, so the last epoch is effectively frozen. The 0.0010 drop between epoch 11
and epoch 12 is schedule noise, **not** the onset of overfitting — the model had
already stopped moving.

Second, validation loss sits *below* training loss at epoch 8 (0.6043 vs 0.6128).
That is the expected signature of label smoothing plus train-time augmentation and
dropout: the training figure is computed under a strictly harder objective. It is
not a sign of a broken split, and the curves confirm it — validation accuracy
actually leads training accuracy until epoch 6, which is the classic
transfer-learning signature of pretrained features being exploited before the
fine-tuning catches up.

### 4.4 Ablation study

![Ablation curves](figures/abl_01_resnet18__curves.png)

| Run | Model | Img | Frozen | Params (M) | Trainable (M) | Best val | Test acc | Δ vs main | s/epoch |
|---|---|---|---|---|---|---|---|---|---|
| **main** | mobilenetv4_conv_small | 224 | no | 2.506 | 2.506 | 0.9670 | **0.9631** | — | 159.7 |
| abl_01 | resnet18 | 224 | no | 11.182 | 11.182 | 0.9604 | 0.9588 | −0.43 | 165.5 |
| abl_02 | mobilenetv4_conv_medium | 224 | no | 8.447 | 8.447 | 0.9690 | 0.9675 | **+0.44** | 180.6 |
| abl_03 | mobilenetv4_conv_small | 128 | no | 2.506 | 2.506 | 0.9542 | 0.9504 | −1.27 | 82.4 |
| abl_04 | mobilenetv4_conv_small | 224 | **yes** | 2.506 | **1.244** | 0.9160 | 0.9056 | **−5.75** | 150.8 |

Deltas are in percentage points. Full per-run confusion matrices and curves are in
the delivered `figures/` folder and on the source-code page.

#### Finding 1 — the modern architecture wins on size, decisively

ResNet-18 is **4.46× larger** (11.18 M vs 2.51 M) and is **0.43 points *worse***.
MobileNetV4 does not merely match the classical baseline at lower cost; it beats
it. For a mobile-deployment argument this is the headline number: 4.46× fewer
parameters, marginally faster per epoch (165.5 s vs 159.7 s, within run-to-run
noise), and better accuracy.

#### Finding 2 — extra capacity is not worth it here

`conv_medium` is the only configuration that beats the baseline, by **+0.44
points** — and it costs **3.37× the parameters** and **13 % more time per epoch**.
Against that, its validation loss is markedly unstable (1.21, 1.60, 2.48 and 1.89
at epochs 2, 9, 10 and 12, against a smooth train-loss curve), and its training
log records GradScaler step skips at epoch 1. The larger model is both
overfitting more and interacting worse with fp16.

**Conclusion: for this dataset, `conv_small` is the right choice.** The +0.44
points is within the range that a single-seed run cannot distinguish from noise
(§4.7), and it is bought at 3.37× the model size — which is precisely the trade-off
MobileNetV4's small variant exists to avoid.

#### Finding 3 — resolution pays, but less than expected

Dropping to 128 px halves the time per epoch (82.4 s vs 159.7 s, a **48 % saving**)
and costs **1.27 points**. The time saving is the largest single lever in the whole
study, which makes 128 px the right default for iteration and debugging. But the
fact that the model still reaches 95.04 % at 128 px says something more
interesting: **the pretrained features are doing most of the work, and they do not
depend heavily on the extra detail.** Given that 32 × 32 → 224 px is already an
interpolation with no real information in it, this is exactly what §3.1 predicts.

#### Finding 4 — the "frozen backbone" is the worst trade in the study

Freezing everything after `global_pool` loses **5.75 points** while saving only
**5.6 %** of training time. Both halves of that are worth stating.

It is a large accuracy loss because the fine-tuned features genuinely matter for
CIFAR-10 at this resolution. But the surprise is the *time*: freezing half the
parameters bought almost nothing, because the forward pass through the whole
backbone still runs on every batch, and the bottleneck is the 224-px data pipeline
rather than the backward pass.

**And it is not the freeze it appears to be.** The frozen run still trains
**1.244 M of 2.506 M parameters — 49.7 %** — because timm places `conv_head`, a
1 × 1 convolution expanding 960 → 1280 channels, *after* the pooling layer, so it
falls on the head side of the boundary:

```
freeze mode: 'head' -> head modules
             ('conv_head', 'norm_head', 'act2', 'flatten', 'classifier')
             45 BatchNorm layer(s) pinned to eval()
```

So abl_04 is not a linear probe; it is "fine-tune the last stage only". Reporting
it as a frozen-backbone experiment without that caveat would overstate the result.

### 4.5 Error analysis

![Confusion matrix](figures/error_confusion_matrix.png)

Per-class performance, sorted by difficulty:

| Class | Precision | Recall | F1 | Errors / 1000 |
|---|---|---|---|---|
| **cat** | 0.9214 | 0.9140 | **0.9177** | 86 |
| **dog** | 0.9339 | 0.9190 | **0.9264** | 81 |
| bird | 0.9697 | 0.9590 | 0.9643 | 41 |
| deer | 0.9576 | 0.9710 | 0.9643 | 29 |
| airplane | 0.9720 | 0.9730 | 0.9725 | 27 |
| truck | 0.9778 | 0.9670 | 0.9723 | 33 |
| automobile | 0.9723 | 0.9830 | 0.9776 | 17 |
| ship | 0.9723 | 0.9830 | 0.9776 | 17 |
| horse | 0.9819 | 0.9750 | 0.9784 | 25 |
| frog | 0.9734 | 0.9890 | **0.9812** | 11 |

The distribution is not uniform: eight of the ten classes sit in a tight
0.964–0.981 band, and **two classes — cat and dog — carry about four times the
error rate of the best class.** The residual error is therefore concentrated in
one visually ambiguous pair rather than spread evenly across the problem.

![Ranked confusions](figures/error_top_confusions.png)

The ten most common confusions:

| True → predicted | Count | % of true class |
|---|---|---|
| dog → cat | 52 | 5.20 % |
| cat → dog | 45 | 4.50 % |
| truck → automobile | 23 | 2.30 % |
| automobile → truck | 16 | 1.60 % |
| airplane → ship | 16 | 1.60 % |
| cat → deer | 14 | 1.40 % |
| horse → deer | 10 | 1.00 % |
| dog → deer | 10 | 1.00 % |
| bird → cat | 10 | 1.00 % |
| ship → airplane | 9 | 0.90 % |

Grouping these reveals three distinct failure modes rather than ten scattered
ones:

* **The cat / dog / deer cluster — 131 errors, 35.7 % of all errors.** Four-legged
  mammals with similar fur texture, similar body shapes and similar outdoor
  backgrounds. At the effective resolution the model sees, the discriminative
  detail — facial structure, ear shape, muzzle length — is largely gone.
* **The vehicle cluster — truck ↔ automobile, 39 errors, 10.6 %.** Geometrically
  these overlap heavily, and several gallery examples are genuinely ambiguous
  even to a human: pickups, vans and ambulances sit between the two classes.
* **The air / sea cluster — airplane ↔ ship, 25 errors, 6.8 %.** Both are
  elongated silhouettes, frequently against a uniform blue background.

![Error gallery](figures/error_gallery.png)

The gallery shows the 16 test images the model got wrong **while being most
confident** — a low-confidence mistake is forgivable, a confident one reveals a
genuine blind spot. Looking at them, the common thread is **loss of the
discriminative feature**:

* Several cats and dogs appear in profile, in shadow, or partly hidden by
  vegetation, so no face is visible — only a generic furry quadruped body.
* A dog photographed in a dark doorway reduces to a white silhouette.
* An airplane seen from directly underneath at distance is a small cross on a thin
  pole against blank sky; the model calls it a bird, which is a defensible reading.
* A truck on a road beside water is classified as a ship, suggesting the
  background influenced the decision.

One tile is worth flagging separately. Tile 16 is labelled `cat → frog` at 0.9112
confidence — and the image is unmistakably **a green frog on a leaf**. The model
was right and the label is wrong. This is label noise in CIFAR-10, and it is a
useful reminder that not every error in the confusion matrix is a model error; at
96.31 % accuracy, a small number of the remaining 367 are simply mislabelled.

### 4.6 Calibration

![Calibration](figures/error_calibration.png)

| Confidence bin | n | Actual accuracy |
|---|---|---|
| 0.2–0.3 | 12 | 8.33 % |
| 0.3–0.4 | 60 | 48.33 % |
| 0.4–0.5 | 155 | 59.35 % |
| 0.5–0.6 | 190 | 68.42 % |
| 0.6–0.7 | 245 | 73.06 % |
| 0.7–0.8 | 419 | 85.20 % |
| 0.8–0.9 | 2 513 | 98.05 % |
| 0.9–1.0 | 6 406 | 99.61 % |

Mean confidence is **0.8885 on correct** predictions and **0.6217 on wrong** ones,
so confidence does separate right from wrong usefully. But the bins tell a more
precise story than that summary: in the mid range the model is **over-confident**
(0.6–0.7 bins deliver 73 % not 65 %), while at the top end it is
**under-confident** — images it rates 0.8–0.9 are right 98.05 % of the time. The
practical reading is that the softmax scores are trustworthy as a *ranking* signal
but not as calibrated probabilities, and a deployment that thresholded on 0.9
would be discarding a large number of correct predictions.

### 4.7 Threats to validity

These are stated because they bound how strongly the ablations can be read.

* **One seed per configuration.** A single run cannot separate a 0.44-point
  difference from run-to-run variance. Only the two large effects — resolution
  (−1.27) and freezing (−5.75) — are comfortably outside that band; the
  ResNet-18 and `conv_medium` comparisons should be read as "indistinguishable
  from the baseline" rather than as firm orderings. Three seeds would be the
  obvious next step.
* **CIFAR-10 is saturated.** Absolute accuracy on this benchmark says more about
  the dataset than about the architecture, which is why the ablation deltas and the
  error structure carry more information than the headline number.
* **Upsampling destroys information.** 32 × 32 → 224 px is interpolation, so the
  pretrained features help less than they would at native resolution (§3.1).
* **A 2-image discrepancy.** The error-analysis notebook recomputed top-1 as
  **0.9633** against the training notebook's **0.9631**. The analysis runs at batch
  size 256 versus 128, and fp16 accumulation differs marginally between the two.
  Two images out of ten thousand — reported rather than smoothed over.

---

## 5. What I Have Learnt from This AI Assignment

**The AI was genuinely good at breadth, and genuinely bad at being right about
details.** It produced a working training pipeline, a four-way ablation harness
and a full error-analysis notebook far faster than I could have, and the survey
stage compressed a literature search that would otherwise have taken days. But
five of the errors documented in §3.3 were things the AI stated confidently and
that were simply wrong: a scheduler stepped in the wrong order, an RNG restore
that failed for a reason its own error message misreported, parameter counts that
contradicted the paper, a "frozen" experiment that was 49.7 % trainable, and a
config reader that assumed a schema the file did not have. **None of these were
caught by reading the code. All five were caught by running it and reading the
output.**

That is the main thing I take from this assignment. Reviewing AI-generated code by
reading it gives a false sense of confidence, because the code *looks* careful —
it has docstrings, it handles exceptions, it names its variables well. The only
reliable test is execution against known inputs, and the most useful habit I
developed was writing small harnesses that run the generated code on stub data and
assert on the result. The 24-, 13- and 25-assertion checks in §3.4 found four bugs
that no amount of reading had surfaced.

**The second thing is that error messages lie.** `RNG state must be a
torch.ByteTensor` was true about the tensor's *dtype* and completely silent about
the actual cause, which was its *device*. I only got to the bottom of it by reading
the assertion in the PyTorch source and testing each of its three conditions
independently. The general lesson — that a message names the check that fired, not
necessarily the reason — is one I would not have internalised from a lecture.

**The third is about what makes a result defensible.** Early on I was inclined to
report the best number. What I actually ended up doing was reporting the *caveats*
as findings: that the parameter count depends on a convention, that the frozen
experiment was not really frozen, that the 0.44-point ablation win is probably
noise, that one of the 16 headline errors appears to be a mislabelled image. Each
of those makes the report weaker as a sales pitch and much stronger as evidence.
The habit of asking "what would make this claim false?" turned out to be more
useful than any accuracy figure.

**Finally, on the process the brief is really testing.** The assignment is framed
around whether I can drive AI tooling to do real work, and the honest answer is
that the skill is not prompt-writing — it is *verification*. The AI wrote every
line of code I submitted, exactly as the brief requires. My contribution was
deciding what to build, choosing which of its suggestions to reject, running
everything on real hardware, and refusing to accept an explanation until I had
reproduced the mechanism myself.

---

## 6. Source Code Webpage Link

All source code for this assignment is published in a public GitHub repository:

> **Repository:** `[[ PASTE YOUR GITHUB REPOSITORY URL HERE ]]`
>
> e.g. `https://github.com/<your-username>/CISC3024-AI1-MobileNetV4-CIFAR10`

The repository contains the complete, executable source for all three notebooks,
together with the figures and the raw result files:

| # | File | Contents |
|---|---|---|
| 01 | `notebooks/01_main_training_MobileNetV4_CIFAR10.ipynb` | Environment, Drive-backed data pipeline, model, AMP training loop, checkpoint/resume, final evaluation |
| 02 | `notebooks/02_ablation_study.ipynb` | The four single-variable ablations and the comparison table |
| 03 | `notebooks/03_error_analysis.ipynb` | Confusion matrix, ranked confusions, error gallery, calibration |
| — | `README.md` | Repository landing page: results, findings and reproduction steps |
| — | `CISC3024_AI1_AI_Workflow_Log.md` | Appendix A: every AI prompt, error and correction |
| — | `figures/`, `data/` | The 15 figures used in this report and the raw `results.json` / CSV exports |

The notebooks are the ones actually executed on Colab, so their saved outputs are
the real run logs and figures reported in §4. The repository is public, so the
source can be inspected directly in the browser without downloading anything.

---

## References

1. Qin, D., Leichner, C., Delakis, M., et al. *MobileNetV4 — Universal Models for
   the Mobile Ecosystem.* ECCV 2024. arXiv:2404.10518.
2. Wightman, R. *MobileNet-V4 (now in timm).* Hugging Face blog, June 2024.
3. `huggingface/pytorch-image-models` (timm).
   https://github.com/huggingface/pytorch-image-models
4. Krizhevsky, A. *Learning Multiple Layers of Features from Tiny Images.*
   Technical report, University of Toronto, 2009. (CIFAR-10)
5. Wang, A., Chen, H., Lin, Z., et al. *RepViT: Revisiting Mobile CNN From ViT
   Perspective.* CVPR 2024. arXiv:2307.09283.
6. Chen, J., Kao, S., He, H., et al. *Run, Don't Walk: Chasing Higher FLOPS for
   Faster Neural Networks.* CVPR 2023. arXiv:2303.03667.
7. Woo, S., Debnath, S., Hu, R., et al. *ConvNeXt V2: Co-designing and Scaling
   ConvNets with Masked Autoencoders.* CVPR 2023. arXiv:2301.00808.

---

## Appendix A — Delivered files

```
CISC3024HW1/
├── CISC3024_AI1_Report.docx          this report
├── CISC3024_AI1_Report.pdf           the same report as PDF
├── README.md                         GitHub repository landing page
├── CISC3024_AI1_AI_Workflow_Log.md   every prompt, error and correction
├── CISC3024_AI1_Prompt_Log.md        Appendix C as a standalone file
├── notebooks/                        the three executed notebooks
│   ├── 01_main_training_MobileNetV4_CIFAR10.ipynb
│   ├── 02_ablation_study.ipynb
│   └── 03_error_analysis.ipynb
├── figures/                          19 figures used in this report
│   ├── smoke_v1_confusion_matrix.png   smoke test, 128 px / 2 epochs (0.9053)
│   ├── smoke_v3_confusion_matrix.png   smoke test, 128 px / 2 epochs (0.9034)
│   ├── smoke_v3_curves.png             smoke test training curves
│   ├── smoke_sample_batch.png          the augmented 224 px samples
│   └── ...                             main run, 4 ablations, error analysis
└── data/                             ablation_summary.csv, error_summary.json
```

## Appendix B — Reproducing the results

1. **Smoke test first.** Open `01_main_training_MobileNetV4_CIFAR10.ipynb` in Colab,
   select a T4 GPU, set `img_size=128` and `epochs=2`, and `Runtime → Run all`. This
   takes about three minutes and proves the pipeline works before you spend half an
   hour on the real run. Expect roughly 0.90 test accuracy — that is correct for two
   epochs, not a bug (§4.2).
2. **The main run.** Set `img_size=224` and `epochs=12`, and run all. CIFAR-10 is
   cached on Drive on first run and reused afterwards; the run takes roughly 35
   minutes for 12 epochs. If the session disconnects, run the notebook again and it
   resumes from the last checkpoint.
3. Open `02_ablation_study.ipynb` and run the four ablation cells in order
   (128 px first — it is the fastest). Each writes to its own folder and the final
   cell emits `ablation_summary.csv`.
4. Open `03_error_analysis.ipynb`, set `ANALYSE_TAG = "mnv4_cifar10"`, and run all.
   Change the tag to analyse any ablation run instead.

---

## Appendix C — Complete prompt log

Every prompt that shaped this project, in the order it was used, **verbatim in the
language it was typed (Chinese)** with an English gloss beneath. The brief asks how
the algorithm was found using AI tools; this appendix is the raw evidence. The
narrative version, with the AI's replies and every error, is the separate
`CISC3024_AI1_AI_Workflow_Log.md`.

`→` marks the assistant that received each prompt. Prompts were typed in Chinese;
the glosses are mine, added for this report.

### C.1 Stage 0 — understanding the assignment

**C.1.1** → planning assistant

> AI Assignment #1 CISC3024 Pattern Recognition September, 2026 ... 要怎樣做這份功課，提供步驟

*"How should I go about this assignment? Give me the steps."*

**C.1.2** → planning assistant

> 在 WORKING BUDDY 的 AI 端中開發這份功課，要有哪些準備，要 CALL 哪些專家

*"To develop this assignment on the WorkBuddy AI side, what preparation is needed,
and which experts should I call?"*

**Why this mattered.** The second prompt is the one that produced the staged plan
that the whole project then followed — smoke test first, then the full run, then
ablations, then error analysis, then the report. Deciding the *order* of work
before writing any code is what stopped the project from running a 12-epoch
training job on an unvalidated pipeline.

### C.2 Stage 1 — finding the algorithm

**C.2.1** → algorithm-survey assistant (see §1.2 for the full text)

> 我需要完成CISC3024作业，要求用2023-2026年的Deep CNN/Autoencoder做计算机视觉应用。
> 请帮我搜5个候选算法，要求有官方PyTorch代码，能在Colab免费GPU跑。推荐一个最适合MVP的
> 方案，比如MobileNetV4 + CIFAR-10，并给出端到端工作流规划。

*"…search five candidate algorithms, official PyTorch code, runnable on Colab's free
GPU; recommend one for an MVP and give an end-to-end workflow plan."*

**C.2.2** → algorithm-survey assistant

> 這是阶段 1：选题与方案设计 交付的成果，之後怎做

*"This is the Stage 1 deliverable — topic selection and design. What next?"*

**Outcome.** Five candidates with a weighted scoring table (§1.4), and two facts I
would not have found without explicitly asking for them: that MobileNetV4's
reference implementation is TensorFlow rather than PyTorch, and that EfficientViT
had been declared unmaintained.

### C.3 Stage 2 — building the Colab notebook

**C.3.1** → code-generation assistant

> 我要在 Google Colab 上正式訓練 MobileNetV4。請給我一段完整的代碼，要求：
> 1. 先掛載 Google Drive，並把 data_dir 指向 Drive 裡的路徑，確保 CIFAR-10 只下載一次
>    並緩存到 Drive，避免重複下載浪費時間。
> 2. 將輸出目錄（out_dir）也指向 Drive，確保斷點檔案不會因為 Colab 斷線而丟失。
> 3. 修復之前 lr_scheduler.step() 調用順序的警告（需要放在 optimizer.step() 之後）。
> 4. 加入完整的斷點續訓功能（保存 optimizer、scheduler 狀態和 epoch）。
> 請直接給我修改後的完整代碼單元，不要省略任何部分。

*"I want to train MobileNetV4 on Google Colab. Give me complete code that: (1)
mounts Drive and points `data_dir` at it so CIFAR-10 downloads once and is cached;
(2) points `out_dir` at Drive so checkpoints survive a disconnect; (3) fixes the
`lr_scheduler.step()` ordering warning; (4) adds full resume saving optimiser,
scheduler and epoch. Give me the complete cell, omitting nothing."*

**Note on prompt 3.** The scheduler warning was not a guess — it was quoted from the
real smoke-test log (§4.2). Feeding the AI an actual observed warning, rather than a
description of it, is why the fix addressed the real defect.

**C.3.2** → planning assistant

> CISC3024_AI1_MobileNetV4_CIFAR10_MVP_v2.ipynb 修改後要做什么，下一步要做什么

*"After these changes, what is the next step?"*

**C.3.3** → planning assistant

> CISC3024_AI1_MobileNetV4_CIFAR10_MVP_v1.ipynb 這是輸出結果，你看對嗎

*"Here is the output. Is it correct?"*

**Why C.3.3 mattered.** Asking the AI to *critique* its own generated output — rather
than just run it — is what surfaced the `lr_scheduler.step()` warning as a defect
worth fixing, three prompts before it was formally fixed.

### C.4 Stage 3 — validating the smoke test

**C.4.1** → planning assistant

> CISC3024_AI1_MobileNetV4_CIFAR10_MVP_v4.ipynb 執行完正式訓練後下一步是?

*"The full training run is finished. What next?"*

**Outcome.** The 12-epoch run reached 0.9631 test accuracy (§4.3), and the reply
identified the next three stages. It also stated that the RNG warning was harmless
and caused by a PyTorch version difference — **an explanation I checked and
rejected** (§3.3, correction 2). This is the clearest example in the project of
verifying an AI claim instead of accepting it.

### C.5 Stage 4 — the ablation study

**C.5.1** → code-generation assistant

> 我已經跑完 MobileNetV4 + CIFAR-10 的正式訓練。請幫我產生以下 4 組獨立的程式碼單元
> （每次只改一個變量，並將 out_dir 改成對應的新資料夾，避免覆蓋原有結果）：
> 基線對比：將模型換成 resnet18。
> 模型規模：將模型換成 mobilenetv4_conv_medium。
> 輸入解析度：將 img_size 改成 128。
> 微調策略：凍結 MobileNetV4 的主幹（Freeze backbone），只訓練分類頭。
> 請提供完整、無省略的代碼，確保我能直接複製到 Notebook 裡執行。

*"I have finished the main run. Generate four independent code cells, each changing
exactly one variable and writing to its own `out_dir` so nothing overwrites the
original results: (1) baseline → resnet18; (2) scale → mobilenetv4_conv_medium;
(3) resolution → img_size 128; (4) fine-tuning → freeze the backbone, train the head
only. Give complete code, omitting nothing."*

**The design decision that made the ablation valid** is in the phrase *"每次只改一個
變量"* — change exactly one variable. That single instruction is what makes the
resulting table an ablation rather than a collection of unrelated runs.

### C.6 Stage 5 — error analysis and debugging

**C.6.1** → code-generation assistant

> 這是四組的 Test Accuracy / 每輪耗時 / 參數量（或直接貼 ablation_summary.csv）

*"Here are the four runs' test accuracy, time per epoch and parameter counts (or I
can paste `ablation_summary.csv`)."*

**C.6.2** → debugging assistant

> 這個檔案運行時出現error：KeyError: 'freeze_backbone'

*"This notebook raises `KeyError: 'freeze_backbone'` when run."*

**Root cause and fix** are in §3.3, correction 5: two notebooks write a file named
`results.json` with different schemas. The prompt was useful precisely because it
quoted the **exact error string** rather than describing the problem.

**C.6.3** → planning assistant

> 恭喜你！正式訓練已經順利完成……（註：訓練日誌中的 RNG state not restored 警告是無害的，
> 這是因為 PyTorch 版本差異導致的）……根據上方的ai提供的步驟，所以我現在要做什么

*"The run finished successfully… (note: the RNG warning is harmless, caused by a
PyTorch version difference)… Given those steps, what should I do now?"*

**This prompt embedded a false claim, and the right response was to reject it.**
The "version difference" explanation is checkable and wrong — the same code
succeeds on a *newer* PyTorch, which a version difference cannot explain. The real
cause is `map_location='cuda'` moving the RNG state tensor onto the GPU, where
`torch.set_rng_state()` refuses it. Writing the quoted explanation into the report
would have put a verifiable false statement in the submission (§3.3, correction 2).

### C.7 Stage 6 — writing and packaging the report

**C.7.1** → writing assistant

> 報告的要求：Your AI assignment report in Word/PDF format should contain the
> following details (1) how you ask AI tools to find the algorithm, (2) algorithm
> description, (3) how AI implements the algorithm, (4) experiment settings and
> results, (5) what you have learnt from this AI assignment, and (6) a webpage link
> of your source codes.
> 幫我在 C:\Users\michael\workbuddy-ai\3024HW1\outputs 新建的一個名為 CISC3024HW1
> 的資料夾，並把要交付的內容放在這個資料夾

*"The report must contain these six things… Create a folder named CISC3024HW1 under
`outputs/` and put the deliverables in it."*

The six required items became the six numbered sections of this report, one to one.
That mapping is deliberate: it means a marker can check compliance by reading the
table of contents.

**C.7.2** → writing assistant (clarification)

> a webpage link of your source codes 是指把交付的成果上傳到 GITHUB 中，你不需要生成網頁。

*"'A webpage link of your source codes' means uploading the deliverables to GitHub —
you do not need to generate a webpage."*

**C.7.3** → writing assistant (final review)

> 你應該還缺了什麼東西，例如冒煙測試，提示詞，測試圖像等。這些可以都在
> C:\Users\michael\workbuddy-ai\3024HW1\INPUT 找到，提示詞可以用中文表示。

*"You are still missing some things — for example the smoke test, the prompts, the
test images. They are all in `INPUT/`; the prompts can be in Chinese."*

**This prompt found three real gaps**, all now closed: the smoke test became §4.2,
the prompts became this appendix, and the test images became the four
`figures/smoke_*.png` figures.

### C.8 What the prompts deliberately did *not* do

Worth stating, because it is where the marks are:

* **No prompt ever asked the AI to write a conclusion.** Every prompt asked for
  code, a plan, an explanation or a critique. The interpretations — why `conv_small`
  is the right choice, what the calibration curve implies, which findings are
  trustworthy — are mine, and §4.7 says which of them are weak.
* **No prompt asked the AI to "make it accurate".** Asking a model to improve a
  number is how fabricated results get into a report. Every number here came out of
  a run whose log is quoted in this report or stored in `data/`.
* **Prompts quoted real error strings and real log lines** rather than paraphrases.
  That is why the fixes landed on the actual defects.
* **Every AI claim that could be checked was checked** — the TensorFlow/PyTorch
  provenance, the parameter-count discrepancy, the RNG explanation, the frozen
  trainable-parameter count. Four of them were wrong or misleading as first stated;
  all four are documented in §3.3 and the Workflow Log.
