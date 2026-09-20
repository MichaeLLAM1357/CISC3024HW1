# CISC3024 Pattern Recognition — AI Assignment #1

**MobileNetV4 (ECCV 2024) fine-tuned on CIFAR-10**

A 2.51 M-parameter MobileNetV4 reaches **96.31 % top-1** and **99.84 % top-5**
accuracy on the CIFAR-10 test set, beating an 11.18 M-parameter ResNet-18 baseline
by 0.43 points at **4.46× fewer parameters** — trained entirely on free Google
Colab GPU time.

> Every line of code in this repository was generated with AI assistance, as the
> assignment brief requires. The report documents what the AI got right, what it
> got wrong, and how each error was caught.
>
> * [`CISC3024_AI1_AI_Workflow_Log.md`](CISC3024_AI1_AI_Workflow_Log.md) — the
>   chronological record: every prompt, every error, every correction.
> * [`CISC3024_AI1_Prompt_Log.md`](CISC3024_AI1_Prompt_Log.md) — the same prompts
>   grouped by stage, in Chinese (the language they were actually typed in).

---

## Results

| Run | Model | Img | Frozen | Params (M) | Trainable (M) | Best val | Test acc | Δ vs main | s/epoch |
|---|---|---|---|---|---|---|---|---|---|
| **main** | `mobilenetv4_conv_small` | 224 | no | 2.506 | 2.506 | 0.9670 | **0.9631** | — | 159.7 |
| abl_01 | `resnet18` | 224 | no | 11.182 | 11.182 | 0.9604 | 0.9588 | −0.43 | 165.5 |
| abl_02 | `mobilenetv4_conv_medium` | 224 | no | 8.447 | 8.447 | 0.9690 | 0.9675 | +0.44 | 180.6 |
| abl_03 | `mobilenetv4_conv_small` | 128 | no | 2.506 | 2.506 | 0.9542 | 0.9504 | −1.27 | 82.4 |
| abl_04 | `mobilenetv4_conv_small` | 224 | **yes** | 2.506 | **1.244** | 0.9160 | 0.9056 | **−5.75** | 150.8 |

Deltas in percentage points. Parameter counts follow the **timm** convention
(batch-norm layers kept separate); the paper's folded counts are 3.77 M and 9.72 M
for the same two models.

### Four findings

1. **The modern architecture wins on size.** ResNet-18 is 4.46× larger and
   0.43 points *worse*.
2. **Extra capacity is not worth it here.** `conv_medium` gains +0.44 points for
   3.37× the parameters and 13 % more time per epoch — inside single-seed noise,
   and its validation loss is markedly unstable.
3. **Resolution is the biggest time lever.** 128 px halves the time per epoch for
   1.27 points, making it the right default for iteration.
4. **"Freezing the backbone" is the worst trade in the study.** It loses 5.75
   points while saving only 5.6 % of time — and it still trains **49.7 %** of
   parameters, because timm places `conv_head` after the pooling layer.

---

## Smoke test

Before committing to a 12-epoch run, the pipeline was validated twice at reduced
scale (`img_size=128`, `epochs=2`). It costs ~3 minutes against ~32 for the real
run, and it paid for itself twice over.

| | Smoke run 1 (`v1`) | Smoke run 2 (`v3`) |
|---|---|---|
| Best validation accuracy | 0.9086 | 0.9128 |
| **Test accuracy** | **0.9053** | **0.9034** |
| GradScaler step skips | not instrumented | 8 at epoch 1 |

Two things came out of this stage that changed the final code:

1. **The scheduler-ordering bug was caught here, not in the main run.** Smoke run 1
   emitted the `lr_scheduler.step()` warning that became correction 1. Left
   unnoticed, the cosine schedule would have been one step out of phase for all 12
   epochs of the real run.
2. **The AMP guard is observably working.** Smoke run 2 skipped 8 steps and its
   first-epoch LR is **2.94e-04** rather than the nominal 3.00e-04 — the scheduler
   was correctly held back on the skipped steps.

Even at two epochs the error structure matches the final model: **cat and dog are
the weakest classes**, and the largest off-diagonal confusion is the cat/dog pair.

Figures: `figures/smoke_v1_confusion_matrix.png`,
`figures/smoke_v3_confusion_matrix.png`, `figures/smoke_v3_curves.png`,
`figures/smoke_sample_batch.png`. Written up in report §4.2.

---

## Error analysis

| | |
|---|---|
| Test top-1 | 0.9633 (recomputed; 0.9631 in training) |
| Test top-5 | 0.9984 |
| Errors | 367 / 10 000 |
| Mean confidence, correct | 0.8885 |
| Mean confidence, wrong | 0.6217 |
| Worst class | cat (recall 0.914) |
| Best class | frog (recall 0.989) |

**The residual error is concentrated, not spread.** Eight of ten classes sit in a
0.964–0.981 F1 band, while cat (0.9177) and dog (0.9264) carry roughly four times
the error rate of the best class. The **cat / dog / deer cluster accounts for 131
errors — 35.7 % of all errors.**

Top confusions:

| True → predicted | Count | % of true class |
|---|---|---|
| dog → cat | 52 | 5.20 % |
| cat → dog | 45 | 4.50 % |
| truck → automobile | 23 | 2.30 % |
| automobile → truck | 16 | 1.60 % |
| airplane → ship | 16 | 1.60 % |

One of the 16 most confidently-wrong images is labelled `cat` but is
unmistakably **a green frog** — label noise in CIFAR-10, and a reminder that not
every error in the confusion matrix is a model error.

---

## Repository contents

```
.
├── CISC3024_AI1_Report.md              the full report (6 required sections)
├── CISC3024_AI1_Report.docx            the same report in Word
├── CISC3024_AI1_Report.pdf             the same report in PDF
├── CISC3024_AI1_AI_Workflow_Log.md     every prompt, error and correction
├── CISC3024_AI1_Prompt_Log.md          the prompt log, in Chinese
├── notebooks/
│   ├── 01_main_training_MobileNetV4_CIFAR10.ipynb
│   ├── 02_ablation_study.ipynb
│   └── 03_error_analysis.ipynb
├── figures/                            19 figures used in the report
│   ├── smoke_*.png                     smoke-test evidence (§4.2)
│   ├── main_*.png                      the main run
│   ├── abl_*.png                       the four ablations
│   └── error_*.png                     error analysis and calibration
└── data/
    ├── ablation_summary.csv
    └── error_summary.json
```

---

## How to reproduce

1. **Smoke test first.** Open `notebooks/01_main_training_MobileNetV4_CIFAR10.ipynb`
   in Google Colab, select a **T4 GPU**, set `img_size=128` and `epochs=2`, and run
   all. This takes ~3 minutes and proves the whole pipeline works before you spend
   half an hour on the real run. Expect ~0.90 test accuracy — that is correct for
   two epochs, not a bug.
2. **The main run.** Set `img_size=224` and `epochs=12`, and run all. CIFAR-10 is
   cached on Google Drive on the first run and reused afterwards; 12 epochs take
   roughly 35 minutes. If the session disconnects, run the notebook again — it
   resumes automatically from the last checkpoint.
3. Open `notebooks/02_ablation_study.ipynb` and run the four ablation cells in
   order (128 px first — it is the fastest). Each writes to its own run directory
   so nothing is overwritten, and the last cell emits `ablation_summary.csv`.
4. Open `notebooks/03_error_analysis.ipynb`, set `ANALYSE_TAG = "mnv4_cifar10"`,
   and run all. Change the tag to analyse any ablation run instead.

### Environment

| | |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| timm | 1.0.29 |
| Hardware | Google Colab free tier, NVIDIA Tesla T4 |
| Seed | 42 |

---

## A note on the resolution trap

MobileNetV4's pretrained weights are ImageNet-1k models at **224 px**; CIFAR-10 is
**32 × 32**. Feeding 32 × 32 directly collapses the feature map to **1 × 1** before
the classifier sees it, because the network has five stride-2 stages. The pipeline
therefore uses the ImageNet protocol — `Resize(256) → RandomCrop(224)` — which is
the single most important engineering detail in the project.

The honest cost: upsampling 32 → 224 px **invents no detail**. The model sees an
interpolated image, which caps how much the pretrained features can help.

---

## Reference

Qin, D., Leichner, C., Delakis, M., et al. *MobileNetV4 — Universal Models for the
Mobile Ecosystem.* ECCV 2024. [arXiv:2404.10518](https://arxiv.org/abs/2404.10518)

PyTorch implementation: [`huggingface/pytorch-image-models`](https://github.com/huggingface/pytorch-image-models) (timm).
MobileNetV4's *reference* implementation is TensorFlow (Google Model Garden); timm
is the authoritative PyTorch port.
