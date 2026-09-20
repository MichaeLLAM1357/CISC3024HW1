# CISC3024 AI Assignment #1 — AI Workflow Log

**Appendix A: detailed record of every AI interaction, error, and correction.**

This log exists to satisfy the assignment requirement to document *detailed steps*
on how the work was done, and the constraint that **no code was written by hand**.
Every prompt below is reproduced verbatim as sent. Every error is reproduced as
observed.

> **Companion file.** `CISC3024_AI1_Prompt_Log.md` collects the same prompts
> grouped by stage, in Chinese, without the error narratives. This log is the
> chronological version; that one is the reference version. The English-annotated
> version is Appendix C of the report.

---

## 0. At a glance

| Phase | Date | AI contribution | Human contribution | Artefacts |
|---|---|---|---|---|
| P0 | 2026-09-19 | Parsed the assignment brief, extracted constraints | Supplied the PDF, asked the first question | — |
| P1 | 2026-09-19 | Screened 5 candidate algorithms, scored them, recommended one | Chose to accept the recommendation | `Algorithm_Survey_and_MVP_Plan.html` |
| P2 | 2026-09-19 | Wrote the v1 → v2 Colab notebook | Requested 4 specific engineering changes | `..._MVP_v2.ipynb` |
| P3a | 2026-09-19 | — | **Ran the smoke test on Colab** (128 px, 2 epochs, twice) | `..._v1.ipynb`, `..._v3.ipynb`, `figures/smoke_*.png` |
| P3b | 2026-09-19 | — | **Ran the 12-epoch main training on Colab** | `..._v4.ipynb`, `best.pt` |
| P4 | 2026-09-19 | Wrote 4 single-variable ablation cells | Ran them | `Ablation_Cells.ipynb` |
| P5 | 2026-09-20 | Wrote the error-analysis notebook | Ran it | `Error_Analysis.ipynb` |
| P6 | 2026-09-20 | Diagnosed and fixed the RNG resume bug | Reported the error output | `..._MVP_v5.ipynb` |
| P7 | 2026-09-20 | Drafted the report sections from real results | Filled remaining sections, proofread | `Report_*.md` |

---

## 1. Phase 0 — Interpreting the brief

**Prompt sent (verbatim):**

> @scene#17:"Agent 应用" "我需要完成CISC3024作业，要求用2023-2026年的Deep CNN/Autoencoder做计算机视觉应用。请帮我搜5个候选算法，要求有官方PyTorch代码，能在Colab免费GPU跑。推荐一个最适合MVP的方案，比如MobileNetV4 + CIFAR-10，并给出端到端工作流规划。" CISC3024作业要求在Alassign1.pdf中 @D:/CISC3024/AIassign1.pdf

**What the AI produced.** The PDF was read and its constraints extracted. Three
matter for the report:

1. The algorithm must be from **2023–2026**.
2. The report must document **detailed steps**.
3. **"You are NOT allowed to write any code by yourself!"** — all code must be
   AI-generated. This is why this log exists.
4. Submission is via UMMoodle within one week, through a Turnitin similarity check.

The AI turned the vague brief into six screening criteria (recency, official
PyTorch implementation, Colab-feasible, parameter budget, reported accuracy,
licence), which are reproduced in §3.1 of the report.

---

## 2. Phase 1 — Algorithm survey

Five candidates were researched and scored:

| Algorithm | Venue | Params | Reported acc. | Weighted score |
|---|---|---|---|---|
| **MobileNetV4** | ECCV 2024 | 3.77 M | 73.8–74.6 % | **4.90** |
| RepViT | CVPR 2024 | 5.1 M | 78.7–79.1 % | 4.78 |
| FasterNet | CVPR 2023 | 3.9 M | 71.9 % | 4.65 |
| ConvNeXt V2 + FCMAE | CVPR 2023 | 3.7 M | 76.7 % | 4.60 |
| DC-AE + EfficientViT | ICLR 2025 | — | rFID | 4.20 |

**An honesty point the AI flagged, which is in the report.** MobileNetV4's
*reference* implementation is TensorFlow (Google Model Garden). The authoritative
**PyTorch** implementation is Ross Wightman's `timm`. The report states this
rather than implying an official PyTorch release exists.

---

## 3. Phase 2 — Building the Colab notebook

**Prompt sent (verbatim):**

> 「我要在 Google Colab 上正式訓練 MobileNetV4。請給我一段完整的代碼，要求：
> 1. 先掛載 Google Drive，並把 data_dir 指向 Drive 裡的路徑，確保 CIFAR-10 只下載一次並緩存到 Drive，避免重複下載浪費時間。
> 2. 將輸出目錄（out_dir）也指向 Drive，確保斷點檔案不會因為 Colab 斷線而丟失。
> 3. 修復之前 lr_scheduler.step() 調用順序的警告（需要放在 optimizer.step() 之後）。
> 4. 加入完整的斷點續訓功能（保存 optimizer、scheduler 狀態和 epoch）。
> 請直接給我修改後的完整代碼單元，不要省略任何部分。」

**What the AI produced.** A 26-cell notebook implementing all four requests.

**The single most important engineering decision** — the AI identified that
MobileNetV4's pretrained weights are ImageNet models at 224 px, but CIFAR-10 is
32 × 32. Feeding 32 × 32 directly passes through five stride-2 stages and
collapses the feature map to 1 × 1 before the classifier ever sees it. The fix is
the standard ImageNet protocol: `Resize(256) → RandomCrop(224)`. This is
documented in §4.3 of the report.

**Request 3 in detail.** The learning-rate scheduler was being stepped before the
optimiser, which raised a PyTorch warning and let the schedule advance on
iterations where the optimiser had not actually moved. The corrected order is:

```python
scale_before = scaler.get_scale()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)            # the real optimiser step
scaler.update()                   # may halve the scale on inf/NaN
if scaler.get_scale() >= scale_before:
    scheduler.step()              # only after a genuine optimiser step
else:
    skipped_steps += 1
```

The `get_scale()` comparison matters because AMP *silently skips*
`optimizer.step()` when it encounters inf/NaN gradients. Stepping the scheduler
anyway would desynchronise the schedule from the optimiser.

---

## 4. Phase 3 — Running it (human step)

**Environment observed:**

```text
Python : 3.13.15
PyTorch: 2.11.0+cu128
CUDA available: True
GPU  : Tesla T4
VRAM : 14.6 GB
timm version: 1.0.29
```

**Smoke test first.** Two reduced-scale runs at `img_size=128`, `epochs=2`, before
committing to the full 12-epoch job. Both logs are quoted in full because they are
where two of the project's findings came from.

```text
# ---- smoke run 1 (v1) -- this is where the scheduler bug surfaced ----
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
epoch  1/2 | train 2.0718 acc 0.5391 | val 1.0299 acc 0.7898 | lr 3.00e-04 |  91.2s   <- best
epoch  2/2 | train 0.9253 acc 0.8235 | val 0.7277 acc 0.9086 | lr 0.00e+00 |  78.6s   <- best
best validation accuracy: 0.9086
TEST  loss 0.7355   accuracy 0.9053

# ---- smoke run 2 (v3) -- after the Drive / resume rewrite ----
        (note: 8 step(s) skipped by GradScaler this epoch)
epoch  1/2 | train 2.1590 acc 0.5269 | val 0.9590 acc 0.8102 | lr 2.94e-04 |  93.7s   <- best
epoch  2/2 | train 0.9362 acc 0.8198 | val 0.7275 acc 0.9128 | lr 3.84e-07 |  90.7s   <- best
best validation accuracy: 0.9128
TEST  loss 0.7386   accuracy 0.9034
```

**Two findings, both cheap to act on and expensive to miss.**

1. **The scheduler-ordering bug was caught here, not in the main run.** Smoke run 1
   emitted the `lr_scheduler.step()` warning that became correction 1 (§7). Left
   unnoticed, the cosine schedule would have been advanced one step out of phase for
   all 12 epochs of the real run. Three minutes of GPU time found it instead of
   thirty.
2. **The AMP guard is observably doing something.** Smoke run 2 skipped 8 steps at
   epoch 1, and its first-epoch learning rate is **2.94e-04** rather than the
   nominal **3.00e-04** — the scheduler was correctly held back on the skipped
   steps. That is the guard described in §3 working on real data, not only in a unit
   test.

Both runs already show the error structure that survives to the final model: **cat
and dog are the weakest classes**, and the largest off-diagonal confusion is the
cat/dog pair. The failure mode was visible before any real training was done.

Figures: `figures/smoke_v1_confusion_matrix.png`,
`figures/smoke_v3_confusion_matrix.png`, `figures/smoke_v3_curves.png`,
`figures/smoke_sample_batch.png`. Written up in report §4.2.

**The arithmetic that justifies this step.** The smoke test costs ~3 minutes; the
full run costs ~32. Running the cheap version first is the single best-value
decision in the project — and it is the step that a first-time reader of the brief
is most likely to skip.

**Main run** (v4 notebook, `img_size=224`, `epochs=12`): the session was
interrupted at epoch 7 and resumed from the Drive checkpoint.

```text
found checkpoint: /content/drive/MyDrive/cisc3024_ai1/runs/mnv4_cifar10/last.pt
  (RNG state not restored: RNG state must be a torch.ByteTensor )
resumed -> next epoch 8, best val acc 0.9508, lr 1.30e-04
epoch  8/12 | train 0.6128 acc 0.9570 | val 0.6043 acc 0.9594 | lr 8.86e-05 | 171.6s   <- best
epoch  9/12 | train 0.5849 acc 0.9693 | val 0.5948 acc 0.9614 | lr 5.25e-05 | 158.2s   <- best
epoch 10/12 | train 0.5656 acc 0.9788 | val 0.5885 acc 0.9652 | lr 2.43e-05 | 156.1s   <- best
epoch 11/12 | train 0.5517 acc 0.9851 | val 0.5869 acc 0.9670 | lr 6.35e-06 | 156.4s   <- best
epoch 12/12 | train 0.5439 acc 0.9883 | val 0.5873 acc 0.9660 | lr 3.18e-09 | 156.2s
```

**Note what worked.** The resume restored the optimiser, scheduler and AMP scaler
state exactly: the learning rate picked up at 1.30e-04 and continued decaying, and
the best-validation tracker retained 0.9508 from before the interruption. The only
casualty was the RNG state — see §6 below.

**Final result:** test accuracy **0.9631**, best validation **0.9670** (epoch 11).

---

## 5. Phase 4 — Ablation study

**Prompt sent (verbatim):**

> 「我已經跑完 MobileNetV4 + CIFAR-10 的正式訓練。請幫我產生以下 4 組獨立的程式碼單元（每次只改一個變量，並將 out_dir 改成對應的新資料夾，避免覆蓋原有結果）：
> 1. 基線對比：將模型換成 resnet18。
> 2. 模型規模：將模型換成 mobilenetv4_conv_medium。
> 3. 輸入解析度：將 img_size 改成 128。
> 4. 微調策略：凍結 MobileNetV4 的主幹（Freeze backbone），只訓練分類頭。
> 請提供完整、無省略的代碼，確保我能直接複製到 Notebook 裡執行。」

**What the AI produced.** One shared setup cell defining
`run_experiment(tag, **overrides)`, plus four one-line calls, plus a comparison
cell that emits `ablation_summary.csv`.

```python
summary_resnet18 = run_experiment("abl_01_resnet18", model_name="resnet18")
summary_medium   = run_experiment("abl_02_mnv4_medium",
                                  model_name="mobilenetv4_conv_medium.e500_r224_in1k")
summary_img128   = run_experiment("abl_03_img128", img_size=128)
summary_frozen   = run_experiment("abl_04_freeze_backbone", freeze_backbone=True)
```

Because `out_dir = os.path.join(RUNS_ROOT, tag)` and the tag is the first
argument, each run is written to its own folder. The main run at
`runs/mnv4_cifar10` is **read** for comparison and never written, so it cannot be
overwritten by any ablation.

**Two findings the AI surfaced while building this, both worth reporting:**

1. **timm's parameter counts disagree with the paper.** `conv_small` = 2.51 M
   (paper: 3.77 M), `conv_medium` = 8.45 M (paper: 9.72 M), `resnet18` = 11.18 M.
   The difference is that the paper folds batch-norm layers into the preceding
   convolutions while timm keeps them separate. The report must state which
   convention it uses.
2. **The freeze-backbone ablation is weaker than it sounds.** Freezing everything
   after `global_pool` still leaves **1.244 M of 2.51 M parameters trainable
   (49.7 %)**, because timm places `conv_head` — a 1 × 1 convolution expanding
   960 → 1280 channels — *after* the pooling layer, so it counts as part of the
   head.

---

## 6. Phase 5 — The RNG resume bug (found, diagnosed, fixed)

This is the most substantive correction in the project and is worth documenting in
full, because the first explanation offered for it was **wrong**.

### 6.1 Symptom

```text
(RNG state not restored: RNG state must be a torch.ByteTensor )
```

The message is misleading: the saved state genuinely *was* a `uint8` tensor.

### 6.2 An incorrect first explanation, and why it was rejected

The initial explanation given was *"a harmless PyTorch version difference."*
This was tested and rejected. The save/load/set round-trip was reproduced
successfully on a different PyTorch version (`2.14.0+cpu`) with the identical
code, which rules out a version incompatibility — if it were a version
difference, the newer version would be the one to fail, not the older one.

### 6.3 Actual root cause

The assertion lives in `ATen/core/Generator.h`:

```cpp
inline void check_rng_state(const c10::TensorImpl& new_state) {
  TORCH_CHECK_TYPE(
    new_state.layout() == kStrided && new_state.device().type() == kCPU
        && new_state.dtype() == kByte,
    "RNG state must be a torch.ByteTensor"
  );
  TORCH_CHECK(new_state.is_contiguous(), "RNG state must be contiguous");
}
```

Three conditions, not one — and the error text only mentions the dtype one. The
failing condition was the **device**: `torch.load(..., map_location=device)` was
called with `device='cuda'`, and `map_location` relocates *every* tensor in the
checkpoint onto the GPU, the RNG state included. `torch.set_rng_state()` requires
a **CPU** ByteTensor.

This was confirmed by testing each branch separately:

| Input to `set_rng_state` | Observed result |
|---|---|
| CPU ByteTensor (normal case) | accepted |
| `int64` (wrong dtype) | `RNG state must be a torch.ByteTensor` |
| non-CPU device | `RNG state must be a torch.ByteTensor` |
| non-contiguous | `RNG state must be contiguous` (different message) |

Since `get_rng_state()` always returns `uint8`, and the "contiguous" message is
distinct, only the device branch can have fired.

**A second, independent defect in the same block.** All three restore calls shared
one `try`. The exception on the first line meant `np.random.set_state()` and
`random.setstate()` **never executed at all** — so the NumPy and Python RNGs were
also not restored. Each is now guarded independently.

### 6.4 Fix

```python
rng = ckpt.get("rng")
if rng:
    try:
        st = rng["torch"] if isinstance(rng, dict) else rng
        if not isinstance(st, torch.Tensor):
            st = torch.as_tensor(st)
        torch.set_rng_state(
            st.detach().to(device="cpu", dtype=torch.uint8).contiguous())
        if isinstance(rng, dict):
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("python") is not None:
                random.setstate(rng["python"])
    except Exception as e:
        print("  (RNG state not restored:", e, ")")
```

**Verified** by extracting the four checkpoint helper functions out of the
*patched notebook source itself* and executing a real save → advance → load →
redraw round-trip: **13 of 13 assertions passed**, including full reproduction of
the torch, NumPy and Python random streams, and tolerance of `int64` and
list-form state.

### 6.5 Impact on the reported results

**None.** The RNG state affects only the augmentation random stream. The resumed
run continued with correct model weights, optimiser state, learning rate and
scheduler position — all of which are independently visible in the log above (LR
resumed at 1.30e-04 and decayed correctly; the best-validation tracker retained
0.9508). The reported 0.9631 test accuracy is unaffected.

The one honest caveat, stated in §10 of the report: a fresh 12-epoch run from
scratch would land within normal run-to-run variation of 0.9631 but would not
reproduce it to the digit, because the augmentation stream after epoch 7 was not
bit-identical to an uninterrupted run.

---

## 7. Other corrections made

| # | What was wrong | How it was caught | Fix |
|---|---|---|---|
| 1 | `lr_scheduler.step()` before `optimizer.step()`; scheduler advanced on AMP-skipped steps | PyTorch warning during the smoke test | Reordered; guarded with a `scaler.get_scale()` comparison |
| 2 | RNG state not restored on resume | Error printed by the resumed run | Pinned the state to CPU (§6) |
| 3 | Parameter counts disagreed with the paper | Comparing timm's `sum(p.numel())` against the paper table | Traced to BN folding; report now states the convention |
| 4 | "Freeze backbone" still trained 49.7 % of parameters | Printing trainable-parameter count | Kept the experiment; documented the `conv_head` boundary caveat |
| 5 | A `CifarWrap` helper referenced a non-existent `self.indices` | Offline execution of the notebook cells | Removed before delivery |
| 6 | Re-running a finished experiment overwrote `sec_per_epoch_mean` with `None` | Re-running an ablation to test resume | Carried the previous value forward from `results.json` |
| 7 | A corrupt HuggingFace cache (`SafetensorError: header too small`) failed silently and retried pointlessly | Reproduced a blocked download | Added explicit detection and an actionable message |
| 8 | The error-analysis notebook assumed the **ablation** `results.json` schema and crashed on the main run with `KeyError: 'freeze_backbone'` | Running it against the main run | Normalised both schemas into one merged view; added a fail-fast guard that lists the available runs |

### 7.1 Detail on correction #8 (two schemas for the same filename)

Two different notebooks write a file called `results.json`, and they do **not**
agree on its shape:

| | main run notebook | ablation notebook |
|---|---|---|
| top-level keys | `test_acc`, `best_val_acc`, `epochs_run`, `history`, `cfg` | those **plus** `tag`, `model_name`, `img_size`, `freeze_backbone`, `freeze_mode`, `total_params`, `trainable_params`, `sec_per_epoch_mean`, `test_loss`, `out_dir` |
| freezing fields | absent | present |
| parameter counts | absent | present |

The error-analysis notebook was written against the ablation shape and indexed
`CFG["freeze_backbone"]` directly. Pointed at the main run — which is exactly
what `ANALYSE_TAG = "mnv4_cifar10"` does — that key does not exist and the cell
dies with `KeyError`.

The fix reads through a single merged view instead of assuming either schema:

```python
CFG = dict(RES.get("cfg", {}))
RUN = {**CFG, **{k: v for k, v in RES.items() if k != "cfg"}}
RUN.setdefault("tag", ANALYSE_TAG)
RUN.setdefault("freeze_backbone", False)   # the main run never freezes anything
RUN.setdefault("freeze_mode", None)
```

All later accesses go through `RUN`. A guard was added to the configuration cell
so a missing or mistyped `ANALYSE_TAG` fails immediately with the list of runs
that do exist, rather than surfacing as a bare `FileNotFoundError` many cells
later:

```text
FileNotFoundError: missing results.json for ANALYSE_TAG='does_not_exist'
  looked in : /content/drive/MyDrive/cisc3024_ai1/runs/does_not_exist
  available : ['abl_01_resnet18', 'abl_03_img128', 'abl_04_freeze_backbone',
               'mnv4_cifar10']
```

**Verified** by executing the patched configuration cell against three
reconstructed schemas (main run, plain ablation, frozen ablation) plus an
`img_size=128` case: **25 of 25 assertions passed**, including that the
pre-processing `RESIZE` follows the analysed run's own `img_size` rather than a
hard-coded constant. The same check was run against the ablation notebook's
comparison cell, which already used `.get()` throughout and needed no change.

---

## 8. Summary of what the human did vs what the AI did

**The AI did:** reading the brief, the literature search, all code, the offline
verification harnesses, the bug diagnosis, and the drafting of the report prose.

**The human did:** supplied the assignment and the environment; ran every cell on
Colab; reported errors back verbatim; made the judgement calls (accepting the
MobileNetV4 recommendation, choosing 12 epochs, choosing to document rather than
hide the freeze-backbone caveat); and verified that the reported numbers match the
actual notebook output.

No line of code in any submitted notebook was written by hand. Where the AI was
wrong, the error was found by running the code and reading the output — which is
why every correction in §7 is accompanied by the observed evidence rather than an
assertion.
