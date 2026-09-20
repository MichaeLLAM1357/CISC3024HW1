# CISC3024 AI Assignment #1 — 提示詞全集（Prompt Log）

**演算法：** MobileNetV4 (ECCV 2024) ｜ **應用：** CIFAR-10 影像分類
**最終結果：** 測試集 top-1 **96.31 %**、top-5 **99.84 %**，2.51 M 參數

---

## 這份文件是什麼

作業明確要求報告要寫「**detailed steps**」，並且要交代「**how you ask AI tools to
find the algorithm**」。這份檔案把整個專案用過的**每一條提示詞**按階段逐字記錄下來，
包含原文（中文）、當時的意圖、以及實際產出了什麼。

> 所有提示詞都是**中文原文**，因為那是我實際輸入的語言。
> 報告正文（`CISC3024_AI1_Report.md`）是英文，其中的 **Appendix C** 是本檔案的英文註解版。

**貫穿全程的四條原則：**

| 原則 | 具體做法 |
|---|---|
| **不叫 AI 下結論** | 所有提示詞只索取「程式碼 / 計畫 / 解釋 / 批評」，結論由我自己下 |
| **不叫 AI「把數字弄好看」** | 要求模型改進準確率＝製造假數據的開始。本專案每個數字都來自可查證的日誌 |
| **貼真實錯誤，不貼描述** | 提示詞直接引用真實報錯字串與日誌行，修復才落在真正的缺陷上 |
| **能查證的 AI 說法一律查證** | 全程有 4 處 AI 說法是錯的或誤導的，全部記錄在報告 §3.3 |

---

## 階段 0 — 讀懂作業

### 0.1 把作業變成可執行的步驟

> AI Assignment #1 CISC3024 Pattern Recognition September, 2026 ... 要怎樣做這份功課，提供步驟

**產出**：一份分階段計畫（選題 → 冒煙測試 → 正式訓練 → 消融 → 錯誤分析 → 報告 → 提交）。

**為什麼重要**：這條提示詞決定了整個專案的**工作順序**。先決定順序再寫程式，
才沒有發生「在未驗證的管線上直接跑 12 個 epoch」這種浪費 30 分鐘 GPU 的事。

### 0.2 決定要用哪些工具／專家

> 在 WORKING BUDDY 的 AI 端中開發這份功課，要有哪些準備，要 CALL 哪些專家

**產出**：工具清單（Colab / Drive / GitHub / UMMoodle）與各階段對應的角色分工。

---

## 階段 1 — 找演算法

### 1.1 核心搜尋提示詞

> 我需要完成CISC3024作业，要求用2023-2026年的Deep CNN/Autoencoder做计算机视觉应用。
> 请帮我搜5个候选算法，要求有官方PyTorch代码，能在Colab免费GPU跑。推荐一个最适合MVP的
> 方案，比如MobileNetV4 + CIFAR-10，并给出端到端工作流规划。

**這條提示詞有三處是刻意設計的：**

1. **點名真正綁死的那個限制** —「**免費** Colab GPU」。不寫這句，得到的會是一堆根本跑不動的
   SOTA 模型。
2. **要 5 個候選，不要 1 個。** 一個推薦無從評估；五個可以互相比較 —— 這正是報告 §1.3
   那張加權評分表的來源。
3. **要工作流，不只要名字。** 模型名字不是計畫。

**產出**：5 個候選演算法 + SC1–SC6 篩選標準 + 加權評分表（MobileNetV4 以 4.90 分居首）。

**兩個不加這句就找不到的事實：**
- MobileNetV4 的**官方參考實作是 TensorFlow**（Google Model Garden），PyTorch 的權威實作是
  Ross Wightman 的 `timm`。
- **EfficientViT 已於 2025-09 宣告停止維護** —— 這正是 DC-AE 雖然最新卻評分最低的原因。

### 1.2 確認階段 1 成果

> 這是阶段 1：选题与方案设计 交付的成果，之後怎做

---

## 階段 2 — 建立 Colab Notebook

### 2.1 四項需求的完整程式碼

> 我要在 Google Colab 上正式訓練 MobileNetV4。請給我一段完整的代碼，要求：
> 1. 先掛載 Google Drive，並把 data_dir 指向 Drive 裡的路徑，確保 CIFAR-10 只下載一次
>    並緩存到 Drive，避免重複下載浪費時間。
> 2. 將輸出目錄（out_dir）也指向 Drive，確保斷點檔案不會因為 Colab 斷線而丟失。
> 3. 修復之前 lr_scheduler.step() 調用順序的警告（需要放在 optimizer.step() 之後）。
> 4. 加入完整的斷點續訓功能（保存 optimizer、scheduler 狀態和 epoch）。
> 請直接給我修改後的完整代碼單元，不要省略任何部分。

**關鍵在於第 3 點不是猜的。** 那句警告是從**真實的冒煙測試日誌**裡抄出來的（見階段 3）。
把「觀察到的真實警告」而不是「對警告的描述」餵給 AI，修復才會落在真正的缺陷上。

**產出**：v2 notebook（Drive 掛載、資料集快取、scheduler 順序修正、原子寫入的完整斷點續訓）。

### 2.2 請 AI 批評自己的產出

> CISC3024_AI1_MobileNetV4_CIFAR10_MVP_v1.ipynb 這是輸出結果，你看對嗎

**為什麼這條重要**：要求 AI **評論**自己生成的東西（而不只是跑它），才把
`lr_scheduler.step()` 警告升級成「值得修的缺陷」—— 比它被正式修好早了三個提示詞。

### 2.3 確認下一步

> CISC3024_AI1_MobileNetV4_CIFAR10_MVP_v2.ipynb 修改後要做什么，下一步要做什么

---

## 階段 3 — 冒煙測試與正式訓練

### 3.1 正式訓練完成後

> CISC3024_AI1_MobileNetV4_CIFAR10_MVP_v4.ipynb 執行完正式訓練後下一步是?

**產出**：12 epoch 訓練完成，測試準確率 0.9631。

⚠️ **同一則回覆裡夾帶了一個錯誤說法**：它說 RNG 警告「是 PyTorch 版本差異導致的，無害」。
我查證後**否定了這個解釋**（見階段 5.3）。這是整個專案裡「查證 AI 說法而不是照單全收」
最清楚的一個例子。

---

## 階段 4 — 消融實驗

### 4.1 四組單變量程式碼

> 我已經跑完 MobileNetV4 + CIFAR-10 的正式訓練。請幫我產生以下 4 組獨立的程式碼單元
> （每次只改一個變量，並將 out_dir 改成對應的新資料夾，避免覆蓋原有結果）：
> 基線對比：將模型換成 resnet18。
> 模型規模：將模型換成 mobilenetv4_conv_medium。
> 輸入解析度：將 img_size 改成 128。
> 微調策略：凍結 MobileNetV4 的主幹（Freeze backbone），只訓練分類頭。
> 請提供完整、無省略的代碼，確保我能直接複製到 Notebook 裡執行。

**「每次只改一個變量」這六個字是整個消融能成立的原因。** 就是它讓最後那張表成為
*消融*（ablation），而不是一堆互不相關的實驗。

**產出**：

| 實驗 | 唯一改變的變數 | Test acc | 參數量 | s/epoch |
|---|---|---|---|---|
| main | —（基準） | 0.9631 | 2.506 M | 159.7 |
| abl_01 | `model_name=resnet18` | 0.9588 | 11.182 M | 165.5 |
| abl_02 | `model_name=conv_medium` | **0.9675** | 8.447 M | 180.6 |
| abl_03 | `img_size=128` | 0.9504 | 2.506 M | 82.4 |
| abl_04 | `freeze_backbone=True` | 0.9056 | 1.244 M 可訓練 | 150.8 |

---

## 階段 5 — 錯誤分析與除錯

### 5.1 提供消融結果

> 這是四組的 Test Accuracy / 每輪耗時 / 參數量（或直接貼 ablation_summary.csv）

### 5.2 回報真實報錯

> 這個檔案運行時出現error：KeyError: 'freeze_backbone'

**根因**：專案裡有**兩個 notebook 都會寫 `results.json`，但 schema 不一致** ——
主實驗寫 5 個鍵，消融寫 17 個鍵。錯誤分析 notebook 是按消融 schema 寫的，指向主實驗時
`freeze_backbone` 這個鍵不存在。

**這條提示詞有用，正是因為它貼的是「完整的錯誤字串」而不是「描述問題」。**

### 5.3 查證 AI 的解釋（本專案最重要的一條）

> 恭喜你！正式訓練已經順利完成……（註：訓練日誌中的 RNG state not restored 警告是無害的，
> 這是因為 PyTorch 版本差異導致的）……根據上方的ai提供的步驟，所以我現在要做什么

**這條提示詞裡夾帶了一個錯誤主張，而正確的回應是拒絕它。**

| | |
|---|---|
| AI 的說法 | 「PyTorch 版本差異導致，無害」 |
| **實際根因** | `map_location='cuda'` 把 checkpoint 裡**每一個** tensor 都搬到了 GPU，**RNG state 也不例外**；而 `torch.set_rng_state()` 要求的是 **CPU 上的 contiguous ByteTensor** |
| **為何「版本差異」被排除** | 同一份程式碼在**更新**的 PyTorch 上**成功**了 —— 版本差異無法解釋「新版成功、舊版失敗」 |

**附帶發現**：原本三個 RNG 恢復呼叫共用同一個 `try`，第一行拋錯後，NumPy 與 Python 的
RNG 也**從來沒有被恢復過**。

**影響評估**：**不影響任何已報告的準確率**，只影響續訓後增強隨機流的逐位可重現性。

> 如果把那句「版本差異」照抄進報告，報告裡就會出現一句**可被查證為假**的陳述。

---

## 階段 6 — 撰寫與打包報告

### 6.1 報告要求與交付路徑

> 報告的要求：Your AI assignment report in Word/PDF format should contain the
> following details (1) how you ask AI tools to find the algorithm, (2) algorithm
> description, (3) how AI implements the algorithm, (4) experiment settings and
> results, (5) what you have learnt from this AI assignment, and (6) a webpage link
> of your source codes.
> 幫我在 C:\Users\michael\workbuddy-ai\3024HW1\outputs 新建的一個名為 CISC3024HW1
> 的資料夾，並把要交付的內容放在這個資料夾

**六項要求 → 報告六個編號章節，一對一。** 這個對應是刻意的：閱卷者只要看目錄就能核對合規。

### 6.2 澄清原始碼連結的形式

> a webpage link of your source codes 是指把交付的成果上傳到 GITHUB 中，你不需要生成網頁。

### 6.3 補齊缺漏（最終覆核）

> 你應該還缺了什麼東西，例如冒煙測試，提示詞，測試圖像等。這些可以都在
> C:\Users\michael\workbuddy-ai\3024HW1\INPUT 找到，提示詞可以用中文表示。

**這條提示詞找出三個真實缺口**，全部已補：

| 缺口 | 補齊方式 |
|---|---|
| **冒煙測試** | 報告 §4.2（v1：128px/2ep → 0.9053；v3 → 0.9034，並記錄 GradScaler 跳步） |
| **提示詞** | 本檔案 + 報告 Appendix C |
| **測試圖像** | `figures/smoke_v1_confusion_matrix.png`、`smoke_v3_confusion_matrix.png`、`smoke_v3_curves.png`、`smoke_sample_batch.png` |

---

## 附：AI 做錯的 4 處（我發現並修正的）

| # | AI 的說法 | 實際情況 | 我怎麼發現的 |
|---|---|---|---|
| 1 | `lr_scheduler.step()` 位置無所謂 | 必須在 `optimizer.step()` **之後**；且 AMP 跳步時**不能**推進 scheduler | 冒煙測試日誌裡的 `UserWarning` |
| 2 | RNG 警告是「PyTorch 版本差異，無害」 | 真因是 `map_location='cuda'` 把 RNG tensor 搬到了 GPU | 同一份程式碼在**更新**的 PyTorch 上成功 → 版本差異無法解釋 |
| 3 | MobileNetV4 有 3.77 M 參數 | timm 報 **2.51 M**；差異來自 **BN folding** | 兩邊數字都對，但必須說明用的是哪種口徑 |
| 4 | 凍結主幹後只剩很少參數可訓練 | MobileNetV4 的 head 含 `conv_head`（1×1 conv 擴到 1280 通道，1.23 M）→ **仍有 49.7 % 可訓練** | 讀 timm 的 `forward_head` 邊界定義 |

---

## 附：AI 產出的驗證方式

| 驗證對象 | 方法 | 結果 |
|---|---|---|
| 消融 harness | stub 資料集（200 張代替 50 000）+ `pretrained=False` 離線權重，四個消融端到端跑完 | **24/24 斷言通過** |
| RNG 修復 | 從**已修補的 notebook 原始碼**抽出 4 個 helper，跑真實 save → advance → load → redraw | **13/13 斷言通過** |
| 錯誤分析修復 | 對三套重建 schema + `img_size=128` 案例執行修補後的 cell | **25/25 斷言通過** |

---

*本檔案為 CISC3024 AI Assignment #1 的提示詞記錄。英文註解版見報告 Appendix C。*
