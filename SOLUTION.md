# 控球判定方案：工作记录与计划

这份文档记录我们在 More than a Hack 2027 Phase 1（逐帧判定控球方）上做过的事、当前结果、怎么在另一台机器上复现，以及接下来的计划。

- 提交截止：**2026-11-17**
- 要超过的官方 baseline：全场 spellF1 **0.2433**（`baseline_smooth50`）
- 代码都在 [`possession/`](possession/) 目录下

---

## 0. 2026-11-14 晚更新：当前版本（v4）

本节是现在的提交版本。后面的 0.1 节（v3）和第 1–8 节是之前的记录；模型、参数和复现步骤以本节为准。

### 现在提交用的是什么
- **两个模型合用**：每一帧的 H/A/D 概率 = **0.35 × LightGBM + 0.65 × GRU**，权重写在 `possession/model/decoder.json` 的 `lgbm_weight`。
  - **LightGBM**：配置和 v3 相同（109 个特征，2 份弄脏副本加主客互换，126 轮），用修好方向的特征重新训练。
  - **GRU**（[`possession/seq.py`](possession/seq.py)）：双向 GRU，读和 LightGBM 一样的 109 个特征，但按时间顺序读：每秒 5 步，一个窗口 60 秒，前后都能看到。训练 6 轮，种子 2027。权重存成 `gru.npy`（所有参数排成一列 float32），其余信息（结构、特征表、特征标准化参数）存在 `gru.json`。
- **解码**：Viterbi，`team_cost=2, dead_cost=64, min_len=75`，和 v3 相同。
- **开球规则**：和 v3 相同，在合用之后、解码之前生效。
- **修了一个进攻方向的 bug**（`features._orient`）：每个半场要先统一方向，让主队防守 −x 一侧。原来的规则是"主队平均站位过了中线就翻转"，可主队整个半场压着对手打时会被判反（德甲 J03WR9 上半场就是这样）。现在改成比较两队的平均站位：主队平均更靠 +x 就翻转。巴萨这场两条规则结论相同（已在两个半场的开球时刻核对），所以巴萨的输入没有变：用 v3 模型重新预测，CSV 逐字节相同。

### 怎么选出来的
考场和 v3 一样：7 场德甲轮流当考题，答案按 2 秒受控口径，用官方 spellF1 打分。GRU 的考场是 [`possession/seq_cv.py`](possession/seq_cv.py)，考题和 `cv.py` 完全相同（输入指纹一致）；它和最终训练用同一个函数 `seq.fit` 训练。

| 版本 | 7 场考场 spellF1 | 和 v3 比 | 7 场里变好 |
|---|---|---|---|
| 官方 baseline_smooth50 | 0.195 | | |
| v3 | 0.642 | — | — |
| LightGBM，修好方向 | 0.659 | +0.017 | 4 场 |
| GRU，修好方向，6 轮 | 0.701 | +0.059 | 7 场 |
| **v4：0.35 × LightGBM + 0.65 × GRU** | **0.712** | **+0.070** | **7 场** |

- 和"修好方向的 LightGBM"比，合用也是 7 场全部变好（+0.053）。
- 公开 5 分钟只在最后对最终版本查了一次，不参与任何选择：spellF1 0.690、macroF1 0.908（v3 是 0.676 / 0.903，官方 baseline_smooth50 是 0.326）。这 5 分钟只有 17 段控球，这点差别在噪声范围内。
- 0.35 和 6 轮是在考场上挑的，算轻微地看了考卷；但 0.20 和 0.50 也有 0.710 和 0.708，结果不依赖恰好选中 0.35。

**试过但没有采用**：
- 触球线索（某队球员 1–2 米内有球、对方没人靠近）：+0.012，但 7 场只赢 4 场，没过门槛；代码留在 `features.py`，默认不启用。
- GRU 训练 30 轮：0.624，比训练 6 轮的 0.657 低（这两次都在修方向之前），学过头了。
- 3 个不同随机种子的 GRU 取平均（`python -m possession.seq_cv --tag g3s3 --epochs 6 --seeds 3`）：单用 GRU 时从 0.701 升到 0.710，7 场全赢；但按 35/65 和 LightGBM 合用后是 0.711，比 v4 的 0.712 低 0.001，只赢 3 场，没过门槛。合用权重 0.35 事先定好不重调。
- 解码改成 `team_cost=4`：只多 0.001，没过门槛。

### 复现
环境：`pip install -r possession/requirements.txt`（推荐 Python 3.12）。v4 新增了 PyTorch 2.7.1。只做预测的话 CPU 版就够；训练时有 NVIDIA GPU 会自动用 GPU。
```bash
python -m possession.train
```
在 RTX 5060 笔记本上约 2.5 分钟，其中 GRU 约 40 秒。LightGBM 和 GRU 用同一批 28 份训练数据（7 场 × 2 份弄脏副本 × 主客互换）。在这台机器上重新训练两次，4 个模型文件（`lgbm.txt`、`decoder.json`、`gru.npy`、`gru.json`）逐字节一致。没有 GPU 时可以用 `--device cpu`，但 GRU 权重会和 GPU 训练出来的略有不同，所以提交以仓库里的模型文件为准。
```bash
python -m possession.predict
```
约 17 秒。GRU 在 CPU 上用 float64 计算，不需要 GPU。输出 `evaluation/results/lgbm_viterbi.csv`，sha256 以 `e82e6b31bb93d690` 开头，连跑两次完全一致。v3 的模型和 CSV 备份在 `.superpowers/backup/v3_2026-11-14/`（不进 git）。

整场比赛上（不看答案）：死球占 46%，活球里巴萨控球 61%，共 242 段控球、平均 14.1 秒；v3 是 238 段、14.5 秒，和 Smart Tagging 的吻合程度两版基本一样。
```bash
python -m possession.seq_cv --tag g3 --epochs 6
```
GRU 的考场，在 GPU 上约 5 分钟，平均 0.7012；重跑一次，7 场的概率逐个相同。LightGBM 的考场还是 `python -m possession.cv --exam-controlled 2 --decoder 2,64,75 --extra-features 1 --copies 2`（平均 0.6592）。合用的分数用 `python -m possession.seq_cv --blend v3o g3` 算，但其中 LightGBM 的考场概率文件 `v3o` 目前是用一个临时脚本生成的，还没放进仓库。

测试用 `python -m pytest possession/tests`，81 个，需要另装 pytest；没装 PyTorch 时 GRU 相关的 4 个会跳过。

### 提交前还要做
- [ ] 提交（commit）并推送 v4 的代码和模型文件。
- [ ] 检查仓库里没有比赛数据，再把仓库改成公开。
- [ ] 写技术文档。
- [ ] 把 CSV 和文档按队名命名（`<team_name>.csv`、`<team_name>.pdf`），在截止前用报名邮件里的表单提交。

---

## 0.1 2026-11-14 更新：v3（已被 v4 取代）

下面是 v3 当时的记录。

### 现在提交用的是什么
- **模型**：LightGBM，84 个基础特征加 25 个新特征（`features.build(extra=1)`）。新特征主要是"之前 / 之后"：过去 5 秒谁离球近、接下来 5 秒两队往哪个方向推进，另外还有 ±20 秒的长窗口和逼抢速度。
- **训练**：每场德甲做 2 份弄脏副本，每份配一份主客互换副本；7 场全部用上，固定训练 126 轮。
- **解码**：Viterbi，`team_cost=2, dead_cost=64, min_len=75`（每段最短 3 秒）。
- **开球规则**：一个半场开球之前，球不可能在比赛中，所以判 D。开球时刻取自 Smart Tagging 的 "Kick Off (Start)"，代码里不写帧号（`predict.pre_kickoff`）。这条规则只改动了下半场开球前那段：判成 D 的比例从 52% 升到 97%。

### 怎么选出来的
- 考场是 [`possession/cv.py`](possession/cv.py)：7 场德甲轮流当考题，其余 6 场训练。考题答案按 2 秒"受控控球"口径改写（[`possession/labels.py`](possession/labels.py)：对方控球不满 2 秒不算换控球），用官方 spellF1 打分。考题的输入有指纹，换了考题会报警告。
- 每个改动必须 7 场里至少赢 5 场、平均至少涨 0.005 才采用。
- 公开 5 分钟没有参与任何选择，只在最后对最终版本检查了一次。

| 版本 | 7 场考场 spellF1（2 秒口径） | 公开 5 分钟 spellF1 / macroF1 |
|---|---|---|
| 官方 baseline_smooth50 | 0.195 | 0.326 / 0.621 |
| v2（11-13，解码 2/32/50） | 0.598 | 0.638 / 0.882 |
| **v3（当前）** | **0.642** | **0.676 / 0.903** |

**试过但没有采用**（每项都是和当时最好的版本比，结果在括号里）：
- 训练 300 轮（+0.002）；
- 叶子数 127（−0.001）；
- 学习率 0.025 训练 400 轮（+0.001）；
- 3 个模型取平均（+0.001）；
- 3 份弄脏副本（−0.003）；
- 第二批"之前 / 之后"特征（+0.004，未过门槛）；
- 球的前进方向特征（−0.007）；
- 更强的正则化（−0.006）；
- 训练答案改用 1 秒或 2 秒口径（−0.003 和 −0.009）；
- 死球单独设最短段长（没有提升）。

### 复现
```bash
python -m possession.train
```
约 1.5 分钟。重新训练两次，`lgbm.txt` 和 `decoder.json` 逐字节一致。
```bash
python -m possession.predict
```
约 6 秒，输出 `evaluation/results/lgbm_viterbi.csv`，sha256 以 `7d9ca13e46a9c78f` 开头。
```bash
python -m possession.cv --exam-controlled 2 --decoder 2,64,75 --extra-features 1 --copies 2
```
考场，约 3 分钟，平均 0.6422。测试用 `python -m pytest possession/tests`，44 个，需要另装 pytest。

---

## 1. 当前结果

方法：**LightGBM 逐帧打分 + Viterbi 分段**。训练只用其他比赛（DFL 德甲公开数据），比赛本场的任何标签都不参与训练或调参。

| 版本 | DFL 验证集 spellF1 | 公开 5 分钟 spellF1 / macroF1 | 比赛上预测的 D 占比 |
|---|---|---|---|
| baseline_smooth50（官方） | — | 0.326 / 0.621 | — |
| v1：第一版 | 0.504 | 0.553 / 0.827 | 55% |
| **v2：修复死球偏多（当前）** | 0.494 | **0.638 / 0.882** | 46% |

注意：
- 公开 5 分钟只有 17 段控球，分数波动很大，**只能用来检查流程对不对，不能用来选版本或调参**（规则明确禁止）。
- DFL 验证集上 0.504 和 0.494 的差距在噪声范围内（spellF1 的噪声带约 0.04）。
- 预测端已验证可复现：同一个模型连跑两次，CSV 的 sha256 完全一致。**训练端（重新训练是否得到完全相同的模型）还没验证。**

---

## 2. 做过的事（按时间顺序）

1. **熟悉仓库和规则**：跑通官方 scorer 和 baseline，确认本地数据和官方一致（重新生成的 baseline CSV 与仓库里的版本一致）。
2. **调研**：检索了 arXiv 和 GitHub 上的相关论文和代码（见第 7 节）。最大的发现是 DFL 公开数据集每帧都带 `BallPossession` 和 `BallStatus`，标签的产生方式和本题 GT 几乎一样，可以合规地用来做监督训练。
3. **搭建流程**（`possession/`）：
   - `atd.py`：读取比赛的 ATD，转成统一格式（以中圈为原点的米制坐标）。
   - `dfl.py`：读取 DFL 的 positions XML，生成每帧 H/A/D 标签。
   - `augment.py`：给干净的 DFL 数据加噪，让它像比赛的 ATD。
   - `features.py`：84 个特征，和球员 slot 的编号无关。
   - `viterbi.py`：带最短持续时间和切换惩罚的 Viterbi 解码。
   - `train.py`：在 5 场 DFL 上训练，用 2 场选解码参数，最后用全部 7 场重新训练。
   - `predict.py`：对比赛数据预测，写出提交 CSV。
4. **第一版结果（v1）**：DFL 验证集上，只取 argmax 时 spellF1 是 0.11（约 2,800 段碎控球）；加上 Viterbi 后升到 0.50。
5. **诊断死球偏多**（`diagnose.py`）：只对比输入数据的分布，不看比赛标签，找到两个原因：
   - ATD 测出的球员速度比 DFL 低约 30%（中位数 1.47 对 2.14 m/s），模型把"跑得慢"当成死球信号。
   - DFL 在球出界后还继续跟踪（20% 的球帧在场外），ATD 基本跟丢（4%）。
6. **修复（v2）**：速度特征按半场中位数归一化；加噪时把出界的球一律挖掉。比赛上预测的 D 占比从 55% 降到 46%，正常比赛的死球大约占 40–45%。

---

## 3. 方案细节

### 3.1 训练数据：Sportec Open DFL
- 7 场德甲和德乙比赛，25 fps，TRACAB 光学追踪，许可证 CC BY 4.0（使用时需注明 DFL 并引用 Bassek et al. 2025）。
- 标签：`BallStatus == 0` 记为 D；否则按 `BallPossession` 记为 H（1）或 A（2）。
- **口径差异**：DFL 每场大约 400 段控球，平均每段约 7 秒；公开 GT 那 5 分钟里平均每段约 14 秒。DFL 对"控球"的定义可能更细，这一点还没处理（见计划）。

### 3.2 加噪：让 DFL 看起来像 ATD
这些统计量都**只来自比赛 ATD 的输入数据**，没有用到标签：
- **球是整段丢的**，不是零星丢帧：ATD 里丢球空档的中位数约 10 秒，最长超过 3 分钟，评分帧里有 41% 看不到球。加噪时，按 ATD 实际的"可见段 / 丢失段"长度分布交替挖空。
- 出界 1 米以外的球一律挖掉（ATD 在球出界后基本跟丢）。
- 每个球员按段随机消失，约 7% 的帧不可见（ATD 平均能看到 20.5 人）。
- 坐标加高斯噪声：球员 0.15 m，球 0.3 m。
- 交换主客队再做一份副本（标签 H 和 A 互换），让模型对两队对称。

### 3.3 特征（84 个）
- 每个半场把方向统一成"主队防守 -x 方向的球门"。
- 球：是否可见、距上次看到 / 下次看到各有多少帧、插值后的位置、离边线和底线的距离、是否出界、速度（按半场中位数归一化）。
- 球员和球的关系：每队离球最近和第二近的距离、5 米和 15 米内各有几人，以及两队之差。
- 阵型：两队的重心、离散程度、x 方向的最前和最后位置、向前推进的速度、可见人数。
- 速度：每队的平均速度和最大速度（按半场中位数归一化）。
- 时间上下文：关键特征在 ±1 秒、±3 秒、±10 秒窗口里的均值，以及前后 2 秒的变化量。

### 3.4 模型和解码
- LightGBM 多分类（H/A/D），`deterministic=True`，固定种子和线程数。训练时每 5 帧取 1 帧。
- Viterbi：每个状态展开成长度为 `min_len` 的链，保证一段至少持续这么多帧；H↔A 切换的代价为 `team_cost`，活球↔死球切换的代价为 `dead_cost`。
- 当前参数（在 DFL 验证集上用官方 spellF1 选出）：`team_cost=2, dead_cost=32, min_len=50, prior_power=0`，存在 `possession/model/decoder.json`。

---

## 4. 在游戏本上复现

比赛数据**不在 git 里**（规则禁止再分发，`.gitignore` 已排除 `data/`），需要另外拷贝过去。

### 4.1 准备环境
```bash
git clone https://github.com/tongzhou1215/more-than-a-hack-2027.git
```
```bash
cd more-than-a-hack-2027
```
```bash
pip install -r possession/requirements.txt
```
推荐 Python 3.12，版本固定为 numpy 2.4.4、pandas 3.0.2、lightgbm 4.7.0。版本不一样时，结果可能不会逐字节一致。

### 4.2 放数据
1. 把比赛数据拷到 `data/tactical/`，文件名保持原样。Smart Tagging 文件要命名为 `BAR-ATH_tactical_smart_tagging.xml`。
2. 下载 DFL 数据（约 2.6 GB）。默认放在仓库**同级**的 `dfl_data/` 目录；想放到别处，可以设置环境变量 `DFL_DATA_DIR`：
```bash
python -m possession.download_dfl
```

### 4.3 跑起来
```bash
python -m possession.train
```
```bash
python -m possession.predict
```
```bash
python evaluation/validate_submission.py evaluation/results/lgbm_viterbi.csv
```
```bash
python evaluation/run_evaluation.py
```
- 训练在 i7-1355U 笔记本 CPU 上大约 7 分钟（含 54 组解码参数搜索），预测大约 40 秒。
- 第一次运行会把解析结果缓存到 `data/cache/`（已被 gitignore），之后会快很多。
- 想检查数据分布是否对齐，可以运行 `python -m possession.diagnose`。
- 只想用仓库里已有的模型出结果，跳过 `train`，直接运行 `predict`。

### 4.4 GPU 检查（只有做 YOLO 或深度模型时才需要）
```bash
nvidia-smi
```
当前这条流程**不需要 GPU**。

---

## 5. 接下来的计划

按优先级排列。截止日是 11 月 17 日。

### 必须做（提交前）
- [ ] **验证训练端复现**：在游戏本上从头运行 `train` 和 `predict`，确认 CSV 的哈希和这台机器上的一致。如果不一致，最终提交时以仓库里的模型文件为准，并在文档里写清楚。
- [ ] **把仓库改成公开**：规则要求提交公开仓库。改成公开前，再检查一遍里面没有比赛数据。
- [ ] **写技术文档**：方法、用了哪些数据（DFL，注明来源和许可）、控球规则、局限、复现步骤。可以直接从这份文档改写。
- [ ] 提交时把 CSV 和文档按队名命名：`<team_name>.csv`、`<team_name>.pdf`。

### 值得做（提分）
- [ ] **扩大解码参数网格**：`team_cost=2` 落在网格边缘，可以加上 1 和 0.5。
- [ ] **处理控球口径差异**：DFL 的控球段比本题更碎。可以试一条算法规则：夹在同一队两段控球之间、短于 N 秒的对方控球（比如折射、未控制的触球）并回原控球方，再用这样处理过的标签训练。N 要按 CHALLENGE.md 的"受控"定义来定，不能用公开 GT 调。
- [ ] **Smart Tagging 先验**：用 SET PIECES 等片段的时间点辅助判断死球和重新开球。它不能参与训练（DFL 没有这个数据），但可以放进解码里作为先验。
- [ ] **剩余的死球偏差**：ATD 上球不见时 P(D) 是 0.81，DFL 上是 0.46，还没完全对齐。
- [ ] 加入 SkillCorner 开放数据（10 场比赛，球员缺失更多，噪声更接近 ATD）一起训练。

### 有 GPU 和余力时再做
- [ ] **YOLO 补球**：只在 ATD 丢球的帧上，围绕上一次的球位置裁剪画面做检测。用公开数据集（SoccerTrack v2、SoccerNet）在云 GPU 上微调。比赛视频是否可以上传到云端，要先看报名条款；最稳妥的是只上传公开数据训练，把权重下载回本地推理。
- [ ] 参考 PathCRF 的 CRF 层，或者用 Ball Radar 补全球位置。

---

## 6. 规则红线（摘自 RULES.md）

- ✅ 任何方法都可以；可以用其他比赛的人工标注；可以在本场比赛上做无监督学习和算法伪标签。
- ❌ 不能对本场比赛做任何人工标注或人工修正。
- ❌ 不能用公开 5 分钟 GT 训练或调参，只能用来检查流程。
- ❌ 不能用本场比赛的任何人工数据（Opta、StatsBomb 等）。
- ❌ 不能写死和本场具体帧号绑定的规则。
- 仓库必须能重新生成**完全相同**的提交文件。

---

## 7. 算力选择

| 方案 | 需要什么 |
|---|---|
| 当前流程（LightGBM + Viterbi） | 笔记本 CPU 就够 |
| YOLO 补球、PathCRF | RTX 4090（24 GB）一类的卡；游戏本的 8 GB 显存可以做推理，微调比较吃紧 |
| 视频基础模型 | A100 或 H100，4 天内不现实 |

- **Colab**：官方的 `google-colab-cli` 只支持 Linux 和 macOS，Windows 上要通过 WSL（Ubuntu）或 Docker 来用，第一次登录需要本人在浏览器里完成 Google OAuth。VS Code 的 Colab 扩展只能连接 notebook，没有命令行接口。
- **租卡**：国内可以考虑 AutoDL、恒源云，海外有 RunPod、Vast.ai。SSH 连接最方便；训练完先下载权重再关机，避免空转计费。

---

## 8. 参考资料

| 来源 | 用途 |
|---|---|
| [PathCRF](https://arxiv.org/abs/2602.12080)（KDD 2026，[代码](https://github.com/hyunsungkim-ds/pathcrf)） | 不用球位置推断控球路径，CRF 加 Viterbi |
| [Ball Radar](https://arxiv.org/abs/2306.08206)（KDD 2023，[代码](https://github.com/hyunsungkim-ds/ballradar)） | 从球员轨迹推断球的位置 |
| [Vidal-Codina et al. 2022](https://arxiv.org/abs/2202.00804) | 基于规则的控球和事件检测 |
| [Peral et al., WACV 2026](https://arxiv.org/abs/2512.19528) | tactical 机位下的球状态和控球者识别 |
| [Yamamoto et al. 2023](https://arxiv.org/abs/2308.05460) | 控球时长的统计分布，可作解码先验 |
| [ELASTIC](https://arxiv.org/abs/2608.30227)（[代码](https://github.com/hyunsungkim-ds/elastic)） | 从轨迹检测触球候选帧 |
| [Sportec Open DFL](https://github.com/spoho-datascience/idsse-data) | 训练数据，CC BY 4.0 |
| [SkillCorner Open Data](https://github.com/SkillCorner/opendata) | 备选训练数据 |
| [Metrica Sample Data](https://github.com/metrica-sports/sample-data) | 和比赛同一格式的样例数据 |
| [SoccerTrack v2](https://github.com/AtomScott/SoccerTrack-v2) | 全景视频和触球标注，可用于 YOLO 微调 |
