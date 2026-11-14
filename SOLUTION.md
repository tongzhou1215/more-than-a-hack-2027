# 控球判定方案：工作记录与计划

这份文档记录我们在 More than a Hack 2027 Phase 1（逐帧判定控球方）上做过的事、当前结果、怎么在另一台机器上复现，以及接下来的计划。

- 提交截止：**2026-11-17**
- 要超过的官方 baseline：全场 spellF1 **0.2433**（`baseline_smooth50`）
- 代码都在 [`possession/`](possession/) 目录下

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
