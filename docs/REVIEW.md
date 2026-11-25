# `meta-rsi` 模拟顶会审稿意见

> - 日期：2026-11-25
> - 审阅对象：`main` @ `fe78c98`，以及本地的 `runs/`、`data/` 产物
> - 视角：NeurIPS / ICLR 主会审稿人
> - 核验方式：文中数字均用仓库自带脚本从本地 `runs/` 重新计算。`runs/` 和 `data/` 已被 gitignore，文中引用的 `runs/...` 路径只在本机存在。
>
> ```bash
> python -m meta_rsi.eval.analysis runs/eval480 --ood-dir runs/eval_ood
> python -m meta_rsi.eval.sprint4_summary
> nix-shell --run "python -m pytest -q tests"   # 33 passed
> ```

## 结论

按现在的样子投 NeurIPS/ICLR 主会，我会给 **3/10（Reject）**，置信度 4/5。

工程量和 benchmark 设计确实是亮点。但核心卖点"label-free 元变换门控打破 Proxy Lock-In"目前站不住，原因有四：

- 验证器是照着隐藏测试变体造的；
- 头条结果有领域提示泄漏；
- 统计检验大多不显著；
- 关键对照缺失或有混杂。

**评分：** Soundness 2/4 · Presentation 2/4 · Contribution 2/4 · Overall 3/10（Reject）· Confidence 4/5

## 我对这篇论文的理解

- **问题：** 自进化 agent harness 时，如果 in-loop 验证器很弱（只在 `low` 纬度检查答案），进化会锁定 EPSG:3857 这类捷径。作者称之为 Proxy Lock-In。
- **方法：** 用重定位、CRS 翻转等不改变答案的元变换做门控；再让 LLM 自己提出新的 probe（verifier 共进化）；最后把轨迹蒸馏，加上 GRPO，写进权重（Two-Timescale）。

## 优点

1. 问题有价值。作者找到了一个确定、可审计的实例（投影畸变），lucky 和 silent_wrong 的划分很清楚。
2. GSB v1/v2 附带方法可分性审计（reference、UTM、3857 各自的通过率），这是整个仓库最扎实的部分。
3. 基线覆盖面广：naive、preserve、darwinx_soft、llm_judge、oracle、handcrafted、runtime_meta。还测了 BF16 抖动，做了 `usage.errors` 审计，维护了风险表。研究规范意识高于平均。
4. "二阶锁定"是真正可推广的发现：`meta_T0` 在 6/6 个种子上都卡在 `estimate_utm_crs` 的极地边界；SciUnits 里也出现了同样性质的单位覆盖缺口。

## 主要问题

### M1. 验证器和测试 oracle 是同一个结构，"label-free"名不副实（最致命）

- 门控用的两个 probe 在 (58.3°N, 6.0°E) 和 (−56.2°, 12.0°)，见 [evaluator.py:90](../meta_rsi/evaluator.py#L90)。
  - 这两个纬度落在 v1 `high` 变体的区间（52–64°）。
  - 6°、12° 恰好是 `zone_edge` 变体使用的 UTM 带边界，见 [gsb/geo.py:22](../benchmarks/gsb/geo.py#L22)。
  - [metamorphic.py:6](../meta_rsi/metamorphic.py#L6) 甚至直接写成 `AEQD(high_lat=58.3, zone_edge_lon=6.0)`。
- GSB 的真值按构造在各个位置都相同。所以"在 low 上答对，并且在 T(x) 上结果一致"几乎就等于"oracle 判对"。Dev Gap 只有 2.8% 是构造决定的，不是方法发现的。
- 后续阶段同样的模式又出现了：
  - v4 共进化"自主发现"的 probe 在 86.5°±0.3°N（`runs/v4_coev_polar/certified_probe.py:24`）。Polar OOD 测试也恰好在 86.5°N（[eval_ood.py:26](../meta_rsi/eval/eval_ood.py#L26)）。
  - GSB v2 的极地 probe 在 86°N，测试的极地变体是 85.3–87.2°（[gsb_v2/geo.py:30](../benchmarks/gsb_v2/geo.py#L30)）。
  - probe 一加进来，所谓的 OOD 测试就变成了同分布测试。
- 这个"发现"受到了很强的引导：
  - verifier 的 prompt 明说已有 probe 是"Mid-latitude … (58.3N, 6.0E)"（[verifier_agent.py:652](../meta_rsi/verifier_agent.py#L652)）。
  - 共进化是在作者看到极地测试失败之后才启动的。
  - 被"发现"的 probe（极地、反子午线、3857 投影的 GPKG、km 单位）都是 GIS 教科书里的经典陷阱，也正是作者设计进 GSB v2 的变体。
  - 现有实验分不清两种解释：方法能发现未知的失效模式，还是元 LLM 的 GIS 先验恰好和作者的 benchmark 设计重合。
- 认证器只检查几何量是否守恒，不检查任务答案在 T 下是否真的不变。凡是依赖绝对位置的真实任务都会被错误认证。

### M2. README 主表（v0.2）有领域提示泄漏，而且一半种子里门控根本没起作用

- v0.2 的诊断原文（`runs/v3_gis_s0_meta/gen_000/dev_eval.json`）是："The script likely uses a distorted projection (e.g. EPSG:3857), hardcoded UTM zone, degree-based geometry/NN, swapped lat/lon, or ignores layer CRS."
  - 同时暴露了变换名 `local_aeqd_shift`。
  - 当时的 meta prompt 还把可用库限定为 geopandas/pyproj/shapely（[meta_agent.py:22](../meta_rsi/meta_agent.py#L22)，v0.2 提交 `0fb5457` 中相同）。
  - 可 README 仍把这张表当主结果，还称方法是"domain-neutral"。
- s0、s2 这两个 metamorphic 种子的 4 个候选全部被接受，门控一个都没拒。这两个种子的提升只能来自带提示的诊断文本和最终选择。
- 去掉泄漏后重做的实验（v4_clean，3 个种子，`runs/v4_wave1_evals/`）：

  | 方法 | 均值 | 各种子 |
  |---|---|---|
  | `meta_neutral`（完整方法） | 74.7% | 84.0 / 84.6 / 55.6 |
  | `naive` | 75.2% | 74.8 / 75.6 / 75.2 |
  | `darwinx_soft` | 76.9% | 78.1 / 75.8 / 76.9 |
  | `meta_gate_only` | 80.8% | 81.7 / 78.5 / 82.3（比 darwinx_soft 高 3.9pp） |
  | in-loop 真 oracle 门控 | 69.0% | 51.2 / 73.8 / 82.1 |

  - 连完美验证器都不能稳定带来收益，说明结果主要由 Meta-Agent 的随机性决定，而不是验证器质量。
  - 在严格弱代理（eq2，|lat| ≤ 2.2°）下，naive 两个种子都没有锁定（77.1 / 76.2）。锁定只出现在 2 个 darwinx_soft 种子中的 1 个。Proxy Lock-In 现象本身也不够稳健。

### M3. 统计证据弱，报告有选择性

- **v0.2：** metamorphic 对 naive，ALL 高 11.6±24.9pp，只在 2/4 个种子上胜出，p=0.42；README 只写了"Lucky 4/4 胜"。对 preserve 的 p=0.09。
- **GSB v2 六种子：**
  - `arch_coev` 对 `darwinx_soft` 的 p=0.15（置换检验 0.19），对 `llm_judge` p=0.88，对 `coev_tip_only` p=0.49。
  - 唯一"显著"的是对 `meta_T0`（作者自己的 Stage-1 消融）的 t 检验 p=0.021。同一个脚本算出的精确置换检验是 p=0.0625，README 只报了前者。
  - `arch_coev` 的 41.0%，低于只跑了单种子的 naive 46.9%、meta_gate_only 53.6%、oracle_soft 55.3%、runtime_meta 57.1%。最强的几个基线只跑了 1 个种子，弱基线跑了 6 个。"leads all 6-seed methods"这个说法是统计口径造成的。
- **81.9% 的"best harness"：** 是按测试分数从 12 个 harness 里挑出来的，比较对象又是单次 zero-shot 的 26B/27B。一边是多轮加执行反馈，一边是单次生成。
- **噪声：** temp=0 时 dev 上的翻转率是 22%（8/36，`runs/v4_jitter_check.json`），可每个 harness 只在测试集上评了一次，没有重复评测的方差。dev 上"+4/−2"这种接受决策落在噪声范围内。
- **其他与文字不符的结论：**
  - 按当前 `analysis.py` 的修正分组，metamorphic 在非投影族上是 60.0%，低于 naive 的 62.5% 和 preserve 的 65.4%，"zero skill contamination"不成立；RESEARCH 里还是旧表。
  - Polar OOD 上 metamorphic 的正确率最低（3.2%，naive 24.1%，preserve 21.8%），却被写成"fails loudly"的优点。
  - SciUnits 上是 10.4±7.5% 对 7.1%（p=0.54），却被写成"autonomous discovery"。

### M4. Two-Timescale 的对照有混杂

- SFT-Weak 和 SFT-Meta 不只是过滤器不同，教师也不同（[collect_v2_trajectories.py:35](../scripts/wave_s4_2/collect_v2_trajectories.py#L35)）：
  - weak 的教师是 darwinx_soft harness，v2 测试 32.2%。
  - meta 的教师是 stage2_coev，v2 测试 62.9%。
  - v1 也一样：`sym26b_s1_darwinx_soft_gen004`（20.4%）对 `sym26b_s1_meta_neutral_gen005`（86.2%）。
  - 两者的差距完全可以由教师质量解释。
- 缺的对照：同一教师配弱过滤器；同一教师不过滤；用 oracle 过滤的上界；从 SFT-Meta 出发做 GRPO-Weak；用同样计算量继续做 SFT。
- 全部只跑了一次：
  - v1 的教师是两个种子里测试 86.2% 的那个（另一个是 42.7%）。选择规则没写，s0 的 archive 也不在 `runs/` 里。
  - v2 的教师恰好是 6 个种子里最好的那个（62.9%，6 种子均值 37.1%）。
- GSB v2 的管线号称"100% open-weight"，可轨迹过滤和 GRPO 用的 3 个 probe 来自闭源的 gemini-4-hydrogen。
- "Gen-2 Compounding RSI"的 harness 只有 1 次被接受的变异（36 道 dev 题上 +4/−2，在抖动范围内），内容是通用的报错重试。收益几乎都来自错误率下降（3.8%→0.4%，7.6%→0.7%）。叫它"递归自我改进的复利"言过其实。

### M5. 外部有效性证据不足

- **GeoAnalystBench：**
  - n=17，评测是纯静态的 AST/字符串检查，不执行代码，也没用这个 benchmark 自己的评测方式。
  - `dynamic_crs` 只判断代码里有没有 `estimate_utm_crs` 这类写法（[geoanalyst_eval.py:127](../meta_rsi/eval/geoanalyst_eval.py#L127)）。这恰恰是训练数据教给模型的写法，测到的是风格迁移，不是正确性。
  - "+71.4pp"实际上是 0/7 变成 5/7。
- **OSM：** 6 个题目模板乘 4 座城市，有效样本大约只有 6；91.7% 和 100% 之间只差 2 道题。

### M6. 评测协议本身的效度

- agent 只看到 `low` 工作区，生成的脚本再被搬到它看不到的 high、zone_edge、polar 上运行，可 prompt 从来没要求写与位置无关的代码。
  - "lucky"和"silent error"是相对隐藏输入定义的；在给定数据上，3857 的结果其实落在 1% 容差内。
  - 需要论证这个威胁模型（比如脚本会被复用），并把 agent 能看到真实数据的 Protocol B 作为主结果。目前只对少数 harness 做过。
- 考察的能力面很窄，基本就是"选对度量投影"这一项。一个 15 分钟手写的 harness 在 v1 上就有 74.2%。

## 次要问题

- **术语：**
  - 只有 4–8 代、1–6 次接受的 scaffold 搜索，被叫作 RSI。
  - 号称 label-free，但 in-loop 弱检查（[evaluator.py:292](../meta_rsi/evaluator.py#L292)）和 GRPO 的 reward（[split_two_timescale.py:97](../meta_rsi/training/split_two_timescale.py#L97)）都用了 `low` 变体的真值。
  - "true OOD"、"breakthrough"等措辞过强；"inverse scaling"的结论只基于 2 个种子。
- **文献：**
  - 主基线 DarwinX 只读了摘要（见 `LITERATURE.md` 的待读清单），代码注释里却引用了它的"§2.1"（[archive.py:82](../meta_rsi/archive.py#L82)）。
  - 缺少几类相关工作：面向 ML/LLM 代码的元变换测试、CodeT/执行一致性、property-based testing、reward over-optimization（Gao et al. 2023），以及 STaR/ReST/RFT。
- **可复现性：**
  - `runs/` 和 `data/` 都被 gitignore，git 只跟踪了 76 个文件，所有"Verified"的数字在公开仓库里都没法核验。
  - Meta 模型是闭源的预览代号。
  - `pyproject.toml` 缺 rasterio、torch 等依赖，版本号写的是 0.2.0，文档里已经是 v0.7。
- **计算量没有对齐：** metamorphic 在 dev 上每道题要多跑几次执行，测试时平均调用次数也是 1.50 对 1.30。
- **写法：** README/RESEARCH 满是加粗和感叹号，读起来像实验日志而不像论文；v0.2 和 v4 两套数字并存，容易误导。

## 给作者的问题

1. 如果测试的失效模式由不知道 verifier 设计的第三方给出，方法还剩多少收益？
2. v0.2 去掉提示后重跑是什么结果？为什么 README 仍用它做主表？
3. naive、meta_gate_only、oracle_soft、runtime_meta 能否在 GSB v2 上补满 6 个种子，并对齐计算量？
4. 同一教师、只换过滤器的 SFT 消融结果如何？
5. verifier 发现这一步能否按种子重复？gemini 一侧一共认证了 5 个 probe，被舍弃的反子午线 probe（r1、r4）是按什么规则舍弃的？
6. 同一个 harness 在 eval450 上重复评测，方差有多大？

## 什么能让我改分

- 拿出盲设计协议下的结果，或者真实、非构造任务上的结果。
- 给一张统一的主表：GSB v2 上所有方法都跑至少 6 个种子，计算量对齐，用配对置换检验加多重比较校正，每个 harness 评测多次。
- 撤下 v0.2 的结果，或者明确标注提示泄漏。
- 补上教师匹配的蒸馏消融。

## 重新定位的建议

- **方向 (a)：** 投 Datasets & Benchmarks track，主贡献是 GSB v1/v2、可分性审计和静默错误分类。
- **方向 (b)：** 写成实证分析论文，主题是"任何固定的验证器都会在自己的覆盖边界上诱发二阶锁定"（`meta_T0` 6/6 个种子、SciUnits 的单位缺口）。这个发现比"我们的门控赢了"更可信，也更通用。
