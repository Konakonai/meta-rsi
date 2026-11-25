# Literature Survey & Positioning Matrix

> Living document for Related Work and manuscript drafting.
> Last updated: 2026-11-20.
> Verification tag legend: `[full-text]` = full paper read; `[abstract/summary]` = verified from abstract, arXiv metadata, or search summary (re-verify exact numbers before manuscript citation).

---

## 1. Recursive Self-Improvement (RSI) & Self-Evolving Agent Harnesses

| Work | Venue / Date | Editable Target | Verification / Selection Mechanism | Key Finding & Limitation | Status |
|---|---|---|---|---|---|
| **ADAS** (Hu et al.) | ICLR 2025 (2024) | Agent Python code (`forward()`) | Archive of discovered agents + ground-truth eval | Automated design of agentic systems outperforms hand-designed scaffolds; relies on full benchmark labels in-loop. | `[abstract/summary]` |
| **Darwin Gödel Machine (DGM)** (Sakana AI & UBC) | arXiv 2025-05 | Python coding agent codebase | Empirical benchmark archive + open-ended Darwinian parent selection | Self-modifying coding agents improve across generations via lineage archive; assumes reliable unit-test verifiers. | `[abstract/summary]` |
| **Hyperagents (DGM-H)** | `arXiv:2603.19461` (Meta FAIR & UBC, 2026-03) | Task Agent + Meta Agent merged into one editable program | Empirical archive + metacognitive self-modification | Moves from bounded harness search to true recursion (improving how the agent improves itself). | `[abstract/summary]` |
| **RHI (Recursive Harness Self-Improvement)** | `arXiv:2607.15524` (2026-07) | Agent harness wrapping low-effort models | Iterative harness refinement | Low-reasoning-effort models + evolved harness beat high-reasoning baselines while cutting token cost up to 60%. | `[abstract/summary]` |
| **DarwinX** (Yifan Zhang et al.) | `arXiv:2608.07545` (2026-08) | Harness on frozen base model | Population lineage archive, recombination, **Preserve-and-Extend** contract, label-free `avg@k` native verifier | Strong gains without weight updates; exposed a critical in-loop vs held-out gap on TerminalWorld (`1.000` in-loop proxy vs `68.3%` held-out). | `[abstract/summary]` |
| **When Self-Evolution Backfires** | `arXiv:2608.05810` (2026-08) | Self-evolved skills/rules | Verifier-as-Gatekeeper (VaG) pre-commit screening | Documents non-monotonic **Skill Contamination** chains where newly added rules degrade previously solved tasks. | `[abstract/summary]` |
| **HarnessLens** | `arXiv:2608.27311` (2026-08) | Agent harness | Behavior-aware sparse verification + attributable-evidence gating | Reduces evaluation cost while preserving signal quality during harness search. | `[abstract/summary]` |
| **RSIGym** | 2026-10 | `Data` / `Harness` / `Joint` tracks | EaaS RSI environment + `RSI-Index` | Formalizes evaluation of RSI systems; documents budget hacking and circular optimization failure modes. | `[abstract/summary]` |
| **Double Ratchet** (*Who Grades the Grader?*) | `arXiv:2607.12790` (2026-07) | Natural-language skill library + drawback-detector metrics | Locked human-labeled **Anchor Set** (`Anchor Guards`) | Co-evolves metrics and skills under a frozen LLM; removing human-labeled anchor guards collapses evolved metrics into degenerate states. | `[abstract/summary]` |
| **Verifier Hardening** (*Scalable Weak-to-Strong Verifier Hardening*) | `arXiv:2610.15924` (2026-10) | Execution sandbox & verifier logic | Hacker-fixer loops + **known-solution replays** | Closes reward-hacking loopholes in agent environments, but requires per-task known gold solutions to verify patches do not reject valid solves. | `[abstract/summary]` |

**Core RSI takeaway for `meta-rsi`**:
Existing harness-evolution systems assume either a strong/unhackable in-loop verifier or rely on strict regression gates (`Preserve-and-Extend`). When the in-loop verifier is *weak* (admitting silent/spurious passes), our v0.2 experiments expose a counter-intuitive failure mode — **Proxy Lock-In** — where `Preserve-and-Extend` locks in early shortcut hacks and performs *worse* than naive hill-climbing.

---

## 2. GeoAI Agents, Executable Benchmarks & Geospatial Silent Errors

### 2.1 Geospatial Tool-Use Agent Lineage (GeoGPT → GTChain → MiniGeoSolver)

| Work | Venue / Year | Approach | Evaluation Paradigm | Relevance to `meta-rsi` | Status |
|---|---|---|---|---|---|
| **GeoGPT** (Zhang et al.) | *JAG* 2024, 131:103976 (324+ cites) | LangChain + GPT-4 + fixed GIS tool descriptions | Case studies + workflow evaluation | Originating GIS agent harness; hand-written static scaffold with closed LLM. | `[full-text]` |
| **MapGPT** | *CaGIS* 2024 | LLM cartographic workflow agent | Case studies + map outputs | Extends static GIS harness to cartography. | `[abstract/summary]` |
| **GeoTool-GPT** | *IJGIS* 2024 | SFT on LLaMA-2-7B (GeoTool / GeoSolution / GeoTask) | Tool-use instruction datasets | Moves GIS tool-use from closed APIs to open-weight 7B models. | `[abstract/summary]` |
| **GTChain** (Zhang et al.) | *JAG* 2024, 104312 | Simulated self-instruct tool chains + open-weight SFT | 500-item text workflow comparison (`AGI-GIS/GTChain`) | Textual workflow matching with virtual paths (no actual execution); `meta-rsi` advances to execution-verified self-evolving harnesses. | `[abstract/summary]` |
| **VGI Mapping Agent** (Song et al.) | *JGGS* 2025 | Prompted LLM agent over OSM VGI data | End-to-end mapping cases | Demonstrates OSM data retrieval + mapping workflow construction. | `[abstract/summary]` |
| **MiniGeoSolver** | *EMS* 2026 | Two-stage SFT on Gemma-2-2B | Tool selection + workflow accuracy | Compresses GIS workflow planning to 2B models; aligns with our small-model (4B) + harness / weight-internalization goal. | `[abstract/summary]` |
| **Text-to-SpatialSQL** (Ma et al.) | *Transactions in GIS* 2026 | Manual-extracted Geo-KG + RAG for spatial SQL | Execution accuracy (EX) & component match (CM) | Shows explicit rule/knowledge retrieval helps spatial queries; motivates comparing hand-written rules vs self-evolved harnesses. | `[abstract/summary]` |

### 2.2 2025–2026 Executable GeoAI Benchmarks

The international frontier shifted in 2025–2026 from text-workflow comparison to **actual execution on real spatial data with automated output scoring**:

| Benchmark | Date / Venue | Scale & Tasks | Scoring | Link / Cross-Benchmark Potential | Status |
|---|---|---|---|---|---|
| **GISAgentBench** | `arXiv:2608.01645` (2026-08) | 349 multi-step tasks from real GIS Stack Exchange questions | Automated comparison vs executable ground truth (numeric, categorical, spatial F1) | Best model (Gemini-3.1-Pro) hits only `32.7%` strict success (open models ~half). High-value target for Python subset transfer. | `[abstract/summary]` |
| **GeoAgentBench** | `arXiv:2604.13888` (*GSIS* 2026) | 57 spatial analysis tasks, 133 atomic GIS tools | Dynamic sandbox execution + intermediate checks | Public dataset (`hf: zhangdw/GABench`); multi-turn tool-call format (requires adapter vs single-script GSB). | `[abstract/summary]` |
| **GeoNatureAgent** | `arXiv:2606.12821` (2026-06) | 93 environmental geo-analysis tasks (18 classes) | Rule-based execution scoring (no LLM judge) | Evaluates closed vs open-weight models on self-hosted geo APIs. | `[abstract/summary]` |
| **GeoAnalystBench** | `arXiv:2509.05881` (2025-09) | 50 Python spatial analysis tasks | Workflow & Python code generation | Closest task format to single-script Python analysis; top priority for external transfer check. | `[abstract/summary]` |
| **GeoSQL-Bench** | `arXiv:2509.25264` (2025-09) | 14,178 PostGIS instances, 340 functions, 82 DBs | Execution leaderboard (24 models) | Saturated SQL track; distinguishes our general Python/GeoPandas/GEE agent setting. | `[abstract/summary]` |

### 2.3 Geospatial Silent Errors & Geo-RL

- **GeoContra** (`arXiv:2605.00782`, 2026-05) `[abstract/summary]`: Identifies LLM-written GIS workflows that "run without error yet are geographically wrong" (distorting accessibility/exposure metrics); proposes 3-layer inference-time checking (static analysis, runtime checks, semantic verification). Unlike `meta-rsi`, it performs *post-hoc inference checking* rather than *recursive harness self-improvement*.
- **Bench4GeoCode** `[abstract/summary]`: Notes CRS mismatches and wrong spatial predicates produce silent logic errors without runtime exceptions.
- **Geospatial RL (primarily vision/multimodal)**: **Geo-R1** (`arXiv:2510.00072`, GRPO on Qwen2.5-VL-7B for street/satellite matching), **GeoSolver** (`arXiv:2603.09551`, 9B remote-sensing process-reward model), **GeoX** (`arXiv:2605.20006`, self-play) `[abstract/summary]`. Execution-verified RSI for geospatial *coding agents* remains under-explored.
- **Urban Mobility Validation (adjacent reference)**: *When Plausible Is Not Realistic* (`arXiv:2606.13835`, 2026-06) `[abstract/summary]` echoes the broader GeoAI validation gap between "looks plausible" and "quantitatively valid".

---

## 3. Spurious Correctness, Metamorphic / Property-Based Testing, Reward Over-Optimization & Iterative Self-Distillation

- **Metamorphic Testing (MT) for Software & ML**: Classical software testing paradigm (Chen et al., 1998; Segura et al., 2016; Chen et al., 2018) addressing the *test oracle problem* by checking **metamorphic relations** — domain invariants where transforming input $x \mapsto T(x)$ induces a known relation between $f(x)$ and $f(T(x))$ without requiring ground-truth labels on $T(x)$; extended to ML programs by Murphy et al. (2008) and LLM code evaluation.
  - In `meta-rsi`, Euclidean/ellipsoidal **geometric invariance** under coordinate relocation (AEQD shift / ellipsoidal isometry), storage CRS flip, and row/column permutation turns MT into a **weak-label + metamorphic self-consistency gate** (requiring no additional OOD labels beyond the equatorial dev truth) inside an evolution or RL loop.
- **Property-Based Testing & Execution-Consistency Verification for Code**: **QuickCheck** (Claessen & Hughes, ICFP 2000) formalizes testing code against randomized invariant properties; **CodeT** (Chen et al., ICLR 2023) ranks generated programs via dual execution agreement against LLM-generated unit tests. Unlike CodeT's unconstrained unit tests (which can themselves inherit shortcut assumptions), `meta-rsi` requires proposed workspace transformations $T$ to pass a deterministic geometric conservation certifier before entering the verifier.
- **Reward Over-Optimization & Goodhart's Law**: **Skalse et al. (NeurIPS 2022)** formally characterize *reward hacking* under proxy objectives, and **Gao et al. (ICML 2023)** quantify scaling laws for *reward model over-optimization* in RLHF/RL. Our **Second-Order Proxy Lock-In** finding extends this phenomenon to iterative verifier-guided agent evolution: even when a first-order proxy (`low` accuracy) is augmented with a fixed metamorphic verifier (`meta_T0`), search systematically concentrates at the *coverage boundary* of the augmented verifier (`estimate_utm_crs()` failing polar/antimeridian/large-extent regimes).
- **Spurious Correctness in LLMs**: **Ishii et al. (IJCNLP-AACL 2025)**, *Fine-grained Confidence Estimation for Spurious Correctness Detection in LLMs* `[abstract/summary]`, studies detecting answers that are *correct for the wrong reasons*. Passing an equatorial geospatial task via Web Mercator (`EPSG:3857`) provides a deterministic, auditable instance of algorithmic spurious correctness.
- **Iterative Self-Distillation & RL (STaR / ReST / RFT / GRPO)**: **STaR** (Zelikman et al., NeurIPS 2022), **ReST** (Gulcehre et al., 2023), and **Rejection Sampling Fine-Tuning (RFT)** (Yuan et al., 2023) filter model-generated trajectories by ground-truth outcome correctness before SFT; **GRPO** (Shao et al., 2024) replaces a learned critic with group-relative advantage normalization. Our Two-Timescale ablation tests whether filtering and GRPO reward signals based only on weak outcomes (`low`) vs. weak outcomes + metamorphic invariance induce or mitigate weight-level proxy lock-in under matched teachers and initializations.

---

## 4. Positioning Matrix & Research Gaps

| Dimension | Static GIS Agents (`GeoGPT`, `GTChain`) | Runtime GIS Checkers (`GeoContra`) | Harness RSI (`DGM`, `DarwinX`, `RHI`) | **`meta-rsi` (Ours)** |
|---|---|---|---|---|
| Executes real geospatial code | No (text workflow) / partial | Yes | Domain-general | **Yes (sandboxed execution)** |
| Evolves agent harness automatically | No | No | Yes | **Yes (lineage archive + recombination)** |
| Resists weak-verifier Proxy Lock-In | N/A | Inference-only | Vulnerable (exposed by our `preserve` study) | **Yes (label-free metamorphic gating)** |
| Avoids hand-written rule contamination | N/A | Static rule set | Partial (VaG / HarnessLens) | **Yes (compact modular harness + invariance gate)** |
| Path to true recursion (P1) | No | No | Hyperagents (meta-level) | **Planned: self-discovered invariants + meta-evolution + weight internalization** |

### Priority Reading Checklist (Next Full-Text Reads)
- [ ] **DarwinX** (`arXiv:2608.07545`) — exact Preserve-and-Extend formulation & TerminalWorld gap numbers.
- [ ] **Hyperagents / DGM-H** (`arXiv:2603.19461`) — metacognitive self-modification interface & cross-generation metrics.
- [ ] **When Self-Evolution Backfires** (`arXiv:2608.05810`) — formal definition of skill contamination & VaG baseline.
- [ ] **GISAgentBench** (`arXiv:2608.01645`) & **GeoAnalystBench** (`arXiv:2509.05881`) — task schema and public code/data availability for external transfer evaluation.
- [ ] **GeoContra** (`arXiv:2605.00782`) — exact error taxonomy & static/runtime checks to compare against metamorphic invariance.
