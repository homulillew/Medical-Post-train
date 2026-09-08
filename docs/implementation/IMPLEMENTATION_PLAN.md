# Repository Takeover：具体实施计划

日期 2026-09-08。研究问题与 Stage 1–6 主线全部保留。本次只完成规划、源码/环境审计和轻量验证；未开始正式训练。六个 Stage 均为 NOT_STARTED，代码架构章节中的模块和命令是待实现接口。

## 结论与关键选择

本机是单张 RTX 5880 Ada，44.99 GiB 可见显存、约 109.63 GiB RAM。BF16 LoRA 有容量上的可行路径，但没有 8B 实测保证。采用一个 immutable Qwen3-8B snapshot、一个共享 r32 LoRA 定义、Transformers/PEFT SFT，以及 verl 原生 GSPO loss + group-relative advantage。把 DAPO correctness filter/refill 接到同一训练 controller，使 Vanilla 和 Dynamic 只在 sampling intervention 上不同。

控制器以 **成功完成并与 checkpoint 一致的训练组** 为预算，另存所有实际生成成本。两个正式 RL run 各保留 5,000 training groups，Dynamic 必须全为 mixed groups。系统退出、agent 中断或达到 step guard 都不等于验收通过。

## 阅读与证据

已完整阅读 AGENTS、README、七份顶层协议/工作流、六阶段规范、全部八份模板（含 BAD_CASE.json）；[read manifest](evidence/contract-read-manifest.json) 固定各文件 hash 与起始 commit。规范中的正式数据隔离、原始日志、负结果、阶段报告/面试材料都是实现 gate。

- [ENVIRONMENT_AUDIT](ENVIRONMENT_AUDIT.md)：实际硬件、软件、路径与已执行探测。
- [DEPENDENCY_STRATEGY](DEPENDENCY_STRATEGY.md)：候选版本、metadata 解算、ABI 风险与环境冻结步骤。
- [UPSTREAM_FINDINGS](UPSTREAM_FINDINGS.md)：固定源码 API 与失效假设。
- [REPOSITORY_ARCHITECTURE](REPOSITORY_ARCHITECTURE.md)：模块接口、流向、配置与测试。
- [EXECUTION_PLAN](EXECUTION_PLAN.md)：Stage 1–6 数据/训练/评估/serving 的实际参数和入口。
- [COMPUTE_BUDGET](COMPUTE_BUDGET.md)：实机内存算式、GPU-hour 情景和不确定性。
- [CHECKPOINT_AND_RESUME](CHECKPOINT_AND_RESUME.md)：脱离会话的执行、恢复与故障矩阵。
- [EXPERIMENT_DATA_MODEL](EXPERIMENT_DATA_MODEL.md)：run/artifact/state/metrics/case/research memory。
- [RISK_REGISTER](RISK_REGISTER.md)、[OPEN_QUESTIONS](OPEN_QUESTIONS.md)、[DECISIONS](DECISIONS.md)：风险、未决问题与变更决策。

## 实施与验收总表

所有命令中的 `mpt` 是未来安装的 `medical_posttrain.cli:main`；**当前不能运行这些 Stage 命令**。入口的 mode 是 smoke/pilot/full；存储分类按 protocol 映射。

| Stage | 主要模块/配置 | smoke → pilot → full 入口 | 必需交付 / verifier |
|---|---|---|---|
| 1 | data/sft adapters、collator、training/sft；s1.yaml | `mpt sft --mode smoke` → `--mode pilot` → `--mode full` | 20k selection、完整 epoch、loss/coverage、adapter reload；`verify_stage.py --stage 1` |
| 2 | cmexam、reward/*、rollout/*；s2.yaml | `mpt profile --mode smoke` → 可选诊断 → `--mode full` | 15k pool、1000 unique×4 responses、所有 reward/case；stage 2 verifier |
| 3 | sampling/*；s3.yaml | `mpt sampling --mode smoke` → 小 refill 诊断 → `--mode full` | 真模型 on-policy refill，256 accepted mixed groups；stage 3 verifier |
| 4 | verl_bridge、gspo_controller；gspo_primary.yaml | `mpt gspo --variant vanilla|dynamic --mode smoke` → `--mode pilot` → `--mode full` | 两份独立 full run，每份 5000 training groups、resume、匹配 settings；stage 4 成对 verifier |
| 5 | selection/predict/paired/plots；s5.yaml | `mpt evaluate --mode smoke` → pipeline 小集诊断 → `--mode full` | 三模型完整 CMExam 6811 + clean CMB 2000，统计/图源/案例；stage 5 verifier |
| 6 | export/launch/client/benchmark；s6.yaml | `mpt serve-check --mode smoke` → workload 诊断 → `mpt serve-benchmark --mode full` | 原生 LoRA API、100 consistency、1/4/8/16 并发，原始 request rows；stage 6 verifier |

每个命令须显式接受 `--config`、`--inputs <artifact-manifest>`、`--run-id`；full 没有 input artifact hash、固定 manifest 或前序 VERIFIED/DONE 证据就拒绝。`mpt run prepare` 生成可审阅 manifest，`mpt run launch --manifest ...` 启动 supervisor。恢复使用 `mpt run resume --run-id ... --checkpoint <manifest>`，不重新创建一个相同 ID 的 run。`mpt verify --stage N` 委托 stage verifier；不存在通用 `--mark-done` 开关。

## 阶段依赖与执行模式

兼容性/单元测试可在 Stage 1 前提前做 stage-0 diagnostic，不训练正式策略。正式顺序固定：S1 DONE 的 adapter → S2 DONE 的 pool/reward → S3 DONE 的 sampler → S4 DONE 的两个 checkpoints → S5 DONE 的分析 → S6 DONE 的 API/benchmark。验证只用于 gate；每阶段随后必须写真实 retrospective 和 interview story 才 DONE。

SMOKE/PILOT 从 immutable 初始输入启动，用独立 IDs；formal 不沿用 pilot 的 optimizer 或消耗预算。Stage 1 pilot 可选，若跳过需 verifier 的显式 N/A 规则/理由；Stage 4 实际 interrupted-resume 检查不可跳过。Stage 2、3、5、6 无硬性独立 pilot 规模，记录可选诊断或 N/A，不伪造 PILOT_PASS。

## 流向与复用合同

Data flow：官方 pinned source → 完整性/去重/隔离 → immutable IDs → tokenized shards/parquet → trainer/rollout；ground_truth/reference explanation 只给 reward/evaluator，不进 actor prompt。

Checkpoint flow：Stage 1 final adapter（不按 test 选）→ 两组 RL 相同初始化 → 每次成功更新后 recoverable checkpoint → validation-only best/final artifact manifests → 三模型评估及服务。final adapter 和 resumable checkpoint 是不同 artifact kinds。

Metrics flow：worker 原始输出 → event WAL → controller 成本/有效更新聚合 → append-only JSONL/Parquet → plot source / summaries；不能仅靠 W&B。W&B 默认为关，可作镜像。

Artifact flow：bulk root 中原子生成、hash/size 注册、Git compact index → verifier 核 hash/counters → stage report/claims ledger。对 raw rollout 保留所有 generated groups，不能过滤后才留日志。

Verifier flow：schema check → inputs/revision/integrity → 根据原始 ID 重算预算 → reload/resume/eval receipts → stage-specific clauses → machine receipt → retrospective linkage → state writer。GPT 的自然语言声明不参与状态转换。

## 当前与未来的边界

当前已有：审计脚本及实测 JSON、官方 source pins 与 hash、十项要求对应文档、候选 requirements、预算合同 JSON、初始 project_state、状态/run schema、planning-only validator。没有实现 trainer、stage verifier、数据处理和部署代码；不是六阶段 IMPLEMENTED。

后续最先完成：metadata lock → 隔离环境 ABI → tokenizer/mask fixtures → 8B memory/sleep/LoRA smoke → resume/counter adversarial tests。只有可行性 gate 通过，才准备数据与正式 SFT。计划中的可选探测预算在 COMPUTE_BUDGET 列出。

## 合同变更请求

**None：本次不更改 20k SFT、15k pool、G=4、两组各 5k、CMExam 6811、CMB 2000 或阶段留证要求。**

存在条件 proposal：512 response cap 升级、MedEmbed 换中文/多语言 encoder、BF16 失败后的量化替代。前两项属于有比较影响的工程/reward 决策，必须记录证据并在两组 formal 前共同冻结；涉及主研究合同/预算/量化则标 project-contract change，在明确批准前不能启用。不能用资源耗尽倒推降低完成门槛。
