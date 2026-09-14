# Stage 4 — Interview Story

Scope: completed Stage4 experiments and owner-supplied scientific narrative.
Stage4 is `FULL_PASS`, with human review pending. This is not a Stage1–6 completion claim.
`READY_FOR_STAGE5=NO`; final tests and selection1024 have not run.

## 30-second version

我在单张约 48 GB GPU 上实现了 Qwen3-8B 医疗 Online GSPO 系统。
correctness-aware Dynamic Sampling 把 correct-vs-wrong 训练 group 密度从约 41%
提高到 100%，但付出约 2.92× rollout token 成本。Frontier 和 matched Random-3×
对照保留了正向 selection-effect 点估计，也表明统计结论仍为 **INCONCLUSIVE**。
系统在真实训练中验证了 transaction-safe resume，没有重复已经提交到 checkpoint 的更新。

## 90-second version

**Problem.** 单卡预算既限制采样，也限制策略更新。我研究的是如何分配更新预算，
同时量化效果和生成成本。Vanilla 与 Dynamic 使用同一份医疗 SFT 初始化。

**Reward reliability.** 答案由固定 parser 与 ground truth 验证。semantic shaping
受 correctness 门控，错误答案不能靠语义相似拿到正确性奖励。采样 eligibility
只看组内 binary correctness，不按总 reward 筛组。

**GRPO advantage + GSPO loss.** 我用 GRPO-style group-relative advantage 做组内
credit assignment，再用 GSPO sequence-level loss 更新策略。G=4，8 groups/window，
32 条轨迹的 old logprobs 在两个 minibatches 前全部冻结。系统没有 critic 或常驻 reference worker。

**Dynamic sampling.** Dynamic 给 mixed groups 分配更新。正式训练中，具有正确/错误
advantage separation 的组占比从 41.02% 变成 100%；但 nonmixed 组并非都没有信号。

**Equal-update / equal-token.** 两个正式条件各完成 5000 组。Dynamic 在全部预注册
正数 equal-update milestones 上点估计领先，最终 monitor 为 68.55% 对 67.58%，
差值约 0.98 pp，但 CI 跨 0。它花了约 2.92× rollout tokens，equal-token 没有稳定优势。

**Frontier falsification.** 我原先关注 held-out SFT-mixed 是否更容易被解决。
1000 题 G4 diagnostic 没支持这个 H1。额外收益点估计主要来自
SFT-G4-all-wrong hard-tail proxy，不能把它称为已证明的原因。

**Random3x causal control.** 新对照匹配 Dynamic512 的 1360 个候选生成组，但按
预冻结 hash 随机选训练组。新验证集上 Random3x 为 62.50%，Dynamic 为 64.65%；
差值约 2.15 pp，CI 仍跨 0，因此 selection effect 有正向点估计，但尚未证实。

**Transaction recovery.** 一次 GPU-release assertion 发生在 actor 已保存 step806、
controller 仍记 step804 的边界。我验证并采用完整 checkpoint，不重放 805/806，
再检查下一 fresh window 到 step808。Random3x 也完成了真实进程终止和恢复测试。

## Common interviewer questions

### Q1. Why GSPO instead of GRPO?

本项目的组合是 **GRPO-style group-relative outcome advantage + GSPO sequence-level
policy loss**。advantage 负责同一 prompt 内的相对 credit assignment；GSPO 用
sequence-level importance ratio 和 clipping，适配 trajectory-level outcome reward。
项目没有 GRPO-loss vs GSPO-loss formal ablation，所以不能回答“实验证明 GSPO 更好”。

### Q2. Why Dynamic Sampling?

它把有限的 policy-update budget 分配给存在可验证 correctness contrast 的 groups。
正式数据中，correctness-discriminative group density 从 41.02% 提高到 100%。
这个指标描述进入 optimizer 的信号组成，不代表 gradient quality 提高 2.44 倍。

### Q3. Is Dynamic more compute-efficient?

在这次 equal-update 比较中，Dynamic 的点估计持续领先；但没有证明 rollout-token
efficiency。正式 rollout token 成本约 2.92×，shared 1M–5M thresholds 仅在 4M 点领先。
这个 token 比例不能直接换成 GPU-hours、费用或端到端 compute 比例。

### Q4. How do you know it is not just seeing 3× more prompts?

Random3x512 与 Dynamic512 都生成 1360 groups，再各训练 512 groups。
Random3x 按生成前固定的身份 hash 选组，不读 correctness 或 reward。
独立验证集上 Random3x=62.5%，Dynamic=64.6484375%；Dynamic−Random3x 为
+2.1484375 pp，95% CI [−1.171875, +5.46875] pp。这个方向符合 correctness-aware
selection 的解释，但不是因果证明。两者匹配的是 generated-group schedule，并非精确 token 数。

### Q5. Did Dynamic ever regress?

有。正式 final monitor 的 paired comparison 中，44 题被纠正，39 题发生回退。
审阅包 `S4-MR-03` 也保留了 Vanilla 正确而 Dynamic 错误的完整回答。
我同时报告 gains、regressions 和 CI，不把个别正例当作总体优势。
这些案例的拟议人工 observation 仍待用户确认。

### Q6. Why no critic or reference model?

组内 relative outcome advantage 不依赖额外 critic。冻结设计没有 KL reward/loss，
也没有 permanent reference worker。old policy 仍然存在：每个窗口用 actor 为全部
32 条轨迹计算并保存 old logprobs，后续 importance ratio 对它们比较。

### Q7. How did a single GPU handle rollout and training?

通过实际的 vLLM sleep/wake 与 actor 生命周期切换。先完成 rollout，vLLM sleep；
actor 加载原生参数、optimizer/scheduler/RNG，完成两个 minibatches 并保存 checkpoint；
actor 退出且真实显存释放后，vLLM wake，同步新 adapter，才提交窗口状态。
这不是并行多卡训练，Stage6 concurrency benchmark 也没有运行。

### Q8. What was the hardest engineering bug?

Vanilla 的 controller 停在 3216 groups / step804，而 actor 已完成 805、806 并
保存完整 checkpoint。GPU-release assertion 在同步/提交之前失败。普通 replay 会
重复优化工作。我把 rollout、update、checkpoint、controller commit 当作事务边界，
检查 checkpoint 身份和 lineage 后采用已有结果：3224 groups / step806，重复更新为 0。
下一 fresh window 到 3232 / step808，验证通过。

原失败时没有保留具体 NVML 数值，不能断言精确 teardown 延迟或特定驱动故障。
修复是在原阈值下有限等待真实释放，并保留读数。另一次 Random3x fresh-process
resume 从 32 groups / step8 恢复到下一次 step10，也没有重复 optimizer step。

### Q9. What changed your interpretation?

Frontier H1 没被支持：SFT-mixed 517 题中，Vanilla 变为 all-correct 的有 264 题，
Dynamic 有 261 题。收益点估计更多来自 SFT-G4-all-wrong hard-tail proxy。
Random3x 的配对 CI 又跨过 0，所以报告保持 **INCONCLUSIVE**，没有将选择效应写成定论。

## Evidence navigation and speaking limits

- [Stage4 report](04_gspo.md)：完整数值、条件和结论边界。
- [Formal and auxiliary evidence handoff](../../experiments/handoffs/stage4_final_docs_to_chatgpt_v1.json)：精确路径和 hash。
- [Recovery bundle](../../experiments/stage4/recovery_interview_case_v1.json)：正式事故及独立恢复案例。
- [Full-response packet](../../experiments/stage4/manual_review_packet_v1.md)：正例、回退与采样边界。

不声称 final monitor 显著胜出、节省 rollout compute、GSPO 实验优于 GRPO、
临床有效性或 final-test superiority。报告中的机制不确定性和 single training seed
限制必须与正向点估计一起说明。

下一步是用户完整审阅 `S4-MR-02`、`S4-MR-03` 并确认或编辑 observation。
人工确认之前不生成最终人工 review artifact，不运行 full Stage4 verifier，也不执行 selection1024。


## GRPO vs GSPO Auxiliary Objective Ablation

独立单 seed、512-group auxiliary validation 对照已完成。原有 5000-group GSPO formal pair 及其结论保持冻结。本节补充此前缺少的 token-level objective 对照；不是 primary formal GRPO 实验。

GRPO 使用 verl `compute_policy_loss_vanilla`，原生 token ratio 与 dual-clip=3，和 GSPO 同用 native GRPO group-relative advantage、`seq-mean-token-mean` 聚合。六组原始固定批次诊断后按预注册规则选择 LR=1e-06、symmetric clip=0.2，没有使用新 validation 调参。原生 clamp、dual-clip 和 clip 尺度均属于 objective-package 差异，不能把全部差异归因于 ratio geometry。

| Model | Correct/N | Accuracy |
|---|---:|---:|
| sft | 313/512 | 61.1328% |
| vanilla_grpo | 321/512 | 62.6953% |
| vanilla_gspo | 322/512 | 62.8906% |
| dynamic_grpo | 328/512 | 64.0625% |
| dynamic_gspo | 326/512 | 63.6719% |

- `vanilla_gspo_minus_grpo`: +0.1953 pp，95% CI [-2.9297, +3.3203] pp；gain/regress=34/33，exact McNemar p=1。
- `dynamic_gspo_minus_grpo`: -0.3906 pp，95% CI [-3.9062, +3.1250] pp；gain/regress=40/42，exact McNemar p=0.912157。
- `dynamic_minus_vanilla_grpo`: +1.3672 pp，95% CI [-1.9531, +4.6875] pp；gain/regress=42/35，exact McNemar p=0.494382。
- `dynamic_minus_vanilla_gspo`: +0.7812 pp，95% CI [-2.3438, +3.9062] pp；gain/regress=36/32，exact McNemar p=0.716301。

描述性 interaction=-0.5859 pp，95% paired bootstrap CI=[-5.078125, 3.90625] pp。预注册判定为 `NO_CLEAR_OBJECTIVE_DIFFERENCE`；Dynamic 跨 objective 的结果为 `POSITIVE_POINT_ESTIMATE_BOTH_OBJECTIVES`。这是单 seed exploratory validation，非 final test，CI 跨零不代表等效。

Dynamic-GRPO 实际生成 1336 组，amplification=2.6094，rollout tokens=1602795；Dynamic-GSPO 对应 1360 组、2.6562、1633072 tokens。生成量未人为匹配。

四个固定长度区间的 ratio、clipping、absolute surrogate、accuracy 和样本分布见 [raw analysis](../../experiments/stage4/loss_objective_ablation_analysis_v1.json)。Per-length parameter-gradient proxy 为 `NOT_IDENTIFIABLE`；on-policy 长度关联不能解释为长度的因果效应。所有反向结果与无显著差异的比较均保留。

两条 GRPO 均完成真实 32→40 groups / 8→10 steps fresh-process resume 与完整原生 raw verifier。五模型统一评估通过。案例仅为机器候选，未伪造人工审阅。Stage4 仍 `FULL_PASS`，Stage5 仍 `NOT_STARTED`，`READY_FOR_STAGE5=NO`；selection1024 与 final tests 未使用，未执行 full Stage4 verifier。证据入口：[handoff](../../experiments/handoffs/loss_objective_ablation_to_chatgpt_v1.json)。
