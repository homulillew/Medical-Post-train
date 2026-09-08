# Stage 1 decisions and observations

## D-017：真实 SFT 的 LoRA 与 batching 候选

2026-09-08；Stage 1；ACCEPTED for smoke/pilot, formal freeze follows pilot。Contract change: NO。模型、20k、epoch 和 r32 不变。

选择 alpha=64，七个 projections，dropout=0，FP32 LoRA 参数 / BF16 base，与 Stage 0 实测的 PEFT 架构一致。早期规划 alpha=32 是起始方案，本轮明确选择 64；不声称它优于 32，未做 alpha 对照。后续 GSPO 两组必须从最终 SFT 同构 adapter 初始化。

真实 smoke `s1_smoke_20260908T145244_1af07b` 使用 microbatch1 / accumulation16，128 条 / 8 updates；update-phase 55,322 tokens / 67.223 s = 822.97 tokens/s，NVML peak 26,188,840,960 bytes。此时尚待新进程重载生成；这些数字不等于全轮完成。

pilot 候选改为 microbatch4 / accumulation4，有效16不变；只在每个固定16样本组内按长度排序执行，sample cursor/order 仍是原16条。loss 按全部有效 assistant tokens 加权，CPU oracle 已验证 microbatch1/2/4 与 padded full-batch 的 loss/参数更新一致。动态 padding 可能提高吞吐，但真实显存与浮点结果须由 pilot 检验；若失败保留原 pilot，再使用新 run ID 调整。全数据最大实际序列1558，继续采用cap2048，无packing、无截断。

完整 epoch 的第一更新 LR=0 来自原生 warmup scheduler；其 forward/backward、loss、样本/token 消费仍记录，不能把 parameter delta=0 的该更新伪装成非零。整个 run 必须有真实 adapter 参数变化。

loss 采用有效 assistant token 的平均 NLL，而非先按样本取均值再等权混合。两源各10k不意味着监督token各占50%：medical-o1有4,947,754个监督tokens，Huatuo有2,793,411个，因此前者约占63.91%。这使较长的真实推理得到相应的token权重；报告必须保留这一解释，不能声称来源梯度贡献严格一半一半。

## O-017：源数据实际形态与官方异常题干

固定 medical-o1 zh 有20,171条，Huatuo有142,248条，后者全部单轮。实现支持多轮角色监督，真实数据不能被描述为多轮训练。Qwen native template 会移除历史 assistant 的 think block，仍保留各轮 answer 和 EOS；单轮原始 reasoning 全部保留。多轮测试明确检查这种实际行为，无虚构历史 CoT。

第一次全量治理 `s1_data_20260908T144417_418b80` 因空规范化题干失败。复核发现原 option regex 误删以“A、B两种药物”开头的题干，已改为题干之后出现多个明确选项标记才剥离选项。官方 train 的两条纯标点题干仍保留ID和异常记录，无可用文本相似度，未按答案修复。

## D-018：词法去重及固定数据划分

2026-09-08；ACCEPTED；Contract change: NO。先对全部 CMExam train/val/test 和 CMB-Exam test question 做投影，再执行全源质量/重复处理。NFKC/空白/标点/题号规范化保留数字、小数、否定和单位；同题重排选项按 stem 排除。完整 char3 Jaccard>=0.65 prefix join 找候选，char5 Jaccard>=0.85 或 SequenceMatcher>=0.90 确认；所有连通 cluster 中含 benchmark 的训练成员都排除，包含传递关系。

这是可复算的词法防泄漏措施，不是语义同义改写完全无泄漏的证明。SequenceMatcher 分支仅在char3检索边界内适用。跨 SFT cluster 优先保留较小的 medical-o1 来源，再以 seed42 的 SHA 排序确定代表；split 在清洁cluster代表上以独立seed42 hash排序，每源前500验证、随后10000训练。所有规则在模型训练之前固定，不使用 benchmark 答案、难度或模型分数。

成功数据 run `s1_data_20260908T144854_bac3a7`：20,000 train +1,000 val，独立重算核验PASS。选中21,000条均低于2048，没有发生最终答案截断，也没有为凑配额过滤短/难样本。未选的清洁候选保留在bulk；长度统计范围是选择时实际扫描的候选和最终split，不冒充全142k的token分布。

## D-019：正式 SFT 采用可扩展 CUDA 分配器 segments

2026-09-08；ACCEPTED；Contract change: NO。设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`；只作用于 SFT 进程，vLLM 的冻结 batch-invariant 配置保持独立。正式训练仍使用 microbatch4 / accumulation4、相同 token-mean loss、同一数据、模型和优化算法。

观察：pilot 前32步实际张量峰值25,906,831,872 bytes，而NVML峰值48,020,193,280 bytes，缓存预留46,917,484,544 bytes；未发生OOM，但可见余量只有285,605,888 bytes。动态长度产生大量不同大小分配是候选解释，不能把 NVML 全部解释为活跃张量。

诊断 `s1_memory_20260908T151118_dfaed6` 从 fresh base 重放相同前32组，仅改变allocator；每步loss差0，step32 adapter最大参数误差0。重放峰值约26.68GiB。再对整个冻结训练集最长16条执行一轮真实更新，最大序列1558，loss1.45381、grad0.155538均有限，NVML峰值34,768,289,792 bytes（32.38GiB），剩余13,537,509,376 bytes（12.61GiB）。32步重放吞吐约1103.10 tokens/s，未证明吞吐提高；收益是显存余量。

这与 [PyTorch 2.11 CUDA memory management](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-memory-management) 描述的动态尺寸分配场景一致。该选项仍为 experimental，本机有限重放不能保证所有平台；正式 run 持续记录实际分配/预留/NVML。它不改变数据配额、梯度有效批量或两组未来 GSPO 的对照合同。

## D-020：正式一轮配置冻结

2026-09-08；ACCEPTED before formal launch；Contract change: NO。精确机器可读配置为 `configs/stages/s1_formal.json`，prepare 记录其SHA256与clean git commit。

选择固定 Qwen3-8B BF16 + r32/alpha64/七projection/dropout0，SDPA、gradient checkpointing、2048 cap、无packing；microbatch4 × accumulation4，组内长度排序，token-mean NLL；AdamW LR1e-4、betas0.9/0.999、eps1e-8、weight decay0、cosine schedule、warmup3%、clip1、seed42。新的base/LoRA/optimizer从头开始；不续训任何smoke/pilot adapter。正式语料20,000条、共9,168,530 tokens，覆盖全部唯一ID的一轮才终止；预计1250updates不是缩减预算的停止捷径。

依据：真实pilot1024条/64updates，验证token NLL从2.04616到1.45209，两源分别从2.36758/1.46733到1.80565/0.81536；四条greedy重载生成均有完整think+answer和EOS。smoke四条全在1024截断的负例仍保留。没有做LR/alpha网格搜索，也不推断临床正确率已改善。真实进程恢复的step33参数/loss误差均0；allocator-only replay和最长batch诊断支持D-019。

checkpoint每100updates或900秒安全边界，保留所有便携adapter与完整optimizer/scheduler/RNG/coverage状态；初始和final全1000验证，中间固定两源各64 monitor，不用test选checkpoint。final-budget adapter是后续初始化，不能按验证或test改选提前checkpoint。

最终生成protocol也在训练前冻结：两源各25条固定validation，Base/SFT同原始prompt，native thinking=True、temperature0.6/top_p1/top_k-1/seed42。先1024；若SFT闭合不足95%或截断超5%，双方在同50题上再测2048。这是结构与长度诊断，不是CMExam/CMB test评分。greedy重载检查与这份采样协议分开解释。

资源估计：实际pilot update-phase约1103 tokens/s，全epoch仅更新阶段约2.31h；加完整/周期validation与checkpoint约2.45h（Estimated）。最终独立重载和paired生成另计；这些不是已完成GPU小时，也不构成提前结束理由。
