# Takeover决策记录

日期均为2026-09-08，Stage=0 / Architecture Planning，相关实际run仅环境probe；没有训练结论。这里用同一文件保存本次逻辑相关的decision，后续每个重大改变使用docs/templates/DECISION_RECORD.md独立归档。

## D-001：以实机容量和兼容release为起点

Status=ACCEPTED（架构选择）；Contract change=NO。证据：ENVIRONMENT_AUDIT、UPSTREAM_FINDINGS与source hashes。选择RTX5880 Ada容量预算、verl v0.9.0源码、vLLM0.24/torch2.11候选，而非直接复用base或跟latest0.28。

备选：直接使用base torch2.12+最新vLLM，改动少但组合未获得当前verl CI依据；旧2025教程版本则丢失当前接口。当前方案牺牲“最新版”，换取可追踪source与显式兼容gate。NCCL冲突不隐藏，metadata解算与运行通过分开。没有此前formal run受影响；环境安装与FA2/GPU验证还未完成，实际GPU成本仅极小probe，后续预计<=2h兼容诊断。

## D-002：复用GSPO与DAPO语义，自有预算/恢复controller

Status=ACCEPTED（设计）；Contract change=NO。证据：固定core_algos、RayDAPOTrainer及checkpoint source。

选同一个MedicalGSPOController跑Vanilla与Dynamic，保留上游GSPO loss/group advantage/engine，扩展完整G验证、filter-before-drop日志、成本ledger、预算完成与checkpoint事务。备选完整重写RL trainer会扩大正确性风险；完全不扩展recipe又缺预算与恢复留证。patch范围仅controller和bridge，禁止悄悄带入DAPO其他loss/reward。formal前要求parity与真实refill/resume。预计多占少量CPU/磁盘与checkpoint时间，两组相同；无现有formal比较失效。

## D-003：中文语义reward有效性先验不足

Status=PROPOSED；Contract change=NO（encoder选择在原MedEmbed可调整范围，若删semantic项/改公式则另开contract/reward decision）。证据：MedEmbed card的language=en、BGE-small-en base、512长度，BGE-M3官方多语言支持；**未测中文准确性**。

备选A保留MedEmbed-small：便宜、医疗检索适配，但中文表现未知；B换BGE-M3 dense：中文覆盖有依据但非专门医学推理评价，CPU更慢；C去掉semantic：改变reward设计，当前不采纳。先64对controlled train-only诊断（原/同义/无关/否定/数字变更），冻结chunk/normalization，比较rank与人工标签。若A明显失效，选B需另记ACCEPTED证据；两组同一reward hash从相同SFT开始。成本上限CPU每encoder15min，当前无generated tokens/测量结果。

## D-004：显式Qwen generation与old-logprob一致性

Status=ACCEPTED（起始诊断配置）；Contract change=NO。temperature=0.6、thinking=True、top_p=1、top_k=-1、无penalty；保留512作为待测cap。官方card建议thinking采样，但其top_p=.95/top_k=20会裁剪分布，而当前actor显式温度logits路径未自动重放该裁剪。先采用无裁剪，减少old/model/sampler定义混淆，不声称质量更优。

备选保留nucleus并引入分布重放/importance correction，复杂度更高，当前不纳入。后续比较使用相同配置与prompt-seeded RNG；实际格式闭合/entropy/length探测后冻结。若512截断触发，升级1024/2048为**PROPOSED**，需共享修改与重新估算，而非当前已改。资源随实际length增加，尚无实测；无既有formal受影响。

## D-005：训练数据不借用test难度注释

Status=ACCEPTED；Contract change=NO。CMExam train header只有Question/Options/Answer/Explanation；阶段合同“preferably stratified by available difficulty/category”是条件建议。改为15k确定性均匀train抽样，difficulty仅用于测试后分析，保留unknown而不推测标签。备选LLM自动标难度增加费用/潜在label noise，也不能冒充原注释。正式15k数量不变；实现后保留selection/dedup counts，数据清理成本CPU为主。

## D-006：状态来源必须是可重算证据

Status=ACCEPTED；Contract change=NO。原DEFINITION_OF_DONE明确反对短跑完成。选schema+contract+verifier receipt+CI重验，cost/effective budgets分账，GPT不能靠自然语言改变状态。备选只在README打勾无法验证；只看trainer global_step不能覆盖Dynamic补采样和恢复。

额外存储是原始events/checkpoint manifests，小于模型权重成本；增量IO测入pilot。当前state保持NOT_STARTED，无stage reports虚构。正式预算全部不变。

## D-007：评估选择与benchmark证据更明确

Status=ACCEPTED（计划）；Contract change=NO。原合同给了validation-only选择、paired统计与最低100服务请求。计划冻结monitor512/selection1024 disjoint validation集合、11monitor点、至多3候选selection、test前部署选择；服务每并发100请求，总400，高于合同下限。代价是额外validation tokens和请求时间，已计compute模型；不可依test表现挑adapter或只报告更好的checkpoint。单seed限制显式报告，改善并非DONE条件。

## 没有采纳的合同变化

没有减少SFT/RL/evaluation预算，没有换backbone/算法或删除Vanilla，没有以量化替代BF16，没有启动任何正式训练。若后续必须变动，记录新decision而不是改写这里的历史。

## D-008：不把配置GSPO等同于证实clipping有效

Status=ACCEPTED（解释与诊断要求）；Contract change=NO。证据：固定release的`ray_trainer.py:_update_actor`、`engine_workers.py:train_mini_batch`与GSPO函数。whole-batch单epoch单optimizer.step下，更新前old/current相同；clipfrac可能0是数学预期，不应捏造clipping改善。

保持原8groups/update、1epoch起始设置；增加synthetic非1ratio梯度测试与pilot ratio观测。备选两个minibatches或多epoch会改变每rollout的优化量，暂不采纳，需要两组共同的新decision且仍各完成5000groups。主问题是sampling intervention，不增加未经对照的“GSPO优于GRPO”简历主张。当前额外成本仅未来小型CPU/numeric诊断，GPU成本未发生。
