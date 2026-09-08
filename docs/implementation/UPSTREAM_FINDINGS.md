# 上游源码审计与文档假设修正

截至 2026-09-08 的事实。固定 source hash/URL 见 [evidence/upstream-audit.json](evidence/upstream-audit.json)。这里覆盖本次发现的冲突；未执行路径的风险不写成已证实 bug。原研究合同没有删改。

| 假设 | 实际证据 | 工程处理 |
|---|---|---|
| “48GB GPU”可直接套统一速度 | 本机 RTX 5880 Ada，可见 44.99 GiB | 用实测容量；吞吐必须后续测量 |
| 新版 vLLM + 当前 torch 自然兼容 | 0.24.0 固定 torch 2.11，base 为 2.12 | 独立候选环境；不要覆盖 base |
| `algorithm=GSPO` 即完整配置 | `core_algos.py:1545` 注册 `gspo` policy loss，advantage 是另一个开关 | `adv_estimator=grpo` + `policy_loss.loss_mode=gspo` |
| DAPO 开关在普通 main_ppo 自动生效 | 独立 recipe 的 RayDAPOTrainer 实现 refill；release recipe gitlink 与 rolling pin 不同 | 封装同一 controller 跑两组，显式 integration gate |
| 过滤连续 total reward 与正确率等价 | recipe 按指定 metric 的 std 判断；默认错误字段会改变选择 | reward 必须返回显式 binary `acc`，检查组完整性与 `0<sum(acc)<4` |
| 原生 refill 已满足证据要求 | recipe 截断超过 batch 的 accepted groups；记录主要是 step；不足 G 的单例可被保留 | 先校验 G=4，记录 overflow/discard、所有 rejected 成本，增加跨恢复账本 |
| LoRA load 就是 resume | PEFT adapter 本身无 optimizer/scheduler；verl checkpoint 另存 model/optim/extra | warm-start 与同 run resume 分开 |
| 长期保留 actor+rollout 两份权重没问题 | 两份 BF16 base 约 30.5 GiB，余量还需 activation/KV/allocator | rollout sleep level 1、actor offload，顺序占 GPU |
| CMExam train 有难度字段 | `train.csv` 仅 Question/Options/Answer/Explanation；test 才有五种注释 | 15k train 均匀确定性抽样；不能偷用 test 难度分层训练 |
| medical-o1 默认 config 就是中文医疗 | card 为 en/zh/en_mix/zh_mix，2025-04 拆分医疗与通用混合 | 精确指定 `zh`，字段 Question/Complex_CoT/Response |
| Huatuo 是 instruction/output | 首条 schema 是 id + conversations[{from,value}] | 保留 turn 边界，解析角色，不拼接成一段 prompt |
| CMB card 的 `exam` 示例仍准确 | 同一 card YAML 是 `CMB-Exam`，test 文件另名 | 用固定文件 JSON loader，答案 release 另核对 |
| MedEmbed 可以直接评中文推理 | small card language=en，BGE-small-en，max_seq_length=512 | 中文相关性/否定反例/UNK/截断探测后冻结，BGE-M3 备选 |
| Qwen 默认输出长度 512 足够 | model card 对 thinking 推荐远长于 512 的输出 | 512 是成本假设，检查闭合率；1024/2048 只作为成对更改 proposal |

## GSPO / group advantage / reward 通路

[固定 loss 源码](https://github.com/verl-project/verl/blob/483b8a009ba3a97563edee3a19887e4862b8094a/verl/trainer/ppo/core_algos.py#L1545) 使用 response-mask 上平均 log ratio 的指数，sequence 聚合并采用 detach surrogate 梯度，内部强制 `seq-mean-token-mean`。不另写一个 token PPO loss 冒称 GSPO。FP32 log-prob accumulation 与 padding/EOS mask 应做数值对照。

[固定 reward manager](https://github.com/verl-project/verl/blob/483b8a009ba3a97563edee3a19887e4862b8094a/verl/experimental/reward_loop/reward_manager/naive.py) 接受 compute_score 返回 dict；若只返回 scalar，它会把 scalar 作为 acc。这对 hybrid reward 是实际接口陷阱。自有函数返回 `{score, acc, sem, format, parse_status}`，各样本 key 集合一致；score 放在最后有效 response token，group filtering 只读 acc。

全对组不等于没有训练信号：同组正确答案仍可能有不同语义/格式 reward，Vanilla 下 group-relative advantage 可以非零。Dynamic 会主动删除这类 reward 对比；“去掉所有零梯度组”不是本项目充分解释。报告同时计算 correctness std、hybrid std 和实际 advantage std。

## Qwen chat / thinking

[模型卡](https://huggingface.co/Qwen/Qwen3-8B)、[固定 tokenizer_config](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/tokenizer_config.json)。对其 Jinja 做 synthetic rendering：enable_thinking=True 的 generation prefix 止于 assistant 换行；False 则加空 think 块。不能总是假设 prefix 已含 `<think>`，也不能无条件再手工添加。模型卡建议 thinking 使用采样，主评估默认也固定 seed 的采样；greedy 只作一致性诊断，不混成主要能力分数。

SFT collator 根据实际模板定位 assistant target，验证保留最后回答的 reasoning；多轮模板可能移除历史 reasoning，需 snapshot fixture。Huatuo 没有 CoT 的回答不虚构 CoT；模板空 reasoning 块允许存在，格式奖励与任务格式另定义。

## 数据与语义来源

- [CMExam 官方仓库](https://github.com/williamliujl/CMExam/tree/fadb22c89beb1b7115dc36460ba792eb96b7b972)：官方表为 54,497/6,811/6,811，属于上游统计，尚未完整本地计数。Options 为含换行的字符串，需合规 CSV parser，不能按物理行切记录。仓库 Apache LICENSE 与 README 的研究用途声明并存；不据此宣称商业使用已获确认。
- [medical-o1 card](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)、[Huatuo card](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K)：metadata 标 Apache-2.0。训练前仍保留来源/去重清单，尤其 o1 题源可能与考试集同源。
- [CMB card](https://huggingface.co/datasets/FreedomIntelligence/CMB)：CMB-Exam 与 CMB-Clin 不混用；公开 test 答案在 2024-02 有修正说明，需固定与题目对应的答案版本。
- [MedEmbed small](https://huggingface.co/abhinand/MedEmbed-small-v0.1)、[BGE-M3](https://huggingface.co/BAAI/bge-m3)：后者支持多语言、dense embedding 可通过 SentenceTransformer 调用，但检索适配也不自动等于医疗逻辑判断正确。

## LoRA / sleep / serving

[verl LoRA 文档](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)、[固定 vLLM LoRA 文档](https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/docs/features/lora.md)、[PEFT checkpoint 文档](https://huggingface.co/docs/peft/developer_guides/checkpoint)。固定 verl server 对 adapter 模式 sleep 使用 level 1（保留 CPU 权重），不能用 level 2 丢弃 base 后仅同步 adapter。LoRA request 要验证 adapter 确实已载入，因为上游逻辑可能在未载入时以无 adapter 生成。Stage 2/4/6 必须通过 adapter identity 与正负对照，不只看 HTTP 200。

## 单次更新的GSPO clipping解释边界

固定`ray_trainer.py:_update_actor`将ppo_mini_batch_size乘n，engine worker按mini-batch进行optimizer更新。当前设计train_batch=8、ppo_mini_batch=8、ppo_epochs=1，整个rollout batch只做一次optimizer.step；在无dropout、同策略同步及相同logprob计算下，梯度计算时current==old，ratio理论上为1，因此clipping可能始终不活跃。GSPO函数仍是正确配置，但此设置无法单独证明“GSPO clipping带来稳定性收益”，一阶更新接近sequence归一化的group policy gradient。

这是算法设置的解释边界，不是把本项目改为GSPO-vs-GRPO实验。保持Vanilla-vs-Dynamic主对照；synthetic loss测试必须覆盖非1 ratio/clip/正负advantage，真实pilot记录ratio/clip分布。如果研究者需要非平凡clipping，可提出两组共同调整ppo_mini_batch或epochs的独立diagnostic与decision，重新说明optimizer-step/预算对应，不静默改正式配置。

## 审计日main与release的实际差异

额外比较`0d3f56a8980a55bfbbf1214ea54fa1b32ca1405c` main与v0.9.0：GRPO outcome advantage函数AST相同；GSPO函数的loss aggregation不同。v0.9.0内部强制sequence mean，而main改成使用传入loss_agg_mode，注释提示默认可为token-mean。主线显式固定`seq-mean-token-mean`并固定release，所以未来升级main不能假设同名gspo具有完全相同聚合语义。差异见[AST审计记录](evidence/gspo-release-main-comparison.json)。
