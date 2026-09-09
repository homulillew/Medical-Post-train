# RTX 5880 Ada 算力与存储预算

本文件保留上一轮规划情景，并在末尾增加 Stage 0 实测校准。旧表属于 **Estimated**，不是 benchmark 或确定性承诺；本轮真实 probe 的数据单独标为 **Measured**。自然 mixed acceptance、医学样本平均长度和正式长期吞吐仍为 **Unknown**。realistic 是工作假设，不是统计期望。

## 显存账本

依据固定Qwen配置：36 layers、hidden4096、intermediate12288、32 query heads、8 KV heads、head_dim128、vocab151936、untied embeddings。按dense权重形状估算约8.19B参数，BF16 base约15.26 GiB。r32在7类linear层约87.29M可训练参数；FP32 adapter+grad+Adam m/v约1.30 GiB（以实际dtype和optimizer实现复核）。

| 活动阶段 | 估算常驻与临时需求 | 实机规划 |
|---|---|---|
| SFT | base15.26 + LoRA/optim1.30 + activations约6–14 + logits/workspace约2–5 GiB | 大致25–36 GiB；2k seq、microbatch1、checkpointing先验证，保留>=3 GiB余量 |
| 独占rollout | base15.26 + LoRA + KV + engine/context/workspace | max_num_seqs16、总context2048；不要直接使用默认超长context |
| actor old/backward | base+LoRA/optim + one microbatch activations | vLLM先sleep；entropy/logprob用chunk避免vocab logits瞬时膨胀 |
| 同时双base | 约30.52 GiB，仅权重，不含KV/激活 | 不能据此说48GB足够，初始化与切换峰值必须测 |

BF16 KV每token理论量：`2(K,V)*36*8*128*2 bytes =147456 bytes=144 KiB`。16条×2048完整context上限约4.5 GiB，32条约9 GiB，此外还有碎片/工作区。这解释了为什么32 accepted trajectories可以拆成最多16条并发rollout、actor按microbatch累积，不能将batch大小等同于同时驻显存数。

原生LoRA多进程sleep/offload使host RAM需要两份base、optimizer、checkpoint临时序列化、数据与Ray object store。预算约60–85 GiB，监控available RAM下限16 GiB；限制CPU embedding与Ray并发，禁止无上限worker。`use_shm`默认关，避免把全部base复制到tmpfs再形成重复占用。

## 时间模型

- SFT：`20,000 * mean_total_tokens / effective_training_tokens_per_second`。
- Vanilla生成：`5,000*4*mean_response_tokens / aggregate_decode_tokens_per_second`。
- Dynamic生成：上述生成量除以当前平均mixed acceptance a；额外overflow/retry另外计。
- 每组RL actor+old：`20,000*(mean_prompt+mean_response)*(1/actor_tps + 1/old_tps)`。
- 两组每组625次更新；切换时间每更新另计，validation为11×512 + 至多3×1024 =8704 responses/变体。
- 最终eval：3×(6811+2000)=26433，再加50×3 open-ended。
- 末尾overhead因子覆盖checkpoint/CPU reward/prefill波动/调度；这些不是无限容量，真实reward CPU若超预算必须重估。

| 假设输入 | best-case | realistic工作假设 | adverse有限情景 |
|---|---:|---:|---:|
| SFT平均总tokens/例 | 512 | 1024 | 2048 |
| SFT有效训练tokens/s | 1000 | 450 | 150 |
| RL平均prompt/output tokens | 256/128 | 384/256 | 768/512 |
| rollout aggregate output tokens/s | 300 | 160 | 60 |
| actor有效total tokens/s | 800 | 350 | 100 |
| old-logprob total tokens/s | 1500 | 700 | 200 |
| Dynamic acceptance a | 0.65 | 0.35 | 0.10 |
| 每update切换秒数 | 5 | 15 | 60 |
| 调度/保存等余量倍数 | 1.10 | 1.20 | 1.35 |

这些吞吐输入都是情景参数，不是已发表或本机测得的RTX5880基准值。特别是LoRA batch=16的aggregate decode不等同于单请求tokens/s。

## 主线预计GPU占用小时

“GPU占用小时”指任务占有该单卡的wall time（含CPU reward等待/切换），不是按NVML utilization积分；额外报告active kernel time时另名。

| 工作 | best-case h | realistic h | adverse h |
|---|---:|---:|---:|
| Stage1：20k完整SFT | 3.13 | 15.17 | 102.40 |
| Stage2：1000×4 profiling | 0.52 | 2.13 | 12.80 |
| Stage3：256 accepted real refill | 0.21 | 1.56 | 32.77 |
| Vanilla GSPO：5000组，含其validation/selection | 9.19 | 36.72 | 249.92 |
| Dynamic GSPO：5000 mixed组，含其validation/selection | 10.60 | 56.53 | 825.92 |
| Stage5：三模型完整eval + open-ended | 3.47 | 14.18 | 85.07 |
| Stage6：consistency + benchmark，请求级调度预留 | 0.50 | 2.00 | 8.00 |
| 合计 | **27.61** | **128.29** | **1316.87** |

算式输入与未四舍五入输出见 [compute-scenarios.json](evidence/compute-scenarios.json)。Stage6的并发1不适用batch aggregate吞吐，所以表中单独采用0.5 / 2 / 8 h的请求级调度预留（仍是假设）；初版仅按aggregate吞吐计算的更低值在JSON中保留为not_adopted，不当成测量。最终用校准后的请求级耗时替换。

realistic情景下Vanilla生成约5.12M output tokens，Dynamic约14.63M（未加overflow）；两组接受的20k trajectories数量相同。一次run不能靠“总共625步”隐去约2.86倍生成放大。Stage4 validation另外每变体约2.23M tokens，成本不可忽略。

严格worst-case**无有限上界**：a趋近0、refill starvation、机器故障或不兼容可使合同无法完成。adverse只是a=0.1、长度饱和且吞吐低的有限场景；这应触发诊断/资源讨论，而不是自动跑数周或降低5000组。`a = 1-p^4-(1-p)^4`仅在同题独立同分布且固定正确率p的模型中成立；真实组相关、题难度异质，不能用平均accuracy代入当实测acceptance。

## 后续校准与探索预算

以下是上一轮规划时保留的诊断/训练预算；本轮 Stage 0 已执行的实际范围见报告，不代表对应正式 Stage 已运行：

| 验证 | 拟定小预算 | 解决的问题 |
|---|---|---|
| 8B SFT memory sweep | 512/1024/2048 tokens各最多2个microbatch；整体<=15min | BF16/LoRA真实峰值；超时保留诊断，不升级为pilot |
| vLLM/actor切换 | 固定8 prompts×4，2个policy同步周期，<=20min | 单卡共存、cold/warm load、adapter真实性 |
| MedEmbed中文有效性 | 64对train-only text，CPU优先；<=15min/encoder | 语义区分、chunk/UNK、CPU服务耗时 |
| length诊断 | 32 train prompts×4，比512/1024，max一次<=20min | `<answer>`闭合率、截断、token成本；扩2048另记decision |
| SFT pilot | 1024例/64 updates | 粗略按formal约5.12%训练量，另有启动/验证 |
| RL pilots | 每变体512 groups（formal的10.24%） | 约best 2h / realistic 10h / adverse>100h；adverse时先停诊断 |

先测有效tokens/s（SFT与actor是total processed，rollout是output aggregate），峰值显存、cold load、switch time、CPU reward、acceptance与长度分布，再重算预算。探测不是Stage done；optional诊断总GPU上限初始2h，pilot另计。约128h主线+10–15h pilot/诊断仅是当前工作预算，可能需数个会话与多天持久调度。

## 正式参数建议

- **20k SFT、15k pool、5000 training groups、G=4保持。** 当前证据不足以证明必须降低。
- 保留r32；估计LoRA状态相对base小，首选减microbatch/关闭graphs/正确offload。
- response512是待验证风险；如果闭合answer率<95%或截断率>5%（train/val diagnostic），**proposal** 同时提高两组到1024，必要时2048，重新测预算和primary generation config。阈值是诊断触发条件，不是删掉难题的规则。
- 缺内存时先顺序切换再考虑架构；BF16→quantized LoRA是project-contract proposal，不自动采用。
- acceptance<0.1持续3个refill窗口时暂停并保存case；提temperature/shared length/组大小探索，不自动替换baseline或生成永久黑名单。

## 磁盘与保留

预留300 GiB工作额度+100 GiB安全余量：base单份约16.4GB、依赖/toolchain/wheels约20–60GiB、原始/处理数据与token cache约10–30GiB、raw rollout与token/logprob records约5–30GiB、两组checkpoints约30–80GiB（adapter-only权重+optimizer/extra若通过），另留export和故障证据。若只能保存完整base的FSDP checkpoints，每10份可增加约150GiB/run，转为预留>=600GiB并调整checkpoint保留策略，不能删除唯一恢复点。

磁盘当前充足不等于无风险；每次prepare/checkpoint前检查free bytes、临时双写空间、目标mount一致。active run至少保留最近2个有效完整恢复点、所有validation便携adapter、最终与选中checkpoint及失败证据。每次关键checkpoint复制到第二个物理盘是本机冗余；外部备份目的地尚未确认，见OPEN_QUESTIONS。

## Stage 0 实测校准（2026-09-08）

### Measured

- Qwen3-8B BF16+r32，512/1024/2048 的NVML峰值18.563/20.031/22.692 GiB（以raw bytes为准），update-phase吞吐584.44/1541.43/1499.00 total tokens/s。后两个点已包含Adam状态；数据为重复合成输入，非医学SFT长期速度。
- 最终batch-invariant V1 vLLM，16条×32 tokens：base419.44，LoRA 114.14 output tokens/s。该短窗口含shape/激活开销，不是稳态benchmark。不能把base速度用于LoRA预算。
- 首次sleep+wake约8.731s；独立actor进程的加载+一次backward+退出约13.604s。后两次热sleep+wake各约2.04s；不能把热路径代替首次成本。
- 非batch-invariant base长度诊断：512平均504.203 tokens、87.5%截断；1024平均751.211 tokens、22.656%截断。不是post-SFT分布。
- CPU语义比较实际执行8个混合长短文本，MedEmbed/BGE的单次编码总时长、RSS和截断长度见semantic原始metrics。长文本主导BGE耗时，短解释服务延迟仍不能由平均值替代。

### Estimated：共享1024候选的条件性占用小时

使用实测LoRA短批吞吐的1.5/1.0/0.5倍作为三种**估计**，有效SFT吞吐取实测1024 update-phase的0.8/0.45/0.2倍；actor/old-logprob速度仍是假设。mixed acceptance仍假设0.65/0.35/0.10；平均prompt384，平均output751.211从base诊断转移，overhead1.10/1.20/1.35，switch10/25/60s。625updates仍按原8-prompt主设置，未采纳D-014。

| 工作 | optimistic h | working h | adverse h |
|---|---:|---:|---:|
| Stage1 20k SFT | 5.07 | 9.84 | 24.91 |
| Stage2 1000×4 | 5.36 | 8.77 | 19.74 |
| Stage3 256 accepted | 2.11 | 6.42 | 50.54 |
| Vanilla 5000 + validation | 51.95 | 86.56 | 211.08 |
| Dynamic 5000 accepted + validation | 66.39 | 168.04 | 1099.55 |
| Stage5 all three checkpoints | 35.64 | 58.32 | 131.21 |
| Stage6 reservation | 0.50 | 2.00 | 8.00 |
| 总计 | 167.03 | 339.95 | 1545.04 |

工作情景从原128.29h变为约339.95h，主要因为输出比原256-token工作假设长，以及LoRA吞吐明显低于base。这个数值没有加入mini=4可能增加的更新开销，也不代表已经批准1024为正式配置。输入、公式和源metrics哈希见 [calibration JSON](../../experiments/stage0/compute_calibration.json)，可用 `scripts/calibrate_stage0.py` 在新输出路径复算；旧分析情景和失败run保留。

### Unknown

post-SFT长度、真实医学数据packing效率、自然mixed acceptance、正式LoRA长窗口decode、完整Ray循环/optimizer/checkpoint开销、正式validation表现、systemd重启和外部备份时延。1024仍未达到闭合阈值，若共同提高到2048须重新预算。严格worst-case仍无有限上界；不能因预计多天运行而缩减20k/5000/G4合同。


## Stage 1 完整实测校准（2026-09-09 CST）

本节来自正式run `s1_formal_20260908T152119_60040b` 和同50条heldout配对评估 `s1_evaluation_20260908T175526_8cfbc0`。上文Stage0估计保留为历史规划。可复算公式、输入和source SHA见 [Stage1 calibration](../../experiments/stage1/compute_calibration.json)，完整阶段结果见 [Stage1报告](../stage_reports/01_medical_sft.md)。

### Measured

完整20,000例、9,168,530 total /7,741,165 supervised tokens；平均458.4265 total tokens/例。正式更新计时8530.12s，吞吐1074.84 total tokens/s；formal worker计时2.537h，NVML峰值37.188GiB。独立重载和配对生成另计；含smoke/pilot/allocator诊断/各次重载/最终生成的互不重叠已计时GPU占用阶段，合计下界2.966h，缺失的早期pilot尾部开销没有补0或反推完整wall。

在max_response1024、temperature0.6/top_p1/top_k-1下，SFT的50条回答全部think/answer闭合，0截断，平均output346.22、reasoning150.62tokens；o1 reasoning301.24、Huatuo0。SFT实际生成17,311tokens/100.6413s=172.01 output tokens/s，含批次prefill/调度，不是单请求TPOT。Base为49/50截断、平均output1020.46；没有answer标签不等于无语义答案。

预先协议只在SFT closure<95%或truncation>5%时追加双方2048，本次未触发。因此1024是下一阶段优先测试候选，不能声称完成了1024对2048的实测优劣比较，也没有把Stage0的考试题长度诊断与本次开放QA当作同一分布。

### Estimated：维持原合同的条件性占用小时

| 工作 | optimistic h | working h | adverse h |
| --- | ---: | ---: | ---: |
| stage1_measured_worker | 2.54 | 2.54 | 2.54 |
| stage2 | 1.64 | 2.68 | 6.04 |
| stage3 | 0.65 | 1.96 | 15.46 |
| vanilla | 21.12 | 36.29 | 92.99 |
| dynamic | 25.53 | 61.21 | 364.72 |
| stage5 | 10.90 | 17.84 | 40.13 |
| stage6 | 0.50 | 2.00 | 8.00 |
| 合计 | 62.87 | 124.52 | 529.88 |

表中只有Stage1 worker为完成后的实测；其他项目仍是假设情景。输出均长346.22由开放SFT-val迁移到考试任务，prompt仍假设384tokens；decode采用实测LoRA窗口的1.5/1/0.5倍。mixed acceptance仍为0.65/0.35/0.10；actor/old-logprob、10/25/60s切换和1.10/1.20/1.35 overhead均为假设。20k、5000 groups、G4与原625-update规划均未缩减，D-014 mini=4的额外更新成本仍未采纳。

工作情景由Stage0约339.95h改为约124.52h，主要是替换了SFT实际token分布、post-SFT开放QA长度和更长LoRA生成窗口的吞吐锚点。这不是同一benchmark上的训练/解码加速实验，也不证明Dynamic降低生成成本。working条件下Dynamic仍比Vanilla估计耗时更长；生成放大由未知的自然mixed acceptance主导。

### Unknown

真实CMExam post-SFT响应分布、自然all-correct/mixed/all-wrong比例、完整Ray/GSPO actor和old-logprob吞吐、切换峰值、长期验证与外部备份。格式50/50通过不能填补医学正确性；源CoT算术错误与生成参考分歧已记录。下一阶段必须在train-only profiling中实测这些未知量。acceptance趋近0时严格最坏成本仍无有限上界，不能据此缩减正式组预算。


## Stage 2 actual profiling calibration

Formal `s2_formal_20260909T025736_f607d9`: measured mixed fraction 0.531000, mean output 237.553 tokens, P95 365.000, bounded-batch LoRA generation 182.274 output tokens/s. Tracked Stage2 GPU-worker wall 1.6214h includes smoke/formal generation and sequential semantic scoring; CPU diagnostics separately recorded.

| Variant | Contract training groups | Estimated generated groups | Estimated output tokens | Estimated generation GPU hours |
| --- | ---: | ---: | ---: | ---: |
| vanilla | 5000 | 5000 | 4751065.0 | 7.240424351495717 |
| dynamic | 5000 | 9416.195856873823 | 8947391.713747645 | 13.635450756112458 |

These are fixed-initial-policy rollout-only estimates. Actor updates, old-logprob recomputation, switching and validation costs remain unmeasured in Stage2 and must be added after their stages. Acceptance changes with policy updates; 1/P_mixed is not a measured Dynamic training amplification. No 5000-group budget reduction is made. Stage1 historical scenarios remain above, not overwritten. Machine-readable source hashes, actual run costs and adverse sensitivity: `experiments/stage2/compute_calibration.json`.


## Stage 3 actual refill calibration (2026-09-09)

Formal `s3_formal_20260909T053101_d99393` really generated496 groups /1984 responses and accepted exactly256 mixed, with1 additional overflow eligible. Measured amplification=1.937500x; P_mixed=257/496=51.814516%. Output478,117 tokens, accepted242,109, rejected234,982, overflow1,026; accepted rollout-token fraction50.638024%, not hardware utilization.

Generation wall=1947.643869s (0.541012h); active GPU worker=2039.387755s (0.566497h), BGE scoring=36.673988s, throughput=245.484818 output tokens/s. Formal NVML peak=32.606384GiB. Mean output=240.986391, P95=372, P99=441.17, max576, zero truncation. This replaces the ~0.7h Stage3 generation forecast with measured0.541012h generation /0.566497h active worker; historical estimates above remain the planning record.

Stage2 request batches contained4 prompts; Stage3 uses16 with the same engine max_num_seqs16. Its higher observed throughput is measured under different request batching and a different prompt slice, not evidence that Dynamic filtering saves rollout compute.

| Stage4 initial fixed-SFT projection | Contract groups | Estimated output tokens | Estimated generation hours |
| --- | ---: | ---: | ---: |
| Vanilla | 5000 generated | 4819727.823 | 5.453752 |
| Dynamic | 5000 accepted | 9338222.656 | 10.566644 |

Dynamic formula: `5000 * 1.9375 * 4 * 240.98639112903226 / 245.48481766436305`. The measured amplification includes final-batch overflow. This is an initial generation-only projection: policy-dependent acceptance, actor/old-logprob, switches, validation and checkpoint costs remain Stage4 measurements. Both mandatory5000-group baselines are retained; no LR/mini-batch/clip decision is made here.

All Stage3 costs are retained in [calibration JSON](../../experiments/stage3/compute_calibration.json): successful32-prompt smoke, failed32-prompt smoke with terminal metadata error, zero-generation failed startup, and formal. Across these runs, returned rollout output totals543,123 tokens; identity-control tokens are separate. Active-runtime checkpoints exclude the deliberate human pause and delayed failure cleanup. The failed worker overlaps the following startup attempt in wall time, so summing these process intervals would double-count GPU reservation; no exact all-stage GPU-kernel busy time is claimed. Raw timestamps, shutdown/termination observations, measured generation and active runtime remain available.
