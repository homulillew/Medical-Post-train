# Stage 1–6 工程执行方案

下面的命令是待实现 CLI 合同，当前不执行。所有 formal 入口都必须先消费固定 manifest；状态与验收见 EXPERIMENT_DATA_MODEL。参数是待 smoke/pilot 冻结的 starting configuration；现有正式样本/组数预算不变。

## 共用数据隔离：必须在 Stage 1 前完成

下载只发生于后续实现/数据准备阶段。模型仅取固定 Qwen3-8B 一份 snapshot；HF source 通过固定 revision 的 `snapshot_download(allow_patterns=...)`，JSON 数组用 datasets JSON reader / streaming parser，CSV 用标准支持 quoted newline 的 parser。CMExam 从固定 Git commit 的三份原始 CSV 下载，逐文件 SHA256；不使用不明 HF mirror。

在 SFT 抽样前，建立 CMExam **整个 train/val/test** 与 CMB-Exam 固定 test 的 question-only exclusion index；处理程序仅用于隔离，测试答案不暴露给 trainer/调参。SFT 对整个 CMExam 排除比仅对将来的 15k 排除更明确。CMB 使用同一个预先固定 overlap 规则；完整 CMExam test 保留原 6811 条，把相撞训练记录剔除，不能为分数删 test 题。若 CMExam 自身重复/标签问题，保留官方全集结果并另报 clean sensitivity；严重问题需明确 contract decision。

规范化：Unicode NFKC、统一空白/标点、去无语义序号；保留否定词、数字、单位。精确 question hash + question/options 内容 fingerprint（能识别 option 重排）。中文 char-5-gram MinHash 找候选，Jaccard >=0.85 为待复核近重复，短题另用编辑距离与人工抽样复核。阈值在只读训练重复样本上固定；跨 split cluster 优先保留 held-out，训练成员剔除。所有 cluster/source IDs、阈值、误杀抽样、exclude reason 留存。没有原 source ID 的样本用 revision/file/logical-row-index + normalized hash，不能用物理 CSV 行号。

选择 seed=42，使用 hash 排序/固定 RNG，不能用 Python 随机 hash。分层缺失标 unknown，不补造 difficulty。最终 manifest 列 source revision、file hash、parser revision、row IDs、dedup cluster、split、原始/处理后 count、token stats。SFT 10k+10k 不足时继续流式读源；仍不足则 BLOCKED，不复制样本凑数。

## Stage 1：Medical SFT

**输入与 schema。** `medical-o1` 使用 `zh` / `medical_o1_sft_Chinese.json`：Question→user、Complex_CoT→assistant reasoning、Response→final；不选 zh_mix。Huatuo 使用 id、conversations.from/value；人类/助手角色 canonicalize，保持多轮次序，拒绝孤立 assistant、空内容和非法交替。不生成不存在的 CoT，不强行把开放 QA 变成单选题。

统一 `SFTExample{sample_id,source,source_revision,source_row,question_hash,cluster_id,messages:[{role,content,reasoning_content?}],split,quality_flags}`。每源质量/dedup 后 10,000 训练 + 暂定 500 validation（总 val=1000，非 20k 内扣除）；按 cluster 分配，val 与所有训练互斥。非医疗/乱码过滤记录可解释规则与抽样，不用 test 结果决定规则。

**Tokenizer。** 固定实际 Qwen tokenizer/template；预统计总 tokens、supervised tokens、均值/P50/P90/P95/max，区分 prompt/answer 和 source。先考虑总序列 cap=2048（pilot 可比较 4096），超过 cap 的训练样本先剔除并从同源补足，报告被剔除分布；不截断到没有 answer。Prompt labels=-100，监督有效 assistant tokens 与 EOS；PAD mask 依 attention mask，不能因 pad_token_id==eos_token_id 而屏蔽所有 EOS。首版关闭 packing，避免跨例 leakage 和 sample counting；如启用，需 block-diagonal attention、coverage 证据与 decision。

**训练。** Transformers Trainer + PEFT `LoraConfig(r=32,lora_alpha=32,lora_dropout=0,task_type=CAUSAL_LM,target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj])`。明确排除 embeddings/lm_head 与 modules_to_save，SFT/RL 同构。BF16 base，LoRA 参数/优化器实际 dtype 留存；梯度检查点、use_cache=False，microbatch=1、accumulation=16，AdamW 起始 LR=1e-4、weight_decay=0、warmup_ratio=0.03、max_grad_norm=1、cosine schedule。LR 可在训练/val pilot 修订。

**命令与预算。** `mpt data prepare --stage 1 --manifest ...` → `mpt sft --mode smoke --config configs/stages/s1.yaml`（128 例/4 updates）→ pilot（1024 例/64 updates）→ `--mode full`（20k 全集、num_train_epochs=1、max_steps=-1）。预计无 packing 1250 updates，只作一致性检查；唯一例/监督 token coverage 才是权威。smoke/pilot 分别重新初始化，不把它们计进 full。

**恢复与验证。** 每 100 updates 或 15min 安全边界保存 adapter、optimizer、scheduler、RNG、dataloader/coverage；训练后新进程 reload final adapter，并做 held-out val loss 和固定 50 条 train/val generation sanity。验收要求计划完整 epoch、覆盖 >=99%（目标100%，差异须列出）、20k manifest、所有损失有限、无严重数据问题，final 与 immutable SFT initialization hash 一致。训练 early-stopping callback 禁用。保留 malformed/extreme/dedup/pre-post case、loss curve 原数据、正式 run report，再做 Stage 1 report/interview。

## Stage 2：Reward 与 rollout profiling

**候选池。** CMExam train 抽 15,000 个清洁唯一 prompt；没有 train difficulty 字段，默认确定性均匀抽样，报告 answer cardinality/长度等描述统计。Stage 1 已对整个 CMExam 去重隔离。暂定从 CMExam val 清洁题冻结 512 条 RL monitor set，另外 1024 条 validation selection set（两者互斥）；其余 val 留 diagnostic reserve。不使用 test 选参数。保留完整 val split index。

统一 `ExamPrompt{prompt_id,source_row,question,options:{A:...},answer_set,reference_explanation,annotations,split,cluster_id}`；用于 verl parquet 的 prompt 只含 system/user question/options，reward_model.ground_truth 保存答案，extra_info 指向 explanation/IDs。禁止 reference explanation 进入 generation message。

**Parser。** ParseResult={answer_set,method,valid_format,ambiguous,error,span_offsets}。优先严格闭合 `<answer>`；只在标签外最终回答区匹配明确“答案/Answer:”或整段仅选项字母的 fallback，不从 think 推理任意搜索字母。NFKC、case fold、允许分隔符、集合排序，未知字母/多个冲突答案/重复标签/尾部矛盾结论判 ambiguous；不能取任意最后一个跨度来刷分。valid_format 要恰好一组 think/answer 且正确闭合；fallback 正确可得 acc，但 format=0。格式判定与答案正确性分开。单/多选答案严格集合相等，无 partial credit。truncation 记 finish_reason，未完成 answer 判错；不能为了 profile 凑 completed responses 丢掉截断轨迹。

**奖励。** 保留 `0.8*acc + 0.15*acc*sem + 0.05*format`。sem 首选固定归一化 `clip(cosine,0,1)`；不做 batch min-max，不用测试答案校准。empty reasoning 得0且记录原因。所有 wrong 的 sem 也计算/保留以挖掘反例，其 gated contribution 为0；wrong total≤0.05、correct total≥0.8。若 embedding 出错不能静默按0当正常 reward，重试失败标 invalid group。逐 token reward 只在最后有效 token 赋总分，acc 单独保存 binary。

**MedEmbed gate。** `SentenceTransformer(model_id,revision=...,device='cpu')`，`encode(texts,normalize_embeddings=True)`；参考解释缓存 key 含 encoder revision、text hash、normalization/chunk policy。MedEmbed-small 最大512 encoder tokens，与 Qwen tokens 不同；长解释按句切到 <=480 encoder tokens，length-weighted mean 再 L2 normalize，保留 chunk counts。先在 64 对中文 train-only controlled pairs（同义/无关/否定/数字替换/重复废话）检查区分能力、UNK/截断和耗时；MedEmbed 与 BGE-M3 比较是后续 diagnostic，当前未下载权重。若否定/错误推理得高分，记录真实案例；不能将 cosine 当成医学逻辑正确性。替换 encoder 前记 D-003，两组使用同一最终 reward hash。

**vLLM。** 只加载 verified SFT adapter，`tensor_parallel_size=1`、BF16、max_model_len 初始2048（prompt cap1536+response512）、n=4，temperature=0.6、top_p=1.0、top_k=-1、min_p=0，固定 sampling seed、无 logits penalties。这里把 Qwen card 的 top-p/top-k 截断建议改成无截断采样，是为了与 actor temperature-scaled log-prob 定义一致；D-004 记录权衡，pilot 测完再共同冻结。单独推理可初始 gpu_memory_utilization=0.70、max_num_seqs=16、eager=True，再测后调；不能把这个独占值复制到 RL。

**实验。** smoke 50 unique×4；formal 固定 seed 选 1000 unique×4=4000 complete request responses。completed 指服务生成完成（含 finish_reason=length），不表示 answer 格式必合格。请求级失败必须重试并记录消耗，缺轨迹不进入完整组；原失败不能删除。保存 raw token IDs、文本、四分量、parser status、length、policy hash、temperature、时序。报告每类 group 比率、reward mean/std、fallback/error、semantic 按正确性切片、length/truncation、吞吐与显存。至少手工审查50条，覆盖歧义/高sem错答/多选；若有系统 parser bug，invalidate 并修复后重新正式 profile。

## Stage 3：Dynamic Sampling 与真实 refill

`group_id` 是当前 policy 下一次 prompt encounter，不是永久 prompt ID；同题再次出现产生新组。只有恰好 G=4 唯一轨迹、同 prompt/policy、acc∈{0,1} 才能分类。16种 binary组合单测；全0/全1拒绝、其余接受，sem/format/total 任意变化不能改变筛选。请求失败/重复 response ID 是 invalid generation，不能套原 recipe 的单例放行分支。

复用 DAPO 的分组/refill 语义，自有 bounded controller 实现完整计数、policy barrier、refill log 和重启；不引入 DAPO overlong penalty 或 token-mean loss。每次 update target=8 groups，gen_batch_size 初始8 prompts；生成完整组后依 encounter 顺序选前8 mixed，剩余标 overflow，不跨 optimizer 更新留旧 policy 队列。Vanilla 用同一 controller，接受所有有效完整组；相同 prompt stream seed 但 Dynamic 自然消费更多候选。

每次补足更新最多尝试32个 generation batches；达到上限时保存部分组与 starvation case，标 BLOCKED/需要诊断，不以少于8组静默更新，不无限循环。重新调 temperature/G/pool 必须新 proposal，不能黑名单化全错/全对题。原循环一次过15k可能耗尽：允许有 seed 的 cyclic candidate stream，epoch+cursor 持久化；重抽题不等于离线 replay。

smoke 先运行极小 real generate→reward→filter 链；formal `mpt sampling --mode full --accepted-groups 256`，至少256真实 mixed groups，固定 SFT policy（不训练）。记录 generated/valid/invalid/mixed/rejected/overflow/accepted、all prompt/output tokens、refill iterations、耗时；验证分类计数守恒，sampling_amplification = valid_generated_groups/accepted_groups，并同时报含失败 attempts 的成本倍率。Stage 4 导入相同代码版本，不能另写不等价 sampler。

## Stage 4：LoRA GSPO 正式对照

**上游配置映射（source-checked，runtime 待验证）。**

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
  use_kl_in_reward: false
  # filter_groups 是项目/DAPO controller 的扩展配置，不是 main_ppo 通用开关
  filter_groups: {enable: true, metric: acc, max_num_gen_batches: 32}
actor_rollout_ref:
  model:
    path: <pinned-local-base>
    lora_adapter_path: <verified-sft-adapter>
    lora_rank: 32
    lora_alpha: 32
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    lora: {merge: false}
    enable_gradient_checkpointing: true
  actor:
    strategy: fsdp2
    policy_loss: {loss_mode: gspo}
    loss_agg_mode: seq-mean-token-mean
    clip_ratio_low: 0.0003
    clip_ratio_high: 0.0004
    ppo_mini_batch_size: 8
    ppo_epochs: 1
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 2048
    use_kl_loss: false
    entropy_coeff: 0
    grad_clip: 1.0
    optim: {lr: 0.00001}
    fsdp_config: {param_offload: true, optimizer_offload: true}
    checkpoint:
      save_contents: [model, optimizer, extra]
      load_contents: [model, optimizer, extra]
  rollout:
    name: vllm
    tensor_model_parallel_size: 1
    n: 4
    load_format: safetensors
    gpu_memory_utilization: 0.45
    max_model_len: 2048
    max_num_seqs: 16
    enable_sleep_mode: true
    enforce_eager: true
    layered_summon: true
trainer: {nnodes: 1, n_gpus_per_node: 1, total_training_steps: 625}
```

data.train_batch_size=8 prompts，data.gen_batch_size=8 prompts；固定 release `ray_trainer.py:_update_actor` 将 ppo_mini_batch_size×rollout.n，因此8→32 trajectories，不能再填32导致128。最终 config compose 须验证这点、checkpoint.save_lora_only dataclass 字段（YAML无此项时由 adapter 显式注入）、无 reference/critic worker、动态 microbatch 实际 token budget。上述0.45是全卡比例，约20.25 GiB预算，约15.26 GiB base后KV/工作区余量紧；按测量适配，不保证可行。

**优化语义。** 每组 hybrid scalar reward 做均值/std归一化，epsilon 与 std定义复用固定 verl；advantage broadcast 到有效 response tokens。GSPO 用 sequence log-ratio/clip，不等同于GRPO token PPO。每次生成前 barrier 冻结 rollout policy v；同一 v 完成全部 refill；actor 在更新前 no-grad 重算 old_log_probs，保存 detached FP32 anchor，温度与 mask 和 vLLM 一致。初始 old/current ratio≈1；optimizer 后差异可测。KL off 不代表 old_logprob 可省，也无需常驻第三份 reference model。

当前whole-batch单minibatch、单epoch时，optimizer.step前old/current相同，ratio理论上为1，clip可能不活跃；必须报告这一解释限制（D-008），不声称已证实clipping收益。synthetic测试仍覆盖非1ratio；若pilot后需要两组共同改变minibatch/epochs，另记decision和预算映射。

不启用 off-policy rollout correction/bypass；主线请求等待同一版本完成。top_p=1/top_k=-1 保证不人为裁剪采样支持集；如果改用 nucleus，必须增加 logprob 分布定义和偏差诊断，不把 raw/model/sampler logprob 混在一起。无 dropout、无条件异步更新，generation 使用 async API 仅表示并发请求，不表示 off-policy training。

**显存切换。** actor 参数/optimizer→CPU；唤醒 rollout base/adapters→KV；生成/CPU reward→sleep level1；actor→GPU 重算 old/前反向；成功更新→checkpoint（如到间隔）→actor offload→仅 adapter 同步到 rollout。先不启CUDA graph。所有 sync记录版本/hash并断言adapter存在；任何base-only generation invalid。顺序阶段内两份base可能暂驻，测memory峰值；若标准hybrid初始化本身OOM，转串行子进程rollout/actor（仍用verl loss/engine），不能偷偷量化，记录时间成本与两组共同配置。

**正式执行。** smoke每变体4 updates，kill/resume测试每变体至少2+2 updates；pilot每变体512 accepted/training groups，可扩到1024但独立run。之后冻结共同config，分别从同一SFT adapter重启 optimizer，使用相同seed=42、LR=1e-5、constant schedule（避免不同 refill 导致按生成量调LR），group/update=8、625成功updates。LR/clip/length如经pilot修改，两组都重开formal。

Vanilla accepted_training_groups=所有实际送入成功update的完整组（包括全对全错），Dynamic=实际送入成功update的mixed组。最终两者各>=5000、accepted trajectories>=20000；generated只是成本，不作为到达预算。controller 以counter循环，625只是预期/守护参数；若dataloader耗尽或step上限触发而不足5000，退出INCOMPLETE，继续原预算的resume，不能FULL_PASS。

每64updates（512groups）或15分钟安全边界保存recoverable checkpoint；每64updates在固定512 monitor val上测accuracy，含0与最终625共11个点；另存便携adapter便于曲线复核。每变体从monitor最高的至多3个checkpoint，在预先冻结1024 selection val上选最终展示checkpoint；tie依次选较早checkpoint、较短响应。主表另报final-budget checkpoint，避免best与final概念混淆；最终服务候选也在测试前用相同validation规则选定。

必须保存reward四分量、correctness/hybrid/advantage std、entropy（若只算子样本需标proxy）、clip fraction、gradnorm、length、truncation、各group类别、累计generated tokens、有效training groups、wall/gpu-reserved time、VRAM、validation点。`pg_clipfrac_lower`在此GSPO实现恒0是占位兼容指标，不当成有效下侧clip测量。语义变化导致全对组仍有advantage的现象专门记录。

## Stage 5：公平评估与统计

锁定SFT、Vanilla、Dynamic checkpoint manifest后，一次固定protocol运行所有6811 CMExam test、2000 clean CMB-Exam外部题，每模型共8811；3模型合26433条primary responses。源数量与答案可用性先核实，任何缺/重复 ID 都拒绝验收。CMB按exam_class分层proportional allocation + largest remainder补齐2000，稀有层至少1，seed=42；去重先于选样，保存初始/排除/最终ID。不能只选表现好的子集。所有模型相同chat/thinking、token cap、sampling参数、按prompt ID派生seed，1 response/question；无few-shot test示例。

difficulty只在评估阶段读CMExam test原注释，先检查字段取值与含义，不猜级别方向。五级到easy/medium/hard的映射在看模型分数前冻结，unknown单列；若确为由易到难1..5，使用1–2/3/4–5，否则按官方语义映射。保留原五级表，不能为了显著性重划bucket。

预测按prompt_id inner/outer join检查完全覆盖后做paired bootstrap（seed固定，10000次question resamples）估计accuracy delta与95%percentile CI；同时McNemar exact（discordant少时）/合适近似。CI是题目抽样不确定性，单个训练seed不能推断训练seed方差或强因果。若同源重复题聚类，补cluster bootstrap敏感性。主要contrast预注册Dynamic−Vanilla；其他bucket/pair是次要分析。

效率图仅用Stage4固定monitor val点：accuracy vs累计generated tokens、accuracy vs成功updates/accepted groups、三类group ratio。matched token比较只在两组共同成本区间，以较早已测checkpoint作预算内对照；不生成虚构interpolated accuracy点，不外推Dynamic或把test用于曲线调参。训练生成、重试/waste、validation token分别列，并补total pipeline cost。

还需固定独立50条open-ended sanity（非SFT/RL/test），三模型同题审查格式僵化、冗长、明显回退；这不是临床安全或准确率主benchmark。自动挖四方向paired改善/退化、medium/hard分歧、reward anomaly；病例选取同时含负例。输出prediction parquet、summary CSV/JSON、difficulty table、paired stats、overlap report、plots及精确源数据/hash。后续严禁依据test结果换checkpoint。

## Stage 6：原生LoRA服务与benchmark

选择在Stage5测试前已由validation规则冻结的deployment artifact。导出仅adapter_config.json、adapter_model.safetensors和base/tokenizer/PEFT manifest；`PeftModel.from_pretrained`重载验证；训练恢复文件不混为deployment adapter。

未来标准launch形态（路径由manifest解析，不原样运行占位符）：

```bash
vllm serve /durable/pinned-qwen3-base --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --enable-lora --max-lora-rank 32 \
  --lora-modules medical=/durable/verified-adapter \
  --max-model-len 2048 --gpu-memory-utilization 0.70
```

本地`/v1/chat/completions`以model=`medical`发请求，chat_template_kwargs.enable_thinking=true；generation参数全部显式，不继承隐藏generation_config默认。主服务先保留原始think/answer文本，不启自动reasoning parser拆分；若启用则客户端需重组reasoning字段与content并做等价验证。单独启动服务时独占GPU，不残留actor。

**一致性。** 固定100条非test的val/workload prompts，验证adapter identity、tokenized prompt hash、EOS/length/stop、输出格式/accuracy与offline同协议；greedy另作工程诊断不替代主采样评估。不能要求跨并发核绝对bitwise一致；预先设accuracy绝对差<=2题为调查触发阈值，任何base-only/模板错误为硬失败，即便总分差小也失败。unexpected mismatch调查留case，未解释不能进入有效benchmark。

**性能。** 并发1/4/8/16，每条件100条真实请求（项目加强为400总请求），条件相同workload与seed，warmup每条件8条另记不计分位数，至少100条一致性另计。smoke只4条API请求。固定prompt/output长度分布，保留真实finish_reason与失败；闭环并发，不声称外推生产负载。采样文本没有被强制变长；报告按长度分桶。

客户端stream timestamps使用monotonic时钟：TTFT=首个非空token事件−send；E2E=complete−send；TPOT=(last-token−first-token)/(output_tokens−1)，输出<=1 token记null不填0。SSE chunk不是token，须保存累计content并按同一tokenizer核对usage；如chunk含多token，把流式TPOT标为request平均估计、不能宣称真实逐token ITL。优先捕获服务侧token timestamp获得精确TPOT；报告来源差异。throughput=测量窗口成功output tokens/窗口秒，request/s及error率均列；失败尝试成本单列不隐去。按条件报告TTFT/TPOT/E2E P50/P95、样本n、输出tokens、峰值NVML GPU used与poll interval（建议100ms）。保存raw requests/stream events、CSV/JSON、launch command、版本/hash、deployment runbook、latency outlier/serving mismatch cases。

## Stage 0 review addition: unique prompt exposure

Stage 4 records `generated_unique_prompts`, `accepted_unique_prompts`, `prompt_repeat_histogram`, and `max_prompt_exposure` under `schemas/prompt_exposure.schema.json`. Prompt identity is the immutable train record ID, independent of rollout group UUID; a repeated cyclic draw increments exposure once per prompt group, not four times for G=4. Histograms and maxima have separate generated and accepted views. Rejected/overflow groups count toward generated cost; only groups consumed by an optimizer update count toward accepted effective exposure.

The future Stage 4 verifier must recompute these from raw prompt/group/update lineage: sum(histogram frequencies)=unique prompts; sum(exposure*frequency)=generated or accepted groups; maximum occupied bin=max exposure; accepted unique <= generated unique. Crash attempts retain physical generated cost, while effective accepted lineage excludes rolled-back updates. Report per-window and cumulative distributions for both conditions. No synthetic Stage 0 exposure is backfilled as a formal measurement.
