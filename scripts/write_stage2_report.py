"""Write a retrospective only after full Stage 2 raw analysis and manual review."""
from pathlib import Path
from medical_posttrain.evidence.stage2 import read,selected_path,INDEX

root=selected_path('formal');s=read(root/'summary.json');smoke=read(selected_path('smoke')/'summary.json')
data=read(selected_path('data')/'summary.json');sem=read(selected_path('semantic')/'summary.json')
cfg=read(root/'config.json');supp=read(INDEX/'supplementary_analysis.json');compute=read(INDEX/'compute_calibration.json');review=read(INDEX/'manual_review.json')
placeholder=read(INDEX/'reference_placeholder_audit.json')
assert s['status']=='PASS' and s['completed_responses']==4000 and s['unique_prompts']==1000 and len(review['entries'])>=50
pct=lambda v:f'{100*v:.3f}%'
amp=s['expected_sampling_amplification'];amptext=f'{amp:.4f}' if amp is not None else '∞（未观察到mixed）'
lines=[]
def add(t=''):lines.append(t)
add('# Stage 2 — CMExam pool、严格解析与正确性门控奖励实测报告')
add(f'''
正式run：`{root.name}`。完整预算为15,000题candidate pool，独立50×4 smoke，以及正式1,000个唯一train prompts × G4 = **4,000条有效完成轨迹**。本报告从全量原始响应、embedding向量、reward分解和人工定性记录生成；验收状态由 [verifier收据](../../experiments/stage2/verification-final.json) 与 `project_state.json` 最终确定。Stage3–6未启动，policy optimizer updates=0。

主要结果：轨迹正确率 **{pct(s['trajectory_accuracy'])}**，至少一条正确的prompt比例（本次实际四次采样的pass@4/any-correct）**{pct(s['any_correct_rate'])}**；all-wrong/mixed/all-correct分别为 **{pct(s['group_fractions']['all_wrong'])}/{pct(s['group_fractions']['mixed'])}/{pct(s['group_fractions']['all_correct'])}**。这些是从清洁CMExam train候选池确定性抽取1000题、按冻结parser和官方答案计分的结果，不是全54,497题准确率或test/临床效果。

证据入口：[selected runs](../../experiments/stage2/selected_runs.json)、[正式摘要](../../experiments/stage2/{root.name}/summary.json)、[group明细](../../experiments/stage2/{root.name}/groups.json)、[奖励manifest](../../experiments/stage2/reward_manifest.json)、[决策记录](../implementation/STAGE2_DECISIONS.md)。完整响应、token IDs和embedding留在 `{root}`，Git只保存摘要/索引/案例和SHA。
''')
add('## 1. 数据源、候选池和验证集保留')
add(f'''
从Stage1 manifest解析固定官方CMExam revision `fadb22c89beb1b7115dc36460ba792eb96b7b972`，重hash train/val CSV并完整读取train54,497条。字段严格为Question、Options、Answer、Explanation；没有可用真实difficulty或category，不补造。原train SHA `3d6ca5e2499c510956534740c95179c483d9d47d2531e39fc5069333ffd23724`。

继承Stage1已封存question-only重复簇：char3 Jaccard>=0.65检索，char5>=0.85或SequenceMatcher>=0.90确认，连通簇整体处理。只消费已有cluster ID成员关系，Stage2没有重新解析CMExam test/CMB test，也没有测试生成、评分或调参。保留heldout成员，移除train侧重叠；词法隔离不能保证所有语义改写无泄漏。

处理记录：heldout cluster关联2617条、纯标点题2条、非连续选项标签11条、非法答案3条、train内部重复簇成员2821条。剩余49,043个清洁唯一候选，按seed42 SHA顺序选15k，每簇一个代表，没有复制补数。正式1000题使用独立`42:profiling:ID`哈希排序；smoke采用随后50题，互斥，不按生成表现选题。完整rejections及来源hash在data run `{selected_path('data').name}`。

pool答案集合基数分布：{data['answer_cardinality']}。共有 **{data['missing_explanations']}/15000** 条解释缺失，照实保留，不用模型补写；对应sem=0并记录missing_reference。用有解释样本筛选pool会改变目标分布，因此本轮没有这么做。

后续完整case复核另发现非空占位参考“请等待更新”：pool共{placeholder['pool_count']}题，formal共{placeholder['formal_prompt_count']}题/{placeholder['formal_trajectory_count']}条响应。冻结pipeline按非空文本编码了它们；对应低sem不能解释为医学推理差。实际分数不回填或改写，单独保留 [placeholder audit](../../experiments/stage2/reference_placeholder_audit.json)。未来若屏蔽此类文本，必须更新两个基线共用的reward版本。

CMExam val冻结为512 monitor、1024 selection与5275 diagnostic reserve，ID互斥；monitor/selection也在清洁簇边界互斥，test关联或剩余重复成员留reserve。所有6811原val IDs可核对。本阶段没有在这些子集上运行模型。未来checkpoint选择只能按后续validation协议，不能使用test。

统一ExamPrompt包含prompt_id/source/revision/row/question/options/answer_set/reference_explanation/split/cluster_id和explanation_missing。actor只消费question、options、固定format instruction；答案、参考解释和reward元数据留side channel。单测通过改变所有side-channel字段确认actor messages不变，verifier逐prompt重新调用Qwen native template核对token IDs。

源完整性限制：pool有{supp['pool_image_reference_prompts']}题命中预先描述性正则“暂无图/无图/如图/见图/下图/图示”，formal中{supp['formal_image_reference_prompts']}题。实际输入全为文本，没有图像附件。train27961明确“暂无图”，参考却描述影像特征，模型有时也讨论假设影像。这些题不事后删除，也不把得分解释为视觉诊断能力。该正则只是case triage，不是伪difficulty/category。
''')
add('## 2. 固定policy与实际vLLM运行时')
add(f'''
主干为官方后训练Qwen/Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218`。从Stage1 VERIFIED final-budget initialization manifest解析adapter实际路径，r32/alpha64，q/k/v/o/gate/up/down七projection，BF16 base，未量化、未merge。adapter SHA `1601e97891e51940bd4b575d8811a77d8278cbeb296c044b7004e41da6d9ea64`。不是smoke/pilot adapter，也没有按本阶段成绩重新选SFT checkpoint。

Formal policy_version：`{cfg['policy_version']}`。formal从clean git `3390cbafe55de6fde91fec6630c1248967772bb9`启动，固定config/source archive及hash，后续只增加分析/案例/报告。正式每批检查核心parser/reward/semantic/prompt/rollout源文件与manifest，开始/结束及批间重hashadapter。没有任何optimizer、GSPO、GRPO训练或refill。

运行时继承vLLM0.24、V1 runner、native sampler、VLLM_BATCH_INVARIANT=1、LoRA shrink split_k=1、TP1、BF16、原生LoRA。eager=True，context4096，max_num_seqs16，gpu_memory_utilization0.65，prefix cache关闭，generation_config='vllm'以避免隐藏采样默认。每个请求批4题，每题n4；真实Qwen chat template、thinking=True，只到assistant generation prefix，不手写替代模板。

采样冻结temperature0.6、top_p1、top_k-1、min_p0、无presence/frequency/repetition额外惩罚、max_response1024。每题seed由SHA(42:prompt_id)前4字节导出；已读当前vLLM parallel_sampling源，每个child seed=parent seed+member_index。相同文本可从不同成员独立生成，属于多样性结果；重复trajectory ID才是无效重复。

每次生成attempt都执行Base→SFT→Base-negative→SFT-repeat及sleep(level1)/wake后的SFT对照。formal中matched prompt-logprob Base/SFT最大差17.57701683，Base负对照、SFT重复和wake误差均0；sleep7.4613s、wake1.2142s，睡眠显存1,278,869,504bytes。这里只验证权重身份与运行时切换，未在睡眠期启动actor或更新adapter。全部控制输出和token成本另存，不计入正式4000。
''')
add('## 3. 严格parser、fallback与软件验收')
add('''
ParseResult保留answer_set、strict_match、fallback_match、valid_format、ambiguous、error_type、matched_span。严格路径允许一个闭合answer-only，或一个完整think后一个answer；答案标签只含合法唯一选项及无害分隔符，大小写/NFKC规范化、多选排序。重复字母AA、非法F、多个/嵌套answer、未闭合think/answer、answer后任何额外文字不算strict。不能从think中的A/B/C/D取最终答案。

Fallback只在最终回答区域匹配明确“答案：C / 答案是C / Answer: C”，或整个区域只有选项；多次结论或冲突不任取最后一个。fallback可得acc1、format0。截断的未完成answer为不可解析，合法length finish仍计入完成轨迹；如果length时恰好已有完整合法答案则按实际文本判定，不丢掉整条。

严格/多选/非法选项/重复/冲突/think-only/length边界与reward不变量有单元测试；分组测试覆盖16种binary组合、缺成员、重复ID、跨policy/config/reward及engine failure。完整仓库测试 **64 PASS**，用冻结train解释器和原分析环境pytest运行，未修改训练依赖。[测试收据](../../experiments/stage2/tests-final.json)。对抽样原文的人为阅读验证parser按这套合同执行，不是医学专家审查。

真实坏格式包括think内嵌answer后又输出最终answer、未闭合think、answer中写“无”。它们没有被transport过滤掉，也没有为了提高正确率放宽规则。正确性是“可按冻结规则恢复的最终选项集合与官方标签完全一致”；它同时受答题和结构遵循影响，不能称为纯粹医学知识准确率。
''')
add('## 4. Semantic诊断、选择与reward freeze')
add('''
比较固定MedEmbed-small-v0.1与BGE-M3，总200对：12组明确synthetic逻辑场景×15类，再加10条train解释各2个surface/unrelated对照。覆盖同义、无关、否定、剂量、年龄、性别、疾病/药物实体、治疗方向、疗程、因果、重复废话、题干复述、正确结论+错误解释、错误结论+高重合。合成剂量/方案只用于文本一致性测试，不是临床建议或验证过的处方。多个pair共享anchor，统计量不具有200个独立临床病例的含义。

| Encoder | 正对均值 | 负对均值 | pairwise AUC | 同anchor正确paraphrase胜出率 | CPU编码秒 | 进程RSS快照 GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |''')
for r in sem['results']:
 add(f"| {r['model_id']} | {r['positive_mean']:.4f} | {r['negative_mean']:.4f} | {r['pairwise_auc']:.4f} | {r['within_anchor_ranking_accuracy']:.4f} | {r['encoding_seconds']:.3f} | {r['cpu_rss_bytes']/2**30:.3f} |")
add('''
CPU诊断的RSS是编码后的进程内存快照，并非连续监测峰值，也不能视为两个encoder各自隔离后的纯模型占用；GPU memory为0，因为受控诊断明确在CPU运行。正式语义评分的GPU峰值另列。

两个模型在12个否定及12个剂量翻转例中，都把错误近似句排在正常同义改写前。MedEmbed否定cos均值0.99981，BGE0.99097；剂量分别0.99448和0.99777。BGE药物实体改动的排序12/12胜出，MedEmbed0/12；无关文本BGE均值0.3457、MedEmbed0.7995。BGE有更有用的中文实体/主题区分，但没有通过医学逻辑判别门槛。

D-022选择 **BGE-M3 revision5617a9f61b028005a4858fdac845db406aefb181**，保留原0.15弱语义shaping权重与correctness gate。没有声称它在正确答案内部能可靠排序推理质量；只将其作为有限的reference alignment，保留这一待训练检验的假设。去掉semantic/降权属于将来的显式共享reward版本决策，本轮不静默修改。

固定 R = 0.8 R_acc + 0.15 (R_acc R_sem) + 0.05 R_format。acc为canonical完整集合相等的binary值，多选无partial credit；format也是binary；sem=clip(normalized embedding cosine,0,1)。错误final answer时semantic贡献必须0、total<=0.05；正确answer的total>=0.8。Sem仍对所有错误响应计算，以观察高重合反例。Reward返回score/acc/sem/format/parse_status完整dict，避免verl把scalar score当作acc。

编码使用<=480 encoder content tokens的连续无重叠chunk，native special tokens逐chunk添加；归一化chunk向量按content token数加权平均后再L2 normalize。所有content IDs重组校验完全相同，原长解释不丢尾部；length、chunk span、UNK和truncated_tokens都保存。empty reasoning=0，missing reference=0，原因分别记录；实际embedding错误会终止该run，不能静默降为0。无batch min-max，cache绑定text/encoder revision/chunk policy。

第一轮semantic run因Transformers5.5.3移除旧tokenizer helper而FAILED；保留全部traceback、原始pairs与source archive。新run通过native backend post_process修复，两encoder的短文本向量与SentenceTransformer.encode最大误差2.98e-8。该失败不伪造成模型能力负例；真正的语义数字/否定失败来自成功diagnostic。
''')
add(f"\n正式reward_version（manifest SHA）：`{cfg['reward_manifest']['sha256']}`。")
add('## 5. Smoke与完整formal执行')
add(f'''
Smoke `{selected_path('smoke').name}`：50题×4=200全部完成，accuracy57%，0/1/2/3/4为10/5/10/11/14，mixed52%。输出mean236.26、P95=353.30、max453，answer-tag closure100%、truncation0%、strict86.5%。48组四条文本都不同、2组三种文本，无极端采样collapse。因未触发truncation>5%或closure<95%，没有比较1536/2048；没有为了制造mixed而提高温度。

Formal `{root.name}`：预冻结1000个唯一prompt全部覆盖，各恰好4个有效唯一trajectory，同policy/config/reward。成功stop或length可计完成；engine异常、缺响应和重复ID不可计。完整raw group以临时文件+fsync+rename提交，之后才做reward，token IDs和原文不被过滤改写。每批request-start/completed WAL记录请求ID、seed、输入长度、输出token和时间。相同run恢复只消费尚未完成ID，保持配置；本轮实际失败/重试见成本表，不把未发生的恢复称为实测成功。

生成阶段完成后由独立GPU进程执行BGE评分，生成与评分按阶段顺序调度；所有评分来自同一冻结manifest，对4000条一并完成。没有actor backward/optimizer、Stage3正式refill或Stage4试训。即便中间已经观察到很多mixed，仍执行完1000题。
''')
add('## 6. 正确率、分组分布与初始sampling成本')
add('| 正确条数 /4 | 组数 | 比例 |\n| --- | ---: | ---: |')
for k in range(5):add(f"| {k}/4 | {s['correct_count_distribution'][str(k)]} | {pct(s['correct_count_distribution'][str(k)]/1000)} |")
add('\n| GT集合大小 | 正式prompt组数 | 轨迹准确率 |\n| --- | ---: | ---: |')
for cardinality,row in s['answer_cardinality'].items():add(f"| {cardinality} | {row['groups']} | {pct(row['accuracy'])} |")
add(f'''
trajectory accuracy={pct(s['trajectory_accuracy'])}，本次四次生成any-correct={pct(s['any_correct_rate'])}。acc_vector逐组公开，无伪造advantage。std使用population ddof0，分别列correctness与hybrid_reward的std；尚未进行policy loss归一化，所以不报告actual advantage std。

P_mixed={s['group_fractions']['mixed']:.6f}，固定SFT policy下的一阶 expected sampling amplification≈**{amptext}**。这不是已经运行Dynamic得到的最终倍率；后续policy、长度和acceptance都会变，prompt encounters也可能相关。{'mixed低于10%，属于后续资源风险，保持G4和5000合同。' if s['group_fractions']['mixed']<.1 else '当前mixed未低于10%的预设资源风险示例阈值；后续仍需实际refill和训练窗口测量。'}

Mixed归因补充：{supp['mixed_group_contrast_counts']}。其中“contrast_only_from_unparseable_responses”意味着可解析响应全部答对，其他成员因格式无法解析而acc0。这样的group满足本项目acc对比定义，但不能全都解释为纯医学知识frontier。完整类别明细在 [supplementary analysis](../../experiments/stage2/supplementary_analysis.json)。

sampling_metric固定 **acc**。即使all-wrong，format可能不同，使total_reward std非零：正式观察到{supp['all_wrong_hybrid_std_nonzero']}组此情况；所有错误响应的raw sem虽可变化，门控后的语义贡献恒0。若按total std过滤，会误收这类缺correctness contrast的组。反过来，all-correct中有{supp['all_correct_hybrid_std_nonzero']}组hybrid std非零，来自语义/格式变化；不能声称它们一律没有任何训练信号。Dynamic按合同删除它们，舍弃这部分可能的shaping排序信号以研究正确性对比干预；效果留给Stage4。

权重大小与未来group-normalized advantage的影响不能直接等同：若一组全正确且format恒1，则R=0.85+0.15sem；忽略稳定epsilon时，中心化后除以组内std会约掉0.15。因此“弱shaping”仅描述raw reward相对correctness间隔，不能保证它在all-correct组的归一化更新也弱。本轮未计算或伪造实际advantage，该影响留给Stage4对照测量。

本阶段generated_unique_prompts=1000、每题encounter1次、repeat histogram为1→1000；accepted_unique_prompts=0，因为没有实际训练或refill接受，另列correctness-contrast eligible数。未来cyclic stream必须继续记录生成/接受两份exposure，不能把4条response当4次prompt encounter。
''')
add('## 7. 长度、格式和语义分层')
add('| token范围 | mean | P50 | P90 | P95 | P99 | max |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
for k,r in s['lengths'].items():add('| '+k+' | '+' | '.join(f'{r[c]:.2f}' for c in ('mean','p50','p90','p95','p99','max'))+' |')
add('\n| group类型 | output mean | output P95 | reasoning mean |\n| --- | ---: | ---: | ---: |')
for k,r in s['lengths_by_group'].items():add(f"| {k} | {r['output_tokens']['mean']} | {r['output_tokens']['p95']} | {r['reasoning_tokens']['mean']} |")
add(f'''
长度由raw text/native tokenizer重算，output_tokens取实际生成IDs，包含终止/标签；reasoning/answer单独重tokenize并去边界标签，两者与output不要求可加。无闭合think时以其剩余文本记录reasoning，保持runaway证据。answer长度基于可解析最终答案跨度，不把不可解析回答说成语义上没有作答。

Formal answer-tag closure={pct(s['closure_rate'])}，think closure={pct(s['think_closure_rate'])}，truncation={pct(s['truncation_rate'])}；strict format={pct(s['strict_format_rate'])}，fallback={pct(s['fallback_rate'])}，ambiguous={pct(s['ambiguous_rate'])}，unparseable={pct(s['unparseable_rate'])}。正确但格式不合格{s['correct_format_failure']}条、错误但格式正确{s['wrong_format_valid']}条。parse errors：{s['parse_errors']}。answer标签闭合不等于严格结构完全合法。

空reasoning{supp['empty_reasoning_count']}条，<=10tokens的近空reasoning{supp['near_empty_reasoning_le10']}条，其中合法格式+空reasoning{supp['format_valid_empty_reasoning']}条。5%的format奖励上界小于correctness0.8，但该项是否影响group内排序由这些真实结构差异决定，不能仅凭SFT源内验证100%格式就说它没有影响。
''')
add('| acc | sem mean | sem P50 | sem P95 | n |\n| --- | ---: | ---: | ---: | ---: |')
for a,r in s['semantic_by_accuracy'].items():add(f"| {a} | {r['mean']} | {r['p50']} | {r['p95']} | {r['count']} |")
add(f'''
wrong+sem>0.9共 **{s['high_semantic_wrong_gt09']}** 条，其中可解析错选与不可解析分别为{supp['high_semantic_wrong_parse_status']}；高相似+acc0本身不能直接判定为医学事实错误。补充固定阈值>0.8为{s['high_semantic_wrong_gt08']}条。correct+sem<0.5为{s['correct_low_semantic_lt05']}条，其中缺解释或空reasoning的0值另由semantic_reason分开：{s['semantic_reasons']}。有参考且实际编码的分布另保存在supplementary analysis，避免把源缺失解释成编码器判错。

相关系数（常量变量时为null）：{s['correlations']}。total主要由binary correctness隔开，由公式不变量和原始值共同验证；相关性不构成因果推断。高语义错答的gated contribution仍为0；正确final answer中的错误解释仍可能获得shaping，这是明确保留的局限。

![全量profiling图](../../experiments/stage2/figures/rollout_profile.png)
''')
add('## 8. 人工定性复核、bad cases和失败保留')
add(f'''
完成 **{len(review['entries'])}** 条完整trajectory的 **manual qualitative review**，由当前执行代理逐条阅读，不是人类医生审定或clinical validation，也没有调用外部GPT/Doubao等LLM Judge API或外部scorer。先读每个0–4桶前三组共60条，再读全量自动挖出的高sem错误/低sem正确/多选/格式/长度等补充样本。每条记录raw输出SHA、完整阅读范围、解析一致性与独立观察。医学事实无法确定时一律NOT_ASSESSED；不把代理阅读包装成人类医生审定。

自动case类别：all-wrong、1/4、2/4、3/4、all-correct、高sem错答、低sem对答、fallback、ambiguous、format、truncation、long/short reasoning、multi-select。实际不存在的类别写NOT_OBSERVED，不生成虚构案例。见 [case coverage](../../experiments/stage2/case_coverage.json)、[逐条review](../../experiments/stage2/manual_review.json)、[重点manual cases](../../experiments/stage2/manual_cases.json)。

重点案例：

1. **train31425 member0（correct_reference_disagreement）**：最终D与GT相符，但把尿比重1.014称高、归因血容量不足；参考解释称低比重且排除容量不足。能观察到表述分歧，不能在未临床审定下宣布参考或模型整体正确。这说明正确final answer也可能带不可靠解释。
2. **train19412 member2（correct_internal_contradiction）**：结尾先否认宿主比重增加，又选恰好表示宿主比重增加的E；最终字母正确。参考为空，因此semantic0是missing_reference，不是encoder识别了这处矛盾。
3. **train30338 member2（date_arithmetic_error）**：原文把4月18日+14天算成4月30日，独立datetime算术为5月2日；又把5月6日说成最接近5月3日，尽管5月2日/4日相差1天而6日相差3天。这里只验证算术，不审定临床排卵日期。
4. **train27961 member2（missing_image）**：题干明确暂无图，模型仍讨论假设龛影。B字母正确不能声称模型实际看过影像。输入完整性风险R24保留。
5. **train41049整组（1/4 frontier）**：四个最终答案分别D/B/C/A，均可解析，仅A匹配GT；与仅由坏格式形成的mixed不同，适合后续跟踪。模型对药物稳定性的解释也互不一致，未做专家裁定。
6. **train39417 member0（最高semantic错答）**：最终E而GT为B，sem=0.950364、format1、total0.05、semantic contribution0。虽然解释大量复用了参考中的ADH/渗透压等主题，最终选择仍不匹配；这是门控作用的真实例子。
7. **train17 member3（占位参考）**：最终C正确，reasoning完整，但参考只有“请等待更新”，sem=0.352775。低分来自比较对象无实质内容，不能据此否定该解释的医学质量。
8. **train20126 member1（最长循环推理）**：reasoning766tokens，错误地从特异度推健康人数800，反复得到160，最后却选表示72的C；从题给数据算式应为(1000−40)×0.2=192。这是直接算术/选项一致性检查，没有临床诊断判断。

初次semantic tokenizer API failure、其FAILED状态及修复后的新run均保留；未完成时的verifier FAIL收据也保留，证明smoke/部分计数没有越过formal门槛。没有发生的OOM、adapter错载、formal crash或重试不得虚构成已处理事故。
''')
add('## 9. 实际成本与Stage4条件估计')
c=s['costs']
gm=read(root/'generation_summary.json');em=s['semantic_memory']
add('| 阶段 | NVML peak GiB | 当前进程torch allocated peak GiB | 当前进程torch reserved peak GiB |\n| --- | ---: | ---: | ---: |')
for label,memory in [('vLLM generation',gm),('BGE semantic scoring',em)]:
 add(f"| {label} | {memory['nvml_peak_bytes']/2**30:.3f} | {memory['allocated_peak_bytes']/2**30:.3f} | {memory['reserved_peak_bytes']/2**30:.3f} |")
add(f'''
Formal生成输出 **{c['generated_output_tokens']}tokens**，本次generation调用总时长 **{c['generation_wall_seconds']:.3f}s**、吞吐 **{c['output_tokens_per_second']:.3f} output tokens/s**；冷加载/控制/sleep/wake另列。生成worker计时{c['generation_process_wall_seconds']:.3f}s，BGE加载+编码{s['semantic_encoding_seconds']:.3f}s。共用单张RTX5880 Ada46,068MiB。生成与scoring峰值NVML、allocated/reserved分别在generation_summary和semantic_memory保存，不能把父进程torch的0 allocated解释为vLLM引擎无显存占用。

每个G4 parent输入计数 **{c['generated_prompt_tokens_shared_prefill']}tokens**，四条trajectory逻辑输入计数 **{c['logical_prompt_tokens_all_trajectories']}tokens**。前者历史字段名带shared_prefill，但本项目只实测请求token IDs，没有测GPU kernel prefill工作量；prefix cache关闭，不声称节省该倍率的prefill计算。控制额外prompt/output={c['control_prompt_tokens']}/{c['control_output_tokens']}tokens。失败token={c['failed_request_tokens']}，retry output={c['retry_output_tokens']}，未结束请求={c['incomplete_requests']}，真实generation attempts={c['attempt_count']}。生成批wall共用于该批response记录，并明确不是每条请求的独立latency；不能将这列累加当总时间。

所有Stage2 GPU-owning worker的记录时长合计 **{compute['measured']['tracked_gpu_worker_hours']:.4f}h**，包含smoke/formal生成与顺序GPU语义评分；CPU数据/semantic diagnostic另计。该口径含worker里的CPU处理/sleep，不是GPU利用率积分，未记录的进程退出尾部不补造。完整成本账本：[compute calibration](../../experiments/stage2/compute_calibration.json)。

| Variant | 保留training groups合同 | 估计generated groups | 估计output tokens | 估计rollout GPU小时 |
| --- | ---: | ---: | ---: | ---: |''')
for name,r in compute['projections'].items():add(f"| {name} | 5000 | {r['estimated_generated_groups']} | {r['estimated_output_tokens']} | {r['estimated_generation_gpu_hours']} |")
add('''
以上仅把当前固定SFT的mixed率、均长和实际LoRA生成吞吐代入。Vanilla约生成5000组，Dynamic约5000/P_mixed组；不是完成过的Stage4性能。actor forward/backward、old-logprob、switch、validation仍UNKNOWN，需后续真实测量后加入，总训练GPU小时不能仅用本表替代。接受率减半/吞吐减半敏感性同文件保留；a→0时无有限worst-case。原5000/每组G4预算不改。
''')
add('## 10. 研究问题逐项回答')
answers=[('Q1 当前CMExam准确率',f"轨迹级{pct(s['trajectory_accuracy'])}，范围为本次1000个clean train prompts上的4000次生成。"),
 ('Q2 all-wrong/mixed/all-correct',str(s['group_counts'])+'；比例'+str(s['group_fractions'])+'。'),
 ('Q3 1/4、2/4、3/4',f"分别{s['correct_count_distribution']['1']}、{s['correct_count_distribution']['2']}、{s['correct_count_distribution']['3']}组。"),
 ('Q4 初始放大率',f'1/P_mixed≈{amptext}；初始固定policy一阶估计，不是实际Dynamic训练倍率。'),
 ('Q5 CMExam长度',f"平均{s['lengths']['output_tokens']['mean']:.2f}、P95={s['lengths']['output_tokens']['p95']:.2f} output tokens。"),
 ('Q6 1024是否足够',f"本次截断{pct(s['truncation_rate'])}、answer closure{pct(s['closure_rate'])}。作为后续共同response cap候选保留1024；未做2048准确率对照，不能外推全部训练中的长尾。"),
 ('Q7 format权重是否无影响',f"严格格式{pct(s['strict_format_rate'])}，仍有结构差异；5%小于correctness主项，但不能称其在group内完全无影响。"),
 ('Q8 semantic是否有用','BGE提供部分实体/主题对齐信息，数字与否定错误仍不敏感；作为gated弱shaping保留实验假设，没有证明提升推理质量。'),
 ('Q9 wrong且high-semantic',f"sem>0.9有{s['high_semantic_wrong_gt09']}条；门控贡献全0，另保留相对最高错误案例。"),
 ('Q10 Stage3 readiness','READY_FOR_STAGE3 = YES：pool/parser/reward/G4/acc分组/cap候选/完整分布及账本已明确，仍以最终verifier PASS为阶段门槛。Stage3没有在本轮执行。')]
for q,a in answers:add('\n### '+q+'\n\n'+a)
add('## 11. 验收、交接与限制')
add(f'''
验收重新读取原始CSV/pool/selection、各raw group、prompt/response token IDs，并用native prompt-prefilled DecodeStream重放全部响应；重算parser、sem cosine、reward、长度和所有group summary。不只读取summary count。Model/adapter/config/reward/source/全部产物SHA可核对；自动测试、200对诊断、定性review、cases/report/成本校准均是必需gate。最初不完整验收被正确拒绝，最终收据另存。

交接：`experiments/stage2/reward_manifest.json`（hash `{cfg['reward_manifest']['sha256']}`）、candidate pool manifest、formal config、1000×4 group acc_vector、response cap1024候选与相同Stage1 adapter。Stage3必须以acc对比验证真实refill，不得用total reward/sem std替代，不得带未冻结的新reward直接进入Stage4。

结论限于固定SFT policy、单seed集合和当前考试文本分布。没有Dynamic Sampling效果提升、没有RL训练、没有test表现、没有临床可靠性或模型看图能力的证据。正确label不能审定reasoning；reference可能缺失/有误；相近词法簇可能误排或漏掉语义改写；样本case不是随机临床错误率调查。Bulk SHA不是外部备份，Git clone不能恢复全部本机权重/响应。
''')
add('## 12. 面试故事：30秒')
add(f'''
我把一个完整医疗SFT checkpoint接成了可核验的在线RL前置流水线：固定15k CMExam train候选池，原生vLLM同policy生成1000题×4次，严格解析最终选项，再按真实标签计算正确性门控的语义和格式奖励。实测accuracy{pct(s['trajectory_accuracy'])}、mixed{pct(s['group_fractions']['mixed'])}，初始采样放大估计{amptext}倍。关键收获是embedding高相似不代表推理正确，且mixed可能来自格式失败；这些原始案例与成本都会保留给后续GSPO对照。
''')
add('## 13. 面试故事：2分钟')
add(f'''
选择CMExam是因为它有可验证选项答案和参考解释，可以先建立清楚的奖励边界，而不依赖外部LLM judge。先复核官方54,497条train字段，发现没有difficulty，因此使用固定seed和Stage1已封存重复簇规则冻结15k候选，不从test推断训练难度。验证集预留monitor与selection，actor只看到题干、选项和格式要求。

解析器严格区分think与最终答案，支持多选完整集合相等；fallback可以正确但格式0，冲突或未闭合结构不任意取最后一个字母。奖励保持0.8acc+0.15acc×sem+0.05format，返回完整dict以防框架把scalar误当acc。200对受控中文对照显示BGE比MedEmbed更能区分一些实体和无关句，但两者都对否定、剂量错误不可靠，所以只能作正确答案内的有限对齐shaping。

在单GPU上继承经过验证的vLLM V1/native sampler/batch invariance，实际做adapter正负对照和sleep/wake校验。50×4 smoke后没有触发长度或温度调整，正式同一固定SFT完成全部4000响应，平均输出{s['lengths']['output_tokens']['mean']:.1f}tokens，mixed{pct(s['group_fractions']['mixed'])}。据此估算后续Dynamic初始生成约放大{amptext}倍，但还没有训练，接受率将随policy变。完整raw、failures、逐组acc_vector、奖励分解和定性review说明了实验何时可信、何时结论必须受限。
''')
add('## 14. 深挖准备')
qa=[('为什么不用LLM Judge？','选项正确性有固定ground truth，严格集合核对便于重算，外部judge会引入额外成本与不稳定；推理事实无法判断时明确NOT_ASSESSED。'),
('为什么semantic不能直接当correctness？','受控否定/剂量翻转仍得接近1的cosine；它度量文本表征重合而非蕴含或医学逻辑，最终选项必须独立验证。'),
('为什么需要gating？','让错误选项的semantic contribution恰好0，wrong total<=0.05、correct>=0.8；原始sem仍保留用于诊断。'),
('为什么sampling metric不能用total reward？','all-wrong中的格式差异能制造非零total std，误当有效对比；acc_vector直接界定0<correct_count<4。'),
('为什么G=4？','这是预定单卡比较合同，每题有四次采样以观察正确性对比，资源可控；本轮没有宣称它是最优G或做G消融。'),
('mixed为什么重要？','它同时有正确与错误响应，可在当前任务奖励下形成方向不同的相对评价；实际收益需Stage4公平训练对照。'),
('all-correct真的没训练信号吗？','不是。纯binary correctness方差0，但hybrid中的semantic/format仍可能使相对奖励不同；本阶段只报告reward std，没有真实advantages。'),
('为什么还过滤all-correct？','干预按预先约定专注correctness contrast，接受舍弃部分shaping信号的代价，不能把这个设计写成已证实更好。'),
('all-wrong不能直接训练吗？','Vanilla仍会包含它们。当前binary正确性无组内区分，格式差异可能有信号；Dynamic按合同过滤它们，不代表任何算法都无法利用全错样本。'),
('为什么1024？','Stage1是候选起点，Stage2 train-only smoke和完整profiling实际测闭合/截断/尾部分位数；没有用test或主观感觉选cap，也没有把长cap优势编出来。'),
('如何避免think里的C当答案？','先验证并剥离唯一闭合think，最终region才作strict/fallback；未闭合、嵌套或重复答案不任取字母。'),
('多选怎样处理？','大小写和分隔符规范化、唯一字母排序，要求与GT集合完全相等；少选、多选、重复/非法选项无partial credit，不给actor泄漏GT基数。'),
('为什么profiling用train而非test？','这是构造奖励和预测训练采样成本的阶段，必须保持test最终评估独立；val也只预留而未用来挑当前结果。'),
('如何处理空解释和长解释？','空解释显式sem0并计missing_reference；长文本按固定encoder-token chunk完整编码，留全部长度/span/向量hash，不静默截尾。'),
('怎么知道adapter真的加载？','读取Stage1 final manifest哈希，实际Base/SFT matched-token logprob正对照、Base负对照、重复与wake一致，批间/末尾继续校验hash。'),
('生成是否可恢复？','已有组原子落盘、同run按冻结ID继续缺失项，WAL保留重试和未知尾部；本轮未崩溃就不声称实测formal恢复。'),
('最需要后续跟踪什么？','低/变化的mixed acceptance、格式造成的对比、正确选项中的错误解释、缺图和参考噪声、实际refill成本以及两个主训练组的公平性。')]
for q,a in qa:add('\n- **'+q+'** '+a)
path=Path('docs/stage_reports/02_reward_rollout.md');assert not path.exists();path.write_text(__import__('re').sub(r'(?m)^(#{1,6} .+)\n(?=\S)',r'\1\n\n','\n'.join(lines))+'\n')
print('Report characters',len(path.read_text()))
