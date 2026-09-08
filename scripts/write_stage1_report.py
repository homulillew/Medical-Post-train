"""Render the Stage 1 retrospective only from completed, audited evidence.

Manual review is a required input. This creates no VERIFIED/DONE state and does
not choose a checkpoint; the immutable initialization is the full-budget adapter.
"""
import argparse
import json
from pathlib import Path

from medical_posttrain.evidence import sha256


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/stage_reports/01_medical_sft.md')
    args = parser.parse_args()
    assert not Path(args.output).exists(), 'Preserve previous reports; use a new output'
    index = Path('experiments/stage1')
    selected = read(index / 'selected_runs.json')
    formal_dir = index / selected['formal']
    manifest = read(formal_dir / 'manifest.json')
    bulk = Path(manifest['artifact_root'])
    formal = read(bulk / 'summary.json')
    config = read(bulk / 'config.json')
    assert formal['run_class'] == 'FORMAL' and formal['unique_examples'] == 20000
    assert formal['coverage_fraction'] == 1 and formal['global_step'] == 1250
    assert formal['initial_trainable_digest'] != formal['final_trainable_digest']
    reload = read(bulk / 'reload_generation/receipt.json')
    assert reload['status'] == 'PASS'
    evaluation = read(index / selected['evaluation'] / 'summary.json')
    assert evaluation['status'] == 'PASS'
    generation_audit = read(index / 'generation_audit.json')
    assert generation_audit['status'] == 'PASS'
    review_path = index / 'manual_case_review.md'
    manual_review = review_path.read_text()
    assert len(manual_review) > 500 and selected['evaluation'] in manual_review
    cases = read(index / 'case_coverage.json')
    calibration = read(index / 'compute_calibration.json')
    data = read(index / selected['data'] / 'summary.json')
    raw = read(index / selected['data'] / 'raw_manifest.json')
    smoke = read(index / selected['smoke'] / 'summary.json')
    pilot = read(index / selected['pilot'] / 'summary.json')
    memory = read(index / selected['memory'] / 'summary.json')
    metrics = [json.loads(line) for line in (bulk / 'canonical_metrics.jsonl').open()]
    physical = [json.loads(line) for path in sorted(bulk.glob('attempt_*/metrics.jsonl')) for line in path.open()]
    assert len(metrics) == 1250
    checkpoint_events = [r for r in physical if r['event'] == 'checkpoint']
    assert checkpoint_events
    validations = formal['validation']
    initial = next(r for r in validations if r['scope'] == 'initial')
    final = next(r for r in validations if r['scope'] == 'final')
    assert initial['examples'] == final['examples'] == 1000
    candidate = calibration['rl_response_candidate']
    pre = evaluation['results'][f'base_{candidate}']
    post = evaluation['results'][f'sft_{candidate}']
    def pct(value):
        return f'{100*value:.2f}%'
    token_rows = []
    for source in ('all', 'medical_o1', 'huatuo'):
        for kind in ('total_tokens', 'supervised_tokens', 'reasoning_tokens', 'answer_tokens'):
            row = data['token_statistics']['train'][source][kind]
            token_rows.append([source, kind, f"{row['total']:,}", f"{row['mean']:.2f}",
                f"{row['p50']:.2f}", f"{row['p90']:.2f}", f"{row['p95']:.2f}", f"{row['p99']:.2f}", row['max']])
    generation_rows = []
    for cap in evaluation['limits']:
        for model in ('base', 'sft'):
            for source in ('all', 'medical_o1', 'huatuo'):
                row = evaluation['results'][f'{model}_{cap}'][source]
                generation_rows.append([cap, model, source, row['count'],
                    pct(row['think_closed_fraction']), pct(row['answer_closed_fraction']),
                    pct(row['format_valid_fraction']), pct(row['truncated_fraction']),
                    f"{row['reasoning_tokens']['mean']:.2f}",
                    f"{row['reasoning_tokens']['p50']:.1f}/{row['reasoning_tokens']['p90']:.1f}/{row['reasoning_tokens']['p95']:.1f}",
                    f"{row['answer_tokens']['mean']:.2f}", f"{row['total_tokens']['mean']:.2f}"])
    val_rows = [[r['global_step'], r['scope'], r['examples'], f"{r['loss']:.6f}",
        f"{r['by_source']['medical_o1']['loss']:.6f}", f"{r['by_source']['huatuo']['loss']:.6f}"] for r in validations]
    resources = table(['指标', '实际值', '范围'], [
        ['唯一训练样本', formal['unique_examples'], '两源各10000；每个ID一次'],
        ['总/监督tokens', f"{formal['processed_tokens']:,} / {formal['supervised_tokens']:,}", '真实非padding序列 / assistant监督'],
        ['更新耗时', f"{formal['update_seconds']:.2f} s ({formal['update_seconds']/3600:.3f} h)", '累计forward/backward/optimizer更新阶段'],
        ['有效训练吞吐', f"{formal['effective_tokens_per_update_second']:.2f} tokens/s", '总非paddingtoken / 更新耗时'],
        ['formal worker耗时', f"{formal['attempt_wall_seconds']:.2f} s ({formal['attempt_wall_seconds']/3600:.3f} h)", '当前attempt加载、验证、保存、训练；若多attempt须另核总wall'],
        ['NVML峰值', f"{formal['nvml_peak_bytes']/2**30:.3f} GiB", '全formal worker，raw bytes见summary'],
        ['活跃张量峰值', f"{formal['allocated_peak_bytes']/2**30:.3f} GiB", 'PyTorch allocated peak'],
        ['缓存预留峰值', f"{formal['reserved_peak_bytes']/2**30:.3f} GiB", 'PyTorch reserved peak'],
        ['checkpoint计时', f"{len(checkpoint_events)} 次，共 {sum(r['seconds'] for r in checkpoint_events):.2f} s；最长 {max(r['seconds'] for r in checkpoint_events):.2f} s", '实际物理attempt保存事件'],
        ['峰值时显存余量', f"{formal['remaining_at_peak_bytes']/2**30:.3f} GiB", 'NVML总量减观察峰值'],
        ['新进程HF重载+4条生成', f"{reload['elapsed_seconds']:.2f} s", '单独诊断，不计进formal更新吞吐'],
        ['配对vLLM生成', f"{evaluation['wall_seconds']:.2f} s", '含cold load和identity probe'],
    ])
    sources = table(['source', 'revision', 'raw SHA256'], [[r['source'], r['revision'], r['sha256']] for r in raw['sources']])
    clean_rows = []
    for source in ('medical_o1', 'huatuo'):
        c = data['counts']
        clean_rows.append([source, c[f'{source}:raw'], c.get(f'{source}:encoding_corruption', 0),
            c.get(f'{source}:extreme_repetition', 0), c[f'{source}:benchmark_cluster_overlap'],
            c[f'{source}:sft_duplicate_cluster'], c[f'{source}:clean_unique'], 10000, 500])
    scenario_rows = []
    for name, scenario in calibration['estimated_scenarios'].items():
        v = scenario['gpu_hours']
        scenario_rows.append([name, *[f'{v[k]:.2f}' for k in ('stage1_measured_worker', 'stage2', 'stage3', 'vanilla', 'dynamic', 'stage5', 'stage6')], f"{scenario['total_gpu_hours']:.2f}"])
    high = cases['training_diagnostics']
    attempt_count = len(list(bulk.glob('attempt_*/metrics.jsonl')))
    assert attempt_count == 1, 'Report writer must reconcile multi-attempt wall time before reporting total cost'
    adapter_hash = sha256(Path(formal['final_adapter']) / 'adapter_model.safetensors')
    assert adapter_hash == formal['adapter_sha256']
    text = f'''# Stage 1 — Medical SFT 实验报告

正式 run `{selected['formal']}` 已实际完成20,000个唯一训练样本的一完整epoch，覆盖率100%，共1250次有效批次更新。本报告依据完整训练、独立重载、1,000条验证和配对生成产物编写；阶段最终状态与全局收据以 `project_state.json` 和 `experiments/stage1/verification-final.json` 为准。Stage 2–6本轮未执行。

本阶段建立可恢复的医疗SFT初始化，并测量输出格式与长度。没有运行CMExam/CMB test评分，没有证明临床正确性、考试准确率提升或后续Dynamic Sampling收益。

## 1. 数据源与来源固定

Medical-o1选择中文医疗文件 `medical_o1_sft_Chinese.json`，保留原始Complex_CoT和Response；Huatuo选择固定GPT4-SFT文件，保留原回答，不生成额外CoT。原始数量分别20,171和142,248；实际Huatuo全部为单轮human/gpt对。两个来源分别提供推理风格与开放医疗问答风格，20k是已冻结的初始化预算；这不是经过规模消融得到的最优数量。

{sources}

完整文件URL、bytes与SHA见 [raw manifest](../../experiments/stage1/{selected['data']}/raw_manifest.json)。固定上游metadata将Qwen和两种SFT源标为Apache-2.0，先前审计见 [upstream findings](../implementation/UPSTREAM_FINDINGS.md)；CMExam仍按研究用途处理。原始数据/权重只留bulk，Git保留索引、哈希和少量案例。

## 2. 清洗、去重与隔离

先投影CMExam train54497、val6811、test6811以及CMB-Exam test11200，共79,319条question-only exclusion。测试答案和difficulty不进入投影、训练或调参；只做预先文本隔离，不根据测试表现调整阈值，也不删除测试题。这里“没有使用test调参”不等于“从未下载带其他列的原始CSV”：下载保留原始字节，解析只读取指定题干字段。

结构清洗检查实际schema/role、空值、编码异常、明显非医疗文本、极端重复及控制标记。规范化采用NFKC、小写、空白/标点/前导题号处理，保留数字、小数、符号、单位与否定；只有题干后存在多个明确选项标记才剥离选项。精确question hash后，以完整char3 Jaccard>=0.65检索候选，char5 Jaccard>=0.85或SequenceMatcher>=0.90确认近重复。后者只在char3候选边界内适用。含benchmark的整个连通簇中的SFT成员均排除，包括传递关系。

{table(['source', 'raw', '编码', '重复文本质量过滤', 'benchmark簇排除', 'SFT簇去重', 'clean unique', 'train', 'val'], clean_rows)}

完整图有10,803条exact、2,990条SequenceMatcher、696条char5确认边。源内边与跨benchmark边均保留；medical-o1与Huatuo直接边实际为0，不能凭空声称两种SFT源之间移除了近重复。489条benchmark关联SFT记录是簇成员数，不是直接边数。原拒绝记录保留；补充 `rejections_resolved.jsonl` 用真实相邻边给出分数，明确相邻边目标和最终cluster代表可能不同。

规范化曾错误把“A、B两种药物”视为选项，导致首个治理run失败；修复发生在数据冻结前。官方CMExam train另有两条纯标点题干，保留ID和异常记录，没有可用词法匹配信息，未用答案修复。保守相似度还会把数字不同的高度相近题型归为同族，例如85%/95%的边界；这有误排风险。词法隔离不是语义改写零泄漏的证明。

## 3. 固定划分与token统计

SFT簇代表优先较小的medical-o1源，同源按seed42哈希确定；在clean代表上使用独立seed42 split哈希排序，每源前500为val、随后10000为train。没有复制、重复采样或降低过滤规则凑配额。训练与验证ID/cluster互斥，并与exclusion簇隔离。原始到messages、native模板、input IDs和labels在独立核验中对全部21,000条重新计算。

{table(['source', 'token范围', 'total', 'mean', 'P50', 'P90', 'P95', 'P99', 'max'], token_rows)}

验证集另有456,624 total /385,092 supervised tokens。长度统计覆盖选中train/val及选择时实际扫描的21k候选，没有把它冒充全142k源数据的分布。所有选中样本最长1558，低于2048，因此无截断、无超长替补。源诊断另外发现未选中Huatuo记录长2477/2200/2101，说明直接硬截断确有丢掉最终回答的风险；策略是整例拒绝并同源补足。

medical-o1平均总长566.98、reasoning316.09，reasoning P95=454/P99=540/max868；Huatuo平均总长349.87、reasoning恒0。两源各10k并不等于token loss权重一半一半：medical-o1占监督token的{pct(4947754/7741165)}。不能由此进一步声称梯度范数贡献等于该百分比。

## 4. 输出格式与label mask

实际Qwen tokenizer的 `apply_chat_template(enable_thinking=True)` 负责role边界；不手写替代ChatML。medical-o1 target为原CoT置于`<think>...</think>`，原Response置于`<answer>...</answer>`；Huatuo保留空think与原回答。未使用其他模型编推理。

system/user/header的labels为-100，实际assistant内容及EOS参与loss，padding按有效长度mask；PAD与EOS即使同ID也不误删真实EOS。多轮fixture验证Qwen native模板会去掉历史assistant的think块、保留历史answer/EOS；实际训练数据为单轮，不能报告成多轮CoT训练。Transformers5.5.3返回类型通过显式`return_dict=False`固定。

完整data verifier重算21k原生编码与监督位置；CPU梯度oracle比较microbatch1/2/4与整批token-mean loss及更新。正式不packing，保留明确样本边界、coverage和EOS追踪。

## 5. 模型、LoRA与正式配置

主干为官方`Qwen/Qwen3-8B`，revision `{config['model_revision']}`，BF16，未量化。它是已经后训练的混合thinking模型，不是`Qwen3-8B-Base`预训练权重；本文“Base”仅指加入本项目医疗adapter之前的同一官方checkpoint。选择它沿用固定研究主干，未做Instruct/Base消融。

LoRA r32、alpha64、dropout0，覆盖q/k/v/o/gate/up/down七种projection，共87,293,952个FP32可训练参数；base权重保持BF16。早期alpha32规划明确在D-017改选64，与已验证的PEFT架构一致；没有证据证明64优于32。后续Vanilla/Dynamic必须从同一最终SFT adapter和同构设置初始化。

AdamW LR1e-4，betas0.9/0.999、eps1e-8、weight decay0；cosine，warmup3%（38个计划更新），grad clip1，seed42。microbatch4×accumulation4，有效16；仅固定16条内部按长度排序以减少padding。目标函数是整批有效assistant token的平均NLL，gradient checkpointing开启，SDPA，max sequence2048，packing=false，`PYTORCH_ALLOC_CONF=expandable_segments:True`。

正式由fresh base +新LoRA/optimizer启动，不延续smoke/pilot。全部源文件/config/data哈希在clean commit `{manifest['git_commit']}` 冻结，训练中未改超参或数据。第一更新LR=0是原生warmup行为，其样本/前后向照实记录；非零adapter变化由全程digest与独立logits证据证明。完成条件是20k唯一ID的epoch cursor，关闭early stopping，不按短max_steps停机，也不按validation选择提前checkpoint。

## 6. Smoke、pilot与失败保留

| Run ID | class | 实际预算 | 主要结果 |
| --- | --- | --- | --- |
| `s1_data_20260908T144417_418b80` | DIAGNOSTIC | 首次全量治理 | FAILED，规范化空题干；原日志保留 |
| `{selected['data']}` | DIAGNOSTIC | 20k train +1k val | 数据及完整token核验PASS |
| `{selected['smoke']}` | SMOKE | 128例/8更新 | 有限loss/grad、adapter更新、重载成功；4条greedy全部1024截断 |
| `{selected['pilot']}` | PILOT | 1024例/64更新 | 实际新进程恢复精确；4条重载格式/EOS通过 |
| `{selected['memory']}` | DIAGNOSTIC | 重放32更新+最长16条1更新 | 无OOM，更新参数一致；内存历史混杂见下文 |
| `{selected['formal']}` | FORMAL | 20,000例/1250更新/1epoch | 实际100%覆盖、完整最终adapter |
| `{selected['evaluation']}` | EVALUATION | 50固定val × Base/SFT × {len(evaluation['limits'])} cap | 原始生成、身份校验、格式/长度实测 |

Smoke update-phase吞吐{smoke['effective_tokens_per_update_second']:.2f} tokens/s；pilot为{pilot['effective_tokens_per_update_second']:.2f}。两者样本数、batch形状和验证/加载比例不同，不能当作严格吞吐消融。pilot的128条验证NLL从2.046165到1.452088；并非正式1000条结果。

下载bootstrap最初请求不存在的CMExam `test.csv` 返回404，随后改为固定revision中的 `test_with_annotations.csv`。早期bootstrap开始时间未采集，保持missing，不从mtime补造。首次data-case audit未写worker status文件，后来依据原PASS summary并独立核对原artifact SHA补记了状态；原结束时间和exit code仍为unknown，旧UNKNOWN索引快照保留。初期部分run仅存源码哈希，后续正式manifest有完整source archive；旧部分artifact seal和失败记录均保留，最终全量索引另行生成。

## 7. 实际resume与checkpoint

pilot在step32/cursor512写完整checkpoint后退出；独立进程恢复optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、LoRA和已消费ID，再执行step33。与原进程的reference-only step33相比，参数最大绝对差0、loss差0、下一批ID一致、scheduler一致。reference-only更新不计入canonical覆盖，真实attempt与canonical成本分开保留。证据：[resume receipt](../../experiments/stage1/{selected['pilot']}/resume_receipt.json)。

formal本次有{attempt_count}个训练attempt，未人为中断制造第二份“恢复成功”。每100updates或900秒安全边界原子保存，临时目录、文件SHA、COMPLETE marker、fsync后rename；保留全部valid checkpoint，包括previous/latest/final。最终路径 `{formal['final_checkpoint']}`。验收会重新hash全部保存点、加载最终optimizer/scheduler/RNG/cursor与canonical metrics，不把“目录存在”当作恢复证据。

## 8. 正式loss、梯度与覆盖

{table(['step', 'scope', 'val例数', '全体token NLL', 'medical-o1 NLL', 'Huatuo NLL'], val_rows)}

初始与最终同1000条验证NLL为{initial['loss']:.6f}→{final['loss']:.6f}。中间128条是各源64条固定monitor，不能将其数值直接与1000条连成同一评价集合的改善曲线。模型选择固定使用最终预算adapter；这些验证值没有用于early stop。

正式更新loss范围{high['loss_min']:.6f}–{high['loss_max']:.6f}，前50/后50更新均值{high['first50_mean']:.6f}/{high['last50_mean']:.6f}；裁剪前grad norm范围{min(r['grad_norm'] for r in metrics):.6f}–{max(r['grad_norm'] for r in metrics):.6f}。事后Q3+3IQR高loss标记阈值{high['descriptive_high_loss_threshold']:.6f}，标记{high['high_loss_batches']}个batch；这是案例检索，不是正式稳定性门槛或早停规则。每条指标对应16个sample IDs，不能把batch loss捏造成单样本loss。

原始loss/grad/LR/样本/token/cursor均保留；处理{formal['processed_tokens']:,} total和{formal['supervised_tokens']:,}监督tokens，与完整冻结train统计相等，20,000唯一ID无遗漏、无重复计数。没有以早期下降趋势代替完整预算。

![完整训练曲线](../../experiments/stage1/figures/sft_training.png)

## 9. 吞吐、显存与单卡取舍

{resources}

含smoke、pilot、memory诊断、各次HF重载和最终配对生成的已记录互不重叠GPU占用阶段，合计至少{calibration['observed_stage1_costs']['recorded_gpu_occupancy_lower_bound_hours']:.3f}h。phase逐项秒数见compute calibration。pilot早期attempt未保留完整结束wall，故该attempt只累加实际计时的load/update/validation/checkpoint（含reference-only update成本），未计时的save/digest/退出尾部仍unknown；不把下界写成全阶段精确总耗时。CPU数据下载/清洗/只读审计不计GPU小时。

GPU为RTX5880 Ada，系统可见约44.99GiB，单卡；不是多GPU/FSDP训练成果。正式预算Stage0工作估计约9.84h，真实token分布比当时1024均长假设短，且真实microbatch优化后的吞吐更高，最终以本表实测为准。update-phase不含独立HF/vLLM验证成本，不能把训练tokens/s写成生成tokens/s。

pilot曾出现NVML峰值44.89GiB、活跃张量仅约26.53GiB的缓存压力，没有发生OOM。memory诊断的32步loss和参数与pilot一致，最长16条更新峰值32.38GiB；但诊断省略了pilot开头128条validation，不能把整体峰值差全部归因于allocator开关。正式进程包括1000条初始验证，故上表的全程峰值是更直接的配置可行性证据。优化保留了有效batch、token加权目标和样本预算，未靠量化或换小模型。

## 10. 独立重载与配对生成

最终adapter SHA256：`{adapter_hash}`。新HF进程重载trainable digest与最终checkpoint完全一致；Base/adapter logits最大差{reload['identity_logit_max_delta']:.6f}，4条greedy诊断格式通过{reload['format_valid_count']}/4、EOS通过{reload['eos_terminated_count']}/4。这个greedy检查用于重载sanity；正式行为比较使用冻结的采样协议。

配对 run `{selected['evaluation']}` 对两源各25条固定validation，保持原始同prompt，无额外格式指令，native thinking=True，temperature0.6/top_p1/top_k-1/seed42。先cap1024；只有SFT answer closure<95%或截断>5%时，双方同50题再测2048。本次实际执行caps={evaluation['limits']}，没有用test回答选择长度。

vLLM为BF16 TP1、max_model_len4096、max_num_seqs16、memory0.65、eager、LoRA r32、无prefix cache；native sampler、V1、batch-invariant并断言LoRA shrink split_k=1。Base/SFT/Base-negative/SFT-repeat identity控制中Base/SFT prompt logprob最大差{evaluation['identity']['base_sft_prompt_logprob_delta']:.6f}，SFT重复差{evaluation['identity']['repeat_sft_prompt_logprob_error']:.6g}。生成audit对{generation_audit['outputs_redecoded_and_remeasured']}条raw output IDs重新decode，重算格式/长度/重复及按源聚合，不只相信summary。

{table(['cap', '模型', '来源', 'n', 'think闭合', 'answer闭合', '严格格式', '截断', 'reason均长', 'reason P50/P90/P95', 'tagged answer均长', 'output均长'], generation_rows)}

这里answer length只统计完整`<answer>`块。Base的无标签正文不等于没有语义答案，不能把其tagged answer_tokens=0解释为医学回答能力为零。开放think未闭合时把剩余内容计入reasoning，以反映截断成本。EOS与length finish reason保存在raw，不从字符串猜测。

在推荐候选cap{candidate}下，Base/SFT平均reasoning分别{pre['all']['reasoning_tokens']['mean']:.2f}/{post['all']['reasoning_tokens']['mean']:.2f}，平均总输出分别{pre['all']['total_tokens']['mean']:.2f}/{post['all']['total_tokens']['mean']:.2f}。这些是同50题的描述统计，单seed、单响应，没有多seed置信区间；格式、长度变化不能推出医学正确率变化。

SFT在Huatuo提示上的非空reasoning比例{pct(post['huatuo']['reasoning_present_fraction'])}、均长{post['huatuo']['reasoning_tokens']['mean']:.2f}；在medical-o1上分别{pct(post['medical_o1']['reasoning_present_fraction'])}、{post['medical_o1']['reasoning_tokens']['mean']:.2f}。这回答实际是否跳过reasoning/是否仍生成推理；不能仅据混合训练前后比较，将变化单独因果归于empty-think数据，需要同预算去掉/替换来源的对照才可隔离作用。推理是否可读与个别错误见人工案例，未进行全量临床审定。

## 11. Bad / good / boundary案例与数据限制

自动案例按冻结prompt顺序保留每种最多3例，完整输出仍在bulk。长度减少标签为SFT<=Base65%且格式完整，Huatuo长推理为>512tokens，near-empty tagged answer为<=3tokens；这些是成本/结构标签，不是医学判错规则。类别总数及NOT_OBSERVED/NOT_ASSESSED分别见 [case coverage](../../experiments/stage1/case_coverage.json)。混合正确性group边界本轮未测，不能编造Stage3案例。

一个关键反例是 `S1-SOURCE-REASONING-001`：选中最长CoT（medical-o1 row14099，868 reasoning tokens）把3.9921875+4错写为9.9921875，并给出前后矛盾的累积浓度推导。按题设纯数学递推C_n=C_(n-1)/2+4，应趋向8，原文末尾“不可达10”的方向正确却又错误写“接近10”。这证明结构/长度/去重合格不等于推理正确。样本由最长长度选出，不能推算全数据错误率。它来自原始数据而非传输或mask损坏；发现时正式数据已冻结，保留原baseline和观察，未事后删例或修改epoch。

下面为实际人工读过的配对案例；其观察与aggregate同时保留，不以少数好案例代替完整分布。

{manual_review}

## 12. 决策与可辩护结论

D-017冻结alpha64和token加权有效batch；D-018冻结词法隔离与source split；D-019选择可扩展allocator，O-019补记初始化validation历史混杂；D-020在clean revision启动独立正式epoch。O-020记录原始CoT算术错误，保留原数据而不事后清洗结果。详见 [决策记录](../implementation/STAGE1_DECISIONS.md)。

本阶段实证是20k完整SFT训练、可核验adapter更新、真实dataloader恢复，以及固定heldout上实测的输出格式/长度变化。训练loss下降说明更符合这些源的监督目标，不能等同医疗事实质量提高；带噪源CoT会进入目标。没有宣称医疗SOTA、临床可用、考试test提升或GSPO已完成。

## 13. 算力重估与Stage 2交接

推荐优先测试 `max_response={candidate}`。候选在当前50条heldout上是否满足95%闭合/5%截断触发条件：{calibration['candidate_meets_heldout_closure_trigger']}。它是下一阶段train-only profiling的候选，不是已证明适合CMExam自然on-policy分布的正式RL设置。

{table(['情景', 'SFT实测h', 'S2估计h', 'S3估计h', 'Vanilla估计h', 'Dynamic估计h', 'S5估计h', 'S6预留h', '合计h'], scenario_rows)}

Stage1 worker时长为Measured；后续数字为Estimated：把本次开放问答SFT-val的长度/decode迁移到考试rollout，mixed acceptance仍假设0.65/0.35/0.10，actor/old-logprob/switch仍是假设。自然acceptance、CMExam post-SFT长度、正式Ray训练循环耗时是Unknown。20k/5000/G4与625-update原规划不变；acceptance趋近0时严格worst-case无有限上界。公式、原始summary哈希见 [compute calibration](../../experiments/stage1/compute_calibration.json)；完整解释追加至COMPUTE_BUDGET，保留Stage0历史估计。

READY_FOR_STAGE2 = YES，含义是Stage1必需产物具备、可进入下一阶段受控smoke/profiling验收；本报告没有启动Stage2正式1k profiling或RL。最终状态仍须以全局verifier PASS收据核定。

精确初始化：`{formal['final_adapter']}`，SHA256 `{adapter_hash}`，固定Base revision `{config['model_revision']}`，r32/alpha64/七projection。该final-budget adapter不按val/test改选，后续Vanilla与Dynamic必须共用。输入继承native Qwen thinking模板，输出目标继承 [OUTPUT_FORMAT_CONTRACT](../implementation/OUTPUT_FORMAT_CONTRACT.md) 的think+answer；Stage2仍需实装严格选项解析和验证初始correctness-gated reward `0.8 R_acc + 0.15 R_acc R_sem + 0.05 R_format`，Dynamic接受指标保持accuracy对比。

开放风险：词法过滤不能排除全部语义同源；原始CoT有事实噪声；开放QA长度不能直接代表考试；MedEmbed/BGE数字/否定反例未解决；自然mixed acceptance与完整GSPO显存/吞吐未知；单机bulk哈希不是外部备份。R06/R07/R09及后续阶段gate仍有效，不能将Stage1 readiness理解为这些风险已关闭。

## 14. 验收映射

| 合同项 | 实际证据 |
| --- | --- |
| 20k与两源10k、独立val1k | frozen IDs、raw/clean/cluster manifest、全21k重新编码receipt |
| 全epoch、无缺失重复、有限loss/grad | canonical 1250 raw updates、coverage IDs、全部token计数、最终state.pt |
| BF16 Qwen3-8B、r32、配置冻结 | clean formal manifest、source archive、env、config SHA |
| 真实smoke/pilot与resume | selected run摘要、physical attempt logs、step33 reference/恢复receipt |
| adapter更新/重载 | 最终safetensors hash、trainable digest、HF logits差与实际生成 |
| 验证与生成行为 | 全1000初始/最终NLL、50同prompt Base/SFT、raw ID/text重算audit |
| 失败/案例/报告/面试 | 原失败run、case coverage、人工案例、本报告及RESUME_EVIDENCE |
| 全局验收 | `scripts/verify_stage.py --stage 1`生成的最终PASS收据；缺少时不得标VERIFIED/DONE |

## 15. 面试故事：30秒版本

我在单张RTX5880 Ada上完成Qwen3-8B的医疗LoRA SFT：两个来源各1万条，经过全考试题干隔离、重复簇清洗和原生tokenizer mask核验后，完整训练20k一轮，处理{formal['processed_tokens']:,}tokens。用真实新进程恢复对照验证optimizer和数据cursor，再对同50条验证prompt比较训练前后格式与推理长度。结果和最终adapter都能从原始记录核验；同时保留了原始CoT算术错误，说明loss或格式改善不等于医学正确性。

## 16. 面试故事：2分钟版本

这个阶段的难点先在数据。medical-o1和考试题可能同源，不能直接下载20k就训练。我先对全CMExam三个split和CMB-Exam test做question-only隔离，不读取答案或难度参与选择，用精确hash和近重复连通簇排除关联SFT成员，再从两源分别固定10k训练和500验证。实际数据全部21k原生tokenize，没有截断最终答案；user/system被mask，assistant内容和EOS被监督。

训练使用BF16八十亿级Qwen、r32/alpha64 LoRA，有效batch16、microbatch4，按整批assistant token加权，避免不同长度microbatch被错误等权。pilot出现缓存显存接近整卡的问题，我记录活跃张量与预留的差别，采用可扩展allocator，并保留初始validation历史混杂这一限制。正式全程峰值{formal['nvml_peak_bytes']/2**30:.2f}GiB，更新吞吐{formal['effective_tokens_per_update_second']:.1f}tokens/s。

我还在真实pilot中退出进程，恢复完整optimizer/scheduler/RNG/数据cursor，再与连续参考执行同一步，参数和loss差都是0。之后正式从base重新开始，按20k唯一样本完成一整轮；重载最终adapter并做全1000验证、同50prompt配对生成。全val NLL从{initial['loss']:.4f}到{final['loss']:.4f}，但我只据配对raw输出讨论格式和长度，不把它宣传成考试或临床提升。最长CoT中实际存在算术错误，也写进报告。这套证据为下一阶段选择response cap和固定共同SFT初始化提供依据。

## 17. 深挖问答与限制

- **为什么用Instruct而不是Base？** 实际固定主干是后训练Qwen3-8B，保留其指令/思考能力并做领域适配；没有Base消融，不能声称该选择最佳。“Base对照”指医疗SFT之前的同checkpoint。
- **为什么20k、为什么这两个来源？** 合同给定紧凑初始化预算，分别提供推理和开放QA风格，各10k避免样本数一边倒。规模与配比未调优，token权重实际不同。
- **怎样避免CMExam/CMB leakage？** 在选择前对完整题干索引做精确/近重复簇隔离，test答案和难度不用；仍有语义漏检与保守误排限制。
- **为什么train_on_inputs=False？** 训练目标是assistant回答，不优化复述用户提示。按真实token边界mask，EOS/PAD按位置区分。
- **为什么不用packing？** 初版优先清晰的样本覆盖、注意力/label边界和恢复cursor；没有测到packing收益，不虚构比较。
- **LoRA为什么r32、alpha64？** r32保留既定单卡架构；alpha64沿用已实测PEFT设置并通过pilot，未做32/64优劣结论。
- **为何LR1e-4、1epoch？** pilot有限loss/grad、验证与重载行为支持冻结起始LR；1epoch是初始化合同，不是val最优选点，不early stop。
- **数据为何不是越多越好？** 词法重复、源CoT错误及风格比例会改变学习目标；增加规模前应先建立可验证质量和公平新版本比较。
- **如何resume？** 原子checkpoint保存LoRA+Adam+scheduler+多种RNG+canonical IDs/cursor；不同进程恢复下一真实batch，与参考参数/损失逐项比对。
- **如何确认adapter真的训练？** 初末trainable digest变化，最终权重SHA，独立HF重载digest一致，以及Base/SFT logits和vLLM身份控制均实际不同。
- **SFT怎样影响RL rollout成本？** 使用同prompt测得think/answer闭合、reasoning/总长和LoRA decode；cap{candidate}作为下一阶段候选，但考试分布和mixed acceptance还必须实测。
- **最大的限制？** 原始推理未全量事实审定；单seed、无临床/考试准确率结论；50条源内验证可描述风格，不能证明泛化。更多资源应优先审计数据事实质量、扩展独立验证与预注册对照，不以挑选好案例代替它们。

简历可用证据单列 [RESUME_EVIDENCE](../RESUME_EVIDENCE.md)，只有全局verifier通过后才允许标为VERIFIED。GSPO、Dynamic Sampling、最终准确率和服务指标不在本阶段成果中。
'''
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
    print(f'Wrote {len(text)} characters from completed evidence to {args.output}')


if __name__ == '__main__':
    main()
