#!/usr/bin/env python3
"""Render the Stage3 report only after full raw analysis and qualitative review."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.evidence.stage2 import read
from medical_posttrain.evidence.stage3 import INDEX


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    selected=read(INDEX/'selected_runs.json');formal=selected['formal'];smoke=selected['smoke']
    out=Path(read(INDEX/formal/'manifest.json')['artifact_root']);s=read(out/'summary.json');c=s['costs'];counts=s['group_counts']
    assert s['status']=='FULL_PASS' and s['accepted_mixed_groups']==256
    sr=read(INDEX/smoke/'summary.json');review=read(INDEX/'manual_review.json');cal=read(INDEX/'compute_calibration.json');overlap=read(INDEX/'stream_overlap.json')
    cases=read(INDEX/'case_coverage.json');notes=read(INDEX/'frontier_case_notes.json')
    assert len(review['entries'])>=50 and 2<=len(notes['cases'])<=5
    header=table(['正式实测指标','数值'],[
        ('generated group attempts / generated groups',s['generated_groups']),('valid groups',s['valid_generated_groups']),
        ('all-wrong / mixed / all-correct',f"{counts['all_wrong']} / {counts['mixed']} / {counts['all_correct']}"),
        ('accepted mixed / overflow eligible',f"256 / {s['overflow_mixed_groups']}"),('invalid groups',s['invalid_groups']),
        ('completed trajectories',s['completed_trajectories']),('sampling amplification',f"{s['sampling_amplification']:.6f}×"),
        ('attempt amplification',f"{s['attempt_amplification']:.6f}×"),('generated output tokens',c['generated_output_tokens']),
        ('accepted-group output tokens',c['output_tokens_by_disposition']['accepted']),('rejected-group output tokens',c['rejected_output_tokens']),
        ('overflow output tokens',c['output_tokens_by_disposition']['overflow_eligible']),('generation wall seconds',f"{c['generation_wall_seconds']:.6f}"),
        ('active GPU worker seconds',f"{c['gpu_worker_active_seconds']:.6f}"),('generation output tokens/s',f"{c['output_tokens_per_second']:.6f}"),
        ('accepted rollout token fraction',f"{c['accepted_token_fraction']:.4%}"),('optimizer updates',0)])
    distribution=table(['正确数','Stage2 profiling：1000组','Stage3 generated valid','Stage3 accepted'],[(f'{k}/4',[235,152,154,225,234][k],s['correct_count_distribution'][str(k)],s['accepted_correct_count_distribution'].get(str(k),0)) for k in range(5)])
    token_table=table(['Disposition','output tokens','均长','P95','P99'],[(k,c['output_tokens_by_disposition'][k],*(f'{s["lengths_by_disposition"][k][v]:.3f}' if s['lengths_by_disposition'][k][v] is not None else 'N/A' for v in ('mean','p95','p99'))) for k in c['output_tokens_by_disposition']])
    subtype_table=table(['mixed slice','全部 mixed','accepted mixed'],[(k,s['mixed_subtypes'].get(k,0),s['accepted_mixed_subtypes'].get(k,0)) for k in ('mixed_parsed_wrong','mixed_unparseable_only','mixed_both')])
    card_table=table(['Ground-truth answer cardinality','all-wrong','mixed','all-correct'],[(k,*[v[x] for x in ('all_wrong','mixed','all_correct')]) for k,v in s['answer_cardinality_by_class'].items()])
    p=s['group_fractions']['mixed'];unparsed=s['accepted_mixed_subtypes'].get('mixed_unparseable_only',0)
    frontier='\n\n'.join(f"### {entry['title']}\n\nGroup `{entry['group_id']}`。{entry['observation']}\n\n{entry['interpretation']}" for entry in notes['cases'])
    text=f'''# Stage 3 — Dynamic Sampling 与真实 refill 集成

{header}

正式 run：`{formal}`。来源：[完整 summary](../../experiments/stage3/{formal}/summary.json)、[逐组摘要](../../experiments/stage3/groups.json)、[成本](../../experiments/stage3/cost_summary.json)、[验收 receipt](../../experiments/stage3/verification-final.json)。本阶段固定 Stage1 SFT policy，完成256个真实 mixed groups 的 sampler 集成；没有任何 policy optimization 或准确率提升实验。

## 1. 定义、有效性与奖励冻结

Stage2的1000×4 profiling发现53.1%的组具有可验证正确性差异。本阶段要验证新生成、奖励、过滤和补采样能否形成一个可恢复、可计量的真实循环。完整G4组中只有 `0 < sum(acc) < 4` 才取得未来训练窗口的资格，0000和1111均拒绝。决策函数仅接收binary acc；semantic、format、total_reward和长度的变化不能改变资格。

先判断validity：恰好4条完成的stop/length轨迹、4个唯一trajectory IDs、相同prompt/group/policy/config/reward版本，且acc为0或1。缺失、重复、跨题、跨policy和transport error单列invalid，不能归成all-wrong。所有16种binary pattern、奖励/长度扰动、invalid、cyclic stream、overflow、starvation和状态恢复有确定性单测。全仓库98项测试通过，包括真实发现的摘要写入回归测试。

继承 Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`，Stage1 final-budget r32/alpha64七投影adapter SHA `1601e97891e51940bd4b575d8811a77d8278cbeb296c044b7004e41da6d9ea64`。policy_version固定 `cdb99bc34f05c71a7ec6411c19e7091562e8f8b3277961ceb83d33e8393a9ff1`。G4、temperature0.6、top_p1、top_k−1、max_response1024不变；奖励仍为 `0.8acc + 0.15acc·sem + 0.05format`，BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`，reward manifest SHA `cc72be1c2348d20f58bf978d047b9b4a2f200d8c70c39f0d0d98506528bfb905`。原六个核心模块不改写。

## 2. Controller、stream 与上游关系

沿用[verl官方DAPO说明](https://verl.readthedocs.io/en/latest/algo/dapo.html#dynamic-sampling-with-group-filtering)的acc过滤/重复生成语义。本阶段复用冻结的vLLM、parser、BGE和hybrid reward，新增持久化控制器；没有执行包含actor更新的DAPO trainer，也没有引入overlong reward、clipping或token-loss变化。详见 [Stage3决策与计划](../implementation/STAGE3_PLAN.md)。

每批16题，engine max_num_seqs16，生成后完整计分并按stream顺序处理。正式target256；最大128批用于starvation保护，未达到目标就BLOCKED，不能无限循环或减预算。实际执行{s['refill_batches']}批。达到256后同批额外mixed标overflow_eligible，成本完整保留，停止补采样。

candidate pool继承Stage2固定15000题，SHA `6fa3e114f47c51977d940ca01b4e0d6510a03a390fe1f1495fe5d0820bf41262`。全部题按seed42、domain `stage3:formal`、epoch和prompt_id的SHA确定顺序；遍历完会进入新epoch的完整排列。没有prompt黑名单或基于历史结果改变概率。prompt_id代表源题；group_id含run/policy/encounter，重复遭遇仍是新组。未来policy更新须开启新的policy window，旧版本轨迹不能混入。

生成唯一prompt {s['exposure']['generated_unique_prompts']}，接受唯一prompt {s['exposure']['accepted_unique_prompts']}，最大曝光{s['exposure']['max_prompt_exposure']}；曝光直方图 `{s['exposure']['prompt_repeat_histogram']}`。结束epoch/cursor={s['exposure']['candidate_epoch']}/{s['exposure']['candidate_cursor']}。正式题与Stage2 profiling重叠{overlap['stage2_formal_prompt_overlap']}题，与Stage3成功smoke重叠{overlap['smoke_formal_prompt_overlap']}题。重叠仅指题号：formal独立run、计数器和domain，所有回答均重新生成，未使用Stage2的531个旧mixed凑数。

## 3. Smoke、真实中断与失败保留

成功smoke `{smoke}`：32题×4，实际mixed/accepted={sr['accepted_mixed_groups']}。第一批16组后保存checkpoint，外部SIGTERM终止拥有的worker进程组，观察旧进程退出、显存恢复15MiB，再启动新PID。新进程恢复同一run的accepted/generated、epoch/cursor、next encounters、dispositions、seed和token ledger；第二批从第17题开始，无重复group IDs。[Smoke验收](../../experiments/stage3/smoke_verification.json)包含raw重算和真实restart gate。

每批保留reservation、raw output/token IDs、BGE vectors/编码元数据、scored responses、commit及state_after。提交后重启以commit重放为准；raw已存但尚未计分可复用同一输出。若生成中断且raw尚未返回，成本不可知则保留FAILED，不静默重试补0。此次真实恢复测试发生在已提交批次边界；未宣称验证所有断电/文件系统故障位置，也未宣称跨进程随机生成逐token完全一致。

保留两个失败：`{selected['failed_smoke']}`完成32组后摘要writer重复run_id导致TypeError，durable commit未丢失；`{selected['failed_startup']}`因前一个失败worker残留显存，在任何generation请求前被vLLM空闲显存保护拒绝。修复摘要合并、显式engine_core.shutdown、异常后的owned-group清理和启动前显存门槛。成功smoke重新执行，不将失败run升级为PASS。失败raw/日志、恢复出的成本与system cases均纳入最终seal。

## 4. 正式分布与Stage2比较

{distribution}

Stage3 observed P_mixed={p:.4%}，Stage2为53.1%；Stage2一阶预估1/0.531=1.883239×，Stage3实测valid/accepted={s['sampling_amplification']:.6f}×。两者分母并不完全相同：Stage3精确target的overflow计入generated而不计accepted；有限样本、独立prompt slice与顺序、批次布局也可能造成差异。本次invalid={s['invalid_groups']}，故attempt amplification与valid amplification{'相同' if s['invalid_groups']==0 else '分别记录'}。不能将几个百分点差异解读为policy能力提升。

轨迹准确率{s['trajectory_accuracy']:.4%}只描述这次训练池采样的SFT输出，不是held-out/test评估。valid={s['valid_generated_groups']}={counts['all_wrong']}+{counts['mixed']}+{counts['all_correct']}；mixed={counts['mixed']}=256+{s['overflow_mixed_groups']} overflow；completed valid trajectories={s['completed_valid_trajectories']}=4×valid。所有被拒绝的完整回答与奖励仍保留。

## 5. Token成本与长度

{token_table}

output总数{c['generated_output_tokens']}，accepted token fraction={c['accepted_token_fraction']:.4%}。这是rollout token eligibility fraction，不是GPU利用率、MFU或occupancy。过滤发生在生成之后，被拒绝的{c['rejected_output_tokens']} tokens已付出生成成本；不能声称这些token被节省。

prompt tokens同时记录共享prefill口径{c['prompt_tokens_shared_prefill']}和4轨迹逻辑口径{c['logical_prompt_tokens_all_trajectories']}。policy identity controls另产生{c['control_output_tokens']}输出tokens，不混入正式G4分布。generation wall={c['generation_wall_seconds']:.3f}s，throughput={c['output_tokens_per_second']:.3f} tokens/s；BGE计分={c['semantic_scoring_seconds']:.3f}s；active worker={c['gpu_worker_active_seconds']:.3f}s含模型加载/身份检查/评分，排除人工pause空档，并不等同GPU kernel busy time。正式NVML峰值{max(r['nvml_peak_bytes'] for r in s['runtimes'])/1024**3:.3f}GiB。Stage2请求批次为4题，Stage3为16题，吞吐差异不能单独归因于过滤算法。原始run start/end和各进程日志保留用于其他wall口径。成功smoke与两个失败run的成本也在compute calibration分别保留，全部已返回rollout共{cal['total_recorded_rollout_output_tokens']}输出tokens；失败worker占用与随后startup尝试有时间重叠，不把两者wall直接相加成GPU小时。

全体均长{s['lengths']['mean']:.3f}，P95={s['lengths']['p95']:.3f}，P99={s['lengths']['p99']:.3f}，max={s['lengths']['max']}；截断率{s['truncation_rate']:.4%}。length完成标志仍属于valid generation，无法解析答案时按冻结规则acc0，不额外添加超长惩罚。

![Refill and retained output-token costs](../../experiments/stage3/figures/refill_costs.png)

## 6. Parser contrast 与多选切片

{subtype_table}

全部mixed中包含parsed-wrong的组{s['mixed_with_parsed_wrong']}；accepted中包含parsed-wrong的组{s['accepted_with_parsed_wrong']}，unparseable-only={unparsed}/256={unparsed/256:.4%}。mixed_parsed_wrong表示只有可解析错误答案形成错误侧；mixed_both同时有parsed-wrong和unparseable。三者互斥，两个包含parsed-wrong的类别合并成mixed_with_parsed_wrong。Stage3全部mixed中unparseable-only为78/257=30.3502%。Stage2的153/531=28.8136% unparseable-only以全部mixed为分母，比较时需保持同一口径。

无法解析的response仍是acc0，包括 `[1,1,1,unparseable]` 的mixed。只做analysis slice，不改变资格。如果未来policy修复格式，这部分contrast可能减少，P_mixed和refill成本可能变化；本阶段未验证变化方向。strict format={s['strict_format_rate']:.4%}，unparseable={s['unparseable_rate']:.4%}，ambiguous={s['ambiguous_rate']:.4%}；parse errors `{s['parse_errors']}`。

{card_table}

answer cardinality来自固定ground truth，多选属于稀少描述性子群。无difficulty/category标签、未按结果删题，不能据此断言某临床专科必然更难。

## 7. Correctness variance 与hybrid reward variance

all-wrong有{s['homogeneous_acc_nonzero_hybrid_variance']['all_wrong']}组correctness std=0而hybrid reward std>0；all-correct有{s['homogeneous_acc_nonzero_hybrid_variance']['all_correct']}组。前者可能来自format，后者也可来自semantic差异。因此精确表述是过滤“没有可验证正确性差异的组”，不能笼统称去掉所有zero-advantage groups。本阶段仅保存population reward/correctness std，未计算真实GRPO-style advantage。

## 8. 完整轨迹阅读与policy frontier cases

Agent逐条阅读{len(review['entries'])}条完整轨迹，覆盖accepted、两种reject、1/2/3正确、unparseable-only和多选。来源：[逐条review](../../experiments/stage3/manual_review.json)、[完整阅读材料](../../experiments/stage3/manual_review_packet.txt)、[自动case覆盖](../../experiments/stage3/case_coverage.json)。这是模型行为、parser与奖励的定性审读，不是医学专家审定。

{frontier}

自动case覆盖all-wrong/all-correct、1/2/3 mixed、unparseable-only、parsed-wrong、高semantic全错、hybrid-variance全对、长/短accepted、长rejected、多选全错/混合。未观察类别明确标NOT_OBSERVED，不补造case。自动极端样本不用于估计总体发生率。

## 9. 验收与下一阶段

READY_FOR_STAGE4 = YES。交接物为冻结SFT initialization、Stage2 pool/reward、Stage3纯sampler与bounded refill、resume receipt、完整raw成本账本和[readiness](../../experiments/stage3/readiness.json)。最终 `python scripts/verify_stage.py --stage 3` 从raw输出、tokenizer增量decode、保存的BGE vectors、parser/reward、stream和commit重新验证，不只信summary计数。

验收范围：前置Stage1/2 DONE与合同hash；16-pattern/adversarial/invalid测试；真实32×4 smoke；新generation反复refill至恰好256；同一policy及G4 lineage；全部拒绝/overflow保留；trajectory/counter/token守恒；真实终止与新进程恢复；>=50条阅读与cases；报告/面试/成本/readiness；失败run和最终bulk seal。最终状态在project_state与verifier receipt中记录。Stage4–6保持NOT_STARTED。

[Compute calibration](../../experiments/stage3/compute_calibration.json)用本阶段A、L、T给出Stage4 Dynamic初始生成投影：5000×A×4×L={cal['stage4_dynamic_output_tokens']:.3f} output tokens，约{cal['stage4_dynamic_generation_hours']:.3f}h generation。两条5000组baseline预算均未降低。实际policy会变化，actor/old-logprob、切换、validation和checkpoint成本仍须Stage4实测，不在此决定LR、mini-batch或clip。

## 10. 限制与可辩护结论

本阶段证明固定SFT下acc-only group filtering/refill及其计数/恢复可工作，测得256个eligible groups的真实生成成本。没有训练对照，因此没有准确率提升、update efficiency提升或rollout节省的因果结论。所有accepted仅取得未来update资格，本阶段尚无optimizer consumption。

继承的BGE语义分数对否定、剂量等细节辨别有限；高相似度不是医学正确性。缺参考解释取semantic0，少量非空占位参考仍按冻结版本编码，缺图题保持文本原样。benchmark label可能有噪声，格式合规不能代替事实审查。独立stream不保证与Stage2题号完全不重叠；所有重叠已量化。deterministic stream/seed保证恢复位置，不声称并行随机解码跨进程bitwise一致。

物理resume验收覆盖提交边界，不等同全部in-flight/断电故障已验证。未来每个policy window仍应记录P_all_wrong/P_mixed/P_all_correct、parsed-wrong/unparseable-only和amplification，持续检查policy barrier；不能永久删掉今天全错或全对的题。

# Interview Story

## 30秒

Stage2发现固定SFT约53%的G4组有正确性差异。我实现了基于acc的DAPO-style过滤与真实refill，并在新生成的{s['generated_groups']}组中恰好接受256组，实测放大{s['sampling_amplification']:.3f}倍，accepted只占输出token的{c['accepted_token_fraction']:.1%}。拒绝组也花了生成计算，所以我同时保存其原始回答和成本，并用真实终止/重启验证计数和stream恢复。本阶段只验sampler，没有训练或准确率提升主张。

## 2分钟

这个阶段的核心是为以后Vanilla GSPO和Dynamic GSPO建立可核验的采样差异。对同题4个回答先检查完整性，再只看acc是否同时包含0和1。全错和全对都缺少可验证正确性差异，但不代表hybrid reward完全一样：format与semantic仍可能使reward std非零，所以不能用total reward variance替代acc过滤，也不把reward std叫advantage。

工程上每批16题，固定adapter和奖励，在15000题的确定性循环stream上不断生成、打分、拒绝和refill，直到256个mixed。拒绝是当次group属性，不是永久题目黑名单；未来policy改变必须重新判断。同一批多出的eligible组单列overflow，raw与成本都保留。最终实测生成{s['generated_groups']}组、{c['generated_output_tokens']}输出tokens，放大{s['sampling_amplification']:.3f}倍，说明过滤不能免费省生成计算。

一个容易忽略的细节是无法解析答案按acc0，所以correct与unparseable也会形成mixed。Stage2有28.8%的mixed只来自这种contrast，Stage3 accepted中该比例为{unparsed/256:.1%}；我把parsed-wrong、unparseable-only和both分开记录但不改资格。未来格式改善可能改变refill成本，这需要持续测量。

另一个收获是生命周期与证据写入也需要真实测试：smoke实际暴露摘要重复键和失败worker占显存的问题。失败日志和成本保留，修复后重新smoke，并在16组checkpoint后真正SIGTERM，再由新PID恢复原计数和下一批题。这些证据让Stage4可以检验训练方法，而不是把sampler的丢组、重计数或隐含成本当成算法效果。
'''
    Path('docs/stage_reports/03_dynamic_sampling.md').write_text(text)
    print('Wrote Stage3 report',len(text),'characters')

if __name__=='__main__':main()
