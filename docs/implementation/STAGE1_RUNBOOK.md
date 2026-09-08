# Stage 1 当前正式运行与恢复

当前 formal run：`s1_formal_20260908T152119_60040b`。冻结代码 `f591a9e`；初始 adapter 为新建LoRA，不是任何pilot checkpoint。全部20k、单完整epoch、r32/alpha64、BF16、microbatch4×accumulation4、LR1e-4、expandable_segments=True。

Bulk目录：`/data/WSH/medical-post-train-artifacts/runs/s1_formal_20260908T152119_60040b`。正式进度仅见该目录 `progress.json` / `heartbeat.json`；Git的state是阶段索引，不是每5秒写入的监控数据库。`status=FULL_RUNNING`不表示完成。

```bash
.venv-train/bin/python scripts/run_stage1.py inspect --run /data/WSH/medical-post-train-artifacts/runs/s1_formal_20260908T152119_60040b
```

工作进程独立session运行，stdout/stderr与所有状态在bulk。无需保持Codex/SSH连接。每5秒heartbeat；每100updates或900秒安全边界保存完整恢复点。进程遇SIGTERM/SIGINT时先完成当前有效batch并checkpoint，再PAUSED；硬崩溃按最近完整恢复点回滚，失败attempt原始物理消耗保留。

如果恢复会话，第一件事是检查**同一run**状态/PID/heartbeat和最新有效checkpoint；不可另起一个短run替代正式预算。恢复入口自动检查COMPLETE marker、文件hash、config、adapter/optimizer/scheduler/RNG/样本游标：

```bash
.venv-train/bin/python scripts/run_stage1.py resume --run /data/WSH/medical-post-train-artifacts/runs/s1_formal_20260908T152119_60040b --checkpoint <latest_checkpoint.json中的确切目录>
```

同一formal run恢复时必须保留冻结source/config，不能拼接不同实现。原始source.zip与每attempt provenance在bulk；若工作区发生相关代码修改，应在冻结commit的隔离checkout恢复。optimizer更新的canonical lineage与物理attempt日志分开，只有真正消费的唯一训练ID计入epoch。

正常终止必须有：20,000 covered IDs / 无missing、1250成功updates、全部9,168,530 total tokens / 7,741,165 supervised tokens、final完整checkpoint。仍须执行新进程reload、固定50条Base/SFT生成、报告/案例/面试材料及`verify_stage.py --stage 1`，之后才可 VERIFIED/DONE。

```bash
.venv-train/bin/python scripts/check_stage1_adapter.py --run /data/WSH/medical-post-train-artifacts/runs/s1_formal_20260908T152119_60040b --count 4
.venv-train/bin/python scripts/evaluate_stage1.py --run /data/WSH/medical-post-train-artifacts/runs/s1_formal_20260908T152119_60040b --protocol configs/stages/s1_evaluation.json
python scripts/plot_stage1.py --run /data/WSH/medical-post-train-artifacts/runs/s1_formal_20260908T152119_60040b
```

最后两条命令消费真实完成结果；不能提前填充报告。plot使用已存在的base Python matplotlib，只做分析、不改base环境。Stage 2–6保持NOT_STARTED，当前授权只完成Stage1。
