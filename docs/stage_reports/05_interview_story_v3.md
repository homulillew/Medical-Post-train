# Stage5 evidence-based project narrative

The project completed matched Vanilla/Dynamic GSPO training from the same SFT initialization and evaluated validation-selected checkpoints plus equal-update endpoints. On held-out CMExam/CMB, Dynamic had small selected-checkpoint gains but all main paired confidence intervals crossed zero. Equal-update results were nearly tied, while Dynamic used about 2.925 times the rollout tokens. The evidence therefore supports an inconclusive capability effect with a measured cost increase, not a compute-efficiency win.

A major evaluation failure was found in the interface between prompt and answer parser. Apparently severe model regressions changed after correcting falsely rejected explicit answers. The response was to retain every raw output and earlier score, test and version the revised parser, replay all models uniformly, and disclose that correction occurred after test outputs were observed. No checkpoint was reselected and no wrong answer was regenerated.

Open medical QA is still pending independent judgment and real human audit. Generation completion is not clinical-quality validation. The next delivery is a frozen anonymous review package, followed by adjudication and final Stage5 verification before serving benchmarks. See [the active report](05_objective_evaluation_v3.md) for numbers and evidence.
