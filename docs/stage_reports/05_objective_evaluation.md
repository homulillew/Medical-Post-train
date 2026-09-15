# Stage 5 objective evaluation and local open QA

Status: OBJECTIVE_EVAL_PASS; OPEN_QA_JUDGE_PENDING. Stage 5 is not DONE. No paid judge was called; no human or clinical review was fabricated.

| Model | CMExam 6809 | CMB 2000 | CMB medical-only 1929 |
|---|---:|---:|---:|
| sft | 45/6809 (0.6609%) | 10/2000 (0.5000%) | 10/1929 (0.5184%) |
| selected_vanilla | 684/6809 (10.0455%) | 188/2000 (9.4000%) | 181/1929 (9.3831%) |
| selected_dynamic | 310/6809 (4.5528%) | 83/2000 (4.1500%) | 81/1929 (4.1991%) |

Scientific equal-update endpoints are reported separately in the JSON results; selected checkpoints were frozen using validation only.

| Scientific endpoint | CMExam 6809 | CMB 2000 |
|---|---:|---:|
| vanilla5000 | 1092/6809 (16.0376%) | 300/2000 (15.0000%) |
| dynamic5000 | 310/6809 (4.5528%) | 83/2000 (4.1500%) |

- CMExam, selected_vanilla_minus_sft: +9.3846 pp; 95% CI [8.67968864737847, 10.089587310912028]; McNemar p=1.43189e-169; W→C=656, C→W=17.
- CMExam, selected_dynamic_minus_sft: +3.8919 pp; 95% CI [3.407255103539433, 4.376560434718755]; McNemar p=8.52813e-62; W→C=284, C→W=19.
- CMExam, selected_dynamic_minus_selected_vanilla: -5.4927 pp; 95% CI [-6.212365986194742, -4.75840798942576]; McNemar p=6.45048e-50; W→C=147, C→W=521.
- CMB, selected_vanilla_minus_sft: +8.9000 pp; 95% CI [7.6499999999999995, 10.151249999999983]; McNemar p=1.00632e-48; W→C=182, C→W=4.
- CMB, selected_dynamic_minus_sft: +3.6500 pp; 95% CI [2.85, 4.5]; McNemar p=3.97576e-20; W→C=75, C→W=2.
- CMB, selected_dynamic_minus_selected_vanilla: -5.2500 pp; 95% CI [-6.6000000000000005, -3.9]; McNemar p=2.99468e-14; W→C=46, C→W=151.
- CMB medical-only, selected_vanilla_minus_sft: +8.8647 pp; 95% CI [7.568688439606014, 10.160705028512181]; McNemar p=1.10442e-46; W→C=175, C→W=4.
- CMB medical-only, selected_dynamic_minus_sft: +3.6807 pp; 95% CI [2.851218247796786, 4.561949196474858]; McNemar p=1.50931e-19; W→C=73, C→W=2.
- CMB medical-only, selected_dynamic_minus_selected_vanilla: -5.1840 pp; 95% CI [-6.531881804043546, -3.784344219803007]; McNemar p=1.84235e-13; W→C=45, C→W=145.

All slices are exploratory, with fixed source-defined membership and unadjusted intervals. Single training seed limits generalization. Safety files are automated review candidates, not clinical validation. All candidate regressions are retained. Stage 6 remains NOT_STARTED.
