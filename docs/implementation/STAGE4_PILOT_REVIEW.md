# Stage4 completed pilot review

2026-09-09. **PILOT_PASS, formal NOT_STARTED, READY_FOR_STAGE5=NO.**
Both independently initialized512-group pilots completed64 policy windows and
128 native optimizer steps. Both passed raw trajectory/reward/loss/checkpoint/
sync/validation verification, physical process resume and final native reload.
Vanilla verification completed18:22 Asia/Shanghai; Dynamic22:10. The detached
pilot queue finished; it did not launch a formal run.

Sources: `experiments/stage4/pilot_analysis_v1.json`, each pilot raw verification
receipt, and `pilot_manual_cases_v1.json`. The analysis script reconstructs
paired outcomes from all512 predictions and training summaries from64 windows.
It binds the source files by hashes. Thirty SVG/PDF artifacts are retained under
`experiments/stage4/pilot_plots/`. Pilot validation plots show only measured
initial/final points, with no intermediate values inferred.

## Validation observation and uncertainty

| Condition | Correct /512 | Accuracy | Change from SFT | Strict format | Mean output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared SFT |282 |55.0781% |— |98.2422% |252.73 |
| Vanilla pilot |290 |56.6406% |+1.5625pp |99.4141% |256.00 |
| Dynamic pilot |308 |60.1563% |+5.0781pp |100% |253.06 |

The two independent initial greedy monitor predictions are exactly identical.
Dynamic exceeds Vanilla by18 questions/3.5156pp at equal512 training groups,
2048 training trajectories and128 optimizer steps. Paired transitions:

| Comparison | Wrong→correct | Correct→wrong | Net correct | Unadjusted exact paired p |
| --- | ---: | ---: | ---: | ---: |
| SFT→Vanilla |41 |33 |+8 |.41599 |
| SFT→Dynamic |61 |35 |+26 |.01035 |
| Vanilla→Dynamic |47 |29 |+18 |.05045 |

These are exploratory two-sided exact discordance tests over questions, with
no multiplicity adjustment. There is one training seed per variant, so they
do not quantify training-seed uncertainty or establish algorithm superiority.
This is the monitor512 set, not test or selection1024. No formal configuration
is chosen by these p-values, and no test-performance claim is made.

Of SFT→Vanilla's41 corrections,3 start unparseable; of SFT→Dynamic's61,
4 start unparseable. Neither has a correct→unparseable regression. All76
Vanilla/Dynamic correctness disagreements have parseable answers on both
sides. Therefore the18-question net difference cannot be explained solely
by rescuing unparseable answers. That does not independently validate the
medical explanation or exclude all effects of format learning.

## Cost and the intervention

| Pilot | Training groups | Generated groups | Output tokens | Prompt+output tokens | Training phase hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla |512 |512 |544483 |612640 |1.8664 |
| Dynamic |512 |1408 |1500402 |1688518 |3.1010 |

Training phase sums real generation, reward, actor process, sleep/wake and sync
durations; it excludes validation, initial startup, pauses and raw verification.
Dynamic uses2.7556× output tokens and1.6614× training phase time. Most actor
work is fixed by the equal update budget, so time does not scale directly with
rollout tokens. Validation/control cost is retained separately.

Dynamic output-token dispositions:553533 selected,529104 rejected all-correct,
322303 rejected all-wrong and95462 expired eligible overflow. Only36.89% of
generated output tokens enter training. Generated599 mixed groups yield512
selected and87 overflow; the observed2.75× group amplification exceeds the
inverse prefilter mixed probability because overflow also costs generation.

Vanilla training includes237 mixed,174 all-correct and101 all-wrong groups.
Saved native advantages are nonzero in all237 mixed groups,149 all-correct
groups and12 all-wrong groups. The hybrid reward can vary within an equal-acc
group, so Vanilla's nonmixed groups must not all be called zero-gradient data.
Dynamic allocates all512 training groups to correctness contrast, over twice
Vanilla's237 mixed groups under the same training-group budget. This is the
intended intervention and a plausible contributor to the observed gain, not
proof of the mechanism from this single pilot pair.

At equal generated-token budget there is no matched evaluated checkpoint in
these pilots: only initial and final validations were scheduled. Consequently
the current evidence supports a promising equal-update-budget result, but
does not establish superior token efficiency. Dividing endpoint gains by token
counts would assume an unobserved learning curve and is not a causal comparison.

## Dynamics and the cost risk

Across Dynamic's successive16-window blocks, prefilter mixed fractions are
48.36%,43.90%,41.94%,37.50%; all-correct fractions are29.28%,32.85%,38.33%,
43.50%. Per-block amplification rises2.375→2.6875→2.8125→3.125.
These blocks contain different prompts under evolving policies. They cannot
by themselves show that particular questions crossed the policy frontier.
The pilot has no repeated training prompts:1408 unique generated and512
unique selected for Dynamic,512 unique for both Vanilla counts.

Mean second-minibatch objective clip activity is37.21% Vanilla and46.39%
Dynamic; maxima62.5% and75%. First minibatches have zero clipping, as expected
because all old logprobs freeze before the first update. Neither second-mini
trace shows persistent near100% clipping. Sequence-ratio extrema are
.96777–1.07861 and.95918–1.04831. Tight clipping and BF16 residual precision
remain documented limitations, not evidence of perfect numerical parity.

Maximum pre-clip gradient norms are1.6003 and1.1470 with configured norm clip1.
Raw checks found finite losses/gradients and changing LoRA parameters. Block
mean policy entropy stays about.74–.79, while mean generated response length
stays about259–273 tokens for Vanilla and263–268 for Dynamic. These traces do
not show persistent entropy or length collapse in512 groups; they do not
guarantee stability over5000. Validation truncation is1/512 Vanilla and0/512
Dynamic at the pilot endpoints.

## Full-response cases

Six complete outputs across SFT/Vanilla/Dynamic were manually read for two
paired cases. This is implementation-agent qualitative review, without an
external judge or clinician adjudication.

- Monitor1514: SFT and Vanilla emit E; Dynamic emits the official B. All
  three formats are valid, and the reasoning changes its interpretation of
  optionB. This illustrates an answer change beyond parser recovery.
- Monitor4061: Vanilla emits official B while SFT/Dynamic emit E, so the
  pair contains a real adverse Dynamic example. Vanilla's explanation itself
  concludes “左心房增大,” which is optionC, while it emits B (“左心室增大”).
  This observable text/option contradiction shows why correct final-answer
  labels cannot be equated with consistent reasoning.

Both favorable and adverse examples and their full raw responses are retained.
No claim about aggregate explanation quality is inferred from selected cases.

## Decision and remaining formal gates

The optimizer candidate LR1e-6/mini4/one epoch with unchanged native GSPO clips
has sufficient pilot evidence to retain it for the planned formal comparison.
There is no pilot evidence requiring a different per-variant LR or a new reward
term. This is a review recommendation, not the formal pair freeze.

Before formal launch, complete the three actual checkpoint transaction fault
injections already required by `CHECKPOINT_AND_RESUME.md`. Existing committed-
boundary restarts and CPU recovery tests do not cover those three kill timings.
Then freeze one shared config/code revision and declare both new formal runs
from the same original SFT. Preserve exactly5000 groups per variant; no pilot
checkpoint can become formal by continuation. Stage5–6 remain NOT_STARTED.

Pilot phase-time extrapolations are18.23h Vanilla and30.28h Dynamic for625
windows, excluding formal validation and setup. Dynamic's decreasing mixed
rate can make later formal windows slower; roughly50–60h for the two formal
runs including validation is a planning range, not a completion guarantee.
