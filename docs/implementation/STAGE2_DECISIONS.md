# Stage 2 decisions and observations

## D-021 — Freeze deterministic CMExam pool and validation partitions

2026-09-09; ACCEPTED; contract change NO. Data run `s2_data_20260909T024052_870aba` validated all 54,497 train rows and the pinned train/val byte hashes. Fields are exactly Question/Options/Answer/Explanation; no train difficulty/category exists. Stage1 sealed question-only cluster graph supplies heldout exclusion; no test answers, metadata or performance accessed in Stage2. Exclude heldout-overlap clusters (2617 train rows), invalid schemas (11 noncontiguous options, 3 invalid answers), two punctuation-only questions, then 2821 within-train duplicate cluster members. 49,043 unique clean candidates remain, sufficient for 15k deterministic seed42 pool. Missing explanations are retained: 2220/15000; they receive sem=0 with explicit `missing_reference`, never substituted text or guessed correctness.

Choose pool by `sha256('42:pool:'+ID)` after deterministic cluster representative selection. Formal IDs are first1000 of independent `sha256('42:profiling:'+ID)` order; smoke next50, disjoint from formal. Val512 monitor and1024 selection use clean distinct cluster representatives; every remaining official val ID is diagnostic reserve with cluster provenance. No val generation in Stage2. Different source difficulty must not be invented from lengths or answer cardinality.

## O-022 — Retained tokenizer compatibility failure

Initial semantic diagnostic `s2_semantic_20260909T024252_815f01` FAILED before scores: Transformers5.5.3 BertTokenizer lacks legacy `build_inputs_with_special_tokens`. Full traceback/config/pairs/source snapshot retained. Replacement run `s2_semantic_20260909T024652_cb6286` uses native tokenizers Encoding truncation with overflow chunks and backend post_process to add native special tokens; asserts concatenated content IDs equal originals exactly. This is chunking, not discarding overflow. Short-text vectors agree with SentenceTransformer.encode to max error 2.98e-8 for both models.

<a id="d-022"></a>

## D-022 — BGE-M3 as limited gated semantic alignment

2026-09-09; ACCEPTED before smoke and formal; contract change NO. Decision based on the 200 train-only/synthetic controlled pairs in `s2_semantic_20260909T024652_cb6286`, never on test or formal profiling results. Compared fixed MedEmbed-small revision40a5850d046cfdb56154e332b4d7099b63e8d50e and BGE-M3 revision5617a9f61b028005a4858fdac845db406aefb181.

Choose BGE-M3, retaining `0.8*acc + 0.15*acc*sem + 0.05*format`. Fixed mapping clip(cosine,0,1), no batch min-max. Both models fail negation/dose sensitivity: all12 controlled negation and dose perturbations rank above their proper paraphrase. Thus neither is accepted as a medical correctness scorer. BGE nevertheless improves drug-entity ranking (12/12 vs0/12), unrelated-text similarity (mean0.346 vs0.799 synthetic; train unrelated0.321 vs0.719), and aggregate within-anchor ranking (0.3371 vs0.2584). Pairwise AUC0.6341 vs0.4797 is descriptive on shared anchors, not a clinical metric. BGE CPU encoding13.09s vs2.48s for the diagnostic text set; memory and token statistics are retained.

Alternatives: retain MedEmbed (weaker Chinese/entity discrimination); remove semantic (would abandon the initial shaping hypothesis before measuring real reward cases); reduce weight (does not fix logical blindness and would add another untested hyperparameter). BGE is chosen only as bounded reference-alignment shaping among correct final answers. A correct final answer with an incorrect explanation can still score highly: retain this failure mode. Correctness gating is mandatory; wrong answers get semantic contribution exactly0 and total<=0.05. Correct answers get>=0.8. The decision freezes an experimentally limited reward, not evidence that semantic shaping improves learning. Future removal/weight change must have its own version and shared comparison decision.

Chunk policy: <=480 encoder content tokens, non-overlapping native token-ID spans, native special tokens per chunk, normalized chunk embeddings → content-length-weighted mean → L2 normalize. All content tokens retained. Same chunk policy for reference and reasoning; empty reasoning or missing reference yields0 with distinct reason. Model-local context limits verified; this deliberately uses the same explicit chunk size even though BGE supports longer inputs. Cache keys/text index bind model revision, text and policy. Quantify source missingness separately from low cosine.

## D-023 — Parser and rollout starting configuration

2026-09-09; ACCEPTED for smoke; formal freeze follows observed smoke. Preserve existing output contract's answer-only acceptance, or exactly one closed think followed by exactly one closed answer. No duplicate tags/options or trailing text for strict format. Harmless lowercase/separators/order canonicalize. Explicit final-region fallback can receive correctness1/format0; thinking never searched for final letters. Ambiguous final conclusions receive no answer. Truncated outputs count as completed trajectories only when the engine returns a valid length finish; partial unclosed answers remain incorrect.

Use verified Stage1 final-budget adapter only, rehash before/after and check positive/negative identity plus sleep/wake without any actor process or optimizer. vLLM0.24 V1 native sampler, batch invariant, LoRA shrink split_k1, TP1 BF16, eager, context4096, memory0.65, seq16, no prefix cache, seed42. Temperature0.6, top_p1, top_k-1, G4, cap1024. Per-prompt request seed is stable SHA-derived; within each n4 request native sampler supplies the four members. Equal text from distinct request/member IDs is legitimate sampling collapse evidence and retained, not discarded as a duplicate request.

Raw groups commit atomically before scoring. Generation process exits before GPU semantic encoding; both consume frozen pipeline manifests. Engine controls and successful/failed attempt costs are separate from formal4000. Correctness classification remains acc-based; no actual advantages or accepted training groups exist in this stage.

## D-024 — Freeze formal profiling after real 50×4 smoke

2026-09-09; ACCEPTED before formal; contract change NO. Smoke `s2_smoke_20260909T024936_746dbe` completed all200 responses from50 disjoint train prompts. Adapter positive/negative/repeated/sleep-wake checks passed. Correctness57% is a smoke observation, not the primary estimate. Groups0/1/2/3/4 correct =10/5/10/11/14; mixed26/50. Text diversity:48 groups have four distinct outputs, two have three. No collapse diagnostic or temperature adjustment warranted.

Mean output236.26, P95353.30, P99396.01, max453tokens; answer-tag closure100%, truncation0%. The predefined >5% truncation /<95% answer-closure trigger is not met: retain1024 and do not run1536/2048. This only supports sufficient length on smoke; formal will quantify a larger CMExam sample.

Strict format173/200 (86.5%), despite raw answer-tag closure100%. Rejections:21 answer tags inside think/duplicated structure,5 unclosed think,1 answer “无”. These are real model outputs rather than transport failures. Retain the strict contract; no post-hoc relaxation to count ambiguous output as correct. Manual read10 full trajectories plus representatives of each structural error found extraction consistent with the specified rules. A correct-label drug answer contains a questionable allergy-related statement, illustrating correct final answer does not certify reasoning; clinical judgment NOT_ASSESSED.

Freeze same SFT policy, reward/BGE/parser, request_batch_prompts4, G4, temperature0.6/top_p1/top_k-1/cap1024 and same engine settings for independent formal1000 prompts. Sampling is not filtered by previous successes/mixed groups. Actual smoke generation47252 outputtokens/254.635s=185.568tokens/s; formal prior estimate now~1.415h at the same mean and throughput, plus startup/scoring. This is a forecast, not a reduced budget. Prompt-token accounting records one input length per parent G4 request and also four logical trajectory input lengths; the legacy field name `generated_prompt_tokens_shared_prefill` denotes the former accounting convention and does not measure kernel prefill work or claim cache reuse (prefix caching is off).

Installed vLLM parallel_sampling.py confirms each seeded n4 child uses seed+member_index. Thus identical text is retained with distinct trajectory lineage, while duplicate IDs are rejected. Record the source hash in runtime audit. Future Dynamic cost is estimated from acc contrast, including unparseable-as-incorrect outcomes under this frozen reward, not independently adjudicated medical uncertainty.

## O-023 — Early immutable formal cases, no configuration changes

Formal run `s2_formal_20260909T025736_f607d9` is still running. Read the first three completed groups in each0/1/2/3/4 correctness bucket (60 complete trajectories) from immutable raw group files. The deterministic category-prefix selection cannot change when later groups append; qualitative worklog retains exact IDs and output SHA. These reads do not select formal prompts or change parser/reward/sampling. Aggregate stage claims wait for all4000.

Observed examples: train30338 member2 contains the explicit date-arithmetic statement “4月18日加14天为4月30日”; direct calendar addition givesMay2. Its nearest-option reasoning also selectsMay6 as closest toMay3 despiteMay2/4 alternatives. train19412 member2 chooses correctE after explicitly denying the same host-weight increase described byE; source explanation is missing. train31425 allfour choose correctD, while descriptions of urine specific gravity/pressure differ from the source reference. These are textual/arithmetic observations, not clinical adjudication. Correctness gating does not detect bad reasoning when the final option is correct.

The prompt at train27961 literally says “如图1（暂无图）”; all responses only receive text, yet some discuss hypothetical imaging findings. The source reference includes a described radiographic sign. Missing-image prompts were not excluded by the pre-frozen schema/lexical rules; preserve them and report their incidence as a data limitation. No image, answer or reference was supplied to the actor, and no post-hoc removal changes the pool or profiling budget. This is an input-completeness risk, not evidence of visual reasoning capability.


## O-024 — Full-run evidence and source placeholder

Formal 1000×4 completed and scored without optimizer updates. 531 mixed groups include 153 whose contrast comes only from unparseable members; 378 contain parsed wrong answers. All 80 selected full trajectories were qualitatively reviewed by the execution agent, not clinically adjudicated.

Four of 15k source references exactly equal `请等待更新`; one formal prompt (four responses) has this placeholder. Its frozen semantic scores remain unchanged and are reported as source-quality artifacts, not explanation-quality rankings. Future masking requires an explicit shared reward version; no retrospective rescore is performed. See `experiments/stage2/reference_placeholder_audit.json`.

After formal scoring finished, the launcher was guarded against duplicate scoring of completed runs. A real replay attempt on the completed smoke was rejected before worker creation, with every existing artifact hash unchanged. This changes launcher validation only; all six frozen execution files and the reward/config remain unchanged. Receipt: `experiments/stage2/completed_run_replay_guard.json`.
