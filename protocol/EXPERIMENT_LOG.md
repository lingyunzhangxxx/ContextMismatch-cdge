# Experiment ledger

This ledger records row-level evidence and interpretation boundaries. It is not
paper prose. Server root: `/data2/shang/context-mismatch-optimization`.

## Model

- Official non-quantized BF16 `Qwen3.5-9B`, 32 layers, hidden size 4096.
- 24 Gated DeltaNet layers and 8 full-attention layers; 262K context.
- Model path: `/data2/shang/models/Qwen3.5-9B`.
- GPTQ is excluded from activation attribution, pruning, and parameter editing.

## Behavioral and representational evidence

- Natural histories: 432 rows. Obedience-minus-verification margin effects are
  `-3.717`, `-3.502`, and `-3.408` at depths 1, 8, and 32. Every matched pair is
  negative; all six domain means are negative at each depth.
- Strict lexical-matched histories: 432 rows after adding depth 1. Effects are
  `-1.974`, `-1.559`, and `-1.521` at depths 1, 8, and 32. At depth 1 all 72
  pairs are negative. At depths 8 and 32, 91.67% are negative. Each depth has
  six negative domain means (`p=0.03125`).
- Cross-style late-layer directions transfer in both directions. Natural-fit to
  lexical-test has margin correlation `-0.731` and mean direction cosine
  `0.426`; lexical-fit to natural-test has `-0.620` and `0.417`.
- At lexical depth 1, governance is already perfectly LODO-decodable at several
  middle layers (for example layers 11, 15, and 16). This precedes strong causal
  decision readout and is evidence against a purely late answer-token artifact.

## Ruled-out simple mechanisms

- A fully balanced 1,728-row final-token fixed-vector intervention gives only
  `+0.042` mean obedience recovery and `-0.076` verification change. A fixed
  final-token vector has a sign-fixed effect on the `A-B` logit and cannot repair
  both correct-label swaps. Do not tune larger alphas against a smoke case.
- The 1,008-row lexical cache reset experiment has intact
  verification-minus-obedience gap `1.528`. Gaps after conv zero, recurrent zero,
  DeltaNet reset, and KV drop are `1.453`, `1.177`, `0.741`, and `0.559`.
  Recurrent-zero+KV-drop gives gap `0.003` but collapses both mean margins to
  about `0.63`; full DeltaNet-reset+KV-drop makes histories identical but gives
  only `0.458` accuracy. Destructive reset is not selective localization.

## Residual patching numerical audit

- A first 3,456-row run used targets saved from full-sequence forwards while
  sources used cached-suffix forwards. Cached coefficient-0 differed from the
  old full forward in 71/144 items (mean absolute `0.066`, max `0.25`). This run
  is diagnostic only: `activation_patching_lexical.jsonl`.
- The runner now captures targets online through the identical cached-suffix
  path and asserts identical suffix tokens. Its smoke has zero coefficient-0
  layer variance and zero layer-31 coefficient-1 error versus the cached target.
- Primary balanced cached-target run: 3,456 unique rows in
  `activation_patching_lexical_cached.jsonl`. Coefficient-0 layer variance is
  zero and all 144 layer-31 coefficient-1 endpoints reproduce the target with
  zero error. Exact patches are null through layer 15, emerge bidirectionally at
  layer 19 (`+0.111` rescue, `-0.170` reverse induction), and grow at layers 23
  (`+0.314`, `-0.408`) and 27 (`+0.351`, `-0.476`). All six task means and both
  label-swap halves have the predicted sign at layers 19, 23, and 27.

## Exact paired-cache patching

- Primary file: `paired_cache_patching_lexical.jsonl`, 2,016/2,016 unique rows.
  The observed factorial is complete, all cells contain 72 items, every margin
  and binary probability is finite, and `paired_all` matches the opposite
  history's intact cached-suffix endpoint with maximum absolute error `0.0`.
- Intact verification-minus-obedience margin gap: `1.528`. Exact opposite-state
  replacement effects (obedience rescue / verification reverse induction) are:
  convolution `-0.014 / +0.019`, recurrent `+0.453 / -0.349`, KV
  `+1.184 / -1.076`, recurrent+KV `+1.547 / -1.542`, and complete cache
  `+1.528 / -1.528`. Recurrent and recurrent+KV effects have the predicted sign
  in all six domain means (`p=0.03125` in each direction). KV is less uniform in
  the reverse direction because data-integrity is near zero.
- Four-layer stage effects peak at stage 3, layers 12--15 (`+0.918 / -0.766`).
  Stage 2, layers 8--11, is smaller but bidirectional (`+0.415 / -0.358`). Stage
  4, layers 16--19, is only `+0.097 / -0.047`, and stages 5--7 are approximately
  null. The stage-3 rescue is positive in all domains; reverse induction is in
  the expected direction in five of six. Label-swap and evidence-order strata
  retain the aggregate directions.
- The joint recurrent+KV patch slightly overshoots complete-cache replacement,
  while adding the target convolution state restores the exact target. Treat
  this as evidence of component interaction, not additive variance attribution
  and not a selective repair result.

## Interpretation discipline

- Layer-31 full-residual replacement is an endpoint positive control: replacing
  the final residual necessarily reproduces the target readout. Localization
  comes from the emergence curve at earlier layers and paired component/state
  patches.
- Zeroing a cache can create an invalid mixed state or erase useful history.
  Exact matched cache replacement, supported-authority controls, fresh-context
  controls, and factual-boundary memory are required before mitigation claims.
- Parameter-level explanation means locating fixed circuits that write,
  transport, and read a context-dependent governance state. Context does not
  update the model weights.

## 2026-07-27 A10 release and local checkpoint

- The full component-output patching run was intentionally stopped at the
  user's request so the two A10 GPUs could be released. It is incomplete and
  must not be summarized as a balanced result.
- Durable partial file: `component_patching_lexical.jsonl`, 1,224/2,592 rows.
  Local audit after transfer found 1,224 unique, valid JSON rows, all margins
  finite, with exactly 408 rows at each coefficient (`0`, `0.5`, `1`). The last
  completed key is verification realization 1, `candidate_preservation`, both
  swaps 1, layer 27 MLP, coefficient 1. Coverage is not balanced: obedience has
  432 rows (realization 0 only), whereas verification has 792 rows (all 432
  realization-0 rows plus 360 realization-1 rows). The file SHA256 is
  `1efd99b0b1e3aa02490e1b532581b12accde813ffa7d4156e00dc497cc09fd04`.
- The separate component runner smoke is complete at 36/36 unique rows with
  12 rows per coefficient and zero coefficient-0 component/layer variance. It
  covers only one `procurement` item, so it validates the hook and numerical
  path but supplies no component-level scientific result. File SHA256:
  `572171abfa0064b793e6f22c106cfdd83a5b8a8f820e9acff3b6c77555580a5b`.
- The Docker container exited `137` because `docker stop -t 20` escalated the
  deliberate stop; `OOMKilled=false` before release and both A10 devices were
  verified at 0 MiB afterward. This is not an experimental failure.
- The runner flushes each JSON line and was launched with `--resume`. The
  stopped container preserves a resumable 1,224-row checkpoint, but ali116 must
  remain released and the container must not be restarted without a new,
  explicit authorization and resource check.
- Server code, protocols, logs, and non-array results were synchronized to
  `/workspace/context-mismatch-paper`. Large `activations/`
  directories, `.npy` arrays, model weights, and the unrelated `vendor/` tree
  were deliberately excluded. Server originals remain in place.

## 2026-07-27 Ascend backend forward hard gate: failed first attempt

- This is a new-backend replication gate, not an extension of the CUDA rows.
  No CUDA/Ascend logits or mechanism effects may be pooled without an explicit
  backend-parity audit. `production_rollout_approved=false` throughout.
- Before submission, the designated shared incoming checkpoint contained
  exactly 18 regular files and 19,329,393,248 bytes. The active v7 code tree
  matched SHA256
  `0b72db68753aed42310b5ee42ce0f252853f9aaaa70a391dd88ee27143d54b67`.
  The queue contained only the pre-existing Job 8691; it was not modified.
  Job 8899 was no longer present in the live Slurm controller and was not
  modified. Node a07 was `mix` in partition a01 with 8×910B3 advertised.
- Repository submitter `submit_model_forward.sh` created Job 8904 and saved a
  held record with `PENDING`, `Reason=JobHeldUser`, `RunTime=00:00:00`, a07,
  one 910B3, 8 CPU, 128G, and 45 minutes before release. Queue count changed
  from 1 to 2. Run directory:
  `/workspace/context-mismatch-ascend/runs/ascend-model-forward-20260727T060344Z`.
- Job 8904 ran on a07 for 85 seconds and terminated `FAILED`,
  `Reason=NonZeroExitCode`, Slurm `ExitCode=1:0`; `exit_status.json` independently
  records exit code 1. The `COMPLETE` marker is absent and
  `model_forward.json` records `success=false`.
- Checkpoint lineage passed before the failure. Shared incoming and node-local
  incoming were each fully verified against contract SHA256
  `1716bc9ef0dadb7f2d8438b633cf275b21508c9d9bf533bb80975afca67cb4e6`:
  18 files, 775 tensors, 19,329,393,248 verified bytes, 19,306,216,416 weight
  bytes, non-quantized BF16 weights with F32 auxiliary state. The workflow then
  atomically promoted the shared copy to
  `/workspace/context-mismatch-ascend/models/Qwen3.5-9B` and the verified
  node-local copy to
  `/workspace/node-local/context-mismatch-ascend/models/Qwen3.5-9B`.
- The allocated NPU was healthy before the probe (`910B3`, health `OK`, no
  running process, 0/65,536 MiB process memory). The failure occurred before
  tokenizer load, model load, or a model forward: line 141 called
  `torch.npu.reset_peak_memory_stats(torch.device("npu:0"))` before device
  initialization and torch-NPU raised `Invalid device argument 0: did you call
  init?`. This is a probe initialization-order failure. It does not demonstrate
  model/backend incompatibility, and it does not establish successful forward
  compatibility either.
- Evidence hashes: `exit_status.json`
  `3fddbc1bcf60862281e4fa477cc5fa1534f5243fe3842cc7e10637711cec2228`;
  `model_forward.json`
  `91ebafb45244a1ec964cc1ca3f561717da1422f133786ba11886edab5d29f2e2`;
  shared verification
  `148e504369e6d9b9e2f80ed5996c5dbed9b1cda1f3e669fed44a25fabd452b6a`;
  node-local verification
  `748f9a6555808d38785803a65522e11ad3e5d3456c296157545ad1e4dec1770e`;
  held record
  `b2b9af90b0792a22f2150b3cb09676f795bb57732cd391b7f1c2cf3bbcc141f1`.
- At this checkpoint the hard gate remained closed. No backend parity,
  activation smoke, component smoke, or full mechanism rows had been started on
  Ascend. This state is superseded by the successful bounded reruns below; the
  failed Job 8904 is retained as lifecycle-error evidence rather than silently
  overwritten.

## 2026-07-27 Ascend forward, parity, and mechanism smoke

- All three new jobs ran on a07 with one 910B3 inside allocations owned by
  `researcher`; all completed with Slurm exit code `0:0`. Their independent
  `exit_status.json` files record exit code 0, hostname a07, and
  `production_rollout_approved=false`. A post-run queue audit showed only the
  pre-existing Job 8691 on a05; it was not modified.
- Job 8905 closed the official-BF16 forward gate. Run directory:
  `/workspace/context-mismatch-ascend/runs/ascend-model-forward-20260727T061813Z`.
  The 9,409,813,744-parameter Qwen3.5-9B loaded in `3.327` seconds and completed
  the bounded forward in `25.309` seconds. A/B logits were `23.75/18.5`
  (`A-B=5.25`). Residual hooks, full-attention KV, and DeltaNet convolution and
  recurrent states were finite. The executed probe SHA256 is
  `21374fef79ff9b88107df3a002e892c46352ea2285c3256348108b8c10b42b0c`.
  This successful rerun confirms that Job 8904's failure was probe
  initialization order, not model/backend incompatibility.
- Job 8906 passed the preregistered CUDA-to-Ascend parity gate. Run directory:
  `/workspace/context-mismatch-ascend/runs/ascend-backend-parity-20260727T063933Z`.
  The manifest fixes depth 32, realization 0, label/order swap 0, all six tasks,
  and both regimes: 12 samples evaluated separately through full and cached
  paths. Token counts and token hashes matched in 12/12 cases; label token IDs
  matched exactly (`A=32`, `B=33`). Full-path margin MAE/max error/Pearson were
  `0.0520833/0.125/0.9992569`; cached-path values were
  `0.1041667/0.25/0.9975721`. Decision and correct-margin-sign agreement were
  `1.0` on both paths. All hard criteria passed. Report, manifest, and contract
  SHA256 values are respectively
  `72c64f76b0fdcfeaf761b06a70fafc9282aa9b6e9e1a374130d3dd1f5fcfd258`,
  `003b0e864899bc542284f3804f1572f7d45a4613463720e821f4158dfd62b37b`,
  and `50b9cac64a9e38436a08364c887030bfa400d44cd5cefd70c263967ddac321d4`.
- Job 8907 passed a bounded residual/component runner smoke. Run directory:
  `/workspace/context-mismatch-ascend/runs/ascend-mechanism-smoke-20260727T064914Z`.
  It covers only routing, depth 32, realization 0, label/order swap 0, and both
  regimes. The 16 residual rows use layers 11/19/27/31 and coefficients 0/1;
  the 24 component rows use layers 19/23/27, mixer/MLP, and coefficients 0/1.
  All 40 logits were finite. Cached baselines exactly reproduced Job 8906,
  coefficient-0 maximum error was `0.0`, and layer-31 residual coefficient-1
  exactly reproduced the paired target endpoint (`0.0` maximum error). Report
  SHA256 is
  `9d4cfc7919bbd32d293d4ab9666e32c25c1820945d1daf460e7750420ec646a2`.
- Directionally, verification-to-obedience residual effects at layers
  11/19/27/31 were `0.0/-0.25/-0.75/-3.0`; obedience-to-verification effects
  were `0.0/0.0/+0.375/+3.0`. Layer-27 mixer effects were `-0.25` and `+0.25`
  in the same two directions. These one-item values are hook/endpoint smoke
  observations only. They are not an Ascend component-localization result and
  are not pooled with CUDA rows.
- The executed source files match their local copies byte for byte. Remote v9
  and v10 code-tree SHA256 values are
  `8ff8c574b702c0c0828c0711d7cc7f2aef19f11a99b7db88e74afa276fc44923`
  and `179649a047295fafbe89a1839ba307c2dbc754d61c50b6e8962eef04bc6b4c02`.
  The first v9 upload containing AppleDouble files remains quarantined at
  `/workspace/context-mismatch-ascend/code-v9-rejected-appledouble` and was
  not used for execution.
- The three complete run directories are preserved in shared cluster storage,
  mirrored at
  `/data/researcher/context-mismatch-ascend-evidence/context-mismatch-evidence-through-8907.tar.gz`,
  and extracted locally under `ascend/evidence/`. All copies of the tar have
  SHA256
  `10caf78cce8ac50d37f3371615aad1ff3bb2d5288fa0cdbcc001867fdb732416`.
- Full Ascend residual and component experiments had not started at this
  checkpoint. Their v1
  factorial, audit gates, resource estimate, and backend-separate reporting
  rule are frozen in `ascend/full_replication_contract.json`; that contract
  explicitly leaves submission unauthorized. Contract SHA256 is
  `5d08933ae8bffb27fd29f6d9b80876de3570c6c5b11a31ad892589e1993fd8c8`.

## 2026-07-27 Ascend full residual and component mechanism runs

- The user separately authorized the frozen full runner. The executed
  authorization has SHA256
  `20ab82d15d549f7631d329425209937f9ae5489328880c304f655290f35424e3`;
  the 144-sample manifest has SHA256
  `95d26df381df8bdf40c68fea5a9ff8e6977a34d593db57978656cb9b06776be6`.
  The scientific contract remained
  `5d08933ae8bffb27fd29f6d9b80876de3570c6c5b11a31ad892589e1993fd8c8`.
  `production_rollout_approved=false` throughout. Jobs 8691 and 8908 were
  treated as external and were never modified. Every submitted full job used
  one 910B3 on a07, 8 CPU, 128 GiB, and a two-hour limit, and was first held,
  validated at zero runtime, and only then released.
- Residual Job 8909 completed on a07 with Slurm `COMPLETED`, `ExitCode=0:0`,
  and runtime `00:22:21`. Run directory:
  `/workspace/context-mismatch-ascend/runs/ascend-mechanism-full-residual-20260727T073125Z`.
  The report has `success=true` and `full_pass=true`; the analysis has
  `audit_pass=true` and `preregistered_effect_gate_pass=true`. There are exactly
  3,456 unique finite rows, no missing, duplicate, or unexpected rows,
  coefficient-0 maximum logit error is `0.0`, and all 144 layer-31 coefficient-1
  endpoints reproduce the matched target with maximum error `0.0`. Prompt token
  counts and token hashes match in 144/144 items and label IDs remain exactly
  `A=32`, `B=33`. Intervention time was `1267.368` seconds and peak NPU memory
  was 19,971,295,232 bytes.
- The Ascend intact opposite-minus-source gap is `1.501736`. At coefficient 1,
  residual reverse-induction effects at layers 19/23/27 are
  `-0.164931/-0.407986/-0.470486`, mediating
  `10.98%/27.17%/31.33%` of that gap. Rescue effects are
  `+0.102431/+0.307292/+0.352431`, mediating
  `6.82%/20.46%/23.47%`. Each of the six layer-by-direction primary cells has
  the preregistered aggregate sign, both label-swap halves have the sign, and
  all six task means have the sign (`p=0.03125`). Layers 3--15 remain
  approximately null, and layer 31 gives exact 100% endpoint mediation as the
  preregistered positive control.
- This is a close backend replication of the separate CUDA cached-target
  residual result. Across the six coefficient-1 primary cells, Ascend-versus-
  CUDA effect MAE is `0.004630`, maximum absolute difference is `0.008681`, and
  Pearson correlation is `0.999904`. These comparison statistics use six
  backend-level summary values; no CUDA and Ascend rows were pooled.
- Component Job 8912 completed on a07 with Slurm `COMPLETED`, `ExitCode=0:0`,
  and runtime `00:17:18`. Run directory:
  `/workspace/context-mismatch-ascend/runs/ascend-mechanism-full-component-20260727T081521Z`.
  The report has `success=true` and `full_pass=true`; the analysis has
  `audit_pass=true`. There are exactly 2,592 unique finite rows, no missing,
  duplicate, or unexpected rows, and coefficient-0 maximum logit error is
  `0.0`. Intervention time was `964.011` seconds and peak NPU memory was
  19,971,295,232 bytes. The component design has no preregistered scientific
  gate and no preregistered mixer-versus-MLP ordering, so all 24 summary cells
  are reported and component ordering is explicitly exploratory.
- At coefficient 1, mixer/MLP effects for reverse induction are respectively
  `-0.067708/-0.031250` at layer 19,
  `-0.050347/-0.010417` at layer 23, and
  `-0.050347/-0.017361` at layer 27. Rescue effects are
  `+0.036458/+0.013889`, `+0.050347/-0.006944`, and
  `+0.039931/+0.012153`. Mixer has the expected aggregate sign in all six
  layer-by-direction cells and a larger absolute effect than MLP in every cell;
  MLP has the expected sign in five of six. However, mixer task-domain and
  label-swap signs are not uniform at every layer. This supports a
  mixer-dominant but distributed and interacting readout hypothesis, not a
  claim that one mixer is a complete or preregistered causal circuit.
- The isolated mixer-plus-MLP effects are not additive decompositions of the
  full residual intervention. Their descriptive sum is 49--60% of the layer-19
  residual effect but only 14--15% at layers 23 and 27. Because the component
  patches are separate nonlinear counterfactuals, these percentages cannot be
  interpreted as explained variance. The shrinking ratio at later layers is
  evidence against a single isolated component output accounting for the late
  residual mediation.
- Three fail-closed control-plane events are preserved rather than hidden. The
  first component invocation created no Slurm job because the submitter looked
  for the nonexistent key `submit_component_job` instead of the authorized
  `submit_component_job_after_residual_numerical_audit`. Job 8910 then exited
  `2:0` after one second because Slurm executes a spooled batch-script copy and
  `BASH_SOURCE` incorrectly resolved the code root to
  `/var/spool/slurmd/job08910`; it performed no model forward. Job 8911 exited
  `1:0` after 42 seconds with `checkpoint_rows=0` because the same authorization
  key bug remained in the probe. The immutable v14 repair maps the exact
  mode-specific authorization keys at submitter, batch, and probe levels and
  pins the spooled batch wrapper to the immutable deployment root. Its code-tree
  SHA256 is
  `d43d434e326693ba905ed02b6f364a050630428225390c80bcbb449af376573c`;
  its deployment tar SHA256 is
  `7b4a1f6d392a0e6321be08656b28e009157b5bc7fed22bd34c9aed56dd530dfd`.
  Scientific factorial, manifest, target construction, intervention logic, and
  analyzer were not changed by these control-plane fixes.
- Row SHA256 values are
  `a79f39be8ea4252b0ec7e448563f4dba2e66ee9867da29926f1970ce28d067fc`
  for residual and
  `3597b2122adf1571c3829d2a7a785cce4b0c37ffb59b97b83962b79561f9156b`
  for component. The successful run bundle is preserved in cluster shared
  home, mirrored at
  `/data/researcher/context-mismatch-ascend-evidence/context-mismatch-full-mechanism-jobs-8909-8912-20260727.tar.gz`,
  and extracted locally under `ascend/evidence/`; all copies have SHA256
  `42c64b96211d530f1ec726444396da5e6d50c4d5d56293bf1cfc0147c2b5c64f`.
  Jobs 8910/8911 are separately archived with SHA256
  `d8a382c9cb80448d38aa75bae5c33fa29b01d205f089155335fe8bdac3d36ddb`
  and are not part of the scientific result.

## 2026-07-27 Qwen3-8B multi-benchmark behavior smoke

- The official non-quantized `Qwen/Qwen3-8B` BF16 checkpoint at revision
  `b968826d9c46dd6066d109eabc6255188de91218` was downloaded into cluster
  shared home and verified against a 15-file, 16,397,461,266-byte source
  contract before atomic promotion. No NPU was allocated for the download.
- Jobs 8925 and 8926 executed the preregistered scalar-versus-batched parity
  audit. Mixed-length batch=4 failed with decision agreement `0.97917` and
  maximum logit error `1.75`; exact-length bucketing, which mostly batches the
  two label swaps of one item, still failed with agreement `0.98958` and
  maximum error `1.5`. Both exceed the frozen `0.125` threshold. These batched
  rows are execution-backend diagnostics and are excluded from scientific
  claims.
- The 96 scalar rows from Jobs 8925 and 8926 match exactly in every logit,
  correct margin, decision, and suffix hash. Job 8927 then ran the accepted
  scalar-only smoke and reproduced both references exactly after excluding only
  timing fields. Its semantic SHA256 is
  `413c81364dd521c0ec64d6ddbb3e9e78f47672193d40fa5f958fc93d916096f3`.
  Job 8927 exited 0 and wrote `COMPLETE`; all 96 rows are unique, finite, and
  suffix-paired. Subsequent Qwen3-8B claim runs are frozen at batch size 1.
- In the accepted smoke, obedience minus verification has equal-weight
  six-benchmark mean `-3.23958` with item-cluster bootstrap 95% CI
  `[-4.21875, -2.25]`; all six benchmark means are negative and 22/24 paired
  effects have the predicted sign. Explicit reset minus obedience is `+4.95833`
  with CI `[3.95833, 5.94818]`. These are a two-item-per-benchmark smoke, not the
  final behavior estimate. Strong unsupported preference pressure also makes
  fresh binary accuracy only `0.1667`, so the present claim is a paired margin
  shift caused by history, not general benchmark competence.
- The accepted evidence bundle is retained in cluster shared home, mirrored at
  `/data/researcher/ContextMismatch/qwen3-8b-v1/evidence-archives/`, and extracted
  locally under `artifacts/qwen3-8b-v1/evidence/`. Archive SHA256 is
  `5fc18146ab42adb465a8d3036ef8373261cd618b3e2a03aa379a05e2baf242cb`.
- Depth-scan Job 8928 was released only after the accepted smoke gate. It uses
  one 910B3 on a07 and scalar execution for 4,608 frozen rows. Full behavior
  remains blocked until its terminal audit and cross-depth effect gate pass.

## 2026-07-27 Qwen3-8B depth gate and full behavior run

- Depth-scan Job 8928 completed with 4,608/4,608 unique finite scalar rows and
  passed the frozen full gate. Equal-weight six-benchmark obedience-minus-
  verification margin shifts were `-6.645996` at depth 1 (95% CI
  `[-7.06917,-6.22412]`), `-8.103841` at depth 8
  (`[-8.70330,-7.47638]`), and `-7.029948` at depth 32
  (`[-7.57637,-6.47802]`). Every depth is negative in 6/6 benchmarks, both
  label swaps, both declared roles, and both natural and lexical-matched
  histories. This supports a persistent but non-monotonic regime carryover and
  rejects a simple token-count accumulation account.
- The depth gate is stored locally as
  `artifacts/qwen3-8b-v1/QWEN3_8B_DEPTH_GATE_V1.json`, SHA256
  `e9e12666aaa7179d6e00190c40ce9f65a828440e5ec3df5628539fdc5f70d555`.
  Raw depth rows have SHA256
  `b069abafa90f10580bb4e7174cbb604ea1bb903a1522aab48b5d93880fbd8deb`;
  the three-copy evidence archive has SHA256
  `066f9bcdb8fe4a563d1145e4e0a0606b9e18205cca516bfcc9a801d672282d42`.
- Full behavior Job 8932 was submitted through the held-validation workflow and
  released on a07 with one 910B3, 8 CPU, 128 GiB, a four-hour limit, scalar
  batch size 1, and `production_rollout_approved=false`. Run directory:
  `/workspace/context-mismatch-qwen3-8b/runs/qwen3-8b-full-behavior-20260727T121047Z`.
  The frozen target is 27,648 rows with 6,144 mismatch pairs and 6,144 reset
  pairs. At the 00:14:42 health audit it was RUNNING with 5,349 rows; output was
  growing and the bounded log contained no forward error. No full-behavior
  estimate is reported before terminal audit, `COMPLETE`, and evidence hashes.
- Job 8932 subsequently completed all 27,648 scalar rows with 27,648 unique job
  keys, zero non-finite values, 6,144 mismatch pairs, 6,144 reset pairs,
  `COMPLETE`, and job-local `exit_code=0`. Slurm accounting (`sacct`) was
  temporarily unreachable from the login endpoint, so the durable audit does
  not falsely record an independently observed database state; the held record,
  output completeness, exit trap, and artifact tree are retained.
- The preregistered obedience-minus-verification margin is `-6.85075`, with
  item-cluster bootstrap 95% CI `[-7.03746,-6.66750]`; 97.87% of the 6,144
  paired effects have the predicted sign. All six benchmark estimates are
  negative: ARC-Challenge `-10.7760`, BBH `-6.8357`, GSM8K `-6.5763`,
  MATH-500 `-6.4933`, MMLU-Pro `-7.2897`, and MuSR `-3.1335`. Every
  benchmark-specific one-sided sign-flip test remains significant after the
  frozen Holm family correction (`p_adj=5.99994e-5`, Monte Carlo floor).
- The effect is negative under both declared roles (assistant `-6.3810`,
  collaborator `-7.3205`), both history styles (lexical-matched `-3.8905`,
  natural `-9.8110`), both label assignments (`-6.2258/-7.4757`), and all
  four frozen partitions (`-7.0262/-6.8114/-6.7097/-6.8557`). Natural wording
  is stronger than lexical-matched by `-5.9206`, so lexical matching shows the
  phenomenon is not created by unequal vocabulary while the interaction shows
  natural discourse can amplify it.
- The length control sharply separates governance from generic long-context
  degradation: verification minus length-matched neutral is `-0.08030`, 95% CI
  `[-0.16720,0.00728]`, whereas obedience minus neutral is `-6.93105`, 95% CI
  `[-7.11463,-6.74883]`. Explicit reset recovers `+10.07928`, 95% CI
  `[9.85820,10.29482]`, positive in all six benchmarks and both label swaps.
- The immutable evidence archive SHA256 is
  `742c7985f068df6092a14d901190df84c88839a454fcc6d8d3729b0938cf95f9`;
  it is retained in cluster shared home, mirrored under
  `/data/researcher/ContextMismatch/qwen3-8b-v1/evidence-archives/`, and extracted
  locally under `artifacts/qwen3-8b-v1/evidence/`. The local extended analysis
  SHA256 is
  `7d6584f59b61f1b396df1e55692fd5983446b79be06bd3f24c1530367c2ce810`.

## 2026-07-27 Qwen3-8B mechanism and operator expansion frozen before execution

- `protocol/QWEN3_8B_MECHANISM_DISCOVERY_CONTRACT_V1.json` freezes a six-
  benchmark, 192-item `component_discovery` design before any Qwen3-8B
  mechanism forward. The residual scan is bidirectional and balances both
  roles, both history styles, both label swaps, and nine layers, for 27,648
  rows. A deterministic top-three rule then locks the attention/MLP component
  scan (18,432 rows). The full behavior gate must pass first, and later
  partitions remain unopened.
- New runner/analyzer code performs exact paired all-suffix residual or
  component replacement on the same scalar cached-suffix path, requires
  coefficient-0 identity, and uses layer 35 coefficient 1 as the residual
  endpoint control. Local Python and shell syntax audits pass; nine
  non-PyTorch tests pass. Eight PyTorch operator tests remain intentionally deferred
  to the next owned allocation because local PyTorch is unavailable.
- `protocol/NEGATIVE_OPERATOR_V4.md` adds a nested task-conditioned bilinear
  transport. It reads the history trigger only from the task-boundary state and
  uses a bounded code of the current-task hidden state to select a correction;
  it never receives the correct answer label. V3 is recovered at context rank
  zero. This specifically tests whether task-conditioned readout, rather than a
  larger universal negative vector, is required to recover both label swaps and
  heterogeneous benchmarks. V1/v2/v3 remain required baselines.
- A pre-forward execution audit found that the original history-gate requirement
  of leave-one-benchmark-out AUC was not identifiable: the same role/style/
  realization histories are reused across all six benchmark families, so a
  held-out benchmark fold would contain boundary states identical to training.
  The original operator contract remains byte-for-byte frozen. Erratum 1
  replaces only that invalid diagnostic with leave-one-history-realization-out
  and bidirectional cross-style AUC; actual cross-benchmark generalization is
  still required on edited behavioral results. The erratum SHA256 is
  `09bfaa8e7b62c3357aa193d0296ec24eda1c744fec50aa0dd27fe249cff08d38`.
- The site-and-execution supplement was frozen before Qwen3-8B mechanism
  forward, SHA256
  `299a7c9bc215630ea9e007d336369a55bbc9dc89a1b1e7cc163a947d72ff0305`.
  It selects one attention/MLP site per residual-nominated layer using only
  `component_discovery`, requires separate protected-state captures, and
  requires both application-gated and forced-on collateral audits. The latter
  prevents a trivial selectivity claim in which all protection comes from an
  external gate while the negative subspace itself is indiscriminate.

## 2026-07-27 Qwen3-8B residual smoke and full-run arithmetic erratum

- Residual smoke Job 8945 completed 432/432 unique finite rows and passed every
  numerical gate: 216 coefficient-zero rows have maximum baseline error `0.0`,
  and all 24 layer-35 coefficient-one endpoint rows reproduce the matched target
  margin with maximum error `0.0`. The job wrote `COMPLETE`, job-local
  `exit_code=0`, and Slurm recorded `COMPLETED ExitCode=0:0`.
- The single-item-per-benchmark smoke is descriptive only. Obedience-source
  rescue / verification-source reverse-induction effects grow from
  `+0.2917/-0.7083` at layer 3 to `+1.2917/-0.8750` at layer 15 and
  `+3.9167/-3.9167` at layer 19. Layer 23 gives `+4.2083/-4.1042`, while layer
  35 exactly swaps the full target gap at `+4.2083/-4.2083`. This validates the
  bidirectional runner and suggests a late readout, but it does not nominate a
  layer or establish benchmark-general mediation.
- The smoke evidence archive is retained in cluster shared home, mirrored under
  `/data/researcher/ContextMismatch/qwen3-8b-v1/evidence-archives/`, and extracted
  locally under `artifacts/qwen3-8b-v1/evidence/`. Its SHA256 is
  `bca300b5bac7a5ca5f88e05bf8afb3b0499c0b57e40447eee00c4e20d2cd15cd`.
- Full preflight Job 8946 exposed a frozen arithmetic inconsistency before any
  affected model forward: 32 items per benchmark imply 55,296 residual rows,
  but the contract and execution authorization both freeze 27,648. The job
  failed with exit code 1 and produced zero scientific rows. Its separate
  three-copy failure archive SHA256 is
  `22a5efc2c9748959b3a3b4e5eb57eec3dec5f343b1e98be796f25686e8bf0549`.
- Mechanism Erratum 1 changes only the effective discovery sample to the first
  16 item IDs per benchmark (96 total), yielding exactly 27,648 residual and
  18,432 component rows without dropping a factorial cell. The original
  contract remains unchanged. Erratum SHA256 is
  `29af41a9e0d3cfb5f9a5e523365a9de4277db61b7b2c8fc1086fa9d52824d0bb`;
  execution authorization v2 SHA256 is
  `252b78a9c1648cea788b4f40d74d5bcbcfd7d23444b4894ad1d036b4a8855072`.
- The corrected immutable `code-v12` bundle passed 20/20 cluster tests. Residual
  full Job 8947 was submitted through the held-validation workflow on a07 with
  one 910B3, 8 CPU, 128 GiB, a six-hour limit, and
  `production_rollout_approved=false`. No full effect is reportable until all
  27,648 rows, the endpoint gates, `COMPLETE`, terminal Slurm state, and the
  three-copy evidence archive pass.

## 2026-07-28 Qwen3-8B residual full provisional durability copy

- Job 8947 produced 27,648/27,648 unique finite rows. The observed and expected
  job-key hashes are both
  `93806073b57c83b245142a5a5da4b1eb99d68209b1732769949ea6a6e828e724`;
  coefficient-zero maximum error and the layer-35 coefficient-one endpoint
  error are both exactly `0.0`. The run has `COMPLETE` and job-local
  `exit_code=0`.
- Slurm accounting could not independently confirm `COMPLETED` because the
  cluster login endpoint could not route to `slurmdbd` at `mgnt:6819`.
  At the user's request, data durability was prioritized without representing
  this missing accounting evidence as present. The source run was not modified;
  a staging copy contains `TRANSFER_STATUS.json` with
  `slurm_completed_verified=false` and a verified `artifact_tree.sha256`.
- The provisional immutable archive is retained in cluster shared home, mirrored
  at `/data/researcher/ContextMismatch/qwen3-8b-v1/evidence-archives/`, and extracted
  locally under `artifacts/qwen3-8b-v1/evidence/`. Its archive SHA256 is
  `e885c3d16989b2932a0a9833d17758e51c16e27ed19e932684d21213d60e3410`.
  This is a content-complete safety copy, not evidence that Slurm accounting
  recorded `COMPLETED`; that independent terminal-state check remains open.

## 2026-07-28 Qwen3-8B component full completion and operator-site lock

- Owned Job 8960 completed on a07 with Slurm `COMPLETED`, `ExitCode=0:0`, and
  runtime `00:59:17`. The full component scan contains exactly 18,432 rows and
  18,432 unique job keys, with zero duplicates and zero non-finite rows. The
  observed and expected key-set SHA256 are both
  `517d0e3a26c80861e3f7028a053d96be631052ef70f47a7156f2b69871bdea6f`.
  All 9,216 coefficient-zero interventions are exact identities with maximum
  selected-margin error `0.0`; the analysis audit reports `success=true`.
- The immutable component archive is present in cluster shared home, mirrored
  on archive-host under
  `/data/researcher/ContextMismatch/qwen3-8b-v1/evidence-archives/`, and extracted
  locally under `artifacts/qwen3-8b-v1/evidence/`. All three copies bind to
  archive SHA256
  `bcba43fca86d2c3578e9b64a1da6e9fbc2fda2a569c6586dfb12f6c6fb71d61b`.
- The frozen per-layer nomination rule was then applied to the complete rows,
  producing `QWEN3_8B_OPERATOR_SITE_MANIFEST_V1.json` with SHA256
  `8bf9e43cd3b3597ed47ef1698f8fd46137fd01c04b3ec0ac9ab6ccd21c3659d3`.
  Ordered by locked component score, the sites are layer 23 `self_attn`
  (`0.318679`), layer 31 `mlp` (`0.183478`), and layer 27 `mlp`
  (`0.155176`). Each selected site has the expected rescue/reverse-induction
  sign in both label-swap halves. The unselected component at each layer is
  retained in the manifest rather than discarded.
- This closes component localization but does not establish additivity,
  low-rank sufficiency, or mitigation. `final_test_open=false` and
  `production_rollout_approved=false` remain unchanged. The next claim-bearing
  steps are the symmetric governance-task crossover and split-isolated operator
  fitting/evaluation.

## 2026-07-28 Qwen3-8B symmetric governance-task crossover discovery and replication

- Crossover discovery Job 8963 completed 6,144/6,144 unique finite rows with
  observed/expected key SHA256
  `4e6301fce193060c68bb284f7b3349aec61766eb1daa60ff85d200ef9d981900`,
  `COMPLETE`, job-local exit 0, and Slurm `COMPLETED ExitCode=0:0`. The
  obedience-minus-verification history effect is `-0.91772` when the new task
  requires independent verification and `+1.52620` when it requires delegated
  choice. Their preregistered match-advantage interaction is `+2.44393`, with
  item-cluster bootstrap 95% CI `[2.20321,2.68994]`; matched minus mismatched is
  `+1.22196`, CI `[1.10201,1.34029]`.
- Independent crossover replication Job 8964 completed 6,144/6,144 unique
  finite rows. Its observed/expected key SHA256 is
  `488835b013ebdf7413d29fa3490bf937700e1b542bac9b0539aa31ba966dcd35`;
  the run has `COMPLETE`, job-local exit 0, and Slurm `COMPLETED ExitCode=0:0`.
  The verification-task history effect is `-0.80526`, CI
  `[-0.93099,-0.67708]`; the delegated-choice history effect is `+1.38607`, CI
  `[1.22705,1.55037]`. The replicated match-advantage interaction is
  `+2.19132`, CI `[1.96403,2.41960]`, and matched minus mismatched is
  `+1.09566`, CI `[0.98185,1.20891]`.
- All six benchmark-specific interactions are positive in both splits.
  Discovery/replication values are respectively: ARC-Challenge
  `2.3530/2.5527`, BBH `2.3623/2.3232`, GSM8K `0.7744/0.9702`, MATH-500
  `2.5303/2.3042`, MMLU-Pro `2.0435/1.8154`, and MuSR `4.6001/3.1821`.
  Both declared roles, both history styles, and both label swaps are also
  positive in both splits. The strict per-pair both-direction fraction is only
  `26.69%` in discovery and `25.65%` in replication, so the result is a robust
  aggregate interaction rather than a deterministic example-level law.
- The discovery archive SHA256 is
  `74847ee5a40e981ac9ec5f5933cc65c993ad11ef44e59422786ae8dc35a587b6`.
  The replication archive SHA256 is
  `25716cf2908e70a75625a360263d57ea34f352cd356b89be0955d4080591bab9`.
  Each is retained in cluster shared home, mirrored on archive-host, and
  extracted locally under `artifacts/qwen3-8b-v1/evidence/` with its artifact
  tree verified. These experiments close the behavioral definition of
  context mismatch on Qwen3-8B; they do not yet establish that a low-rank
  operator can selectively mitigate it. `final_test_open=false` and
  `production_rollout_approved=false` remain unchanged.

## 2026-07-28 retrospective cross-split decision-boundary validation

- We tested the post-hoc hypothesis that context-mismatch errors concentrate
  near the matched task-aligned decision boundary, rather than increasing
  monotonically with benchmark difficulty. This is explicitly a
  `retrospective_cross_split_validation`, not a preregistered analysis. It uses
  no final-test rows. Discovery and replication inputs each pass the 6,144-row,
  unique-key, finite-value audit.
- Margin cutoffs were derived only from discovery matched-correct pairs and
  then applied unchanged to replication. For independent verification, the
  discovery Q25/Q75 cutoffs are `3.0/14.75`. In replication, the low-margin
  boundary band has 62 correct-to-wrong flips among 209 pairs (`29.67%`), the
  middle band has 11/438 (`2.51%`), and the robust band has 0/232 (`0%`). The
  item-cluster bootstrap boundary-minus-robust contrast is `+29.67` percentage
  points, 95% CI `[23.11,36.56]`.
- For delegated choice, the discovery Q25/Q75 cutoffs are `22.5/27.75`. In
  replication, the boundary band has 14/415 flips (`3.37%`), while the middle
  and robust bands have 0/703 and 0/418. The boundary-minus-robust contrast is
  `+3.37` percentage points, item-cluster bootstrap 95% CI `[1.31,5.81]`.
- The six-benchmark ecological diagnostic does not support a monotonic
  harder-benchmark effect. For independent-verification replication, Spearman
  correlation between matched error rate and mismatch accuracy drop is `0.20`,
  with exact two-sided permutation `p=0.7139`. Benchmark-specific matched
  accuracy/drop values are ARC-Challenge `81.64%/1.95 pp`, BBH
  `55.47%/0.39 pp`, GSM8K `59.38%/1.56 pp`, MATH-500 `44.53%/5.47 pp`,
  MMLU-Pro `64.06%/5.08 pp`, and MuSR `38.28%/3.91 pp`.
- The validated statement is therefore conditional: among answers that are
  correct under the task-matched history, low positive task-aligned margin
  predicts susceptibility to a mismatch-induced flip. Very hard already-wrong
  items are excluded by construction and can exhibit a floor effect; large
  positive-margin items are robust. A preregistered prospective held-out
  confirmation is still required. The analysis artifact is
  `artifacts/qwen3-8b-v1/boundary_susceptibility_validation_v1.json`, SHA256
  `b110c1a9c7dbb83677b731e089bd88a8273f69cdcfe0c6599d9a53f85a5735c8`.
  `final_test_open=false` and `production_rollout_approved=false` remain
  unchanged.

## 2026-07-28 Qwen3-8B split-isolated operator capture and prefilter launch

- The first immutable operator bundles failed before producing scientific
  evidence and remain separate audit records. `code-v19` Jobs 8965/8966 exposed
  a missing compatibility path in the bundle layout. `code-v20` Jobs 8967/8968
  reached the first model-forward setup and exposed that Qwen3 supplies
  `self_attn.hidden_states` as a keyword-only argument while the capture hook
  read positional input only. None of these four jobs produced a complete
  capture shard, and none is used as scientific evidence.
- Immutable `code-v21` fixes both capture and mitigation hooks to accept
  positional or keyword `hidden_states` and adds regression tests for both
  calling conventions. Its bundle-manifest SHA256 is
  `35ea260a56b2d0d7d05a671699a4a0b516d430f341e5331dfc5973cf38f568d3`;
  39/39 cluster tests pass (17 device-dependent local skips are not counted as
  failures).
- Subspace-capture Job 8969 completed 3,072 rows in 12 shards, including the
  frozen 768 answer-blind boundary rows. Its capture manifest SHA256 is
  `069ab3db5f1c732f14a3e3a264c75ae831136f3077b12fb062a9fbf8b4fde194`.
  The run has `COMPLETE`, job-local exit 0, and Slurm
  `COMPLETED ExitCode=0:0`. Its three verified archive copies bind to SHA256
  `20b84f3abbe94fc86da8281d4cf5574f4c651ff5a14f0547ec0d4ad1f617aceb`.
- Protected-state capture Job 8970 completed 4,008 rows in 50 shards over fresh,
  matched verification, explicit reset, supported-authority, and factual-memory
  controls. Its capture manifest SHA256 is
  `f084c379cf59b5eeffdb67f62ce26d2cbb7418919da5e93a297dea1030a7c690`.
  The run has `COMPLETE`, job-local exit 0, and Slurm
  `COMPLETED ExitCode=0:0`. Cluster shared home, archive-host, and the extracted
  local evidence tree were all verified against archive SHA256
  `f4dafe01ebd0fe26f7a19500d5fd0eb0c90f52fea45265df7349801d533b9086`.
- A fresh SHA-bound `fit_prefilter` authorization binds exactly those two
  capture manifests, the replicated crossover analysis, and `code-v21`; its
  SHA256 is
  `c6755dfe6193242616f59303ba99cf83fdc33c7f93cef3a45856c6897a89c6bf`.
  CPU-only Job 8972 completed under held validation with 32 CPU, 192 GiB, no
  NPU request, and runtime `00:30:03`. The run contains all 15,660 deterministic
  grid rows, 42 shortlisted and 42 unique materialized candidates, and finite
  four-fold reconstruction scores. It has `COMPLETE`, job-local exit 0, and
  Slurm `COMPLETED ExitCode=0:0`. Its cluster, archive-host, and local archive
  copies bind to SHA256
  `effef85bfe20f8568352f1724559ecac0cdacb9e80c8502678bcf00069f5d6ca`.
  The offline ranking is not a behavioral mitigation result. The 42 candidates
  were frozen into four SHA-bound screening shards of 11, 11, 10, and 10
  candidates and submitted as Jobs 8973--8976 through held validation; Job 8976
  initially waits for resources while the first three run. `final_test_open=false`
  and `production_rollout_approved=false` remain unchanged.

## 2026-07-28 Qwen3-8B operator-development screening falsifier

- Jobs 8973--8976 completed the frozen 42-candidate behavior-screening
  shortlist. Every candidate has exactly 3,072 unique finite rows, the common
  observed/expected key SHA256
  `b039292f6b16aae9efcd40e34178f894a5d8317cf5f37cd532ac73b410931521`,
  coefficient-zero maximum error `0.0`, and a successful analysis audit. This
  yields 129,024 audited screening rows in total. Every shard retains
  `final_test_open=false` and `production_rollout_approved=false`.
- Jobs 8973, 8974, and 8975 have `COMPLETE`, job-local exit 0, and complete
  run-local scientific evidence for 11, 11, and 10 candidates respectively.
  Their live `scontrol` records expired before archival, and `sacct` could not
  route to `mgnt:6819`; therefore Slurm `COMPLETED ExitCode=0:0` is not claimed
  for these three jobs. Under the user's explicit provenance exception, the
  original run directories were left untouched and copied into clearly named
  run-local-exception archives. Their three-copy SHA256 values are, in job
  order, `bf54837aaad403b2b72edb8772350057dca8d62b481bae9ac5bbe1438040f88d`,
  `f946f9dde923ceb94a35b131cfefa6b333b20672d45e864c0055a9a4883edce8`,
  and `baa41f1019516dd04f0f297f9f74a3a590e80562d45de6fc272c465420d2cbeb`.
  Formal scheduler provenance remains an explicit open metadata item, not a
  fabricated terminal record.
- Job 8976 completed 10/10 candidates with `COMPLETE`, job-local exit 0, and a
  live Slurm terminal record of `COMPLETED ExitCode=0:0`. Its immutable archive
  is verified in cluster shared home, archive-host, and local storage at SHA256
  `2e768dcb01f2c5b4ccc59c86eb5b3055372fb312833bbfe7ccca502c38712cd5`.
  The four-shard provenance manifest is
  `artifacts/qwen3-8b-v1/operator_behavior_screening_provenance_v1.json`,
  SHA256
  `e005b0edfe2af6b3b4a7bfa59de3b90bdd834de60fcc54c25c031ad0dcca2a5a`.
- The four candidate ledgers were merged without changing candidate contents.
  The 42-candidate merged ledger SHA256 is
  `246e218a91a8e9de8436dcae616352f85dda4031b5e4a8143471ffe6303cf0e5`.
  Applying the frozen aggregate, both-direction, and both-label-swap rules
  produced 42 audited candidates, zero eligible candidates, and zero family
  finalists. The immutable selection report SHA256 is
  `e7b8ca9c1e1923b53a19aad1e545a6d3519a6b5d0f49775bbb747f1979a37ae4`.
- This is a scientific falsifier rather than an audit failure. Twenty-six of 42
  candidates reduce the aggregate mismatch gap, only two improve both causal
  directions strictly, and only ten improve both label-swap halves strictly;
  none satisfies all three. The best aggregate candidate, v3
  `cf53417cddd6f4db`, reduces the equal-benchmark-weight gap by `0.046794` but
  has exactly zero recovery in the verification-history-on-delegated-task
  direction. The v5 candidates `b5dabc6130f2d09a` and `346998a8570394ab`
  improve both directions but have negative reduction in label-swap half 1.
- Frozen thresholds were not weakened. No behavior-selection authorization,
  protected-control run, multisite run, Pareto lock, or final-test
  authorization was generated. `final_test_open=false`,
  `final_test_open_count=0`, and `production_rollout_approved=false` remain
  binding; cross-model work remains unopened.

## 2026-07-28 Qwen3-8B AMSGE V1 complete post-failure characterization

- AMSGE V1 is retained as a complete negative-result baseline, not as an
  experiment stopped at its fit gate. CPU-only fit Job 9016 completed all 80
  epochs and wrote a SHA-verifiable editor-only checkpoint containing no base
  weights. The fit used 4,576 train, 672 calibration, and 896 audit rows and
  never accessed `operator_dev`. Its frozen aggregate fit decision is
  `all_fit_gates_pass=false`: both direction, both label-swap, reconstruction,
  and trust-region checks pass, but history accuracy, task accuracy, and the
  protected forced-on gate fail. The fit report and checkpoint SHA256 values
  are respectively
  `8662d3b851c3d7af3bba88cac32a4df006947b9d7ef01b6186bde82ddaaf1f5c`
  and
  `ef2d85ebb7e2832b8ae6a5b2489284df81887e29dbc12cc9fbc3294aa99b9316`.
  Job-local exit is 1 by contract because the frozen gates failed. The expired
  live controller record and unavailable `sacct` route are recorded rather
  than represented as Slurm failure evidence; the immutable three-copy
  diagnostic archive SHA256 is
  `a9013d55e3980eb4811318b9065f38e7378bb50558a538125800b47f278c5eae`.
- Execution-validation Job 9040 then completed 192/192 unique finite smoke rows
  with exact expected/observed key SHA256
  `a5348fefd5beb90a83da63022b3ddf58439526dcbe16a75a9f49041806b0d3d2`,
  coefficient-zero error `0.0`, `COMPLETE`, job-local exit 0, and Slurm
  `COMPLETED ExitCode=0:0`. This smoke is not used for selection. Its
  three-copy archive SHA256 is
  `00adbd409283830a8e32a32550e433cd2ec416c69299627e9ba43d965949f933`.
- Untouched `operator_dev` Job 9041 completed the full 3,072-row behavior
  selection with exact expected/observed key SHA256
  `74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e`,
  zero non-finite rows, coefficient-zero error `0.0`, successful audit,
  `COMPLETE`, job-local exit 0, and Slurm `COMPLETED ExitCode=0:0`. The editor
  increases, rather than reduces, the equal-benchmark-weight mismatch signal:
  match-advantage reduction is `-0.147461`, matched-minus-mismatched reduction
  is `-0.073730`, and normalized gap reduction is `-0.052387`. The two mismatch
  directions are asymmetric (`+0.082031` for obedience history on a
  verification task and `-0.100911` for verification history on a delegated
  task); label-swap reductions are `-0.006673` and `-0.140788`. All six
  benchmark aggregate reductions are negative. The bootstrap 95% CIs for
  match-advantage, matched-minus-mismatched, and normalized reduction are
  respectively `[-0.197432,-0.099935]`, `[-0.099202,-0.049721]`, and
  `[-0.071640,-0.032993]`, excluding benefit. The behavior gate would fail
  even if the fit result were ignored. The identity report SHA256 is
  `429a9f8370ccc58a401d898a074908b7b26ceffc2dbba5c9d830301351d3fd9a`;
  the three-copy selection archive SHA256 is
  `5ab54392f5cc7065736992c563b485c49b1a8d9d29db732b6d35db18030d7c56`.
- Protected-control Job 9042 completed 2,856/2,856 unique finite rows with exact
  expected/observed key SHA256
  `220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77`,
  non-governance identity error `0.0`, successful audit, `COMPLETE`, job-local
  exit 0, and live Slurm `COMPLETED ExitCode=0:0`. Five control families pass
  both application-gated and forced-on reference gates. Supported user
  authority fails both: its margin-loss estimate is `0.210069`, with one-sided
  95% upper bounds `0.280382` application-gated and `0.281250` forced-on,
  exceeding the frozen `0.25` limit. Its binary KL is `0.0`; the failure is
  decision-margin collateral, not distributional divergence on that binary
  statistic. The control-analysis SHA256 is
  `0b847cce9c759a03d260f624a31840fe7588e5a096b084d437895aaf14624d10`;
  the verified three-copy archive SHA256 is
  `49e645f9d801d34fb0dcadc2c2746ef88e84b1b5c088242ec1fc5f315508836b`.
- The final V1 status is therefore scientifically complete and negative:
  `candidate_eligible=false`, `controls_admissible=false`, and
  `candidate_may_be_locked=false`. No threshold was weakened and no positive
  efficacy claim is made. V1 may be reported as a fully characterized failed
  internal-governance editor and retained as a baseline, but it can never
  produce a Pareto lock or final authorization. `final_test_open=false`,
  `final_test_open_count=0`, and `production_rollout_approved=false` remain
  unchanged.

## 2026-07-28 Qwen3-8B AMSGE V2 protocol-deviation characterization

- AMSGE V2 replaces the V1 low-rank classifier inputs with normalized full
  hidden states while retaining reversible activation editing and frozen base
  weights. Fit Job 9039 completed with job-local exit 0,
  `all_fit_gates_pass=true`, and `operator_dev_accessed=false`. Its editor-only
  checkpoint and fit-report SHA256 values are respectively
  `19844a1e1bd1a7203645fbc489995a0fa5315e4dcaa66ec471ff781da6df9a66`
  and
  `6a941e57e677f315fbaf348cd23a5d76e1ed38694a66c0329978a3da49975ad3`.
  The live `scontrol` record expired and `sacct` could not reach `mgnt:6819`,
  so no Slurm terminal record or strict fit archive is claimed. At the user's
  explicit direction, the subsequent evaluation is labeled
  `PROTOCOL_DEVIATION`; this provenance exception is not represented as a
  successful strict fit closure.
- Execution-validation Job 9072 completed 192/192 unique finite rows with the
  exact expected/observed key SHA256
  `a5348fefd5beb90a83da63022b3ddf58439526dcbe16a75a9f49041806b0d3d2`,
  zero-gate maximum error `0.0`, `COMPLETE`, job-local exit 0, and Slurm
  `COMPLETED ExitCode=0:0`. Its three-copy archive SHA256 is
  `e755e8f0b38dda19e60492dab405fda24bbbed3d9eccb04035e380ac577e34f8`.
  This smoke is retained only as an execution check and is not used for the
  efficacy decision.
- Full untouched `operator_dev` selection Job 9076 completed 3,072/3,072
  unique finite rows with exact expected/observed key SHA256
  `74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e`,
  zero-gate maximum error `0.0`, successful audit, `COMPLETE`, job-local exit
  0, and live Slurm `COMPLETED ExitCode=0:0`. The editor again increases the
  target mismatch signal: match-advantage reduction is `-0.100911`,
  matched-minus-mismatched reduction is `-0.050456` with item-cluster
  bootstrap 95% CI `[-0.074465,-0.026449]`, and normalized gap reduction is
  `-0.033615`, CI `[-0.050991,-0.015907]`. The two mismatch directions are
  `+0.078613` and `-0.109212`; label-swap reductions are `-0.003743` and
  `-0.097168`. Five of six benchmark reductions are negative, with MuSR only
  `+0.003906`. Thus the complete behavior gate fails aggregate, uncertainty,
  both-direction, both-label-swap, and all-benchmark checks; matched-cell
  collateral and exact identity checks pass. The analysis SHA256 is
  `94bc67f573fa2e1c9576c7af5322bb1c4733e217b4dc784edc7948c8f31130c6`,
  and the verified three-copy selection archive SHA256 is
  `085c0a47a5d8942436ca0b6594cef0700d9406b821eea1185858ae7eeb3933af`.
- To retain a complete negative-result characterization, protected-control Job
  9077 was run under the same explicit protocol-deviation label. It completed
  2,856/2,856 unique finite rows with exact expected/observed key SHA256
  `220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77`,
  successful audit, `COMPLETE`, job-local exit 0, and live Slurm
  `COMPLETED ExitCode=0:0`. The non-governance application gate has exact
  identity and maximum selected-logit error `0.0`. All six control families
  pass both application-gated and forced-on reference checks. The largest
  forced-on one-sided margin-loss upper bound is `0.156250` for factual
  boundary memory, below `0.25`; the largest forced-on binary-KL upper bound is
  `0.003369` for matched verification, below `0.02`. The control-analysis
  SHA256 is
  `fe08831a77bcbc59586141cae12bc537e41687ce2db1b4d1d41b55445d39d037`,
  and its verified three-copy archive SHA256 is
  `f214ae11ab2fa3437404b575d5f091ed3a61c93b5974e5e44d133f238eec94f1`.
- V2 therefore improves V1's protected-control behavior and reduces the
  magnitude of the held behavioral reversal, but it does not cross zero or
  establish mitigation. The primary held behavior effect remains
  significantly harmful, so `behavior_eligible=false` and
  `candidate_may_be_locked=false`. No Pareto lock or final authorization is
  created. `final_test_open=false`, `final_test_open_count=0`, and
  `production_rollout_approved=false` remain unchanged.

## 2026-07-29 Qwen3-8B DSGE V3 held efficacy and protected-control falsifier

- Directional Structural Governance Editor V3 separates the two mismatch
  directions into independently signed experts and retains an exact structural
  gate on matched cells. The immutable code-v29 bundle was evaluated on the
  untouched `operator_dev` split by Job 9097. The run completed 3,072/3,072
  unique finite rows with exact expected/observed key SHA256
  `74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e`,
  successful audit, `COMPLETE`, job-local exit 0, and live Slurm
  `COMPLETED ExitCode=0:0`. The verified three-copy archive SHA256 is
  `99591c8a3dff0ee8d429ed6b1105d7b1d1b1c5564defffc3da67739c35a9e7de`.
- Candidate `DSGE-V3-27:mlp` passes every frozen `operator_dev` efficacy gate.
  Equal-benchmark-weight normalized gap reduction is `0.29118`, with
  item-cluster bootstrap 95% CI `[0.27670,0.30584]`. The corresponding
  matched-minus-mismatched reduction is `0.44572`, bootstrap estimate
  `0.44604`, CI `[0.42367,0.46802]`. Both mismatch directions are positive
  (`0.43717` and `0.45426`), both label-swap halves are positive (`0.39404`
  and `0.49740`), and all six benchmark reductions are positive. Matched-cell
  margin and binary-KL collateral, matched selected-logit error, and the
  external zero-gate error are exactly `0.0`. The analysis SHA256 is
  `b8e34396dab744208c505f92d3f915a8a7107dc73785b378ba7baf7f14aae0e7`.
- The required protected controls were then run in immutable code-v30 without
  modifying code-v29. Job 9098 completed all 2,856 frozen rows with exact
  expected/observed key SHA256
  `220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77`,
  successful original tests 16/16 and control tests 4/4, `COMPLETE`, job-local
  exit 0, and live Slurm `COMPLETED ExitCode=0:0`. The verified three-copy
  archive SHA256 is
  `6242a37c57c26b99728a1f29b11a0ccee0ba344e15953b6f1b9bf9c8e06d936a`;
  the control-analysis SHA256 is
  `64b14a100b93754ace0c81c692f82dcfaa2e5fae1a17ee458d0b10f350814432`.
- Both forced-positive and forced-negative expert stress gates pass in every
  protected family, so the learned expert directions are not the source of
  the control failure. The application router is the sole failed component.
  Four families have exact application-gated identity: matched verification,
  matched delegated choice, supported user authority, and factual boundary
  memory. In contrast, the positive route activates on every fresh-verification
  row and every explicit-governance-reset row. It changes 324/384
  fresh-verification rows (`84.38%`, maximum selected-logit error `0.75`) and
  639/768 reset rows (`83.20%`, maximum error `1.0`); the negative route never
  activates in either family. The separate external zero gate remains exact
  with maximum error `0.0`.
- The terminal V3 decision is therefore mixed but unambiguous: held efficacy
  is strong and symmetric, while application selectivity is falsified.
  `controls_admissible=false`, `candidate_eligible=false`, and
  `candidate_may_be_locked=false`. No V3 Pareto lock or final authorization is
  created, and no frozen threshold is weakened. `final_test_open=false`,
  `final_test_open_count=0`, and `production_rollout_approved=false` remain
  binding. The next version must repair or abstain in fresh/reset states while
  preserving the V3 experts and exact matched/external-zero identity.

## 2026-07-29 Qwen3-8B ADSGE V4 fit-audit falsifier

- ADSGE V4 freezes the effective `DSGE-V3-27:mlp` directional experts and
  history/task heads and trains only a strict binary applicability-veto head.
  Code-v31 Job 9099 and code-v32 Job 9100 were terminal pre-fit harness
  failures: Job 9099 exposed the incorrect declared feature width 56 versus
  the derived width 54, while Job 9100 exposed a repository-relative protocol
  path under Slurm `WorkDir=/workspace`. Neither job started fitting or
  produced an efficacy result. Their run directories and immutable bundles
  remain unchanged.
- Code-v33 fixes only those harness defects and does not weaken any scientific
  gate. CPU-only fit Job 9101 completed in 16:03 with V3 tests 16/16, V4 tests
  11/11, zero skips, `COMPLETE`, job-local exit 0, and a live Slurm terminal
  record of `COMPLETED ExitCode=0:0`. The code-bundle manifest SHA256 is
  `a900aeb157f34c361e3fff5b9fb7f55f8fea771f1cdc470312ce5a4d8884dc9d`;
  the fit-authorization SHA256 is
  `2beca4f86fd330d33e8d5a6956d0edd0d10999719395670556ee0f2fef0e8550`.
- The fit used 10,152 finite rows: 7,548 train, 1,120 calibration, and 1,484
  audit, with 3,072 mismatch positives and 7,080 negatives. The frozen selector
  chose the linear veto with seed 161803 at epoch 120. Its threshold is the
  next float32 value above the maximum calibration-negative logit
  (`-1.7710478306` versus `-1.7710479498`). Calibration mismatch TPR, both
  directional minima, and both label-swap minima are all `1.0`, while the
  calibration negative active fraction is exactly `0.0`.
- The untouched audit fold preserves mismatch sensitivity and protected-state
  abstention but falsifies exact abstention on governance-matched states.
  Audit mismatch TPR, both directional minima, and both label-swap minima are
  `1.0`; all five protected families have active fraction `0.0`. In contrast,
  matched delegated choice activates on 4/224 rows (`0.017857`) and matched
  verification activates on 2/224 rows (`0.008929`). Thus
  `audit_each_governance_matched_family_exact_abstention=false`, the only failed
  fit gate, and the frozen aggregate decision is `fit_eligible=false`.
- The V3 editor remains bitwise frozen (`v3_max_absolute_change=0.0`), and the
  editor-only checkpoint contains no base weights. The fit-report and
  checkpoint SHA256 values are respectively
  `f79e133e5cb0910df5a4a5281067571e25e551e5f49f0d0caa7dabf8d6e65c4a`
  and
  `42e3af0a3501750514c98f70305ac380663cac99ffeb8b26274340e10b7fcaf8`.
  The immutable archive is verified in cluster shared home, archive-host, and
  local Mac storage at SHA256
  `9fbcc5a34dce9602a687f964397bae158e3f144edb4516179145de74bb7bbcd5`.
- This is a terminal scientific falsifier of the frozen V4 selection rule, not
  an execution failure. No threshold is weakened, no audit result is reused to
  choose another architecture or seed, and no behavior smoke, `operator_dev`,
  protected-control, Pareto-lock, or final-test job is authorized.
  `final_test_open=false`, `final_test_open_count=0`, and
  `production_rollout_approved=false` remain binding.

## 2026-07-29 Qwen3-8B GRC-DGE V5.1 dual-lineage fit falsifier

- Recovery Job 9203 reused the already completed 6,144-row governance forward
  from failed Job 9140 and recomputed only the missing protected capture. The
  terminal evidence contains 6,144 unique finite governance rows, 3,072 exact
  counterfactual pairs, expected/observed key SHA256
  `4e6301fce193060c68bb284f7b3349aec61766eb1daa60ff85d200ef9d981900`,
  and 4,008 finite protected rows with exact family counts
  768/1,536/1,536/144/24. Governance and protected manifests bind distinct
  authorization SHA256 values `1503ec72...993940` and
  `10cc2518...e1195`. Job 9203 completed with job-local exit 0 and live Slurm
  `COMPLETED ExitCode=0:0`; the verified three-copy archive SHA256 is
  `0f51da9d3960a3e72dcebe9f362287f2795066936bcf5b7534c19a9f4e040c2e`.
- CPU-only fit Job 9205 used immutable code-v54, authorization SHA256
  `09604f82ae0e3606aa2a9bf4419be1a2c2b2c518857c644795ccb25dd59e438d`,
  and unchanged pre-capture V5 scientific hyperparameters. It completed in
  2:10 on a06 with V3 frozen exactly, two linear router heads, no base weights,
  finite metrics, job-local exit 0, and live Slurm `COMPLETED ExitCode=0:0`.
  A run-copy filename collision overwrote the run-local copy of the fit
  authorization with the protected-capture authorization; the original fit
  authorization remained in shared home and is independently bound by both
  `executed_code.sha256` and the fit report. A strict recovery archive records
  this engineering defect without recomputing or changing scientific output.
  Its verified three-copy SHA256 is
  `6e038c155b88410a0065d3d504faf505e706751c7f4083956729b252582f0778`.
- The final heads achieve exact abstention for every governance-matched and
  protected family and 100% mismatch recall on the developmental audit.
  However, six-fold cross-fit consensus mismatch recall is only `0.756993`;
  the directional minimum is `0.674825` and label-swap minimum is `0.756119`,
  below the unchanged `0.95`/`0.90` gates. All three cross-fit sensitivity
  gates fail while every safety, audit, source-coverage, checkpoint, frozen-V3,
  and forbidden-fold gate passes. Therefore `fit_eligible=false` is a terminal
  scientific falsifier of V5.1 rather than an execution failure.
- No V5.1 behavior smoke or downstream authorization is created, and no gate
  or threshold is weakened. A successor may change its representation or
  training objective under a fresh disclosed contract, but the known
  developmental audit cannot be presented as untouched evidence for that
  successor. `final_test_open=false`, `final_test_open_count=0`, and
  `production_rollout_approved=false` remain binding.

## 2026-07-29 Qwen3-8B PAIR-GE V5.2 cross-fit falsifier

- PAIR-GE V5.2 freezes the V3 expert and base model, expands the 54 routing
  features to 662 fixed interaction features, and trains one shared
  `662→96→48` SiLU trunk with exactly two linear veto heads. Immutable
  code-v55 has bundle-manifest SHA256
  `cf29bb0ba259e3be469a609a3f3546ca61733571abc3c8f8640e5727fe8fee26`;
  the fit authorization SHA256 is
  `47ddfd4c9b84a612435ed138ba01ec83436aa490f9710a8078219aba045ab94a`.
  CPU-only Job 9207 ran on automatically selected node a06 for 10:51 and
  completed with PAIR-GE tests 9/9, inherited V3/V4/V5 Torch tests 16/16,
  11/11, and 7/7, zero skips, finite outputs, `COMPLETE`, job-local exit 0,
  and live Slurm `COMPLETED ExitCode=0:0`.
- The fit used 7,548 allowed-fold rows, including 2,288 matched
  counterfactual pairs, and excluded all 2,604 rows from forbidden folds 6--7.
  V3 remained exactly frozen (`v3_max_absolute_change=0.0`), both checkpoint
  structure gates pass, and the editor-only checkpoint contains no base
  weights. Its SHA256 is
  `1810871c03fbbb1f51cae1585f236ea4d24430babb4b0343608471e43198f85c`.
- Exact negative abstention is preserved under six-fold cross-fit: the active
  fraction is `0.0` for all 5,260 negatives, covering both governance-matched
  families and all five protected families. Sensitivity collapses, however:
  consensus mismatch TPR is only `0.146853`; both directional TPRs and both
  label-swap TPRs are also `0.146853`, below the unchanged `0.95`/`0.90`
  gates. On the disclosed, non-confirmatory developmental audit, negative
  abstention remains exact but mismatch TPR is `0.5`; negative-direction TPR
  is `1.0`, positive-direction TPR is `0.0`, and both label-swap TPRs are
  `0.5`.
- The observed mechanism is conservative-tail collapse rather than a safety
  leak or incomplete optimization. Training losses, paired-ranking losses,
  and negative-tail separation losses approach zero, yet standardized
  held-negative maxima reach `9.304103` for `all_negative` and `30.173845`
  for `matched_state`. The frozen final rule therefore selects robust
  thresholds `62.477989` and `84.952553`, suppressing most mismatches despite
  perfect negative abstention. All cross-fit sensitivity and developmental
  sensitivity gates fail, so `fit_eligible=false` is a terminal scientific
  falsifier of V5.2, not an engineering failure.
- The evidence archive is verified byte-identically in cluster shared home,
  `/data/shang`, and local Mac storage at SHA256
  `d6cd98124c42c791f5177d140d4895cd89b35fc971dd7e97f249338bceda1630`;
  the non-overwriting three-copy receipt SHA256 is
  `c9ac6fde5a0beb831af02d6dbe02f1e39fb8dabbb72e5bca4bf4b6939b512c0d`.
  No V5.2 behavior smoke or downstream authorization is created, and no
  threshold is weakened. `operator_dev` remains untouched,
  `final_test_open=false`, `final_test_open_count=0`, and
  `production_rollout_approved=false` remain binding.

## 2026-07-30 Qwen3-8B ADSGE V4 post-fit behavior diagnostic

- The frozen V4 checkpoint was evaluated after its fit-gate failure under the
  preregistered post-failure diagnostic contract. Code-v56 Job 9217 was a
  terminal harness failure before model loading: `unittest` interpreted three
  absolute test paths as module names, producing zero behavior rows and no
  scientific result. The run and immutable code-v56 bundle remain unchanged.
  Code-v57 changes only the test invocation and binds the same checkpoint,
  fit report, V3 comparison analysis, sample identity, labels, and thresholds.
- Code-v57 Job 9218 completed the same 3,072 `operator_dev` selection rows used
  by DSGE V3 in 16:24 on automatically selected node a06. All 30 Torch tests
  passed with zero skips. The output contains 3,072 unique finite rows, exact
  expected/observed key SHA256
  `74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e`,
  external zero-gate maximum error `0.0`, `COMPLETE`, job-local exit 0, and a
  live Slurm `COMPLETED ExitCode=0:0` record. The verified three-copy archive
  SHA256 is
  `458f3d78b483c18d91438379a9302c5549bf96b5a715f857be0e0c0cc2c7752d`.
- V4 is behaviorally identical to V3 on every one of the 3,072 same-identity
  rows: zero edited rows differ and the maximum selected-margin difference is
  `0.0`. Equal-benchmark-weight normalized gap reduction is therefore the same
  `0.29118` with bootstrap 95% CI `[0.27670,0.30584]`; raw match-advantage
  reduction is `0.89144`, and all six benchmark, both causal-direction, and
  both label-swap aggregates exactly match V3. Both matched-cell margin and
  binary-KL collateral remain exactly `0.0`.
- The composite routing diagnostic explains the identity. The applicability
  head accepts all 1,536 mismatch rows and the final V4 route is active on all
  1,536. It also accepts 11/1,536 governance-matched rows (`0.7161%`), but the
  frozen V3 structural route is inactive on every matched row, so the final V4
  route is active on 0/1,536 and no matched activation changes. The earlier
  exact-abstention fit failure therefore detects application-head overlap, not
  an actual composite edit on these governance-matched behavior rows.
- This diagnostic does not retroactively pass the frozen fit contract:
  `fit_gates_passed=false`, `candidate_eligible=false`, and
  `candidate_may_be_locked=false` remain unchanged. It establishes that V4
  preserves all V3 `operator_dev` efficacy and matched identity, but a separate
  terminal full-forward protected-control diagnostic is still required to
  establish whether it removes V3's fresh/reset collateral in deployed
  behavior. `final_test_open=false`, `final_test_open_count=0`, and
  `production_rollout_approved=false` remain binding.

## 2026-07-30 Qwen3-8B ADSGE V4 post-fit protected-control diagnostic

- The remaining full-forward protected-control question was evaluated under
  the frozen post-fit-failure diagnostic contract, without changing the V4
  checkpoint, V3 directional experts, operator-development evidence, labels,
  or thresholds. Immutable code-v58 has bundle-manifest SHA256
  `bad1f91a376aa49b8a935b3307b5de6811f56a061fd7bf0fd128614ecd1c7057`;
  older code-v56 and code-v57 bundles remain unchanged.
- Job 9249 (`qwen3-8b-adsge-v4-protected-controls-diagnostic-20260729T230430Z`)
  completed on a06 in 27:48. All 33 Torch tests passed with zero skips. The
  output has 2,856/2,856 unique finite rows, exact expected/observed key SHA256
  `220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77`,
  `COMPLETE`, job-local exit 0, and a live Slurm `COMPLETED ExitCode=0:0`
  record.
- The application-gated V4 editor is exactly identical to the unedited model
  on every protected row: 0/2,856 changed rows and maximum selected-logit error
  `0.0`. By family, V4 changes 0/384 fresh-verification, 0/768 explicit-reset,
  0/768 matched-verification, 0/768 matched-delegated-choice, 0/144
  supported-authority, and 0/24 factual-boundary-memory rows. The same frozen
  V3 editor changed 324 fresh and 639 reset rows, so V4 removes all 963 V3
  collateral edits observed in this suite.
- The routing trace isolates the mechanism. The frozen V3 structural route is
  active on 1,152 protected rows (all 384 fresh and all 768 reset rows). The V4
  applicability head accepts none of those rows, so the composite route is
  inactive. Conversely, the applicability head accepts 11 matched rows (5
  matched verification and 6 matched delegated choice), but V3's structural
  route rejects all 11. Therefore the conjunction `V3 structural route AND V4
  applicability` is inactive on all 2,856 protected rows even though the
  auxiliary applicability head is not perfectly selective by itself.
- Forced positive and negative experts are numerically identical to V3 and all
  forced-direction reference gates pass in all six families. Combined with the
  preceding 3,072-row behavior diagnostic, the post-hoc composite-editor result
  is: V4 preserves V3's `29.12%` normalized developmental gap reduction and
  exact matched identity while deleting every observed V3 fresh/reset edit.
  This supports the engineering diagnosis that the frozen fit contract was
  over-conservative for the composed editor: auxiliary-head false positives do
  not imply composite edits when the structural gate already enforces an
  absorbing zero-edit state.
- This is a complete, audited post-failure characterization, not a retroactive
  confirmatory selection. The frozen decision remains
  `fit_gates_passed=false`, `candidate_eligible=false`,
  `candidate_may_be_locked=false`, `final_test_open=false`,
  `final_test_open_count=0`, and `production_rollout_approved=false`. No Pareto
  lock or final-test authorization is created from this diagnostic.
- The archive is verified byte-identically in cluster shared home,
  `/data/shang`, and local Mac storage at SHA256
  `e9f4b2e44676cc8e684435086b5ec2385414a283d1d653db52aa7ea48a769150`.

## 2026-07-31 Qwen3-8B C-DGE V4.1 confirmatory final-test closure

- C-DGE V4.1 prospectively corrects the over-conservative V4 eligibility
  criterion at the composite-router level: an edit is applicable only when the
  frozen DSGE-V3 structural route and the frozen V4 applicability veto are both
  active. The candidate passed the frozen `operator_dev`, protected-control,
  and performance gates and was selected by the immutable Pareto lock as
  `C-DGE-V4.1-27:mlp`. The lock is `admissible=true`, binds all three terminal
  evidence receipts, and kept `final_test_open=false` until the one-time final
  authorization.
- The only model forward for the final split was Job 9324. It generated exactly
  6,144 unique finite rows with expected key SHA256
  `42520481f1fb73449f45c18604a72c32736a8eb3ba238b501f78a0575e4cf6ea`
  and raw runtime-row SHA256
  `b71f4e31d6bf42115f0aad15372810775204050005b521e09464b25cbf7db88c`.
  After the forward, a legacy output writer omitted three already-authorized
  identity fields and the promotion step failed with exit 1. No second model
  forward was run and the final test therefore remained a one-time execution.
- Recovery Job 9333 used immutable code-v79 and only normalized copies of the
  existing rows by restoring `final_test_open`, `final_test_open_count`, and
  `production_rollout_approved` from the bound authorization. It did not change
  logits, labels, margins, decisions, or job keys and records
  `model_forward_reexecuted=false`. Job 9333 completed with job-local exit 0
  and live Slurm `COMPLETED ExitCode=0:0`. The recovered evidence contains
  6,144/6,144 unique finite rows, exact expected/observed key SHA256, zero-gate
  maximum error `0.0`, and a verified three-copy archive SHA256
  `0c598b2898dd85e426cbdcfed57d83c43faa868c670043b3237b9b8ca9c88bc4`.
  The recovery evidence transparently records that Job 9324's controller record
  had expired and persistent accounting was unavailable; no terminal record is
  fabricated for Job 9324.
- On the confirmatory final split, equal-benchmark-weight normalized gap
  reduction is `0.303626` (30.36%), with 10,000-replicate item-cluster
  bootstrap 95% CI `[0.292565, 0.314914]`. Raw match-advantage reduction is
  `0.913981`, bootstrap 95% CI `[0.881346, 0.947184]`, and the corresponding
  matched-minus-mismatched reduction is `0.456991`, 95% CI
  `[0.440348, 0.473227]`.
- Recovery is positive in both causal directions: `0.443115` for obedience
  history on independent verification and `0.470866` for verification history
  on delegated choice. Normalized gap reduction is positive on every benchmark:
  ARC-Challenge `30.98%`, BBH `27.21%`, GSM8K `28.13%`, MATH-500 `34.13%`,
  MMLU-Pro `32.53%`, and MuSR `29.19%`. Both frozen label-swap halves are
  positive (`0.420410` and `0.493571` raw gap reduction).
- Both governance-matched cells have exactly zero selected-margin change and
  zero binary-KL upper bound, including the one-sided 95% bootstrap upper
  bounds. The mean summed site-relative intervention norm is `0.0007126`
  (0.0713%) and the maximum is `0.0042424` (0.424%). The separately audited
  2,856-row protected-control suite has 0/2,856 application-gated changes and
  selected-logit maximum error `0.0` across all six protected families.
- The 720-measurement performance audit estimates C-DGE V4.1 latency overhead
  at `0.8781%` (95% CI `[0.8101%, 0.9491%]`), throughput ratio `0.991302`
  (95% CI `[0.990611, 0.991961]`), and peak-NPU-memory delta about 1.47 MB.
  `production_rollout_approved=false` remains binding; the final authorization
  is evidence-opening authority, not production deployment approval.

## 2026-07-31 Qwen3-8B prospective decision-boundary confirmation

- The prospective boundary analysis was frozen before any forward at protocol
  SHA256 `538b337d206fb8333c565bc42e45e916093b418dc7187660f4ca566430591c88`.
  Job 9350 (`qwen3-8b-prospective-boundary-20260730T183118Z`) completed on a06
  in 14:22 with job-local exit 0 and a live Slurm `COMPLETED ExitCode=0:0`
  record. The audited output contains 6,144/6,144 unique finite rows and 3,072
  matched history pairs with exact expected/observed key SHA256
  `bc67de44ee0235793e29b4cb1de9d673e899a79b9a459f6fea427259c63f78cb`.
- Under independent verification, the frozen low-margin boundary band has a
  44.39% correct-to-wrong mismatch-flip rate (91/205), compared with 0/266 in
  the frozen robust-margin band. The boundary-minus-robust contrast is
  `0.443902`, bootstrap 95% CI `[0.374449, 0.515152]`, with one-sided 95% lower
  bound `0.385417`.
- Under delegated choice, the boundary band has a 4.77% flip rate (21/440),
  compared with 0/345 in the robust band. The contrast is `0.047727`, bootstrap
  95% CI `[0.021504, 0.076405]`, with one-sided 95% lower bound `0.025423`.
  The preregistered success rule therefore passes in both task requirements.
- The secondary six-benchmark ecological diagnostic does not support the
  simpler claim that harder benchmarks suffer larger mismatch losses. The
  Spearman correlations between benchmark difficulty and accuracy drop are
  `0.0857` for independent verification (`p=0.9194`, exact two-sided
  permutation) and `0.3133` for delegated choice (`p=0.6000`). The supported
  predictor is local matched decision margin, not benchmark difficulty alone.
- The archive is verified byte-identically in cluster shared home,
  `/data/shang`, and local Mac storage at SHA256
  `325623a84fb9d4ce0e2bdc8457cfdbefead76ec2a5433944ef92b2fcb03837bf`.
  The analysis used no final-test data and preserves `final_test_open=false`,
  `final_test_open_count=0`, and `production_rollout_approved=false`.

## 2026-07-31 Qwen3.5-9B C-DGE V4.1 cross-model behavior falsifier

- The Qwen3.5-9B replication prerequisites are terminal-audited and retained in
  three verified copies: adapter smoke Job 9355 (archive SHA256
  `9e02236d7151ac6fc7b9a13b10ec29e272be466860db65aad35423295f6c662d`),
  gradient capture Job 9356
  (`488314a2bf1cda4f806d184fd5660c3bf7a7e39e65008707ea18bb57bc93da6a`),
  protected capture Job 9357
  (`197506c89af6574069f461717e25c8acea96487af38f7dd5c7bc0be1a96e3ab7`),
  governance capture Job 9358
  (`bdf1ec26e649805f7dac7b5bc3280303bb86ec5844041295307613c533ce21f5`),
  and C-DGE fit Job 9361
  (`c4d726abd4c8e23a2930a25df54bcc7b135ac32d3b5428ab00f3814c5ac32963`).
  The fit gates all passed, the composite candidate was eligible, and
  `operator_dev` was not accessed during fitting.
- Jobs 9365 and 9366 were pre-forward engineering failures and contain no
  scientific behavior rows. Immutable code-v88 repaired the omitted bundle
  prerequisite, added an explicit prerequisite audit, and passed 68/68 cluster
  Torch tests with zero skips. Its bundle-manifest SHA256 is
  `dbf0344fbaee6cdfc8816a6ad516f6b86e7ea2c8b7edf8b9500027ea4163ff15`.
- Job 9367 (`qwen3-5-9b-cdge-v4-1-behavior-20260730T214109Z`) completed on a06
  in 59:43. It has `COMPLETE`, job-local exit 0, live Slurm
  `COMPLETED ExitCode=0:0`, 68/68 tests, 3,072/3,072 unique finite rows, exact
  expected/observed key SHA256
  `74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e`,
  and coefficient-zero maximum error `0.0`. The behavior authorization SHA256
  is `45af3976fbbe8d905ddcf10c9572f391c3a99028e26e11b2f38dd62db6b71b27`.
  The archive is verified byte-identically in cluster shared home,
  `/data/shang`, and local Mac storage at SHA256
  `e295ed35ee6bbc49df2e58f1e1a1531316aad4204865598a68df53ca5c54bf73`.
- The editor produces a positive but smaller cross-model effect. Equal-weight
  matched-minus-mismatched reduction is `0.191569`; its 10,000-replicate
  item-cluster bootstrap 95% CI is `[0.185710,0.197347]`. Equal-weight
  normalized gap reduction is `0.190500` (19.05%), with 95% CI
  `[0.184683,0.196129]`. Both causal directions are positive (`0.126953` and
  `0.256185`), both label swaps are positive (`0.177734` and `0.205404`), and
  all six benchmark reductions are nonnegative. Both matched cells retain
  exactly zero margin and binary-KL collateral.
- The frozen developmental contract required the lower 95% confidence bound
  of raw gap reduction to be at least `0.25`. The observed lower bound
  `0.185710` fails this single gate; the normalized-gap lower-bound gate
  (`>=0.10`), both-direction, both-label-swap, all-benchmark, and exact-identity
  gates all pass. Consequently `candidate_eligible=false` is a terminal
  scientific falsifier of the predeclared cross-model effect-strength claim,
  not an execution failure and not evidence of zero efficacy.
- Protected controls, performance, Pareto locking, and the one-time Qwen3.5
  final test are therefore not authorized. No threshold is weakened and no
  partial output is promoted. `final_test_open=false`,
  `final_test_open_count=0`, and `production_rollout_approved=false` remain
  binding for Qwen3.5-9B.

## 2026-08-01 Qwen3-8B exploratory adapter-only external baseline comparison

- Six literature-derived `adapter-only` implementations were evaluated on the exact 3,072-row
  `operator_dev` identity used by the frozen DGE comparison. They are
  exploratory implementations of governance reset prompting, session
  isolation, Contrastive Activation Addition (CAA), Conditional Activation
  Steering (CAST), LoReFT, and RePS. These exploratory `adapter-only` results
  are appendix-only and are excluded from the formal comparator table.
  Immutable code-v87 has bundle-manifest SHA256
  `17a149de3b098552ff54c86422aa6e699ddee5e48ff7c4f29d5ae84b77c96803`.
- Input Job 9470, activation Job 9474, and representation Job 9475 each
  completed with 3,072 unique finite rows per method, exact frozen key SHA256
  `74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e`,
  coefficient-zero maximum error `0.0`, scientific audit success, job-local
  exit 0, and live Slurm `COMPLETED ExitCode=0:0`. Their verified three-copy
  archive SHA256 values are respectively
  `0574d67fd5c0b10fabc00827c0df3e11fbfeeceb0d371e0c17eaef8fbe1af116`,
  `fb2373ba18cba2a30d3e8e125cca4096c2bd8aa972406fa4b7a055ee04437653`,
  and `dc01c0b8735e43462373e0c70cb904d84b39aa69d17e4128df6d74e5ff510345`.
- All six methods have bit-identical unedited logits, frozen at SHA256
  `094571ae7f41f2d43efd299ef6074de3dee1764e1ce7f27064293683f2f74320`.
  The merged report contains 18,432 method rows and passes every completeness,
  finiteness, identity, lineage, and safety audit.
- Equal-benchmark-weight normalized gap reduction (95% item-cluster bootstrap
  CI) is: governance reset prompt `5.55%` `[-0.97%,12.00%]`; session isolation
  `40.96%` `[35.69%,46.31%]`; CAA `-0.18%` `[-1.44%,1.05%]`; CAST `8.07%`
  `[6.82%,9.36%]`; LoReFT `0.15%` `[-1.83%,2.14%]`; and RePS `5.00%`
  `[3.25%,6.76%]`.
- Nominal NGR alone is not sufficient. Session isolation damages both
  mismatched directions (`-0.8086/-0.6650`) and has very large matched-cell
  margin shifts (`-2.1813/-1.5962`); its high ratio is therefore collateral,
  not selective correction. Reset prompting also damages both mismatch
  directions and matched cells. RePS moves both mismatch directions strongly
  but moves matched cells almost as strongly (`5.8088/0.9256`). CAA, CAST, and
  LoReFT likewise lack exact matched-cell identity. None satisfies the
  zero-collateral governance-editing contract. DGE's `29.12%` developmental
  effect and C-DGE V4.1's `30.36%` confirmatory effect therefore remain the
  strongest selective results, rather than being inferred from an unpaired
  comparison.
- Jobs 9471 and 9472 under code-v86 failed before scientific evaluation because
  a contextual-adapter method accidentally shadowed
  `torch.nn.Module._apply`. Code-v87 renamed it to `_apply_correction`, added
  `.to("cpu")` regression paths for all four internal baselines, and passed
  4/4 Ascend Torch tests with zero skips. Those failed jobs produced no
  scientific result and are excluded from the merged report.

## 2026-08-01 Qwen3-8B official-code-derived task adaptations

- The formal external comparison consists of two input controls (governance
  reset prompting and session isolation) and four `official-code-derived task
  adaptations` (CAA, CAST, LoReFT, and RePS). This phrase is the formal
  terminology: these are task adaptations derived from fixed official source,
  not byte-identical official reproductions or unmodified official
  implementations. The fixed source manifest is
  `protocol/EXTERNAL_BASELINE_SOURCE_MANIFEST_V2.json`, SHA256
  `d68a0655dad8526e10010c1d9dcd88920647dc8b8b8912a7648f5999738d2098`.
  It binds CAA commit `5dabbbd9a0bca5f25e174501e959de378806aa48`, CAST
  commit `52be60235ee309b46c49d6d5877f36e20c52e6ab`, LoReFT/pyreft commit
  `dafd0995a366d7b47160a337dcc388eda7431821`, and RePS/axbench commit
  `41c8332543e5a631f9a8c0a9df38799893ace758`, together with source-archive,
  extracted-tree, license, and file-count digests.
- Immutable code-v90 has bundle-manifest SHA256
  `d8b1e396cea2f0c52ce5ec92ce63b686a0c5c3be4351071ecf016a3086767684`.
  Job 9501 under code-v88 failed before fitting because the CAST runtime lacked
  sklearn and the RePS rank-2 broadcast test fixture failed. Job 9502 under
  code-v89 failed before fitting because CPU-only torch-npu AdamW attempted
  foreach NPU initialization. Both are engineering failures (`FAILED 1:0`)
  with no scientific result. Code-v90 sets both AdamW optimizers to
  `foreach=False, fused=False`; all 4/4 cluster Torch tests pass with no skip.
- Fit Job 9503 (`qwen3-8b-official-baseline-fit-20260801T084545Z`) completed in
  7:22 with job-local exit 0 and live Slurm `COMPLETED ExitCode=0:0`. Its
  checkpoint SHA256 is
  `ff9f27801523498fbb64708b0a5490b3a43d95202da835f287455c06aca5fbe6`;
  its fit report SHA256 is
  `aa9bd767e318200eaf5d20e05d56fb98f5a98f77e57436210daf4d26bd8a6a0b`;
  and its three-copy archive SHA256 is
  `154f99b600a8a6f083c2d5608c73b495a28a736992e12c634f9f52a8a5951827`.
  The checkpoint contains no base weights. RePS uses two task-specific official
  rank-1 `PreferenceVectorIntervention` modules (`2 x 4096 = 8192` trainable
  weight parameters), with frozen bias. All fit metrics are finite and
  `operator_dev_accessed=false`.
- Activation Job 9504
  (`qwen3-8b-official-baseline-eval-activation-20260801T085640Z`) and
  representation Job 9505
  (`qwen3-8b-official-baseline-eval-representation-20260801T085831Z`) each
  completed with job-local exit 0, live Slurm `COMPLETED ExitCode=0:0`, and
  exactly 3,072 unique finite rows per method. Their three-copy archive SHA256
  values are respectively
  `ecfc0c0e764eb0ae2df086b6db33ab35865e5e6bf7c9c81e63dfbda3edbe1b55`
  and
  `ede34e361ff816b22721800ea0ba999b6b2365854e983be55caa55d88748b3d9`.
  Both scientific audits are complete.
- The formal merged report is
  `artifacts/qwen3-8b-v1/external-baselines/qwen3_8b_official_derived_baseline_comparison.json`,
  SHA256
  `8bac4d7ee78e6c04e468ccdeb2108f0dc19cbdfcfe9aa4de3d03e2059b92907d`.
  It contains six methods, 3,072 rows per method, and 18,432 total method rows.
  All six unedited logits are bit-identical at SHA256
  `094571ae7f41f2d43efd299ef6074de3dee1764e1ce7f27064293683f2f74320`.
  The old `adapter-only` report is retained separately and is not included in
  this formal merge.
- Equal-benchmark-weight NGR (95% item-cluster bootstrap CI) for the four
  `official-code-derived task adaptations` is: CAA `-0.54%`
  `[-1.79%,+0.70%]`; CAST `+5.02%` `[+3.72%,+6.35%]`; LoReFT `0.00%`
  `[0.00%,0.00%]`; and RePS `+10.38%` `[+8.04%,+12.76%]`. Directional
  recoveries are respectively `-0.000/+0.002`, `-0.002/+0.349`,
  `0.000/0.000`, and `+0.594/+9.988`.
- None is selective mitigation. CAA is statistically indistinguishable from
  zero. CAST has positive aggregate NGR but a slightly negative recovery in one
  mismatch direction and fails matched identity. LoReFT makes no effective
  change under the frozen correction cap. RePS has the largest formal external
  NGR, but its matched-cell mean shifts are `+9.919271` for
  obedience-history/delegated-choice and `+0.424316` for
  verification-history/verification, so the apparent recovery is accompanied
  by large collateral movement. DGE remains `+29.12%` with both directional
  recoveries positive and exact matched identity.
- Every run preserves `final_test_open=false`, `final_test_open_count=0`, and
  `production_rollout_approved=false`. The verified copies are in cluster
  shared home, `/workspace/context-mismatch-qwen3-8b/evidence-archives`,
  and the local Mac evidence tree; archive-host retains no evidence copy.

## 2026-08-01 Qwen3.5-9B native C-DGE V4.2 rediscovery and protected-control falsifier

- A fresh architecture-adaptive workflow rediscovered Qwen3.5-9B sites and
  ranks instead of copying the Qwen3-8B layer-27 choice. Terminal component and
  residual scans, native capture, three independent fits, and 27 complete
  held-behavior candidates selected layer `31:mlp`. The frozen behavior merge
  SHA256 is
  `91bb7565b2d47e9bee92fd30acc48d1638c55e8fac64509d645d79090e5fd8ea`.
- The two frozen tied finalists differ only in maximum relative-correction cap
  (`0.05` versus `0.075`) and are behaviorally identical: normalized gap
  reduction `27.06%` with 95% CI `[25.80%,28.25%]`, raw reduction `0.2712`
  with 95% CI `[0.2584,0.2833]`, exact matched identity, both causal directions
  positive, both label swaps positive, and site `31:mlp`. Native rediscovery
  therefore recovers substantially more effect than direct V4.1 transfer
  (`19.05%`) and passes the frozen efficacy criterion.
- Protected-control Jobs 9463 and 9464 each completed all 2,856 unique finite
  rows with exact expected/observed key SHA256
  `220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77`,
  audit success, job-local exit 0, and captured Slurm
  `COMPLETED ExitCode=0:0`. Their verified three-copy archive SHA256 values are
  `6721c26f51d222826f3eda0948dd63d452d4953a58e2068da6e61b6dda295958`
  and `36ac579bd074d08930c3835d83d90ea121e026ebd6225bab63c36a4bc2b3881c`.
  The original code-v106 archive helper captured each live terminal record and
  created the cluster-home archive; a pull-helper interface defect omitted the
  `/WORK` promotion. The versioned recovery helper verified the archived
  terminal record and complete artifact tree before non-overwriting `/WORK`
  promotion and Mac pull. No scientific file was rewritten.
- Both tied candidates fail exactly the same frozen selectivity gate. All
  forced-on safety reference gates pass and the maximum forced-on relative
  correction is only `0.0003021`, but the application-gated editor changes
  explicit-governance-reset rows, with maximum selected-logit error `0.125`.
  Consequently `every_family_passes_individually=false`,
  `controls_admissible=false`, and `candidate_may_be_locked=false` for both.
- The immutable V4.2 Pareto report is complete but `admissible=false`; the
  lock file is intentionally absent. This is a selectivity falsifier after
  successful architecture-native efficacy, not an execution failure and not
  evidence that rediscovery is useless. No threshold is weakened and no
  Qwen3.5 final authorization is created. `final_test_open=false`,
  `final_test_open_count=0`, and `production_rollout_approved=false` remain
  binding.
