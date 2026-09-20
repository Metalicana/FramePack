# External baseline comparison: local code, user-run cluster jobs

**Current requested scope: 15 matched 60-second videos.** The full-study commands
below still describe the original 180-second implementation; do not launch that
stage. We are first repairing the engineering smoke, reusing its original first
trajectory so the framing/precision change can be compared. Selecting the exact
60-second cohort and updating the full launcher/evaluation remains subsequent work.

## Revised smoke: preserve aspect ratio and benchmark BF16

After pushing/pulling FramePack changes, run from the cluster FramePack checkout:

```bash
export STUDY_PYTHON="$CONDA_PREFIX/bin/python"
bash external_baselines/run.sh nominal-smoke \
  --config "$HOME/external-baselines.json" --gpu 0 \
  --smoke-spatial letterbox --smoke-precision bf16
```

For the observed 640×360 source PNG, the initial image is resized isotropically to
256×144 and placed at y=56 inside a 256×256 black canvas. The camera intrinsics
use that exact scale and padding translation. It preserves all original framing
and avoids the earlier square squeeze. This is an experimental letterboxed input
to a square-trained checkpoint, not evidence of native rectangular support.
The inspected U-ViT positional geometry and DFoT ray generation both assume a
square image, so simply changing height/width would not be a supported fix.

Outputs are isolated in `STUDY/nominal_smoke_letterbox_bf16/`. Native square videos
remain under `smoke/`. `preview_640x352.mp4` removes only the padding region and
resizes that content to MemCam's display dimensions. This preview is upscaled
256×144 content, not native 640×352 generated detail, and is not eligible for
metrics. The display resize mirrors MemCam's full-image 640×360-to-640×352 resize.

BF16 autocast applies to eligible network operations; model weights, diffusion
state and upstream camera processing stay float32. Sampling steps, guidance and
seed are unchanged. Numerical outputs may change. Nonfinite output pixels fail
the run rather than being encoded silently. The resources JSON records autocast
precision and measured time/memory; speedup is not established until this runs on
the cluster. Use `--smoke-precision float32` for a separate framing-only check;
it writes to `nominal_smoke_letterbox_float32/`.

The existing original smoke is preserved. No full rollout is launched by either
command, and nominal calibration is not promoted to verified calibration.

All new code is in `FramePack/external_baselines/`. Push this directory with the
FramePack repository, then pull it on the cluster. Nothing here uses SSH, submits
Slurm jobs, trains models, downloads model weights, or regenerates MemCam videos.
Use CECSL directly inside your own tmux session. On Newton, run inside an allocation
you obtain under the site's actual rules; no allocation details have been assumed.

**Status:** implemented locally, with CPU contract tests. No checkpoint load,
GPU smoke, generated comparison video, or metric run has been validated here.
The first cluster action is the audit below. Paths in the example are placeholders
or previously recorded locations, not confirmed cluster inventory.

Stock FramePack is excluded at your request. Both inspected workers (original and
F1) accept an image and text, but no numeric camera trajectory. A camera-conditioned
extension with compatible trained weights needs a separate adapter audit before
it can be included. This implementation supports only official pixel-space DFoT
RE10K EMA weights; these are trained on RealEstate10K, not the shared CaM baseline.

## 1. Pull, configure, and audit

Create the separate DFoT environment first, on the cluster:

```bash
conda create -n dfot -c conda-forge python=3.10 pip numpy=1.26 pillow ffmpeg -y
conda activate dfot
export STUDY_PYTHON="$CONDA_PREFIX/bin/python"
```

This follows the checked-out DFoT README's Python 3.10 choice and its NumPy 1.x
requirement. It installs the prerequisites for the inventory and CPU tests;
it is not yet the complete inference environment. Run the audit before choosing
the CUDA-enabled PyTorch build, so that choice can match the installed NVIDIA
driver. The upstream dependency file leaves Torch and many other versions
unpinned; do not treat a fresh unconstrained install as a tested environment.
Keep the existing `memcam` and `vbench` environments separate.

The supplied CECSL audit reports two RTX PRO 6000 Blackwell GPUs (97,887 MiB each),
driver 610.57.04, and environments under `/home/ab575577/miniconda3/envs/`.
Update all three configured Python paths accordingly. The dataset and 15-row
manifest exist; source manifest SHA256 is
`f0676a252e4d528e8ee7b512bd2bb9bf3090ebc32f3e67ef9b5b6ac9073130a8`.
The sampled pose has position/rotation/scale only, so it does not establish
intrinsics. No DFoT checkpoint was reported in the searched roots; also check
the DFoT repository's `huggingface/` directory, which its downloader uses.

For this Blackwell machine, begin with the official CUDA 12.8 Torch wheel pair:

```bash
conda activate dfot
python -m pip install setuptools==81.0.0
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c "$HOME/FramePack/external_baselines/constraints-blackwell.txt" -r "$HOME/diffusion-forcing-transformer/requirements.txt"
python -m pip check
```

PyTorch 2.7 introduced Blackwell/CUDA 12.8 support:
https://pytorch.org/blog/pytorch-2-7/ . The paired 2.7.1/0.22.1 installation is
listed at https://pytorch.org/get-started/previous-versions/ . These constraints
prevent downstream requirements from replacing the selected Torch pair or NumPy
1.x; they are not a tested complete dependency lock. Check actual CUDA execution
and DFoT imports before considering installation successful. Cluster DFoT revision
`0fac21e2df8636dfa1dd47bf74572c5ad80778fe` differs from the locally inspected
revision, so compatibility must be established by the import/model smoke checks.

The supplied CECSL install log confirms Torch 2.7.1+cu128 executes CUDA matrix
multiplication successfully. DFoT import then fails in TorchMetrics 0.11.4 because
Setuptools 84 removed `pkg_resources`. The Setuptools 81.0.0 pin above restores
that legacy API; rerun the DFoT import after installing it. Passing `pip check`
alone does not verify these runtime imports.

From the cluster FramePack checkout:

```bash
cp external_baselines/config.example.json /tmp/external-baselines.json
# Edit /tmp/external-baselines.json to match your actual paths.
export STUDY_PYTHON="$CONDA_PREFIX/bin/python"
bash external_baselines/run.sh audit --config /tmp/external-baselines.json
```

The audit needs standard Python only. Subsequent stages need NumPy and Pillow in
`STUDY_PYTHON`. Choose an existing absolute interpreter; the launcher does not
activate environments. Generation uses `dfot_python`; quality uses `metric_python`;
standard VBench uses `vbench_python`. Keep the DFoT environment separate from the
working MemCam/VBench environments. Use the checked-out DFoT README/requirements
when preparing that environment. No package installations are done by this code.

Audit writes `STUDY/audit.json`: hostname, GPUs, disk, environments, repository
commits/source hashes, configured path existence, and checkpoint candidates within
the explicitly configured search roots. Candidate filenames do not establish
checkpoint provenance. Record the verified source/revision in
`checkpoint_provenance`; the launcher hashes the checkpoint itself. It fails if
the checkpoint is not a released EMA checkpoint or its model keys are incompatible.
Only load a checkpoint whose source you trust: upstream Lightning checkpoints use
Python pickle via `torch.load`.

Use the original 15-row `context_memory_180s/manifest.jsonl`. If it is only on
Newton, transfer that exact file yourself and set `source_manifest` to the copied
file. `source_dataset_root` is the prefix actually present in its path fields;
`dataset_root` is the destination dataset prefix. The launcher preserves original
bytes, all non-path fields, relative file identities, and row order. It never
rebuilds the split. Required values are 30 FPS, 5,397 frames, nominal 180 seconds.
The effective encoded duration is 179.9 seconds; the last timestamp is 5396/30.

## 2. Camera metadata is required before smoke

### Published-settings engineering smoke

Appendix B reports 24 mm focal length, 52.67-degree FOV, and f/10:
https://arxiv.org/html/2506.03141v2#A2 . The FOV axis and its mapping to the
distributed PNGs remain assumptions. Do not use MemCam's retrieval FOV constants.

Once the checkpoint has been downloaded and its path/source set in the config:

```bash
cd "$HOME/FramePack"
export STUDY_PYTHON="$CONDA_PREFIX/bin/python"
bash external_baselines/run.sh nominal-smoke --config "$HOME/external-baselines.json" --gpu 0
```

This separate engineering stage reads the first manifest input PNG's actual W,H,
sets `f = W/(2*tan(52.67 degrees/2))`, and forms normalized intrinsics
`[f/W, f/H, 0.5, 0.5]`. The adapter resizes the full image to 256×256 using Pillow
bicubic, with no crop or padding. Its image-edge coordinate transform is
`diag(256/W,256/H,1)`, so normalized intrinsics remain unchanged. Pixel centers
are i+0.5, matching the inspected DFoT ray code. Native square pixels do not imply
equal fx/fy after this anisotropic resize. The 24 mm value and aperture are
recorded as renderer provenance, not substituted into pixel focal-length fields.

Only the first trajectory's 65 frames are generated by default (hard cap 200).
This stage requires no evaluator configuration or detector weights. It writes
`STUDY/nominal_smoke/inputs/nominal_calibration.json`, camera previews, provenance,
and `STUDY/nominal_smoke/smoke/` video/attempt receipts with the explicit label
`nominal_paper_smoke_only`. It never sets `verified=true`, evaluates metrics,
or produces the approval estimate required by the full-run stage. Missing
checkpoint/calibration compatibility still fails visibly. Inspect the resulting
video and metadata before establishing the full protocol.

### Verified calibration for the matched study

Set `calibration_file` to a JSON object keyed by the exact manifest scene names.
Each scene needs the following record. **This example is intentionally unverified;
do not turn verification on until the source camera convention and intrinsics
have been checked.**

```json
{
  "EXACT_SCENE_NAME": {
    "verified": false,
    "evidence": "Record camera export metadata/FOV, image dimensions, axis and known-motion checks here",
    "position_units_per_meter": 100,
    "opencv_camera_to_ue_camera": [[0, 0, 1], [1, 0, 0], [0, -1, 0]],
    "normalized_intrinsics_after_resize": [0.0, 0.0, 0.5, 0.5]
  }
}
```

The matrix shown maps OpenCV right/down/forward camera axes into UE
forward/right/up axes. It is a candidate convention, not verified dataset metadata.
The code matches MemCam's `Rz(yaw) @ Ry(pitch) @ Rx(roll)` and centimeter-to-meter
conversion, changes world and camera bases together, then inverts to w2c. DFoT
receives `[fx, fy, cx, cy, flattened_w2c_3x4]`, and its upstream code performs
reference-camera normalization. Intrinsics must be normalized to image width and
height. Full-image resizing preserves these normalized values; there is no crop.
Do not guess focal lengths from another dataset. If focal length varies across a
trajectory, this constant-per-scene adapter must be extended before running.

The preflight checks all requested pose indices and writes first/second/last
converted poses to `inputs/camera_preview_*.json`. It requires consecutive dataset
frame keys, rather than silently treating sparse keys as consecutive frames.
Known translation and yaw tests verify the conversion math, but do not establish
that the real dataset's export convention matches the candidate basis.

## 3. Freeze the actual evaluation configuration

Set `quality_reference_summary` to an existing 180-second source result's
`summary.json`. The launcher requires stride-30, 224-pixel LPIPS and StyleGAN-V I3D
FVD with 16-frame clips, four clips/video, stride four, 224 pixels, epsilon 1e-6.
Set `fvd_detector` and `fvd_detector_sha256` from that source run's actual detector.
Use `sha256sum /path/to/detector` on Linux. Quality downloads are disabled.

`vbench_reference_config` is a JSON record transcribed from the actual saved
standard VBench invocation and weights, with these fields:

```json
{
  "source_evidence": "/absolute/path/to/source/invocation-or-config",
  "mode": "custom_input",
  "dimensions": ["subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"],
  "imaging_quality_preprocessing_mode": "longer",
  "checkpoint_sha256": {
    "/absolute/path/to/each/local/metric/checkpoint": "actual_sha256"
  }
}
```

Include all weights actually used, not one representative detector. The launcher
verifies these hashes and calls VBench with local checkpoints enabled. Verify its
cache lookup matches those paths. Set `vbench_imaging_preprocessing` to the same
saved setting. The launcher copies the reference records into `inputs/` and
fingerprints the code and environments. A record of defaults alone is not evidence
of what generated the earlier results.

DFoT uses a full-image bicubic resize to 256×256 and float32 inference. It ignores
text because this checkpoint has no prompt interface. Native DFoT files are kept
untouched. Output index k is GT `start_frame+k`; there is no FPS relabeling, repeated
output padding, or future-GT interpolation. The first output is the resized
initial image. The other observations used for continuation are generated.

The MemCam evaluator resizes GT to each generator's native frame size before
LPIPS's 224-pixel preprocessing. Consequently DFoT and MemCam have different native
resolution paths even with the same evaluator flags. Review and disclose this
when approving the protocol; if you require a single identical spatial resampling
path, implement that evaluator change and reevaluate existing MemCam videos under
it before combining scores. Do not copy historical numbers under a changed protocol.

## 4. Smoke, inspect, then approve the full run

```bash
export STUDY_PYTHON=/absolute/path/to/python-with-numpy-and-pillow
bash external_baselines/run.sh smoke --config /tmp/external-baselines.json --gpu 0
```

This runs the first trajectory for 65 frames by default, in `STUDY/smoke/`, and
never marks it as a completed study video. Inspect its video, converted poses,
`frame_map.csv`, `resolved.yaml`, logs, and `smoke/estimate.json`. Preflight checks
all initial images and all GT filenames; generation only opens the initial image
and pose JSON. Future GT images are read only by evaluation.

Inference calls upstream `_predict_videos`, with keyframe density 0.0625,
stabilized-vanilla prediction guidance 4.0/stabilization 0.02, vanilla interpolation
guidance 1.5, and interpolation batch size four. Sampling steps and remaining
settings are recorded in `resolved.yaml`. Compilation is disabled. A narrow
override pads only a short final camera-conditioning window using upstream's
`_pad_to_max_tokens`; those masked slots do not create output frames.

**Full-length GPU memory is not established by a short smoke.** Upstream retains
full prediction tensors and constructs interpolation buffers; its memory usage
grows with length. The linear time estimate is approximate and excludes metrics.
If the first full-length video fails for memory, keep its log and adjust the
implementation/configuration in a new study directory; do not silently split the
trajectory into GT-initialized chunks.

After inspecting smoke and confirming available GPU time, one command handles
missing generation, validation, quality metrics, VBench, and exports:

```bash
bash external_baselines/run.sh run \
  --config /tmp/external-baselines.json --gpu 0 \
  --protocol-approved --approved-hours 24
```

Replace 24 with your actual approved budget, allowing time for metrics. This is
an admission/between-stage budget check, not a hard process-kill timer: a running
video or metric stage can finish beyond it. Use allocation limits for a hard wall
time. Generation and metrics run sequentially on the explicitly selected GPU.
The script does not launch workers on other GPUs. Reissue the same command to
resume; it validates code/config/environment/input/checkpoint hashes, the video
hash, full decoding, exact frame count, resolution and frame timestamps. Changed
inputs or settings require a separate study root. Filename existence is insufficient.

Successful outputs are atomically renamed from per-attempt directories. Failed
attempts and metric directories are retained. A crash between the video rename
and receipt write leaves an untrusted video: move it aside for inspection before
retrying. The launcher will not silently overwrite it. A nonzero exit indicates
requested work is incomplete; `failure.json` and `tables/coverage.csv` explain why.

## 5. Existing MemCam results and optional VBench-Long

Add verified source rows to `existing_results` before smoke/provenance freezing:

```json
{
  "run": "ACTUAL_OUTPUT_DIRECTORY_NAME",
  "manifest": "/absolute/path/to/canonical/manifest.jsonl",
  "dataset_root": "/prefix/used/in/that/manifest",
  "quality_dir": "/absolute/path/to/180s/quality/results",
  "vbench_dir": "/absolute/path/to/180s/standard/vbench/results",
  "checkpoint": "verified MemCam checkpoint/revision"
}
```

The collector checks path-independent manifest identity, every quality record's
scene/start/count/status, one complete 60-clip cohort FVD, quality parameters, and
all 15 identities in each VBench dimension. It reuses MemCam's `validate_bench`,
including imaging-quality scaling exactly once. Source artifacts are hashed into
`inputs/source_RUN.json`. Verify saved generation configs for seed, checkpoint,
resolution, and initial observations yourself before registering these historical
rows: those differing config formats are not automatically parsed. Historical
headlines are never used as fallback values. An empty `existing_results` generates
only the DFoT score row and is not yet the complete MemCam comparison.

Optionally add `vbench_long_command` as an argv array beginning with an absolute
Python executable and your existing validated wrapper invocation, using its saved
configuration and this exact cohort. It runs after standard metrics, logs to
`logs/vbench_long.log`, and remains a separate result family. The wrapper must
validate its own outputs; its zero-exit receipt is not standard VBench validation.
No CUT3R calibration or 60-second suite is launched.

## 6. Files produced and local checks

Under the configured study root:

- `inputs/`: original/remapped manifests, hashes, reference configs, camera previews.
- `provenance.json`: checkpoint, code/environment fingerprints, hardware and protocol.
- `smoke/`: short output and cost estimate, separate from the real cohort.
- `dfot_re10k/`: 15 final `output_prefix + custom.mp4` files and receipts; attempt
  subdirectories hold resolved settings, cameras, index maps, resources and logs.
- `metrics/`: quality, exact-cohort standard VBench inputs/results, collected scores.
- `logs/`: import checks, evaluators, collection, and optional long-evaluator log.
- `tables/coverage.csv`, `tables/scores.csv`, `tables/comparison.tex`: unavailable
  scores remain blank; FramePack exclusion appears in coverage. VBench is [0,1].

Resource columns distinguish startup/checkpoint loading, synchronized generation,
and video encoding/transfer time. Peak CUDA allocated/reserved bytes and peak
process RSS are separate. Retrieval latency is N/A. These timings must not be
compared to MemCam's retrieval-only CPU microbenchmark as generation speed.

Run local tests from the FramePack checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s external_baselines/tests -v
```

Local source audit: FramePack `97fe5dbe06ac1f337ece08935b1076a35eefeeb9`,
DFoT `530f8bf4db91a21964993f214fa4050dbd529591`,
MemCam `035efc375b35a7db8499fbdcedf4541f57b473a2`.
No canonical dataset/manifest or matching checkpoints were found locally. The
cluster audit and smoke are required to establish compatibility and feasibility.
