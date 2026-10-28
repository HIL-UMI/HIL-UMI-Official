<h1 align="center">HIL-UMI</h1>

<h3 align="center">
  Bringing Human-in-the-Loop Post-Training of<br>
  Vision-Language-Action Models to Universal Manipulation Interface
</h3>

<p align="center">
  Zimu Han<sup>*1,4</sup> &nbsp;·&nbsp;
  Yiming Zeng<sup>*1,4</sup> &nbsp;·&nbsp;
  Jiyao Zhang<sup>*‡1,2,3</sup> &nbsp;·&nbsp;
  Zihao Zhao<sup>1</sup> &nbsp;·&nbsp;
  Yuanfei Wang<sup>1,2,3</sup> &nbsp;·&nbsp;
  Yixiang Jin<sup>5</sup><br>
  Shiqi Li<sup>5</sup> &nbsp;·&nbsp;
  Shuangben Chen<sup>1</sup> &nbsp;·&nbsp;
  Wei Huang<sup>1</sup> &nbsp;·&nbsp;
  Ruodai Li<sup>5</sup> &nbsp;·&nbsp;
  Hui Shen<sup>5</sup> &nbsp;·&nbsp;
  Hao Dong<sup>†1,2,3</sup>
</p>

<p align="center">
  <sub>
    <sup>1</sup> Center on Frontier Computing Studies, School of Computer Science, Peking University<br>
    <sup>2</sup> National Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University<br>
    <sup>3</sup> PrimeBot &nbsp;·&nbsp; <sup>4</sup> Xi’an Jiaotong University &nbsp;·&nbsp; <sup>5</sup> JD Technology<br>
    <sup>*</sup> Equal contribution &nbsp; <sup>‡</sup> Project lead &nbsp; <sup>†</sup> Corresponding author
  </sub>
</p>

<p align="center">
  <a href="https://hil-umi.github.io/"><img src="docs/assets/button-website.svg" alt="Project Website" height="42"></a>
  <a href="https://arxiv.org/pdf/2609.20659"><img src="docs/assets/button-paper.svg" alt="Paper PDF" height="42"></a>
  <a href="#video"><img src="docs/assets/button-video.svg" alt="Video" height="42"></a>
  <a href="#citation"><img src="docs/assets/button-citation.svg" alt="BibTeX Citation" height="42"></a>
</p>

<p align="center">English &nbsp;|&nbsp; <a href="README.zh-CN.md">简体中文</a></p>

<p align="center"><strong>Robot-free data collection for human-in-the-loop VLA post-training.</strong></p>

<!-- Figure 1 from the HIL-UMI paper (arXiv:2609.20659v2). -->
<p align="center">
  <a href="https://arxiv.org/pdf/2609.20659">
    <img src="docs/assets/teaser.png" alt="HIL-UMI teaser: robot-free UMI collection with online OOD detection, collection efficiency, and task progress across post-training rounds" width="100%">
  </a>
</p>

<p align="center">
  <sub>Policy-guided UMI collection improves task performance while collecting data 5.63× faster than HG-DAgger on Clean Up Table.</sub>
</p>

---

## Overview

HIL-UMI queries the current policy while an operator demonstrates a task with a
handheld Universal Manipulation Interface (UMI). Online feedback identifies
segments for policy improvement and Advantage refinement, connecting data
collection to iterative vision-language-action (VLA) post-training.

This repository contains the corresponding π₀.₅ training, Advantage estimation
and annotation, collection adapters, and single-Franka evaluation code.

<p align="center">
  <a href="#video">Video</a> &nbsp;·&nbsp;
  <a href="#method">Method</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#collect-a-round-and-update-iteratively">Iterative workflow</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

<a id="video"></a>

## Video

The preview shows a successful Clean Up Table rollout, followed by the
comparison with HG-DAgger. Click the preview to open the full overview video.

<p align="center">
  <a href="docs/assets/hil-umi.mp4?raw=1">
    <img src="docs/assets/video-preview-success.gif" alt="Watch HIL-UMI: successful Clean Up Table rollout and results comparison" width="100%">
  </a>
</p>

<p align="center">
  <a href="docs/assets/hil-umi.mp4?raw=1"><strong>▶ Watch / download the full video</strong></a> &nbsp;·&nbsp; 3 minutes
</p>

<a id="method"></a>

## Method

<!-- Figure 2 from the HIL-UMI paper (arXiv:2609.20659v2). -->
<p align="center">
  <a href="docs/assets/method.png">
    <img src="docs/assets/method.png" alt="HIL-UMI method: online Advantage and policy data collection, Advantage estimator refinement, and advantage-conditioned behavioral cloning" width="100%">
  </a>
</p>

Each post-training round connects four steps:

1. **Collect Advantage data.** Low online advantage predictions identify segments
   for refining the progress-based Advantage estimator.
2. **Collect policy data.** The Energy Score compares human demonstrations with
   the current policy’s action distribution to identify OOD segments.
3. **Refine the estimator.** Update Advantage with the newly collected data and
   use it to label policy-training demonstrations.
4. **Update the policy.** Continue advantage-conditioned behavioral cloning
   (ACBC) on a balanced mixture of base demonstrations and new policy data.

---

## Prerequisites

- Use a Linux/CUDA host for model training and inference, with Git, `uv`, and the
  build tools required by the dependency manifests. CPU checks can run separately.
- Obtain task data, pretrained π₀.₅ weights, and tokenizer assets under their
  providers' licenses. This source release **does not include** HIL-UMI datasets
  or checkpoints, and does not provide a data or model download service.
- Users need their own UMI data collection system, including the necessary
  hardware and software.
- Real-robot evaluation requires a calibrated single Franka, cameras, ROS Noetic,
  libfranka/controllers, and the corresponding hardware SDKs.


## Installation

Use the distinct environments named in the table below. Model and robot
environment paths are relative to the repository root. Policy and Advantage
retain their own dependency locks and Transformers integration patches.

| Purpose | Environment | Python import | Interpreter selector |
| --- | --- | --- | --- |
| Policy training and inference | `.venv-policy` | `training`, external `openpi` | `POLICY_PYTHON` |
| Advantage training, annotation, and monitoring | `.venv-advantage` | `hil_umi_advantage` | `ADVANTAGE_PYTHON` |
| Franka hardware control | `.venv-franka` | `training` robot runtime | `FRANKA_PYTHON` |
| External recorder | User-provided external environment | External `data_recorder` | `RECORDER_PYTHON` |

`OOD_PYTHON` overrides the mode-specific interpreter; normally leave it unset.

```bash
# From the repository root, on the Linux/CUDA training host
export HIL_UMI_ROOT="$(pwd -P)"
bash requirements/install.sh embodied --model openpi --env franka --venv .venv-policy

# Separate robot-control environment; ROS/controllers remain external prerequisites
bash requirements/install.sh franka --venv .venv-franka

# Separate Advantage environment (do not activate the policy environment here)
(cd advantage && UV_PROJECT_ENVIRONMENT="$HIL_UMI_ROOT/.venv-advantage" \
  GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --group ood)

# Apply the Advantage Transformers patch only in its own environment
export ADVANTAGE_PYTHON="$HIL_UMI_ROOT/.venv-advantage/bin/python"
cp -R "$HIL_UMI_ROOT/advantage/src/hil_umi_advantage/models_pytorch/transformers_replace/." \
  "$("$ADVANTAGE_PYTHON" -c 'import transformers; print(transformers.__path__[0])')/"

# Lightweight collection adapters; the external recorder stays in its own environment
uv pip install --python .venv-policy/bin/python -e ./collection
uv pip install --python .venv-advantage/bin/python -e ./collection
```

Model environments use Python 3.11; the external recorder uses a separate
Python 3.12 environment. The patch above targets `transformers==4.53.2` and must
only be applied to the Advantage environment. When reusing existing environments,
select their interpreters with the variables in the table. Replay encoding also
requires a system `ffmpeg` installation.

## Data and model preparation

Use local working copies of LeRobot **v2.1** datasets. Raw UMI data contains
47-dimensional `observation.state`, 30-dimensional `action`, and two video
streams, `cam_left` and `cam_front`. The converter produces 10-dimensional
single-Franka `state` and `actions` (XYZ, rotation-6D, and gripper), along with
`image` and `extra_view_image`. An archive containing only front-view videos
cannot serve as complete training data.

```text
dataset/<task>_umi/                         # raw UMI LeRobot v2.1 data
dataset/<task>_franka/                      # converted single-Franka data
advantage/data/<task>/base/origin/           # canonical base demonstrations
advantage/data/<task>/stage1/origin/         # newly collected round-1 data
checkpoints/pi05/base/model.safetensors     # policy initialization
advantage/checkpoints/pi05_base/            # Advantage initialization
checkpoints/advantage/                      # trained Advantage estimators
outputs/checkpoint_offload/                 # policy training states
collection_dataset/                        # recorded correction datasets
```

These paths are storage examples; the corresponding assets must be prepared
separately. Data, weights, environments, and runtime outputs must not be
committed. Keep the original recordings and annotate a working copy. For the
towel example, first convert the base demonstrations:

```bash
source .venv-policy/bin/activate
export PI05_BASE_CKPT="$HIL_UMI_ROOT/checkpoints/pi05/base"
export TASK_PROMPT="fold the towel on the table and put it into the basket"
python toolkits/lerobot/convert_umi_left_to_single_franka.py \
  --src dataset/fold_towel_umi \
  --dst advantage/data/fold_towel/base/origin \
  --prompt "$TASK_PROMPT"
```

## Train the initial Advantage estimator

The GT command modifies parquet and metadata **in place**. Each episode uses one
linear progress curve from 0 at its first frame to 1 at its last frame. For other
tasks, configure the matching prompt and data schema.

```bash
cd "$HIL_UMI_ROOT/advantage"
export ADV_BASE_DATASET="$PWD/data/fold_towel/base/origin"
export ADVANTAGE_PI05_BASE="$PWD/checkpoints/pi05_base"
export ADVANTAGE_CHECKPOINT_ROOT="$HIL_UMI_ROOT/checkpoints"

# Inspect first; omit --dry-run to write annotations to your working copy
UV_PROJECT_ENVIRONMENT="$HIL_UMI_ROOT/.venv-advantage" uv run python scripts/annotate_stage_progress_gt.py \
  --dataset-dir "$ADV_BASE_DATASET" --linear --dry-run
UV_PROJECT_ENVIRONMENT="$HIL_UMI_ROOT/.venv-advantage" uv run python scripts/annotate_stage_progress_gt.py \
  --dataset-dir "$ADV_BASE_DATASET" --linear

CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
bash advantage train-base towel_base_v1 \
  --data.repo-id="$ADV_BASE_DATASET" \
  --num-train-steps=5000 --save-interval=1000 --batch-size=64
```

## Label demonstrations and train the policy

Offline annotation creates a new dataset without modifying the input. The policy
uses its `task_index` and positive/negative prompts.

```bash
cd "$HIL_UMI_ROOT/advantage"
export BASE_ADV_CKPT="$HIL_UMI_ROOT/checkpoints/advantage/ADVANTAGE_TORCH_TOWEL_FOLD/towel_base_v1/5000"
bash scripts/label_advantage_datasets.sh \
  --checkpoint "$BASE_ADV_CKPT" \
  --output-root "$HIL_UMI_ROOT/dataset/fold_towel_acbc" \
  --dataset base="$PWD/data/fold_towel/base/origin" \
  --devices cuda:0,cuda:1 --batch-size 32 \
  --future-frames 50 --top-percent 30

cd "$HIL_UMI_ROOT"
source .venv-policy/bin/activate
export PI05_BASE_CKPT="$HIL_UMI_ROOT/checkpoints/pi05/base"
bash scripts/pi05_acbc.sh dataset/fold_towel_acbc/base fold_towel_acbc 0-1
```

## Collect a round and update iteratively

Copy `collection/config/recorder.example.yaml` to the ignored
`collection/config/recorder.local.yaml`, then configure the devices and output
path in that local file. Set the policy checkpoint, normalization statistics,
task prompt, and camera mappings in `collection/config/ood.yaml`; configure the
estimator in `collection/config/advantage_ood.yaml`. Replace all hardware
placeholders. The recorder requires the OOD compatibility patch. If it has not
been applied, generate a patched copy of your installer (the recorder environment
needs the `wheel` package):

```bash
RECORDER_PYTHON=/path/to/external-recorder-env/bin/python \
  bash "$HIL_UMI_ROOT/collection/scripts/patch_recorder_for_ood.sh" \
  /path/to/data_recorder-0.1.21-py3-none-any.whl
```

Output goes to `collection/.local/patched-wheels/`; install it manually into the
recorder environment. The input is not overwritten.

```bash
cd "$HIL_UMI_ROOT/collection"
export RECORDER_PYTHON=/path/to/external-recorder-env/bin/python
export RECORD_BIN=/path/to/external-recorder-env/bin/record
export POLICY_PYTHON="$HIL_UMI_ROOT/.venv-policy/bin/python"
export ADVANTAGE_PYTHON="$HIL_UMI_ROOT/.venv-advantage/bin/python"
unset OOD_PYTHON
bash run_ood_teleop.sh --mode pi05 config/recorder.local.yaml

# For the separate Advantage collection loop
bash run_ood_teleop.sh --mode advantage config/recorder.local.yaml
```

The two commands correspond to the two collection modes. Add `--check` before
launching to validate configuration and dependencies without opening devices or
loading model weights. The dashboard defaults to `http://127.0.0.1:8765`;
A starts, B saves, X discards, and Y quits. Monitoring provides feedback without
executing predictions on the robot or automatically annotating training parquet.

After each collection round:

1. Validate complete recordings and organize them under
   `advantage/data/fold_towel/stage1/origin/`.
2. Annotate that working dataset, initialize from the previous estimator
   checkpoint, and update Advantage using the base data and this round's data.
3. Relabel the original base demonstrations with the updated estimator, and
   prepare this round's policy data using the existing annotation entry points.
4. Start `scripts/pi05_mix_acbc.sh` from the previous policy checkpoint, set the mixture
   ratio (0.5/0.5 in the example), recompute statistics using the same data and
   ratio, then convert and evaluate the new policy.

### Collection interface

| Data | Interface contract |
| --- | --- |
| State | Unix datagram JSON containing `message_type=state`, `sys_ts`, 4×4 `left_pose`/`right_pose` transforms, gripper feedback/commands, and recording state. |
| Lifecycle | `message_type=lifecycle`, with `timestamp`, `state`, `episode`, and `run_root`, forwards recording, saving, and discard events; replay uses the save-completion event. |
| Images | `dr_<camera_name>_rgb` shared memory with RGB `uint8[H,W,3]`; camera names, dimensions, frame rate, and `ring_slots` must match the recorder configuration. |

Poses use the recorder's tracking frame. Advantage's `frame_offset` must be
smaller than `ring_slots`. The launcher locates the shared-memory module through
`RECORDER_PYTHON`; set `COLLECTION_RECORDER_RING_MODULE` for a custom location.

## SFT baseline

The entry point for ordinary SFT on raw UMI demonstrations is:

```bash
cd "$HIL_UMI_ROOT"
source .venv-policy/bin/activate
bash scripts/pi05_sft.sh dataset/fold_towel_umi fold_towel \
  "fold the towel on the table and put it into the basket" 0-1
```

This script converts the data, computes statistics, and trains for 20,000 steps
by default. Append options such as `--max_steps` or `--lr` to adjust training.
Policy training scripts live in `scripts/`: `pi05_sft.sh` runs ordinary SFT,
`pi05_acbc.sh` runs Advantage-conditioned training, `pi05_mix_acbc.sh` runs
iterative mixed training, and `pi05_post-train.sh` runs ordinary mixed post-training.
They share `scripts/run_vla_sft.sh`. Typical runs only need data, task, GPU and
other run-specific inputs; extra arguments override the default training settings.

## Convert and deploy on Franka

The `full_weights.pt` file in a training state cannot be used directly as a
deployment model. Convert its containing `global_step_<N>` directory, using the
statistics from **the same training run**. Fill in the device variables before
running; ordinary SFT policies should use the task prompt without a conditioning
suffix.

The default policy interface returns 20-frame action chunks with 10 values per
frame: XYZ, 6D rotation, and gripper. Deployment converts each frame's action into
an absolute Franka endpoint pose target and a separate gripper command. The ROS
path publishes `PoseStamped` targets to
`/cartesian_impedance_controller/equilibrium_pose`; the external Cartesian
impedance controller handles low-level robot control.

```bash
cd "$HIL_UMI_ROOT"
source .venv-policy/bin/activate
export POLICY_CKPT=/path/to/global_step_5000
export POLICY_STATS=/path/to/matching/norm_stats.json
export DEPLOY_MODEL="$HIL_UMI_ROOT/checkpoints/pi05/fold_towel_deploy"
python -m training.utils.ckpt_convertor.openpi.convert --mode sft2new \
  --ckpt "$POLICY_CKPT" --input-norm-stats "$POLICY_STATS" \
  --output-model "$DEPLOY_MODEL" \
  --output-norm-stats "$DEPLOY_MODEL/norm_stats/norm_stats.json"

export POLICY_PYTHON="$HIL_UMI_ROOT/.venv-policy/bin/python"
export FRANKA_PYTHON="$HIL_UMI_ROOT/.venv-franka/bin/python"
export ROS_NOETIC_SETUP=/opt/ros/noetic/setup.bash
export FRANKA_CATKIN_SETUP=/path/to/franka_ws/devel/setup.bash
bash evaluations/run_umi_left_franka_eval.sh \
  --model_path "$DEPLOY_MODEL" \
  --norm_stats_path "$DEPLOY_MODEL/norm_stats" \
  --robot_ip "$FRANKA_ROBOT_IP" \
  --wrist_camera "$FRANKA_WRIST_CAMERA_SERIAL" \
  --front_camera "$FRANKA_FRONT_CAMERA_SERIAL" \
  --keyboard_device "$TRAINING_KEYBOARD_DEVICE" \
  "env.eval.override_cfg.task_description='fold the towel on the table and put it into the basket, Advantage: positive'"
```

Append `--dry-run` to the deployment wrapper to preview the command without ROS,
models, or devices, and without creating logs. It only checks path arguments and
command structure; it does not run inference. `evaluations/run_eval.sh` accepts
only the retained `realworld` single-Franka configuration.

`POLICY_CKPT`, `POLICY_STATS`, and the device variables must be supplied by the
user. Statistics must be recomputed when the data mixture ratio changes.
Deployment weights retain the original model's tensor format; direct recovery
of old complete training states after the package rename is not guaranteed.


## License

See [LICENSE](LICENSE), the [Advantage license](advantage/LICENSE), and
[Gemma terms](advantage/LICENSE_GEMMA.txt).
Data and model licenses are separate from the code license.

<a id="citation"></a>

## Citation

Zimu Han, etc. *HIL-UMI: Bringing Human-in-the-Loop Post-Training of Vision-Language-Action Models to Universal Manipulation Interface*. arXiv:2609.20659, 2026.

If you use HIL-UMI, please cite the paper using the BibTeX entry below.

```bibtex
@article{han2026hilumi,
  title={HIL-UMI: Bringing Human-in-the-Loop Post-Training of Vision-Language-Action Models to Universal Manipulation Interface},
  author={Han, Zimu and Zeng, Yiming and Zhang, Jiyao and Zhao, Zihao and Wang, Yuanfei and Jin, Yixiang and Li, Shiqi and Chen, Shuangben and Huang, Wei and Li, Ruodai and Shen, Hui and Dong, Hao},
  journal={arXiv preprint arXiv:2609.20659},
  year={2026}
}
```
