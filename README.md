# H2F


## 🛠️ Requirements and Installation

Basic dependencies:

* Python 3.9 (recommended)
* PyTorch >= 1.10.0
* torchvision
* transformers >= 4.28.0
* CUDA-compatible GPU for distributed training and evaluation

**[Online Mode]** Install the dependencies in a Conda environment (recommended for development):

```bash
conda create -n h2f python=3.9 -y
conda activate h2f

# Install PyTorch for your CUDA version. The following example uses CUDA 11.8.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**[Package Mode]** Install h2f as an editable Python package (recommended for direct use):

```bash
conda activate h2f
pip install --upgrade pip
pip install -e .
```

On Apple Silicon, replace `decord` with `eva-decord` in `requirements.txt` before installation. The provided training scripts use CUDA and `torchrun`; Apple Silicon is mainly suitable for code inspection and limited local experiments.

## 📊 Datasets

The current project configurations cover four video question answering benchmarks:

| Dataset | Task | Annotation / Homepage | Visual Data | Project Config |
|---|---|---|---|---|
| MSVD-QA | Open-ended VideoQA | [MSVD-QA annotations](https://github.com/boheumd/MA-LMM/tree/main/data) | Extracted frames | `lavis/projects/malmm/qa_msvd.yaml` |
| ActivityNet-QA | Open-ended VideoQA | [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) | Extracted frames | `lavis/projects/malmm/qa_activitynet.yaml` |
| NExT-QA | Multiple-choice VideoQA | [NExT-QA](https://github.com/doc-doc/NExT-QA) | Videos | `lavis/projects/malmm/qa_nextqa.yaml` |
| MVBench | Multi-task VideoQA | [MVBench](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2/mvbench) | Videos | `lavis/projects/malmm/qa_mvbench.yaml` |

Extract frames for frame-based datasets at 10 FPS. An example preprocessing script is available in the [MA-LMM data tools](https://github.com/boheumd/MA-LMM/blob/main/data/extract_frames.py). Different FFmpeg versions may produce slightly different frame counts, so update the frame lengths in the annotations when necessary.

The dataset root defaults to the LAVIS cache directory configured by `cache_root` in `lavis/configs/default.yaml`. Arrange the data as follows:

```text
data/
├── msvd/
│   ├── annotation/
│   │   ├── qa_train.json
│   │   ├── qa_val.json
│   │   ├── qa_test.json
│   │   └── qa_ans2label.json
│   └── frames/
├── activitynet/
│   ├── annotation/
│   │   ├── qa_train.json
│   │   ├── qa_val.json
│   │   └── qa_test.json
│   └── frames/
├── nextqa/
│   ├── annotations/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── videos/
└── mvbench/
    ├── annotations/
    │   └── mvbench.json
    └── videos/
```

If your data is stored elsewhere, update the corresponding `storage` entries under `lavis/configs/datasets/`.

## 🗝️ Training & Evaluation

**[Pre-trained Models]**

The model is initialized from [InstructBLIP Vicuna-7B](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth) and uses Vicuna-7B as its language model. Download the Vicuna weights by following the [FastChat instructions](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md), then arrange them as:

```text
llm/
└── vicuna-7b/
```

The default project configurations expect the language model at `llm/vicuna-7b`.

**[Training]**

The provided scripts launch distributed training on four GPUs:

```bash
# MSVD-QA
bash run_scripts/msvd/train_qa.sh

# ActivityNet-QA
bash run_scripts/activitynet/train.sh

# NExT-QA
bash run_scripts/nextqa/train.sh
```

MVBench currently provides an evaluation script only.

**[Testing]**

Pass the fine-tuned checkpoint path to the matching evaluation script:

```bash
bash run_scripts/msvd/test_qa.sh /path/to/checkpoint.pth
bash run_scripts/activitynet/test.sh /path/to/checkpoint.pth
bash run_scripts/nextqa/test.sh /path/to/checkpoint.pth
bash run_scripts/mvbench/test.sh /path/to/checkpoint.pth
```

The scripts use `torchrun --nproc_per_node=4`. Adjust this value and the batch size in the scripts to match your hardware.

## 🧠 Model Components

### Question-Conditioned Video MoD

`QuestionConditionedMoD` routes temporal visual tokens according to the question. Given `video_feats [B,T,D]` and `question_tokens [B,Nq,D]`, it produces:

```text
mod_video_feats [B,T,D]
evidence_prior [B,T]
depth_logits [B,T]
depth_mask [B,T]
```

Set the token keep ratio ρ with `model.mod_keep_ratio` (default: `0.5`).

### Diffusion Semantic Fusion (DSF)

DSF refines question-conditioned temporal evidence before pooling it into the visual token supplied to the language model. The reusable components are available from:

```python
from lavis.models.blip2_models.qmod_dsf import (
    QuestionConditionedMoD,
    DiffusionSemanticFusion,
    DSFConditionalDenoiser,
    DSFRefiner,
    VideoEvidenceAdapter,
)
```

The effective configuration options are `model.mod_keep_ratio`, `model.mod_expansion`, `model.mod_dropout`, `model.lambda_mod`, `model.lambda_dsf`, `model.dsf_steps`, `model.dsf_init_noise`, and `model.dsf_target_temp`. To override the MoD keep ratio at runtime:

```bash
torchrun --nproc_per_node=4 train.py \
  --cfg-path lavis/projects/diffusion_mod/qa_msvd.yaml \
  --options model.mod_keep_ratio 0.5
```

Run the Q-MoD/DSF unit tests with:

```bash
python -m pytest tests/models/test_qmod_dsf.py -q
```

## 📝 Citation


## 🙏 Acknowledgement

This codebase builds upon [MA-LMM](https://github.com/boheumd/MA-LMM) and [LAVIS](https://github.com/salesforce/LAVIS).

