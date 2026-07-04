#!/usr/bin/env bash
set -euo pipefail

checkpoint_path=${1:?"Usage: bash run_scripts/mvbench/test.sh CHECKPOINT_PATH"}
torchrun --nproc_per_node=4 \
    --master_port=34652 \
    train.py \
    --cfg-path lavis/projects/malmm/qa_mvbench.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 10 \
    model.num_frames 16 \
    model.max_num_frames 120 \
    model.temporal_layers 2 \
    model.moe_num_experts 4 \
    model.moe_top_k 2 \
    model.moe_expansion 2 \
    model.moe_dropout 0.1 \
    model.lambda_moe 0.01 \
    model.lambda_diff 1.0 \
    model.evidence_diffusion_steps 20 \
    model.evidence_init_noise 0.1 \
    model.evidence_target_temp 0.07 \
    run.evaluate True \
    run.train_splits "[]" \
    run.valid_splits "[]" \
    run.test_splits "['test']" \
    run.prefix moe_diffusion_test \
    run.resume_ckpt_path "${checkpoint_path}"
