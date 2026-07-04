#!/usr/bin/env bash
set -euo pipefail

checkpoint_path=${1:?"Usage: bash run_scripts/activitynet/test.sh CHECKPOINT_PATH"}
torchrun --nproc_per_node=4 \
    --master_port=34652 \
    train.py \
    --cfg-path lavis/projects/malmm/qa_activitynet.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 10 \
    model.num_frames 20 \
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
    run.init_lr 1e-4 \
    run.max_epoch 5 \
    run.num_beams 5 \
    run.batch_size_train 32 \
    run.batch_size_eval 32 \
    run.accum_grad_iters 1 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate True \
    run.valid_splits "['val', 'test']" \
    run.report_metric True \
    run.prefix moe_diffusion_test \
    run.resume_ckpt_path "${checkpoint_path}"
