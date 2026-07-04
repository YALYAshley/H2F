import importlib.util
from pathlib import Path

import torch


def load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "lavis"
        / "models"
        / "blip2_models"
        / "qmod_dsf.py"
    )
    spec = importlib.util.spec_from_file_location("qmod_dsf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qmod_shapes_capacity_identity_temporal_order_and_padding():
    module = load_module()
    torch.manual_seed(0)
    batch_size, num_visual, num_question, hidden_size = 2, 6, 4, 8
    visual_features = torch.randn(batch_size, num_visual, hidden_size)
    question_features = torch.randn(batch_size, num_question, hidden_size)
    question_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    visual_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool
    )
    qmod = module.QuestionConditionedMoD(
        hidden_size=hidden_size,
        num_routing_blocks=2,
        routing_dim=4,
        router_hidden_dim=12,
        capacity_ratios=[0.5, 0.5],
        attention_heads=2,
        dropout=0.0,
    )

    output = qmod(
        visual_features,
        question_features,
        question_mask=question_mask,
        visual_mask=visual_mask,
    )

    assert output["final_visual_state"].shape == (batch_size, num_visual, hidden_size)
    assert len(output["visual_states"]) == 3
    assert output["affinities"][0].shape == (batch_size, num_visual, num_question)
    assert output["delta_selected"][0].shape == (batch_size, 3, hidden_size)
    assert output["effective_depth"].shape == (batch_size, num_visual)

    for block_idx, routing_mask in enumerate(output["routing_masks"]):
        assert routing_mask.shape == (batch_size, num_visual)
        assert torch.all(routing_mask.sum(dim=1) == 3)
        assert torch.all(routing_mask[~visual_mask] == 0)
        assert torch.all(
            output["selected_indices"][block_idx]
            == output["selected_indices"][block_idx].sort(dim=1).values
        )
        unchanged = routing_mask == 0
        assert torch.allclose(
            output["visual_states"][block_idx][unchanged],
            output["visual_states"][block_idx + 1][unchanged],
        )

    assert torch.all(output["affinities"][0][:, :, ~question_mask[0]].sum() >= 0)
    assert torch.allclose(
        output["affinities"][0][0, :, 3],
        torch.zeros(num_visual),
        atol=1e-6,
    )


def test_dsf_shapes_and_gate_monotonicity():
    module = load_module()
    torch.manual_seed(1)
    batch_size, num_visual, num_question = 3, 5, 4
    hidden_size, answer_dim, num_steps = 8, 7, 6
    denoiser = module.DSFConditionalDenoiser(
        answer_dim=answer_dim,
        hidden_size=hidden_size,
        num_steps=num_steps,
        attention_dim=4,
        dropout=0.0,
    )
    x_t = torch.randn(batch_size, answer_dim)
    timesteps = torch.tensor([5, 3, 0], dtype=torch.long)
    question_features = torch.randn(batch_size, num_question, hidden_size)
    U_g = torch.randn(batch_size, num_visual, hidden_size)
    U_p = torch.randn(batch_size, num_visual, hidden_size)
    question_mask = torch.ones(batch_size, num_question, dtype=torch.bool)
    visual_mask = torch.ones(batch_size, num_visual, dtype=torch.bool)

    output = denoiser(
        x_t,
        timesteps,
        question_features,
        U_g,
        U_p,
        question_mask=question_mask,
        visual_mask=visual_mask,
    )
    assert output["answer_logits"].shape == (batch_size, answer_dim)
    assert output["answer_prob"].shape == (batch_size, answer_dim)
    assert output["depth_gate"].shape == (batch_size,)
    assert output["global_context"].shape == (batch_size, hidden_size)
    assert output["deep_context"].shape == (batch_size, hidden_size)

    reverse_timesteps = torch.arange(num_steps - 1, -1, -1)
    gates = denoiser._depth_gate(reverse_timesteps)
    assert torch.all(gates[1:] + 1e-6 >= gates[:-1])


def test_qmod_dsf_gradients_and_qmod_reuse():
    module = load_module()
    torch.manual_seed(2)
    batch_size, num_visual, num_question = 2, 5, 3
    hidden_size, answer_dim = 8, 4
    refiner = module.DSFRefiner(
        answer_dim=answer_dim,
        hidden_size=hidden_size,
        num_steps=4,
        qmod_num_routing_blocks=2,
        qmod_routing_dim=4,
        qmod_router_hidden_dim=10,
        qmod_capacity_ratios=[0.6, 0.4],
        qmod_attention_heads=2,
        dsf_attention_dim=4,
        dropout=0.0,
    )
    visual_features = torch.randn(batch_size, num_visual, hidden_size)
    question_features = torch.randn(batch_size, num_question, hidden_size)
    question_mask = torch.ones(batch_size, num_question, dtype=torch.bool)
    visual_mask = torch.ones(batch_size, num_visual, dtype=torch.bool)
    answer_target = torch.tensor([1, 3], dtype=torch.long)

    loss, _ = refiner.training_losses(
        answer_target,
        visual_features,
        question_features,
        question_mask=question_mask,
        visual_mask=visual_mask,
    )
    loss.backward()

    router_grad = refiner.qmod.routing_blocks[0].router_fc2.weight.grad
    attention_grad = refiner.qmod.routing_blocks[0].visual_proj.weight.grad
    denoiser_grad = refiner.denoiser.denoising_mlp[-1].weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0
    assert attention_grad is not None and attention_grad.abs().sum() > 0
    assert denoiser_grad is not None and denoiser_grad.abs().sum() > 0

    refiner.qmod_forward_count = 0
    refiner.sample(
        visual_features,
        question_features,
        question_mask=question_mask,
        visual_mask=visual_mask,
    )
    assert refiner.qmod_forward_count == 1


def test_video_evidence_adapter_uses_dsf_outputs():
    module = load_module()
    torch.manual_seed(3)
    adapter = module.VideoEvidenceAdapter(
        hidden_size=8,
        max_num_frames=6,
        temporal_layers=1,
        mod_keep_ratio=0.5,
        mod_expansion=2,
        dropout=0.0,
        dsf_steps=3,
        dsf_init_noise_scale=0.0,
        dsf_target_temperature=0.1,
    )
    frame_tokens = torch.randn(2, 4, 2, 8)
    question_tokens = torch.randn(2, 3, 8)
    question_mask = torch.ones(2, 3, dtype=torch.bool)

    output = adapter(
        frame_tokens,
        question_tokens,
        question_mask,
        training=True,
    )

    assert output["llm_visual_tokens"].shape == (2, 1, 8)
    assert output["refined_evidence"].shape == (2, 4)
    assert output["mod_aux_loss"].ndim == 0
    assert output["dsf_loss"].ndim == 0
