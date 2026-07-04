"""Question-conditioned Mixture-of-Depths and diffusion semantic fusion."""

import torch
import torch.nn.functional as F
from torch import nn


def masked_mean(values, mask=None, dim=1):
    if mask is None:
        return values.mean(dim=dim)
    weights = mask.to(device=values.device, dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    summed = (values * weights).sum(dim=dim)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return summed / count


def sinusoidal_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    exponent = -torch.log(
        timesteps.new_tensor(10000.0)
    ) * torch.arange(half_dim, device=timesteps.device) / max(half_dim - 1, 1)
    frequencies = torch.exp(exponent)
    args = timesteps.to(frequencies.dtype).unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def batched_gather(values, indices):
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
    return values.gather(1, gather_indices)


def batched_scatter(base, indices, updates):
    scatter_indices = indices.unsqueeze(-1).expand_as(updates)
    return base.scatter(1, scatter_indices, updates)


class QuestionConditionedRoutingBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        routing_dim,
        router_hidden_dim,
        capacity_ratio,
        attention_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        if not 0 < capacity_ratio <= 1:
            raise ValueError("capacity_ratio must be in (0, 1].")
        self.capacity_ratio = capacity_ratio
        self.visual_proj = nn.Linear(hidden_size, routing_dim)
        self.question_proj = nn.Linear(hidden_size, routing_dim)
        self.context_proj = nn.Linear(hidden_size, hidden_size)
        self.visual_ln = nn.LayerNorm(hidden_size)
        self.question_ln = nn.LayerNorm(hidden_size)
        self.router_fc1 = nn.Linear(hidden_size * 4, router_hidden_dim)
        self.router_fc2 = nn.Linear(router_hidden_dim, 1)
        self.ln_query = nn.LayerNorm(hidden_size)
        self.ln_memory = nn.LayerNorm(hidden_size)
        num_heads = attention_heads if hidden_size % attention_heads == 0 else 1
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, visual_state, question_features, question_mask=None, visual_mask=None):
        batch_size, num_visual_tokens, hidden_size = visual_state.shape
        capacity = int(torch.ceil(visual_state.new_tensor(
            self.capacity_ratio * num_visual_tokens
        )).item())
        capacity = max(1, min(capacity, num_visual_tokens))

        v_proj = self.visual_proj(visual_state)
        q_proj = self.question_proj(question_features)
        affinity_logits = torch.matmul(v_proj, q_proj.transpose(-1, -2))
        affinity_logits = affinity_logits / (q_proj.size(-1) ** 0.5)
        if question_mask is not None:
            question_mask = question_mask.to(
                device=question_features.device, dtype=torch.bool
            )
            affinity_logits = affinity_logits.masked_fill(
                ~question_mask.unsqueeze(1), torch.finfo(affinity_logits.dtype).min
            )
        affinity = torch.softmax(affinity_logits, dim=-1)
        question_context = self.context_proj(torch.matmul(affinity, question_features))

        h_norm = self.visual_ln(visual_state)
        c_norm = self.question_ln(question_context)
        router_input = torch.cat(
            [h_norm, c_norm, h_norm * c_norm, torch.abs(h_norm - c_norm)], dim=-1
        )
        routing_logits = self.router_fc2(
            F.gelu(self.router_fc1(router_input))
        ).squeeze(-1)
        if visual_mask is not None:
            visual_mask = visual_mask.to(device=visual_state.device, dtype=torch.bool)
            routing_logits = routing_logits.masked_fill(
                ~visual_mask, torch.finfo(routing_logits.dtype).min
            )
        routing_gates = torch.sigmoid(routing_logits)

        _, topk_indices = torch.topk(routing_logits, k=capacity, dim=1)
        selected_indices = topk_indices.sort(dim=1).values
        if visual_mask is not None:
            selected_valid = visual_mask.gather(1, selected_indices)
            first_valid = visual_mask.to(torch.long).argmax(dim=1, keepdim=True)
            selected_indices = torch.where(
                selected_valid, selected_indices, first_valid.expand_as(selected_indices)
            )
            selected_indices = selected_indices.sort(dim=1).values

        routing_masks = visual_state.new_zeros(batch_size, num_visual_tokens)
        routing_masks.scatter_(1, selected_indices, 1.0)
        if visual_mask is not None:
            routing_masks = routing_masks * visual_mask.to(routing_masks.dtype)

        h_selected = batched_gather(visual_state, selected_indices)
        c_selected = batched_gather(question_context, selected_indices)
        g_selected = routing_gates.gather(1, selected_indices)
        u_selected = h_selected + c_selected

        memory = self.ln_memory(visual_state)
        r_selected, _ = self.cross_attention(
            query=self.ln_query(u_selected),
            key=memory,
            value=memory,
            key_padding_mask=None if visual_mask is None else ~visual_mask,
            need_weights=False,
        )
        delta_selected = r_selected + self.ffn(self.ln_ffn(u_selected + r_selected))
        updated_selected = h_selected + g_selected.unsqueeze(-1) * delta_selected
        next_state = batched_scatter(visual_state.clone(), selected_indices, updated_selected)

        return {
            "next_state": next_state,
            "question_context": question_context,
            "affinity": affinity,
            "routing_logits": routing_logits,
            "routing_gates": routing_gates,
            "routing_mask": routing_masks,
            "selected_indices": selected_indices,
            "delta_selected": delta_selected,
        }


class QuestionConditionedMoD(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_routing_blocks=2,
        routing_dim=None,
        router_hidden_dim=None,
        capacity_ratios=None,
        attention_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        if num_routing_blocks < 1:
            raise ValueError("num_routing_blocks must be at least 1.")
        routing_dim = routing_dim or hidden_size
        router_hidden_dim = router_hidden_dim or hidden_size
        if capacity_ratios is None:
            capacity_ratios = [0.5] * num_routing_blocks
        if len(capacity_ratios) != num_routing_blocks:
            raise ValueError("capacity_ratios length must match num_routing_blocks.")
        self.routing_blocks = nn.ModuleList(
            [
                QuestionConditionedRoutingBlock(
                    hidden_size=hidden_size,
                    routing_dim=routing_dim,
                    router_hidden_dim=router_hidden_dim,
                    capacity_ratio=capacity_ratios[idx],
                    attention_heads=attention_heads,
                    dropout=dropout,
                )
                for idx in range(num_routing_blocks)
            ]
        )

    def forward(
        self,
        visual_features,
        question_features,
        question_mask=None,
        visual_mask=None,
    ):
        if visual_features.ndim != 3:
            raise ValueError("visual_features must have shape [B, Nv, D].")
        if question_features.ndim != 3:
            raise ValueError("question_features must have shape [B, Nq, D].")
        if (
            visual_features.size(0) != question_features.size(0)
            or visual_features.size(2) != question_features.size(2)
        ):
            raise ValueError(
                "visual_features and question_features must share batch and hidden sizes."
            )
        if visual_mask is not None:
            visual_mask = visual_mask.to(device=visual_features.device, dtype=torch.bool)
            if torch.any(visual_mask.sum(dim=1) == 0):
                raise ValueError("Each sample must contain at least one valid visual token.")
        if question_mask is not None:
            question_mask = question_mask.to(
                device=question_features.device, dtype=torch.bool
            )
            if torch.any(question_mask.sum(dim=1) == 0):
                raise ValueError("Each question must contain at least one valid token.")

        visual_states = [visual_features]
        routing_logits_list = []
        routing_gates_list = []
        routing_masks_list = []
        selected_indices_list = []
        affinity_list = []
        delta_selected_list = []

        state = visual_features
        for routing_block in self.routing_blocks:
            block_output = routing_block(
                state,
                question_features,
                question_mask=question_mask,
                visual_mask=visual_mask,
            )
            state = block_output["next_state"]
            visual_states.append(state)
            routing_logits_list.append(block_output["routing_logits"])
            routing_gates_list.append(block_output["routing_gates"])
            routing_masks_list.append(block_output["routing_mask"])
            selected_indices_list.append(block_output["selected_indices"])
            affinity_list.append(block_output["affinity"])
            delta_selected_list.append(block_output["delta_selected"])

        effective_depth = torch.stack(routing_masks_list, dim=0).sum(dim=0)

        return {
            "final_visual_state": visual_states[-1],
            "visual_states": visual_states,
            "routing_logits": routing_logits_list,
            "routing_gates": routing_gates_list,
            "routing_masks": routing_masks_list,
            "effective_depth": effective_depth,
            "selected_indices": selected_indices_list,
            "affinities": affinity_list,
            "delta_selected": delta_selected_list,
        }


class EvidenceDenoiser(nn.Module):
    def __init__(self, hidden_size, num_steps, dropout=0.1):
        super().__init__()
        input_size = hidden_size * 3 + 4
        num_heads = 8 if hidden_size % 8 == 0 else 1
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.question_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.timestep_embedding = nn.Embedding(num_steps, hidden_size)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        m_k,
        timesteps,
        evidence_prior,
        depth_logits,
        depth_mask,
        mod_video_feats,
        question_tokens,
        video_mask=None,
        question_mask=None,
    ):
        if question_mask is None:
            question_mask = question_tokens.new_ones(
                question_tokens.shape[:2], dtype=torch.long
            )
        question_weights = question_mask.to(question_tokens.dtype)
        question_feat = torch.einsum(
            "bl,bld->bd", question_weights, question_tokens
        )
        question_feat = question_feat / question_weights.sum(
            1, keepdim=True
        ).clamp_min(1.0)
        question_frames = question_feat.unsqueeze(1).expand_as(mod_video_feats)
        depth_logits = depth_logits.masked_fill(
            video_mask == 0, 0.0
        ) if video_mask is not None else depth_logits

        hidden = self.input_proj(
            torch.cat(
                [
                    m_k.unsqueeze(-1),
                    evidence_prior.unsqueeze(-1),
                    depth_logits.unsqueeze(-1),
                    depth_mask.unsqueeze(-1),
                    mod_video_feats,
                    question_frames,
                    mod_video_feats * question_frames,
                ],
                dim=-1,
            )
        )
        question_context, _ = self.question_attention(
            hidden,
            question_tokens,
            question_tokens,
            key_padding_mask=question_mask == 0,
            need_weights=False,
        )
        hidden = (
            hidden
            + question_context
            + self.timestep_embedding(timesteps).unsqueeze(1)
        )
        logits = self.output_head(hidden).squeeze(-1)
        if video_mask is not None:
            logits = logits.masked_fill(video_mask == 0, -1e4)
        return torch.softmax(logits, dim=-1)


class DiffusionSemanticFusion(nn.Module):
    """Refine and fuse question-conditioned temporal evidence."""
    def __init__(
        self,
        hidden_size,
        num_steps=20,
        dropout=0.1,
        init_noise_scale=0.1,
    ):
        super().__init__()
        if num_steps < 2:
            raise ValueError("dsf_steps must be at least 2.")
        self.num_steps = num_steps
        self.init_noise_scale = init_noise_scale
        self.denoiser = EvidenceDenoiser(hidden_size, num_steps, dropout)
        betas = torch.linspace(1e-4, 2e-2, num_steps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_ab", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_omab", (1.0 - alphas_cumprod).sqrt())

    @staticmethod
    def _extract(values, timesteps, target):
        return values.gather(0, timesteps).view(-1, 1).to(target.dtype)

    @staticmethod
    def _distribution(logits, video_mask):
        if video_mask is not None:
            logits = logits.masked_fill(
                video_mask == 0, torch.finfo(logits.dtype).min
            )
        return torch.softmax(logits, dim=-1)

    def training_losses(
        self,
        evidence_target,
        evidence_prior,
        depth_logits,
        depth_mask,
        mod_video_feats,
        question_tokens,
        video_mask=None,
        question_mask=None,
    ):
        if evidence_target.shape != evidence_prior.shape:
            raise ValueError(
                "evidence_target and evidence_prior must both have shape [B, T]."
            )
        batch_size = evidence_target.size(0)
        timesteps = torch.randint(
            self.num_steps, (batch_size,), device=evidence_target.device
        )
        s0 = torch.log(evidence_target.clamp_min(1e-8))
        noise = torch.randn_like(s0)
        s_k = (
            self._extract(self.sqrt_ab, timesteps, s0) * s0
            + self._extract(self.sqrt_omab, timesteps, s0) * noise
        )
        m_k = self._distribution(s_k, video_mask)
        m0_pred = self.denoiser(
            m_k,
            timesteps,
            evidence_prior,
            depth_logits,
            depth_mask,
            mod_video_feats,
            question_tokens,
            video_mask,
            question_mask,
        )
        dsf_loss = (
            -evidence_target * torch.log(m0_pred.clamp_min(1e-8))
        ).sum(-1).mean()
        return dsf_loss, m0_pred

    @torch.no_grad()
    def sample(
        self,
        evidence_prior,
        depth_logits,
        depth_mask,
        mod_video_feats,
        question_tokens,
        video_mask=None,
        question_mask=None,
    ):
        # The diffusion state is a temporal logit, so initialize around the
        # learned evidence distribution instead of pure Gaussian noise.
        s_k = torch.log(evidence_prior.clamp_min(1e-8))
        s_k = s_k + self.init_noise_scale * torch.randn_like(s_k)
        for step in reversed(range(self.num_steps)):
            timestep = torch.full(
                (s_k.size(0),), step, dtype=torch.long, device=s_k.device
            )
            m0_pred = self.denoiser(
                self._distribution(s_k, video_mask),
                timestep,
                evidence_prior,
                depth_logits,
                depth_mask,
                mod_video_feats,
                question_tokens,
                video_mask,
                question_mask,
            )
            if step == 0:
                return m0_pred
            s0_pred = torch.log(m0_pred.clamp_min(1e-8))
            noise_pred = (
                s_k - self.sqrt_ab[step] * s0_pred
            ) / self.sqrt_omab[step].clamp_min(1e-8)
            s_k = (
                self.sqrt_ab[step - 1] * s0_pred
                + self.sqrt_omab[step - 1] * noise_pred
            )
        return self._distribution(s_k, video_mask)


class DSFConditionalDenoiser(nn.Module):
    """Denoise an answer state using shallow and deep Q-MoD features."""
    def __init__(
        self,
        answer_dim,
        hidden_size,
        num_steps,
        attention_dim=None,
        dropout=0.1,
        global_only=False,
        deep_only=False,
        static_fusion=False,
        reversed_gate=False,
        shared_condition_attention=False,
    ):
        super().__init__()
        if answer_dim < 2:
            raise ValueError("answer_dim must be at least 2.")
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2.")
        attention_dim = attention_dim or hidden_size
        self.answer_dim = answer_dim
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.global_only = global_only
        self.deep_only = deep_only
        self.static_fusion = static_fusion
        self.reversed_gate = reversed_gate
        self.shared_condition_attention = shared_condition_attention

        self.answer_projection = nn.Linear(answer_dim, hidden_size)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.query_projection = nn.Linear(hidden_size * 3, hidden_size)
        self.query_ln = nn.LayerNorm(hidden_size)
        self.global_query = nn.Linear(hidden_size, attention_dim)
        self.global_key = nn.Linear(hidden_size, attention_dim)
        self.global_value = nn.Linear(hidden_size, hidden_size)
        if shared_condition_attention:
            self.deep_query = self.global_query
            self.deep_key = self.global_key
            self.deep_value = self.global_value
        else:
            self.deep_query = nn.Linear(hidden_size, attention_dim)
            self.deep_key = nn.Linear(hidden_size, attention_dim)
            self.deep_value = nn.Linear(hidden_size, hidden_size)
        self.kappa_raw = nn.Parameter(torch.zeros(()))
        self.delta_raw = nn.Parameter(torch.zeros(()))
        self.static_lambda_raw = nn.Parameter(torch.zeros(()))
        self.denoising_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, answer_dim),
        )

    @staticmethod
    def _condition_attention(query, keys, values, visual_mask=None):
        scores = torch.einsum("bd,bnd->bn", query, keys)
        scores = scores / (keys.size(-1) ** 0.5)
        if visual_mask is not None:
            visual_mask = visual_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~visual_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bn,bnd->bd", weights, values)
        return context, weights

    def _depth_gate(self, timesteps):
        if self.global_only:
            return timesteps.new_zeros(timesteps.shape, dtype=torch.float32)
        if self.deep_only:
            return timesteps.new_ones(timesteps.shape, dtype=torch.float32)
        if self.static_fusion:
            return torch.sigmoid(self.static_lambda_raw).expand_as(
                timesteps.to(torch.float32)
            )

        eta = 1.0 - timesteps.to(torch.float32) / float(self.num_steps - 1)
        if self.reversed_gate:
            eta = 1.0 - eta
        kappa = F.softplus(self.kappa_raw) + 1e-6
        delta = torch.sigmoid(self.delta_raw)
        return torch.sigmoid(kappa * (eta - delta))

    def forward(
        self,
        x_t,
        timesteps,
        question_features,
        U_g,
        U_p,
        question_mask=None,
        visual_mask=None,
    ):
        if x_t.ndim != 2 or x_t.size(-1) != self.answer_dim:
            raise ValueError("x_t must have shape [B, Na].")
        g_t = self.answer_projection(x_t)
        tau_t = self.timestep_mlp(
            sinusoidal_timestep_embedding(timesteps, self.hidden_size)
        )
        q = masked_mean(question_features, question_mask, dim=1)
        r_t = self.query_ln(self.query_projection(torch.cat([q, g_t, tau_t], dim=-1)))

        c_g_t, global_weights = self._condition_attention(
            self.global_query(r_t),
            self.global_key(U_g),
            self.global_value(U_g),
            visual_mask=visual_mask,
        )
        c_p_t, deep_weights = self._condition_attention(
            self.deep_query(r_t),
            self.deep_key(U_p),
            self.deep_value(U_p),
            visual_mask=visual_mask,
        )
        lambda_t = self._depth_gate(timesteps).to(dtype=x_t.dtype, device=x_t.device)
        v_t = (1.0 - lambda_t.unsqueeze(-1)) * c_g_t + lambda_t.unsqueeze(-1) * c_p_t
        answer_logits = self.denoising_mlp(torch.cat([g_t, q, tau_t, v_t], dim=-1))

        return {
            "answer_logits": answer_logits,
            "answer_prob": answer_logits.softmax(dim=-1),
            "depth_gate": lambda_t,
            "global_context": c_g_t,
            "deep_context": c_p_t,
            "global_attention": global_weights,
            "deep_attention": deep_weights,
        }


class DSFRefiner(nn.Module):
    """Depth-aware semantic fusion over the Q-MoD visual hierarchy."""
    def __init__(
        self,
        answer_dim,
        hidden_size,
        num_steps=20,
        qmod_num_routing_blocks=2,
        qmod_routing_dim=None,
        qmod_router_hidden_dim=None,
        qmod_capacity_ratios=None,
        qmod_attention_heads=8,
        dsf_attention_dim=None,
        dropout=0.1,
        init_noise_scale=1.0,
        disable_qmod=False,
        global_only=False,
        deep_only=False,
        static_fusion=False,
        reversed_gate=False,
        shared_condition_attention=False,
    ):
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2.")
        self.num_steps = num_steps
        self.init_noise_scale = init_noise_scale
        self.disable_qmod = disable_qmod
        self.qmod_forward_count = 0
        self.qmod = QuestionConditionedMoD(
            hidden_size=hidden_size,
            num_routing_blocks=qmod_num_routing_blocks,
            routing_dim=qmod_routing_dim,
            router_hidden_dim=qmod_router_hidden_dim,
            capacity_ratios=qmod_capacity_ratios,
            attention_heads=qmod_attention_heads,
            dropout=dropout,
        )
        self.denoiser = DSFConditionalDenoiser(
            answer_dim=answer_dim,
            hidden_size=hidden_size,
            num_steps=num_steps,
            attention_dim=dsf_attention_dim,
            dropout=dropout,
            global_only=global_only,
            deep_only=deep_only,
            static_fusion=static_fusion,
            reversed_gate=reversed_gate,
            shared_condition_attention=shared_condition_attention,
        )
        betas = torch.linspace(1e-4, 2e-2, num_steps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_ab", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_omab", (1.0 - alphas_cumprod).sqrt())

    @staticmethod
    def _extract(values, timesteps, target):
        return values.gather(0, timesteps).view(-1, 1).to(target.dtype)

    def _build_conditions(
        self,
        visual_features,
        question_features,
        question_mask=None,
        visual_mask=None,
    ):
        if self.disable_qmod:
            qmod_output = {
                "final_visual_state": visual_features,
                "visual_states": [visual_features],
                "routing_logits": [],
                "routing_gates": [],
                "routing_masks": [],
                "effective_depth": visual_features.new_zeros(visual_features.shape[:2]),
            }
        else:
            self.qmod_forward_count += 1
            qmod_output = self.qmod(
                visual_features,
                question_features,
                question_mask=question_mask,
                visual_mask=visual_mask,
            )
        return qmod_output["visual_states"][0], qmod_output["final_visual_state"], qmod_output

    def training_losses(
        self,
        answer_target,
        visual_features,
        question_features,
        question_mask=None,
        visual_mask=None,
    ):
        if answer_target.ndim == 1:
            x_0 = F.one_hot(answer_target, self.denoiser.answer_dim).to(
                dtype=visual_features.dtype, device=visual_features.device
            )
            ce_target = answer_target.to(device=visual_features.device)
        else:
            x_0 = answer_target.to(dtype=visual_features.dtype, device=visual_features.device)
            ce_target = None
        batch_size = x_0.size(0)
        timesteps = torch.randint(self.num_steps, (batch_size,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            self._extract(self.sqrt_ab, timesteps, x_0) * x_0
            + self._extract(self.sqrt_omab, timesteps, x_0) * noise
        )
        U_g, U_p, qmod_output = self._build_conditions(
            visual_features,
            question_features,
            question_mask=question_mask,
            visual_mask=visual_mask,
        )
        denoiser_output = self.denoiser(
            x_t,
            timesteps,
            question_features,
            U_g,
            U_p,
            question_mask=question_mask,
            visual_mask=visual_mask,
        )
        if ce_target is not None:
            loss = F.cross_entropy(denoiser_output["answer_logits"], ce_target)
        else:
            loss = F.kl_div(
                F.log_softmax(denoiser_output["answer_logits"], dim=-1),
                x_0,
                reduction="batchmean",
            )
        return loss, {**denoiser_output, "qmod_output": qmod_output, "timesteps": timesteps}

    @torch.no_grad()
    def sample(
        self,
        visual_features,
        question_features,
        question_mask=None,
        visual_mask=None,
    ):
        U_g, U_p, qmod_output = self._build_conditions(
            visual_features,
            question_features,
            question_mask=question_mask,
            visual_mask=visual_mask,
        )
        x_t = self.init_noise_scale * torch.randn(
            visual_features.size(0),
            self.denoiser.answer_dim,
            device=visual_features.device,
            dtype=visual_features.dtype,
        )
        denoiser_output = None
        for step in reversed(range(self.num_steps)):
            timesteps = torch.full(
                (x_t.size(0),), step, dtype=torch.long, device=x_t.device
            )
            denoiser_output = self.denoiser(
                x_t,
                timesteps,
                question_features,
                U_g,
                U_p,
                question_mask=question_mask,
                visual_mask=visual_mask,
            )
            x_0_pred = denoiser_output["answer_prob"]
            if step == 0:
                break
            noise_pred = (
                x_t - self.sqrt_ab[step] * x_0_pred
            ) / self.sqrt_omab[step].clamp_min(1e-8)
            x_t = (
                self.sqrt_ab[step - 1] * x_0_pred
                + self.sqrt_omab[step - 1] * noise_pred
            )
        return {**denoiser_output, "qmod_output": qmod_output}


class VideoEvidenceAdapter(nn.Module):
    """Temporal encoder, question-conditioned MoD, and DSF pooling."""

    def __init__(
        self,
        hidden_size,
        max_num_frames=120,
        temporal_layers=2,
        mod_keep_ratio=0.5,
        mod_expansion=2,
        dropout=0.1,
        dsf_steps=20,
        dsf_init_noise_scale=0.1,
        dsf_target_temperature=0.07,
    ):
        super().__init__()
        num_heads = 8 if hidden_size % 8 == 0 else 1
        temporal_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_position = nn.Embedding(max_num_frames, hidden_size)
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, temporal_layers
        )
        self.mod = QuestionConditionedMoD(
            hidden_size=hidden_size,
            num_routing_blocks=1,
            routing_dim=hidden_size,
            router_hidden_dim=hidden_size * mod_expansion,
            capacity_ratios=[mod_keep_ratio],
            attention_heads=num_heads,
            dropout=dropout,
        )
        self.dsf = DiffusionSemanticFusion(
            hidden_size,
            dsf_steps,
            dropout,
            dsf_init_noise_scale,
        )
        if dsf_target_temperature <= 0:
            raise ValueError("dsf_target_temperature must be positive.")
        self.dsf_target_temperature = dsf_target_temperature

    @staticmethod
    def _pool_question(question_tokens, question_mask):
        weights = question_mask.to(question_tokens.dtype)
        pooled = torch.einsum("bl,bld->bd", weights, question_tokens)
        return pooled / weights.sum(1, keepdim=True).clamp_min(1.0)

    def forward(
        self,
        frame_query_tokens,
        question_tokens,
        question_mask,
        training=True,
        video_mask=None,
    ):
        if frame_query_tokens.ndim != 4:
            raise ValueError(
                "frame_query_tokens must have shape [B, T, Q, D]."
            )
        batch_size, num_frames, _, _ = frame_query_tokens.shape
        if num_frames > self.temporal_position.num_embeddings:
            raise ValueError(
                f"Received {num_frames} frames, but max_num_frames is "
                f"{self.temporal_position.num_embeddings}."
            )
        if video_mask is None:
            video_mask = torch.ones(
                batch_size,
                num_frames,
                dtype=torch.bool,
                device=frame_query_tokens.device,
            )
        else:
            video_mask = video_mask.to(
                device=frame_query_tokens.device, dtype=torch.bool
            )
        if torch.any(question_mask.sum(dim=1) == 0):
            raise ValueError("Each question must contain at least one valid token.")
        position_ids = torch.arange(
            num_frames, device=frame_query_tokens.device
        ).unsqueeze(0)
        video_feats = frame_query_tokens.mean(dim=2)
        video_feats = self.temporal_encoder(
            video_feats + self.temporal_position(position_ids),
            src_key_padding_mask=~video_mask,
        )
        question_feat = self._pool_question(question_tokens, question_mask)
        mod_output = self.mod(
            video_feats,
            question_tokens,
            question_mask=question_mask,
            visual_mask=video_mask,
        )
        mod_video_feats = mod_output["final_visual_state"]
        depth_logits = mod_output["routing_logits"][-1]
        depth_mask = mod_output["routing_masks"][-1]
        evidence_prior = torch.softmax(depth_logits, dim=-1)
        evidence_prior = evidence_prior.masked_fill(~video_mask, 0.0)
        evidence_prior = evidence_prior / evidence_prior.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        target_logits = torch.einsum(
            "btd,bd->bt",
            F.normalize(mod_video_feats, dim=-1),
            F.normalize(question_feat, dim=-1),
        )
        target_logits = target_logits.masked_fill(
            ~video_mask, torch.finfo(target_logits.dtype).min
        )
        evidence_target = torch.softmax(
            target_logits / self.dsf_target_temperature, dim=-1
        ).detach()
        if training:
            dsf_loss, refined_evidence = self.dsf.training_losses(
                evidence_target,
                evidence_prior,
                depth_logits,
                depth_mask,
                mod_video_feats,
                question_tokens,
                video_mask,
                question_mask,
            )
        else:
            dsf_loss = video_feats.new_zeros(())
            refined_evidence = self.dsf.sample(
                evidence_prior,
                depth_logits,
                depth_mask,
                mod_video_feats,
                question_tokens,
                video_mask,
                question_mask,
            )

        evid_feat = torch.einsum(
            "bt,btd->bd", refined_evidence, mod_video_feats
        )
        target_keep = video_feats.new_tensor(self.mod.routing_blocks[-1].capacity_ratio)
        valid_counts = video_mask.to(video_feats.dtype).sum(1).clamp_min(1.0)
        actual_keep = depth_mask.sum(1) / valid_counts
        mod_aux_loss = F.mse_loss(actual_keep, target_keep.expand_as(actual_keep))
        return {
            "llm_visual_tokens": evid_feat.unsqueeze(1),
            "mod_video_feats": mod_video_feats,
            "mod_aux_loss": mod_aux_loss,
            "dsf_loss": dsf_loss,
            "evidence_target": evidence_target,
            "refined_evidence": refined_evidence,
            "evidence_prior": evidence_prior,
            "depth_logits": depth_logits,
            "depth_mask": depth_mask,
            "visual_states": mod_output["visual_states"],
            "effective_depth": mod_output["effective_depth"],
        }
