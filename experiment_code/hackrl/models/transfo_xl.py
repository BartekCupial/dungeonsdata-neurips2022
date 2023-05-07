from functools import partial

import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F

from hackrl.core import nest
from hackrl.models.trajectory_transfo_xl import TransfoXLModel
from hackrl.models.decision_transformer import DecisionTransformer


class TransfoXL(DecisionTransformer):
    def __init__(
        self,    
        shape,
        action_space,
        flags,
        device,
    ):
        super().__init__(shape, action_space, flags, device)

        kwargs = dict(
            d_model=self.hidden_dim, 
            d_embed=self.hidden_dim, 
            n_layer=flags.n_layer,
            n_head=flags.n_head,
            d_head=self.hidden_dim // flags.n_head,
            d_inner=4 * self.hidden_dim,
            div_val=1,
            pre_lnorm=True,
            mem_len=flags.mem_len,

            dropout=flags.dropout,
            dropatt=flags.dropout,
        )

        config = transformers.TransfoXLConfig(vocab_size=1, **kwargs)

        # note: the only difference between this TransformerXL and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.core = TransfoXLModel(config)
        self.core.hidden_size = self.hidden_dim
        self.core.num_layers = flags.n_layer


    def forward(self, inputs, core_state):
        org_inputs = inputs
        OT = org_inputs["screen_image"].shape[0]
        K = max(self.max_length, OT)

        core_state = dict(sorted({key: value for key, value in core_state.items() if key in inputs.keys()}.items()))
        inputs = dict(sorted({key: value for key, value in inputs.items() if key in core_state.keys()}.items()))
        inputs = nest.map_many(partial(torch.cat, dim=0), *[core_state, inputs])
        inputs = nest.map(lambda x: x[-K :], inputs)

        T, B, C, H, W = inputs["screen_image"].shape

        if self.use_tty_only:
            topline = inputs["tty_chars"][..., 0, :]
            bottom_line = inputs["tty_chars"][..., -2:, :]
        else:
            topline = inputs["message"]
            bottom_line = inputs["blstats"]
            assert False, "TODO: topline shape transpose "

        topline = topline.permute(1, 0, 2)
        bottom_line = bottom_line.permute(1, 0, 2, 3)

        st = [
            self.topline_encoder(
                topline.float(memory_format=torch.contiguous_format).reshape(T * B, -1)
            ),
            self.bottomline_encoder(
                bottom_line.float(memory_format=torch.contiguous_format).reshape(
                    T * B, -1
                )
            ),
            self.screen_encoder(
                inputs["screen_image"]
                .permute(1, 0, 2, 3, 4)
                .float(memory_format=torch.contiguous_format)
                .reshape(T * B, C, H, W)
            ),
        ]
        if self.use_prev_action:
            actions = inputs["prev_action"].permute(1, 0).float().long()
            st.append(
                self.action_encoder(
                    actions
                ).reshape(T * B, -1)
            )
        if self.use_returns:
            if self.return_to_go:
                target_score = inputs["max_scores"] - inputs["scores"]
            else:
                target_score = inputs["max_scores"]
            st.append(
                self.return_encoder(
                    target_score.T.reshape(T * B, -1) / self.score_scale
                )
            )

        st = torch.cat(st, dim=1)
        core_input = st.view(B, T, -1)
        inputs_embeds = self.embed_input(core_input)

        if self.use_timesteps:
            timesteps = inputs["timesteps"].permute(1, 0).unsqueeze(-1)
            if self.linear_time_embeddings:
                timesteps = timesteps.float() / self.flags.env.max_episode_steps
            else:
                timesteps = timesteps.long().squeeze(-1)
        else:
            timesteps = (
                torch.arange(T, device=inputs["mask"].device)
                .view(1, -1, 1)
                .repeat(B, 1, 1)
            )
            if self.linear_time_embeddings:
                timesteps = timesteps.float()  
            else:
                timesteps = timesteps.long().squeeze(-1)

        time_embeddings = self.embed_timestep(timesteps)
        inputs_embeds = inputs_embeds + time_embeddings

        inputs_embeds = self.embed_ln(inputs_embeds)

        attention_mask = inputs["mask"].T
        causal_mask = (
            torch.tril(torch.ones((B, T, T), dtype=torch.uint8))
            .view(B, 1, T, T)
            .to(attention_mask.device)
        )
        if inputs["done"].any():
            # # for breakpoint
            # if "actions_converted" in org_inputs:
            #     print("breakpoint")

            # modify causal mask, to prevent attending to states between games
            mask = torch.ones_like(causal_mask)
            xs, ys = torch.where(inputs["done"].T)
            for x, y in zip(xs, ys):
                mask[x] = 0
                mask[x, :, y:, y:] = 1
                mask[x, :, :y, :y] = 1

                # reset state if episode finished
                init_state = self.initial_state(K=OT)
                init_state = nest.map(torch.squeeze, init_state)
                nest.slice(inputs, (slice(None), x), init_state)
            causal_mask *= mask

        core_output = self.core(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
        )
        x = core_output["last_hidden_state"]

        # we only want to predict the same number of actions as seq len of org_inputs
        x = x[:, -OT:]

        # -- [B' x A]
        policy_logits = self.policy(x)

        # -- [B' x 1]
        baseline = self.baseline(x).squeeze(-1)

        action = torch.multinomial(F.softmax(policy_logits.view(B * OT, -1), dim=1), num_samples=1)

        policy_logits = policy_logits.permute(1, 0, 2)
        baseline = baseline.permute(1, 0)
        action = action.view(B, OT).permute(1, 0)

        version = torch.ones_like(action) * self.version

        output = dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
            version=version,
        )

        if self.use_inverse_model:
            inverse_action_logits = self.inverse_model(core_input)
            output["encoded_state"] = core_input
            output["inverse_action_logits"] = inverse_action_logits
        return (output, inputs)
