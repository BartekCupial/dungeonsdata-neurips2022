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
        self.mem_len = flags.mem_len

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

    def initial_state(self, batch_size=1, K=None):
        if K is None:
            K = self.mem_len
        return dict(
            memory=self.core.init_mems(batch_size, mem_len=K),
            timesteps=torch.zeros(K, batch_size),
            mask=torch.zeros(K, batch_size).to(torch.bool),
        )

    def forward(self, inputs, core_state):
        T, B, C, H, W = inputs["screen_image"].shape
        mems = core_state["memory"]
        state_mask = core_state["mask"]
        # TODO: use timesteps as positional embeds

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
        else:
            timesteps = (
                torch.arange(T, device=inputs["mask"].device)
                .view(1, -1, 1)
                .repeat(B, 1, 1)
            )
        timesteps = timesteps.long().squeeze(-1)
        # pos_embs = self.core.pos_emb(timesteps)

        inputs_embeds = self.embed_ln(inputs_embeds)

        qlen = T
        mlen = mems[0].size(0)
        klen = mlen + qlen
        all_ones = inputs_embeds.new_ones((qlen, klen), dtype=torch.uint8)
        mask_len = klen - self.mem_len
        if mask_len > 0:
            mask_shift_len = qlen - mask_len
        else:
            mask_shift_len = qlen
        dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len))[:, :, None].repeat(1, 1, B)
        # update with mask from inputs
        state_mask = torch.cat([state_mask, inputs["mask"]], dim=0)
        dec_attn_mask = dec_attn_mask | (~state_mask.unsqueeze(0)).to(torch.uint8)
        

        if inputs["done"].any():
            # # for breakpoint
            # if T == 32:
            #     print("breakpoint")

            # modify to prevent attending to states between games
            mask = torch.zeros_like(dec_attn_mask)
            xs, ys = torch.where(inputs["done"])
            for x, y in zip(xs, ys):
                mask[:, :, y] = 1
                mask[x:, x+T:, y] = 0
                mask[:x, :x+T, y] = 0

                # reset state if episode finished
                init_state = self.initial_state(K=x+T)
                state_mask[:x+T, y] = init_state["mask"].squeeze(-1)
            dec_attn_mask = dec_attn_mask | mask

        core_output = self.core(
            inputs_embeds=inputs_embeds,
            mems=mems,
            dec_attn_mask=dec_attn_mask,
            # pos_embs=pos_embs,
        )
        x = core_output["last_hidden_state"]
        new_memory = core_output["mems"]

        # -- [B' x A]
        policy_logits = self.policy(x)

        # -- [B' x 1]
        baseline = self.baseline(x).squeeze(-1)

        action = torch.multinomial(F.softmax(policy_logits.view(B * T, -1), dim=1), num_samples=1)

        policy_logits = policy_logits.permute(1, 0, 2)
        baseline = baseline.permute(1, 0)
        action = action.view(B, T).permute(1, 0)

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

        core_state["memory"] = new_memory
        core_state["mask"] = state_mask[-self.mem_len:]
        return (output, core_state)
