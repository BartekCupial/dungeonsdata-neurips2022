from functools import partial

import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F

from hackrl.core import nest
from hackrl.models.trajectory_gpt2 import GPT2Model
from hackrl.models.chaotic_dwarf import ChaoticDwarvenGPT5


class DecisionTransformer(ChaoticDwarvenGPT5):
    """This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)."""

    def __init__(
        self,
        shape,
        action_space,
        flags,
        device,
    ):
        super().__init__(shape, action_space, flags, device)

        self.max_length = flags.ttyrec_unroll_length
        self.use_returns = flags.use_returns
        self.use_actions = flags.use_actions
        self.return_to_go = flags.return_to_go
        self.score_scale = flags.score_scale
        self.use_timesteps = flags.use_timesteps
        self.hidden_dim = flags.hidden_dim

        self.n = 1 + self.use_prev_action * 1 + self.use_returns * 1

        self.h_dim = sum(
            [
                self.topline_encoder.hidden_dim,
                self.bottomline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
            ]
        )

        # self.embed_timestep = nn.Embedding(flags.env.max_episode_steps, self.hidden_dim)
        self.embed_timestep = nn.Linear(1, self.hidden_dim)

        self.embed_state = nn.Linear(self.h_dim, self.hidden_dim)
        self.embed_action = nn.Embedding(self.num_actions, self.hidden_dim)
        self.embed_return = nn.Linear(1, self.hidden_dim)

        self.embed_ln = nn.LayerNorm(self.hidden_dim)

        kwargs = dict(
            n_layer=flags.n_layer,
            n_head=flags.n_head,
            n_inner=4 * self.hidden_dim,
            activation_function=flags.activation_function,
            resid_pdrop=flags.dropout,
            attn_pdrop=flags.dropout,
        )

        config = transformers.GPT2Config(vocab_size=1, n_embd=self.hidden_dim, **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.core = GPT2Model(config)

        self.policy = nn.Linear(self.hidden_dim, self.num_actions)
        self.baseline = nn.Linear(self.hidden_dim, 1)

    def initial_state(self, batch_size=1):
        return dict(
            blstats=torch.zeros(self.max_length, batch_size, 27).to(torch.long),
            done=torch.zeros(self.max_length, batch_size).to(torch.bool),
            message=torch.zeros(self.max_length, batch_size, 256).to(torch.uint8),
            screen_image=torch.zeros(self.max_length, batch_size, 3, 108, 108).to(torch.uint8),
            tty_chars=torch.zeros(self.max_length, batch_size, 24, 80).to(torch.uint8),
            tty_colors=torch.zeros(self.max_length, batch_size, 24, 80).to(torch.int8),
            tty_cursor=torch.zeros(self.max_length, batch_size, 2).to(torch.uint8),
            prev_action=torch.zeros(self.max_length, batch_size).to(torch.long),
            timesteps=torch.zeros(self.max_length, batch_size),
            scores=torch.zeros(self.max_length, batch_size),
            max_scores=torch.zeros(self.max_length, batch_size),
            mask=torch.zeros(self.max_length, batch_size).to(torch.bool),
        )

    def forward(self, inputs, core_state):
        org_inputs = inputs
        OT = org_inputs["screen_image"].shape[0]

        core_state = dict(sorted({key: value for key, value in core_state.items() if key in inputs.keys()}.items()))
        inputs = dict(sorted({key: value for key, value in inputs.items() if key in core_state.keys()}.items()))
        inputs = nest.map_many(partial(torch.cat, dim=0), *[core_state, inputs])
        inputs = nest.map(lambda x: x[-self.max_length :], inputs)

        T, B, C, H, W = inputs["screen_image"].shape

        # time embeddings
        if self.use_timesteps:
            timesteps = (
                inputs["timesteps"].permute(1, 0).unsqueeze(-1)
                / self.flags.env.max_episode_steps
            )
            time_embeddings = self.embed_timestep(timesteps)
        else:
            timesteps = (
                torch.arange(T, device=inputs["mask"].device)
                .view(1, -1, 1)
                .repeat(B, 1, 1)
                .float()
            )
            time_embeddings = self.embed_timestep(timesteps)

        inputs_embeds = []

        # return embeddings
        if self.use_returns:
            if self.return_to_go:
                target_score = inputs["max_scores"] - inputs["scores"]
            else:
                target_score = inputs["max_scores"]
            target_score = target_score.T.unsqueeze(-1) / self.score_scale
            return_embeddings = self.embed_return(target_score) + time_embeddings
            inputs_embeds.append(return_embeddings)

        # state embeddings
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
        st = torch.cat(st, dim=1)
        state_input = st.view(B, T, -1)
        state_embeddings = self.embed_state(state_input) + time_embeddings
        inputs_embeds.append(state_embeddings)

        if self.use_prev_action:
            actions = inputs["prev_action"].T.float().long()
            action_embeddings = self.embed_action(actions) + time_embeddings
            inputs_embeds.append(action_embeddings)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(inputs_embeds, dim=1)
            .permute(0, 2, 1, 3)
            .reshape(B, self.n * T, self.hidden_dim)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        attention_mask = inputs["mask"].T
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask,) * self.n, dim=1).permute(0, 2, 1).reshape(B, self.n * T)
        )

        causal_mask = (
            torch.tril(torch.ones((B, T * self.n, T * self.n), dtype=torch.uint8))
            .view(B, 1, T * self.n, T * self.n)
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
                y = y * self.n
                mask[x] = 0
                mask[x, :, y:, y:] = 1
                mask[x, :, :y, :y] = 1

                # reset state if episode finished
                init_state = self.initial_state()
                init_state = nest.map(torch.squeeze, init_state)
                nest.slice(inputs, (slice(None), x), init_state)
            causal_mask *= mask

        core_output = self.core(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            causal_mask=causal_mask,
        )
        x = core_output["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(B, T, self.n, self.hidden_dim).permute(0, 2, 1, 3)

        # we want to look at every 3rd (nth) element, we predict action every (r, t, a) 
        # and since we are passing prev_action not action we include last action from sequence
        # thats why for all (r, s, a) we pass index 2 instead of 1
        x = x[:, self.n - 1] # -1 because we index from 0
        
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
            # TODO: pass the same input to inverse model as in ChaoticDwarven
            inverse_action_logits = self.inverse_model(core_input)
            output["encoded_state"] = core_input
            output["inverse_action_logits"] = inverse_action_logits
        return (output, inputs)
